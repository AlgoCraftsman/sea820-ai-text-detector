"""Analyze frozen-test errors without changing any trained model or threshold."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DEFAULT_PREDICTIONS_PATH = Path("results/frozen_test_predictions.csv.gz")
DEFAULT_PREPARED_DATASET_PATH = Path("data/processed/distilbert_cased_seed42")
DEFAULT_CHECKPOINT_PATH = Path(
    "checkpoints/distilbert-full-epoch1/checkpoint-23212"
)
DEFAULT_SUMMARY_PATH = Path("results/frozen_error_analysis.json")
DEFAULT_MODEL_PATH = Path("results/frozen_error_analysis_by_model.csv")
DEFAULT_SLICES_PATH = Path("results/frozen_error_analysis_slices.csv")
DEFAULT_OVERLAP_PATH = Path("results/frozen_error_overlap.csv")
DEFAULT_EXAMPLES_PATH = Path("results/frozen_error_examples.csv.gz")
DEFAULT_CONFUSION_FIGURE = Path("results/figures/frozen_confusion_matrices.svg")
DEFAULT_SLICE_FIGURE = Path("results/figures/frozen_error_rate_by_truncation.svg")

EXPECTED_PREDICTIONS_SHA256 = (
    "11b6e98055bd0aeef177c5202b2478c3dc7c689218a2e66afb70853e1e413422"
)
EXPECTED_TEST_ROWS = 46_423
EXPECTED_LABEL_COUNTS = {0: 28_446, 1: 17_977}
MAX_LENGTH = 512

MODEL_SPECS = {
    "Logistic Regression": {
        "slug": "logistic_regression",
        "prediction": "logistic_regression_pred",
        "score": "logistic_regression_probability",
        "score_type": "label_1_probability",
    },
    "Linear SVM": {
        "slug": "linear_svm",
        "prediction": "linear_svm_pred",
        "score": "linear_svm_decision_score",
        "score_type": "label_1_decision_function",
    },
    "DistilBERT": {
        "slug": "distilbert",
        "prediction": "distilbert_pred",
        "score": "distilbert_probability",
        "score_type": "label_1_probability",
    },
}

LENGTH_BIN_ORDER = ["0-100", "101-250", "251-500", "501-750", "751+"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_word_length_bin(word_count: int) -> str:
    if word_count < 0:
        raise ValueError("Word count cannot be negative")
    if word_count <= 100:
        return "0-100"
    if word_count <= 250:
        return "101-250"
    if word_count <= 500:
        return "251-500"
    if word_count <= 750:
        return "501-750"
    return "751+"


def classify_outcomes(y_true: Any, y_pred: Any) -> np.ndarray:
    truth = np.asarray(y_true, dtype=np.int8)
    predicted = np.asarray(y_pred, dtype=np.int8)
    if truth.shape != predicted.shape:
        raise ValueError("True and predicted labels must have the same shape")
    if not np.isin(truth, (0, 1)).all() or not np.isin(predicted, (0, 1)).all():
        raise ValueError("Outcomes require binary labels 0 and 1")
    outcomes = np.empty(truth.shape, dtype="<U2")
    outcomes[(truth == 0) & (predicted == 0)] = "TN"
    outcomes[(truth == 0) & (predicted == 1)] = "FP"
    outcomes[(truth == 1) & (predicted == 0)] = "FN"
    outcomes[(truth == 1) & (predicted == 1)] = "TP"
    return outcomes


def overlap_signature(error_flags: dict[str, bool]) -> str:
    errored = [name for name, is_error in error_flags.items() if is_error]
    return "none" if not errored else " + ".join(errored)


def classify_opening_style(text: str) -> str:
    """Assign a compact, descriptive opening style without model-dependent tuning."""

    normalized = " ".join(text.split()).lower()
    if normalized.startswith("dear ") or normalized.startswith("far principal"):
        return "salutation"
    if "?" in normalized[:150]:
        return "opening_question"
    return "other"


def compute_binary_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int8)
    predicted = np.asarray(y_pred, dtype=np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def summarize_slice(
    frame: pd.DataFrame,
    *,
    model: str,
    prediction_column: str,
    slice_name: str,
    slice_value: str,
) -> dict[str, Any]:
    outcomes = classify_outcomes(frame["true_label"], frame[prediction_column])
    support = len(frame)
    errors = int(np.isin(outcomes, ("FP", "FN")).sum())
    return {
        "model": model,
        "slice": slice_name,
        "value": slice_value,
        "support": support,
        "errors": errors,
        "error_rate": errors / support if support else 0.0,
        "false_positives": int((outcomes == "FP").sum()),
        "false_negatives": int((outcomes == "FN").sum()),
        **compute_binary_metrics(frame["true_label"], frame[prediction_column]),
    }


def _validate_predictions(frame: pd.DataFrame) -> None:
    required = {"source_row_id", "true_label"}
    for spec in MODEL_SPECS.values():
        required.update((spec["prediction"], spec["score"]))
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS:,} predictions, got {len(frame):,}")
    if not frame["source_row_id"].is_unique:
        raise ValueError("Prediction source_row_id values must be unique")
    if frame[list(required)].isna().any().any():
        raise ValueError("Prediction inputs cannot contain missing values")
    counts = {
        int(label): int(count)
        for label, count in frame["true_label"].value_counts().sort_index().items()
    }
    if counts != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Frozen label counts differ: {counts}")
    for spec in MODEL_SPECS.values():
        if not frame[spec["prediction"]].isin((0, 1)).all():
            raise ValueError(f"{spec['prediction']} must contain only 0 and 1")


def _load_aligned_test(
    predictions: pd.DataFrame,
    prepared_dataset_path: Path,
) -> tuple[Any, np.ndarray]:
    from datasets import load_from_disk

    prepared = load_from_disk(str(prepared_dataset_path))
    if "test" not in prepared:
        raise ValueError("Prepared DatasetDict is missing test")
    test = prepared["test"]
    required = {"text", "labels", "source_row_id", "input_ids", "attention_mask"}
    missing = required.difference(test.column_names)
    if missing:
        raise ValueError(f"Prepared test split is missing: {sorted(missing)}")
    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError("Prepared test row count differs from frozen predictions")

    prepared_ids = np.asarray(test["source_row_id"], dtype=np.int64)
    prepared_labels = np.asarray(test["labels"], dtype=np.int8)
    if len(np.unique(prepared_ids)) != len(prepared_ids):
        raise ValueError("Prepared test source IDs must be unique")
    prediction_lookup = predictions.set_index("source_row_id")["true_label"]
    try:
        aligned_labels = prediction_lookup.loc[prepared_ids].to_numpy(dtype=np.int8)
    except KeyError as exc:
        raise ValueError("Prepared test membership differs from predictions") from exc
    if not np.array_equal(prepared_labels, aligned_labels):
        raise ValueError("Prepared test labels differ from frozen predictions")
    return test, prepared_ids


def _measure_text(
    test: Any,
    *,
    checkpoint_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path), use_fast=True, local_files_only=True
    )
    character_counts = np.empty(len(test), dtype=np.int32)
    word_counts = np.empty(len(test), dtype=np.int32)
    token_lengths = np.empty(len(test), dtype=np.int32)
    excerpts: list[str] = [""] * len(test)

    for start in range(0, len(test), batch_size):
        stop = min(start + batch_size, len(test))
        texts = test[start:stop]["text"]
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
            verbose=False,
        )
        token_lengths[start:stop] = np.asarray(encoded["length"], dtype=np.int32)
        for offset, text in enumerate(texts, start=start):
            character_counts[offset] = len(text)
            word_counts[offset] = len(text.split())
            excerpts[offset] = " ".join(text.split())[:500]
    return character_counts, word_counts, token_lengths, excerpts


def _build_analysis_frame(
    predictions: pd.DataFrame,
    test: Any,
    prepared_ids: np.ndarray,
    *,
    checkpoint_path: Path,
    batch_size: int,
) -> pd.DataFrame:
    by_id = predictions.set_index("source_row_id")
    analysis = by_id.loc[prepared_ids].reset_index()
    character_counts, word_counts, token_lengths, excerpts = _measure_text(
        test, checkpoint_path=checkpoint_path, batch_size=batch_size
    )
    analysis["character_count"] = character_counts
    analysis["word_count"] = word_counts
    analysis["token_length"] = token_lengths
    analysis["truncated_at_512"] = token_lengths > MAX_LENGTH
    analysis["word_length_bin"] = pd.Categorical(
        [assign_word_length_bin(int(count)) for count in word_counts],
        categories=LENGTH_BIN_ORDER,
        ordered=True,
    )
    analysis["text_excerpt"] = excerpts
    analysis["opening_style"] = [classify_opening_style(text) for text in excerpts]
    for spec in MODEL_SPECS.values():
        analysis[f"{spec['slug']}_outcome"] = classify_outcomes(
            analysis["true_label"], analysis[spec["prediction"]]
        )
        analysis[f"{spec['slug']}_error"] = (
            analysis[spec["prediction"]] != analysis["true_label"]
        )
    return analysis


def _model_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, spec in MODEL_SPECS.items():
        outcomes = analysis[f"{spec['slug']}_outcome"]
        error_mask = analysis[f"{spec['slug']}_error"]
        error_scores = analysis.loc[error_mask, spec["score"]].to_numpy(dtype=float)
        error_predictions = analysis.loc[
            error_mask, spec["prediction"]
        ].to_numpy(dtype=np.int8)
        if spec["score_type"] == "label_1_probability":
            strengths = np.where(
                error_predictions == 1,
                error_scores,
                1.0 - error_scores,
            )
            strength_name = "incorrect_predicted_class_probability"
            high_90: int | str = int((strengths >= 0.90).sum())
            high_99: int | str = int((strengths >= 0.99).sum())
        else:
            strengths = np.abs(error_scores)
            strength_name = "absolute_decision_function_margin"
            high_90 = ""
            high_99 = ""
        rows.append(
            {
                "model": model,
                **compute_binary_metrics(
                    analysis["true_label"], analysis[spec["prediction"]]
                ),
                "true_negatives": int((outcomes == "TN").sum()),
                "false_positives": int((outcomes == "FP").sum()),
                "false_negatives": int((outcomes == "FN").sum()),
                "true_positives": int((outcomes == "TP").sum()),
                "total_errors": int(analysis[f"{spec['slug']}_error"].sum()),
                "score_type": spec["score_type"],
                "error_strength_statistic": strength_name,
                "median_error_strength": float(np.median(strengths)),
                "errors_with_strength_at_least_0_90": high_90,
                "errors_with_strength_at_least_0_99": high_99,
            }
        )
    return pd.DataFrame(rows)


def _slice_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    slice_definitions = {
        "truncation": analysis["truncated_at_512"].map(
            {False: "not_truncated", True: "truncated"}
        ),
        "word_length_bin": analysis["word_length_bin"].astype(str),
        "true_label": analysis["true_label"].map({0: "human", 1: "AI-generated"}),
        "opening_style": analysis["opening_style"],
    }
    for model, spec in MODEL_SPECS.items():
        for slice_name, values in slice_definitions.items():
            for slice_value in values.drop_duplicates():
                subset = analysis.loc[values == slice_value]
                rows.append(
                    summarize_slice(
                        subset,
                        model=model,
                        prediction_column=spec["prediction"],
                        slice_name=slice_name,
                        slice_value=str(slice_value),
                    )
                )
    return pd.DataFrame(rows)


def _overlap_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    slugs = [spec["slug"] for spec in MODEL_SPECS.values()]
    signatures = [
        overlap_signature(
            {slug: bool(row[f"{slug}_error"]) for slug in slugs}
        )
        for _, row in analysis.iterrows()
    ]
    counts = pd.Series(signatures).value_counts().rename_axis("error_models").reset_index(
        name="rows"
    )
    counts["fraction"] = counts["rows"] / len(analysis)
    return counts.sort_values(["rows", "error_models"], ascending=[False, True])


def _error_examples(analysis: pd.DataFrame) -> pd.DataFrame:
    error_columns = [f"{spec['slug']}_error" for spec in MODEL_SPECS.values()]
    columns = [
        "source_row_id",
        "true_label",
        "character_count",
        "word_count",
        "token_length",
        "truncated_at_512",
        "word_length_bin",
        "text_excerpt",
    ]
    for spec in MODEL_SPECS.values():
        columns.extend(
            [
                spec["prediction"],
                spec["score"],
                f"{spec['slug']}_outcome",
            ]
        )
    return analysis.loc[analysis[error_columns].any(axis=1), columns].sort_values(
        ["true_label", "source_row_id"]
    )


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression: str | dict[str, Any]
    if path.suffix == ".gz":
        compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    else:
        compression = "infer"
    frame.to_csv(path, index=False, compression=compression)


def _plot_confusion(model_summary: pd.DataFrame, path: Path) -> None:
    width, height = 1120, 390
    body = [
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="560" y="35" text-anchor="middle" font-size="24" '
        'font-family="Arial" font-weight="bold">Frozen-test confusion matrices</text>',
    ]
    panel_width = 360
    for panel, row in enumerate(model_summary.to_dict("records")):
        matrix = np.asarray(
            [
                [row["true_negatives"], row["false_positives"]],
                [row["false_negatives"], row["true_positives"]],
            ]
        )
        origin_x = 55 + panel * panel_width
        origin_y = 105
        cell_size = 115
        maximum = float(matrix.max())
        model = html.escape(str(row["model"]))
        body.append(
            f'<text x="{origin_x + cell_size}" y="72" text-anchor="middle" '
            f'font-size="18" font-family="Arial" font-weight="bold">{model}</text>'
        )
        for (y, x), value in np.ndenumerate(matrix):
            intensity = 0.18 + 0.68 * (float(value) / maximum if maximum else 0.0)
            blue = int(255 - 115 * intensity)
            fill = f"rgb({blue},{blue + 25},{255})"
            cell_x = origin_x + x * cell_size
            cell_y = origin_y + y * cell_size
            body.extend(
                [
                    f'<rect x="{cell_x}" y="{cell_y}" width="{cell_size}" '
                    f'height="{cell_size}" fill="{fill}" stroke="#4b5563"/>',
                    f'<text x="{cell_x + cell_size / 2}" '
                    f'y="{cell_y + cell_size / 2 + 7}" text-anchor="middle" '
                    f'font-size="19" font-family="Arial">{int(value):,}</text>',
                ]
            )
        body.extend(
            [
                f'<text x="{origin_x + cell_size / 2}" y="{origin_y - 12}" '
                'text-anchor="middle" font-size="14" font-family="Arial">Human</text>',
                f'<text x="{origin_x + 1.5 * cell_size}" y="{origin_y - 12}" '
                'text-anchor="middle" font-size="14" font-family="Arial">AI</text>',
                f'<text x="{origin_x - 10}" y="{origin_y + cell_size / 2 + 5}" '
                'text-anchor="end" font-size="14" font-family="Arial">Human</text>',
                f'<text x="{origin_x - 10}" y="{origin_y + 1.5 * cell_size + 5}" '
                'text-anchor="end" font-size="14" font-family="Arial">AI</text>',
                f'<text x="{origin_x + cell_size}" y="{origin_y + 2 * cell_size + 35}" '
                'text-anchor="middle" font-size="14" font-family="Arial">Predicted label</text>',
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(body) + "</svg>\n",
        encoding="utf-8",
    )


def _plot_truncation_slices(slice_summary: pd.DataFrame, path: Path) -> None:
    selected = slice_summary.loc[slice_summary["slice"] == "truncation"].copy()
    order = ["not_truncated", "truncated"]
    models = list(MODEL_SPECS)
    values_by_slice = {
        value: (
            selected.loc[selected["value"] == value]
            .set_index("model")
            .loc[models, "error_rate"]
            .to_numpy()
        )
        for value in order
    }
    maximum = max(float(values.max()) for values in values_by_slice.values())
    chart_max = maximum * 1.18 if maximum else 1.0
    svg_width, svg_height = 900, 500
    chart_left, chart_top, chart_width, chart_height = 85, 80, 760, 330
    colors = {"not_truncated": "#7aa6c2", "truncated": "#d97757"}
    body = [
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="35" text-anchor="middle" font-size="23" '
        'font-family="Arial" font-weight="bold">Frozen-test error rate by 512-token truncation</text>',
    ]
    for tick in range(6):
        rate = chart_max * tick / 5
        y = chart_top + chart_height - chart_height * rate / chart_max
        body.extend(
            [
                f'<line x1="{chart_left}" y1="{y}" x2="{chart_left + chart_width}" '
                'y2="{y}" stroke="#d1d5db" stroke-width="1"/>',
                f'<text x="{chart_left - 10}" y="{y + 5}" text-anchor="end" '
                f'font-size="12" font-family="Arial">{rate:.3%}</text>',
            ]
        )
    group_width = chart_width / len(models)
    bar_width = 62
    for model_index, model in enumerate(models):
        center = chart_left + group_width * (model_index + 0.5)
        for slice_index, value in enumerate(order):
            rate = float(values_by_slice[value][model_index])
            bar_height = chart_height * rate / chart_max
            x = center + (slice_index - 0.5) * (bar_width + 10) - bar_width / 2
            y = chart_top + chart_height - bar_height
            body.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
                    f'fill="{colors[value]}"/>',
                    f'<text x="{x + bar_width / 2}" y="{max(chart_top + 12, y - 6)}" '
                    f'text-anchor="middle" font-size="12" font-family="Arial">{rate:.3%}</text>',
                ]
            )
        body.append(
            f'<text x="{center}" y="{chart_top + chart_height + 30}" '
            f'text-anchor="middle" font-size="14" font-family="Arial">{html.escape(model)}</text>'
        )
    for index, value in enumerate(order):
        legend_x = 300 + index * 220
        body.extend(
            [
                f'<rect x="{legend_x}" y="455" width="18" height="18" '
                f'fill="{colors[value]}"/>',
                f'<text x="{legend_x + 26}" y="469" font-size="14" '
                f'font-family="Arial">{value.replace("_", " ")}</text>',
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}">' + "".join(body) + "</svg>\n",
        encoding="utf-8",
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument(
        "--prepared-dataset", type=Path, default=DEFAULT_PREPARED_DATASET_PATH
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--slices-output", type=Path, default=DEFAULT_SLICES_PATH)
    parser.add_argument("--overlap-output", type=Path, default=DEFAULT_OVERLAP_PATH)
    parser.add_argument("--examples-output", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--confusion-figure", type=Path, default=DEFAULT_CONFUSION_FIGURE)
    parser.add_argument("--slice-figure", type=Path, default=DEFAULT_SLICE_FIGURE)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing complete error-analysis result set.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.tokenizer_batch_size <= 0:
        raise ValueError("Tokenizer batch size must be positive")
    required_paths = [args.predictions, args.prepared_dataset, args.checkpoint]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required local artifacts are missing: " + ", ".join(missing))
    outputs = [
        args.summary,
        args.model_output,
        args.slices_output,
        args.overlap_output,
        args.examples_output,
        args.confusion_figure,
        args.slice_figure,
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Error-analysis outputs already exist: " + ", ".join(existing))

    predictions_hash = sha256_file(args.predictions)
    if predictions_hash != EXPECTED_PREDICTIONS_SHA256:
        raise ValueError(
            f"Frozen predictions SHA-256 mismatch: expected {EXPECTED_PREDICTIONS_SHA256}, "
            f"got {predictions_hash}"
        )
    predictions = pd.read_csv(args.predictions)
    _validate_predictions(predictions)
    test, prepared_ids = _load_aligned_test(predictions, args.prepared_dataset)
    analysis = _build_analysis_frame(
        predictions,
        test,
        prepared_ids,
        checkpoint_path=args.checkpoint,
        batch_size=args.tokenizer_batch_size,
    )
    model_summary = _model_summary(analysis)
    slice_summary = _slice_summary(analysis)
    overlap_summary = _overlap_summary(analysis)
    examples = _error_examples(analysis)

    summary = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_path": args.predictions.as_posix(),
        "predictions_sha256": predictions_hash,
        "prepared_dataset_path": args.prepared_dataset.as_posix(),
        "checkpoint_path": args.checkpoint.as_posix(),
        "test_rows": len(analysis),
        "label_counts": EXPECTED_LABEL_COUNTS,
        "max_length": MAX_LENGTH,
        "truncated_rows": int(analysis["truncated_at_512"].sum()),
        "truncated_fraction": float(analysis["truncated_at_512"].mean()),
        "rows_with_any_model_error": len(examples),
        "outputs": {
            "model_summary": args.model_output.as_posix(),
            "slices": args.slices_output.as_posix(),
            "overlap": args.overlap_output.as_posix(),
            "error_examples": args.examples_output.as_posix(),
            "confusion_figure": args.confusion_figure.as_posix(),
            "truncation_figure": args.slice_figure.as_posix(),
        },
        "policy": {
            "models_or_thresholds_changed": False,
            "test_used_for_tuning": False,
            "text_storage": "500-character excerpts for rows misclassified by any model",
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".frozen-error-analysis-", dir=args.summary.parent
    ) as staging_directory:
        staging_root = Path(staging_directory)
        staged_pairs = [
            (staging_root / args.model_output.name, args.model_output),
            (staging_root / args.slices_output.name, args.slices_output),
            (staging_root / args.overlap_output.name, args.overlap_output),
            (staging_root / args.examples_output.name, args.examples_output),
            (staging_root / args.confusion_figure.name, args.confusion_figure),
            (staging_root / args.slice_figure.name, args.slice_figure),
            (staging_root / args.summary.name, args.summary),
        ]
        _write_dataframe(model_summary, staged_pairs[0][0])
        _write_dataframe(slice_summary, staged_pairs[1][0])
        _write_dataframe(overlap_summary, staged_pairs[2][0])
        _write_dataframe(examples, staged_pairs[3][0])
        _plot_confusion(model_summary, staged_pairs[4][0])
        _plot_truncation_slices(slice_summary, staged_pairs[5][0])
        staged_pairs[6][0].write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        for staged, final in staged_pairs:
            final.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(final)
    print(model_summary.to_string(index=False))
    print(overlap_summary.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
