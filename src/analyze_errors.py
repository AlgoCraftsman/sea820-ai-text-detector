"""Analyze errors made by the selected Week 2 DistilBERT model.

The workflow reads the predictions exported by ``transformer_finetuning.ipynb``,
aligns them with the original source text, and describes errors by outcome,
length, truncation, opening style, and unsupervised topic. It never retrains the
model or changes a prediction threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "sea820-current-error-analysis"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)


DEFAULT_PREDICTIONS_PATH = Path("results/transformer_test_predictions.csv.gz")
DEFAULT_SOURCE_CSV_PATH = Path("data/AI_Human.csv")
DEFAULT_TOKENIZER_PATH = Path("checkpoints/week2-distilbert-final")
DEFAULT_SUMMARY_PATH = Path("results/error_analysis_summary.json")
DEFAULT_METRICS_PATH = Path("results/error_analysis_metrics.csv")
DEFAULT_SLICES_PATH = Path("results/error_analysis_slices.csv")
DEFAULT_TOPICS_PATH = Path("results/error_analysis_topics.csv")
DEFAULT_TOPIC_TERMS_PATH = Path("results/error_analysis_topic_terms.csv")
DEFAULT_EXAMPLES_PATH = Path("results/error_examples.csv.gz")
DEFAULT_CONFUSION_FIGURE = Path("results/figures/error_confusion_matrix.svg")
DEFAULT_LENGTH_FIGURE = Path("results/figures/error_rate_by_length.svg")
DEFAULT_TOPIC_FIGURE = Path("results/figures/error_rate_by_topic.svg")

EXPECTED_PREDICTIONS_SHA256 = (
    "9fdcb0c14e18e4182343c6fecaca140a696be8df6ab9cbd8a592d4fea2435031"
)
EXPECTED_TEST_ROWS = 46_423
EXPECTED_LABEL_COUNTS = {0: 28_447, 1: 17_976}
MAX_LENGTH = 256
TOPIC_RANDOM_SEED = 42
DEFAULT_TOPIC_COUNT = 8
DEFAULT_TOPIC_FEATURES = 8_000
DEFAULT_TOPIC_MIN_DF = 20
DEFAULT_TOPIC_TOP_TERMS = 8
LENGTH_BIN_ORDER = ["0-100", "101-250", "251-500", "501-750", "751+"]
WHITESPACE_RE = re.compile(r"\s+")


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


def classify_opening_style(text: str) -> str:
    """Assign a small descriptive category without using prediction outcomes."""

    normalized = WHITESPACE_RE.sub(" ", text).strip().lower()
    if normalized.startswith(("dear ", "to whom", "hello ", "hi ")):
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
    slice_name: str,
    slice_value: str,
) -> dict[str, Any]:
    outcomes = classify_outcomes(frame["true_label"], frame["distilbert_prediction"])
    support = len(frame)
    errors = int(np.isin(outcomes, ("FP", "FN")).sum())
    return {
        "slice": slice_name,
        "value": slice_value,
        "support": support,
        "errors": errors,
        "error_rate": errors / support if support else 0.0,
        "false_positives": int((outcomes == "FP").sum()),
        "false_negatives": int((outcomes == "FN").sum()),
        **compute_binary_metrics(frame["true_label"], frame["distilbert_prediction"]),
    }


def _validate_predictions(frame: pd.DataFrame) -> None:
    required = {
        "source_row_id",
        "true_label",
        "distilbert_prediction",
        "distilbert_ai_probability",
    }
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
        raise ValueError(f"Test label counts differ: {counts}")
    if not frame["true_label"].isin((0, 1)).all():
        raise ValueError("true_label must contain only 0 and 1")
    if not frame["distilbert_prediction"].isin((0, 1)).all():
        raise ValueError("distilbert_prediction must contain only 0 and 1")
    probabilities = frame["distilbert_ai_probability"].to_numpy(dtype=float)
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("distilbert_ai_probability must be between 0 and 1")


def load_source_rows(
    predictions: pd.DataFrame,
    source_csv_path: Path,
    *,
    chunk_size: int,
) -> pd.DataFrame:
    """Load only predicted source rows from the 1.1 GB CSV and preserve test order."""

    requested_ids = predictions["source_row_id"].to_numpy(dtype=np.int64)
    requested_set = set(requested_ids.tolist())
    selected_chunks: list[pd.DataFrame] = []
    offset = 0

    for chunk in pd.read_csv(
        source_csv_path,
        usecols=["text", "generated"],
        chunksize=chunk_size,
    ):
        source_ids = np.arange(offset, offset + len(chunk), dtype=np.int64)
        mask = np.fromiter(
            (int(source_id) in requested_set for source_id in source_ids),
            dtype=bool,
            count=len(source_ids),
        )
        if mask.any():
            selected = chunk.loc[mask, ["text", "generated"]].copy()
            selected["source_row_id"] = source_ids[mask]
            selected_chunks.append(selected)
        offset += len(chunk)

    if not selected_chunks:
        raise ValueError("No prediction source IDs were found in the source CSV")
    source = pd.concat(selected_chunks, ignore_index=True)
    if not source["source_row_id"].is_unique:
        raise ValueError("Source CSV produced duplicate source_row_id values")
    if len(source) != len(predictions):
        missing = requested_set.difference(source["source_row_id"].tolist())
        raise ValueError(f"Source CSV is missing {len(missing)} predicted rows")

    source["generated"] = pd.to_numeric(source["generated"], errors="raise").astype(int)
    source["text"] = (
        source["text"].fillna("").astype(str).map(lambda text: WHITESPACE_RE.sub(" ", text).strip())
    )
    source = source.set_index("source_row_id").loc[requested_ids].reset_index()
    expected_labels = predictions["true_label"].to_numpy(dtype=np.int8)
    if not np.array_equal(source["generated"].to_numpy(dtype=np.int8), expected_labels):
        raise ValueError("Source CSV labels do not match saved prediction labels")
    if source["text"].eq("").any():
        raise ValueError("Aligned test text cannot be empty")
    return source[["source_row_id", "text"]]


def assign_nmf_topics(
    texts: Iterable[str],
    *,
    n_topics: int,
    max_features: int,
    min_df: int,
    top_terms: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Discover deterministic descriptive topics without labels or error outcomes."""

    text_values = list(texts)
    if n_topics < 2 or n_topics >= len(text_values):
        raise ValueError("n_topics must be at least 2 and smaller than the row count")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        min_df=min_df,
        max_df=0.90,
        max_features=max_features,
        ngram_range=(1, 1),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(text_values)
    if matrix.shape[1] < n_topics:
        raise ValueError("Topic vocabulary is smaller than the requested topic count")

    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=TOPIC_RANDOM_SEED,
        max_iter=400,
        tol=5e-4,
    )
    document_topics = model.fit_transform(matrix)
    assignments = document_topics.argmax(axis=1)
    strengths = document_topics.max(axis=1)
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    term_rows: list[dict[str, Any]] = []
    for topic_index, weights in enumerate(model.components_):
        ordered = np.argsort(weights)[::-1][:top_terms]
        topic = f"topic_{topic_index + 1}"
        for rank, feature_index in enumerate(ordered, start=1):
            term_rows.append(
                {
                    "topic": topic,
                    "rank": rank,
                    "term": str(feature_names[feature_index]),
                    "weight": float(weights[feature_index]),
                }
            )
    labels = np.asarray([f"topic_{index + 1}" for index in assignments])
    return labels, strengths, pd.DataFrame(term_rows)


def _measure_token_lengths(
    texts: list[str],
    *,
    tokenizer_path: Path,
    batch_size: int,
) -> np.ndarray:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=True,
        local_files_only=True,
    )
    lengths = np.empty(len(texts), dtype=np.int32)
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        encoded = tokenizer(
            texts[start:stop],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
            verbose=False,
        )
        lengths[start:stop] = np.asarray(encoded["length"], dtype=np.int32)
    return lengths


def _build_analysis_frame(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    *,
    tokenizer_path: Path,
    tokenizer_batch_size: int,
    topic_count: int,
    topic_features: int,
    topic_min_df: int,
    topic_top_terms: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = predictions.merge(
        source,
        on="source_row_id",
        how="left",
        validate="one_to_one",
    )
    texts = analysis["text"].tolist()
    analysis["character_count"] = [len(text) for text in texts]
    analysis["word_count"] = [len(text.split()) for text in texts]
    analysis["token_length"] = _measure_token_lengths(
        texts,
        tokenizer_path=tokenizer_path,
        batch_size=tokenizer_batch_size,
    )
    analysis["truncated_at_256"] = analysis["token_length"] > MAX_LENGTH
    analysis["word_length_bin"] = pd.Categorical(
        [assign_word_length_bin(int(count)) for count in analysis["word_count"]],
        categories=LENGTH_BIN_ORDER,
        ordered=True,
    )
    analysis["opening_style"] = [classify_opening_style(text) for text in texts]
    topics, strengths, topic_terms = assign_nmf_topics(
        texts,
        n_topics=topic_count,
        max_features=topic_features,
        min_df=topic_min_df,
        top_terms=topic_top_terms,
    )
    analysis["topic"] = topics
    analysis["topic_strength"] = strengths
    analysis["outcome"] = classify_outcomes(
        analysis["true_label"],
        analysis["distilbert_prediction"],
    )
    analysis["is_error"] = analysis["outcome"].isin(("FP", "FN"))
    probability = analysis["distilbert_ai_probability"].to_numpy(dtype=float)
    prediction = analysis["distilbert_prediction"].to_numpy(dtype=np.int8)
    analysis["predicted_class_probability"] = np.where(
        prediction == 1,
        probability,
        1.0 - probability,
    )
    analysis["text_excerpt"] = [
        WHITESPACE_RE.sub(" ", text).strip()[:750] for text in texts
    ]
    return analysis, topic_terms


def _metrics_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    outcomes = analysis["outcome"]
    error_strengths = analysis.loc[
        analysis["is_error"], "predicted_class_probability"
    ].to_numpy(dtype=float)
    row = {
        "model": "Fine-tuned DistilBERT",
        **compute_binary_metrics(
            analysis["true_label"],
            analysis["distilbert_prediction"],
        ),
        "true_negatives": int((outcomes == "TN").sum()),
        "false_positives": int((outcomes == "FP").sum()),
        "false_negatives": int((outcomes == "FN").sum()),
        "true_positives": int((outcomes == "TP").sum()),
        "total_errors": int(analysis["is_error"].sum()),
        "error_rate": float(analysis["is_error"].mean()),
        "median_incorrect_class_probability": float(np.median(error_strengths)),
        "errors_at_least_0_90_confident": int((error_strengths >= 0.90).sum()),
        "errors_at_least_0_99_confident": int((error_strengths >= 0.99).sum()),
    }
    return pd.DataFrame([row])


def _slice_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    slice_definitions = {
        "truncation": analysis["truncated_at_256"].map(
            {False: "not_truncated", True: "truncated"}
        ),
        "word_length_bin": analysis["word_length_bin"].astype(str),
        "true_label": analysis["true_label"].map({0: "human", 1: "AI-generated"}),
        "opening_style": analysis["opening_style"],
        "topic": analysis["topic"],
    }
    for slice_name, values in slice_definitions.items():
        for slice_value in values.drop_duplicates():
            subset = analysis.loc[values == slice_value]
            rows.append(
                summarize_slice(
                    subset,
                    slice_name=slice_name,
                    slice_value=str(slice_value),
                )
            )
    return pd.DataFrame(rows)


def _topic_summary(
    slice_summary: pd.DataFrame,
    topic_terms: pd.DataFrame,
) -> pd.DataFrame:
    topics = slice_summary.loc[slice_summary["slice"] == "topic"].copy()
    keywords = (
        topic_terms.sort_values(["topic", "rank"])
        .groupby("topic")["term"]
        .agg(lambda values: ", ".join(values))
        .rename("top_terms")
    )
    return topics.merge(keywords, left_on="value", right_index=True, how="left").sort_values(
        ["error_rate", "support"],
        ascending=[False, False],
    )


def _error_examples(analysis: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_row_id",
        "true_label",
        "distilbert_prediction",
        "distilbert_ai_probability",
        "predicted_class_probability",
        "outcome",
        "character_count",
        "word_count",
        "token_length",
        "truncated_at_256",
        "word_length_bin",
        "opening_style",
        "topic",
        "topic_strength",
        "text_excerpt",
    ]
    return analysis.loc[analysis["is_error"], columns].sort_values(
        ["outcome", "predicted_class_probability", "source_row_id"],
        ascending=[True, False, True],
    )


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression: str | dict[str, Any]
    if path.suffix == ".gz":
        compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    else:
        compression = "infer"
    frame.to_csv(path, index=False, compression=compression)


def _save_svg(figure: Any, path: Path) -> None:
    """Save deterministic SVG and remove Matplotlib's path-line trailing spaces."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", metadata={"Date": None})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def _plot_confusion(metrics: pd.DataFrame, path: Path) -> None:
    row = metrics.iloc[0]
    matrix = np.asarray(
        [
            [row["true_negatives"], row["false_positives"]],
            [row["false_negatives"], row["true_positives"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Human", "AI-generated"],
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format=",d")
    axis.set_title("Fine-tuned DistilBERT: held-out test errors")
    figure.tight_layout()
    _save_svg(figure, path)
    plt.close(figure)


def _plot_length_slices(slice_summary: pd.DataFrame, path: Path) -> None:
    selected = slice_summary.loc[
        slice_summary["slice"] == "word_length_bin"
    ].copy()
    selected["value"] = pd.Categorical(
        selected["value"],
        categories=LENGTH_BIN_ORDER,
        ordered=True,
    )
    selected = selected.sort_values("value")
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.bar(
        selected["value"].astype(str),
        selected["error_rate"],
        color="#4c78a8",
    )
    axis.set_title("DistilBERT error rate by text length")
    axis.set_xlabel("Word count")
    axis.set_ylabel("Error rate")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    maximum = float(selected["error_rate"].max())
    axis.set_ylim(0, maximum * 1.22 if maximum else 1.0)
    labels = [
        f"{rate:.2%}\n({errors}/{support:,})"
        for rate, errors, support in zip(
            selected["error_rate"],
            selected["errors"],
            selected["support"],
        )
    ]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8)
    figure.tight_layout()
    _save_svg(figure, path)
    plt.close(figure)


def _plot_topic_slices(topic_summary: pd.DataFrame, path: Path) -> None:
    selected = topic_summary.sort_values("error_rate", ascending=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(selected["value"], selected["error_rate"], color="#d97757")
    axis.set_title("DistilBERT error rate by unsupervised topic")
    axis.set_xlabel("Topic ID (keywords are listed in the topic table)")
    axis.set_ylabel("Error rate")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    maximum = float(selected["error_rate"].max())
    axis.set_ylim(0, maximum * 1.24 if maximum else 1.0)
    labels = [
        f"{rate:.2%}\n({errors}/{support:,})"
        for rate, errors, support in zip(
            selected["error_rate"],
            selected["errors"],
            selected["support"],
        )
    ]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8)
    figure.tight_layout()
    _save_svg(figure, path)
    plt.close(figure)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--slices-output", type=Path, default=DEFAULT_SLICES_PATH)
    parser.add_argument("--topics-output", type=Path, default=DEFAULT_TOPICS_PATH)
    parser.add_argument("--topic-terms-output", type=Path, default=DEFAULT_TOPIC_TERMS_PATH)
    parser.add_argument("--examples-output", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--confusion-figure", type=Path, default=DEFAULT_CONFUSION_FIGURE)
    parser.add_argument("--length-figure", type=Path, default=DEFAULT_LENGTH_FIGURE)
    parser.add_argument("--topic-figure", type=Path, default=DEFAULT_TOPIC_FIGURE)
    parser.add_argument("--source-chunk-size", type=int, default=20_000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--topic-count", type=int, default=DEFAULT_TOPIC_COUNT)
    parser.add_argument("--topic-features", type=int, default=DEFAULT_TOPIC_FEATURES)
    parser.add_argument("--topic-min-df", type=int, default=DEFAULT_TOPIC_MIN_DF)
    parser.add_argument("--topic-top-terms", type=int, default=DEFAULT_TOPIC_TOP_TERMS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing complete error-analysis result set.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    positive_arguments = {
        "source chunk size": args.source_chunk_size,
        "tokenizer batch size": args.tokenizer_batch_size,
        "topic count": args.topic_count,
        "topic features": args.topic_features,
        "topic minimum document frequency": args.topic_min_df,
        "topic top terms": args.topic_top_terms,
    }
    invalid = [name for name, value in positive_arguments.items() if value <= 0]
    if invalid:
        raise ValueError("These arguments must be positive: " + ", ".join(invalid))

    required_paths = [args.predictions, args.source_csv, args.tokenizer]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required local artifacts are missing: " + ", ".join(missing))

    outputs = [
        args.summary,
        args.metrics_output,
        args.slices_output,
        args.topics_output,
        args.topic_terms_output,
        args.examples_output,
        args.confusion_figure,
        args.length_figure,
        args.topic_figure,
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Error-analysis outputs already exist: " + ", ".join(existing))

    predictions_hash = sha256_file(args.predictions)
    if predictions_hash != EXPECTED_PREDICTIONS_SHA256:
        raise ValueError(
            f"Prediction SHA-256 mismatch: expected {EXPECTED_PREDICTIONS_SHA256}, "
            f"got {predictions_hash}"
        )
    predictions = pd.read_csv(args.predictions)
    _validate_predictions(predictions)
    source = load_source_rows(
        predictions,
        args.source_csv,
        chunk_size=args.source_chunk_size,
    )
    analysis, topic_terms = _build_analysis_frame(
        predictions,
        source,
        tokenizer_path=args.tokenizer,
        tokenizer_batch_size=args.tokenizer_batch_size,
        topic_count=args.topic_count,
        topic_features=args.topic_features,
        topic_min_df=args.topic_min_df,
        topic_top_terms=args.topic_top_terms,
    )
    metrics = _metrics_summary(analysis)
    slices = _slice_summary(analysis)
    topics = _topic_summary(slices, topic_terms)
    examples = _error_examples(analysis)

    summary = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "distilbert-base-uncased",
        "predictions_path": args.predictions.as_posix(),
        "predictions_sha256": predictions_hash,
        "source_csv_path": args.source_csv.as_posix(),
        "tokenizer_path": args.tokenizer.as_posix(),
        "test_rows": len(analysis),
        "label_counts": EXPECTED_LABEL_COUNTS,
        "max_length": MAX_LENGTH,
        "truncated_rows": int(analysis["truncated_at_256"].sum()),
        "truncated_fraction": float(analysis["truncated_at_256"].mean()),
        "false_positives": int((analysis["outcome"] == "FP").sum()),
        "false_negatives": int((analysis["outcome"] == "FN").sum()),
        "total_errors": int(analysis["is_error"].sum()),
        "topic_method": {
            "algorithm": "TF-IDF plus non-negative matrix factorization",
            "topics": args.topic_count,
            "maximum_features": args.topic_features,
            "minimum_document_frequency": args.topic_min_df,
            "random_seed": TOPIC_RANDOM_SEED,
            "uses_labels_or_predictions": False,
        },
        "outputs": {
            "metrics": args.metrics_output.as_posix(),
            "slices": args.slices_output.as_posix(),
            "topics": args.topics_output.as_posix(),
            "topic_terms": args.topic_terms_output.as_posix(),
            "error_examples": args.examples_output.as_posix(),
            "confusion_figure": args.confusion_figure.as_posix(),
            "length_figure": args.length_figure.as_posix(),
            "topic_figure": args.topic_figure.as_posix(),
        },
        "policy": {
            "model_or_threshold_changed": False,
            "test_used_for_training_or_selection": False,
            "topic_model_uses_labels_or_error_outcomes": False,
            "text_storage": "750-character excerpts for misclassified rows only",
        },
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".error-analysis-",
        dir=args.summary.parent,
    ) as staging_directory:
        staging_root = Path(staging_directory)
        staged_pairs = [
            (staging_root / args.metrics_output.name, args.metrics_output),
            (staging_root / args.slices_output.name, args.slices_output),
            (staging_root / args.topics_output.name, args.topics_output),
            (staging_root / args.topic_terms_output.name, args.topic_terms_output),
            (staging_root / args.examples_output.name, args.examples_output),
            (staging_root / args.confusion_figure.name, args.confusion_figure),
            (staging_root / args.length_figure.name, args.length_figure),
            (staging_root / args.topic_figure.name, args.topic_figure),
            (staging_root / args.summary.name, args.summary),
        ]
        _write_dataframe(metrics, staged_pairs[0][0])
        _write_dataframe(slices, staged_pairs[1][0])
        _write_dataframe(topics, staged_pairs[2][0])
        _write_dataframe(topic_terms, staged_pairs[3][0])
        _write_dataframe(examples, staged_pairs[4][0])
        _plot_confusion(metrics, staged_pairs[5][0])
        _plot_length_slices(slices, staged_pairs[6][0])
        _plot_topic_slices(topics, staged_pairs[7][0])
        staged_pairs[8][0].write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        for staged, final in staged_pairs:
            final.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(final)

    print(metrics.to_string(index=False))
    print(topics.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
