"""Evaluate frozen classic baselines and DistilBERT on the frozen test split.

The default mode performs preflight validation only. The test split is scored only
when ``--confirm-test-evaluation`` is supplied. Final outputs are write-once: the
script refuses to run if any requested result artifact already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC


DEFAULT_CSV_PATH = Path("data/AI_Human.csv")
DEFAULT_MANIFEST_PATH = Path("results/transformer_split_manifest.csv.gz")
DEFAULT_PREPARED_DATASET_PATH = Path("data/processed/distilbert_cased_seed42")
DEFAULT_CHECKPOINT_PATH = Path(
    "checkpoints/distilbert-full-epoch1/checkpoint-23212"
)
DEFAULT_METRICS_PATH = Path("results/frozen_test_metrics.csv")
DEFAULT_PREDICTIONS_PATH = Path("results/frozen_test_predictions.csv.gz")
DEFAULT_AUDIT_PATH = Path("results/frozen_test_evaluation.json")
DEFAULT_TRANSFORMER_OUTPUT_DIR = Path("runs/frozen-test-evaluation")

EXPECTED_MANIFEST_SHA256 = (
    "16a0ac74326c633f390329c287335518c47fcf4728adc923753a68034adbdd45"
)
EXPECTED_SPLIT_COUNTS = {"train": 371_381, "validation": 46_423, "test": 46_423}
EXPECTED_LABEL_COUNTS = {
    "train": {0: 227_572, 1: 143_809},
    "validation": {0: 28_447, 1: 17_976},
    "test": {0: 28_446, 1: 17_977},
}
EXPECTED_LABELS = {0: "human", 1: "AI-generated"}
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}

URL_RE = re.compile(r"http\S+|www\.\S+", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FrozenConfiguration:
    """The fixed final-comparison settings selected before test evaluation."""

    stop_words: str = "english"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 5
    max_features: int = 50_000
    sublinear_tf: bool = True
    logistic_max_iter: int = 1_000
    logistic_c: float = 1.0
    linear_svc_c: float = 1.0
    transformer_eval_batch_size: int = 4
    seed: int = 42


FROZEN_CONFIGURATION = FrozenConfiguration()


@dataclass(frozen=True)
class ManifestLookup:
    split_codes: np.ndarray
    labels: np.ndarray
    counts: dict[str, int]


def clean_for_classic(text: str) -> str:
    """Apply the exact Week 1 classic-baseline cleaning rules."""

    cleaned = text.lower()
    cleaned = URL_RE.sub(" ", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_payload(config: FrozenConfiguration) -> dict[str, Any]:
    return {
        "cleaning": {
            "lowercase": True,
            "url_pattern": r"http\S+|www\.\S+",
            "collapse_whitespace": True,
            "strip": True,
        },
        "tfidf": {
            "stop_words": config.stop_words,
            "ngram_range": list(config.ngram_range),
            "min_df": config.min_df,
            "max_features": config.max_features,
            "sublinear_tf": config.sublinear_tf,
        },
        "models": {
            "logistic_regression": {
                "class": "LogisticRegression",
                "max_iter": config.logistic_max_iter,
                "C": config.logistic_c,
                "other_parameters": "scikit-learn defaults",
            },
            "linear_svm": {
                "class": "LinearSVC",
                "C": config.linear_svc_c,
                "other_parameters": "scikit-learn defaults",
            },
            "distilbert": {
                "checkpoint": DEFAULT_CHECKPOINT_PATH.as_posix(),
                "eval_batch_size": config.transformer_eval_batch_size,
                "dynamic_padding": True,
                "retained_columns": ["input_ids", "attention_mask", "labels"],
            },
        },
        "positive_label": 1,
        "seed": config.seed,
    }


def validate_frozen_configuration(config: FrozenConfiguration) -> None:
    """Reject accidental changes to settings frozen before test evaluation."""

    if config != FROZEN_CONFIGURATION:
        raise ValueError(
            "Final comparison configuration differs from the frozen Week 1/Week 2 "
            "settings; do not tune configuration during test evaluation"
        )


def compute_binary_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute label-1 accuracy, precision, recall, and F1."""

    truth = np.asarray(y_true, dtype=np.int64)
    predictions = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != predictions.shape:
        raise ValueError("True and predicted label arrays must have the same shape")
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_manifest_lookup(manifest: pd.DataFrame) -> ManifestLookup:
    """Validate manifest membership and build constant-time source-row lookups."""

    required = {"split", "source_row_id", "labels"}
    if set(manifest.columns) != required:
        raise ValueError(
            f"Manifest columns must be exactly {sorted(required)}, got "
            f"{sorted(manifest.columns)}"
        )
    if manifest.empty:
        raise ValueError("Manifest is empty")
    if manifest["source_row_id"].isna().any() or manifest["labels"].isna().any():
        raise ValueError("Manifest source IDs and labels cannot be missing")

    source_ids = manifest["source_row_id"].to_numpy(dtype=np.int64)
    labels = manifest["labels"].to_numpy(dtype=np.int8)
    if (source_ids < 0).any():
        raise ValueError("Manifest source_row_id values must be nonnegative")
    if pd.Series(source_ids).duplicated().any():
        raise ValueError("A source row appears in more than one manifest split")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Manifest labels must contain only 0 and 1")

    split_names = manifest["split"].astype(str).to_numpy()
    unknown = set(np.unique(split_names)).difference(SPLIT_CODES)
    if unknown:
        raise ValueError(f"Manifest contains unknown split names: {sorted(unknown)}")

    max_source_id = int(source_ids.max())
    split_lookup = np.full(max_source_id + 1, -1, dtype=np.int8)
    label_lookup = np.full(max_source_id + 1, -1, dtype=np.int8)
    split_lookup[source_ids] = np.fromiter(
        (SPLIT_CODES[name] for name in split_names),
        dtype=np.int8,
        count=len(split_names),
    )
    label_lookup[source_ids] = labels
    counts = {
        split_name: int((split_names == split_name).sum())
        for split_name in SPLIT_CODES
    }
    return ManifestLookup(split_lookup, label_lookup, counts)


def validate_expected_manifest(manifest: pd.DataFrame, lookup: ManifestLookup) -> None:
    if lookup.counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Manifest split counts differ from frozen counts: {lookup.counts}"
        )
    for split_name, expected_counts in EXPECTED_LABEL_COUNTS.items():
        actual = (
            manifest.loc[manifest["split"] == split_name, "labels"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        actual = {int(label): int(count) for label, count in actual.items()}
        if actual != expected_counts:
            raise ValueError(
                f"Manifest {split_name} label counts differ from frozen counts: {actual}"
            )


def _coerce_source_labels(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("Source labels must contain only numeric 0 and 1 values")
    return numeric.astype(np.int8)


def iter_split_texts(
    csv_path: Path,
    lookup: ManifestLookup,
    split_name: str,
    *,
    source_ids: list[int],
    labels: list[int],
    chunk_size: int,
) -> Iterator[str]:
    """Stream cleaned rows selected by original CSV row ID and verify labels."""

    if split_name not in SPLIT_CODES:
        raise ValueError(f"Unknown split: {split_name}")
    split_code = SPLIT_CODES[split_name]
    offset = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=["text", "generated"],
        chunksize=chunk_size,
    ):
        row_ids = np.arange(offset, offset + len(chunk), dtype=np.int64)
        in_lookup = row_ids < len(lookup.split_codes)
        selected = np.zeros(len(chunk), dtype=bool)
        selected[in_lookup] = lookup.split_codes[row_ids[in_lookup]] == split_code
        selected_positions = np.flatnonzero(selected)
        if selected_positions.size:
            selected_ids = row_ids[selected_positions]
            source_labels = _coerce_source_labels(
                chunk.iloc[selected_positions]["generated"]
            )
            expected_labels = lookup.labels[selected_ids]
            mismatches = source_labels != expected_labels
            if mismatches.any():
                bad_id = int(selected_ids[np.flatnonzero(mismatches)[0]])
                raise ValueError(
                    f"Source label does not match manifest label at source_row_id={bad_id}"
                )
            selected_texts = chunk.iloc[selected_positions]["text"]
            if selected_texts.isna().any():
                raise ValueError(f"Source {split_name} membership contains missing text")
            source_ids.extend(selected_ids.tolist())
            labels.extend(source_labels.astype(int).tolist())
            for text in selected_texts.astype(str):
                yield clean_for_classic(text)
        offset += len(chunk)

    expected_count = lookup.counts[split_name]
    if len(source_ids) != expected_count:
        raise ValueError(
            f"Reconstructed {len(source_ids):,} {split_name} rows; "
            f"expected {expected_count:,}"
        )


def validate_checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    config_path = checkpoint_path / "config.json"
    model_path = checkpoint_path / "model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"Frozen checkpoint is missing config.json or model.safetensors: "
            f"{checkpoint_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    id2label = {int(key): value for key, value in config.get("id2label", {}).items()}
    label2id = {key: int(value) for key, value in config.get("label2id", {}).items()}
    if id2label != EXPECTED_LABELS:
        raise ValueError(f"Checkpoint id2label mapping is not frozen mapping: {id2label}")
    if label2id != {value: key for key, value in EXPECTED_LABELS.items()}:
        raise ValueError(f"Checkpoint label2id mapping is not frozen mapping: {label2id}")
    if config.get("architectures") != ["DistilBertForSequenceClassification"]:
        raise ValueError("Checkpoint is not the frozen DistilBERT classifier")
    if int(config.get("max_position_embeddings", -1)) != 512:
        raise ValueError("Checkpoint does not use the frozen 512-position configuration")
    return config


def validate_prepared_membership(
    source_ids: np.ndarray,
    labels: np.ndarray,
    lookup: ManifestLookup,
    split_name: str,
) -> None:
    if source_ids.ndim != 1 or labels.ndim != 1 or source_ids.shape != labels.shape:
        raise ValueError("Prepared source IDs and labels must be aligned one-dimensional arrays")
    if len(source_ids) != lookup.counts[split_name]:
        raise ValueError("Prepared split row count does not match manifest")
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("Prepared split contains duplicate source IDs")
    if (source_ids < 0).any() or (source_ids >= len(lookup.split_codes)).any():
        raise ValueError("Prepared split contains source IDs outside manifest lookup")
    if not np.all(lookup.split_codes[source_ids] == SPLIT_CODES[split_name]):
        raise ValueError("Prepared split membership differs from frozen manifest")
    if not np.array_equal(lookup.labels[source_ids], labels.astype(np.int8)):
        raise ValueError("Prepared labels differ from frozen manifest labels")


def _fit_classic_models(
    *,
    csv_path: Path,
    lookup: ManifestLookup,
    config: FrozenConfiguration,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    train_ids: list[int] = []
    train_labels: list[int] = []
    vectorizer = TfidfVectorizer(
        stop_words=config.stop_words,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
    )

    vectorizer_start = time.perf_counter()
    x_train = vectorizer.fit_transform(
        iter_split_texts(
            csv_path,
            lookup,
            "train",
            source_ids=train_ids,
            labels=train_labels,
            chunk_size=chunk_size,
        )
    )
    train_y = np.asarray(train_labels, dtype=np.int8)

    test_ids: list[int] = []
    test_labels: list[int] = []
    x_test = vectorizer.transform(
        iter_split_texts(
            csv_path,
            lookup,
            "test",
            source_ids=test_ids,
            labels=test_labels,
            chunk_size=chunk_size,
        )
    )
    vectorizer_seconds = time.perf_counter() - vectorizer_start
    test_y = np.asarray(test_labels, dtype=np.int8)

    predictions = pd.DataFrame(
        {"source_row_id": test_ids, "true_label": test_y.astype(int)}
    )
    metric_rows: list[dict[str, Any]] = []

    logistic = LogisticRegression(
        max_iter=config.logistic_max_iter,
        C=config.logistic_c,
    )
    logistic_start = time.perf_counter()
    logistic.fit(x_train, train_y)
    logistic_pred = logistic.predict(x_test).astype(np.int8)
    logistic_score = logistic.predict_proba(x_test)[:, 1]
    logistic_seconds = time.perf_counter() - logistic_start
    predictions["logistic_regression_pred"] = logistic_pred
    predictions["logistic_regression_probability"] = logistic_score
    metric_rows.append(
        {
            "model": "Logistic Regression",
            **compute_binary_metrics(test_y, logistic_pred),
            "test_loss": "",
            "evaluation_runtime_seconds": logistic_seconds,
            "score_type": "label_1_probability",
            "train_rows": len(train_y),
            "test_rows": len(test_y),
        }
    )

    linear_svm = LinearSVC(C=config.linear_svc_c)
    svm_start = time.perf_counter()
    linear_svm.fit(x_train, train_y)
    svm_pred = linear_svm.predict(x_test).astype(np.int8)
    svm_score = linear_svm.decision_function(x_test)
    svm_seconds = time.perf_counter() - svm_start
    predictions["linear_svm_pred"] = svm_pred
    predictions["linear_svm_decision_score"] = svm_score
    metric_rows.append(
        {
            "model": "Linear SVM",
            **compute_binary_metrics(test_y, svm_pred),
            "test_loss": "",
            "evaluation_runtime_seconds": svm_seconds,
            "score_type": "label_1_decision_function",
            "train_rows": len(train_y),
            "test_rows": len(test_y),
        }
    )

    details = {
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_matrix_shape": list(x_train.shape),
        "test_matrix_shape": list(x_test.shape),
        "vectorization_runtime_seconds": vectorizer_seconds,
        "source_train_labels_verified": len(train_y),
        "source_test_labels_verified": len(test_y),
    }
    return metric_rows, predictions, details


def _evaluate_transformer(
    *,
    prepared_dataset_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    lookup: ManifestLookup,
    config: FrozenConfiguration,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    import torch
    import transformers
    from datasets import load_from_disk
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; frozen Transformer evaluation requires GPU")

    set_seed(config.seed)
    prepared = load_from_disk(str(prepared_dataset_path))
    if "test" not in prepared:
        raise ValueError("Prepared DatasetDict is missing the frozen test split")
    test = prepared["test"]
    required_columns = {"input_ids", "attention_mask", "labels", "source_row_id"}
    missing_columns = required_columns.difference(test.column_names)
    if missing_columns:
        raise ValueError(f"Prepared test split is missing: {sorted(missing_columns)}")

    source_ids = np.asarray(test["source_row_id"], dtype=np.int64)
    labels = np.asarray(test["labels"], dtype=np.int8)
    validate_prepared_membership(source_ids, labels, lookup, "test")
    model_test = test.select_columns(["input_ids", "attention_mask", "labels"])
    del prepared, test

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path),
        use_fast=True,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_path),
        local_files_only=True,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        do_train=False,
        do_eval=False,
        do_predict=True,
        per_device_eval_batch_size=config.transformer_eval_batch_size,
        fp16=True,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=lambda prediction: compute_binary_metrics(
            prediction.label_ids,
            np.argmax(prediction.predictions, axis=-1),
        ),
    )

    prediction_output = trainer.predict(model_test, metric_key_prefix="test")
    logits = np.asarray(prediction_output.predictions)
    predicted = np.argmax(logits, axis=-1).astype(np.int8)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    metrics = {
        "model": "DistilBERT",
        **compute_binary_metrics(labels, predicted),
        "test_loss": prediction_output.metrics.get("test_loss", ""),
        "evaluation_runtime_seconds": prediction_output.metrics.get("test_runtime", ""),
        "score_type": "label_1_probability",
        "train_rows": EXPECTED_SPLIT_COUNTS["train"],
        "test_rows": len(labels),
    }
    prediction_frame = pd.DataFrame(
        {
            "source_row_id": source_ids,
            "distilbert_pred": predicted,
            "distilbert_probability": probabilities[:, 1],
        }
    )
    details = {
        "prepared_test_rows_verified": len(labels),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return metrics, prediction_frame, details


def _atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression: str | dict[str, Any]
    if path.suffix == ".gz":
        compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    else:
        compression = "infer"
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_output_paths(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Final evaluation outputs are write-once and already exist: "
            + ", ".join(existing)
        )


def run_preflight(args: argparse.Namespace) -> tuple[pd.DataFrame, ManifestLookup, str]:
    validate_frozen_configuration(FROZEN_CONFIGURATION)
    required_paths = [args.csv, args.manifest, args.prepared_dataset, args.checkpoint]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required local artifacts are missing: " + ", ".join(missing))
    _validate_output_paths([args.metrics, args.predictions, args.audit])

    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Manifest SHA-256 mismatch: expected {EXPECTED_MANIFEST_SHA256}, "
            f"got {manifest_hash}"
        )
    manifest = pd.read_csv(args.manifest)
    lookup = build_manifest_lookup(manifest)
    validate_expected_manifest(manifest, lookup)
    source_columns = set(pd.read_csv(args.csv, nrows=0).columns)
    if not {"text", "generated"}.issubset(source_columns):
        raise ValueError("Source CSV must contain text and generated columns")
    validate_checkpoint_config(args.checkpoint)
    return manifest, lookup, manifest_hash


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--prepared-dataset", type=Path, default=DEFAULT_PREPARED_DATASET_PATH
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument(
        "--transformer-output-dir", type=Path, default=DEFAULT_TRANSFORMER_OUTPUT_DIR
    )
    parser.add_argument("--csv-chunk-size", type=int, default=10_000)
    parser.add_argument(
        "--confirm-test-evaluation",
        action="store_true",
        help="Required to score the frozen test split and create final artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.csv_chunk_size <= 0:
        raise ValueError("CSV chunk size must be positive")
    _, lookup, manifest_hash = run_preflight(args)

    preflight = {
        "status": "preflight_passed",
        "manifest_sha256": manifest_hash,
        "split_counts": lookup.counts,
        "checkpoint": args.checkpoint.as_posix(),
        "configuration": configuration_payload(FROZEN_CONFIGURATION),
    }
    if not args.confirm_test_evaluation:
        print(json.dumps(preflight, indent=2))
        print("Frozen test metrics were not evaluated. Re-run with --confirm-test-evaluation.")
        return 0

    started_at = datetime.now(timezone.utc)
    classic_metrics, predictions, classic_details = _fit_classic_models(
        csv_path=args.csv,
        lookup=lookup,
        config=FROZEN_CONFIGURATION,
        chunk_size=args.csv_chunk_size,
    )
    transformer_metrics, transformer_predictions, transformer_details = (
        _evaluate_transformer(
            prepared_dataset_path=args.prepared_dataset,
            checkpoint_path=args.checkpoint,
            output_dir=args.transformer_output_dir,
            lookup=lookup,
            config=FROZEN_CONFIGURATION,
        )
    )
    predictions = predictions.merge(
        transformer_predictions,
        on="source_row_id",
        how="left",
        validate="one_to_one",
    )
    if predictions[["distilbert_pred", "distilbert_probability"]].isna().any().any():
        raise ValueError("Transformer predictions do not cover every frozen test row")

    metric_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "test_loss",
        "evaluation_runtime_seconds",
        "score_type",
        "train_rows",
        "test_rows",
    ]
    metrics = pd.DataFrame(
        [*classic_metrics, transformer_metrics], columns=metric_columns
    )
    finished_at = datetime.now(timezone.utc)
    audit = {
        **preflight,
        "status": "completed",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
        "source_csv": args.csv.as_posix(),
        "prepared_dataset": args.prepared_dataset.as_posix(),
        "metrics_path": args.metrics.as_posix(),
        "predictions_path": args.predictions.as_posix(),
        "classic_details": classic_details,
        "transformer_details": transformer_details,
        "test_policy": {
            "configuration_frozen_before_test": True,
            "test_used_for_model_selection": False,
            "tuning_after_test_evaluation": False,
            "statement": (
                "The frozen test split was evaluated for the final comparison only. "
                "No model or preprocessing tuning occurred after test results were observed."
            ),
        },
        "environment": {
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
        },
    }

    _atomic_write_dataframe(metrics, args.metrics)
    _atomic_write_dataframe(predictions, args.predictions)
    _atomic_write_json(audit, args.audit)
    print(metrics.to_string(index=False))
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
