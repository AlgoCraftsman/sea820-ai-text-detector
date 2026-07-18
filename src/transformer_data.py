"""Reproducible Hugging Face data preparation for Transformer training.

The module deliberately keeps test-set evaluation out of data preparation. It
creates a stratified 80/10/10 split, measures truncation only on training and
validation data, tokenizes without padding, and saves a DatasetDict for later
use with a dynamic-padding data collator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CSV_PATH = Path("data/AI_Human.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/distilbert_cased_seed42")
DEFAULT_RESULTS_PATH = Path("results/transformer_data_preparation.json")
DEFAULT_MANIFEST_PATH = Path("results/transformer_split_manifest.csv.gz")
DEFAULT_CACHE_DIR = Path("data/huggingface-cache")
DEFAULT_MODEL_NAME = "distilbert/distilbert-base-cased"
DEFAULT_MAX_LENGTH = 512
DEFAULT_SEED = 42

URL_RE = re.compile(r"http\S+|www\.\S+", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_for_transformer(text: str) -> str:
    """Collapse whitespace while retaining case, punctuation, and stop words."""

    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_for_dedup(text: str) -> str:
    """Match the Week 1 leakage-guard normalization exactly."""

    normalized = text.lower()
    normalized = URL_RE.sub(" ", normalized)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def _dedup_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def _coerce_label(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Label {value!r} is not numeric") from exc

    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"Label {value!r} is not one of 0 or 1")
    return int(numeric)


def _prepare_batch(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
    texts = batch["text"]
    labels = batch["generated"]
    if len(texts) != len(labels) or len(texts) != len(indices):
        raise ValueError("Text, label, and source-index batch lengths do not match")

    cleaned_texts: list[str] = []
    output_labels: list[int] = []
    hashes: list[str] = []
    non_empty: list[bool] = []

    for text, label in zip(texts, labels):
        if text is None:
            raise ValueError("Found a missing text value")
        if not isinstance(text, str):
            text = str(text)

        cleaned = clean_for_transformer(text)
        cleaned_texts.append(cleaned)
        output_labels.append(_coerce_label(label))
        hashes.append(_dedup_hash(text))
        non_empty.append(bool(cleaned))

    return {
        "text": cleaned_texts,
        "labels": output_labels,
        "source_row_id": indices,
        "dedup_hash": hashes,
        "is_non_empty": non_empty,
    }


def load_and_clean_dataset(
    csv_path: Path,
    *,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    num_proc: int | None = None,
    max_rows: int | None = None,
) -> tuple[Any, dict[str, int]]:
    """Load the local CSV through Hugging Face Datasets and remove leakage risks."""

    try:
        import numpy as np
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-transformer.txt before preparing the dataset"
        ) from exc

    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place AI_Human.csv under data/ "
            "or pass --csv."
        )

    raw = load_dataset(
        "csv",
        data_files={"full": str(csv_path)},
        split="full",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    required_columns = {"text", "generated"}
    missing_columns = required_columns.difference(raw.column_names)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    original_rows = len(raw)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        raw = raw.select(range(min(max_rows, len(raw))))

    prepared = raw.map(
        _prepare_batch,
        batched=True,
        with_indices=True,
        num_proc=num_proc,
        remove_columns=raw.column_names,
        desc="Cleaning text and creating deduplication hashes",
    )
    non_empty = prepared.filter(
        lambda flag: flag,
        input_columns=["is_non_empty"],
        num_proc=num_proc,
        desc="Dropping empty texts",
    ).remove_columns("is_non_empty")

    # Only the compact hash column is materialized in pandas. The 1.1 GB text
    # column remains Arrow-backed, avoiding a second full in-memory text copy.
    hash_frame = non_empty.select_columns(["dedup_hash"]).to_pandas()
    keep_mask = ~hash_frame["dedup_hash"].duplicated(keep="first")
    keep_indices = np.flatnonzero(keep_mask.to_numpy()).tolist()
    deduplicated = non_empty.select(keep_indices)
    deduplicated = deduplicated.class_encode_column("labels")

    stats = {
        "source_rows": original_rows,
        "loaded_rows": len(raw),
        "empty_rows_dropped": len(prepared) - len(non_empty),
        "normalized_duplicates_dropped": len(non_empty) - len(deduplicated),
        "usable_rows": len(deduplicated),
    }
    return deduplicated, stats


def create_stratified_splits(dataset: Any, *, seed: int = DEFAULT_SEED) -> Any:
    """Create an 80/10/10 DatasetDict with a fixed, isolated test split."""

    from datasets import DatasetDict

    train_and_holdout = dataset.train_test_split(
        test_size=0.20,
        seed=seed,
        stratify_by_column="labels",
    )
    validation_and_test = train_and_holdout["test"].train_test_split(
        test_size=0.50,
        seed=seed,
        stratify_by_column="labels",
    )
    return DatasetDict(
        {
            "train": train_and_holdout["train"],
            "validation": validation_and_test["train"],
            "test": validation_and_test["test"],
        }
    )


def build_tokenizer_and_collator(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> tuple[Any, Any]:
    """Load the selected tokenizer and its dynamic-padding collator."""

    try:
        from transformers import AutoTokenizer, DataCollatorWithPadding
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-transformer.txt before loading the tokenizer"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    return tokenizer, collator


def _tokenize_batch(
    batch: dict[str, list[str]],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[list[int]]]:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )


def tokenize_splits(
    splits: Any,
    *,
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_proc: int | None = None,
) -> Any:
    """Tokenize every split without padding so batches can be padded dynamically."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    model_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_limit, int) and model_limit < 1_000_000 and max_length > model_limit:
        raise ValueError(
            f"Requested max_length={max_length} exceeds tokenizer limit {model_limit}"
        )

    return splits.map(
        _tokenize_batch,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        desc=f"Tokenizing with max_length={max_length} and no static padding",
    )


def _token_length_batch(
    batch: dict[str, list[str]],
    *,
    tokenizer: Any,
) -> dict[str, list[int]]:
    encoded = tokenizer(
        batch["text"],
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_length=True,
        verbose=False,
    )
    return {"token_length": encoded["length"]}


def measure_truncation(
    dataset: Any,
    *,
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_proc: int | None = None,
) -> dict[str, Any]:
    """Measure token lengths for a non-test split without retaining token IDs."""

    import numpy as np

    length_dataset = dataset.map(
        _token_length_batch,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        desc="Measuring untruncated token lengths",
    )
    lengths = np.asarray(length_dataset["token_length"], dtype=np.int64)
    truncated = lengths > max_length
    return {
        "rows": int(lengths.size),
        "max_length": int(max_length),
        "rows_exceeding_max_length": int(truncated.sum()),
        "fraction_exceeding_max_length": round(float(truncated.mean()), 6),
        "token_length_p50": round(float(np.percentile(lengths, 50)), 2),
        "token_length_p75": round(float(np.percentile(lengths, 75)), 2),
        "token_length_p90": round(float(np.percentile(lengths, 90)), 2),
        "token_length_p95": round(float(np.percentile(lengths, 95)), 2),
        "token_length_p99": round(float(np.percentile(lengths, 99)), 2),
        "token_length_max": int(lengths.max()),
    }


def _label_counts(split: Any) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for label in split["labels"]:
        counts[str(int(label))] += 1
    return counts


def write_split_manifest(splits: Any, path: Path) -> dict[str, Any]:
    """Save compact, version-controlled split membership for future models."""

    import pandas as pd

    frames = []
    for split_name in ("train", "validation", "test"):
        frame = splits[split_name].select_columns(
            ["source_row_id", "labels"]
        ).to_pandas()
        frame.insert(0, "split", split_name)
        frames.append(frame)

    manifest = pd.concat(frames, ignore_index=True)
    if not manifest["source_row_id"].is_unique:
        raise ValueError("A source row appears in more than one prepared split")

    path.parent.mkdir(parents=True, exist_ok=True)
    compression: str | dict[str, Any]
    if path.suffix == ".gz":
        compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    else:
        compression = "infer"
    manifest.to_csv(path, index=False, compression=compression)
    digest = hashlib.sha256()
    with path.open("rb") as manifest_file:
        for chunk in iter(lambda: manifest_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.as_posix()),
        "rows": len(manifest),
        "sha256": digest.hexdigest(),
    }


def build_summary(
    *,
    csv_path: Path,
    output_dir: Path,
    model_name: str,
    max_length: int,
    seed: int,
    cleaning_stats: dict[str, int],
    splits: Any,
    truncation: dict[str, dict[str, Any]],
    split_manifest: dict[str, Any],
    max_rows: int | None,
) -> dict[str, Any]:
    return {
        "dataset_path": str(csv_path.as_posix()),
        "prepared_dataset_path": str(output_dir.as_posix()),
        "development_row_limit": max_rows,
        "random_seed": seed,
        "split_strategy": "stratified 80/10/10",
        "test_set_policy": "Do not evaluate on test until final model comparison.",
        "model_name": model_name,
        "tokenizer_max_length": max_length,
        "padding": "dynamic per batch",
        "cleaning": cleaning_stats,
        "splits": {
            name: {"rows": len(split), "label_counts": _label_counts(split)}
            for name, split in splits.items()
        },
        "split_manifest": split_manifest,
        "truncation_analysis": truncation,
        "truncation_analysis_excludes_test": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Development-only row limit. Never use for the final prepared dataset.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Choose a new --output-dir to avoid overwriting prepared data."
        )

    dataset, cleaning_stats = load_and_clean_dataset(
        args.csv,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
        max_rows=args.max_rows,
    )
    splits = create_stratified_splits(dataset, seed=args.seed)
    tokenizer, _ = build_tokenizer_and_collator(
        args.model_name,
        cache_dir=args.cache_dir,
    )

    # Max-length selection may use training/validation data, but the test set
    # remains unexamined until the final baseline-vs-Transformer comparison.
    truncation = {
        split_name: measure_truncation(
            splits[split_name],
            tokenizer=tokenizer,
            max_length=args.max_length,
            num_proc=args.num_proc,
        )
        for split_name in ("train", "validation")
    }
    tokenized_splits = tokenize_splits(
        splits,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )
    tokenized_splits.save_to_disk(str(args.output_dir))
    split_manifest = write_split_manifest(tokenized_splits, args.manifest)

    summary = build_summary(
        csv_path=args.csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        seed=args.seed,
        cleaning_stats=cleaning_stats,
        splits=splits,
        truncation=truncation,
        split_manifest=split_manifest,
        max_rows=args.max_rows,
    )
    _write_json(args.results, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
