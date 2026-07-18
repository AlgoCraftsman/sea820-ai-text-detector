"""Fine-tune DistilBERT on the prepared train/validation splits.

The frozen test split is intentionally never passed to ``Trainer``. Development
runs may select deterministic stratified subsets, while an unbounded full run
requires an explicit confirmation flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DEFAULT_DATASET_PATH = Path("data/processed/distilbert_cased_seed42")
DEFAULT_OUTPUT_DIR = Path("checkpoints/distilbert-cased-seed42")
DEFAULT_EXPERIMENT_LOG = Path("results/transformer_experiments.csv")
DEFAULT_CACHE_DIR = Path("data/huggingface-cache")
DEFAULT_MODEL_NAME = "distilbert/distilbert-base-cased"
DEFAULT_SEED = 42

ID2LABEL = {0: "human", 1: "AI-generated"}
LABEL2ID = {label: label_id for label_id, label in ID2LABEL.items()}
MODEL_COLUMNS = ("input_ids", "attention_mask", "labels")

EXPERIMENT_FIELDS = (
    "run_name",
    "run_type",
    "started_at_utc",
    "finished_at_utc",
    "status",
    "model_name",
    "seed",
    "dataset_path",
    "full_train_rows",
    "full_validation_rows",
    "train_rows_used",
    "validation_rows_used",
    "epochs",
    "max_steps",
    "train_batch_size",
    "eval_batch_size",
    "gradient_accumulation_steps",
    "effective_train_batch_size",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "warmup_steps",
    "fp16",
    "gradient_checkpointing",
    "dynamic_padding",
    "test_split_used",
    "train_runtime_seconds",
    "train_samples_per_second",
    "validation_runtime_seconds",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "best_checkpoint",
    "output_dir",
    "resumed_from",
    "estimated_full_run_hours",
    "max_gpu_memory_mb",
    "gpu",
    "cuda_version",
    "python_version",
    "torch_version",
    "transformers_version",
    "datasets_version",
    "accelerate_version",
)


def compute_binary_metrics(eval_prediction: Any) -> dict[str, float]:
    """Compute label-1 metrics; F1 is the primary model-selection score."""

    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(np.asarray(logits), axis=-1)
    labels = np.asarray(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def validate_subset_size(size: int | None, available: int, split_name: str) -> None:
    if size is None:
        return
    if size < 2:
        raise ValueError(f"{split_name} subset size must be at least 2")
    if size > available:
        raise ValueError(
            f"{split_name} subset size {size:,} exceeds {available:,} available rows"
        )


def classify_run(
    train_subset_size: int | None,
    validation_subset_size: int | None,
    max_steps: int,
) -> str:
    """Return an explicit run label so reduced experiments cannot look full."""

    if train_subset_size is not None or validation_subset_size is not None:
        return "development_smoke"
    if max_steps > 0:
        return "bounded_full_data"
    return "full"


def estimate_full_run_hours(
    *,
    full_train_rows: int,
    full_validation_rows: int,
    epochs: float,
    train_samples_per_second: float | None,
    validation_rows_used: int,
    validation_runtime_seconds: float | None,
) -> float | None:
    """Scale measured throughput into a rough train-plus-validation estimate."""

    if not train_samples_per_second or train_samples_per_second <= 0:
        return None
    train_seconds = full_train_rows * epochs / train_samples_per_second
    validation_seconds = 0.0
    if validation_runtime_seconds and validation_rows_used > 0:
        validation_seconds = (
            validation_runtime_seconds
            * full_validation_rows
            / validation_rows_used
            * max(1, math.ceil(epochs))
        )
    return round((train_seconds + validation_seconds) / 3600.0, 2)


def compute_warmup_steps(
    *,
    train_rows: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    epochs: float,
    max_steps: int,
    warmup_ratio: float,
) -> int:
    """Convert the configured ratio for the current Transformers API."""

    if max_steps > 0:
        total_steps = max_steps
    else:
        batches_per_epoch = math.ceil(train_rows / train_batch_size)
        updates_per_epoch = math.ceil(
            batches_per_epoch / gradient_accumulation_steps
        )
        total_steps = math.ceil(updates_per_epoch * epochs)
    return math.ceil(total_steps * warmup_ratio)


def _select_stratified_subset(dataset: Any, size: int | None, seed: int) -> Any:
    if size is None or size == len(dataset):
        return dataset
    return dataset.train_test_split(
        train_size=size,
        seed=seed,
        stratify_by_column="labels",
    )["train"]


def _model_ready_dataset(dataset: Any) -> Any:
    missing = set(MODEL_COLUMNS).difference(dataset.column_names)
    if missing:
        raise ValueError(f"Prepared dataset is missing model columns: {sorted(missing)}")
    # DistilBERT does not accept token_type_ids. Removing every metadata/text
    # column also makes the exact Trainer input contract visible and auditable.
    return dataset.select_columns(list(MODEL_COLUMNS))


def _append_experiment(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=EXPERIMENT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in EXPERIMENT_FIELDS})


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-log", type=Path, default=DEFAULT_EXPERIMENT_LOG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--train-subset-size", type=int, default=None)
    parser.add_argument("--validation-subset-size", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path, or 'latest' to resume the newest output-dir checkpoint.",
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required when running all rows without a max-step bound.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "epochs": args.epochs,
        "train batch size": args.train_batch_size,
        "eval batch size": args.eval_batch_size,
        "gradient accumulation steps": args.gradient_accumulation_steps,
        "learning rate": args.learning_rate,
        "save total limit": args.save_total_limit,
        "logging steps": args.logging_steps,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("warmup ratio must be between 0 and 1")
    if args.weight_decay < 0:
        raise ValueError("weight decay cannot be negative")

    run_type = classify_run(
        args.train_subset_size,
        args.validation_subset_size,
        args.max_steps,
    )
    if run_type == "full" and not args.confirm_full_run:
        raise ValueError(
            "An unbounded full-data run requires --confirm-full-run. Run a measured "
            "development subset first and review its duration estimate."
        )


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)

    try:
        import accelerate
        import datasets
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
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-transformer.txt and a CUDA-enabled PyTorch build first"
        ) from exc

    if not args.dataset_path.is_dir():
        raise FileNotFoundError(f"Prepared DatasetDict not found: {args.dataset_path}")
    use_fp16 = not args.no_fp16
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to begin a long CPU training run")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if args.resume_from_checkpoint is None:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. Choose a new path "
                "or pass --resume-from-checkpoint."
            )

    set_seed(args.seed)
    prepared = load_from_disk(str(args.dataset_path))
    required_splits = {"train", "validation"}
    missing_splits = required_splits.difference(prepared.keys())
    if missing_splits:
        raise ValueError(f"Prepared DatasetDict is missing: {sorted(missing_splits)}")

    full_train_rows = len(prepared["train"])
    full_validation_rows = len(prepared["validation"])
    validate_subset_size(args.train_subset_size, full_train_rows, "train")
    validate_subset_size(args.validation_subset_size, full_validation_rows, "validation")
    train_dataset = _model_ready_dataset(
        _select_stratified_subset(prepared["train"], args.train_subset_size, args.seed)
    )
    validation_dataset = _model_ready_dataset(
        _select_stratified_subset(
            prepared["validation"], args.validation_subset_size, args.seed
        )
    )
    del prepared

    run_type = classify_run(
        args.train_subset_size,
        args.validation_subset_size,
        args.max_steps,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"distilbert-{run_type}-{timestamp}"
    started_at = datetime.now(timezone.utc)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        cache_dir=str(args.cache_dir),
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        cache_dir=str(args.cache_dir),
    )
    warmup_steps = compute_warmup_steps(
        train_rows=len(train_dataset),
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=run_name,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_binary_metrics,
    )

    resume: str | bool | None = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    wall_start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume)
    train_wall_seconds = time.perf_counter() - wall_start
    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    trainer.save_model()
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("validation", validation_metrics)

    train_rate = train_result.metrics.get("train_samples_per_second")
    validation_runtime = validation_metrics.get("validation_runtime")
    estimated_hours = estimate_full_run_hours(
        full_train_rows=full_train_rows,
        full_validation_rows=full_validation_rows,
        epochs=args.epochs,
        train_samples_per_second=float(train_rate) if train_rate else None,
        validation_rows_used=len(validation_dataset),
        validation_runtime_seconds=(
            float(validation_runtime) if validation_runtime else None
        ),
    )
    gpu_name = torch.cuda.get_device_name(0)
    max_gpu_memory_mb = round(torch.cuda.max_memory_allocated(0) / 1024**2, 1)
    finished_at = datetime.now(timezone.utc)
    row = {
        "run_name": run_name,
        "run_type": run_type,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "status": "completed",
        "model_name": args.model_name,
        "seed": args.seed,
        "dataset_path": args.dataset_path.as_posix(),
        "full_train_rows": full_train_rows,
        "full_validation_rows": full_validation_rows,
        "train_rows_used": len(train_dataset),
        "validation_rows_used": len(validation_dataset),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_train_batch_size": (
            args.train_batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "fp16": use_fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dynamic_padding": True,
        "test_split_used": False,
        "train_runtime_seconds": round(
            float(train_result.metrics.get("train_runtime", train_wall_seconds)), 3
        ),
        "train_samples_per_second": train_rate,
        "validation_runtime_seconds": validation_runtime,
        "accuracy": validation_metrics.get("validation_accuracy"),
        "precision": validation_metrics.get("validation_precision"),
        "recall": validation_metrics.get("validation_recall"),
        "f1": validation_metrics.get("validation_f1"),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "output_dir": args.output_dir.as_posix(),
        "resumed_from": args.resume_from_checkpoint,
        "estimated_full_run_hours": estimated_hours,
        "max_gpu_memory_mb": max_gpu_memory_mb,
        "gpu": gpu_name,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "accelerate_version": accelerate.__version__,
    }
    _append_experiment(args.experiment_log, row)
    print(json.dumps(row, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
