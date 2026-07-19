# Frozen Test Comparison

## Evaluation policy

The final configuration was frozen before test evaluation. The test split is used once for
the final apples-to-apples comparison; it is not used for feature selection, checkpoint
selection, threshold selection, or any other tuning. No model or preprocessing changes may
be justified using the resulting test metrics.

The evaluator defaults to preflight-only mode and requires the explicit
`--confirm-test-evaluation` flag to score test rows. It refuses to overwrite any existing
final output, making an accidental repeated or partial replacement visible.

## Frozen membership and integrity checks

- Manifest: `results/transformer_split_manifest.csv.gz`
- Expected SHA-256:
  `16a0ac74326c633f390329c287335518c47fcf4728adc923753a68034adbdd45`
- Frozen rows: 371,381 train; 46,423 validation; 46,423 test
- Positive label: `1` (AI-generated); negative label: `0` (human)
- Classic rows are reconstructed from `data/AI_Human.csv` using original
  `source_row_id` values.
- Source labels must match manifest labels for every reconstructed training and test row.
- Prepared Transformer test membership and labels must independently match the manifest.

## Classic baseline configuration

Text is lowercased, URLs matching `http\S+|www\.\S+` are removed, repeated whitespace is
collapsed, and leading/trailing whitespace is stripped. `TfidfVectorizer` removes English
stop words and uses `ngram_range=(1, 2)`, `min_df=5`, `max_features=50000`, and
`sublinear_tf=True`.

The two fixed classic classifiers are:

- `LogisticRegression(max_iter=1000, C=1.0)`
- `LinearSVC(C=1.0)`

All unspecified parameters retain the installed scikit-learn defaults. TF-IDF and both
classifiers are fitted only on frozen training rows. Validation and test rows are never
included in fitting.

## Transformer configuration

- Checkpoint: `checkpoints/distilbert-full-epoch1/checkpoint-23212`
- Prepared DatasetDict: `data/processed/distilbert_cased_seed42`
- Test input columns: `input_ids`, `attention_mask`, and `labels` only
- Dynamic longest-in-batch padding
- Evaluation batch size: 4; FP16 CUDA inference
- Fixed mappings: `0 = human`, `1 = AI-generated`

The checkpoint is the model frozen after the first full training epoch. Its validation F1
was used for checkpoint selection before the test split was evaluated.

## Validation before final scoring

- The complete repository unit-test suite passes, including pure split-membership, label,
  metric, cleaning, and frozen-configuration tests.
- Python syntax compilation passes.
- Preflight passes against the real local manifest, source schema, prepared dataset path,
  and checkpoint.
- Validation-only CUDA inference passes through the installed Hugging Face `Trainer` path
  with dynamic padding and two-class logits.

No frozen test metric was computed during these checks.

## Final artifacts

The confirmed run creates:

- `results/frozen_test_metrics.csv`: accuracy, label-1 precision, recall, F1, available
  test loss, runtime, and score semantics for each model.
- `results/frozen_test_predictions.csv.gz`: source-row ID, true label, predicted labels,
  Logistic Regression label-1 probability, Linear SVM decision score, and DistilBERT
  label-1 probability.
- `results/frozen_test_evaluation.json`: manifest/configuration audit, integrity checks,
  environment details, timing, and the no-post-test-tuning statement.

## Measured results

Pending the one confirmed frozen-test evaluation.

Near-ceiling results must be interpreted as evidence about this dataset, not as proof of a
universally reliable AI-text detector. Generator-specific and dataset-construction artifacts
remain a likely explanation and must be addressed in error analysis, limitations, ethics,
the final report, and the presentation.
