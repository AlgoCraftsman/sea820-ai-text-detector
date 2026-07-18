# Transformer Data Preparation

This workflow is the first Week 2 deliverable. It creates the frozen data inputs for all
Transformer experiments and prevents the final test set from influencing model selection.

## Decisions

- Load `data/AI_Human.csv` with Hugging Face Datasets.
- Validate the required `text` and `generated` columns and binary labels.
- Drop empty texts and remove normalized duplicates before splitting, using the same
  lowercase, URL-removal, and whitespace-normalization key as the Week 1 baseline.
- Retain minimally cleaned Transformer text: repeated whitespace is collapsed, while case,
  punctuation, and stop words remain intact.
- Create stratified 80/10/10 train/validation/test splits with seed 42.
- Use [`distilbert/distilbert-base-cased`](https://huggingface.co/distilbert/distilbert-base-cased)
  first. The cased checkpoint is intentional because capitalization is one of the linguistic
  cues the Transformer pipeline preserves.
- Tokenize to at most 512 tokens, without static padding. Training will use dynamic padding
  to the longest sequence in each batch.
- Measure untruncated token lengths only on training and validation data. Do not calculate
  test metrics or inspect test prediction errors until the final model comparison.

The saved DatasetDict retains `source_row_id` and `dedup_hash`, so later predictions can be
joined back to source examples and the split membership can be audited. Prepared datasets
are written under `data/processed/` and remain ignored by Git along with the source CSV.
The compressed split manifest and small preparation summary under `results/` are intended
for version control.

The Week 1 baseline metrics were calculated on a 20% holdout, not this new frozen 10% test
split. Keep those scores as the Week 1 baseline record, but rerun the selected classic model
on the manifest's test rows before making the final apples-to-apples model comparison.

## Run

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-transformer.txt
python -m src.transformer_data
```

For a small integration check, use a separate output and result path so it cannot be
confused with the full prepared dataset:

```powershell
python -m src.transformer_data `
  --max-rows 2000 `
  --output-dir data/processed/smoke-test `
  --results data/processed/smoke-test-summary.json `
  --manifest data/processed/smoke-test-manifest.csv.gz
```

Do not commit smoke-test results. A full run writes:

- `data/processed/distilbert_cased_seed42/`: tokenized DatasetDict (ignored)
- `results/transformer_data_preparation.json`: counts, class balance, and train/validation
  truncation statistics
- `results/transformer_split_manifest.csv.gz`: source row ID, label, and frozen split assignment

## Full Dataset Outcome

The full preparation run completed successfully:

| Stage | Rows |
| --- | ---: |
| Source CSV | 487,235 |
| Empty rows removed | 4 |
| Normalized duplicates removed | 23,004 |
| Usable rows | 464,227 |
| Training split | 371,381 |
| Validation split | 46,423 |
| Reserved test split | 46,423 |

At 512 cased DistilBERT tokens, 124,985 training examples (33.6541%) and 15,588
validation examples (33.5782%) require truncation. Training token length has a median of
443, a 75th percentile of 571, and a 95th percentile of 869. These measurements confirm
that truncation affects a substantial portion of the dataset, but 512 remains the correct
first-run limit because it is the model's maximum context size. Error analysis should
separately examine long, truncated examples.

The compressed manifest contains all 464,227 split assignments. Its SHA-256 is
`16a0ac74326c633f390329c287335518c47fcf4728adc923753a68034adbdd45`.

## RTX 3060 Training Guardrails

Data preparation does not allocate GPU memory. For the first fine-tuning run on the 6 GB
RTX 3060, start with mixed precision, a per-device batch size of 4, gradient accumulation
of 4, and dynamic padding at 512 tokens. Reduce the per-device batch size to 2 if an
out-of-memory error occurs. The effective batch size can remain 16 through gradient
accumulation. Final values will be recorded in the experiment log rather than treated as
fixed before the first measured run.
