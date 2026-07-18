# sea820-ai-text-detector

SEA 820 NLP final project comparing classic TF-IDF classifiers and fine-tuned Transformer
models for detecting AI-generated vs. human-written text, with evaluation, error analysis,
and an ethical discussion.

## Project overview

Large Language Models make it increasingly hard to tell human writing from machine writing.
This project builds and compares two families of models for that classification task:

1. Classic baseline: TF-IDF features with Logistic Regression, plus Naive Bayes and Linear
   SVM for comparison.
2. Transformer: a fine-tuned DistilBERT (Week 2).

The classic baseline establishes the score the Transformer must beat.

## Repository structure

```
sea820-ai-text-detector/
├── notebooks/
│   └── aiTextClassifier.ipynb   # Week 1: EDA and classic TF-IDF baseline (this deliverable)
├── data/                        # dataset lands here at runtime (not committed, too large)
├── src/                         # shared preprocessing and training code
├── results/                     # saved metrics and figures
├── reports/                     # written report
├── slides/                      # presentation
└── README.md
```

## Dataset

- Source: [AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) (Kaggle, `shanegerami/ai-vs-human-text`).
- Size: about 487,000 text excerpts, roughly 1.1 GB uncompressed.
- Columns: `text` (the essay) and `generated` (the label), where `0.0` is human-written
  and `1.0` is AI-generated.
- Class balance: about 63% human and 37% AI. This is mildly imbalanced, so we report
  precision, recall, and F1 and use stratified splits rather than accuracy alone.

The notebook loads the data automatically, with no Kaggle credentials required. It resolves the
CSV in this order:

1. the path in the `AIHUMAN_CSV` environment variable, if set;
2. `data/AI_Human.csv`, if present;
3. `AI_Human.csv` in the working directory, if present;
4. `../data/AI_Human.csv`, if present;
5. otherwise it downloads and unzips the public Kaggle archive into `data/`.

## Setup

### Option A: Google Colab (recommended)

Open `notebooks/aiTextClassifier.ipynb` in Colab and run all cells. Every required library
(`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`) is pre-installed.

### Option B: Local

```bash
# Python 3.10+ recommended
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook notebooks/aiTextClassifier.ipynb
```

If you already have the CSV locally, skip the download by pointing the notebook at it:

```bash
export AIHUMAN_CSV=/path/to/AI_Human.csv
```

### Week 2 Transformer environment

Create a separate environment and install the Transformer dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-transformer.txt
```

Confirm that PyTorch sees the RTX 3060 before training:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is unavailable, use the platform-specific command from the
[official PyTorch installer](https://pytorch.org/get-started/locally/) instead of beginning a
long CPU training run.

The RTX 3060 environment was verified with PyTorch 2.12.1 and its CUDA 13.0 wheel on
Windows. Because PyTorch wheel commands are platform-specific, confirm the current command
in the official selector before reproducing the install on another machine.

## How to run

Open the notebook and run all cells. Top to bottom it will:

1. Load the dataset (download if needed).
2. Inspect schema, labels, missing values, duplicates, and class balance.
3. Run EDA: text length, vocabulary, class distribution, sample texts.
4. Save reusable EDA summaries, plots, and baseline tables under `results/`.
5. Build the TF-IDF preprocessing pipeline with a stratified 80/20 split.
6. Train and evaluate three classic classifiers.
7. Produce the comparison table and inspect the most indicative tokens.

Prepare the deterministic Week 2 Hugging Face DatasetDict with:

```bash
python -m src.transformer_data
```

This command creates a stratified 80/10/10 split with seed 42, measures 512-token
truncation only on training/validation data, tokenizes with DistilBERT without static
padding, and saves the prepared DatasetDict under ignored `data/processed/`. The test split
is reserved for the final model comparison, and its membership is frozen in the compressed
`results/transformer_split_manifest.csv.gz` file. See
[`reports/transformer_data_preparation.md`](reports/transformer_data_preparation.md) for the
full policy and smoke-test command.

Run the first validation-only training smoke test with:

```powershell
python -m src.train_transformer `
  --run-name distilbert-smoke-256-128 `
  --output-dir checkpoints/distilbert-smoke-256-128 `
  --train-subset-size 256 `
  --validation-subset-size 128 `
  --epochs 1 `
  --train-batch-size 4 `
  --eval-batch-size 4 `
  --gradient-accumulation-steps 4 `
  --gradient-checkpointing
```

The command loads only the prepared `train` and `validation` splits into `Trainer`, uses
dynamic padding, selects checkpoints by F1 for label 1 (AI-generated), and appends measured
metadata to `results/transformer_experiments.csv`. Checkpoints and raw Trainer output stay
under ignored `checkpoints/`. An unbounded all-row run additionally requires
`--confirm-full-run`; review the smoke-run duration estimate before using it. See
[`reports/transformer_training.md`](reports/transformer_training.md) for the policy and
configuration details.

Full execution takes about 5 to 8 minutes on Colab or a typical laptop. The TF-IDF step over
roughly 490k texts is the main cost.

## Current results (Week 1 baseline)

Test set: 20% stratified hold-out (about 93k texts), after de-duplicating on the cleaned text.
TF-IDF uses unigrams and bigrams with 50k features.

| Model                    | Accuracy | Precision | Recall | F1     |
|--------------------------|:--------:|:---------:|:------:|:------:|
| Linear SVM               | 0.9996   | 0.9997    | 0.9991 | 0.9994 |
| Logistic Regression      | 0.9947   | 0.9974    | 0.9889 | 0.9931 |
| Multinomial Naive Bayes  | 0.9776   | 0.9818    | 0.9599 | 0.9708 |

The classic baseline is already very strong on this dataset. We revisit this in the error
analysis and ethics discussion, since near-perfect separability points to generator-specific
artifacts rather than a robust human-vs-AI signal.

## Roadmap

- Week 1, Foundations and classic model: EDA and TF-IDF baseline (`aiTextClassifier.ipynb`). Done.
- Week 2, Transformer: fine-tune DistilBERT with the Hugging Face `Trainer` API and compare to the baseline.
- Week 3, Analysis and reporting: error analysis, ethical discussion, report, and slides.

## Team

| Member | Focus |
|--------|-------|
| George | Data acquisition, EDA, classic baseline, README |
| Kasra  | Preprocessing constraints, Transformer fine-tuning |
