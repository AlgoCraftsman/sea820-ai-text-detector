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

The Week 2 notebook evaluates DistilBERT against the Logistic Regression and Linear SVM
classic baselines on identical held-out test membership.

## Repository structure

```
sea820-ai-text-detector/
├── notebooks/
│   ├── aiTextClassifier.ipynb          # Week 1: EDA and classic TF-IDF baseline
│   └── transformer_finetuning.ipynb   # Week 2: Transformer fine-tuning
├── data/                               # dataset lands here at runtime (not committed)
├── src/                                # Week 3 error-analysis utility
├── results/                            # saved metrics and figures
├── reports/                            # written reports
├── slides/                             # presentation
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

Full execution takes about 5 to 8 minutes on Colab or a typical laptop. The TF-IDF step over
roughly 490k texts is the main cost.

### Week 2 Transformer workflow

The lab-style [`notebooks/transformer_finetuning.ipynb`](notebooks/transformer_finetuning.ipynb)
is the complete Week 2 workflow. Run every cell from top to bottom. It loads and cleans the
dataset, creates a stratified 80/10/10 split, compares four configurations on fixed
development subsets, selects by validation F1, trains the selected uncased DistilBERT
configuration at 256 tokens on all training rows, and compares it with the Logistic Regression
and Linear SVM classic baselines on identical test membership.

Run B (`5e-5`, batch size `4`, one epoch) was selected with development-subset F1 `0.981982`.
On the notebook's test split, the Linear SVM was the strongest model with F1 `0.999360`,
followed by DistilBERT at `0.993361` and Logistic Regression at `0.992735`. The fine-tuned
Transformer does not beat the best classic baseline. The saved outputs are:

- `results/split_summary.csv`
- `results/hyperparameter_experiments.csv`
- `results/transformer_test_metrics.csv`
- `results/transformer_test_predictions.csv.gz`
- `results/model_comparison.csv`

See
[`reports/week2_transformer_notebook.md`](reports/week2_transformer_notebook.md) for the
complete run record and validation.

### Week 3 error analysis

After running the Week 2 notebook, analyze the selected DistilBERT's held-out
mistakes with:

```powershell
python -m src.analyze_errors --overwrite
```

The workflow verifies and aligns the saved predictions, identifies every false
positive and false negative, measures error patterns by length, 256-token
truncation, opening style, and label-blind NMF topic, saves representative
examples, and creates three figures. See
[`reports/error_analysis.md`](reports/error_analysis.md) for the findings and
failure hypotheses.

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
- Week 2, Transformer: fine-tune DistilBERT and complete the held-out test comparison. Done.
- Week 3, Analysis and reporting: current-model error analysis done; ethical discussion,
  final report, and slides remain.

## Team

| Member | Focus |
|--------|-------|
| George | Data acquisition, EDA, classic baseline, README |
| Kasra  | Preprocessing constraints, Transformer fine-tuning |
