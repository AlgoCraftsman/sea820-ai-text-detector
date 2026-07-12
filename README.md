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
├── src/                         # shared helper code
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

## How to run

Open the notebook and run all cells. Top to bottom it will:

1. Load the dataset (download if needed).
2. Inspect schema, labels, missing values, duplicates, and class balance.
3. Run EDA: text length, vocabulary, class distribution, sample texts.
4. Build the TF-IDF preprocessing pipeline with a stratified 80/20 split.
5. Train and evaluate three classic classifiers.
6. Produce the comparison table and inspect the most indicative tokens.

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
