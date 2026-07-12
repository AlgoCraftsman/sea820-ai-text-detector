# Preprocessing Constraints for Model Training

This note records the preprocessing decisions from the Week 1 EDA review and should guide the Week 2 model-training work.

## Verified Dataset Facts

- Use `data/AI_Human.csv` locally, but do not commit it to Git.
- Required columns are `text` and `generated`.
- Cast `generated` to integer labels:
  - `0` = human-written
  - `1` = AI-generated
- Verified EDA facts:
  - 487,235 total rows
  - 305,797 human rows and 181,438 AI-generated rows
  - about 63 percent human and 37 percent AI
  - no missing values in `text` or `generated`
  - no exact duplicate texts
  - 4 empty or whitespace-only texts
  - 23,004 cleaned-text duplicates after empty rows are removed

## Split Rules

- Drop empty or whitespace-only texts before splitting.
- Create a deduplication key before splitting:
  - lowercase text
  - remove URLs
  - collapse repeated whitespace
  - strip leading and trailing whitespace
- Drop duplicate rows by that deduplication key before any train, validation, or test split.
- Use stratified splits because the labels are imbalanced.
- Use a separate validation split for model selection. Recommended:
  - 80 percent train
  - 10 percent validation
  - 10 percent test
- Keep `random_state=42` unless a later experiment explicitly changes it.

## Classic Baseline Preprocessing

- Use the cleaned text field for TF-IDF.
- The Week 1 baseline settings are acceptable:
  - lowercase
  - strip URLs
  - collapse whitespace
  - English stop-word removal in `TfidfVectorizer`
  - unigrams and bigrams
  - `min_df=5`
  - `max_features=50000`
  - `sublinear_tf=True`
- Do not add stemming or lemmatization unless it is a clearly labeled comparison experiment.

## Transformer Preprocessing

- Use minimal cleaning before Transformer tokenization:
  - drop empty texts
  - collapse repeated whitespace
  - strip leading and trailing whitespace
- Do not remove stop words, punctuation, capitalization, or other linguistic cues.
- Use the tokenizer from the selected pretrained model.
- Start with `max_length=512`, truncation enabled, and dynamic padding.
- Track truncation impact because the median text is 363 words and the 75th percentile is 471 words.
- If GPU memory is tight, reduce batch size before reducing sequence length.

## Evaluation Constraints

- Report accuracy, precision, recall, and F1.
- Treat F1 for the AI-generated class as the primary comparison metric.
- Compare all models on the same held-out test set.
- Inspect false positives and false negatives separately, especially:
  - short texts
  - long texts affected by truncation
  - formulaic human writing
  - texts with generator-specific phrasing

## Leakage And Interpretation Notes

- Never fit vectorizers, feature selectors, or other learned preprocessing on test data.
- For pretrained Transformer tokenizers, tokenization can be applied after splitting to keep the workflow clean and reproducible.
- Save split indices or a split manifest if later notebooks need to reproduce the same partitions.
- Do not treat the near-perfect classic baseline as proof that AI-text detection is solved. It likely reflects dataset or generator artifacts and should be discussed in error analysis and ethics.
