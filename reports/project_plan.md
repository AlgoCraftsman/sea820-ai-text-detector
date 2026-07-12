# SEA 820 Project Plan

## Project Goal

Build and evaluate a binary text classifier that distinguishes human-written text from AI-generated text using the Kaggle AI vs Human Text dataset. The project compares a classic TF-IDF baseline against a fine-tuned Transformer model, then analyzes errors, limitations, and ethical risks.

## Approach

Week 1 establishes the foundation: data loading, EDA, preprocessing, TF-IDF feature extraction, and classic baseline models. The current baseline uses cleaned text, cleaned-text deduplication before splitting, stratified train/test evaluation, and TF-IDF unigram/bigram features. Logistic Regression is the required classic baseline, with Multinomial Naive Bayes and Linear SVM included for comparison.

Week 2 will fine-tune a pretrained Transformer, starting with DistilBERT for a practical balance of speed and performance. Transformer preprocessing will stay minimal: drop empty texts, collapse repeated whitespace, preserve punctuation/case/stop words, tokenize with the pretrained tokenizer, use truncation, and track how often examples exceed the token limit. The Transformer will be compared against the same held-out evaluation standard used for the classic baseline.

Week 3 focuses on interpretation and communication: error analysis, ethical discussion, final report, cleaned repository, and presentation slides. Because the classic baseline is already near-perfect, the report will explicitly discuss possible dataset artifacts and why high scores do not prove that AI-text detection is solved in the real world.

## Task Assignments

| Area | George | Kasra |
| --- | --- | --- |
| Data and EDA | Completed dataset loading, class balance, text length, vocabulary, sample text inspection | Review EDA findings and convert them into model-training constraints |
| Classic baseline | Completed TF-IDF preprocessing, Logistic Regression baseline, Naive Bayes and Linear SVM comparisons | Validate baseline metrics and ensure saved results are reusable |
| Transformer model | Support experiment logging and result comparison | Lead Hugging Face Dataset setup, tokenization, GPU fine-tuning, and hyperparameter experiments |
| Evaluation | Help maintain metrics tables and figures | Compare baseline vs Transformer using accuracy, precision, recall, and F1 |
| Error analysis | Inspect false positives/false negatives and summarize qualitative patterns | Extract prediction errors, connect failures to preprocessing/model behavior |
| Ethics and report | Draft dataset, EDA, baseline, and ethics sections | Draft Transformer method, results comparison, error analysis, and limitations |
| Presentation | Prepare slides for dataset, EDA, baseline, and ethics | Prepare slides for Transformer setup, results, comparison, and conclusions |

## Timeline

| Timeframe | Deliverables | Owner |
| --- | --- | --- |
| Week 1 wrap-up | EDA notebook, baseline metrics, reusable result artifacts, preprocessing constraints, project plan | George and Kasra |
| Week 2 start | Hugging Face dataset loading, train/validation/test split, tokenizer setup, first DistilBERT training run | Kasra |
| Week 2 end | Hyperparameter log, best Transformer checkpoint/results, baseline vs Transformer comparison table | Kasra leads, George supports |
| Week 3 start | Error analysis examples, false positive/false negative patterns, ethical implications tied to findings | Both |
| Week 3 end | Final report PDF, presentation slides PDF, cleaned README/repository, final submission check | Both |

## Reproducibility Notes

- Keep `data/AI_Human.csv` local and uncommitted.
- Save reusable metrics and figures under `results/`.
- Keep random seeds fixed unless an experiment explicitly changes them.
- Use stratified splits and report accuracy, precision, recall, and F1.
- Document any hyperparameter run that affects model comparison.
