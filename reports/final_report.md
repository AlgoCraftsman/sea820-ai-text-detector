# Detecting AI-Generated Text: Final Report

SEA 820 NLP Final Project. Team: George and Kasra.

## 1. Introduction

Large Language Models have made machine-generated text hard to tell apart from human writing. This matters for academic integrity, content moderation, and misinformation. The goal of this project is to build, evaluate, and compare NLP models that classify a piece of text as human-written or AI-generated.

We compare two families of models on the same data. The first is a classic baseline that uses TF-IDF features with linear classifiers. The second is a fine-tuned Transformer, DistilBERT. We report accuracy, precision, recall, and F1, run a detailed error analysis on the best model, and discuss the ethical risks of deploying this kind of detector.

The main finding is that the task is easy to score highly on this dataset but hard to solve in a way that generalizes. The strongest model is the classic Linear SVM, and the fine-tuned Transformer does not beat it. The near-perfect scores appear to come from dataset and generator artifacts rather than a robust signal of authorship, which we return to in the error analysis and ethics sections.

## 2. Dataset

We use the AI vs Human Text dataset from Kaggle (`shanegerami/ai-vs-human-text`). Each row has a `text` field and a `generated` label, where `0` is human-written and `1` is AI-generated.

Verified facts from the raw file:

- 487,235 total rows.
- 305,797 human rows and 181,438 AI-generated rows, about 63 percent human and 37 percent AI.
- No missing values in `text` or `generated`.
- No exact duplicate texts, 4 empty or whitespace-only texts, and 23,004 cleaned-text duplicates.

The class imbalance is mild but real, so we use stratified splits and treat F1 for the AI-generated class as the primary metric rather than accuracy alone.

![Class distribution of the dataset. About 63 percent of the rows are human-written and 37 percent are AI-generated.](../results/figures/class_distribution.png)

## 3. Exploratory Data Analysis

The texts are essays. The median length is 363 words and the 75th percentile is 471 words, with a mean of about 393 words and 2,270 characters.

Length differs by class. Human essays are longer on average (mean 422 words, median 389) than AI essays (mean 344 words, median 337), but the distributions overlap heavily. Length on its own is only mildly discriminative, so we do not expect it to explain a near-perfect classifier.

![Word-count distribution by class. Human essays skew slightly longer, but the two distributions overlap heavily.](../results/figures/text_length_distribution.png)

The vocabulary is large. The most frequent content words are almost the same for both classes (`people`, `students`, `school`, `car`, `electoral`, `college`), which shows the two classes share the same prompt topics. This overlap is important later: because both classes write about the same subjects, topic words are weak evidence of authorship, and a model that scores highly is likely keying on style and surface patterns instead.

![Most frequent content words per class. The overlap shows the classes share the same prompt topics.](../results/figures/top_words_by_class.png)

Cleaning is light. We remove empty rows, collapse repeated whitespace, and de-duplicate on a normalized key before splitting.

## 4. Methodology

### 4.1 Preprocessing and splitting

We build a normalized deduplication key by lowercasing the text, removing URLs, and collapsing whitespace, then drop duplicate rows by that key before any split. De-duplicating before the split prevents formatting-only variants from leaking across train and test.

The Week 2 comparison uses a stratified 80/10/10 train, validation, and test split with `random_state=42`. After cleaning and de-duplication, 464,227 usable rows remain, split into 371,381 train, 46,423 validation, and 46,423 test rows. Every split keeps the same class balance of about 61 percent human and 39 percent AI-generated. The Week 1 classic notebook uses an 80/20 split of the same cleaned data, which gives a larger 92,846-row test set that the error analysis in Section 8 reuses. Test rows are held out and only used for final evaluation.

### 4.2 Classic baseline

The classic baseline uses TF-IDF features with linear classifiers, following the Week 1 setup. Text is lowercased, URLs are stripped, and whitespace is collapsed. The `TfidfVectorizer` uses unigrams and bigrams, English stop-word removal, `min_df=5`, `max_features=50000`, and `sublinear_tf=True`.

We train three classic models: Logistic Regression, Multinomial Naive Bayes, and a Linear SVM. Logistic Regression is the required baseline, and the other two are included for comparison. The strongest classic model sets the score the Transformer must beat.

![Week 1 classic baseline comparison of the three classic models. The Linear SVM is the strongest classic model.](../results/figures/baseline_model_comparison.png)

### 4.3 Transformer

The Transformer is `distilbert-base-uncased`, fine-tuned with the Hugging Face `Trainer` API. Transformer preprocessing stays minimal: we drop empty texts and collapse whitespace but keep punctuation, capitalization, and stop words, because those are cues a Transformer can use. Text is tokenized with the pretrained DistilBERT tokenizer, truncated to a maximum length, and padded dynamically per batch with `DataCollatorWithPadding`.

We set the maximum sequence length to 256 tokens, a practical choice for training time on the available GPU. Because a large share of essays are longer than this, the Transformer sees only their first 256 tokens, which is a limitation we note in Section 10.

## 5. Experiments

We ran four controlled fine-tuning experiments that each change one hyperparameter, using a fixed stratified subset of 8,000 training and 2,000 validation rows to keep the search affordable.

| Run | Learning rate | Batch size | Epochs | Validation F1 |
| --- | ---: | ---: | ---: | ---: |
| B | 5e-5 | 4 | 1 | 0.9820 |
| D | 2e-5 | 4 | 2 | 0.9808 |
| C | 2e-5 | 8 | 1 | 0.9776 |
| A | 2e-5 | 4 | 1 | 0.9733 |

Run B, with the larger learning rate of 5e-5, had the highest validation F1 and was selected before any test evaluation. The larger learning rate helped most; a second epoch (Run D) and a larger batch size (Run C) both helped less than raising the learning rate.

The selected configuration was then trained on the full 371,381-row training split for one epoch. On the validation split it reached accuracy 0.9948 and F1 0.9933. Training took about 1.5 hours on an NVIDIA RTX 3060 laptop GPU.

## 6. Results and Model Comparison

All three models are scored on the same 46,423-row held-out test split. The classic models are refit on the same training rows and evaluated on the same test rows as DistilBERT, so the comparison is fair rather than a comparison across different splits.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Linear SVM | 0.9995 | 0.9998 | 0.9989 | 0.9994 |
| Fine-tuned DistilBERT | 0.9949 | 0.9963 | 0.9905 | 0.9934 |
| TF-IDF + Logistic Regression | 0.9944 | 0.9974 | 0.9882 | 0.9927 |

The Linear SVM is the strongest model by every metric, with an F1 of 0.9994. The fine-tuned DistilBERT reaches F1 0.9934, which does not beat the Linear SVM and only roughly matches Logistic Regression. On this dataset, a linear model over TF-IDF features that trains in minutes on a CPU outperforms a Transformer that takes about 1.5 hours to fine-tune on a GPU.

This result is the opposite of what the effort put into each model might suggest, and it is the central point of the analysis. When a simple linear model already scores near the ceiling, there is little headroom for a Transformer to improve, and the gap between the two is within the range that dataset-specific patterns can produce. We do not read this as evidence that Transformers are weak. We read it as evidence that this classification task is being solved by shallow, dataset-specific cues that a linear model captures just as well.

## 7. Model Behavior

To see what the best model learns, we inspect the Linear SVM's TF-IDF coefficients.

The most AI-indicative tokens are formal connectors and essay-structure words: `additionally`, `essay`, `important`, `essential`, `firstly`, `significant`, `conclusion`, `sincerely`, and `ensure`. The most human-indicative tokens are informal or prompt-specific words: `people`, `student`, `paragraph`, `schools`, `students`, `going`, `percent`, `driving`, and `venus`.

This is a useful and slightly uncomfortable result. The model is not detecting authorship in a general sense. It is detecting a register. AI essays in this dataset tend to use polished connectors and explicit structure, while the human essays tend to use informal phrasing, spelling variants, and prompt-specific vocabulary. A classifier that keys on `additionally` and `firstly` will work well here, but it encodes a stylistic stereotype rather than a reliable authorship signal, and that stereotype is exactly what raises the fairness concerns discussed in the ethics section.

## 8. Error Analysis

We analyze the errors of the best model, the TF-IDF Linear SVM. This analysis uses the Week 1 notebook's 92,846-row (80/20) held-out test split, a different partition than the 46,423-row split used for the Section 6 comparison. No model, threshold, or preprocessing rule was changed during the analysis. Following the Week 11 lab, we isolate the false positives and false negatives, compare word counts, inspect examples, and read the SVM's TF-IDF coefficients.

### 8.1 Overall mistakes

On the 92,846-row test split the Linear SVM made only 40 errors: 9 false positives (human text predicted as AI) and 31 false negatives (AI text predicted as human), against 92,806 correct predictions. The errors are asymmetric. False negatives outnumber false positives by more than three to one, so the model was more likely to miss an AI essay than to flag a human essay by mistake. The overall error rate is about 0.04 percent, which is why the near-perfect score should be read with caution rather than treated as a solved task.

### 8.2 Text length

| Outcome | Count | Mean words | Median words |
| --- | ---: | ---: | ---: |
| Correct | 92,806 | 393.6 | 362 |
| False negative | 31 | 304.5 | 356 |
| False positive | 9 | 322.0 | 335 |

| Word count | Test texts | Errors | Error rate |
| --- | ---: | ---: | ---: |
| 0 to 100 | 248 | 1 | 0.40% |
| 101 to 250 | 16,288 | 12 | 0.07% |
| 251 to 500 | 57,156 | 27 | 0.05% |
| 501 or more | 19,154 | 0 | 0.00% |

Misclassified essays are shorter on average than correct ones (about 305 to 322 words versus 394), and all 40 errors are at or below 500 words. The shortest group has the highest rate, but it holds only 248 texts and a single error, so the cautious reading is that mistakes concentrate in shorter essays, which give a TF-IDF model fewer word and bigram cues to work with.

### 8.3 Representative errors

False positives are human essays predicted as AI. They include a personal travel essay with repeated first-person sentence patterns, a moral-argument essay that restates one claim in a formulaic structure, and a career essay built on a repeated question-and-reason pattern. In each case the organized, repetitive structure resembles patterns the SVM associates with AI writing, and it outweighs human-looking spelling and character errors. Grammar mistakes do not protect a human essay from being flagged.

False negatives are AI essays predicted as human. They include an argument about first impressions full of character substitutions and student-level phrasing, a short 116-word education passage with few cues, and an informal problem-solving essay that opens "Wey, it's me, your average 8th grade student!" with an emoji and misspellings. Character corruption, informal student voice, and short length make generated text resemble the human class. The SVM's strongest human-indicative features (`people`, `student`, `paragraph`, `schools`, `students`) are exactly the student-essay vocabulary several of these false negatives use.

### 8.4 Why the model fails

The examples and the coefficient inspection support three cautious explanations. First, formulaic human essays share repeated, organized patterns with AI-labelled essays, so structure alone is weak evidence of authorship. Second, AI text containing misspellings, character substitutions, emojis, and an informal student voice can look like the human class. Third, shorter texts give a TF-IDF model fewer cues, making its vocabulary-based decisions less stable. These are associations in one held-out split, not proof that length causes an error. The near-perfect score likely depends partly on dataset-specific lexical and formatting patterns, so the model should not be treated as a universal AI-text detector.

## 9. Ethical Discussion

The results connect directly to the ethics of deploying an AI-text detector.

The clearest risk is harm to non-native English speakers. Section 7 shows the best model keys on formal connectors and explicit essay structure to mark text as AI. Many non-native writers are taught to write with exactly these connectors and structures. A detector trained on this data could disproportionately flag competent non-native writing as machine-generated, which in an academic setting means false accusations against students who did nothing wrong.

The near-perfect score makes this risk easy to overstate. A model that scores above 0.99 looks authoritative, but Sections 7 and 8 show it is separating a writing register and reacting to lexical cues, not proving who wrote a text. Presenting such an output as proof of authorship would be misleading. The scores describe this dataset, not a universal test of who wrote a text.

There is also dataset and generator bias. The dataset draws on a narrow set of prompts (school essays, the Electoral College, driverless cars, Mars) and a limited set of generators. The models learn artifacts of those prompts and generators, including specific phrasings and even character-corruption patterns, rather than a general property of AI text. A detector this good on this data would likely fail on text from a different generator or a different genre, so a near-perfect score here should not be marketed as a reliable product.

Finally, the two error types carry different costs. A false positive can wrongly penalize a human writer, while a false negative lets AI text pass as human. Our best model made more false negatives than false positives (31 versus 9), but in an academic-integrity setting the false positives are the more serious harm, because they damage individual students. Any real use of a detector should treat its output as a weak signal for human review, never as an automatic judgment.

## 10. Limitations

This study uses one dataset with a limited set of prompts and generators, so the results may not transfer to other sources of AI text. We fine-tuned one Transformer for one epoch at a 256-token limit, so a larger model, more epochs, or a longer sequence length could change the Transformer numbers, though the classic baseline leaves little headroom to beat. The error-analysis patterns are associations from a single held-out split rather than causal claims, and the error analysis covers only the best model rather than every model in the comparison.

## 11. Conclusion

We built and compared classic and Transformer models for detecting AI-generated text. On a shared held-out test split, the strongest model is the classic TF-IDF Linear SVM with an F1 of 0.9994, and the fine-tuned DistilBERT (F1 0.9934) does not beat it and roughly matches Logistic Regression. The scores are near-perfect, but the error analysis and feature inspection show that this reflects dataset and generator artifacts rather than a robust signal of authorship. The model keys on register and lexical cues, and its few errors cluster in shorter, more formulaic essays.

The practical takeaway is twofold. First, a simple linear baseline is a strong and efficient choice for this task, and a Transformer is not required to reach the ceiling on this data. Second, and more important, a high score here does not mean AI-text detection is solved. A detector built on this data would risk flagging non-native writers and would likely fail on new generators, so it should be treated as a weak signal for human review rather than a reliable judgment of authorship.
