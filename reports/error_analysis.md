# Error Analysis of the Selected DistilBERT

## Scope and safeguards

This analysis examines the held-out predictions exported by
`notebooks/transformer_finetuning.ipynb`. It uses the selected uncased
DistilBERT, its 256-token limit, and the exact 46,423-row test membership used
for the reported Week 2 metrics. The prediction artifact is verified by
SHA-256, and every `source_row_id` and label is aligned with the original CSV
before analysis.

No model, threshold, preprocessing rule, or checkpoint was changed. Topic
discovery uses TF-IDF and eight-component non-negative matrix factorization
(NMF) across all held-out texts. It does not use labels, predictions, or error
outcomes, so the topic categories were not chosen to make the errors look more
systematic.

## Overall mistakes

| Outcome | Rows |
| --- | ---: |
| True negatives | 28,380 |
| False positives | 67 |
| False negatives | 171 |
| True positives | 17,805 |
| Total errors | 238 |

![DistilBERT held-out confusion matrix](../results/figures/error_confusion_matrix.svg)

The model achieved accuracy `0.994873`, precision `0.996251`, recall
`0.990487`, and F1 `0.993361`, matching the Week 2 result exactly. Errors were
asymmetric: false negatives outnumbered false positives by about 2.55 to 1.
The AI-generated class had an error rate of `0.9513%`, compared with `0.2355%`
for human-written rows. In this test set, the model was more likely to miss an
AI-generated essay than to flag a human essay incorrectly.

The mistakes were usually confident. The median probability assigned to the
incorrect predicted class was `0.997661`; 217 of 238 errors (`91.2%`) were at
least 0.90 confident, and 192 (`80.7%`) were at least 0.99 confident. A high
DistilBERT probability therefore should not be interpreted as reliable proof
of authorship.

## Text length and truncation

| Word count | Support | Errors | Error rate | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-100 | 127 | 0 | 0.0000% | 0 | 0 |
| 101-250 | 8,004 | 61 | 0.7621% | 12 | 49 |
| 251-500 | 28,572 | 137 | 0.4795% | 21 | 116 |
| 501-750 | 7,791 | 33 | 0.4236% | 27 | 6 |
| 751+ | 1,929 | 7 | 0.3629% | 7 | 0 |

![Error rate by word-count bin](../results/figures/error_rate_by_length.svg)

The very shortest bin contained only 127 examples and no errors, so it does not
support the claim that extremely short text is consistently difficult.
Moderately short essays of 101-250 words were the hardest length group, with an
error rate about 1.59 times the 251-500-word rate. Their errors were primarily
false negatives.

Longer groups had lower total error rates but a different error mix. All seven
errors above 750 words were false positives, and 27 of 33 errors between 501
and 750 words were false positives. A plausible explanation is that long,
well-structured human essays can resemble the organization and fluency the
model associates with generated prose, while only the first 256 tokens are
available to the classifier.

The tokenizer found that 41,309 test rows (`88.98%`) exceeded 256 tokens.
Truncated rows had a `0.5011%` error rate, compared with `0.6062%` for
non-truncated rows. This descriptive result does not show that truncation
helps: text length, label balance, topic, and writing style differ between the
groups. It does show that truncation alone does not explain most mistakes.

## Writing style

| Opening style | Support | Errors | Error rate | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opening question | 4,859 | 35 | 0.7203% | 5 | 30 |
| Other | 36,532 | 191 | 0.5228% | 61 | 130 |
| Salutation | 5,032 | 12 | 0.2385% | 1 | 11 |

Essays opening with a question had the highest error rate, about 1.38 times the
rate for other openings. Thirty of their 35 errors were false negatives.
Rhetorical questions and direct-address school essays occur in both labels, so
they provide weak authorship evidence even though they are recurring stylistic
cues.

Manual inspection also found spelling errors, character substitutions,
informal phrasing, repeated claims, and rigid multi-reason essay structures in
both false positives and false negatives. The overlap suggests that the model
has learned dataset-specific style and prompt patterns that do not map cleanly
to human versus AI authorship.

## Topic patterns

The NMF topics are descriptive groups identified by their highest-weight
terms. They are not manually assigned ground-truth topics.

| Topic | Interpretation from top terms | Support | Errors | Error rate | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Topic 1 | General opinion and personal reasoning | 8,661 | 89 | 1.0276% | 29 | 60 |
| Topic 5 | School, students, classes, and teachers | 13,729 | 80 | 0.5827% | 24 | 56 |
| Topic 7 | Driving, driverless cars, and phones | 5,168 | 23 | 0.4450% | 3 | 20 |
| Topic 6 | Technology, coding, facial expressions | 3,084 | 9 | 0.2918% | 1 | 8 |
| Topic 4 | Venus and planetary exploration | 2,543 | 7 | 0.2753% | 2 | 5 |
| Topic 3 | Car use, transportation, and pollution | 5,782 | 15 | 0.2594% | 7 | 8 |
| Topic 8 | Mars, the "face," and landforms | 2,339 | 6 | 0.2565% | 1 | 5 |
| Topic 2 | Electoral College and voting | 5,117 | 9 | 0.1759% | 0 | 9 |

![Error rate by unsupervised topic](../results/figures/error_rate_by_topic.svg)

General-opinion essays had the highest topic error rate, about twice the
overall `0.5127%` rate. Their broad vocabulary and varied personal arguments
make them a heterogeneous group with fewer topic-specific cues. School essays
produced the second-most elevated rate and 80 errors. Together, the
general-opinion and school topics account for 169 of 238 errors (`71.0%`).
These are also the two largest topics, so both support and rate should be
reported rather than raw error count alone.

The topic result is evidence of association, not causation. Topic vocabulary
may be entangled with prompt family, generator, text length, corruption style,
or class prevalence.

## Representative false positives

A false positive is a human-labeled row predicted as AI-generated.

| Source row | Incorrect-class confidence | Short excerpt | Hypothesis |
| ---: | ---: | --- | --- |
| 9615 | 0.999990 | "Summary The 'Face on Mars' was found during a search..." | The encyclopedic summary format and familiar Mars-prompt vocabulary resemble generated explanatory prose. |
| 62511 | 0.999966 | "Dear TEACHER_NAME HEY IM SORRY... but i don't agree..." | Despite informal spelling and punctuation, the direct-address school prompt and repeated reasons may dominate the model's decision. |
| 264816 | 0.999968 | "In the life today, education is important for people..." | The essay combines a formulaic education argument with extensive word corruption, a pattern that may resemble synthetic or transformed training examples. |

These cases show that grammatical errors do not reliably protect human text
from being flagged. Topic, prompt structure, and organization can outweigh
surface-level human imperfections.

## Representative false negatives

A false negative is an AI-labeled row predicted as human-written.

| Source row | Incorrect-class confidence | Short excerpt | Hypothesis |
| ---: | ---: | --- | --- |
| 237649 | 0.999986 | "Dear state senator, I want to talk about how we elect the president." | The simple student voice, direct address, and spelling substitutions make the generated essay resemble an authentic classroom response. |
| 151905 | 0.999985 | "Last MMK, I had a math test on my first period." | A personal anecdote plus numerous character substitutions provides strong human-like noise. |
| 161325 | 0.999985 | "Do you even Wonder how school would be like if you could pick..." | Rhetorical questioning, repetition, and student-level errors imitate the style of human school essays. |

The test set contains visibly near-parallel variants of some false-negative
essays. For example, rows 237649 and 89430 share the same Electoral College
letter structure, while rows 161325 and 131400 share the same elective-class
argument with different character substitutions. This does not prove train-test
leakage, but it suggests that prompt families and text perturbations are
important dataset characteristics. Exact duplicate removal cannot eliminate
semantic near-duplicates or mechanically altered variants.

## Why the model fails

The evidence supports four cautious hypotheses:

1. **Prompt and topic cues overlap across labels.** School letters, rhetorical
   questions, and formulaic arguments occur in both human and AI-labeled rows.
2. **Artificial-looking corruption weakens authorship cues.** Character
   substitutions and spelling noise can make AI text look human and human text
   look transformed.
3. **The model relies on dataset-specific regularities.** Very high confidence
   on wrong predictions indicates that it has learned strong but imperfect
   correlations rather than a general test of authorship.
4. **Near-parallel text families complicate random splitting.** Exact
   de-duplication does not group paraphrases, prompt siblings, or corrupted
   variants before splitting.

These are descriptive hypotheses from one held-out dataset. Testing them
causally would require a new experiment with prompt-family grouping,
near-duplicate detection, and evaluation on unseen generators and genres.

## Reproducible outputs

- `results/error_analysis_metrics.csv`: overall metrics, confusion counts, and
  confidence summaries.
- `results/error_analysis_slices.csv`: length, truncation, label, opening-style,
  and topic slices.
- `results/error_analysis_topics.csv`: topic support, terms, and error rates.
- `results/error_analysis_topic_terms.csv`: ranked NMF terms and weights.
- `results/error_examples.csv.gz`: all 238 false-positive and false-negative
  rows with compact excerpts.
- `results/error_analysis_summary.json`: input hash, model configuration,
  safeguards, and output manifest.

Run the complete analysis after executing the Week 2 notebook:

```powershell
python -m src.analyze_errors --overwrite
```

The explicit flag is required only when replacing an existing complete result
set.
