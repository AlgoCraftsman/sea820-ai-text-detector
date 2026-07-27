# Linear SVM Error Analysis

The Linear SVM is the project's best tested model (F1 `0.9994`). This analysis
uses its predictions on the same 92,846-row held-out test set used in
`notebooks/aiTextClassifier.ipynb`.

The implementation follows the Week 11 lab rather than a separate analysis
pipeline: align texts and predictions in a pandas DataFrame, isolate false
positives and false negatives with Boolean filters, compare word counts, print
examples, and inspect the Linear SVM's TF-IDF coefficients.

## Error counts

| Outcome | Count |
| --- | ---: |
| Correct | 92,806 |
| False positive (human predicted as AI) | 9 |
| False negative (AI predicted as human) | 31 |

False negatives account for 31 of the 40 mistakes. The model is therefore more
likely to miss AI-labelled text than to flag human-labelled text on this test
split.

## Text-length pattern

| Word count | Test texts | Errors | Error rate |
| --- | ---: | ---: | ---: |
| 0-100 | 248 | 1 | 0.4032% |
| 101-250 | 16,288 | 12 | 0.0737% |
| 251-500 | 57,156 | 27 | 0.0472% |
| 501+ | 19,154 | 0 | 0.0000% |

False negatives averaged 304.5 words, false positives averaged 322.0 words,
and correct predictions averaged 393.6 words. All 40 errors were at or below
500 words. The shortest group has the highest rate, but it contains only 248
texts and one error. The cautious conclusion is that mistakes in this run were
concentrated in shorter texts, which provide fewer TF-IDF word and bigram cues.

## False positives

A false positive is human-labelled text predicted as AI-generated.

| Processed row | Text type and short excerpt | Hypothesis |
| ---: | --- | --- |
| 68,543 | Personal travel essay: "The first place I would want to go is the Bahamas..." | Repeated first-person sentence patterns and a simple, orderly narrative may resemble patterns the SVM associated with AI writing. |
| 272,725 | Moral argument: "Have you ever made a awful mistake..." | The essay repeatedly states and restates one claim, creating a formulaic structure despite its grammatical errors. |
| 301,640 | Travel/career essay: "Have you ever wanted ... to another country..." | A repeated question-and-reason structure may outweigh mechanical substitutions such as `TM` that look human or corrupted. |

These examples show that spelling and grammar problems do not prevent a human
essay from being flagged. A TF-IDF Linear SVM uses weighted words and bigrams,
not evidence about who produced the text.

## False negatives

A false negative is AI-labelled text predicted as human-written.

| Processed row | Text type and short excerpt | Hypothesis |
| ---: | --- | --- |
| 225,486 | Argument about first impressions: "You may or may not believe that first impression can change..." | Character substitutions and awkward student-level phrasing make the AI-labelled text resemble noisy human writing. |
| 246,071 | Short education passage: "Before long, I saw a photo in which a child stood with a rifle." | At 116 words, the passage supplies relatively few cues; its education vocabulary also overlaps the student essays in the human class. |
| 398,539 | Informal problem-solving essay: "Wey, it's me, your average 8th grade student!" | An emoji, informal voice, misspellings, and merged words strongly imitate authentic student writing. |

The SVM's strongest human-indicative features include `people`, `student`,
`paragraph`, `schools`, and `students`. Several false negatives use this
student-writing vocabulary or style, which may pull them toward the human
class.

## Why the model failed

The examples and coefficient inspection support three descriptive hypotheses:

1. Formulaic human essays can share repeated, organized patterns with
   AI-labelled essays.
2. AI-labelled text containing misspellings, character substitutions, emojis,
   and informal student voice can resemble the human class.
3. Shorter texts provide fewer TF-IDF cues, making vocabulary-based decisions
   less stable.

These observations are associations in one held-out split, not evidence that
length or topic causes an error. The model's near-perfect score likely depends
partly on dataset-specific lexical and formatting patterns, so it should not be
treated as a universal AI-text detector.

## Reproduction

Run `notebooks/aiTextClassifier.ipynb` from top to bottom. Sections 8 and 9
contain the lab-aligned SVM feature inspection and error analysis.
