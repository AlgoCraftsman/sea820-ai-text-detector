# Week 2 Transformer Notebook Results

## Run status

`notebooks/transformer_finetuning.ipynb` completed from top to bottom on
2026-07-25 using the project virtual environment and an NVIDIA GeForce RTX
3060 Laptop GPU. All nine code cells have consecutive execution counts, all
required outputs were created, and the final notebook contains no traceback or
interrupted cell.

The source CSV contained 487,235 rows. Cleaning removed four empty rows and
23,004 normalized duplicates, leaving 464,227 unique usable rows.

| Split | Rows | Human | Machine-generated |
| --- | ---: | ---: | ---: |
| Train | 371,381 | 227,572 | 143,809 |
| Validation | 46,423 | 28,446 | 17,977 |
| Test | 46,423 | 28,447 | 17,976 |

The complete successful execution took approximately 1 hour 55 minutes.
Controlled tuning took 9 minutes 14 seconds, the final full training and
validation cell took 1 hour 31 minutes, test inference took 2 minutes 26
seconds, and the classic baseline cell took 5 minutes 27 seconds.

## Project brief acceptance

Every Week 2 requirement in the project brief and task checklist is satisfied:

- Hugging Face Datasets loads the source CSV.
- The uncased DistilBERT tokenizer formats the text with truncation and dynamic
  padding.
- `AutoModelForSequenceClassification` and the Hugging Face `Trainer` API
  fine-tune DistilBERT on the RTX 3060.
- Controlled experiments vary learning rate, batch size, and epoch count, with
  all settings and validation results saved.
- The best configuration is selected by validation F1 before test evaluation.
- The selected Transformer is evaluated on the held-out test split.
- The Logistic Regression and Linear SVM classic baselines are retrained on the
  same training membership and scored on the same test membership.
- Accuracy, precision, recall, and F1 are reported for all three models, including
  the exact performance difference.
- The fine-tuned model checkpoint and initial comparison results are saved.

The observed test F1 of `0.993361` is acceptable under the rubric's requirement
that a standard fine-tuned Transformer achieve reasonable performance.

## Controlled experiments

Each run used the same stratified subset of 8,000 training rows and 2,000
validation rows. The test split was not passed to the tuning code.

| Run | Learning rate | Batch size | Epochs | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B | 5e-5 | 4 | 1 | 0.9860 | 0.978205 | 0.985788 | **0.981982** |
| D | 2e-5 | 4 | 2 | 0.9850 | 0.969697 | 0.992248 | 0.980843 |
| C | 2e-5 | 8 | 1 | 0.9825 | 0.967130 | 0.988372 | 0.977636 |
| A | 2e-5 | 4 | 1 | 0.9790 | 0.958647 | 0.988372 | 0.973282 |

Run B had the highest validation F1 and was selected before test evaluation:
learning rate `5e-5`, per-device batch size `4`, and one epoch. The final
full-data validation metrics were accuracy `0.994809`, precision `0.996084`,
recall `0.990488`, and F1 `0.993278`.

## Same-membership test comparison

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.994399 | 0.997361 | 0.988151 | 0.992735 |
| TF-IDF + Linear SVM | 0.999505 | 0.999777 | 0.998943 | 0.999360 |
| Fine-tuned DistilBERT | 0.994873 | 0.996251 | 0.990487 | 0.993361 |

On this split, the Linear SVM is the strongest model with F1 `0.999360`.
DistilBERT reached F1 `0.993361`, which does not beat the Linear SVM and only
slightly exceeds Logistic Regression at `0.992735`. The fine-tuned Transformer
offers no advantage over the best classic baseline here. These near-ceiling
results describe this dataset and are not evidence of universal detector
reliability.

Independent validation rebuilt the seed-42 split and baseline from the source
CSV. It verified all 46,423 test source IDs and labels, reproduced the Logistic
Regression and Transformer saved metrics to floating-point precision, and confirmed
that the baseline and Transformer used the same test membership.

The notebook satisfies every Week 2 task in the project brief. Its result files
are:

- `results/split_summary.csv`
- `results/hyperparameter_experiments.csv`
- `results/transformer_test_metrics.csv`
- `results/transformer_test_predictions.csv.gz`
- `results/model_comparison.csv`

## Operational notes

The first execution attempt stopped before data preparation because the
default Hugging Face datasets cache was not writable in the managed
environment. The cache was redirected to ignored `runs/` storage. A second
pre-training attempt exposed a host-memory failure caused by retaining unused
full-text columns and superseded tables. The notebook now keeps only source
ID, text, and label in each split and releases superseded tables before Arrow
conversion. These changes did not alter membership, seed, model, sequence
length, batch size, or hyperparameters.

Model checkpoints and temporary caches remain under ignored `checkpoints/`
and `runs/` directories.
