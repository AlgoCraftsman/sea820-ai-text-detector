# Baseline Metrics Validation

Validated on: 2026-07-12

## Source Artifacts

- Notebook: `notebooks/aiTextClassifier.ipynb`
- Saved metrics table: `results/baseline_model_comparison.csv`
- Saved chart: `results/figures/baseline_model_comparison.png`

## Validation Checks

- Confirmed the saved CSV contains all three baseline models:
  - Linear SVM
  - Logistic Regression
  - Multinomial NB
- Confirmed the saved CSV metrics match the executed notebook output to four decimal places.
- Confirmed the evaluated split reported by the notebook:
  - 487,231 non-empty rows before cleaned-text deduplication
  - 464,227 rows after cleaned-text deduplication
  - 371,381 train rows
  - 92,846 test rows

## Validated Metrics

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Linear SVM | 0.9996 | 0.9997 | 0.9991 | 0.9994 |
| Logistic Regression | 0.9947 | 0.9974 | 0.9889 | 0.9931 |
| Multinomial NB | 0.9776 | 0.9818 | 0.9599 | 0.9708 |

## Conclusion

The baseline metrics are validated and saved. Linear SVM is the strongest classic baseline by F1, while Logistic Regression remains the primary required TF-IDF baseline model.
