import unittest

import numpy as np

from src.train_transformer import (
    classify_run,
    compute_binary_metrics,
    estimate_full_run_hours,
    validate_subset_size,
)


class TransformerTrainingHelperTests(unittest.TestCase):
    def test_metrics_treat_label_one_as_positive(self) -> None:
        logits = np.asarray(
            [
                [4.0, 1.0],
                [1.0, 4.0],
                [1.0, 4.0],
                [4.0, 1.0],
            ]
        )
        labels = np.asarray([0, 1, 0, 1])

        metrics = compute_binary_metrics((logits, labels))

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_metrics_use_zero_division_zero(self) -> None:
        logits = np.asarray([[3.0, 1.0], [3.0, 1.0]])
        labels = np.asarray([0, 1])

        metrics = compute_binary_metrics((logits, labels))

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_reduced_run_is_never_labeled_full(self) -> None:
        self.assertEqual(classify_run(128, 64, -1), "development_smoke")
        self.assertEqual(classify_run(None, None, 10), "bounded_full_data")
        self.assertEqual(classify_run(None, None, -1), "full")

    def test_subset_size_validation(self) -> None:
        validate_subset_size(None, 10, "train")
        validate_subset_size(10, 10, "train")
        with self.assertRaisesRegex(ValueError, "at least 2"):
            validate_subset_size(1, 10, "train")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_subset_size(11, 10, "train")

    def test_full_run_estimate_scales_train_and_validation(self) -> None:
        hours = estimate_full_run_hours(
            full_train_rows=3_600,
            full_validation_rows=360,
            epochs=2,
            train_samples_per_second=2.0,
            validation_rows_used=36,
            validation_runtime_seconds=10.0,
        )
        self.assertEqual(hours, 1.06)


if __name__ == "__main__":
    unittest.main()
