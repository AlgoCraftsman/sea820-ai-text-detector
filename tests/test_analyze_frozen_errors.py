import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.analyze_frozen_errors import (
    _plot_confusion,
    _plot_truncation_slices,
    assign_word_length_bin,
    classify_opening_style,
    classify_outcomes,
    overlap_signature,
    summarize_slice,
)


class FrozenErrorAnalysisHelperTests(unittest.TestCase):
    def test_word_length_bins_have_fixed_boundaries(self) -> None:
        self.assertEqual(assign_word_length_bin(0), "0-100")
        self.assertEqual(assign_word_length_bin(100), "0-100")
        self.assertEqual(assign_word_length_bin(101), "101-250")
        self.assertEqual(assign_word_length_bin(500), "251-500")
        self.assertEqual(assign_word_length_bin(751), "751+")
        with self.assertRaisesRegex(ValueError, "negative"):
            assign_word_length_bin(-1)

    def test_outcomes_use_human_zero_and_ai_one(self) -> None:
        outcomes = classify_outcomes([0, 0, 1, 1], [0, 1, 0, 1])
        np.testing.assert_array_equal(outcomes, ["TN", "FP", "FN", "TP"])

    def test_overlap_signature_is_deterministic(self) -> None:
        self.assertEqual(
            overlap_signature({"logistic": True, "svm": False, "distilbert": True}),
            "logistic + distilbert",
        )
        self.assertEqual(overlap_signature({"logistic": False}), "none")

    def test_opening_style_is_descriptive_and_fixed(self) -> None:
        self.assertEqual(classify_opening_style("Dear Principal, please consider..."), "salutation")
        self.assertEqual(classify_opening_style("Have you ever wondered? Here is why."), "opening_question")
        self.assertEqual(classify_opening_style("This essay begins directly."), "other")

    def test_slice_summary_counts_error_types_and_metrics(self) -> None:
        frame = pd.DataFrame(
            {"true_label": [0, 0, 1, 1], "prediction": [0, 1, 0, 1]}
        )
        summary = summarize_slice(
            frame,
            model="model",
            prediction_column="prediction",
            slice_name="example",
            slice_value="all",
        )
        self.assertEqual(summary["support"], 4)
        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["false_positives"], 1)
        self.assertEqual(summary["false_negatives"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["f1"], 0.5)

    def test_lab_style_plots_are_written_as_svg(self) -> None:
        model_summary = pd.DataFrame(
            [
                {
                    "model": model,
                    "true_negatives": 8,
                    "false_positives": 2,
                    "false_negatives": 1,
                    "true_positives": 9,
                }
                for model in ("Logistic Regression", "Linear SVM", "DistilBERT")
            ]
        )
        slice_summary = pd.DataFrame(
            [
                {
                    "model": model,
                    "slice": "truncation",
                    "value": value,
                    "error_rate": rate,
                }
                for model in ("Logistic Regression", "Linear SVM", "DistilBERT")
                for value, rate in (
                    ("not_truncated", 0.01),
                    ("truncated", 0.02),
                )
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            confusion_path = root / "confusion.svg"
            repeated_confusion_path = root / "confusion-repeated.svg"
            slice_path = root / "truncation.svg"
            repeated_slice_path = root / "truncation-repeated.svg"
            _plot_confusion(model_summary, confusion_path)
            _plot_confusion(model_summary, repeated_confusion_path)
            _plot_truncation_slices(slice_summary, slice_path)
            _plot_truncation_slices(slice_summary, repeated_slice_path)

            self.assertIn("<svg", confusion_path.read_text(encoding="utf-8"))
            self.assertIn("<svg", slice_path.read_text(encoding="utf-8"))
            self.assertEqual(
                confusion_path.read_bytes(),
                repeated_confusion_path.read_bytes(),
            )
            self.assertEqual(slice_path.read_bytes(), repeated_slice_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
