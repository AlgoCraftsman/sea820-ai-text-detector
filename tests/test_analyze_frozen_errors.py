import unittest

import numpy as np
import pandas as pd

from src.analyze_frozen_errors import (
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


if __name__ == "__main__":
    unittest.main()
