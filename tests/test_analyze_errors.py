import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.analyze_errors import (
    _plot_confusion,
    _plot_length_slices,
    _plot_topic_slices,
    assign_nmf_topics,
    assign_word_length_bin,
    classify_opening_style,
    classify_outcomes,
    load_source_rows,
    summarize_slice,
)


class ErrorAnalysisHelperTests(unittest.TestCase):
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

    def test_opening_style_is_descriptive_and_fixed(self) -> None:
        self.assertEqual(classify_opening_style("Dear Principal, please consider..."), "salutation")
        self.assertEqual(
            classify_opening_style("Have you ever wondered? Here is why."),
            "opening_question",
        )
        self.assertEqual(classify_opening_style("This essay begins directly."), "other")

    def test_slice_summary_counts_error_types_and_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "true_label": [0, 0, 1, 1],
                "distilbert_prediction": [0, 1, 0, 1],
            }
        )
        summary = summarize_slice(
            frame,
            slice_name="example",
            slice_value="all",
        )
        self.assertEqual(summary["support"], 4)
        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["false_positives"], 1)
        self.assertEqual(summary["false_negatives"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["f1"], 0.5)

    def test_source_rows_align_by_original_row_number(self) -> None:
        predictions = pd.DataFrame(
            {
                "source_row_id": [3, 0],
                "true_label": [1, 0],
            }
        )
        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.csv"
            pd.DataFrame(
                {
                    "text": ["zero", "one", "two", "three"],
                    "generated": [0, 0, 1, 1],
                }
            ).to_csv(source_path, index=False)
            aligned = load_source_rows(predictions, source_path, chunk_size=2)
        self.assertEqual(aligned["source_row_id"].tolist(), [3, 0])
        self.assertEqual(aligned["text"].tolist(), ["three", "zero"])

    def test_topic_assignment_is_deterministic_and_ignores_labels(self) -> None:
        texts = [
            "school student teacher classroom homework",
            "student classroom lesson teacher school",
            "education teacher homework student",
            "classroom school learning student",
            "car road driving vehicle traffic",
            "vehicle traffic road driver car",
            "driving car highway traffic",
            "road vehicle driver highway",
        ]
        first = assign_nmf_topics(
            texts,
            n_topics=2,
            max_features=50,
            min_df=1,
            top_terms=3,
        )
        second = assign_nmf_topics(
            texts,
            n_topics=2,
            max_features=50,
            min_df=1,
            top_terms=3,
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        pd.testing.assert_frame_equal(first[2], second[2])
        self.assertEqual(len(set(first[0][:4])), 1)
        self.assertEqual(len(set(first[0][4:])), 1)
        self.assertNotEqual(first[0][0], first[0][4])

    def test_plots_are_written_as_deterministic_svg(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "true_negatives": 8,
                    "false_positives": 2,
                    "false_negatives": 1,
                    "true_positives": 9,
                }
            ]
        )
        length_slices = pd.DataFrame(
            [
                {
                    "slice": "word_length_bin",
                    "value": value,
                    "support": 20,
                    "errors": index + 1,
                    "error_rate": (index + 1) / 20,
                }
                for index, value in enumerate(
                    ["0-100", "101-250", "251-500", "501-750", "751+"]
                )
            ]
        )
        topic_slices = pd.DataFrame(
            [
                {
                    "value": f"topic_{index}",
                    "support": 20,
                    "errors": index,
                    "error_rate": index / 20,
                }
                for index in range(1, 4)
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            functions_and_frames = (
                (_plot_confusion, metrics, "confusion"),
                (_plot_length_slices, length_slices, "length"),
                (_plot_topic_slices, topic_slices, "topic"),
            )
            for function, frame, name in functions_and_frames:
                first = root / f"{name}.svg"
                second = root / f"{name}-repeated.svg"
                function(frame, first)
                function(frame, second)
                self.assertIn("<svg", first.read_text(encoding="utf-8"))
                self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
