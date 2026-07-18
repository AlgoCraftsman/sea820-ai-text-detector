import unittest

from src.transformer_data import (
    _prepare_batch,
    clean_for_transformer,
    normalize_for_dedup,
)


class TransformerCleaningTests(unittest.TestCase):
    def test_transformer_cleaning_preserves_linguistic_cues(self) -> None:
        text = "  Keep CASE!\n\nKeep punctuation, and stop words.  "
        self.assertEqual(
            clean_for_transformer(text),
            "Keep CASE! Keep punctuation, and stop words.",
        )

    def test_dedup_normalization_matches_week_one_rules(self) -> None:
        first = " Example HTTPS://example.com/page\nText "
        second = "example   text"
        self.assertEqual(normalize_for_dedup(first), normalize_for_dedup(second))

    def test_prepare_batch_keeps_source_ids_and_integer_labels(self) -> None:
        prepared = _prepare_batch(
            {"text": [" Human text ", "\nAI text"], "generated": [0.0, "1"]},
            [11, 12],
        )
        self.assertEqual(prepared["text"], ["Human text", "AI text"])
        self.assertEqual(prepared["labels"], [0, 1])
        self.assertEqual(prepared["source_row_id"], [11, 12])
        self.assertEqual(prepared["is_non_empty"], [True, True])

    def test_prepare_batch_rejects_invalid_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "not one of 0 or 1"):
            _prepare_batch({"text": ["text"], "generated": [2]}, [0])


if __name__ == "__main__":
    unittest.main()
