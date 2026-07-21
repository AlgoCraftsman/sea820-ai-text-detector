import unittest

import numpy as np
import pandas as pd

from src.evaluate_frozen_comparison import (
    FROZEN_CONFIGURATION,
    FrozenConfiguration,
    build_manifest_lookup,
    clean_for_classic,
    compute_binary_metrics,
    validate_frozen_configuration,
    validate_prepared_membership,
)


class FrozenComparisonHelperTests(unittest.TestCase):
    def test_classic_cleaning_matches_week_one_contract(self) -> None:
        text = "  Visit HTTPS://Example.com/a\nMixed CASE  text.  "
        self.assertEqual(clean_for_classic(text), "visit mixed case text.")

    def test_manifest_membership_is_unique_and_addressable(self) -> None:
        manifest = pd.DataFrame(
            {
                "split": ["train", "validation", "test"],
                "source_row_id": [4, 1, 7],
                "labels": [0, 1, 1],
            }
        )
        lookup = build_manifest_lookup(manifest)

        self.assertEqual(lookup.counts, {"train": 1, "validation": 1, "test": 1})
        validate_prepared_membership(
            np.asarray([7]), np.asarray([1]), lookup, "test"
        )
        with self.assertRaisesRegex(ValueError, "membership differs"):
            validate_prepared_membership(
                np.asarray([4]), np.asarray([0]), lookup, "test"
            )

    def test_duplicate_manifest_membership_is_rejected(self) -> None:
        manifest = pd.DataFrame(
            {
                "split": ["train", "test"],
                "source_row_id": [2, 2],
                "labels": [0, 0],
            }
        )
        with self.assertRaisesRegex(ValueError, "more than one"):
            build_manifest_lookup(manifest)

    def test_metrics_treat_label_one_as_positive(self) -> None:
        metrics = compute_binary_metrics([0, 1, 0, 1], [0, 1, 1, 0])
        self.assertEqual(
            metrics,
            {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
        )

    def test_metrics_use_zero_division_zero(self) -> None:
        metrics = compute_binary_metrics([0, 1], [0, 0])
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_only_frozen_configuration_is_accepted(self) -> None:
        validate_frozen_configuration(FROZEN_CONFIGURATION)
        changed = FrozenConfiguration(max_features=40_000)
        with self.assertRaisesRegex(ValueError, "do not tune"):
            validate_frozen_configuration(changed)


if __name__ == "__main__":
    unittest.main()
