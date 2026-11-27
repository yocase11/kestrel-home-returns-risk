import unittest

import numpy as np
import pandas as pd

from src.train_pre_dispatch_v2 import (
    PreDispatchFeatureEngineer,
    _confidence_95_math,
    _high_precision_slice,
    _safe_note_text,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_placed_at": [
                "2025-04-01 09:00",
                "2025-04-02 09:00",
                "2025-04-03 09:00",
            ],
            "sku": ["sku-a", "sku-a", "sku-b"],
            "sales_channel": ["app", "web", "app"],
            "payment_mode": ["cod", "prepaid_upi", "cod"],
            "discount_pct": [10, 30, 20],
            "qty": [1, 1, 2],
            "promised_delivery_days": [3, 5, 4],
            "delivery_pincode": ["000000", "560734", "560735"],
            "is_gift": ["N", "Y", "N"],
            "customer_prior_orders": [0, 1, 2],
            "customer_prior_returns": [0, 0, 1],
            "delivery_note": [
                "Gate code 1234 call before delivery",
                "URGENT office",
                None,
            ],
            "product_family": ["family-a", "family-a", "family-b"],
            "product_model_name": ["model-a", "model-a", "model-b"],
            "launch_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
        }
    )


class PreDispatchFeatureTests(unittest.TestCase):
    def test_interactions_and_temporal_encodings(self) -> None:
        X = _rows()
        y = np.asarray([1, 0, 1])
        features = PreDispatchFeatureEngineer().fit_transform(X, y)

        self.assertEqual(features.loc[0, "payment_channel"], "cod_app")
        self.assertEqual(features.loc[1, "payment_zero_pincode"], "prepaid_upi_0")
        self.assertEqual(features.loc[1, "family_high_discount"], "family-a_1")
        self.assertEqual(features.loc[0, "first_time_payment"], "1_cod")
        self.assertAlmostEqual(features.loc[0, "sku_target_rate"], 1.14 / 10)
        self.assertAlmostEqual(features.loc[1, "sku_target_rate"], 2.14 / 11)
        self.assertAlmostEqual(features.loc[2, "sku_target_rate"], 1.14 / 10)
        self.assertTrue(features.loc[0, "note_text"] == "call")
        self.assertNotIn("1234", features.loc[0, "note_text"])

    def test_validation_encodings_use_training_fold_only(self) -> None:
        X = _rows()
        engineer = PreDispatchFeatureEngineer().fit(X.iloc[:2], np.asarray([1, 0]))
        validation = engineer.transform(X.iloc[[2]])
        self.assertAlmostEqual(validation.iloc[0]["sku_target_rate"], 1.14 / 10)
        self.assertEqual(validation.iloc[0]["sku_frequency"], 0)

    def test_notes_are_keyword_whitelisted_and_credentials_are_not_retained(self) -> None:
        self.assertEqual(
            _safe_note_text("Gate code 4725; call on 9876543210"),
            "call",
        )
        self.assertEqual(_safe_note_text(None), "missing_note")
        self.assertEqual(_safe_note_text("Beside side gate"), "unmodeled_note")

    def test_accuracy_budget_and_minimum_support_for_precision_claim(self) -> None:
        budget = _confidence_95_math(2126, 245)
        self.assertEqual(budget["errors_at_95_percent_accuracy_max"], 106)
        self.assertEqual(
            budget["minimum_true_positives_if_zero_false_positives"], 139
        )
        y = np.asarray([1, 0, 0, 0, 1, 0])
        scores = np.asarray([0.99, 0.98, 0.97, 0.96, 0.95, 0.94])
        slice_metrics = _high_precision_slice(y, scores, 0.90, min_support=5)
        self.assertFalse(slice_metrics["available"])


if __name__ == "__main__":
    unittest.main()
