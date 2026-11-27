import unittest

import numpy as np
import pandas as pd

from src.train_calibrated_ensemble import (
    FEATURE_NAMES,
    FORBIDDEN_COLUMNS,
    FoldFeatureEngineer,
    ProbabilityCalibrator,
    _prepare_model_inputs,
    evaluate_scores,
)


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["order-1", "order-2"],
            "order_placed_at": ["2025-09-01 09:00", "2025-11-02 19:00"],
            "sku": ["sku-a", "sku-a"],
            "sales_channel": ["app", "web"],
            "payment_mode": ["cod", "prepaid_upi"],
            "discount_pct": [10, 30],
            "qty": [1, 2],
            "promised_delivery_days": [3, 5],
            "delivery_pincode": ["000000", "560734"],
            "is_gift": ["N", "Y"],
            "customer_prior_orders": [0, 2],
            "customer_prior_returns": [0, 1],
            "delivery_note": [None, "URGENT - CALL office"],
            "product_family": ["family-a", "family-a"],
            "product_model_name": ["model-a", "model-a"],
            "launch_date": ["2025-01-01", "2025-01-01"],
        }
    )


class CalibratedEnsembleFeatureTests(unittest.TestCase):
    def test_features_use_training_fold_family_mean_and_drop_raw_note(self) -> None:
        train = _raw_rows()
        engineer = FoldFeatureEngineer().fit(train)
        validation = train.iloc[[1]].copy()
        validation.loc[:, "discount_pct"] = 40
        features = engineer.transform(validation)

        self.assertAlmostEqual(features.loc[0, "discount_diff_from_family"], 20.0)
        self.assertAlmostEqual(features.loc[0, "smoothed_prior_return_rate"], 2.14 / 12)
        self.assertEqual(features.loc[0, "is_first_time_buyer"], 0)
        self.assertEqual(features.loc[0, "pincode_prefix"], "56")
        self.assertEqual(features.loc[0, "note_has_urgent"], 1)
        self.assertEqual(features.loc[0, "note_has_call"], 1)
        self.assertEqual(features.loc[0, "note_has_office"], 1)
        self.assertNotIn("delivery_note", features.columns)
        self.assertEqual(list(features.columns), list(FEATURE_NAMES))

    def test_first_time_and_zero_pincode_features(self) -> None:
        rows = _raw_rows().iloc[[0]]
        features = FoldFeatureEngineer().fit(rows).transform(rows)

        self.assertEqual(features.loc[0, "is_zero_pincode"], 1)
        self.assertEqual(features.loc[0, "pincode_prefix"], "0")
        self.assertEqual(features.loc[0, "is_first_time_buyer"], 1)
        self.assertAlmostEqual(features.loc[0, "smoothed_prior_return_rate"], 0.114)
        self.assertEqual(features.loc[0, "has_note"], 0)

    def test_fold_mean_is_not_updated_by_transform_rows(self) -> None:
        train = _raw_rows()
        engineer = FoldFeatureEngineer().fit(train)
        original_mean = engineer.family_discount_means_["family-a"]
        later = train.iloc[[1]].copy()
        later.loc[:, "discount_pct"] = 100

        engineer.transform(later)

        self.assertEqual(engineer.family_discount_means_["family-a"], original_mean)
        self.assertAlmostEqual(original_mean, 20.0)

    def test_input_preparation_drops_explicitly_forbidden_columns(self) -> None:
        train = _raw_rows()
        for column in FORBIDDEN_COLUMNS:
            train[column] = "not-a-feature"
        train["source"] = "crm"
        products = pd.DataFrame(
            {
                "sku": ["sku-a"],
                "family": ["family-a"],
                "model_name": ["model-a"],
                "launch_date": ["2025-01-01"],
            }
        )

        prepared = _prepare_model_inputs(train, products, "test")

        self.assertTrue(set(FORBIDDEN_COLUMNS).isdisjoint(prepared.columns))
        self.assertTrue(
            {"order_id", "order_placed_at", "delivery_note"}.issubset(prepared.columns)
        )


class CalibrationAndEvaluationTests(unittest.TestCase):
    def test_sigmoid_and_isotonic_calibrators_return_probabilities(self) -> None:
        raw = np.asarray([0.05, 0.1, 0.15, 0.3, 0.7, 0.9])
        target = np.asarray([0, 0, 0, 0, 1, 1])

        for method in ("sigmoid", "isotonic"):
            calibrated = ProbabilityCalibrator(method).fit(raw, target).predict(raw)
            self.assertTrue(np.isfinite(calibrated).all())
            self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())

    def test_capacity_metrics_and_policy_net_savings(self) -> None:
        y = np.asarray([0, 0, 1, 1, 1, 0])
        scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.9, 0.8])

        metrics = evaluate_scores(y, scores)
        top20 = metrics["top_20_percent"]

        self.assertEqual(top20["flagged_count"], 2)
        self.assertEqual(top20["true_positive_count"], 1)
        self.assertEqual(top20["precision"], 0.5)
        self.assertAlmostEqual(top20["recall"], 1 / 3)
        self.assertAlmostEqual(top20["f1"], 0.4)
        self.assertAlmostEqual(
            top20["expected_net_savings_rs"],
            (0.9 + 0.8) * 0.35 * 1150 - 2 * 45,
        )


if __name__ == "__main__":
    unittest.main()
