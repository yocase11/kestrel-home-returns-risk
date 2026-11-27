import unittest

import pandas as pd

from src.data_processing import FEATURE_COLUMNS
from src.modeling import (
    _feature_frame,
    build_chronological_calibrator,
    build_logistic_pipeline,
    build_random_forest_pipeline,
    chronological_split,
    evaluate_predictions,
    evaluate_subgroups,
    observation_window_experiment,
)


def _sample_frame() -> pd.DataFrame:
    rows = []
    dates = [
        "2026-03-28 09:00",
        "2026-03-29 09:00",
        "2026-03-30 09:00",
        "2026-04-01 09:00",
        "2026-04-02 09:00",
        "2026-04-03 09:00",
    ]
    for index, order_time in enumerate(dates):
        row = {feature: 1 for feature in FEATURE_COLUMNS}
        row.update(
            {
                "sales_channel": "app" if index % 2 == 0 else "web",
                "payment_mode": "cod",
                "delivery_pincode": "000000",
                "is_gift": "N",
                "sku": "sku-a",
                "product_family": "family-a",
                "product_model_name": "model-a",
                "order_id": f"order-{index}",
                "order_placed_at": order_time,
                "returned": index % 2,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class ModelingPreparationTests(unittest.TestCase):
    def test_feature_allowlist_excludes_sensitive_and_unverified_columns(self) -> None:
        forbidden = {
            "customer_prior_orders",
            "customer_prior_returns",
            "order_value_inr",
            "source",
            "pickup_scheduled_at",
            "last_service_event_type",
        }
        self.assertTrue(forbidden.isdisjoint(FEATURE_COLUMNS))

    def test_chronological_split_is_order_unique_and_sorted(self) -> None:
        frame = _sample_frame().iloc[[4, 0, 5, 2, 1, 3]]
        split = chronological_split(frame)
        self.assertEqual(len(split.train), 3)
        self.assertEqual(len(split.validation), 3)
        self.assertTrue(set(split.train.order_id).isdisjoint(split.validation.order_id))
        self.assertLess(
            pd.to_datetime(split.train.order_placed_at).max(),
            pd.to_datetime(split.validation.order_placed_at).min(),
        )
        self.assertEqual(
            pd.to_datetime(split.train.order_placed_at).tolist(),
            sorted(pd.to_datetime(split.train.order_placed_at).tolist()),
        )

    def test_window_experiment_marks_proxy_as_unverified(self) -> None:
        windows, evidence = observation_window_experiment(_sample_frame(), (1, 2))
        self.assertFalse(evidence["delivery_date_column_available"])
        self.assertFalse(evidence["extract_cutoff_available"])
        self.assertTrue(windows.verified_mature_rows.eq(0).all())
        self.assertTrue(windows.maturity_status.str.contains("unverifiable").all())

    def test_logistic_and_calibration_factories_are_unfitted(self) -> None:
        pipeline = build_logistic_pipeline()
        self.assertFalse(hasattr(pipeline.named_steps["classifier"], "coef_"))
        forest = build_random_forest_pipeline()
        self.assertFalse(hasattr(forest.named_steps["classifier"], "estimators_"))
        calibrated = build_chronological_calibrator(pipeline, n_splits=3)
        self.assertEqual(calibrated.cv.n_splits, 3)

    def test_metric_and_subgroup_helpers_return_expected_shapes(self) -> None:
        metrics = evaluate_predictions(
            [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], threshold=0.5
        )
        self.assertEqual(metrics["confusion_matrix_labels_0_1"], [[2, 0], [0, 2]])
        self.assertIn("pr_auc", metrics)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertIn("brier_score", metrics)
        subgroup_metrics = evaluate_subgroups(
            pd.DataFrame({"sales_channel": ["app", "app", "web", "web"]}),
            [0, 1, 0, 1],
            [0.1, 0.9, 0.2, 0.8],
            min_support=2,
            subgroup_columns=("sales_channel",),
        )
        self.assertEqual(len(subgroup_metrics["sales_channel"]), 2)

    def test_interactions_are_derived_only_from_allowed_order_time_features(self) -> None:
        frame = _sample_frame()
        frame["source"] = "leakage-not-used"
        features = _feature_frame(frame)
        self.assertEqual(
            features.loc[0, "channel_payment_mode"],
            f"{features.loc[0, 'sales_channel']}__{features.loc[0, 'payment_mode']}",
        )
        self.assertEqual(
            features.loc[0, "discount_qty_interaction"],
            features.loc[0, "discount_pct"] * features.loc[0, "qty"],
        )
        self.assertNotIn("source", features.columns)


if __name__ == "__main__":
    unittest.main()
