import unittest
from pathlib import Path

import pandas as pd

from src.data_processing import (
    FEATURE_COLUMNS,
    _prepare_product_reference,
    build_return_request_label_audit,
    canonicalize_train,
)


class DataProcessingTests(unittest.TestCase):
    def test_duplicate_conflict_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "order_id": "O1",
                    "order_placed_at": "2026-01-01",
                    "customer_id": "C1",
                    "sku": "S1",
                    "sales_channel": "app",
                    "payment_mode": "cod",
                    "discount_pct": 0,
                    "qty": 1,
                    "order_value_inr": 100.0,
                    "promised_delivery_days": 5,
                    "delivery_pincode": 411001,
                    "is_gift": "N",
                    "customer_prior_orders": 0,
                    "customer_prior_returns": 0,
                    "delivery_note": None,
                    "last_service_event_type": "NONE",
                    "pickup_scheduled_at": None,
                    "source": "crm",
                    "returned": 0,
                },
                {
                    "order_id": "O1",
                    "order_placed_at": "2026-01-01",
                    "customer_id": "C1",
                    "sku": "S1",
                    "sales_channel": "app",
                    "payment_mode": "cod",
                    "discount_pct": 0,
                    "qty": 2,
                    "order_value_inr": 100.0,
                    "promised_delivery_days": 5,
                    "delivery_pincode": 411001,
                    "is_gift": "N",
                    "customer_prior_orders": 0,
                    "customer_prior_returns": 0,
                    "delivery_note": None,
                    "last_service_event_type": "NONE",
                    "pickup_scheduled_at": None,
                    "source": "partner_feed",
                    "returned": 0,
                },
            ]
        )
        with self.assertRaises(ValueError):
            canonicalize_train(frame)

    def test_primary_allowlist_is_conservative(self) -> None:
        forbidden = {
            "customer_prior_orders",
            "customer_prior_returns",
            "order_value_inr",
            "source",
            "pickup_scheduled_at",
            "last_service_event_type",
            "customer_id",
        }
        self.assertTrue(forbidden.isdisjoint(FEATURE_COLUMNS))

    def test_original_datasets_directory_exists(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "datasets").is_dir())

    def test_serving_product_reference_contains_only_safe_lookup_fields(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reference = _prepare_product_reference(root / "datasets")
        self.assertEqual(
            list(reference.columns),
            ["sku", "family", "model_name", "launch_date"],
        )
        self.assertFalse(reference["sku"].duplicated().any())

    def test_return_request_audit_only_labels_supported_positives(self) -> None:
        orders = pd.DataFrame(
            [
                {
                    "order_id": "source-return",
                    "returned": 1,
                    "pickup_scheduled_at": None,
                    "last_service_event_type": "NONE",
                },
                {
                    "order_id": "pickup-request",
                    "returned": 0,
                    "pickup_scheduled_at": "2026-01-03 12:00:00",
                    "last_service_event_type": "NONE",
                },
                {
                    "order_id": "reverse-request",
                    "returned": 0,
                    "pickup_scheduled_at": None,
                    "last_service_event_type": "REVERSE_PICKUP",
                },
                {
                    "order_id": "unknown-outcome",
                    "returned": 0,
                    "pickup_scheduled_at": None,
                    "last_service_event_type": "NONE",
                },
            ]
        )

        audit = build_return_request_label_audit(orders).set_index("order_id")

        self.assertEqual(
            int(audit.loc["source-return", "return_request_label_proxy"]), 1
        )
        self.assertEqual(
            int(audit.loc["pickup-request", "return_request_label_proxy"]), 1
        )
        self.assertEqual(
            int(audit.loc["reverse-request", "return_request_label_proxy"]), 1
        )
        self.assertTrue(
            pd.isna(audit.loc["unknown-outcome", "return_request_label_proxy"])
        )
        self.assertEqual(
            audit.loc["unknown-outcome", "label_status"],
            "unknown_no_request_evidence",
        )
        self.assertFalse(audit["return_request_label_proxy"].eq(0).any())


if __name__ == "__main__":
    unittest.main()
