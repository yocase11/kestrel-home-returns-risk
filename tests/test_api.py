import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.api import app
from src.prediction_service import get_prediction_status


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def _valid_payload(cls) -> dict[str, object]:
        root = Path(__file__).resolve().parents[1]
        row = pd.read_csv(
            root / "derived_data" / "test_modeling.csv",
            dtype={"order_id": "string", "delivery_pincode": "string"},
            nrows=1,
        ).iloc[0]
        return {
            "order_placed_at": row["order_placed_at"],
            "sku": row["sku"],
            "sales_channel": row["sales_channel"],
            "payment_mode": row["payment_mode"],
            "discount_pct": int(row["discount_pct"]),
            "qty": int(row["qty"]),
            "promised_delivery_days": int(row["promised_delivery_days"]),
            "delivery_pincode": row["delivery_pincode"],
            "is_gift": row["is_gift"],
            "customer_prior_orders": 0,
            "customer_prior_returns": 0,
            "delivery_note": "",
            "shield_member_verified_at_dispatch": False,
        }

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_reports_exploratory_model_and_blocked_production(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model_status"], "exploratory")
        self.assertEqual(payload["production_status"], "blocked")

    def test_predict_returns_exploratory_score_and_readable_reasons(self) -> None:
        response = self.client.post("/predict", json=self._valid_payload())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "exploratory")
        self.assertEqual(payload["production_status"], "blocked")
        self.assertGreaterEqual(payload["risk_score"], 0)
        self.assertLessEqual(payload["risk_score"], 1)
        self.assertEqual(payload["score"], payload["risk_score"])
        self.assertIn(payload["risk_tier"], {"Low", "Medium", "High"})
        self.assertIn("dispatch", payload["recommended_action"].lower())
        self.assertTrue(payload["reasons"])
        self.assertIn("not causes", payload["reasoning_note"])

    def test_verified_shield_member_gets_soft_call_action(self) -> None:
        payload = self._valid_payload()
        payload["shield_member_verified_at_dispatch"] = True
        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("soft call only", response.json()["recommended_action"])

    def test_invalid_prior_return_count_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["customer_prior_orders"] = 1
        payload["customer_prior_returns"] = 2
        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)

    def test_invalid_pincode_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["delivery_pincode"] = "411"
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_missing_model_status_is_truthful(self) -> None:
        status = get_prediction_status(Path("empty-project-root"))
        self.assertEqual(status["model_status"], "blocked")
        self.assertIn("No saved exploratory model", status["reason"])

    def test_timezone_aware_order_time_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["order_placed_at"] = "2026-05-10T14:30:00+05:30"
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_unknown_product_sku_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["sku"] = "SKU-NOT-IN-CATALOG"
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("not present in the product reference", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
