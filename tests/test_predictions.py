import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.generate_predictions import generate_predictions


class PredictionGenerationTests(unittest.TestCase):
    def test_generation_is_blocked_by_current_gate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as _:
            with self.assertRaises(PermissionError):
                generate_predictions(root)

    def test_predictions_match_test_order_and_range(self) -> None:
        root = Path(__file__).resolve().parents[1]
        predictions = pd.read_csv(
            root / "predictions.csv", dtype={"order_id": "string"}
        )
        test_path = next((root / "datasets").glob("*test_unlabelled.csv"))
        test = pd.read_csv(
            test_path,
            dtype={"order_id": "string"},
            usecols=["order_id"],
        )
        self.assertEqual(list(predictions.columns), ["order_id", "score"])
        self.assertEqual(len(predictions), 2096)
        self.assertEqual(
            predictions["order_id"].tolist(),
            test["order_id"].tolist(),
        )
        self.assertTrue(predictions["score"].between(0, 1).all())
        self.assertGreater(predictions["score"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
