import unittest
from pathlib import Path
import tempfile

from src.train import (
    fit_and_evaluate_if_approved,
    fit_exploratory_models,
    read_training_gate,
)


class TrainingGateTests(unittest.TestCase):
    def test_current_repository_gate_is_not_approved(self) -> None:
        root = Path(__file__).resolve().parents[1]
        status, reason = read_training_gate(root)
        self.assertEqual(status, "EXPLORATORY ONLY")
        self.assertTrue(reason)

    def test_fit_never_runs_under_current_gate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(PermissionError):
            fit_and_evaluate_if_approved(root)

    def test_exploratory_fit_rejects_blocked_gate_before_accessing_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gate = root / "reports" / "final_training_gate.md"
            gate.parent.mkdir()
            gate.write_text("**Training status: BLOCKED**\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                fit_exploratory_models(root)


if __name__ == "__main__":
    unittest.main()
