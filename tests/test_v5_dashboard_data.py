import tempfile
import unittest
from pathlib import Path
import shutil

from src.v5_dashboard_data import load_v5_dashboard_artifacts


ROOT = Path(__file__).resolve().parents[1]
V5_ARTIFACTS = ROOT / "derived_data" / "exploratory_v5_v5_run_012"
V5_REPORT = ROOT / "reports" / "exploratory_v5_v5_run_012" / "experiment_report.md"


def _copy_safe_v5_outputs(root: Path) -> None:
    artifact_dir = root / "derived_data" / "exploratory_v5_v5_run_012"
    report_dir = root / "reports" / "exploratory_v5_v5_run_012"
    artifact_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    for name in ("experiment_metadata.json", "metrics.json", "subgroup_metrics.json"):
        shutil.copy2(V5_ARTIFACTS / name, artifact_dir / name)
    shutil.copy2(V5_REPORT, report_dir / "experiment_report.md")


class V5DashboardDataTests(unittest.TestCase):
    def test_aggregate_dashboard_loads_without_row_level_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _copy_safe_v5_outputs(root)
            artifacts = load_v5_dashboard_artifacts(root)

            self.assertIsNotNone(artifacts.metadata)
            self.assertIsNotNone(artifacts.metrics)
            self.assertIsNone(artifacts.validation_predictions)
            self.assertEqual(artifacts.issues, ())

    def test_saved_report_table_matches_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _copy_safe_v5_outputs(root)
            artifacts = load_v5_dashboard_artifacts(root)
            self.assertIsNotNone(artifacts.metrics)
            self.assertIsNotNone(artifacts.report_text)

            report_rows = {}
            for line in artifacts.report_text.splitlines():
                if line.startswith("| ") and not line.startswith("|---"):
                    fields = [field.strip() for field in line.strip("|").split("|")]
                    if len(fields) >= 5:
                        report_rows[fields[0]] = fields[1:5]

            for model_name in (
                "majority_class_baseline",
                "lightgbm",
                "catboost",
                "xgboost",
                "hybrid_stacking_default_threshold",
                "hybrid_stacking",
            ):
                expected = artifacts.metrics[model_name]
                actual = report_rows[model_name]
                self.assertEqual(
                    actual,
                    [
                        f"{expected['accuracy']:.4f}",
                        f"{expected['precision']:.4f}",
                        f"{expected['recall']:.4f}",
                        f"{expected['f1']:.4f}",
                    ],
                )

    def test_missing_v5_outputs_are_reported_as_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = load_v5_dashboard_artifacts(Path(temporary_directory))

        self.assertIsNone(artifacts.metadata)
        self.assertIsNone(artifacts.metrics)
        self.assertIsNone(artifacts.validation_predictions)
        self.assertTrue(artifacts.issues)
        self.assertEqual(artifacts.model_files, ())


if __name__ == "__main__":
    unittest.main()
