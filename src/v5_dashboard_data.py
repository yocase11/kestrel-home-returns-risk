"""Read-only loading and consistency checks for the saved V5 experiment."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ARTIFACT_RELATIVE_PATH = Path("derived_data/exploratory_v5_v5_run_012")
REPORT_RELATIVE_PATH = Path("reports/exploratory_v5_v5_run_012/experiment_report.md")

PREDICTION_COLUMNS = (
    "order_id",
    "order_placed_at",
    "actual_returned",
    "lightgbm_probability",
    "catboost_probability",
    "xgboost_probability",
    "hybrid_probability",
    "predicted_returned",
)

PROBABILITY_COLUMNS = {
    "lightgbm": "lightgbm_probability",
    "catboost": "catboost_probability",
    "xgboost": "xgboost_probability",
    "hybrid_stacking_default_threshold": "hybrid_probability",
    "hybrid_stacking": "hybrid_probability",
}


@dataclass(frozen=True)
class V5DashboardArtifacts:
    metadata: dict[str, Any] | None
    metrics: dict[str, Any] | None
    subgroup_metrics: dict[str, Any] | None
    validation_predictions: pd.DataFrame | None
    report_text: str | None
    output_files: tuple[str, ...]
    model_files: tuple[str, ...]
    issues: tuple[str, ...]


def _read_json(path: Path, issues: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        issues.append(f"{path.name} is unavailable.")
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(f"Could not read {path.name}: {exc}")
        return None

    if not isinstance(value, dict):
        issues.append(f"{path.name} must contain a JSON object.")
        return None
    return value


def _read_predictions(path: Path, issues: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        issues.append(f"{path.name} is unavailable.")
        return None

    try:
        predictions = pd.read_csv(path, dtype={"order_id": "string"})
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        issues.append(f"Could not read {path.name}: {exc}")
        return None

    missing_columns = set(PREDICTION_COLUMNS).difference(predictions.columns)
    if missing_columns:
        issues.append(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}."
        )
        return None

    numeric_columns = (
        "actual_returned",
        "lightgbm_probability",
        "catboost_probability",
        "xgboost_probability",
        "hybrid_probability",
        "predicted_returned",
    )
    for column in numeric_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

    if predictions[list(numeric_columns)].isna().any().any():
        issues.append(f"{path.name} contains missing or non-numeric predictions.")
        return None

    if not predictions["actual_returned"].isin((0, 1)).all():
        issues.append(f"{path.name} actual_returned values must be binary.")
        return None
    if not predictions["predicted_returned"].isin((0, 1)).all():
        issues.append(f"{path.name} predicted_returned values must be binary.")
        return None

    probability_columns = (
        "lightgbm_probability",
        "catboost_probability",
        "xgboost_probability",
        "hybrid_probability",
    )
    for column in probability_columns:
        values = predictions[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            issues.append(
                f"{path.name} {column} values must be finite and between 0 and 1."
            )
            return None

    if predictions["order_id"].isna().any():
        issues.append(f"{path.name} contains missing order IDs.")
    elif predictions["order_id"].duplicated().any():
        issues.append(f"{path.name} contains duplicate order IDs.")

    return predictions


def _matches(stored: Any, computed: float) -> bool:
    try:
        return math.isclose(float(stored), computed, rel_tol=0, abs_tol=1e-10)
    except (TypeError, ValueError):
        return False


def _check_saved_metrics(
    metrics: dict[str, Any] | None,
    predictions: pd.DataFrame | None,
    issues: list[str],
) -> None:
    if metrics is None or predictions is None:
        return

    actual = predictions["actual_returned"].astype(int)
    confusion_labels = [0, 1]

    for model_name in (
        "majority_class_baseline",
        "lightgbm",
        "catboost",
        "xgboost",
        "hybrid_stacking_default_threshold",
        "hybrid_stacking",
    ):
        saved = metrics.get(model_name)
        if not isinstance(saved, dict):
            continue

        if model_name == "majority_class_baseline":
            predicted = pd.Series(0, index=predictions.index, dtype=int)
            probability = None
        else:
            probability_column = PROBABILITY_COLUMNS[model_name]
            probability = predictions[probability_column]
            try:
                threshold = float(saved["threshold"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{model_name} has no usable saved threshold.")
                continue
            predicted = (probability >= threshold).astype(int)

        computed = {
            "support": len(actual),
            "positive_count": int(actual.sum()),
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(
                precision_score(actual, predicted, zero_division=0)
            ),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "confusion_matrix_labels_0_1": confusion_matrix(
                actual,
                predicted,
                labels=confusion_labels,
            ).tolist(),
        }
        if probability is not None and actual.nunique() == 2:
            computed["roc_auc"] = float(roc_auc_score(actual, probability))
            computed["pr_auc"] = float(
                average_precision_score(actual, probability)
            )

        mismatches = []
        for field in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ):
            if field in computed and field in saved:
                if not _matches(saved[field], computed[field]):
                    mismatches.append(field)
        for field in ("support", "positive_count"):
            if field in saved and saved[field] != computed[field]:
                mismatches.append(field)
        if (
            saved.get("confusion_matrix_labels_0_1") is not None
            and saved["confusion_matrix_labels_0_1"]
            != computed["confusion_matrix_labels_0_1"]
        ):
            mismatches.append("confusion matrix")
        if mismatches:
            issues.append(
                f"{model_name} metrics in metrics.json disagree with "
                f"validation_predictions.csv: {', '.join(mismatches)}."
            )

    selected = metrics.get("hybrid_stacking")
    if isinstance(selected, dict) and "threshold" in selected:
        expected = (
            predictions["hybrid_probability"]
            >= float(selected["threshold"])
        ).astype(int)
        if not np.array_equal(
            expected.to_numpy(),
            predictions["predicted_returned"].astype(int).to_numpy(),
        ):
            issues.append(
                "predicted_returned in validation_predictions.csv does not "
                "match the saved hybrid threshold."
            )


def load_v5_dashboard_artifacts(root: Path) -> V5DashboardArtifacts:
    """Load saved V5 results without fitting or modifying any model."""
    root = Path(root)
    artifact_dir = root / ARTIFACT_RELATIVE_PATH
    issues: list[str] = []

    metadata = _read_json(artifact_dir / "experiment_metadata.json", issues)
    metrics = _read_json(artifact_dir / "metrics.json", issues)
    subgroup_metrics = _read_json(
        artifact_dir / "subgroup_metrics.json",
        issues,
    )
    predictions_path = artifact_dir / "validation_predictions.csv"
    predictions = (
        _read_predictions(predictions_path, issues)
        if predictions_path.is_file()
        else None
    )

    if (
        metadata is not None
        and predictions is not None
        and "validation_rows" in metadata
        and int(metadata["validation_rows"]) != len(predictions)
    ):
        issues.append(
            "validation_predictions.csv row count does not match experiment_metadata.json."
        )

    _check_saved_metrics(metrics, predictions, issues)

    report_path = root / REPORT_RELATIVE_PATH
    if report_path.is_file():
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            issues.append(f"Could not read {report_path.name}: {exc}")
            report_text = None
    else:
        issues.append(f"{report_path.name} is unavailable.")
        report_text = None

    try:
        output_files = tuple(
            sorted(path.name for path in artifact_dir.iterdir() if path.is_file())
        )
    except OSError as exc:
        issues.append(f"Could not list V5 output files: {exc}")
        output_files = ()

    model_dir = artifact_dir / "models"
    try:
        model_files = tuple(
            sorted(path.name for path in model_dir.glob("*.joblib") if path.is_file())
        )
    except OSError as exc:
        issues.append(f"Could not list V5 model artifacts: {exc}")
        model_files = ()
    return V5DashboardArtifacts(
        metadata=metadata,
        metrics=metrics,
        subgroup_metrics=subgroup_metrics,
        validation_predictions=predictions,
        report_text=report_text,
        output_files=output_files,
        model_files=model_files,
        issues=tuple(issues),
    )
