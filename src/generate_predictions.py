"""Generate production-approved predictions or explicit exploratory submission scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.prediction_service import MODEL_RELATIVE_PATH
from src.train import read_training_gate


def _read_test_and_sample(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_path = root / "derived_data" / "test_modeling.csv"
    test = pd.read_csv(
        test_path,
        dtype={"order_id": "string", "delivery_pincode": "string"},
    )
    sample_matches = list((root / "datasets").glob("*sample_submission.csv"))
    if len(sample_matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one sample_submission.csv in datasets/."
        )
    sample = pd.read_csv(sample_matches[0], dtype={"order_id": "string"})
    if list(sample.columns) != ["order_id", "score"]:
        raise ValueError("sample_submission.csv must contain order_id,score columns.")
    if test["order_id"].tolist() != sample["order_id"].tolist():
        raise ValueError("Test order IDs do not match sample submission order and row.")
    if test["order_id"].isna().any() or test["order_id"].duplicated().any():
        raise ValueError("Test order IDs must be present and unique.")
    return test, sample


def _score_and_write(root: Path, model_path: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}.")
    test, _ = _read_test_and_sample(root)
    model = joblib.load(model_path)
    probabilities = model.predict_proba(test)[:, 1]
    if probabilities.shape[0] != len(test):
        raise AssertionError("Model output row count differs from test row count.")
    if not pd.Series(probabilities).between(0, 1).all():
        raise AssertionError("Model scores must be between 0 and 1.")

    output = pd.DataFrame(
        {"order_id": test["order_id"].astype("string"), "score": probabilities}
    )
    output_path = root / "predictions.csv"
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} exists; remove it explicitly before regenerating."
        )
    output.to_csv(output_path, index=False)
    return output_path


def generate_predictions(root: Path, model_name: str = "logistic_regression") -> Path:
    """Generate production predictions only after the approval gate is open."""
    status, reason = read_training_gate(root)
    if status != "APPROVED":
        raise PermissionError(
            f"Prediction generation is blocked: {status}. {reason}. "
            "No predictions.csv was generated."
        )

    model_path = root / "models" / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model at {model_path}. Train an approved model first."
        )

    return _score_and_write(root, model_path)


def generate_exploratory_submission(root: Path) -> Path:
    """Write an assignment-only score file without changing production approval."""
    model_path = root / MODEL_RELATIVE_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No saved exploratory model at {model_path}. No predictions.csv was generated."
        )
    return _score_and_write(root, model_path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate production predictions after approval, or an explicitly "
            "exploratory assignment submission."
        )
    )
    parser.add_argument(
        "--model",
        default="logistic_regression",
        help="Saved model name without the .joblib suffix.",
    )
    parser.add_argument(
        "--exploratory-submission",
        action="store_true",
        help=(
            "Use the saved exploratory operating-point model for assignment "
            "scores; production status remains blocked."
        ),
    )
    args = parser.parse_args()
    try:
        if args.exploratory_submission:
            path = generate_exploratory_submission(root)
        else:
            path = generate_predictions(root, args.model)
    except (PermissionError, FileNotFoundError, ValueError, AssertionError, FileExistsError) as exc:
        print(f"PREDICTION GENERATION FAILED: {exc}", flush=True)
        return 2
    mode = "EXPLORATORY SUBMISSION ONLY" if args.exploratory_submission else "APPROVED"
    print(f"Generated {mode} scores: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
