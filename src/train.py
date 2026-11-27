"""Training entry point with separate production and exploratory gates."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.modeling import (
    ChronologicalSplit,
    build_logistic_pipeline,
    build_random_forest_pipeline,
    build_encoded_tree_pipeline,
    build_chronological_calibrator,
    chronological_split,
    evaluate_predictions,
    evaluate_subgroups,
)


GATE_REPORT = "reports/final_training_gate.md"
TRAIN_DATA = "derived_data/train_modeling.csv"
VALIDATION_START = pd.Timestamp("2026-04-01")


def read_training_gate(root: Path) -> tuple[str, str]:
    gate = root / GATE_REPORT
    if not gate.exists():
        return "BLOCKED", "missing reports/final_training_gate.md"
    content = gate.read_text(encoding="utf-8")
    if "**Training status: APPROVED**" in content:
        return "APPROVED", "the final training gate is marked APPROVED"
    if "**Training status: EXPLORATORY ONLY**" in content:
        return "EXPLORATORY ONLY", "the gate allows no production claims"
    if "**Training status: BLOCKED**" in content:
        return "BLOCKED", "the label-maturity evidence gate is BLOCKED"
    return "BLOCKED", "the gate report lacks an exact recognized status"


def fit_and_evaluate_if_approved(root: Path) -> dict[str, Any]:
    """Fit production candidates only after an APPROVED gate."""
    status, reason = read_training_gate(root)
    if status != "APPROVED":
        raise PermissionError(
            f"Training is not allowed: {status}. {reason}. "
            "No estimator was fit and no predictions.csv was generated."
        )

    frame = pd.read_csv(
        root / TRAIN_DATA,
        dtype={"order_id": "string", "delivery_pincode": "string"},
    )
    split: ChronologicalSplit = chronological_split(frame, VALIDATION_START)
    X_train = split.train
    y_train = split.train["returned"].astype(int)
    X_validation = split.validation
    y_validation = split.validation["returned"].astype(int)

    candidates: dict[str, Any] = {
        "logistic_regression": build_logistic_pipeline(),
    }

    candidates["random_forest"] = build_random_forest_pipeline()

    try:
        candidates["catboost"] = importlib.import_module(
            "src.modeling"
        ).build_catboost_pipeline()
    except ImportError:
        pass

    try:
        xgb = importlib.import_module("xgboost")
        candidates["xgboost"] = build_encoded_tree_pipeline(
            xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            )
        )
    except ImportError:
        pass

    try:
        lgbm = importlib.import_module("lightgbm")
        candidates["lightgbm"] = build_encoded_tree_pipeline(
            lgbm.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=-1,
                num_leaves=31,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        )
    except ImportError:
        pass

    results: dict[str, Any] = {}
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_validation)[:, 1]
        metrics = evaluate_predictions(
            y_validation,
            probabilities,
            threshold=0.5,
        )
        metrics["exploratory_or_production"] = "production_ready_only_after_gate"
        results[name] = metrics
        joblib.dump(estimator, models_dir / f"{name}.joblib")

    comparison_path = reports_dir / "model_comparison.json"
    comparison_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    return results


def fit_exploratory_models(root: Path) -> dict[str, Any]:
    """Fit isolated experiments; outputs cannot be mistaken for production artifacts."""
    status, reason = read_training_gate(root)
    if status != "EXPLORATORY ONLY":
        raise PermissionError(
            f"Exploratory training requires an EXPLORATORY ONLY gate: "
            f"{status}. {reason}. No estimator was fit."
        )

    frame = pd.read_csv(
        root / TRAIN_DATA,
        dtype={"order_id": "string", "delivery_pincode": "string"},
    )
    split: ChronologicalSplit = chronological_split(frame, VALIDATION_START)
    X_train = split.train
    y_train = split.train["returned"].astype(int)
    X_validation = split.validation
    y_validation = split.validation["returned"].astype(int)
    if split.train["order_id"].isin(split.validation["order_id"]).any():
        raise AssertionError("An order_id crossed the exploratory split.")
    if pd.to_datetime(split.train["order_placed_at"]).max() >= pd.to_datetime(
        split.validation["order_placed_at"]
    ).min():
        raise AssertionError("Exploratory split is not strictly chronological.")

    candidates: dict[str, Any] = {
        "logistic_regression": build_logistic_pipeline(),
        "random_forest": build_random_forest_pipeline(),
        "logistic_regression_sigmoid_calibrated": build_chronological_calibrator(
            build_logistic_pipeline(), n_splits=3
        ),
    }

    output_dir = root / "derived_data" / "exploratory"
    model_dir = output_dir / "models"
    report_dir = root / "reports" / "exploratory"
    if output_dir.exists() or report_dir.exists():
        raise FileExistsError(
            "Exploratory output already exists. Preserve it and explicitly "
            "choose a new run directory before running another experiment."
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_validation)[:, 1]
        validation_probabilities[name] = probabilities
        metrics = evaluate_predictions(y_validation, probabilities, threshold=0.5)
        metrics["evidence_status"] = "EXPLORATORY_ONLY_LABEL_MATURITY_UNVERIFIED"
        metrics["validation_threshold"] = 0.5
        metrics["threshold_policy"] = (
            "Fixed at 0.5 for descriptive comparison only; not tuned or approved "
            "for operational decisions."
        )
        results[name] = metrics
        joblib.dump(estimator, model_dir / f"{name}.joblib")

    prevalence = float(y_train.mean())
    baseline_probabilities = np.full(len(y_validation), prevalence, dtype=float)
    baseline_metrics = evaluate_predictions(
        y_validation, baseline_probabilities, threshold=0.5
    )
    baseline_metrics["evidence_status"] = (
        "EXPLORATORY_ONLY_LABEL_MATURITY_UNVERIFIED"
    )
    baseline_metrics["threshold_policy"] = (
        "Constant training prevalence; descriptive baseline only."
    )
    results["training_prevalence_baseline"] = baseline_metrics

    subgroup_results = {
        name: evaluate_subgroups(
            split.validation,
            y_validation,
            probabilities,
            threshold=0.5,
            min_support=50,
        )
        for name, probabilities in validation_probabilities.items()
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "subgroup_metrics.json").write_text(
        json.dumps(subgroup_results, indent=2), encoding="utf-8"
    )

    unavailable_models = [
        name
        for name, module in (
            ("CatBoost", "catboost"),
            ("XGBoost", "xgboost"),
            ("LightGBM", "lightgbm"),
        )
        if importlib.util.find_spec(module) is None
    ]
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "status": "EXPLORATORY ONLY",
                "label_maturity": "UNVERIFIED",
                "verified_mature_rows": 0,
                "validation_start": split.validation_start.date().isoformat(),
                "train_rows": len(split.train),
                "validation_rows": len(split.validation),
                "train_positive_count": int(y_train.sum()),
                "validation_positive_count": int(y_validation.sum()),
                "validation_order_date_range": [
                    pd.to_datetime(split.validation["order_placed_at"])
                    .min()
                    .isoformat(),
                    pd.to_datetime(split.validation["order_placed_at"])
                    .max()
                    .isoformat(),
                ],
                "models_not_run_missing_optional_dependencies": unavailable_models,
                "predictions_generated": False,
                "production_artifacts_created": False,
                "warning": (
                    "Labels may be right-censored or otherwise immature. Metrics "
                    "are not valid evidence for deployment, customer decisions, "
                    "financial decisions, or operational threshold selection."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_exploratory_plots(
        report_dir,
        y_validation.to_numpy(),
        validation_probabilities,
    )
    markdown = _exploratory_report_markdown(
        split, y_train, y_validation, results, unavailable_models
    )
    (report_dir / "exploratory_model_results.md").write_text(
        markdown, encoding="utf-8"
    )
    return results


def _write_exploratory_plots(
    report_dir: Path,
    y_true: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
) -> None:
    """Write visual diagnostics to the exploratory report directory only."""
    fig, axis = plt.subplots(figsize=(7, 6))
    for name, probabilities in probabilities_by_model.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    axis.set(xlabel="False positive rate", ylabel="True positive rate")
    axis.set_title("Exploratory ROC curves — label maturity unverified")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    for name, probabilities in probabilities_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        axis.plot(recall, precision, label=name)
    axis.axhline(y_true.mean(), linestyle="--", color="grey", label="prevalence")
    axis.set(xlabel="Recall", ylabel="Precision")
    axis.set_title("Exploratory precision-recall curves — maturity unverified")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "precision_recall_curves.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    for name, probabilities in probabilities_by_model.items():
        observed, predicted = calibration_curve(
            y_true, probabilities, n_bins=10, strategy="uniform"
        )
        axis.plot(predicted, observed, marker="o", label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction")
    axis.set_title("Exploratory calibration — label maturity unverified")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "calibration_curves.png", dpi=150)
    plt.close(fig)


def _exploratory_report_markdown(
    split: ChronologicalSplit,
    y_train: pd.Series,
    y_validation: pd.Series,
    results: dict[str, Any],
    unavailable_models: list[str],
) -> str:
    header = (
        "# Exploratory model results — not business evidence\n\n"
        "> **EXPLORATORY ONLY. Label maturity is unverified.** Delivery dates "
        "and a verified extraction cutoff are unavailable; labels may be "
        "right-censored. These results must not support deployment, customer "
        "decisions, financial decisions, or operational threshold selection.\n\n"
        "## Protocol\n\n"
        f"- Chronological training partition: {len(split.train):,} rows, "
        f"{int(y_train.sum()):,} positive labels; orders before "
        f"{split.validation_start.date()}.\n"
        f"- Chronological holdout: {len(split.validation):,} rows, "
        f"{int(y_validation.sum()):,} positive labels; "
        f"{pd.to_datetime(split.validation['order_placed_at']).min().date()} "
        "through "
        f"{pd.to_datetime(split.validation['order_placed_at']).max().date()}.\n"
        "- Duplicate order IDs do not cross partitions.\n"
        "- Imputation, scaling, and one-hot encoding are fit within training "
        "partitions. Logistic calibration uses expanding chronological folds "
        "from training data only; the validation holdout is not used to fit it.\n"
        "- Model comparison threshold is fixed at 0.5; it was not tuned.\n"
        f"- Optional model libraries unavailable and not run: "
        f"{', '.join(unavailable_models) if unavailable_models else 'none'}.\n\n"
        "## Validation metrics\n\n"
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | "
        "Brier | Log loss | Confusion matrix (true 0/1 × predicted 0/1) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    rows = []
    for name, metrics in results.items():
        confusion = metrics["confusion_matrix_labels_0_1"]
        rows.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} "
            f"| {metrics['recall']:.4f} | {metrics['f1']:.4f} "
            f"| {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} "
            f"| {metrics['brier_score']:.4f} | {metrics['log_loss']:.4f} "
            f"| {confusion} |"
        )
    return header + "\n".join(rows) + (
        "\n\nCalibration curves, Brier score, log loss, and expected calibration "
        "error are descriptive only. See "
        "`../../derived_data/exploratory/metrics.json` and "
        "`../../derived_data/exploratory/subgroup_metrics.json` for detailed "
        "metrics. No prediction file or production model was generated.\n"
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Fit only explicitly authorized production or exploratory models."
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit models only when the corresponding gate permits it.",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Opt into isolated exploratory training; requires EXPLORATORY ONLY gate.",
    )
    args = parser.parse_args()
    status, reason = read_training_gate(root)
    if args.exploratory and not args.fit:
        parser.error("--exploratory requires --fit.")
    if not args.fit:
        print(
            f"MODEL DEVELOPMENT: gate status={status}. "
            "No fit was requested. No predictions.csv was generated.",
            flush=True,
        )
        if status == "APPROVED":
            print(
                "Use --fit only after reviewing this approved gate and fixed split.",
                flush=True,
            )
        elif status == "EXPLORATORY ONLY":
            print(
                "Use --fit --exploratory only for isolated, non-business evidence.",
                flush=True,
            )
        return 0
    try:
        if args.exploratory:
            fit_exploratory_models(root)
        else:
            fit_and_evaluate_if_approved(root)
    except PermissionError as exc:
        print(f"TRAINING BLOCKED: {exc}", flush=True)
        return 2
    if args.exploratory:
        print(
            "EXPLORATORY TRAINING COMPLETED. Outputs are isolated under "
            "derived_data/exploratory/ and reports/exploratory/. "
            "No predictions.csv or production model was generated.",
            flush=True,
        )
        return 0
    print("TRAINING COMMAND COMPLETED; review reports/model_comparison.json.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
