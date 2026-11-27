"""Chronological, exploratory CatBoost/LightGBM/logistic ensemble benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import canonicalize_train
from src.calibrated_types import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FoldFeatureEngineer as StableFoldFeatureEngineer,
    NOTE_KEYWORDS,
    NUMERIC_FEATURES,
    ProbabilityCalibrator as StableProbabilityCalibrator,
)

FoldFeatureEngineer = StableFoldFeatureEngineer
ProbabilityCalibrator = StableProbabilityCalibrator

Q2_START = pd.Timestamp("2026-04-01")
Q3_START = pd.Timestamp("2026-07-01")
BASELINE_ROC_AUC = 0.7057
BASELINE_PR_AUC = 0.2584
RETURN_HANDLING_COST = 1150.0
CALL_COST = 45.0
CALL_EFFECT = 0.35
BLEND_WEIGHTS = {"logistic_regression": 0.55, "catboost": 0.45}
BLEND_MODEL_NAMES = tuple(BLEND_WEIGHTS)
OOF_SPLITS = 3
RANDOM_STATE = 42

FORBIDDEN_COLUMNS = (
    "pickup_scheduled_at",
    "last_service_event_type",
    "source",
    "order_value_inr",
)
NOTE_KEYWORDS = ("gift", "urgent", "call", "office", "leave", "open box")
CATEGORICAL_FEATURES = (
    "sales_channel",
    "payment_mode",
    "is_gift",
    "sku",
    "product_family",
    "product_model_name",
    "pincode_prefix",
)
NUMERIC_FEATURES = (
    "discount_pct",
    "qty",
    "promised_delivery_days",
    "customer_prior_orders",
    "smoothed_prior_return_rate",
    "is_first_time_buyer",
    "is_zero_pincode",
    "has_note",
    "note_length",
    *(f"note_has_{keyword.replace(' ', '_')}" for keyword in NOTE_KEYWORDS),
    "order_hour",
    "day_of_week",
    "is_weekend",
    "is_post_oct2025",
    "days_since_launch",
    "discount_diff_from_family",
)
FEATURE_NAMES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
MODEL_NAMES = ("catboost", "lightgbm", "logistic_regression")


def _find_one(data_dir: Path, suffix: str) -> Path:
    matches = sorted(data_dir.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*{suffix}' file in {data_dir}; found {len(matches)}."
        )
    return matches[0]


def _prepare_model_inputs(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    orders = orders.copy()
    if "launch_date" in orders.columns:
        orders = orders.drop(columns="launch_date")
    if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
        raise ValueError(f"{name} must contain unique, non-null order IDs.")
    if set(FORBIDDEN_COLUMNS).intersection(orders.columns):
        orders = orders.drop(columns=list(FORBIDDEN_COLUMNS))
    required = {
        "order_id",
        "order_placed_at",
        "sku",
        "sales_channel",
        "payment_mode",
        "discount_pct",
        "qty",
        "promised_delivery_days",
        "delivery_pincode",
        "is_gift",
        "customer_prior_orders",
        "customer_prior_returns",
        "delivery_note",
    }
    missing = sorted(required.difference(orders.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    product_features = products[["sku", "family", "model_name", "launch_date"]].copy()
    if product_features["sku"].isna().any() or product_features["sku"].duplicated().any():
        raise ValueError("Product SKUs must be present and unique.")
    product_features["launch_date"] = pd.to_datetime(
        product_features["launch_date"], errors="raise"
    )
    joined = orders.merge(
        product_features,
        on="sku",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        missing_skus = joined.loc[joined["_merge"].ne("both"), "sku"].unique().tolist()
        raise ValueError(f"{name} contains SKUs missing from products.csv: {missing_skus}")
    joined = joined.drop(columns="_merge").rename(
        columns={"family": "product_family", "model_name": "product_model_name"}
    )
    return joined[
        [
            "order_id",
            "order_placed_at",
            "sku",
            "sales_channel",
            "payment_mode",
            "discount_pct",
            "qty",
            "promised_delivery_days",
            "delivery_pincode",
            "is_gift",
            "customer_prior_orders",
            "customer_prior_returns",
            "delivery_note",
            "product_family",
            "product_model_name",
            "launch_date",
        ]
    ].copy()


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data_dir = root / "datasets"
    train_raw = pd.read_csv(
        _find_one(data_dir, "train.csv"),
        dtype={"order_id": "string", "delivery_note": "string"},
    )
    test_raw = pd.read_csv(
        _find_one(data_dir, "test_unlabelled.csv"),
        dtype={"order_id": "string", "delivery_note": "string"},
    )
    products = pd.read_csv(
        _find_one(data_dir, "products.csv"),
        dtype={"sku": "string"},
    )
    sample = pd.read_csv(
        _find_one(data_dir, "sample_submission.csv"),
        dtype={"order_id": "string"},
    )
    if list(sample.columns) != ["order_id", "score"]:
        raise ValueError("sample_submission.csv must have order_id,score columns.")
    if test_raw["order_id"].tolist() != sample["order_id"].tolist():
        raise ValueError("test and sample_submission order IDs or row order differ.")

    canonical, _, duplicate_stats = canonicalize_train(train_raw)
    if duplicate_stats["input_rows"] != 11155 or duplicate_stats["canonical_rows"] != 10504:
        raise ValueError(f"Unexpected train deduplication counts: {duplicate_stats}")
    if not canonical["returned"].isin({0, 1}).all():
        raise ValueError("returned must contain binary 0/1 labels.")
    y = canonical["returned"].astype("int8").reset_index(drop=True)

    train_input = _prepare_model_inputs(canonical, products, "canonical train")
    test_input = _prepare_model_inputs(test_raw, products, "test")
    metadata = {
        **duplicate_stats,
        "train_positive_count": int(y.sum()),
        "train_negative_count": int(len(y) - y.sum()),
        "test_rows": len(test_input),
        "test_order_ids_match_sample": True,
    }
    return train_input, y, test_input, sample, metadata


def _categorical_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def build_estimators() -> dict[str, Pipeline]:
    try:
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost and LightGBM are required; install the versions pinned in "
            "requirements.txt before running this benchmark."
        ) from exc

    categorical = list(CATEGORICAL_FEATURES)
    numeric = list(NUMERIC_FEATURES)
    common = ("feature_engineer", StableFoldFeatureEngineer())
    catboost = Pipeline(
        [
            common,
            (
                "classifier",
                CatBoostClassifier(
                    depth=4,
                    l2_leaf_reg=8,
                    iterations=600,
                    learning_rate=0.03,
                    loss_function="Logloss",
                    random_seed=RANDOM_STATE,
                    cat_features=categorical,
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    )
    tree_preprocessor = ColumnTransformer(
        [
            ("categorical", _categorical_preprocessor(), categorical),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )
    lightgbm = Pipeline(
        [
            ("feature_engineer", StableFoldFeatureEngineer()),
            ("preprocessor", tree_preprocessor),
            (
                "classifier",
                LGBMClassifier(
                    max_depth=4,
                    num_leaves=15,
                    min_child_samples=50,
                    colsample_bytree=0.8,
                    subsample=0.8,
                    subsample_freq=1,
                    learning_rate=0.03,
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    logistic_preprocessor = ColumnTransformer(
        [
            ("categorical", _categorical_preprocessor(), categorical),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )
    logistic = Pipeline(
        [
            ("feature_engineer", StableFoldFeatureEngineer()),
            ("preprocessor", logistic_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {"catboost": catboost, "lightgbm": lightgbm, "logistic_regression": logistic}


class CalibratedProbabilityEnsemble:
    """Serializable model bundle for the optional exploratory FastAPI route."""

    def __init__(
        self,
        estimators: dict[str, Pipeline],
        calibrators: dict[str, ProbabilityCalibrator],
        weights: dict[str, float] | None = None,
        feature_explanations: dict[str, Any] | None = None,
        risk_tier_thresholds: list[float] | None = None,
    ) -> None:
        self.estimators = estimators
        self.calibrators = calibrators
        self.weights = dict(weights or BLEND_WEIGHTS)
        self.feature_explanations = feature_explanations or {}
        self.risk_tier_thresholds = risk_tier_thresholds or []

    def predict_components(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        calibrated: dict[str, np.ndarray] = {}
        for name in BLEND_MODEL_NAMES:
            raw = self.estimators[name].predict_proba(X)[:, 1]
            calibrated[name] = self.calibrators[name].predict(raw)
        calibrated["weighted_blend"] = sum(
            self.weights[name] * calibrated[name] for name in BLEND_MODEL_NAMES
        )
        return calibrated

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_components(X)["weighted_blend"]
        return np.column_stack([1.0 - probabilities, probabilities])


def _build_oof_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    estimators: dict[str, Pipeline],
    n_splits: int = OOF_SPLITS,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    probabilities = {
        name: np.full(len(X), np.nan, dtype=float) for name in MODEL_NAMES
    }
    fold_ids = np.full(len(X), -1, dtype=int)
    fold_details: list[dict[str, Any]] = []
    for fold_number, (train_indices, valid_indices) in enumerate(
        splitter.split(X), start=1
    ):
        y_train = y.iloc[train_indices]
        if y_train.nunique() != 2:
            raise ValueError(f"OOF fold {fold_number} training data lacks a class.")
        if y.iloc[valid_indices].nunique() != 2:
            raise ValueError(f"OOF fold {fold_number} validation data lacks a class.")
        print(
            f"[OOF fold {fold_number}/{n_splits}] "
            f"train={len(train_indices)} validation={len(valid_indices)}",
            flush=True,
        )
        fold_estimators = build_estimators()
        for name in MODEL_NAMES:
            estimator = fold_estimators[name]
            estimator.fit(X.iloc[train_indices], y_train)
            probabilities[name][valid_indices] = estimator.predict_proba(
                X.iloc[valid_indices]
            )[:, 1]
        fold_ids[valid_indices] = fold_number
        fold_details.append(
            {
                "fold": fold_number,
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "train_last_timestamp": str(
                    pd.to_datetime(X.iloc[train_indices]["order_placed_at"]).max()
                ),
                "validation_start": str(
                    pd.to_datetime(X.iloc[valid_indices]["order_placed_at"]).min()
                ),
                "validation_end": str(
                    pd.to_datetime(X.iloc[valid_indices]["order_placed_at"]).max()
                ),
            }
        )
    return probabilities, fold_ids, fold_details


def _select_and_fit_calibrators(
    oof_probabilities: dict[str, np.ndarray],
    fold_ids: np.ndarray,
    y: pd.Series,
) -> tuple[dict[str, ProbabilityCalibrator], dict[str, Any]]:
    oof_mask = fold_ids > 0
    last_fold = int(fold_ids.max())
    selection_train_mask = (fold_ids > 0) & (fold_ids < last_fold)
    selection_valid_mask = fold_ids == last_fold
    if not selection_train_mask.any() or not selection_valid_mask.any():
        raise ValueError("Insufficient expanding OOF folds for calibration selection.")

    calibrators: dict[str, ProbabilityCalibrator] = {}
    details: dict[str, Any] = {}
    target = y.to_numpy(dtype=int)
    for name in MODEL_NAMES:
        raw = oof_probabilities[name]
        if not np.isfinite(raw[oof_mask]).all():
            raise ValueError(f"{name} OOF probabilities contain non-finite values.")
        candidate_scores: dict[str, float] = {}
        for method in ("sigmoid", "isotonic"):
            candidate = StableProbabilityCalibrator(method).fit(
                raw[selection_train_mask], target[selection_train_mask]
            )
            calibrated = candidate.predict(raw[selection_valid_mask])
            candidate_scores[method] = float(
                brier_score_loss(target[selection_valid_mask], calibrated)
            )
        selected_method = min(
            candidate_scores,
            key=lambda method: (candidate_scores[method], method != "sigmoid"),
        )
        calibrators[name] = StableProbabilityCalibrator(selected_method).fit(
            raw[oof_mask], target[oof_mask]
        )
        details[name] = {
            "selected_method": selected_method,
            "selection_fold": last_fold,
            "selection_brier_scores": candidate_scores,
            "oof_calibration_rows": int(oof_mask.sum()),
            "calibrator_training_rows": int(oof_mask.sum()),
        }
    return calibrators, details


def _fit_final_estimators(
    X: pd.DataFrame, y: pd.Series
) -> dict[str, Pipeline]:
    fitted: dict[str, Pipeline] = {}
    estimators = build_estimators()
    for name in MODEL_NAMES:
        print(f"[final fit] {name}: {len(X)} pre-holdout rows", flush=True)
        estimator = estimators[name]
        estimator.fit(X, y)
        fitted[name] = estimator
    return fitted


def _top_rate_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    fraction: float,
) -> dict[str, Any]:
    if not 0 < fraction <= 1:
        raise ValueError("Flagging fraction must be in (0, 1].")
    flagged_count = int(math.ceil(len(y) * fraction))
    order = np.argsort(-probabilities, kind="stable")[:flagged_count]
    flags = np.zeros(len(y), dtype=bool)
    flags[order] = True
    true_positive = int(np.sum(flags & (y == 1)))
    precision = true_positive / flagged_count
    positives = int(np.sum(y == 1))
    recall = true_positive / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    expected_net_savings = float(
        np.sum(probabilities[order] * CALL_EFFECT * RETURN_HANDLING_COST)
        - flagged_count * CALL_COST
    )
    return {
        "flagged_count": flagged_count,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "expected_net_savings_rs": expected_net_savings,
        "true_positive_count": true_positive,
    }


def evaluate_scores(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or scores.ndim != 1 or len(target) != len(scores):
        raise ValueError("Evaluation labels and scores must be aligned 1D arrays.")
    if not set(np.unique(target)).issubset({0, 1}) or len(np.unique(target)) != 2:
        raise ValueError("Holdout evaluation requires both binary classes.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Evaluation scores must be finite probabilities in [0, 1].")
    return {
        "roc_auc": float(roc_auc_score(target, scores)),
        "pr_auc": float(average_precision_score(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
        "top_15_percent": _top_rate_metrics(target, scores, 0.15),
        "top_20_percent": _top_rate_metrics(target, scores, 0.20),
    }


def _decile_calibration(
    y: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, Any]]:
    target = np.asarray(y, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(target) != len(scores) or len(target) == 0:
        raise ValueError("Calibration labels and scores must be non-empty and aligned.")
    bins = pd.qcut(
        pd.Series(scores).rank(method="first"),
        q=min(10, len(scores)),
        labels=False,
    )
    frame = pd.DataFrame(
        {"decile": bins + 1, "actual": target, "predicted": scores}
    )
    grouped = frame.groupby("decile", observed=True).agg(
        rows=("actual", "size"),
        observed_return_rate=("actual", "mean"),
        mean_predicted_probability=("predicted", "mean"),
    )
    return [
        {
            "decile": int(index),
            "rows": int(row["rows"]),
            "observed_return_rate": float(row["observed_return_rate"]),
            "mean_predicted_probability": float(row["mean_predicted_probability"]),
        }
        for index, row in grouped.iterrows()
    ]


def _slice_metrics(
    y: np.ndarray, probabilities: np.ndarray, description: str
) -> dict[str, Any]:
    target = np.asarray(y, dtype=int)
    result: dict[str, Any] = {
        "description": description,
        "rows": int(len(target)),
        "positive_count": int(target.sum()),
        "positive_rate": float(target.mean()) if len(target) else None,
    }
    if len(target) and len(np.unique(target)) == 2:
        result.update(evaluate_scores(target, probabilities))
    else:
        result["ranking_metrics_available"] = False
    return result


def _q2_failure_slices(
    root: Path,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    data_dir = root / "datasets"
    train_raw = pd.read_csv(
        _find_one(data_dir, "train.csv"),
        dtype={
            "order_id": "string",
            "customer_id": "string",
            "delivery_note": "string",
        },
    )
    canonical, _, _ = canonicalize_train(train_raw)
    raw_by_id = canonical.set_index("order_id")
    holdout_raw = raw_by_id.loc[X_holdout["order_id"].tolist()].reset_index()
    if holdout_raw["order_id"].tolist() != X_holdout["order_id"].tolist():
        raise AssertionError("Raw Q2 diagnostic rows do not align with holdout order.")

    def add_slice(name: str, mask: pd.Series, description: str) -> None:
        selected = np.asarray(mask, dtype=bool)
        slices[name] = _slice_metrics(
            y_holdout[selected], probabilities[selected], description
        )

    slices: dict[str, Any] = {}
    post_dispatch_evidence = (
        holdout_raw["last_service_event_type"].fillna("NONE").ne("NONE")
        | holdout_raw["pickup_scheduled_at"].notna()
    )
    add_slice(
        "post_dispatch_evidence",
        post_dispatch_evidence,
        "Orders with a pickup timestamp or non-NONE service event; excluded from model features.",
    )
    zero_pincode = pd.to_numeric(
        holdout_raw["delivery_pincode"], errors="raise"
    ).eq(0)
    add_slice(
        "zero_pincode_walk_ins",
        zero_pincode,
        "Walk-in partner orders using the 000000 pincode sentinel.",
    )
    order_times = pd.to_datetime(holdout_raw["order_placed_at"], errors="raise")
    add_slice(
        "pre_oct_2025_migration",
        order_times.lt(pd.Timestamp("2025-10-01")),
        "Orders before the October 2025 migration boundary; expected empty in Q2 2026.",
    )
    add_slice(
        "post_oct_2025_migration",
        order_times.ge(pd.Timestamp("2025-10-01")),
        "Orders on or after the October 2025 migration boundary.",
    )

    customer_files = sorted(data_dir.glob("*customers.csv"))
    if len(customer_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one customer reference file; found {len(customer_files)}."
        )
    customers = pd.read_csv(customer_files[0], dtype={"customer_id": "string"})
    membership = holdout_raw[["customer_id"]].merge(
        customers[["customer_id", "shield_member"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )["shield_member"]
    if membership.isna().any():
        raise ValueError("Some Q2 rows have no Shield reference status.")
    shield_member = membership.astype("string").str.upper().eq("Y")
    add_slice(
        "current_shield_members",
        shield_member,
        "Current customer reference says Shield member; historical membership effective dates are unavailable.",
    )
    add_slice(
        "current_non_shield_members",
        ~shield_member,
        "Current customer reference says non-member; historical membership effective dates are unavailable.",
    )
    return slices


def _baseline_scores(root: Path, holdout_start: pd.Timestamp) -> tuple[np.ndarray | None, str]:
    baseline_path = (
        root / "derived_data" / "exploratory" / "model_operating_point" / "selected_model.joblib"
    )
    if not baseline_path.is_file():
        return None, "Saved baseline artifact unavailable; reference AUCs are report-only."
    frame_path = root / "derived_data" / "train_modeling.csv"
    if not frame_path.is_file():
        return None, "Prepared baseline validation table unavailable; reference AUCs are report-only."
    frame = pd.read_csv(
        frame_path,
        dtype={"order_id": "string", "delivery_pincode": "string"},
    )
    times = pd.to_datetime(frame["order_placed_at"], errors="raise")
    holdout = frame.loc[(times >= holdout_start) & (times < Q3_START)]
    if holdout.empty:
        raise ValueError("Prepared baseline table has no Q2 2026 holdout rows.")
    model = joblib.load(baseline_path)
    scores = np.asarray(model.predict_proba(holdout)[:, 1], dtype=float)
    return scores, "Predictions from the saved prior Logistic Regression pipeline."


def _feature_explanations(estimators: dict[str, Pipeline]) -> dict[str, Any]:
    explanations: dict[str, Any] = {}
    for name, estimator in estimators.items():
        if name == "catboost":
            classifier = estimator.named_steps["classifier"]
            values = np.asarray(classifier.feature_importances_, dtype=float)
            pairs = [
                {"feature": feature, "importance": float(value)}
                for feature, value in zip(FEATURE_NAMES, values)
            ]
            pairs.sort(key=lambda item: item["importance"], reverse=True)
            explanations[name] = pairs[:20]
            continue

        preprocessor = estimator.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        classifier = estimator.named_steps["classifier"]
        if name == "lightgbm":
            values = np.asarray(classifier.feature_importances_, dtype=float)
            pairs = [
                {"feature": str(feature), "importance": float(value)}
                for feature, value in zip(feature_names, values)
            ]
            pairs.sort(key=lambda item: item["importance"], reverse=True)
            explanations[name] = pairs[:20]
        else:
            coefficients = np.asarray(classifier.coef_[0], dtype=float)
            pairs = [
                {
                    "feature": str(feature),
                    "coefficient": float(value),
                    "direction": "raises" if value > 0 else "lowers",
                }
                for feature, value in zip(feature_names, coefficients)
            ]
            pairs.sort(key=lambda item: abs(item["coefficient"]), reverse=True)
            explanations[name] = pairs[:20]
    return explanations


def _render_table(results: dict[str, dict[str, Any]]) -> str:
    header = (
        "| Model | ROC-AUC | PR-AUC | Brier | P/R/F1 top 15% | Net savings 15% | "
        "P/R/F1 top 20% | Net savings 20% |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for name, metrics in results.items():
        if metrics.get("scores_available") is False:
            rows.append(
                f"| {name} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
                "n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        top15 = metrics["top_15_percent"]
        top20 = metrics["top_20_percent"]
        rows.append(
            f"| {name} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
            f"{metrics['brier_score']:.4f} | "
            f"{top15['precision']:.3f}/{top15['recall']:.3f}/{top15['f1']:.3f} | "
            f"Rs {top15['expected_net_savings_rs']:.0f} | "
            f"{top20['precision']:.3f}/{top20['recall']:.3f}/{top20['f1']:.3f} | "
            f"Rs {top20['expected_net_savings_rs']:.0f} |"
        )
    return "\n".join(rows)


def _render_decile_table(deciles: list[dict[str, Any]]) -> str:
    rows = [
        "| Score decile (low to high) | Rows | Mean predicted risk | Observed return rate |",
        "|---:|---:|---:|---:|",
    ]
    rows.extend(
        f"| {row['decile']} | {row['rows']} | "
        f"{row['mean_predicted_probability']:.3f} | {row['observed_return_rate']:.3f} |"
        for row in deciles
    )
    return "\n".join(rows)


def _render_failure_slices(slices: dict[str, Any]) -> str:
    rows = [
        "| Slice | Rows | Returns | Return rate | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in slices.items():
        rate = result["positive_rate"]
        roc_auc = result.get("roc_auc")
        pr_auc = result.get("pr_auc")
        rows.append(
            f"| {name} | {result['rows']} | {result['positive_count']} | "
            f"{rate:.3f}" if rate is not None else
            f"| {name} | {result['rows']} | {result['positive_count']} | n/a"
        )
        rows[-1] += (
            f" | {roc_auc:.3f} | {pr_auc:.3f} |"
            if roc_auc is not None and pr_auc is not None
            else " | n/a | n/a |"
        )
    return "\n".join(rows)


def _write_report(
    path: Path,
    results: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    baseline_note: str,
) -> None:
    blend = results["weighted_blend"]
    beats = (
        blend["roc_auc"] > BASELINE_ROC_AUC and blend["pr_auc"] > BASELINE_PR_AUC
    )
    content = f"""# Calibrated ensemble benchmark

> **EXPLORATORY ONLY — label maturity is unverified.** Q2 2026 is a reused
> chronological cohort, not an untouched estimate of future performance.

## Result

- Holdout: 2026-04-01 through 2026-06-30, {metadata["holdout_rows"]:,} orders.
- Training period: before 2026-04-01, {metadata["training_rows"]:,} orders.
- Prior baseline gate: ROC-AUC > {BASELINE_ROC_AUC:.4f} **and** PR-AUC > {BASELINE_PR_AUC:.4f}.
- Final blend (55% Logistic Regression + 45% CatBoost): ROC-AUC {blend["roc_auc"]:.4f}; PR-AUC {blend["pr_auc"]:.4f}.
- Top-15% recall is {blend["top_15_percent"]["recall"]:.3f}; random selection at the same capacity would capture about 0.150 of returns ({blend["top_15_percent"]["recall"] / 0.15:.2f}x as many, descriptively).
- Both baseline gates beaten: **{"YES" if beats else "NO"}**.
- Submission file written: **{"Yes, as predictions.csv" if metadata["submission_written"] else "No"}**.
- Baseline prediction source: {baseline_note}

## Q2 2026 holdout metrics

{_render_table(results)}

Expected net savings is computed for flagged orders as
`sum(calibrated_probability × 35% × Rs 1,150) - Rs 45 × flagged_count`.
It is a policy-based estimate, not measured causal savings. Precision, recall,
and F1 use observed holdout labels. The 15%/20% cutoffs are ranked capacities,
not policy-approved actions.

## Calibration and failure slices

The decile table is a descriptive check on the reused Q2 holdout. Shield
membership comes from a current reference without an effective date;
post-dispatch fields are used only for error slicing and never as predictors.
Q2 contains only post-October-2025 migration orders, so a pre-migration Q2
comparison is unavailable.

{_render_decile_table(metadata["decile_calibration"])}

{_render_failure_slices(metadata["failure_slices"])}

## Protocol and feature controls

- CRM rows retained for exact CRM/partner-feed duplicate pairs; non-source
  fields, including labels, must agree. Discarded partner records are audited
  in memory and summarized in experiment metadata; source data is not changed.
- CatBoost, LightGBM, and Logistic Regression use the requested regularized
  configurations. Blend weights are 0.55 Logistic Regression and 0.45
  CatBoost; LightGBM remains comparison-only.
- Three expanding-window folds produce out-of-fold probabilities. Sigmoid and
  isotonic calibration are compared on the final training-side OOF fold only;
  selected calibrators are refit on OOF predictions before scoring Q2.
- Product-family discount averages are fitted inside each estimator fold and
  applied to that fold's validation rows.
- `pickup_scheduled_at`, `last_service_event_type`, `source`, and
  `order_value_inr` are dropped before model input.
- Current `signup_date` and unversioned catalog prices are excluded because
  historical effective dates are unavailable. Thus `customer_tenure_days`,
  `is_retroactive_signup`, and `catalog_net_value` are deliberately not used.
- Prior customer counters are included only as the supplied pre-order counts;
  their underlying construction was not independently verified. Confirm their
  as-of semantics before production use.
- Delivery-note text is transformed locally into length and six binary keyword
  flags; the raw note is not passed to estimators or copied into outputs.
- July-September test outcomes are unavailable and were not read or used in
  calibration, method selection, or holdout evaluation.

## Limitations

The supplied Q2 cohort has been reused in earlier experiments, and label
maturity, delivery dates, and label extraction cutoff remain unverified. Its
metrics and policy savings are descriptive only. No production promotion,
live-order decision, or change to the existing `predictions.csv` is authorized.
"""
    path.write_text(content, encoding="utf-8")


def run_benchmark(root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; preserve the prior run and choose a new output directory."
        )
    report_name = (
        "calibrated_ensemble_benchmark.md"
        if output_dir.name == "calibrated_ensemble_benchmark"
        else f"calibrated_ensemble_benchmark_{output_dir.name}.md"
    )
    report_path = root / "reports" / report_name
    if report_path.exists():
        raise FileExistsError(
            f"{report_path} already exists; preserve it and choose a new report name."
        )

    X_all, y_all, X_test, sample, input_metadata = load_inputs(root)
    times = pd.to_datetime(X_all["order_placed_at"], errors="raise")
    train_mask = times < Q2_START
    holdout_mask = (times >= Q2_START) & (times < Q3_START)
    outside_mask = ~(train_mask | holdout_mask)
    if outside_mask.any():
        raise ValueError(
            "Labeled training rows exist outside the pre-Q2 and Q2 holdout periods."
        )
    X_train = X_all.loc[train_mask].reset_index(drop=True)
    y_train = y_all.loc[train_mask].reset_index(drop=True)
    X_holdout = X_all.loc[holdout_mask].reset_index(drop=True)
    y_holdout = y_all.loc[holdout_mask].to_numpy(dtype=int)
    if X_train.empty or X_holdout.empty:
        raise ValueError("Chronological train and Q2 holdout must both be non-empty.")
    if y_train.nunique() != 2 or len(np.unique(y_holdout)) != 2:
        raise ValueError("Train and Q2 holdout must contain both target classes.")
    train_last = pd.to_datetime(X_train["order_placed_at"]).max()
    holdout_first = pd.to_datetime(X_holdout["order_placed_at"]).min()
    holdout_last = pd.to_datetime(X_holdout["order_placed_at"]).max()
    if train_last >= holdout_first:
        raise AssertionError("Training data reaches or overlaps the Q2 holdout.")

    print(
        f"Deduplicated {input_metadata['input_rows']} -> "
        f"{input_metadata['canonical_rows']} orders; "
        f"train={len(X_train)}, Q2 holdout={len(X_holdout)}",
        flush=True,
    )
    base_estimators = build_estimators()
    oof_probabilities, fold_ids, fold_details = _build_oof_probabilities(
        X_train, y_train, base_estimators
    )
    calibrators, calibration_details = _select_and_fit_calibrators(
        oof_probabilities, fold_ids, y_train
    )
    fitted_estimators = _fit_final_estimators(X_train, y_train)
    holdout_components: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        raw_holdout = fitted_estimators[name].predict_proba(X_holdout)[:, 1]
        holdout_components[name] = calibrators[name].predict(raw_holdout)
    holdout_components["weighted_blend"] = sum(
        BLEND_WEIGHTS[name] * holdout_components[name] for name in BLEND_MODEL_NAMES
    )

    results: dict[str, dict[str, Any]] = {
        name: evaluate_scores(y_holdout, scores)
        for name, scores in holdout_components.items()
    }
    baseline_scores, baseline_note = _baseline_scores(root, Q2_START)
    if baseline_scores is not None:
        if len(baseline_scores) != len(y_holdout):
            raise ValueError("Saved baseline prediction row count differs from Q2 holdout.")
        results["saved_logistic_baseline"] = evaluate_scores(
            y_holdout, baseline_scores
        )
        baseline_auc_matches = math.isclose(
            results["saved_logistic_baseline"]["roc_auc"],
            BASELINE_ROC_AUC,
            abs_tol=0.005,
        )
        baseline_ap_matches = math.isclose(
            results["saved_logistic_baseline"]["pr_auc"],
            BASELINE_PR_AUC,
            abs_tol=0.005,
        )
        if not (baseline_auc_matches and baseline_ap_matches):
            baseline_note += (
                " The artifact's measured AUCs do not reproduce the user-stated "
                "reference within 0.005; both results are reported separately."
            )
    results["user_stated_baseline_reference"] = {
        "roc_auc": BASELINE_ROC_AUC,
        "pr_auc": BASELINE_PR_AUC,
        "scores_available": False,
    }
    holdout_prevalence = float(y_holdout.mean())
    results["majority_class_baseline"] = {
        "roc_auc": 0.5,
        "pr_auc": holdout_prevalence,
        "scores_available": False,
        "accuracy": float(1 - holdout_prevalence),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    beats_baseline = (
        results["weighted_blend"]["roc_auc"] > BASELINE_ROC_AUC
        and results["weighted_blend"]["pr_auc"] > BASELINE_PR_AUC
    )

    decile_calibration = _decile_calibration(
        y_holdout, holdout_components["weighted_blend"]
    )
    failure_slices = _q2_failure_slices(
        root, X_holdout, y_holdout, holdout_components["weighted_blend"]
    )
    output_dir.mkdir(parents=True)
    metrics = {
        "status": "EXPLORATORY ONLY",
        "label_maturity": "UNVERIFIED",
        "holdout_reused_in_prior_experiments": True,
        "holdout_start": str(holdout_first),
        "holdout_end": str(holdout_last),
        "training_end": str(train_last),
        "training_rows": len(X_train),
        "holdout_rows": len(X_holdout),
        "holdout_positive_count": int(y_holdout.sum()),
        "holdout_positive_rate": float(y_holdout.mean()),
        "class_weights": "none",
        "blend_weights": BLEND_WEIGHTS,
        "blend_models": list(BLEND_MODEL_NAMES),
        "calibration": calibration_details,
        "oof_folds": fold_details,
        "features": list(FEATURE_NAMES),
        "excluded_features": list(FORBIDDEN_COLUMNS)
        + [
            "customer_tenure_days",
            "is_retroactive_signup",
            "catalog_net_value",
        ],
        "feature_time_safety": (
            "Fold-local family discount means; prior counters are used as documented "
            "pre-order counts; current signup and unversioned price references excluded."
        ),
        "baseline_gate": {
            "roc_auc_strictly_greater_than": BASELINE_ROC_AUC,
            "pr_auc_strictly_greater_than": BASELINE_PR_AUC,
            "weighted_blend_beats_both": beats_baseline,
        },
        "baseline_prediction_source": baseline_note,
        "results": results,
        "decile_calibration": decile_calibration,
        "failure_slices": failure_slices,
        "input": input_metadata,
        "submission_written": False,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )

    predictions = pd.DataFrame(
        {
            "order_id": X_all.loc[holdout_mask, "order_id"].reset_index(drop=True),
            "actual_returned": y_holdout,
            **{
                f"{name}_probability": scores
                for name, scores in holdout_components.items()
            },
        }
    )
    predictions.to_csv(output_dir / "q2_holdout_predictions.csv", index=False)

    if beats_baseline:
        print(
            "Both user-stated AUC gates are beaten; fitting final artifact on all "
            "labeled orders for the unlabeled submission.",
            flush=True,
        )
        full_oof, full_fold_ids, full_folds = _build_oof_probabilities(
            X_all, y_all, build_estimators()
        )
        full_calibrators, full_calibration_details = _select_and_fit_calibrators(
            full_oof, full_fold_ids, y_all
        )
        oof_mask = full_fold_ids > 0
        calibrated_oof = {
            name: full_calibrators[name].predict(full_oof[name][oof_mask])
            for name in BLEND_MODEL_NAMES
        }
        blended_oof = sum(
            BLEND_WEIGHTS[name] * calibrated_oof[name]
            for name in BLEND_MODEL_NAMES
        )
        tier_thresholds = np.quantile(blended_oof, [1 / 3, 2 / 3]).tolist()
        full_estimators = _fit_final_estimators(X_all, y_all)
        explanations = _feature_explanations(full_estimators)
        predictor = CalibratedProbabilityEnsemble(
            full_estimators,
            full_calibrators,
            BLEND_WEIGHTS,
            explanations,
            tier_thresholds,
        )
        test_scores = predictor.predict_proba(X_test)[:, 1]
        if (
            len(test_scores) != len(sample)
            or len(sample) != 2096
            or not np.isfinite(test_scores).all()
        ):
            raise AssertionError("Test scores must be finite and match sample row count.")
        if ((test_scores < 0) | (test_scores > 1)).any():
            raise AssertionError("Test scores must be probabilities in [0, 1].")
        if X_test["order_id"].tolist() != sample["order_id"].tolist():
            raise AssertionError("Test order IDs no longer match sample order.")
        if len(np.unique(test_scores)) < 2:
            raise AssertionError("Submission scores must not be constant.")

        model_dir = output_dir / "models"
        model_dir.mkdir()
        joblib.dump(
            {
                "estimators": full_estimators,
                "calibrators": full_calibrators,
                "weights": BLEND_WEIGHTS,
                "risk_tier_thresholds": tier_thresholds,
                "risk_tier_basis": (
                    "Training-side OOF score tertiles; relative bands, not business thresholds."
                ),
                "feature_explanations": explanations,
                "status": "EXPLORATORY ONLY",
                "production_status": "BLOCKED",
            },
            model_dir / "calibrated_ensemble.joblib",
        )
        (output_dir / "feature_explanations.json").write_text(
            json.dumps(explanations, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (output_dir / "model_metadata.json").write_text(
            json.dumps(
                {
                    "status": "EXPLORATORY ONLY",
                    "production_status": "BLOCKED",
                    "feature_names": list(FEATURE_NAMES),
                    "categorical_features": list(CATEGORICAL_FEATURES),
                    "blend_weights": BLEND_WEIGHTS,
                    "risk_tier_thresholds": tier_thresholds,
                    "risk_tier_basis": (
                        "Training-side OOF score tertiles; relative bands, not business thresholds."
                    ),
                    "calibration": full_calibration_details,
                    "oof_folds": full_folds,
                    "label_maturity": "UNVERIFIED",
                    "q2_holdout_was_used_for_gate": True,
                    "test_labels_used": False,
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        submission_path = root / "predictions.csv"
        temporary_path = root / "predictions.csv.tmp"
        pd.DataFrame(
            {
                "order_id": sample["order_id"].astype("string"),
                "score": test_scores,
            }
        ).to_csv(temporary_path, index=False)
        temporary_path.replace(submission_path)
        metrics["submission_written"] = True
        metrics["submission_path"] = str(submission_path)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, allow_nan=False, default=str),
            encoding="utf-8",
        )
    else:
        print(
            "At least one AUC gate was not beaten; predictions.csv and final model "
            "artifact were not created.",
            flush=True,
        )

    _write_report(
        report_path,
        results,
        metrics,
        baseline_note,
    )
    print("\n" + _render_table(results), flush=True)
    print(f"\nReport: {report_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a calibrated ensemble on the chronological Q2 holdout."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "derived_data" / "calibrated_ensemble_final",
    )
    args = parser.parse_args()
    run_benchmark(ROOT, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
