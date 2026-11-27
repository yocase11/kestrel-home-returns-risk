"""Model and evaluation utilities; this module does not fit models or score orders."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data_processing import FEATURE_COLUMNS


TARGET_COLUMN = "returned"
ORDER_ID_COLUMN = "order_id"
ORDER_TIME_COLUMN = "order_placed_at"
VALIDATION_START = pd.Timestamp("2026-04-01")
OBSERVATION_WINDOWS_DAYS = (10, 7, 30)

CATEGORICAL_FEATURES = (
    "sales_channel",
    "payment_mode",
    "delivery_pincode",
    "is_gift",
    "sku",
    "product_family",
    "product_model_name",
    "channel_payment_mode",
)
NUMERIC_FEATURES = tuple(
    feature for feature in FEATURE_COLUMNS if feature not in CATEGORICAL_FEATURES
) + ("discount_qty_interaction", "promised_days_weekend_interaction")
SUBGROUP_COLUMNS = (
    "sales_channel",
    "payment_mode",
    "product_family",
    "is_gift",
    "has_default_pincode",
    "order_month",
)
PERFORMANCE_CATEGORICAL_FEATURES = (
    "channel_product_family",
    "payment_product_family",
    "order_hour_band",
    "discount_band",
    "promised_days_band",
)


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    validation_start: pd.Timestamp


def _validate_model_table(frame: pd.DataFrame, require_target: bool = True) -> None:
    required = {ORDER_ID_COLUMN, ORDER_TIME_COLUMN, *FEATURE_COLUMNS}
    if require_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Model table is missing required columns: {missing}")
    if frame[ORDER_ID_COLUMN].isna().any() or frame[ORDER_ID_COLUMN].duplicated().any():
        raise ValueError("Model table must contain one non-null row per order_id.")
    if frame[ORDER_TIME_COLUMN].isna().any():
        raise ValueError("Model table contains a missing order_placed_at.")
    if require_target and not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("returned must contain only binary 0/1 values.")


def observation_window_experiment(
    frame: pd.DataFrame,
    windows_days: Sequence[int] = OBSERVATION_WINDOWS_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Count provisional order-date proxy candidates without asserting label maturity."""
    _validate_model_table(frame)
    if any(days <= 0 for days in windows_days):
        raise ValueError("Observation windows must be positive calendar-day counts.")

    order_times = pd.to_datetime(frame[ORDER_TIME_COLUMN], errors="raise")
    if order_times.dt.tz is not None:
        raise ValueError("order_placed_at must be timezone-naive IST.")
    proxy_cutoff = order_times.max().normalize()

    rows: list[dict[str, Any]] = []
    for window in windows_days:
        latest_proxy_eligible_date = proxy_cutoff - pd.Timedelta(
            value=window, unit="D"
        )
        candidate_mask = order_times.dt.normalize() <= latest_proxy_eligible_date
        eligible_labels = frame.loc[candidate_mask, TARGET_COLUMN]
        rows.append(
            {
                "window_days": int(window),
                "verified_delivery_date_available": False,
                "verified_extract_cutoff_available": False,
                "proxy_cutoff_date": proxy_cutoff.date().isoformat(),
                "proxy_start_date": latest_proxy_eligible_date.date().isoformat(),
                "candidate_rows_order_date_proxy": int(candidate_mask.sum()),
                "candidate_excluded_rows_order_date_proxy": int(
                    (~candidate_mask).sum()
                ),
                "candidate_positive_labels": int(eligible_labels.sum()),
                "candidate_negative_labels": int(eligible_labels.eq(0).sum()),
                "verified_mature_rows": 0,
                "maturity_status": "unverifiable_missing_delivery_date_and_extract_cutoff",
            }
        )

    evidence = {
        "delivery_date_column_available": False,
        "extract_cutoff_available": False,
        "verified_mature_row_count": 0,
        "proxy_cutoff_definition": (
            "Calendar date of the latest order_placed_at in the prepared train table; "
            "this is not asserted to be the label extraction cutoff."
        ),
        "eligibility_proxy_definition": (
            "Candidate order_placed_at calendar dates on or before proxy cutoff minus "
            "the named number of days. This is not delivery-date-based maturity."
        ),
    }
    return pd.DataFrame(rows), evidence


def chronological_split(
    frame: pd.DataFrame,
    validation_start: pd.Timestamp = VALIDATION_START,
) -> ChronologicalSplit:
    """Create a deterministic expanding-history / latest-quarter holdout."""
    _validate_model_table(frame)
    validation_start = pd.Timestamp(validation_start)
    if validation_start.tz is not None:
        raise ValueError("validation_start must be timezone-naive IST.")
    order_times = pd.to_datetime(frame[ORDER_TIME_COLUMN], errors="raise")
    if order_times.dt.tz is not None:
        raise ValueError("order_placed_at must be timezone-naive IST.")

    train_mask = order_times < validation_start
    validation_mask = ~train_mask
    train = frame.loc[train_mask].copy()
    validation = frame.loc[validation_mask].copy()
    if train.empty or validation.empty:
        raise ValueError("Chronological split produced an empty partition.")
    if set(train[ORDER_ID_COLUMN]).intersection(validation[ORDER_ID_COLUMN]):
        raise AssertionError("An order_id crossed the chronological split.")

    train_times = pd.to_datetime(train[ORDER_TIME_COLUMN])
    validation_times = pd.to_datetime(validation[ORDER_TIME_COLUMN])
    if train_times.max() >= validation_times.min():
        raise AssertionError("Chronological split contains future-to-past overlap.")
    train = train.assign(_split_time=train_times).sort_values(
        "_split_time", kind="stable"
    ).drop(columns="_split_time")
    validation = validation.assign(_split_time=validation_times).sort_values(
        "_split_time", kind="stable"
    ).drop(columns="_split_time")
    return ChronologicalSplit(
        train=train,
        validation=validation,
        validation_start=validation_start,
    )


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing model features: {missing}")
    features = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in CATEGORICAL_FEATURES:
        if column not in features.columns:
            continue
        values = features[column].astype("string")
        if column == "delivery_pincode":
            values = values.str.zfill(6)
        if values.isna().any():
            raise ValueError(f"Categorical feature {column!r} contains missing values.")
        features[column] = values.astype(object)
    features["channel_payment_mode"] = (
        features["sales_channel"].astype(str)
        + "__"
        + features["payment_mode"].astype(str)
    )
    features["discount_qty_interaction"] = (
        pd.to_numeric(features["discount_pct"], errors="coerce")
        * pd.to_numeric(features["qty"], errors="coerce")
    )
    features["promised_days_weekend_interaction"] = (
        pd.to_numeric(features["promised_delivery_days"], errors="coerce")
        * pd.to_numeric(features["is_weekend"], errors="coerce")
    )
    return features


def performance_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive dispatch-time interactions and cyclical features for experiments."""
    features = _feature_frame(frame)
    features["channel_product_family"] = (
        features["sales_channel"].astype(str)
        + "__"
        + features["product_family"].astype(str)
    )
    features["payment_product_family"] = (
        features["payment_mode"].astype(str)
        + "__"
        + features["product_family"].astype(str)
    )
    hour = pd.to_numeric(features["order_hour"], errors="coerce")
    weekday = pd.to_numeric(features["order_day_of_week"], errors="coerce")
    month = pd.to_numeric(features["order_month"], errors="coerce")
    discount = pd.to_numeric(features["discount_pct"], errors="coerce")
    promised_days = pd.to_numeric(
        features["promised_delivery_days"], errors="coerce"
    )
    features["order_hour_band"] = pd.cut(
        hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["overnight", "morning", "afternoon", "evening"],
    ).astype("string")
    features["discount_band"] = pd.cut(
        discount,
        bins=[-np.inf, 0, 10, 20, np.inf],
        labels=["none", "low", "medium", "high"],
        include_lowest=True,
    ).astype("string")
    features["promised_days_band"] = pd.cut(
        promised_days,
        bins=[-np.inf, 1, 3, 7, np.inf],
        labels=["0-1", "2-3", "4-7", "8+"],
        include_lowest=True,
    ).astype("string")
    for column in PERFORMANCE_CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("unknown").astype(object)
    features["order_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["order_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["order_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["order_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    features["order_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    features["order_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return features


def build_logistic_pipeline() -> Pipeline:
    """Construct an unfitted Logistic Regression pipeline on the safe allowlist."""
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("select_features", FunctionTransformer(_feature_frame, validate=False)),
            ("features", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Construct an unfitted, encoded Random Forest pipeline."""
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("select_features", FunctionTransformer(_feature_frame, validate=False)),
            ("features", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_encoded_tree_pipeline(
    classifier: BaseEstimator,
    feature_selector: BaseEstimator | None = None,
) -> Pipeline:
    """Construct an unfitted encoded pipeline for tree libraries without native cats."""
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )
    steps: list[tuple[str, Any]] = [
        ("select_features", FunctionTransformer(_feature_frame, validate=False)),
        ("features", preprocessor),
    ]
    if feature_selector is not None:
        steps.append(("feature_selector", feature_selector))
    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)


def build_dense_tree_pipeline(
    classifier: BaseEstimator,
    feature_selector: BaseEstimator | None = None,
) -> Pipeline:
    """Construct a dense one-hot pipeline for scikit-learn histogram/tree models."""
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )
    steps: list[tuple[str, Any]] = [
        ("select_features", FunctionTransformer(_feature_frame, validate=False)),
        ("features", preprocessor),
    ]
    if feature_selector is not None:
        steps.append(("feature_selector", feature_selector))
    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)



def build_catboost_pipeline() -> Pipeline:
    """Construct an unfitted CatBoost pipeline, requiring CatBoost only when called."""
    try:
        catboost = importlib.import_module("catboost")
    except ImportError as exc:
        raise ImportError(
            "CatBoost is required to construct this pipeline. Install the approved "
            "CatBoost dependency before model training."
        ) from exc

    classifier = catboost.CatBoostClassifier(
        cat_features=tuple(CATEGORICAL_FEATURES),
        iterations=500,
        depth=5,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=1.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    return Pipeline(
        steps=[
            ("features", FunctionTransformer(_feature_frame, validate=False)),
            ("classifier", classifier),
        ]
    )


def build_chronological_calibrator(
    estimator: BaseEstimator,
    n_splits: int = 3,
) -> CalibratedClassifierCV:
    """Wrap an estimator in expanding chronological-fold sigmoid calibration."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    return CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=n_splits),
        ensemble=True,
    )


def _validated_probabilities(
    y_true: Sequence[int],
    probabilities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if truth.ndim != 1 or probability.ndim != 1 or len(truth) != len(probability):
        raise ValueError("y_true and probabilities must be equal-length 1D arrays.")
    if len(truth) == 0 or not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("Evaluation requires non-empty binary labels.")
    if not np.isfinite(probability).all() or (
        (probability < 0) | (probability > 1)
    ).any():
        raise ValueError("Predicted probabilities must be finite values in [0, 1].")
    if len(np.unique(truth)) != 2:
        raise ValueError("ROC-AUC and PR-AUC require both classes in evaluation data.")
    return truth, probability


def evaluate_predictions(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Compute classification and calibration metrics without fitting a model."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1.")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2.")
    truth, probability = _validated_probabilities(y_true, probabilities)
    predicted = (probability >= threshold).astype(int)
    observed_fraction, mean_probability = calibration_curve(
        truth,
        probability,
        n_bins=calibration_bins,
        strategy="uniform",
    )
    bin_weights = np.histogram(
        probability, bins=np.linspace(0, 1, calibration_bins + 1)
    )[0]
    if len(bin_weights) != len(observed_fraction) or bin_weights.sum() == 0:
        bin_weights = np.ones(len(observed_fraction), dtype=float)
    else:
        bin_weights = bin_weights[: len(observed_fraction)]
        if bin_weights.sum() == 0:
            bin_weights = np.ones(len(observed_fraction), dtype=float)
    ece = float(
        np.average(np.abs(observed_fraction - mean_probability), weights=bin_weights)
    )
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(matrix[0, 0]),
        int(matrix[0, 1]),
        int(matrix[1, 0]),
        int(matrix[1, 1]),
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    return {
        "support": int(len(truth)),
        "positive_count": int(truth.sum()),
        "positive_rate": float(truth.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float((specificity + recall_score(
            truth, predicted, zero_division=0
        )) / 2),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "brier_score": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
        "expected_calibration_error": ece,
        "calibration_bins": [
            {
                "mean_predicted_probability": float(mean),
                "observed_positive_fraction": float(observed),
            }
            for mean, observed in zip(mean_probability, observed_fraction)
        ],
    }


def evaluate_subgroups(
    frame: pd.DataFrame,
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
    min_support: int = 50,
    subgroup_columns: Sequence[str] = SUBGROUP_COLUMNS,
) -> dict[str, list[dict[str, Any]]]:
    """Return descriptive group metrics; small groups are omitted, not pooled."""
    truth, probability = _validated_probabilities(y_true, probabilities)
    if len(frame) != len(truth):
        raise ValueError("Subgroup frame length must match evaluation labels.")
    if min_support < 1:
        raise ValueError("min_support must be positive.")
    results: dict[str, list[dict[str, Any]]] = {}
    for column in subgroup_columns:
        if column not in frame.columns:
            raise ValueError(f"Subgroup column {column!r} is missing.")
        groups: list[dict[str, Any]] = []
        values = frame[column].astype("string").fillna("<MISSING>")
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            support = int(mask.sum())
            if support < min_support:
                continue
            if len(np.unique(truth[mask])) < 2:
                groups.append(
                    {
                        "group": str(value),
                        "support": support,
                        "positive_count": int(truth[mask].sum()),
                        "positive_rate": float(truth[mask].mean()),
                        "metrics_status": "undefined_single_class_group",
                    }
                )
                continue
            metrics = evaluate_predictions(
                truth[mask],
                probability[mask],
                threshold=threshold,
            )
            metrics["group"] = str(value)
            groups.append(metrics)
        results[column] = groups
    return results



