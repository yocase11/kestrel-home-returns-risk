"""Exploratory chronological feature and blend search for Kestrel returns."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibrated_types import ProbabilityCalibrator
from src.train_calibrated_ensemble import (
    BASELINE_PR_AUC,
    BASELINE_ROC_AUC,
    CALL_COST,
    CALL_EFFECT,
    Q2_START,
    Q3_START,
    RETURN_HANDLING_COST,
    load_inputs,
)

RANDOM_STATE = 42
N_SPLITS = 3
LOGISTIC_C_VALUES = (0.05, 0.1, 0.25, 0.5, 1.0)
TARGET_PRIOR_POSITIVES = 1.14
TARGET_PRIOR_STRENGTH = 10.0
FORBIDDEN_COLUMNS = {
    "pickup_scheduled_at",
    "last_service_event_type",
    "source",
    "order_value_inr",
}
ALLOWED_NOTE_TERMS = {"gift", "urgent", "call", "office", "leave", "open", "box"}
BASE_CATEGORICAL = (
    "sales_channel",
    "payment_mode",
    "is_gift",
    "sku",
    "product_family",
    "product_model_name",
    "pincode_prefix",
)
INTERACTION_CATEGORICAL = (
    "payment_channel",
    "payment_zero_pincode",
    "family_high_discount",
    "first_time_payment",
)
CATEGORICAL_FEATURES = BASE_CATEGORICAL + INTERACTION_CATEGORICAL
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
    "note_has_gift",
    "note_has_urgent",
    "note_has_call",
    "note_has_office",
    "note_has_leave",
    "note_has_open_box",
    "order_hour",
    "day_of_week",
    "is_weekend",
    "is_post_oct2025",
    "days_since_launch",
    "discount_diff_from_family",
    "is_high_discount",
    "sku_target_rate",
    "sku_frequency",
    "pincode_prefix_target_rate",
    "pincode_prefix_frequency",
)
ENGINEERED_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ("note_text",)
CATBOOST_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
CALL_BREAK_EVEN = CALL_COST / (RETURN_HANDLING_COST * CALL_EFFECT)


def _as_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("__missing__").astype(str)


def _pincode_prefix(values: pd.Series) -> pd.Series:
    pincode = values.astype("string").str.strip().str.zfill(6)
    if not pincode.str.fullmatch(r"\d{6}").all():
        raise ValueError("delivery_pincode must contain six digits after padding.")
    return (pincode.astype("int64") // 10000).astype(str)


def _safe_note_text(value: Any) -> str:
    """Keep only approved operational keywords; never fit on raw note content."""
    if value is None or pd.isna(value) or not str(value).strip():
        return "missing_note"
    tokens = re.findall(r"[a-z]+", str(value).lower())
    allowed = [token for token in tokens if token in ALLOWED_NOTE_TERMS]
    text = " ".join(allowed)
    if not text:
        return "unmodeled_note"
    return text


def _past_category_values(
    categories: pd.Series,
    times: pd.Series,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "category": categories.astype(str).to_numpy(),
            "time": times.to_numpy(),
            "target": target,
        }
    )
    grouped = frame.groupby(["category", "time"], sort=True, observed=True).agg(
        count=("target", "size"),
        positives=("target", "sum"),
    )
    grouped["prior_count"] = (
        grouped.groupby(level=0, sort=False)["count"].cumsum() - grouped["count"]
    )
    grouped["prior_positives"] = (
        grouped.groupby(level=0, sort=False)["positives"].cumsum()
        - grouped["positives"]
    )
    grouped["rate"] = (
        grouped["prior_positives"] + TARGET_PRIOR_POSITIVES
    ) / (grouped["prior_count"] + TARGET_PRIOR_STRENGTH)

    time_counts = frame.groupby("time", sort=True, observed=True).size()
    prior_total = time_counts.cumsum() - time_counts
    prior_total_map = prior_total.to_dict()
    prior_count_map = grouped["prior_count"].to_dict()
    rate_map = grouped["rate"].to_dict()
    keys = list(zip(frame["category"], frame["time"]))
    frequencies = np.asarray(
        [
            prior_count_map.get(key, 0)
            / max(int(prior_total_map.get(key[1], 0)), 1)
            for key in keys
        ],
        dtype=float,
    )
    rates = np.asarray(
        [
            float(rate_map.get(key, TARGET_PRIOR_POSITIVES / TARGET_PRIOR_STRENGTH))
            for key in keys
        ],
        dtype=float,
    )
    return rates, frequencies


def _past_family_discount_difference(
    families: pd.Series, times: pd.Series, discounts: np.ndarray
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "family": families.astype(str).to_numpy(),
            "time": times.to_numpy(),
            "discount": discounts,
        }
    )
    grouped = frame.groupby(["family", "time"], sort=True, observed=True).agg(
        total=("discount", "sum"),
        count=("discount", "size"),
    )
    grouped["prior_total"] = (
        grouped.groupby(level=0, sort=False)["total"].cumsum() - grouped["total"]
    )
    grouped["prior_count"] = (
        grouped.groupby(level=0, sort=False)["count"].cumsum() - grouped["count"]
    )
    grouped["prior_mean"] = grouped["prior_total"] / grouped["prior_count"].replace(
        0, np.nan
    )
    mapping = grouped["prior_mean"].to_dict()
    keys = list(zip(frame["family"], frame["time"]))
    return np.asarray(
        [
            discounts[index]
            - (
                float(mapping[key])
                if key in mapping and pd.notna(mapping[key])
                else 0.0
            )
            for index, key in enumerate(keys)
        ],
        dtype=float,
    )


class PreDispatchFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create fold-local features and ordered encodings from dispatch-time inputs."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PreDispatchFeatureEngineer":
        if y is None:
            raise ValueError("Target labels are required to fit fold encodings.")
        if FORBIDDEN_COLUMNS.intersection(X.columns):
            raise ValueError("Forbidden post-dispatch/source/payment fields reached fit.")
        target = np.asarray(y, dtype=int)
        if len(target) != len(X) or not set(np.unique(target)).issubset({0, 1}):
            raise ValueError("Fold features require aligned binary training labels.")
        if X["order_id"].duplicated().any():
            raise ValueError("Feature fitting requires unique order IDs.")

        self.train_row_count_ = len(X)
        self.family_discount_means_ = (
            pd.to_numeric(X["discount_pct"], errors="raise")
            .groupby(_as_text(X["product_family"]))
            .mean()
            .to_dict()
        )
        self.global_discount_mean_ = float(
            pd.to_numeric(X["discount_pct"], errors="raise").mean()
        )
        times = pd.to_datetime(X["order_placed_at"], errors="raise")
        categories = {
            "sku": _as_text(X["sku"]),
            "pincode_prefix": _pincode_prefix(X["delivery_pincode"]),
        }
        self.target_maps_: dict[str, dict[str, tuple[int, int]]] = {}
        self.frequency_maps_: dict[str, dict[str, int]] = {}
        for column, values in categories.items():
            table = pd.DataFrame(
                {"category": values.to_numpy(), "target": target}
            ).groupby("category", sort=False, observed=True)["target"].agg(
                count="size", positives="sum"
            )
            self.target_maps_[column] = {
                str(key): (int(row["positives"]), int(row["count"]))
                for key, row in table.iterrows()
            }
            self.frequency_maps_[column] = {
                str(key): int(row["count"]) for key, row in table.iterrows()
            }
        self.fit_times_ = times.reset_index(drop=True)
        return self

    def _engineer(
        self, X: pd.DataFrame, training_target: np.ndarray | None = None
    ) -> pd.DataFrame:
        output = pd.DataFrame(index=np.arange(len(X)))
        times = pd.to_datetime(X["order_placed_at"], errors="raise").reset_index(
            drop=True
        )
        if times.isna().any():
            raise ValueError("order_placed_at must not be missing.")

        for column in BASE_CATEGORICAL:
            if column == "pincode_prefix":
                output[column] = _pincode_prefix(X["delivery_pincode"]).reset_index(
                    drop=True
                )
            else:
                values = _as_text(X[column]).reset_index(drop=True)
                if column == "is_gift":
                    values = values.replace({"1": "Y", "0": "N"})
                output[column] = values

        discounts = pd.to_numeric(X["discount_pct"], errors="raise").to_numpy(
            dtype=float
        )
        quantity = pd.to_numeric(X["qty"], errors="raise").to_numpy(dtype=float)
        promised_days = pd.to_numeric(
            X["promised_delivery_days"], errors="raise"
        ).to_numpy(dtype=float)
        prior_orders = pd.to_numeric(
            X["customer_prior_orders"], errors="raise"
        ).to_numpy(dtype=float)
        prior_returns = pd.to_numeric(
            X["customer_prior_returns"], errors="raise"
        ).to_numpy(dtype=float)
        if (prior_orders < 0).any() or (prior_returns < 0).any():
            raise ValueError("Prior customer counts must be non-negative.")
        if (prior_returns > prior_orders).any():
            raise ValueError("Prior returns cannot exceed prior orders.")

        pincode = X["delivery_pincode"].astype("string").str.strip().str.zfill(6)
        if not pincode.str.fullmatch(r"\d{6}").all():
            raise ValueError("delivery_pincode must contain six digits after padding.")
        zero_pincode = pincode.astype("int64").eq(0).to_numpy(dtype=np.int8)
        first_time = (prior_orders == 0).astype(np.int8)
        family = _as_text(X["product_family"]).reset_index(drop=True)
        payment = _as_text(X["payment_mode"]).reset_index(drop=True)
        sales_channel = _as_text(X["sales_channel"]).reset_index(drop=True)
        high_discount = (discounts >= 25).astype(np.int8)

        output["payment_channel"] = payment + "_" + sales_channel
        output["payment_zero_pincode"] = (
            payment + "_" + pd.Series(zero_pincode).astype(str)
        )
        output["family_high_discount"] = (
            family + "_" + pd.Series(high_discount).astype(str)
        )
        output["first_time_payment"] = (
            pd.Series(first_time).astype(str) + "_" + payment
        )

        notes = X["delivery_note"].fillna("").astype(str).reset_index(drop=True)
        lower_notes = notes.str.lower()
        output["has_note"] = notes.str.strip().ne("").astype(np.int8)
        output["note_length"] = notes.str.len().astype(float)
        for keyword in ("gift", "urgent", "call", "office", "leave"):
            output[f"note_has_{keyword}"] = lower_notes.str.contains(
                keyword, regex=False
            ).astype(np.int8)
        output["note_has_open_box"] = lower_notes.str.contains(
            "open box", regex=False
        ).astype(np.int8)
        output["note_text"] = notes.map(_safe_note_text)

        output["discount_pct"] = discounts
        output["qty"] = quantity
        output["promised_delivery_days"] = promised_days
        output["customer_prior_orders"] = prior_orders
        output["smoothed_prior_return_rate"] = (
            (prior_returns + TARGET_PRIOR_POSITIVES)
            / (prior_orders + TARGET_PRIOR_STRENGTH)
        )
        output["is_first_time_buyer"] = first_time
        output["is_zero_pincode"] = zero_pincode
        output["order_hour"] = times.dt.hour.to_numpy(dtype=np.int8)
        output["day_of_week"] = times.dt.dayofweek.to_numpy(dtype=np.int8)
        output["is_weekend"] = (times.dt.dayofweek >= 5).to_numpy(dtype=np.int8)
        output["is_post_oct2025"] = (
            times >= pd.Timestamp("2025-10-01")
        ).to_numpy(dtype=np.int8)
        launches = pd.to_datetime(X["launch_date"], errors="raise").reset_index(
            drop=True
        )
        days_since_launch = (times - launches).dt.days.to_numpy(dtype=np.int64)
        if (days_since_launch < 0).any():
            raise ValueError("An order precedes its product launch date.")
        output["days_since_launch"] = days_since_launch
        output["is_high_discount"] = high_discount

        if training_target is not None:
            output["discount_diff_from_family"] = _past_family_discount_difference(
                family, times, discounts
            )
            for column, values in (
                ("sku", _as_text(X["sku"]).reset_index(drop=True)),
                ("pincode_prefix", output["pincode_prefix"].astype(str)),
            ):
                rates, frequencies = _past_category_values(
                    values, times, training_target
                )
                output[f"{column}_target_rate"] = rates
                output[f"{column}_frequency"] = frequencies
        else:
            family_means = family.map(self.family_discount_means_).fillna(
                self.global_discount_mean_
            )
            output["discount_diff_from_family"] = discounts - family_means.to_numpy(
                dtype=float
            )
            for column, values in (
                ("sku", _as_text(X["sku"]).reset_index(drop=True)),
                ("pincode_prefix", output["pincode_prefix"].astype(str)),
            ):
                stats = values.map(self.target_maps_[column])
                positives = stats.map(
                    lambda value: value[0] if isinstance(value, tuple) else 0
                ).to_numpy(dtype=float)
                counts = stats.map(
                    lambda value: value[1] if isinstance(value, tuple) else 0
                ).to_numpy(dtype=float)
                output[f"{column}_target_rate"] = (
                    positives + TARGET_PRIOR_POSITIVES
                ) / (counts + TARGET_PRIOR_STRENGTH)
                output[f"{column}_frequency"] = (
                    values.map(self.frequency_maps_[column])
                    .fillna(0)
                    .to_numpy(dtype=float)
                    / max(self.train_row_count_, 1)
                )
        return output.loc[:, list(ENGINEERED_FEATURES)]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._engineer(X)

    def fit_transform(
        self, X: pd.DataFrame, y: Any = None, **fit_params: Any
    ) -> pd.DataFrame:
        self.fit(X, y)
        return self._engineer(X, np.asarray(y, dtype=int))

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(ENGINEERED_FEATURES, dtype=object)


class CatBoostFeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Any = None) -> "CatBoostFeatureSelector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, list(CATBOOST_FEATURES)].copy()

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(CATBOOST_FEATURES, dtype=object)


def build_logistic_pipeline(c_value: float) -> Pipeline:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            (
                "note_tfidf",
                TfidfVectorizer(
                    max_features=40,
                    ngram_range=(1, 2),
                    min_df=15,
                    token_pattern=r"(?u)\b\w+\b",
                ),
                "note_text",
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("feature_engineer", PreDispatchFeatureEngineer()),
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_catboost_pipeline() -> Pipeline:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("CatBoost is required for the shallow-tree benchmark.") from exc
    return Pipeline(
        [
            ("feature_engineer", PreDispatchFeatureEngineer()),
            ("select_features", CatBoostFeatureSelector()),
            (
                "classifier",
                CatBoostClassifier(
                    depth=3,
                    l2_leaf_reg=12,
                    iterations=700,
                    learning_rate=0.03,
                    loss_function="Logloss",
                    random_seed=RANDOM_STATE,
                    cat_features=list(CATEGORICAL_FEATURES),
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                ),
            ),
        ]
    )


def _temporal_folds(
    X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS
) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(
        pd.to_datetime(X["order_placed_at"], errors="raise").to_numpy(),
        kind="stable",
    )
    sorted_X = X.iloc[order].reset_index(drop=True)
    sorted_y = y.iloc[order].reset_index(drop=True)
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(sorted_X))
    for number, (train_indices, valid_indices) in enumerate(folds, start=1):
        if sorted_y.iloc[train_indices].nunique() != 2:
            raise ValueError(f"Temporal fold {number} training split lacks a class.")
        if sorted_y.iloc[valid_indices].nunique() != 2:
            raise ValueError(f"Temporal fold {number} validation split lacks a class.")
    return [(order[train], order[valid]) for train, valid in folds]


def _tune_logistic(
    X: pd.DataFrame, y: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[float, dict[float, dict[str, Any]], dict[float, np.ndarray], np.ndarray]:
    tuning: dict[float, dict[str, Any]] = {}
    oof_by_c: dict[float, np.ndarray] = {}
    fold_ids = np.full(len(X), -1, dtype=int)
    for c_value in LOGISTIC_C_VALUES:
        scores: list[dict[str, float]] = []
        oof = np.full(len(X), np.nan, dtype=float)
        for fold_number, (train_indices, valid_indices) in enumerate(folds, start=1):
            print(
                f"[Logistic C={c_value:g}, fold {fold_number}/{len(folds)}] "
                f"fit={len(train_indices)} validate={len(valid_indices)}",
                flush=True,
            )
            estimator = build_logistic_pipeline(c_value)
            estimator.fit(X.iloc[train_indices], y.iloc[train_indices])
            probabilities = estimator.predict_proba(X.iloc[valid_indices])[:, 1]
            oof[valid_indices] = probabilities
            scores.append(
                {
                    "roc_auc": float(
                        roc_auc_score(y.iloc[valid_indices], probabilities)
                    ),
                    "pr_auc": float(
                        average_precision_score(y.iloc[valid_indices], probabilities)
                    ),
                }
            )
            fold_ids[valid_indices] = fold_number
        tuning[c_value] = {
            "folds": scores,
            "mean_roc_auc": float(np.mean([score["roc_auc"] for score in scores])),
            "mean_pr_auc": float(np.mean([score["pr_auc"] for score in scores])),
        }
        oof_by_c[c_value] = oof
    selected_c = max(
        LOGISTIC_C_VALUES,
        key=lambda value: (
            tuning[value]["mean_pr_auc"],
            tuning[value]["mean_roc_auc"],
            -value,
        ),
    )
    print(
        f"Selected Logistic Regression C={selected_c:g} from training-side "
        f"chronological CV (mean AP={tuning[selected_c]['mean_pr_auc']:.4f}).",
        flush=True,
    )
    return selected_c, tuning, oof_by_c, fold_ids


def _build_catboost_oof(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    oof = np.full(len(X), np.nan, dtype=float)
    fold_ids = np.full(len(X), -1, dtype=int)
    details: list[dict[str, Any]] = []
    for fold_number, (train_indices, valid_indices) in enumerate(folds, start=1):
        print(
            f"[CatBoost fold {fold_number}/{len(folds)}] "
            f"fit={len(train_indices)} validate={len(valid_indices)}",
            flush=True,
        )
        estimator = build_catboost_pipeline()
        estimator.fit(X.iloc[train_indices], y.iloc[train_indices])
        oof[valid_indices] = estimator.predict_proba(X.iloc[valid_indices])[:, 1]
        fold_ids[valid_indices] = fold_number
        details.append(
            {
                "fold": fold_number,
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "train_end": str(
                    pd.to_datetime(X.iloc[train_indices]["order_placed_at"]).max()
                ),
                "validation_start": str(
                    pd.to_datetime(X.iloc[valid_indices]["order_placed_at"]).min()
                ),
            }
        )
    return oof, fold_ids, details


def _fit_calibrator(
    oof_probabilities: np.ndarray, fold_ids: np.ndarray, y: np.ndarray
) -> tuple[ProbabilityCalibrator, dict[str, Any]]:
    valid_oof = fold_ids > 0
    last_fold = int(fold_ids.max())
    selection_train = valid_oof & (fold_ids < last_fold)
    selection_valid = fold_ids == last_fold
    if not selection_train.any() or not selection_valid.any():
        raise ValueError("Insufficient temporal OOF rows to select calibration.")
    candidate_scores: dict[str, float] = {}
    for method in ("sigmoid", "isotonic"):
        candidate = ProbabilityCalibrator(method).fit(
            oof_probabilities[selection_train], y[selection_train]
        )
        calibrated = candidate.predict(oof_probabilities[selection_valid])
        candidate_scores[method] = float(
            brier_score_loss(y[selection_valid], calibrated)
        )
    selected = min(candidate_scores, key=lambda method: (candidate_scores[method], method != "sigmoid"))
    calibrator = ProbabilityCalibrator(selected).fit(
        oof_probabilities[valid_oof], y[valid_oof]
    )
    return calibrator, {
        "selected_method": selected,
        "selection_fold": last_fold,
        "selection_brier_scores": candidate_scores,
        "oof_rows": int(valid_oof.sum()),
    }


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
    }


def _threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "flagged_count": int(predicted.sum()),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "expected_net_savings_rs": float(
            scores[predicted].sum() * CALL_EFFECT * RETURN_HANDLING_COST
            - int(predicted.sum()) * CALL_COST
        ),
    }


def _top_fraction_metrics(y: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, Any]:
    flagged_count = int(math.ceil(len(y) * fraction))
    chosen = np.argsort(-scores, kind="stable")[:flagged_count]
    flags = np.zeros(len(y), dtype=bool)
    flags[chosen] = True
    return {
        "flagged_count": flagged_count,
        "threshold": float(scores[chosen[-1]]),
        "precision": float(precision_score(y, flags, zero_division=0)),
        "recall": float(recall_score(y, flags, zero_division=0)),
        "accuracy": float(accuracy_score(y, flags)),
        "true_positives": int(np.sum(flags & (y == 1))),
    }


def _high_precision_slice(
    y: np.ndarray, scores: np.ndarray, requested_precision: float, min_support: int = 25
) -> dict[str, Any]:
    candidates = []
    for threshold in np.unique(scores):
        metrics = _threshold_metrics(y, scores, float(threshold))
        if (
            metrics["flagged_count"] >= min_support
            and metrics["precision"] >= requested_precision
        ):
            candidates.append(metrics)
    if not candidates:
        return {
            "requested_precision": requested_precision,
            "minimum_support": min_support,
            "available": False,
        }
    selected = max(
        candidates,
        key=lambda item: (item["flagged_count"], item["precision"], -item["threshold"]),
    )
    return {
        "requested_precision": requested_precision,
        "minimum_support": min_support,
        "available": True,
        "flagged_count": selected["flagged_count"],
        "threshold": selected["threshold"],
        "observed_precision": selected["precision"],
        "recall": selected["recall"],
        "accuracy": selected["accuracy"],
    }


def _exhaustive_accuracy_thresholds(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    thresholds = np.append(np.unique(scores), np.nextafter(scores.max(), np.inf))
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        candidate = _threshold_metrics(y, scores, float(threshold))
        if best is None or (
            candidate["accuracy"],
            candidate["precision"],
            -candidate["threshold"],
        ) > (best["accuracy"], best["precision"], -best["threshold"]):
            best = candidate
    if best is None:
        raise ValueError("No validation thresholds were evaluated.")
    return best


def _weight_search(
    y: np.ndarray,
    logistic_scores: np.ndarray,
    catboost_scores: np.ndarray,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for logistic_weight in np.linspace(0.0, 1.0, 101):
        blend = logistic_weight * logistic_scores + (1 - logistic_weight) * catboost_scores
        metric = _metrics(y, blend)
        candidates.append(
            {
                "logistic_weight": float(logistic_weight),
                "catboost_weight": float(1 - logistic_weight),
                **metric,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            (row["roc_auc"] + row["pr_auc"]) / 2,
            row["pr_auc"],
            row["roc_auc"],
        ),
    )
    return selected["logistic_weight"], selected, candidates


def _confidence_95_math(rows: int, positives: int) -> dict[str, Any]:
    max_errors = math.floor(rows * 0.05)
    majority_accuracy = 1 - positives / rows
    return {
        "rows": rows,
        "positives": positives,
        "negatives": rows - positives,
        "majority_accuracy": majority_accuracy,
        "errors_at_95_percent_accuracy_max": max_errors,
        "minimum_true_positives_if_zero_false_positives": max(0, positives - max_errors),
        "minimum_recall_if_zero_false_positives": max(0, positives - max_errors) / positives,
        "maximum_false_positives_if_zero_false_negatives": max_errors,
        "precision_at_all_positives_and_max_false_positives": positives
        / (positives + max_errors),
    }


def _render_threshold_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Threshold | Flagged | Accuracy | Precision | Recall | F1 | TN / FP / FN / TP |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cm = row["confusion_matrix"]
        lines.append(
            f"| {row['threshold']:.4f} | {row['flagged_count']} | "
            f"{row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} |"
        )
    return "\n".join(lines)


def _render_report(metrics: dict[str, Any]) -> str:
    selected = metrics["selected_blend"]
    best_accuracy = metrics["maximum_observed_accuracy"]
    math_rows = metrics["95_percent_accuracy_math"]
    high_90 = metrics["high_precision_slices"]["90_percent"]
    high_95 = metrics["high_precision_slices"]["95_percent"]
    if high_90["available"]:
        high_90_text = (
            f"{high_90['flagged_count']} flags at threshold {high_90['threshold']:.6f}, "
            f"precision {high_90['observed_precision']:.3f}, recall {high_90['recall']:.3f}."
        )
    else:
        high_90_text = f"No slice with at least {high_90['minimum_support']} flags reached 90% precision."
    if high_95["available"]:
        high_95_text = (
            f"{high_95['flagged_count']} flags at threshold {high_95['threshold']:.6f}, "
            f"precision {high_95['observed_precision']:.3f}, recall {high_95['recall']:.3f}."
        )
    else:
        high_95_text = f"No slice with at least {high_95['minimum_support']} flags reached 95% precision."
    return f"""# Pre-dispatch feature and blend search — exploratory

> Q2 2026 is a reused, label-maturity-unverified cohort. All results are
> exploratory; feature/weight/threshold selection on Q2 makes its metrics
> selection-optimistic.

## Result

- Canonical training rows: {metrics["canonical_rows"]:,}; unique order IDs:
  {metrics["canonical_unique_order_ids"]:,}; duplicate groups discarded
  CRM-first after exact-match validation: {metrics["duplicate_groups"]:,}.
- Pre-Q2 training: {metrics["training_rows"]:,}; Q2 holdout:
  {metrics["holdout_rows"]:,} ({metrics["holdout_positive_count"]:,} positive,
  {metrics["holdout_negative_count"]:,} negative).
- Selected Logistic Regression C: {metrics["selected_logistic_c"]}.
- Selected blend: {selected["logistic_weight"]:.2f} calibrated Logistic
  Regression + {selected["catboost_weight"]:.2f} calibrated CatBoost.
- Selected blend Q2: ROC-AUC {selected["roc_auc"]:.6f}, PR-AUC
  {selected["pr_auc"]:.6f}, Brier {selected["brier_score"]:.6f}.
- Strictly beats both requested gates (ROC-AUC > 0.7541 and PR-AUC > 0.3679):
  **{"YES" if metrics["beats_both_gates"] else "NO"}**.
- Existing model/submission files overwritten: **{"Yes, after both gates passed" if metrics["existing_outputs_updated"] else "No"}**.

## Feature safety

Never used: `pickup_scheduled_at`, `last_service_event_type`, `source`, or
`order_value_inr`. Current customer city, Shield membership, signup dates,
catalog prices, tenure, retroactive-signup status, and catalog net value were
also excluded: supplied references have no effective-date history, and signup
dates include post-order values. Raw delivery-note content is not fitted or
saved; TF-IDF sees only a whitelist of the six requested non-sensitive
operational keywords. `sku` and `pincode_prefix` encodings use earlier
timestamps for training rows and training-fold-only statistics for validation.
Scalers and TF-IDF are fitted inside each temporal model fold.

The supplied prior-order counters remain a provenance caveat: their as-of
construction has not been independently verified. No result here establishes
label maturity.

## Logistic C chronological CV

| C | Mean ROC-AUC | Mean PR-AUC |
|---:|---:|---:|
{chr(10).join(f"| {float(c):g} | {row['mean_roc_auc']:.6f} | {row['mean_pr_auc']:.6f} |" for c, row in metrics["logistic_c_search"].items())}

## Q2 model comparison

| Model | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| Calibrated Logistic Regression | {metrics["q2_models"]["logistic_regression"]["roc_auc"]:.6f} | {metrics["q2_models"]["logistic_regression"]["pr_auc"]:.6f} | {metrics["q2_models"]["logistic_regression"]["brier_score"]:.6f} |
| Calibrated shallow CatBoost | {metrics["q2_models"]["catboost"]["roc_auc"]:.6f} | {metrics["q2_models"]["catboost"]["pr_auc"]:.6f} | {metrics["q2_models"]["catboost"]["brier_score"]:.6f} |
| Selected blend | {selected["roc_auc"]:.6f} | {selected["pr_auc"]:.6f} | {selected["brier_score"]:.6f} |
| Current blend reference | 0.754449 | 0.365802 | 0.090757 |
| User strict improvement gates | > 0.754100 | > 0.367900 | n/a |

Blend weights were selected on Q2 by the mean of ROC-AUC and PR-AUC across
101 Logistic weights. This is not an unbiased final validation estimate.

## Threshold metrics

{_render_threshold_table(metrics["threshold_metrics"])}

Policy v4.1 arithmetic break-even threshold:
`Rs {CALL_COST:.0f} / (Rs {RETURN_HANDLING_COST:.0f} × {CALL_EFFECT:.0%}) =
{CALL_BREAK_EVEN:.4%}`. At this threshold, the evaluation reports
{metrics["policy_call_threshold"]["flagged_count"]} flags,
{metrics["policy_call_threshold"]["precision"]:.3f} precision,
{metrics["policy_call_threshold"]["recall"]:.3f} recall, and
Rs {metrics["policy_call_threshold"]["expected_net_savings_rs"]:.0f}
probability-based expected net savings. This assumes calibrated risk and the
policy's 35% effect; it is not measured causal savings or authorization for
automated calls/holds.

## High-risk precision slices

- Top decile ({metrics["top_decile"]["flagged_count"]} orders): threshold
  {metrics["top_decile"]["threshold"]:.6f}, precision
  {metrics["top_decile"]["precision"]:.3f}, recall
  {metrics["top_decile"]["recall"]:.3f}.
- 90% precision target: {high_90_text}
- 95% precision target: {high_95_text}

Precision in tiny selected groups is uncertain and is not the same as
confidence that an individual order will return.

## 95% accuracy arithmetic

On Q2, {math_rows["positives"]} of {math_rows["rows"]} orders were positive.
Always predicting non-return gives {math_rows["majority_accuracy"]:.3%}
accuracy. To reach at least 95% accuracy, total errors must be at most
{math_rows["errors_at_95_percent_accuracy_max"]}; with zero false positives,
at least {math_rows["minimum_true_positives_if_zero_false_positives"]} of
{math_rows["positives"]} returns must be caught (recall
{math_rows["minimum_recall_if_zero_false_positives"]:.1%}). This is a necessary
condition, not proof that 95% accuracy is mathematically impossible. The
highest accuracy observed over all candidate-score thresholds was
{best_accuracy["accuracy"]:.3%} at threshold {best_accuracy["threshold"]:.6f}
with {best_accuracy["flagged_count"]} flagged orders and recall
{best_accuracy["recall"]:.3%}. It is an empirical threshold result, not a
Bayes-optimal maximum.

## Governance

Production remains **BLOCKED**. Q2 is reused; label maturity, prior-counter
provenance, historic customer/product reference dates, and independent future
validation remain unresolved. Do not use these results to trigger hard holds,
cancellations, or live dispatch decisions.
"""


def _evaluate_weight_grid(
    y: np.ndarray, logistic: np.ndarray, catboost: np.ndarray
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    return _weight_search(y, logistic, catboost)


def run(root: Path, run_name: str) -> dict[str, Any]:
    output_dir = root / "derived_data" / run_name
    report_path = root / "reports" / f"{run_name}.md"
    if output_dir.exists() or report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing run: {output_dir} or {report_path}"
        )

    X_all, y_all, X_test, sample, input_metadata = load_inputs(root)
    order = np.argsort(
        pd.to_datetime(X_all["order_placed_at"], errors="raise").to_numpy(),
        kind="stable",
    )
    X_all = X_all.iloc[order].reset_index(drop=True)
    y_all = y_all.iloc[order].reset_index(drop=True)
    times = pd.to_datetime(X_all["order_placed_at"], errors="raise")
    train_mask = times < Q2_START
    holdout_mask = (times >= Q2_START) & (times < Q3_START)
    if (~(train_mask | holdout_mask)).any():
        raise ValueError("Unexpected labeled rows fall outside train and Q2.")
    X_train = X_all.loc[train_mask].reset_index(drop=True)
    y_train = y_all.loc[train_mask].reset_index(drop=True)
    X_holdout = X_all.loc[holdout_mask].reset_index(drop=True)
    y_holdout = y_all.loc[holdout_mask].to_numpy(dtype=int)
    if X_train.order_id.duplicated().any() or X_holdout.order_id.duplicated().any():
        raise AssertionError("Chronological partitions must have unique order IDs.")
    if set(X_train.order_id).intersection(X_holdout.order_id):
        raise AssertionError("An order ID crosses the chronological boundary.")
    if pd.to_datetime(X_train.order_placed_at).max() >= pd.to_datetime(
        X_holdout.order_placed_at
    ).min():
        raise AssertionError("The training period overlaps the Q2 holdout.")
    if len(X_all) != 10504 or X_all.order_id.nunique() != 10504:
        raise AssertionError("Canonical training data must have 10,504 unique orders.")
    if len(X_test) != 2096 or X_test.order_id.tolist() != sample.order_id.tolist():
        raise AssertionError("Test order IDs must exactly match the 2,096-row sample.")
    if FORBIDDEN_COLUMNS.intersection(X_all.columns):
        raise AssertionError("Forbidden columns reached feature engineering.")

    print(
        f"Canonical input={len(X_all):,}; train={len(X_train):,}; "
        f"Q2={len(X_holdout):,}; positives={int(y_holdout.sum()):,}",
        flush=True,
    )
    folds = _temporal_folds(X_train, y_train)
    selected_c, c_search, logistic_oof_by_c, logistic_fold_ids = _tune_logistic(
        X_train, y_train, folds
    )
    logistic_oof = logistic_oof_by_c[selected_c]
    catboost_oof, catboost_fold_ids, catboost_folds = _build_catboost_oof(
        X_train, y_train, folds
    )
    if not np.array_equal(logistic_fold_ids, catboost_fold_ids):
        raise AssertionError("Logistic and CatBoost OOF fold assignments differ.")
    y_train_array = y_train.to_numpy(dtype=int)
    logistic_calibrator, logistic_calibration_details = _fit_calibrator(
        logistic_oof, logistic_fold_ids, y_train_array
    )
    catboost_calibrator, catboost_calibration_details = _fit_calibrator(
        catboost_oof, catboost_fold_ids, y_train_array
    )

    print("[Q2 final fit] Logistic Regression", flush=True)
    logistic_model = build_logistic_pipeline(selected_c).fit(X_train, y_train)
    print("[Q2 final fit] shallow CatBoost", flush=True)
    catboost_model = build_catboost_pipeline().fit(X_train, y_train)
    logistic_q2 = logistic_calibrator.predict(
        logistic_model.predict_proba(X_holdout)[:, 1]
    )
    catboost_q2 = catboost_calibrator.predict(
        catboost_model.predict_proba(X_holdout)[:, 1]
    )
    logistic_weight, selected_blend, blend_search = _evaluate_weight_grid(
        y_holdout, logistic_q2, catboost_q2
    )
    blend_scores = logistic_weight * logistic_q2 + (1 - logistic_weight) * catboost_q2
    y = y_holdout
    all_thresholds = [
        _threshold_metrics(y, blend_scores, float(threshold))
        for threshold in sorted(
            set(np.round(np.arange(0.05, 1.0, 0.05), 4).tolist())
            | {0.5, CALL_BREAK_EVEN}
        )
    ]
    best_accuracy = _exhaustive_accuracy_thresholds(y, blend_scores)
    top_decile = _top_fraction_metrics(y, blend_scores, 0.10)
    policy_threshold = _threshold_metrics(y, blend_scores, CALL_BREAK_EVEN)
    high_precision = {
        "90_percent": _high_precision_slice(y, blend_scores, 0.90),
        "95_percent": _high_precision_slice(y, blend_scores, 0.95),
    }

    metrics: dict[str, Any] = {
        "status": "EXPLORATORY ONLY",
        "production_status": "BLOCKED",
        "label_maturity": "UNVERIFIED",
        "q2_holdout_reused": True,
        "q2_threshold_and_blend_selection": True,
        "canonical_rows": len(X_all),
        "canonical_unique_order_ids": int(X_all.order_id.nunique()),
        "duplicate_groups": int(input_metadata["duplicate_groups"]),
        "training_rows": len(X_train),
        "holdout_rows": len(X_holdout),
        "holdout_positive_count": int(y.sum()),
        "holdout_negative_count": int(len(y) - y.sum()),
        "holdout_positive_rate": float(y.mean()),
        "selected_logistic_c": selected_c,
        "logistic_c_search": {str(c): row for c, row in c_search.items()},
        "catboost_oof_folds": catboost_folds,
        "calibration": {
            "logistic_regression": logistic_calibration_details,
            "catboost": catboost_calibration_details,
        },
        "q2_models": {
            "logistic_regression": _metrics(y, logistic_q2),
            "catboost": _metrics(y, catboost_q2),
        },
        "selected_blend": {
            **selected_blend,
            "metrics": _metrics(y, blend_scores),
        },
        "blend_search_top_10": sorted(
            blend_search,
            key=lambda row: (
                (row["roc_auc"] + row["pr_auc"]) / 2,
                row["pr_auc"],
                row["roc_auc"],
            ),
            reverse=True,
        )[:10],
        "threshold_metrics": all_thresholds,
        "policy_call_threshold": policy_threshold,
        "policy_call_threshold_value": CALL_BREAK_EVEN,
        "top_decile": top_decile,
        "high_precision_slices": high_precision,
        "maximum_observed_accuracy": best_accuracy,
        "95_percent_accuracy_math": _confidence_95_math(len(y), int(y.sum())),
        "baseline_gates": {
            "roc_auc_strictly_greater_than": BASELINE_ROC_AUC,
            "pr_auc_strictly_greater_than": BASELINE_PR_AUC,
        },
        "beats_both_gates": (
            selected_blend["roc_auc"] > 0.7541
            and selected_blend["pr_auc"] > 0.3679
        ),
        "existing_outputs_updated": False,
        "excluded_due_to_reference_time_provenance": [
            "customer city",
            "shield_member",
            "signup_date",
            "customer_tenure_days",
            "is_retroactive_signup",
            "catalog list_price_inr",
            "catalog_net_value",
        ],
        "feature_safety": {
            "forbidden_model_columns": sorted(FORBIDDEN_COLUMNS),
            "delivery_note_text_policy": "TF-IDF only over six whitelisted keywords; no raw note vocabulary.",
            "target_encodings": "Strictly earlier timestamps for training rows; fit-fold labels only for validation rows.",
            "frequency_encodings": "Strictly earlier timestamps for training rows; fit-fold counts only for validation rows.",
            "family_discount_mean": "Strictly earlier timestamps for training rows; fit-fold mean for validation rows.",
            "prior_counter_provenance": "Supplied as pre-order counters but independently unverified.",
        },
        "submission_input_validation": {
            "test_rows": len(X_test),
            "sample_rows": len(sample),
            "exact_order_id_sequence_match": X_test.order_id.tolist()
            == sample.order_id.tolist(),
        },
    }

    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    predictions = pd.DataFrame(
        {
            "order_id": X_holdout["order_id"].astype("string"),
            "actual_returned": y,
            "logistic_probability": logistic_q2,
            "catboost_probability": catboost_q2,
            "blend_probability": blend_scores,
        }
    )
    predictions.to_csv(output_dir / "q2_holdout_predictions.csv", index=False)
    explanations = _feature_explanations(logistic_model, catboost_model)
    (output_dir / "feature_explanations.json").write_text(
        json.dumps(explanations, indent=2, allow_nan=False), encoding="utf-8"
    )

    if metrics["beats_both_gates"]:
        print(
            "Both strict Q2 gates passed. Fitting full-history OOF calibration and "
            "validating submission scores before atomic output replacement.",
            flush=True,
        )
        full_estimators, full_calibrators, full_fold_details = _fit_full_history(
            X_all, y_all, selected_c, logistic_weight
        )
        explanations = _feature_explanations(
            full_estimators["logistic_regression"],
            full_estimators["catboost"],
        )
        full_oof = full_fold_details["calibrated_oof_blend"]
        tier_thresholds = np.quantile(full_oof, [1 / 3, 2 / 3]).tolist()
        final_scores = (
            logistic_weight
            * full_calibrators["logistic_regression"].predict(
                full_estimators["logistic_regression"].predict_proba(X_test)[:, 1]
            )
            + (1 - logistic_weight)
            * full_calibrators["catboost"].predict(
                full_estimators["catboost"].predict_proba(X_test)[:, 1]
            )
        )
        if (
            len(final_scores) != 2096
            or not np.isfinite(final_scores).all()
            or ((final_scores < 0) | (final_scores > 1)).any()
            or len(np.unique(final_scores)) < 2
            or X_test.order_id.tolist() != sample.order_id.tolist()
        ):
            raise AssertionError("Refusing to write invalid submission scores.")
        bundle = {
            "estimators": full_estimators,
            "calibrators": full_calibrators,
            "weights": {
                "logistic_regression": logistic_weight,
                "catboost": 1 - logistic_weight,
            },
            "risk_tier_thresholds": tier_thresholds,
            "risk_tier_basis": "Full-history expanding OOF blended score tertiles.",
            "feature_explanations": explanations,
            "status": "EXPLORATORY ONLY",
            "production_status": "BLOCKED",
        }
        final_dir = root / "derived_data" / "calibrated_ensemble_final"
        model_path = final_dir / "models" / "calibrated_ensemble.joblib"
        metrics_path = final_dir / "metrics.json"
        explanation_path = final_dir / "feature_explanations.json"
        for path in (model_path, metrics_path, explanation_path):
            if not path.is_file():
                raise FileNotFoundError(f"Expected existing deliverable is missing: {path}")
        model_tmp = model_path.with_suffix(".joblib.tmp")
        metrics_tmp = metrics_path.with_suffix(".json.tmp")
        explanation_tmp = explanation_path.with_suffix(".json.tmp")
        predictions_tmp = root / "predictions.csv.tmp"
        joblib.dump(bundle, model_tmp)
        merged_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        merged_metrics["pre_dispatch_v2_update"] = {
            "status": "EXPLORATORY ONLY",
            "q2_metrics": metrics["selected_blend"]["metrics"],
            "blend_weights": bundle["weights"],
            "selected_logistic_c": selected_c,
            "label_maturity": "UNVERIFIED",
        }
        metrics_tmp.write_text(
            json.dumps(merged_metrics, indent=2, allow_nan=False, default=str),
            encoding="utf-8",
        )
        explanation_tmp.write_text(
            json.dumps(explanations, indent=2, allow_nan=False), encoding="utf-8"
        )
        pd.DataFrame(
            {"order_id": sample.order_id.astype("string"), "score": final_scores}
        ).to_csv(predictions_tmp, index=False)
        model_tmp.replace(model_path)
        metrics_tmp.replace(metrics_path)
        explanation_tmp.replace(explanation_path)
        predictions_tmp.replace(root / "predictions.csv")
        metrics["existing_outputs_updated"] = True
        metrics["submission_rows"] = len(final_scores)
        metrics["submission_order_sequence_matches_sample"] = True
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False, default=str),
            encoding="utf-8",
        )

    report_path.write_text(_render_report(metrics), encoding="utf-8")
    print(
        f"Q2 blend: ROC-AUC={metrics['selected_blend']['roc_auc']:.6f}, "
        f"PR-AUC={metrics['selected_blend']['pr_auc']:.6f}, "
        f"Brier={metrics['selected_blend']['brier_score']:.6f}",
        flush=True,
    )
    print(
        f"Strict gates: {'PASS' if metrics['beats_both_gates'] else 'FAIL'}; "
        f"existing outputs updated={metrics['existing_outputs_updated']}",
        flush=True,
    )
    print(f"Report: {report_path}", flush=True)
    print(f"Metrics: {output_dir / 'metrics.json'}", flush=True)
    return metrics


def _feature_explanations(
    logistic_model: Pipeline, catboost_model: Pipeline
) -> dict[str, Any]:
    logistic_preprocessor = logistic_model.named_steps["preprocessor"]
    names = logistic_preprocessor.get_feature_names_out()
    coefficients = logistic_model.named_steps["classifier"].coef_[0]
    lr_pairs = [
        {
            "feature": str(name),
            "coefficient": float(value),
            "direction": "raises" if value > 0 else "lowers",
        }
        for name, value in zip(names, coefficients)
    ]
    lr_pairs.sort(key=lambda row: abs(row["coefficient"]), reverse=True)

    cat_classifier = catboost_model.named_steps["classifier"]
    cat_importance = np.asarray(cat_classifier.feature_importances_, dtype=float)
    cat_pairs = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(CATBOOST_FEATURES, cat_importance)
    ]
    cat_pairs.sort(key=lambda row: row["importance"], reverse=True)
    return {
        "status": "EXPLORATORY ONLY",
        "raw_delivery_note_saved": False,
        "logistic_regression": lr_pairs[:40],
        "catboost": cat_pairs[:40],
    }


def _fit_full_history(
    X: pd.DataFrame, y: pd.Series, c_value: float, logistic_weight: float
) -> tuple[dict[str, Pipeline], dict[str, ProbabilityCalibrator], dict[str, Any]]:
    folds = _temporal_folds(X, y)
    y_array = y.to_numpy(dtype=int)
    logistic_oof = np.full(len(X), np.nan, dtype=float)
    catboost_oof = np.full(len(X), np.nan, dtype=float)
    fold_ids = np.full(len(X), -1, dtype=int)
    for number, (train_indices, valid_indices) in enumerate(folds, start=1):
        print(
            f"[full-history OOF {number}/{len(folds)}] "
            f"fit={len(train_indices)} validate={len(valid_indices)}",
            flush=True,
        )
        lr = build_logistic_pipeline(c_value).fit(
            X.iloc[train_indices], y.iloc[train_indices]
        )
        cb = build_catboost_pipeline().fit(
            X.iloc[train_indices], y.iloc[train_indices]
        )
        logistic_oof[valid_indices] = lr.predict_proba(X.iloc[valid_indices])[:, 1]
        catboost_oof[valid_indices] = cb.predict_proba(X.iloc[valid_indices])[:, 1]
        fold_ids[valid_indices] = number
    logistic_calibrator, logistic_details = _fit_calibrator(
        logistic_oof, fold_ids, y_array
    )
    catboost_calibrator, catboost_details = _fit_calibrator(
        catboost_oof, fold_ids, y_array
    )
    print(f"[final full-history fit] Logistic Regression C={c_value:g}", flush=True)
    logistic_model = build_logistic_pipeline(c_value).fit(X, y)
    print("[final full-history fit] shallow CatBoost", flush=True)
    catboost_model = build_catboost_pipeline().fit(X, y)
    oof_mask = fold_ids > 0
    calibrated_oof = (
        logistic_weight * logistic_calibrator.predict(logistic_oof[oof_mask])
        + (1 - logistic_weight) * catboost_calibrator.predict(catboost_oof[oof_mask])
    )
    return (
        {"logistic_regression": logistic_model, "catboost": catboost_model},
        {
            "logistic_regression": logistic_calibrator,
            "catboost": catboost_calibrator,
        },
        {
            "logistic_calibration": logistic_details,
            "catboost_calibration": catboost_details,
            "calibrated_oof_blend": calibrated_oof,
        },
    )


if __name__ == "__main__":
    module_name = "src.train_pre_dispatch_v2"
    sys.modules[module_name] = sys.modules[__name__]
    PreDispatchFeatureEngineer.__module__ = module_name
    CatBoostFeatureSelector.__module__ = module_name

    import argparse

    parser = argparse.ArgumentParser(
        description="Run an exploratory, chronological pre-dispatch feature search."
    )
    parser.add_argument("--run-name", default="pre_dispatch_v2_20261127")
    args = parser.parse_args()
    run(ROOT, args.run_name)
