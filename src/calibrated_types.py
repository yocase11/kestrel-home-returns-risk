from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


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


class FoldFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Any = None) -> "FoldFeatureEngineer":
        required = {
            "order_placed_at", "launch_date", "sales_channel", "payment_mode",
            "is_gift", "sku", "product_family", "product_model_name",
            "delivery_pincode", "discount_pct", "qty",
            "promised_delivery_days", "customer_prior_orders",
            "customer_prior_returns", "delivery_note",
        }
        missing = sorted(required.difference(X.columns))
        if missing:
            raise ValueError(f"Feature source is missing columns: {missing}")
        discounts = pd.to_numeric(X["discount_pct"], errors="raise")
        families = X["product_family"].astype("string")
        if families.isna().any():
            raise ValueError("product_family must be populated before feature fitting.")
        self.family_discount_means_ = discounts.groupby(families).mean().to_dict()
        self.global_discount_mean_ = float(discounts.mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=X.index)
        for column in CATEGORICAL_FEATURES:
            if column == "pincode_prefix":
                continue
            values = X[column].astype("string")
            if column == "is_gift":
                values = values.replace({"1": "Y", "0": "N"})
            if values.isna().any():
                raise ValueError(f"{column} contains missing categorical values.")
            output[column] = values.astype(str)

        numeric = {
            column: pd.to_numeric(X[column], errors="raise")
            for column in (
                "discount_pct", "qty", "promised_delivery_days",
                "customer_prior_orders", "customer_prior_returns",
            )
        }
        prior_orders = numeric["customer_prior_orders"]
        prior_returns = numeric["customer_prior_returns"]
        if (prior_orders < 0).any() or (prior_returns < 0).any():
            raise ValueError("Customer prior counts must be non-negative.")
        if (prior_returns > prior_orders).any():
            raise ValueError("Customer prior returns cannot exceed prior orders.")

        pincode = X["delivery_pincode"].astype("string").str.strip().str.zfill(6)
        if not pincode.str.fullmatch(r"\d{6}").all():
            raise ValueError("delivery_pincode must contain six digits after padding.")
        pincode_int = pincode.astype("int64")
        output["pincode_prefix"] = (pincode_int // 10000).astype(str)

        order_time = pd.to_datetime(X["order_placed_at"], errors="raise")
        launch_date = pd.to_datetime(X["launch_date"], errors="raise")
        if order_time.isna().any() or launch_date.isna().any():
            raise ValueError("Order and product launch timestamps must be present.")
        days_since_launch = (order_time - launch_date).dt.days
        if (days_since_launch < 0).any():
            raise ValueError("An order precedes its product launch date.")

        notes = X["delivery_note"].fillna("").astype(str).str.strip()
        lower_notes = notes.str.lower()
        output["has_note"] = notes.ne("").astype("int8")
        output["note_length"] = notes.str.len().astype("int64")
        for keyword in NOTE_KEYWORDS:
            output[f"note_has_{keyword.replace(' ', '_')}"] = (
                lower_notes.str.contains(keyword, regex=False).astype("int8")
            )

        output["discount_pct"] = numeric["discount_pct"].astype(float)
        output["qty"] = numeric["qty"].astype(float)
        output["promised_delivery_days"] = numeric["promised_delivery_days"].astype(float)
        output["customer_prior_orders"] = prior_orders.astype(float)
        output["smoothed_prior_return_rate"] = (
            (prior_returns + 1.14) / (prior_orders + 10.0)
        ).astype(float)
        output["is_first_time_buyer"] = prior_orders.eq(0).astype("int8")
        output["is_zero_pincode"] = pincode_int.eq(0).astype("int8")
        output["order_hour"] = order_time.dt.hour.astype("int8")
        output["day_of_week"] = order_time.dt.dayofweek.astype("int8")
        output["is_weekend"] = order_time.dt.dayofweek.ge(5).astype("int8")
        output["is_post_oct2025"] = order_time.ge(pd.Timestamp("2025-10-01")).astype("int8")
        output["days_since_launch"] = days_since_launch.astype("int64")
        family_means = X["product_family"].astype("string").map(self.family_discount_means_)
        output["discount_diff_from_family"] = (
            numeric["discount_pct"] - family_means.fillna(self.global_discount_mean_)
        ).astype(float)
        return output.loc[:, list(FEATURE_NAMES)].reset_index(drop=True)

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(FEATURE_NAMES, dtype=object)


class ProbabilityCalibrator:
    def __init__(self, method: str) -> None:
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("Calibration method must be sigmoid or isotonic.")
        self.method = method

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        raw = np.asarray(probabilities, dtype=float)
        target = np.asarray(y, dtype=int)
        if raw.ndim != 1 or len(raw) != len(target) or len(raw) == 0:
            raise ValueError("Calibration scores and labels must be aligned 1D arrays.")
        if len(np.unique(target)) != 2:
            raise ValueError("Calibration fitting requires both target classes.")
        if self.method == "sigmoid":
            clipped = np.clip(raw, 1e-6, 1 - 1e-6)
            logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            self.estimator_ = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
            self.estimator_.fit(logit, target)
        else:
            self.estimator_ = IsotonicRegression(out_of_bounds="clip")
            self.estimator_.fit(raw, target)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        raw = np.asarray(probabilities, dtype=float)
        if self.method == "sigmoid":
            clipped = np.clip(raw, 1e-6, 1 - 1e-6)
            logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            return self.estimator_.predict_proba(logit)[:, 1]
        return np.asarray(self.estimator_.predict(raw), dtype=float)
