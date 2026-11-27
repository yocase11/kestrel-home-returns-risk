"""Exploratory order scoring and service-status helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


MODEL_RELATIVE_PATH = Path(
    "derived_data/calibrated_ensemble_final/models/calibrated_ensemble.joblib"
)

_FRIENDLY_FEATURES = {
    "sales_channel": "Sales channel",
    "payment_mode": "Payment method",
    "is_gift": "Gift order",
    "sku": "Product",
    "product_family": "Product family",
    "product_model_name": "Product model",
    "discount_pct": "Discount percent",
    "qty": "Quantity",
    "promised_delivery_days": "Promised delivery days",
    "customer_prior_orders": "Number of earlier orders",
    "smoothed_prior_return_rate": "Earlier return-history signal",
    "is_first_time_buyer": "First-order indicator",
    "is_zero_pincode": "Address-capture status",
    "has_note": "Delivery note",
    "note_length": "Delivery note length",
    "order_hour": "Order hour",
    "day_of_week": "Order weekday",
    "is_weekend": "Weekend order",
    "is_post_oct2025": "Post-October-2025 migration period",
    "days_since_launch": "Days since product launch",
    "discount_diff_from_family": "Discount relative to product-family average",
}


class InvalidOrderError(ValueError):
    """Raised when a valid order record cannot be scored."""


class UnknownSkuError(InvalidOrderError):
    """Raised when a SKU has no matching product reference record."""


class ProductReferenceError(RuntimeError):
    """Raised when the serving-time product lookup is invalid."""


def find_saved_model(root: Path) -> list[Path]:
    """Find production model artifacts, if any."""
    models_dir = root / "models"
    if not models_dir.exists():
        return []
    return sorted(
        path
        for path in models_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".joblib", ".pkl", ".pickle", ".cbm"}
    )


def get_prediction_status(root: Path) -> dict[str, Any]:
    """Report exploratory scoring separately from production approval."""
    model_path = root / MODEL_RELATIVE_PATH
    if model_path.is_file():
        return {
            "status": "exploratory_available",
            "model_status": "exploratory",
            "production_status": "blocked",
            "reason": (
                "The calibrated Logistic Regression/CatBoost blend is available for "
                "local exploratory scoring only. "
                "Label maturity and independent validation remain unresolved; "
                "do not use scores for live dispatch decisions."
            ),
        }

    artifacts = find_saved_model(root)
    if artifacts:
        return {
            "status": "model_artifact_found_but_not_admitted",
            "model_status": "blocked",
            "production_status": "blocked",
            "reason": (
                "A model artifact exists, but this repository has not admitted it "
                "as a validated, label-mature returns-risk model."
            ),
        }
    return {
        "status": "unavailable",
        "model_status": "blocked",
        "production_status": "blocked",
        "reason": (
            "No saved exploratory model is available. Label maturity, the actual "
            "extraction cutoff, and delivery-date evidence remain unresolved."
        ),
    }


@lru_cache(maxsize=2)
def _load_model(model_path: str) -> Any:
    return joblib.load(model_path)


@lru_cache(maxsize=2)
def _load_product_reference(products_path: str) -> pd.DataFrame:
    products = pd.read_csv(products_path, dtype={"sku": "string"})
    required = {"sku", "family", "model_name", "launch_date"}
    missing = required.difference(products.columns)
    if missing:
        raise ProductReferenceError(
            f"Product reference is missing columns: {sorted(missing)}"
        )
    if products["sku"].isna().any() or products["sku"].duplicated().any():
        raise ProductReferenceError(
            "Product reference SKU keys must be present and unique."
        )
    return products.set_index("sku")


def _product_reference(root: Path) -> pd.DataFrame:
    prepared_reference = root / "derived_data" / "product_reference.csv"
    if prepared_reference.is_file():
        return _load_product_reference(str(prepared_reference.resolve()))

    matches = sorted((root / "datasets").glob("*products.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected derived_data/product_reference.csv or exactly one "
            "products.csv reference file in datasets/."
        )
    return _load_product_reference(str(matches[0].resolve()))


def _build_feature_row(
    order: Mapping[str, Any], products: pd.DataFrame
) -> pd.DataFrame:
    order_time = pd.Timestamp(order["order_placed_at"])
    if order_time.tzinfo is not None:
        raise InvalidOrderError("order_placed_at must be timezone-naive IST.")

    sku = str(order["sku"])
    if sku not in products.index:
        raise UnknownSkuError(f"SKU {sku!r} is not present in the product reference.")
    product = products.loc[sku]
    launch_date = pd.Timestamp(product["launch_date"])
    if launch_date.tzinfo is not None:
        raise ProductReferenceError("Product launch_date must be timezone-naive.")
    product_age_days = int((order_time - launch_date).total_seconds() // 86400)
    if product_age_days < 0:
        raise InvalidOrderError("Order date precedes the product launch date.")

    pincode = str(order["delivery_pincode"]).strip().zfill(6)
    if not pincode.isdigit() or len(pincode) != 6:
        raise InvalidOrderError("delivery_pincode must be 000000 or six digits.")
    prior_orders = int(order["customer_prior_orders"])
    prior_returns = int(order["customer_prior_returns"])
    if prior_orders < 0 or prior_returns < 0 or prior_returns > prior_orders:
        raise InvalidOrderError(
            "Customer prior counts must be non-negative and returns cannot exceed orders."
        )
    raw_gift = order["is_gift"]
    is_gift = "Y" if raw_gift in (1, "1", "Y") else "N"
    row: dict[str, Any] = {
        "order_placed_at": order_time,
        "launch_date": launch_date,
        "sales_channel": order["sales_channel"],
        "payment_mode": order["payment_mode"],
        "discount_pct": order["discount_pct"],
        "qty": order["qty"],
        "promised_delivery_days": order["promised_delivery_days"],
        "delivery_pincode": pincode,
        "is_gift": is_gift,
        "customer_prior_orders": prior_orders,
        "customer_prior_returns": prior_returns,
        "delivery_note": order.get("delivery_note") or "",
        "sku": sku,
        "product_family": product["family"],
        "product_model_name": product["model_name"],
    }
    if any(pd.isna(row[column]) for column in row):
        raise InvalidOrderError("Order input or product reference contains missing values.")
    if product_age_days < 0:
        raise InvalidOrderError("Order date precedes the product launch date.")
    return pd.DataFrame([row])


def _readable_reasons(
    estimator: Any, feature_row: pd.DataFrame
) -> list[str]:
    """Translate the local Logistic Regression contributions to plain English."""
    features = estimator.named_steps["feature_engineer"].transform(feature_row)
    transformed = estimator.named_steps["preprocessor"].transform(features)
    classifier = estimator.named_steps["classifier"]
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    contributions = np.asarray(dense)[0] * classifier.coef_[0]
    names = estimator.named_steps["preprocessor"].get_feature_names_out()

    indexes = np.argsort(np.abs(contributions))[::-1]
    chosen = [int(index) for index in indexes[:4] if contributions[index] != 0]
    if not chosen:
        return ["No single input signal stood out in the Logistic Regression component."]

    reasons = []
    for index in chosen:
        name = str(names[index])
        direction = "raises" if contributions[index] > 0 else "lowers"
        detail = "Order characteristics"
        if name.startswith("categorical__"):
            category_feature = name[len("categorical__") :]
            for column, friendly in _FRIENDLY_FEATURES.items():
                prefix = f"{column}_"
                if category_feature.startswith(prefix):
                    category_value = category_feature[len(prefix) :].replace("_", " ")
                    if column == "is_gift":
                        category_value = "Yes" if category_value == "Y" else "No"
                    detail = f"{friendly}: {category_value}"
                    break
        elif name.startswith("numeric__"):
            feature = name[len("numeric__") :]
            if feature.startswith("note_has_"):
                keyword = feature.removeprefix("note_has_").replace("_", " ")
                detail = f"Delivery note includes the word “{keyword}”"
            else:
                detail = _FRIENDLY_FEATURES.get(feature, "Order characteristics")
                if feature in {
                    "is_first_time_buyer",
                    "is_zero_pincode",
                    "has_note",
                    "is_weekend",
                    "is_post_oct2025",
                }:
                    value = features.iloc[0][feature]
                    detail = f"{detail} is {'present' if value else 'absent'}"
        reasons.append(
            f"{detail} {direction} the score in the Logistic Regression component."
        )
    return reasons


def predict_order(root: Path, order: Mapping[str, Any]) -> dict[str, Any]:
    """Score one order with the calibrated, exploratory LR/CatBoost blend."""
    status = get_prediction_status(root)
    if status["model_status"] != "exploratory":
        raise FileNotFoundError(status["reason"])

    model_path = root / MODEL_RELATIVE_PATH
    bundle = _load_model(str(model_path.resolve()))
    feature_row = _build_feature_row(order, _product_reference(root))
    weights = bundle["weights"]
    components: dict[str, float] = {}
    for model_name in weights:
        raw_probability = bundle["estimators"][model_name].predict_proba(
            feature_row
        )[:, 1]
        components[model_name] = float(
            bundle["calibrators"][model_name].predict(raw_probability)[0]
        )
    score = float(sum(weights[name] * components[name] for name in weights))
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ProductReferenceError("The calibrated model returned an invalid score.")

    tier_thresholds = bundle.get("risk_tier_thresholds")
    if (
        not isinstance(tier_thresholds, (list, tuple))
        or len(tier_thresholds) != 2
        or not np.isfinite(tier_thresholds).all()
        or tier_thresholds[0] > tier_thresholds[1]
    ):
        raise ProductReferenceError("Model risk-tier metadata is invalid.")
    if score < tier_thresholds[0]:
        risk_tier = "Low"
    elif score < tier_thresholds[1]:
        risk_tier = "Medium"
    else:
        risk_tier = "High"

    shield_member = bool(order.get("shield_member_verified_at_dispatch", False))
    call_break_even = 45.0 / (1150.0 * 0.35)
    if shield_member:
        action = (
            "Pilot-only: protect the verified Shield member with a soft call only; "
            "never place an automated hold."
        )
    elif score >= call_break_even:
        action = (
            "Pilot-only: consider a Rs 45 pre-dispatch confirmation call; do not "
            "hold or cancel dispatch."
        )
    else:
        action = "Proceed with the normal dispatch workflow; do not hold or cancel."

    logistic_estimator = bundle["estimators"]["logistic_regression"]
    return {
        "status": "exploratory",
        "model_status": "exploratory",
        "production_status": "blocked",
        "score": score,
        "risk_tier": risk_tier,
        "recommended_action": action,
        "risk_score": score,
        "score_interpretation": (
            "Calibrated blend score from 0 to 1; calibration and ranking are "
            "exploratory on reused, maturity-unverified labels and are not a "
            "validated live-order risk estimate."
        ),
        "reasons": _readable_reasons(logistic_estimator, feature_row),
        "reasoning_note": (
            "Reasons describe associations in the Logistic Regression component, "
            "not causes or intervention effects. Call suggestions are pilot-only; "
            "no automated hold or cancellation is approved."
        ),
    }
