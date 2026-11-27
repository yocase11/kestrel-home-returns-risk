"""FastAPI service for exploratory Kestrel returns-risk scoring."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.prediction_service import (
    InvalidOrderError,
    ProductReferenceError,
    get_prediction_status,
    predict_order,
)


ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(
    title="Kestrel Home returns-risk review API",
    version="1.2.0-exploratory",
    description=(
        "Local exploratory calibrated Logistic Regression/CatBoost scoring. "
        "The reused holdout and unverified labels do not establish production "
        "calibration. Action suggestions are pilot-only; production use remains blocked."
    ),
    openapi_tags=[
        {
            "name": "status",
            "description": "Service and model-gate status.",
        },
        {
            "name": "scoring",
            "description": "Single-order exploratory scoring using dispatch-time fields.",
        },
    ],
)


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_placed_at: datetime = Field(
        description="Timezone-naive local order timestamp, interpreted as IST."
    )
    sku: str = Field(
        min_length=1,
        max_length=100,
        description="SKU present in the packaged product reference.",
    )
    sales_channel: Literal["app", "web", "marketplace", "partner_outlet"] = Field(
        description="Sales channel recorded at order time."
    )
    payment_mode: Literal["prepaid_upi", "prepaid_card", "cod", "emi"] = Field(
        description="Payment mode recorded at order time."
    )
    discount_pct: int = Field(ge=0, le=100, description="Discount percentage.")
    qty: int = Field(ge=1, le=100, description="Order quantity.")
    promised_delivery_days: int = Field(
        ge=1,
        le=365,
        description="Promised delivery duration in days.",
    )
    delivery_pincode: str = Field(
        description="Six-digit pincode, or 000000 when missing."
    )
    is_gift: Literal[0, 1, "Y", "N"] = Field(
        description="Gift indicator as 0/1 or Y/N."
    )
    customer_prior_orders: int = Field(
        ge=0,
        le=10000,
        description="Pre-order count known at dispatch; do not use a current/future count.",
    )
    customer_prior_returns: int = Field(
        ge=0,
        le=10000,
        description="Pre-order return count known at dispatch.",
    )
    delivery_note: str = Field(
        default="",
        max_length=500,
        description="Checkout/outlet delivery note; do not include names or phone numbers.",
    )
    shield_member_verified_at_dispatch: bool = Field(
        default=False,
        description="Policy-only input; true only when membership is verified as-of dispatch.",
    )

    @field_validator("order_placed_at")
    @classmethod
    def validate_order_time(cls, value: datetime) -> datetime:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("order_placed_at must be timezone-naive IST.")
        return value

    @field_validator("delivery_pincode")
    @classmethod
    def validate_pincode(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned == "000000":
            return cleaned
        if not cleaned.isdigit() or len(cleaned) != 6:
            raise ValueError("delivery_pincode must be 000000 or a six-digit pincode.")
        return cleaned

    @field_validator("sku")
    @classmethod
    def strip_identifier(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Identifiers must not be blank.")
        return cleaned


class PredictionResponse(BaseModel):
    status: Literal["exploratory"] = Field(
        description="Confirms this is an exploratory response."
    )
    model_status: Literal["exploratory"]
    production_status: Literal["blocked"]
    score: float = Field(
        ge=0,
        le=1,
        description="Exploratory calibrated blend score; not approved for live decisions.",
    )
    risk_tier: Literal["Low", "Medium", "High"]
    recommended_action: str = Field(
        description="Pilot-only policy suggestion; never an automated hold/cancellation.",
    )
    risk_score: float = Field(
        ge=0,
        le=1,
        description="Backward-compatible alias of score.",
    )
    score_interpretation: str = Field(
        description="Limits on interpreting the score."
    )
    reasons: list[str] = Field(
        description="Largest local linear feature contributions, not causal reasons."
    )
    reasoning_note: str = Field(
        description="Caveat on model associations and operational use."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_status: Literal["exploratory", "blocked"]
    production_status: Literal["blocked"]
    prediction_status: str
    reason: str


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["status"],
    summary="Report service and model-gate status",
)
def health() -> HealthResponse:
    status = get_prediction_status(ROOT)
    return HealthResponse(
        status="ok",
        model_status=status["model_status"],
        production_status=status["production_status"],
        prediction_status=status["status"],
        reason=status["reason"],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["scoring"],
    summary="Score one order for exploratory review",
    description=(
        "Returns an exploratory calibrated blend score, risk band, pilot-only "
        "action suggestion, and human-readable local model signals. "
        "Invalid order fields or unknown SKUs are rejected; an unavailable "
        "exploratory artifact returns HTTP 503."
    ),
    responses={
        422: {"description": "Invalid dispatch-time input or unknown SKU."},
        503: {"description": "Exploratory scoring artifact is unavailable."},
    },
)
def predict(order: OrderRequest) -> PredictionResponse:
    status = get_prediction_status(ROOT)
    if status["model_status"] != "exploratory":
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "model_status": "blocked",
                "reason": status["reason"],
            },
        )
    try:
        result = predict_order(ROOT, order.model_dump())
    except InvalidOrderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, ProductReferenceError) as exc:
        raise HTTPException(
            status_code=503,
            detail={"status": "unavailable", "model_status": "blocked", "reason": str(exc)},
        ) from exc
    return PredictionResponse(**result)
