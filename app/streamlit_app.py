"""Employee-facing review screen for exploratory Kestrel returns-risk scoring."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.v5_model_dashboard import render_v5_dashboard


API_URL = os.environ.get("KESTREL_API_URL", "http://127.0.0.1:8000").rstrip("/")
PRODUCT_REFERENCE = ROOT / "derived_data" / "product_reference.csv"
METRICS_PATH = ROOT / "derived_data" / "calibrated_ensemble_final" / "metrics.json"
REQUEST_TIMEOUT_SECONDS = 5
CALL_COST_RS = 45
RETURN_COST_RS = 1150
CALL_PREVENTION_RATE = 0.35
LONG_HOLD_CANCELLATION_RATE = 0.12


@st.cache_data(ttl=300)
def load_final_metrics(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    if not isinstance(metrics, dict):
        raise ValueError("Saved model metrics must be a JSON object.")

    current = metrics.get("pre_dispatch_v2_update")
    blend = current.get("q2_metrics") if isinstance(current, dict) else None
    top_15 = (
        current.get("top_15_percent") if isinstance(current, dict) else None
    )
    required = (
        isinstance(blend, dict)
        and isinstance(top_15, dict)
        and isinstance(blend.get("roc_auc"), (int, float))
        and isinstance(blend.get("pr_auc"), (int, float))
        and isinstance(blend.get("brier_score"), (int, float))
        and isinstance(top_15.get("return_capture_rate"), (int, float))
        and isinstance(top_15.get("lift_vs_random"), (int, float))
    )
    if not required:
        raise ValueError("Saved metrics are missing final blend KPI values.")
    return metrics


@st.cache_data(ttl=3600)
def load_product_reference(path: str) -> pd.DataFrame:
    products = pd.read_csv(path, dtype={"sku": "string"})
    required = {"sku", "family", "model_name"}
    missing = required.difference(products.columns)
    if missing:
        raise ValueError(f"Product lookup is missing columns: {sorted(missing)}")
    if products.empty:
        raise ValueError("Product lookup contains no selectable products.")
    if products["sku"].isna().any() or products["sku"].duplicated().any():
        raise ValueError("Product lookup SKU values must be present and unique.")
    return products.sort_values("sku", kind="stable").reset_index(drop=True)


def product_label(products: pd.DataFrame, sku: str) -> str:
    product = products.loc[products["sku"].eq(sku)].iloc[0]
    return f"{sku} · {product['model_name']} ({product['family']})"


def parse_api_error(response: requests.Response) -> str:
    try:
        payload: Any = response.json()
    except requests.exceptions.JSONDecodeError:
        return f"The scoring service returned HTTP {response.status_code}."

    detail = payload.get("detail", "The scoring service rejected the request.")
    if isinstance(detail, dict):
        return str(detail.get("reason", detail))
    if isinstance(detail, list):
        messages = [
            str(item.get("msg", "Invalid input"))
            for item in detail
            if isinstance(item, dict)
        ]
        return "; ".join(messages) or "The order record is invalid."
    return str(detail)


st.set_page_config(
    page_title="Returns risk review | Kestrel Home",
    page_icon=":material/analytics:",
    layout="wide",
)

st.html(
    """
    <style>
    .st-key-kpi_capture,
    .st-key-kpi_break_even,
    .st-key-kpi_policy,
    .st-key-kpi_validation {
        border: 1px solid #D6E0E8;
        border-radius: 0.75rem;
        background: #FFFFFF;
        padding: 0.7rem 0.85rem;
    }
    .st-key-kpi_capture h3,
    .st-key-kpi_break_even h3,
    .st-key-kpi_policy h3,
    .st-key-kpi_validation h3 {
        color: #18232F;
        font-size: clamp(1.25rem, 1.8vw, 1.65rem);
        line-height: 1.15;
        margin: 0.15rem 0 0.35rem;
        overflow-wrap: anywhere;
    }
    .st-key-kpi_capture p,
    .st-key-kpi_break_even p,
    .st-key-kpi_policy p,
    .st-key-kpi_validation p {
        line-height: 1.35;
        overflow-wrap: anywhere;
        white-space: normal;
    }
    .st-key-decision_summary {
        border: 1px solid #B9D2CF;
        border-left: 4px solid #0B6B66;
        border-radius: 0.85rem;
        background: #FFFFFF;
        padding: 1rem;
    }
    .st-key-risk_gauge {
        border-radius: 0.65rem;
        background: #F1F7F7;
        padding: 0.65rem 0.8rem;
    }
    .st-key-risk_driver_0,
    .st-key-risk_driver_1,
    .st-key-risk_driver_2,
    .st-key-risk_driver_3 {
        border: 1px solid #DFE7EC;
        border-radius: 0.65rem;
        background: #FFFFFF;
        padding: 0.55rem 0.7rem;
    }
    </style>
    """
)

selected_view = st.sidebar.segmented_control(
    "Workspace",
    ["Order review", "Model performance"],
    default="Order review",
    key="workspace_view",
    width="stretch",
)

if selected_view == "Model performance":
    render_v5_dashboard(ROOT)
    st.stop()

st.title("Returns risk review", icon=":material/analytics:")
st.caption("Kestrel Home · Dispatch-time decision-support prototype")

try:
    dashboard_metrics = load_final_metrics(str(METRICS_PATH))
except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    st.error(f"Final model KPI metrics are unavailable: {exc}")
    st.stop()

current_metrics = dashboard_metrics["pre_dispatch_v2_update"]
blend_metrics = current_metrics["q2_metrics"]
top_15_metrics = current_metrics["top_15_percent"]
call_break_even_risk = CALL_COST_RS / (
    RETURN_COST_RS * CALL_PREVENTION_RATE
)
with st.grid(4, min_column_width=210, gap="small"):
    for key, label, value, delta in (
        (
            "capture",
            "Top-15% Return Capture",
            f"{top_15_metrics['return_capture_rate']:.1%}",
            f"{top_15_metrics['lift_vs_random']:.2f}× vs random baseline",
        ),
        (
            "break_even",
            "Call Break-Even Risk",
            f"{call_break_even_risk:.1%}",
            "₹45 call vs ₹1,150 return",
        ),
        (
            "policy",
            "Policy Guardrail",
            "Call First",
            "Avoids 12% hold cancellation",
        ),
        (
            "validation",
            "Q2 Holdout ROC-AUC",
            f"{blend_metrics['roc_auc']:.3f}",
            f"PR-AUC {blend_metrics['pr_auc']:.3f} · "
            f"Brier {blend_metrics['brier_score']:.3f}",
        ),
    ):
        with st.container(key=f"kpi_{key}"):
            st.caption(label)
            st.markdown(f"### {value}")
            st.caption(delta)

st.warning(
    "This calibrated blend is exploratory, not approved for live operations. "
    "Historical labels are not verified as mature, and Q2 2026 was reused during "
    "selection. Treat call suggestions as pilot-only; never hold or cancel dispatch.",
    icon=":material/warning:",
)

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    health_response.raise_for_status()
    health = health_response.json()
except requests.RequestException as exc:
    st.error(
        f"The scoring API is unavailable at {API_URL}. Start it using the README "
        "instructions before requesting a score.",
        icon=":material/cloud_off:",
    )
    health = None

if health is not None:
    status_left, status_right = st.columns(2)
    with status_left:
        if health.get("model_status") == "exploratory":
            st.badge(
                "Exploratory scoring available",
                color="green",
                icon=":material/check_circle:",
            )
        else:
            st.badge(
                "Scoring unavailable",
                color="red",
                icon=":material/error:",
            )
    with status_right:
        st.badge(
            "Production use blocked",
            color="orange",
            icon=":material/block:",
        )

try:
    products = load_product_reference(str(PRODUCT_REFERENCE))
except (FileNotFoundError, pd.errors.ParserError, ValueError) as exc:
    st.error(
        f"The packaged product lookup is unavailable or invalid: {exc}",
        icon=":material/database_off:",
    )
    st.stop()

with st.container(border=True):
    st.subheader("Order details")
    st.caption(
        "Use dispatch-time values only. Customer identifiers and monetary fields "
        "are not collected; do not enter personal information in delivery notes."
    )

    default_values = {
        "order_placed_at": datetime.now().replace(microsecond=0).isoformat(sep=" "),
        "sku": str(products.iloc[0]["sku"]),
        "sales_channel": "app",
        "payment_mode": "prepaid_upi",
        "discount_pct": 0,
        "qty": 1,
        "promised_delivery_days": 4,
        "delivery_pincode": "000000",
        "is_gift": "N",
        "customer_prior_orders": 0,
        "customer_prior_returns": 0,
        "delivery_note": "",
        "shield_member_verified_at_dispatch": False,
    }
    presets = {
        "standard": {
            **default_values,
            "payment_mode": "prepaid_upi",
            "delivery_pincode": "411001",
        },
        "review": {
            **default_values,
            "payment_mode": "cod",
            "delivery_pincode": "000000",
            "customer_prior_orders": 2,
            "customer_prior_returns": 1,
            "delivery_note": "Please call before delivery",
        },
        "shield": {
            **default_values,
            "payment_mode": "prepaid_card",
            "delivery_pincode": "411001",
            "customer_prior_orders": 3,
            "customer_prior_returns": 1,
            "shield_member_verified_at_dispatch": True,
        },
    }
    for field, value in default_values.items():
        if field not in st.session_state:
            st.session_state[field] = value

    st.markdown("**One-click example orders**")
    preset_columns = st.columns(3)
    for column, preset_name, label in zip(
        preset_columns,
        ("standard", "review", "shield"),
        ("Standard order", "Review example", "Verified Shield example"),
    ):
        if column.button(label, key=f"preset_{preset_name}", width="stretch"):
            for field, value in presets[preset_name].items():
                st.session_state[field] = value
            st.rerun()

    with st.form("order_form", clear_on_submit=False):
        order_placed_at = st.text_input(
            "Order placed at (IST)",
            key="order_placed_at",
            help="Enter a timezone-naive timestamp, for example 2026-11-27 10:30:00.",
        )
        sku = st.selectbox(
            "Product",
            products["sku"].astype(str).tolist(),
            format_func=lambda value: product_label(products, value),
            key="sku",
        )

        channel_col, payment_col = st.columns(2)
        sales_channel = channel_col.selectbox(
            "Sales channel",
            ["app", "web", "marketplace", "partner_outlet"],
            key="sales_channel",
        )
        payment_mode = payment_col.selectbox(
            "Payment mode",
            ["prepaid_upi", "prepaid_card", "cod", "emi"],
            key="payment_mode",
        )

        discount_col, quantity_col = st.columns(2)
        discount_pct = discount_col.number_input(
            "Discount (%)",
            min_value=0,
            max_value=100,
            step=1,
            key="discount_pct",
        )
        qty = quantity_col.number_input(
            "Quantity",
            min_value=1,
            max_value=100,
            step=1,
            key="qty",
        )

        delivery_col, gift_col = st.columns(2)
        promised_delivery_days = delivery_col.number_input(
            "Promised delivery (days)",
            min_value=1,
            max_value=365,
            step=1,
            key="promised_delivery_days",
        )
        is_gift = gift_col.selectbox("Gift order", ["N", "Y"], key="is_gift")

        delivery_pincode = st.text_input(
            "Delivery pincode",
            help="Use 000000 only when the pincode is unavailable.",
            key="delivery_pincode",
        )
        history_col, note_col = st.columns(2)
        customer_prior_orders = history_col.number_input(
            "Earlier orders (known at dispatch)",
            min_value=0,
            max_value=10000,
            step=1,
            key="customer_prior_orders",
        )
        customer_prior_returns = history_col.number_input(
            "Earlier returns (known at dispatch)",
            min_value=0,
            max_value=10000,
            step=1,
            key="customer_prior_returns",
        )
        delivery_note = note_col.text_area(
            "Delivery note (no personal information)",
            max_chars=500,
            key="delivery_note",
        )
        shield_member_verified = st.checkbox(
            "Shield membership verified as of dispatch",
            key="shield_member_verified_at_dispatch",
            help="Policy-only input; it is not a model feature.",
        )
        submitted = st.form_submit_button(
            "Calculate exploratory score",
            type="primary",
            icon=":material/analytics:",
            width="stretch",
        )

if submitted:
    payload = {
        "order_placed_at": order_placed_at.strip(),
        "sku": sku,
        "sales_channel": sales_channel,
        "payment_mode": payment_mode,
        "discount_pct": int(discount_pct),
        "qty": int(qty),
        "promised_delivery_days": int(promised_delivery_days),
        "delivery_pincode": delivery_pincode.strip(),
        "is_gift": is_gift,
        "customer_prior_orders": int(customer_prior_orders),
        "customer_prior_returns": int(customer_prior_returns),
        "delivery_note": delivery_note,
        "shield_member_verified_at_dispatch": shield_member_verified,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        st.error(
            "The scoring request timed out. No score was returned; retry when the API is responsive.",
            icon=":material/schedule:",
        )
    except requests.ConnectionError:
        st.error(
            "The scoring API connection failed. No score was returned.",
            icon=":material/cloud_off:",
        )
    except requests.RequestException as exc:
        st.error(f"The scoring request failed: {exc}", icon=":material/error:")
    else:
        if not response.ok:
            st.error(
                parse_api_error(response),
                title=f"Scoring failed (HTTP {response.status_code})",
                icon=":material/error:",
            )
        else:
            result = response.json()
            score = float(result["score"])
            score_to_break_even = score / call_break_even_risk
            action_text = str(result["recommended_action"])
            if "soft call only" in action_text.casefold():
                action_badge = "Kestrel Shield Protected: Gentle Verification Call Only"
                action_color = "blue"
                action_icon = ":material/shield:"
            elif "confirmation call" in action_text.casefold():
                action_badge = "Trigger ₹45 Pre-Dispatch Confirmation Call"
                action_color = "orange"
                action_icon = ":material/call:"
            else:
                action_badge = "Proceed with Standard Dispatch"
                action_color = "green"
                action_icon = ":material/local_shipping:"

            st.subheader("Decision summary")
            with st.container(key="decision_summary"):
                gauge_column, action_column = st.columns([1.15, 1.85])
                with gauge_column:
                    with st.container(key="risk_gauge"):
                        st.markdown("**Calibrated return probability**")
                        st.progress(
                            min(score_to_break_even, 1.0),
                            text=f"{score:.1%} probability",
                        )
                        if score >= call_break_even_risk:
                            comparison = "At or above"
                        else:
                            comparison = "Below"
                        st.caption(
                            f"{comparison} the {call_break_even_risk:.1%} "
                            "call break-even reference."
                        )
                    st.badge(
                        f"{result['risk_tier']} relative risk",
                        color="orange" if result["risk_tier"] != "Low" else "green",
                        icon=":material/monitoring:",
                    )

                with action_column:
                    st.markdown("**Operational action**")
                    st.badge(
                        action_badge,
                        color=action_color,
                        icon=action_icon,
                    )
                    st.caption(action_text)

                st.caption(result["score_interpretation"])
                st.markdown("**Key Risk Drivers (Ops View)**")
                driver_columns = st.columns(2)
                reasons = result["reasons"]
                for index, reason in enumerate(reasons[:4]):
                    with driver_columns[index % 2]:
                        with st.container(key=f"risk_driver_{index}"):
                            st.markdown(f"- {reason}")
                st.caption(result["reasoning_note"])

            st.warning(
                "This output does not establish that an order will be returned "
                "and does not approve any live operational action.",
                title="Pilot-only suggestion; no hold or cancellation",
                icon=":material/do_not_disturb_on:",
            )

with st.expander("Model limitations and intended use"):
    st.markdown(
        "- The blend uses sigmoid/isotonic calibration selected from training-side chronological OOF predictions; Q2 validation and labels remain reused/unverified.\n"
        "- The historical return labels have no verified maturity cutoff.\n"
        "- Validation data was reused for model and threshold selection.\n"
        "- Feature signals are associations, not causes or intervention effects.\n"
        "- Customer identifiers, monetary amounts, and post-dispatch events are excluded.\n"
        "- Product catalog prices and signup-derived tenure are excluded because reference effective dates are unavailable."
    )

st.caption("Internal demonstration · No customer data is stored by this screen.")
