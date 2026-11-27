"""Streamlit rendering for read-only V5 model performance results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.train import read_training_gate
from src.v5_dashboard_data import V5DashboardArtifacts, load_v5_dashboard_artifacts


ACTIVE_BLEND_KEY = "active_calibrated_blend"
MODEL_LABELS = {
    ACTIVE_BLEND_KEY: "Active calibrated LR + CatBoost blend",
    "majority_class_baseline": "Majority baseline (no return detection)",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "hybrid_stacking_default_threshold": "Hybrid stacking (0.50 threshold)",
    "hybrid_stacking": "Hybrid stacking (OOF-selected threshold)",
}
MODEL_ORDER = (
    ACTIVE_BLEND_KEY,
    "majority_class_baseline",
    "lightgbm",
    "catboost",
    "xgboost",
    "hybrid_stacking_default_threshold",
    "hybrid_stacking",
)
CHART_MODEL_LABELS = {
    ACTIVE_BLEND_KEY: "Active blend",
    "majority_class_baseline": "Majority baseline",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "hybrid_stacking_default_threshold": "Hybrid · 0.50",
    "hybrid_stacking": "Hybrid · OOF",
}
BASE_MODEL_ORDER = ("lightgbm", "catboost", "xgboost")
CORE_METRICS = ("accuracy", "precision", "recall", "f1")


@st.cache_data(ttl=300)
def _load_cached_results(root: str) -> V5DashboardArtifacts:
    return load_v5_dashboard_artifacts(Path(root))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


@st.cache_data(ttl=300)
def _load_active_blend_metrics(root: str) -> dict[str, Any]:
    path = Path(root) / "derived_data" / "calibrated_ensemble_final" / "metrics.json"
    with path.open(encoding="utf-8") as metrics_file:
        saved_metrics = json.load(metrics_file)
    if not isinstance(saved_metrics, dict):
        raise ValueError("Current candidate metrics must be a JSON object.")

    current = saved_metrics.get("pre_dispatch_v2_update")
    if not isinstance(current, dict):
        raise ValueError("Current candidate metrics are missing the V2 update.")
    q2_metrics = current.get("q2_metrics")
    top_15 = current.get("top_15_percent")
    if not isinstance(q2_metrics, dict) or not isinstance(top_15, dict):
        raise ValueError("Current candidate Q2 or top-15% metrics are unavailable.")

    total_rows = current.get("holdout_rows")
    positive_count = current.get("holdout_positive_count")
    flagged_count = top_15.get("flagged_count")
    true_positive_count = top_15.get("true_positive_count")
    count_values = (total_rows, positive_count, flagged_count, true_positive_count)
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in count_values
    ):
        raise ValueError("Current candidate holdout counts must be integers.")
    if (
        total_rows <= 0
        or positive_count <= 0
        or positive_count > total_rows
        or flagged_count <= 0
        or flagged_count > total_rows
        or true_positive_count < 0
        or true_positive_count > min(positive_count, flagged_count)
    ):
        raise ValueError("Current candidate holdout counts are inconsistent.")

    required_scores = ("roc_auc", "pr_auc", "brier_score")
    scores = {name: _number(q2_metrics.get(name)) for name in required_scores}
    if any(value is None for value in scores.values()):
        raise ValueError("Current candidate Q2 scores are incomplete.")

    false_positives = flagged_count - true_positive_count
    false_negatives = positive_count - true_positive_count
    true_negatives = total_rows - true_positive_count - false_positives - false_negatives
    precision = true_positive_count / flagged_count
    recall = true_positive_count / positive_count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **scores,
        "accuracy": (true_positive_count + true_negatives) / total_rows,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": None,
        "operating_point": "Top 15% capacity",
        "confusion_matrix_labels_0_1": [
            [true_negatives, false_positives],
            [false_negatives, true_positive_count],
        ],
    }


def _format_percent(value: Any) -> str:
    number = _number(value)
    return f"{number:.2%}" if number is not None else "Unavailable"


def _format_threshold(value: Any) -> str:
    number = _number(value)
    return f"{number:.6f}" if number is not None else "Unavailable"


def _format_table_percent(value: Any) -> str:
    number = _number(value)
    return f"{number:.1%}" if number is not None else "Unavailable"


def _format_decimal(value: Any) -> str:
    number = _number(value)
    return f"{number:.4f}" if number is not None else "Unavailable"


def _metric_table(metrics: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_rows = []
    chart_rows = []

    for model_name in MODEL_ORDER:
        result = metrics.get(model_name)
        result = result if isinstance(result, dict) else {}
        threshold = _number(result.get("threshold"))
        operating_point = result.get("operating_point")
        if isinstance(operating_point, str):
            point_label = operating_point
        elif threshold is not None:
            point_label = f"Threshold {threshold:.4f}"
        else:
            point_label = "Unavailable"
        display_rows.append(
            {
                "Model": MODEL_LABELS[model_name],
                "Operating point": point_label,
                "Accuracy": _format_table_percent(result.get("accuracy")),
                "Precision": _format_table_percent(result.get("precision")),
                "Recall": _format_table_percent(result.get("recall")),
                "F1": _format_table_percent(result.get("f1")),
                "ROC-AUC": _format_decimal(result.get("roc_auc")),
                "PR-AUC": _format_decimal(result.get("pr_auc")),
                "Brier": _format_decimal(result.get("brier_score")),
            }
        )
        if all(_number(result.get(metric)) is not None for metric in CORE_METRICS):
            chart_rows.append(
                {
                    "Model": MODEL_LABELS[model_name],
                    "Chart model": CHART_MODEL_LABELS[model_name],
                    **{
                        f"{metric.title()} (%)": _number(result[metric]) * 100
                        for metric in CORE_METRICS
                    },
                }
            )

    return pd.DataFrame(display_rows), pd.DataFrame(chart_rows)


def _class_distribution(
    artifacts: V5DashboardArtifacts,
) -> pd.DataFrame:
    metadata = artifacts.metadata or {}
    rows = []

    training_rows = metadata.get("train_rows")
    training_rate = _number(metadata.get("train_positive_rate"))
    if training_rows is not None and training_rate is not None:
        total = int(training_rows)
        positives = round(total * training_rate)
        rows.append(
            {
                "Cohort": "Training",
                "Returned": positives,
                "Not returned": total - positives,
                "Return rate (%)": training_rate * 100,
            }
        )

    validation_rows = metadata.get("validation_rows")
    validation_positive_count = None
    if artifacts.validation_predictions is not None:
        actual = artifacts.validation_predictions["actual_returned"].astype(int)
        validation_positive_count = int(actual.sum())
        validation_rows = len(actual)
    elif artifacts.metrics is not None:
        baseline = artifacts.metrics.get("majority_class_baseline")
        if isinstance(baseline, dict):
            validation_positive_count = baseline.get("positive_count")

    validation_rate = _number(metadata.get("validation_positive_rate"))
    if validation_rows is not None and validation_positive_count is None and validation_rate is not None:
        validation_positive_count = round(int(validation_rows) * validation_rate)
    if validation_rows is not None and validation_positive_count is not None:
        total = int(validation_rows)
        positives = int(validation_positive_count)
        rows.append(
            {
                "Cohort": "Validation",
                "Returned": positives,
                "Not returned": total - positives,
                "Return rate (%)": (
                    positives / total * 100 if total else None
                ),
            }
        )

    return pd.DataFrame(rows)


def _base_model_leader(metrics: dict[str, Any], metric_name: str) -> str | None:
    values = {
        model_name: _number(
            (metrics.get(model_name) or {}).get(metric_name)
        )
        for model_name in BASE_MODEL_ORDER
    }
    available = {name: value for name, value in values.items() if value is not None}
    if not available:
        return None
    best = max(available.values())
    leaders = [name for name, value in available.items() if value == best]
    return ", ".join(leaders)


def _role_table(metrics: dict[str, Any]) -> pd.DataFrame:
    roles = (
        ("Accuracy-focused", "lightgbm", "accuracy"),
        ("Precision-focused", "catboost", "precision"),
        ("Recall-focused", "xgboost", "recall"),
        ("F1-based final base candidate", "xgboost", "f1"),
    )
    rows = []
    for role, model_name, metric_name in roles:
        model_metrics = metrics.get(model_name)
        value = (
            model_metrics.get(metric_name)
            if isinstance(model_metrics, dict)
            else None
        )
        leader = _base_model_leader(metrics, metric_name)
        rows.append(
            {
                "Assigned evaluation role": role,
                "Model": MODEL_LABELS[model_name],
                "Observed metric": _format_percent(value),
                "Observed base-model leader": (
                    MODEL_LABELS.get(leader, leader)
                    if leader is not None
                    else "Unavailable"
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_dataset_summary(artifacts: V5DashboardArtifacts) -> None:
    metadata = artifacts.metadata
    if metadata is None:
        st.info("Experiment metadata is unavailable.")
        return

    train_rows = metadata.get("train_rows")
    validation_rows = metadata.get("validation_rows")
    total_rows = (
        int(train_rows) + int(validation_rows)
        if train_rows is not None and validation_rows is not None
        else None
    )
    status_left, status_right = st.columns(2)
    status_left.metric(
        "Labeled rows in experiment",
        f"{total_rows:,}" if total_rows is not None else "Unavailable",
        border=True,
    )
    status_right.metric(
        "OOF predictions recorded",
        f"{int(metadata['oof_rows']):,}"
        if metadata.get("oof_rows") is not None
        else "Unavailable",
        border=True,
    )
    with st.container(horizontal=True):
        st.metric(
            "Training rows",
            f"{int(train_rows):,}" if train_rows is not None else "Unavailable",
            f"{_format_percent(metadata.get('train_positive_rate'))} returned",
            border=True,
        )
        st.metric(
            "Validation rows",
            f"{int(validation_rows):,}"
            if validation_rows is not None
            else "Unavailable",
            f"{_format_percent(metadata.get('validation_positive_rate'))} returned",
            border=True,
        )
        st.metric(
            "Validation begins",
            str(metadata.get("validation_start", "Unavailable")),
            border=True,
        )


def render_v5_dashboard(root: Path) -> None:
    """Render the saved V5 experiment metrics and their integrity limitations."""
    root = Path(root)
    artifacts = _load_cached_results(str(root))

    st.title("Mixed hybrid model performance", icon=":material/analytics:")
    experiment_name = (artifacts.metadata or {}).get("experiment", "v5_run_012")
    st.caption(
        f"Kestrel Home · {experiment_name} · Results loaded from saved V5 artifacts; "
        "the dashboard does not retrain models."
    )

    metadata = artifacts.metadata
    experiment_status = (
        str(metadata.get("status", "Unavailable"))
        if metadata is not None
        else "Unavailable"
    )
    gate_status, gate_reason = read_training_gate(root)

    with st.container(horizontal=True):
        st.badge(
            f"Experiment: {experiment_status}",
            color="orange" if experiment_status != "APPROVED" else "green",
            icon=":material/science:",
        )
        st.badge(
            f"Production gate: {gate_status}",
            color="green" if gate_status == "APPROVED" else "red",
            icon=":material/block:",
        )

    st.warning(
        "V5 is exploratory, not production-approved. Label maturity is unverified "
        "and the point-in-time feature audit is incomplete. Do not use these results "
        "to hold, cancel, or prioritize live orders.",
        title="Exploratory results only",
        icon=":material/warning:",
    )
    st.caption(f"Training gate detail: {gate_reason}.")

    gate_path = root / "reports" / "final_training_gate.md"
    try:
        gate_text = gate_path.read_text(encoding="utf-8").lower()
    except (OSError, UnicodeDecodeError):
        gate_text = ""
    if "no longer an untouched holdout" in gate_text:
        st.info(
            "V5 metadata says the validation cohort was not used for selection in "
            "this run. The project training gate records that the same cohort was "
            "reused in earlier experiments, so these results are not an independent "
            "project-level test."
        )

    if artifacts.issues:
        with st.expander(
            f"Artifact availability and consistency ({len(artifacts.issues)})"
        ):
            for issue in artifacts.issues:
                st.write(f"- {issue}")

    if metadata is not None:
        _render_dataset_summary(artifacts)
    else:
        st.info("Dataset counts and experiment details are unavailable.")

    if artifacts.metrics is None:
        st.error(
            "Model comparisons are unavailable because metrics.json could not be loaded."
        )
        return

    metrics = artifacts.metrics
    try:
        active_blend = _load_active_blend_metrics(str(root))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Active calibrated blend metrics are unavailable: {exc}")
        return
    comparison_metrics = {ACTIVE_BLEND_KEY: active_blend, **metrics}
    comparison_table, chart_data = _metric_table(comparison_metrics)

    st.subheader("Model performance comparison")
    st.caption(
        "The active calibrated Logistic + CatBoost blend is highlighted first. "
        "Its classification metrics use top-15% capacity; other rows use saved "
        "thresholds. ROC-AUC, PR-AUC and Brier are threshold-independent."
    )
    if chart_data.empty:
        st.info("Comparison charts are unavailable because saved metrics are incomplete.")
    else:
        metric_columns = [f"{metric.title()} (%)" for metric in CORE_METRICS]
        comparison_long = chart_data.melt(
            id_vars=["Model", "Chart model"],
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Score",
        )
        comparison_chart = (
            alt.Chart(comparison_long)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "Chart model:N",
                    title="Model",
                    sort=chart_data["Chart model"].tolist(),
                    axis=alt.Axis(labelLimit=160),
                ),
                yOffset=alt.YOffset(
                    "Metric:N",
                    sort=metric_columns,
                ),
                x=alt.X(
                    "Score:Q",
                    title="Validation score (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(format=".0f", labelLimit=60),
                ),
                color=alt.Color(
                    "Metric:N",
                    title=None,
                    legend=alt.Legend(
                        orient="bottom",
                        columns=2,
                        labelLimit=120,
                    ),
                ),
                tooltip=[
                    "Model:N",
                    "Metric:N",
                    alt.Tooltip("Score:Q", format=".1f", title="Score (%)"),
                ],
            )
            .properties(height=max(340, len(chart_data) * 44))
        )
        scatter_data = chart_data.rename(
            columns={
                "Chart model": "model",
                "Precision (%)": "precision",
                "Recall (%)": "recall",
                "F1 (%)": "f1",
            }
        )
        precision_recall_chart = (
            alt.Chart(scatter_data)
            .mark_circle()
            .encode(
                x=alt.X(
                    "precision:Q",
                    title="Precision (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                y=alt.Y(
                    "recall:Q",
                    title="Recall (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                color=alt.Color(
                    "model:N",
                    title="Model",
                    legend=alt.Legend(
                        orient="bottom",
                        columns=2,
                        labelLimit=120,
                    ),
                ),
                size=alt.Size(
                    "f1:Q",
                    title="F1 (%)",
                    scale=alt.Scale(range=[80, 500]),
                ),
                tooltip=[
                    alt.Tooltip("Model:N", title="Model"),
                    alt.Tooltip("precision:Q", format=".1f", title="Precision (%)"),
                    alt.Tooltip("recall:Q", format=".1f", title="Recall (%)"),
                    alt.Tooltip("f1:Q", format=".1f", title="F1 (%)"),
                ],
            )
            .properties(height=400)
        )
        comparison_left, comparison_right = st.columns(2)
        with comparison_left:
            st.altair_chart(
                comparison_chart,
                alt="Accuracy, precision, recall and F1 comparison across saved V5 models",
                width="stretch",
            )
        with comparison_right:
            st.altair_chart(
                precision_recall_chart,
                alt="Validation precision versus recall, with marker size representing F1",
                width="stretch",
            )

    highlighted_table = comparison_table.style.apply(
        lambda row: [
            "background-color: #E8F3F1; font-weight: 600;"
            if row.name == 0
            else ""
            for _ in row
        ],
        axis=1,
    )
    st.dataframe(
        highlighted_table,
        hide_index=True,
        alt="Active calibrated blend and saved V5 validation metrics",
        width="stretch",
    )

    if not chart_data.empty:
        st.subheader("F1 comparison")
        f1_data = chart_data[["Model", "Chart model", "F1 (%)"]].rename(
            columns={"F1 (%)": "F1"}
        )
        f1_chart = (
            alt.Chart(f1_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "F1:Q",
                    title="F1 score (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(format=".0f", labelLimit=60),
                ),
                y=alt.Y(
                    "Chart model:N",
                    title=None,
                    sort="-x",
                    axis=alt.Axis(labelLimit=160),
                ),
                color=alt.Color("Chart model:N", legend=None),
                tooltip=[
                    alt.Tooltip("Model:N", title="Model"),
                    alt.Tooltip("F1:Q", format=".1f", title="F1 (%)"),
                ],
            )
            .properties(height=max(280, len(f1_data) * 38))
        )
        st.altair_chart(
            f1_chart,
            alt="F1 score comparison across base models, hybrid stacking and baseline",
            width="stretch",
        )

    baseline = metrics.get("majority_class_baseline")
    if isinstance(baseline, dict):
        st.info(
            "The majority-class baseline has "
            f"{_format_percent(baseline.get('accuracy'))} accuracy, but "
            f"{_format_percent(baseline.get('recall'))} recall and "
            f"{_format_percent(baseline.get('f1'))} F1. It detects no meaningful "
            "share of returns and is shown only as a class-imbalance reference."
        )

    st.subheader("Assigned specialist roles")
    st.caption(
        "These are intended evaluation roles, not guaranteed rankings. The final "
        "column reports the leader among LightGBM, CatBoost and XGBoost using their "
        "saved validation metrics."
    )
    st.dataframe(
        _role_table(metrics),
        hide_index=True,
        alt="Assigned model evaluation roles and observed base-model metric leaders",
    )

    xgboost_metrics = metrics.get("xgboost")
    hybrid_metrics = metrics.get("hybrid_stacking")
    with st.container(border=True):
        st.subheader("V5 final base-model candidate: XGBoost")
        if isinstance(xgboost_metrics, dict):
            st.metric(
                "XGBoost validation F1",
                _format_percent(xgboost_metrics.get("f1")),
                help="Saved result at the threshold recorded in metrics.json.",
            )
        else:
            st.metric("XGBoost validation F1", "Unavailable")
        if isinstance(hybrid_metrics, dict):
            st.caption(
                "The separately thresholded hybrid stacking result has "
                f"{_format_percent(hybrid_metrics.get('f1'))} F1. "
                "Both are exploratory validation results; neither is promoted."
            )
        else:
            st.caption("Hybrid stacking comparison is unavailable.")

    st.subheader("Threshold comparison")
    threshold_rows = []
    for model_name in MODEL_ORDER:
        result = metrics.get(model_name)
        if isinstance(result, dict):
            threshold = _number(result.get("threshold"))
            if threshold is None:
                continue
            threshold_rows.append(
                {
                    "Model": MODEL_LABELS[model_name],
                    "Chart model": CHART_MODEL_LABELS[model_name],
                    "Threshold": threshold,
                }
            )
    threshold_data = pd.DataFrame(threshold_rows).dropna(subset=["Threshold"])
    if threshold_data.empty:
        st.info("Saved model thresholds are unavailable.")
    else:
        threshold_chart = (
            alt.Chart(threshold_data)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "Chart model:N",
                    title=None,
                    sort=threshold_data["Chart model"].tolist(),
                    axis=alt.Axis(labelLimit=160),
                ),
                x=alt.X(
                    "Threshold:Q",
                    title="Decision threshold",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".2f", labelLimit=60),
                ),
                color=alt.Color("Chart model:N", legend=None),
                tooltip=[
                    "Model:N",
                    alt.Tooltip("Threshold:Q", format=".4f"),
                ],
            )
            .properties(height=max(280, len(threshold_data) * 38))
        )
        st.altair_chart(
            threshold_chart,
            alt="Saved decision thresholds for the baseline, base models and hybrid",
            width="stretch",
        )
        st.dataframe(
            threshold_data[["Model", "Threshold"]],
            hide_index=True,
            alt="Decision thresholds used for saved validation metrics",
            width="stretch",
        )

    st.subheader("Class distribution")
    distribution = _class_distribution(artifacts)
    if distribution.empty:
        st.info("Class distribution is unavailable because the required metadata is missing.")
    else:
        distribution_long = distribution.melt(
            id_vars=["Cohort"],
            value_vars=["Returned", "Not returned"],
            var_name="Outcome",
            value_name="Orders",
        )
        class_distribution_chart = (
            alt.Chart(distribution_long)
            .mark_bar()
            .encode(
                x=alt.X("Cohort:N", title=None),
                y=alt.Y(
                    "Orders:Q",
                    title="Share of cohort",
                    stack="normalize",
                    axis=alt.Axis(format=".0%"),
                ),
                color=alt.Color("Outcome:N", title="Outcome"),
                tooltip=[
                    "Cohort:N",
                    "Outcome:N",
                    alt.Tooltip("Orders:Q", format=","),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(
            class_distribution_chart,
            alt="Returned and non-returned shares in the training and validation cohorts",
            width="stretch",
        )
        st.dataframe(
            distribution,
            hide_index=True,
            alt="Training and validation class counts and return rates",
        )

    st.subheader("Confusion matrix")
    matrix_options = [
        key
        for key in MODEL_ORDER
        if isinstance(metrics.get(key), dict)
        and isinstance(
            metrics[key].get("confusion_matrix_labels_0_1"),
            list,
        )
        and len(metrics[key]["confusion_matrix_labels_0_1"]) == 2
    ]
    if matrix_options:
        default_matrix = (
            "xgboost"
            if "xgboost" in matrix_options
            else matrix_options[0]
        )
        selected_matrix = st.selectbox(
            "Model",
            matrix_options,
            index=matrix_options.index(default_matrix),
            format_func=lambda key: MODEL_LABELS[key],
        )
        confusion = metrics[selected_matrix]["confusion_matrix_labels_0_1"]
        matrix_frame = pd.DataFrame(
            confusion,
            index=["Actual: no return", "Actual: return"],
            columns=["Predicted: no return", "Predicted: return"],
        )
        st.dataframe(
            matrix_frame,
            alt=f"Confusion matrix for {MODEL_LABELS[selected_matrix]}",
        )
    else:
        st.info("Confusion matrices are unavailable in the saved metric output.")

    st.subheader("Experiment details")
    if metadata is None:
        st.info("Experiment metadata is unavailable.")
    else:
        detail_left, detail_right = st.columns(2)
        detail_left.write(
            {
                "Architecture": metadata.get("architecture", "Unavailable"),
                "Base models": metadata.get("models_trained", "Unavailable"),
                "Meta-learner": metadata.get("meta_learner", "Unavailable"),
                "Calibration": metadata.get("calibration", "Unavailable"),
                "Selected class weights": metadata.get(
                    "selected_class_weights",
                    "Unavailable",
                ),
            }
        )
        detail_right.write(
            {
                "OOF source rows": metadata.get("oof_source_rows", "Unavailable"),
                "OOF rows": metadata.get("oof_rows", "Unavailable"),
                "Selected hybrid threshold": _format_threshold(
                    metadata.get("threshold")
                ),
                "Threshold objective": metadata.get(
                    "threshold_objective",
                    "Unavailable",
                ),
                "Validation used for V5 selection": metadata.get(
                    "validation_holdout_used_for_selection",
                    "Unavailable",
                ),
                "Label maturity verified": metadata.get(
                    "label_maturity_verified",
                    "Unavailable",
                ),
                "Point-in-time audit completed": metadata.get(
                    "point_in_time_feature_audit_completed",
                    "Unavailable",
                ),
            }
        )

        threshold_details = metadata.get("threshold_selection_details")
        if isinstance(threshold_details, dict):
            st.markdown("**OOF threshold-selection slice**")
            st.caption(
                f"Threshold selection used {threshold_details.get('threshold_selection_rows', 'Unavailable')} "
                f"OOF rows; the full experiment recorded {metadata.get('oof_rows', 'Unavailable')} "
                "OOF rows. These persisted metrics describe the threshold-selection slice, "
                "not the full OOF population."
            )
            oof_metrics = pd.DataFrame(
                [
                    {
                        "Threshold": _format_threshold(
                            threshold_details.get("selected_threshold")
                        ),
                        "Precision": _format_percent(
                            threshold_details.get("oof_precision_at_threshold")
                        ),
                        "Recall": _format_percent(
                            threshold_details.get("oof_recall_at_threshold")
                        ),
                        "F1": _format_percent(
                            threshold_details.get("oof_f1_at_threshold")
                        ),
                        "Target status": threshold_details.get(
                            "target_status",
                            "Unavailable",
                        ),
                    }
                ]
            )
            st.dataframe(
                oof_metrics,
                hide_index=True,
                alt="OOF threshold-selection slice metrics and target status",
            )
            fold_details = metadata.get("oof_fold_details")
            if isinstance(fold_details, list) and fold_details:
                with st.expander("Chronological OOF fold details"):
                    st.dataframe(
                        pd.DataFrame(fold_details),
                        hide_index=True,
                        alt="Chronological out-of-fold row counts, dates and positive rates",
                    )

        if not any(
            "oof" in name.lower() and name.lower().endswith(".csv")
            for name in artifacts.output_files
        ):
            st.info(
                "A full OOF prediction CSV is not present in the V5 artifact folder. "
                "The dashboard shows only the OOF threshold-selection metrics persisted "
                "in experiment_metadata.json."
            )

    st.subheader("Saved model artifacts")
    if artifacts.model_files:
        st.dataframe(
            pd.DataFrame({"Artifact": artifacts.model_files}),
            hide_index=True,
            alt="Saved V5 model and hybrid component artifact filenames",
        )
        st.caption(
            "The hybrid is represented by its saved base models, Logistic Regression "
            "meta-learner, sigmoid calibrator and configuration; this page only reads "
            "the files and does not load or fit estimators."
        )
    else:
        st.info("Saved model artifacts are unavailable.")

    if artifacts.report_text:
        with st.expander("Original V5 experiment report"):
            st.markdown(artifacts.report_text)
