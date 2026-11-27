import unittest
from pathlib import Path
import json
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from app.v5_model_dashboard import (
    ACTIVE_BLEND_KEY,
    _load_active_blend_metrics,
    _metric_table,
)


ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def test_review_form_renders_with_exploratory_guardrails(self) -> None:
        app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(
            timeout=20
        )

        self.assertFalse(app.exception)
        visible_copy = [
            element.value
            for element in (*app.markdown, *app.caption)
        ]
        for expected in (
            "Top-15% Return Capture",
            "41.2%",
            "2.75× vs random baseline",
            "Call Break-Even Risk",
            "11.2%",
            "₹45 call vs ₹1,150 return",
            "Policy Guardrail",
            "Call First",
            "Avoids 12% hold cancellation",
            "Q2 Holdout ROC-AUC",
            "0.757",
            "PR-AUC 0.374 · Brier 0.091",
        ):
            self.assertTrue(
                any(expected in value for value in visible_copy),
                f"Expected dashboard text not rendered: {expected}",
            )
        self.assertEqual(
            [button.label for button in app.button],
            [
                "Standard order",
                "Review example",
                "Verified Shield example",
                "Calculate exploratory score",
            ],
        )
        self.assertIn("Product", [widget.label for widget in app.selectbox])
        self.assertTrue(
            any("not approved for live operations" in warning.value for warning in app.warning)
        )

    def test_model_performance_view_loads_saved_v5_results(self) -> None:
        app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(
            timeout=30
        )
        app.sidebar.segmented_control[0].set_value("Model performance").run(
            timeout=30
        )

        self.assertFalse(app.exception)
        self.assertIn(
            "Mixed hybrid model performance",
            [title.value for title in app.title],
        )
        self.assertTrue(
            any(
                "v5_run_012" in caption.value
                for caption in app.caption
            )
        )
        self.assertTrue(
            any(
                "highlighted first" in caption.value
                for caption in app.caption
            )
        )

    def test_active_blend_is_first_and_auc_values_are_precise(self) -> None:
        v5_metrics_path = (
            ROOT
            / "derived_data"
            / "exploratory_v5_v5_run_012"
            / "metrics.json"
        )
        v5_metrics = json.loads(v5_metrics_path.read_text(encoding="utf-8"))
        metrics = {
            ACTIVE_BLEND_KEY: _load_active_blend_metrics(str(ROOT)),
            **v5_metrics,
        }

        table, chart_data = _metric_table(metrics)

        self.assertEqual(
            table.iloc[0]["Model"],
            "Active calibrated LR + CatBoost blend",
        )
        self.assertEqual(table.iloc[0]["Accuracy"], "83.0%")
        self.assertEqual(table.iloc[0]["ROC-AUC"], "0.7568")
        self.assertEqual(table.iloc[0]["PR-AUC"], "0.3736")
        self.assertEqual(table.iloc[0]["Brier"], "0.0908")
        self.assertEqual(chart_data.iloc[0]["Chart model"], "Active blend")

    def test_all_presets_post_and_render_plain_english_reasons(self) -> None:
        health_response = Mock()
        health_response.json.return_value = {"model_status": "exploratory"}

        def prediction_response_for_order(
            _url: str,
            *,
            json: dict[str, object],
            timeout: int,
        ) -> Mock:
            response = Mock()
            response.ok = True
            shield_member = json["shield_member_verified_at_dispatch"]
            score = 0.24 if json["payment_mode"] == "cod" or shield_member else 0.04
            if shield_member:
                action = "Pilot-only: protect the verified Shield member with a soft call only."
            elif score >= 45 / (1150 * 0.35):
                action = "Pilot-only: consider a Rs 45 pre-dispatch confirmation call."
            else:
                action = "Proceed with the normal dispatch workflow."
            response.json.return_value = {
                "score": score,
                "risk_tier": "Medium" if score >= 0.1 else "Low",
                "recommended_action": action,
                "score_interpretation": "Exploratory risk score.",
                "reasons": [
                    "Cash-on-delivery orders returned more often in this sample.",
                    "Longer promised delivery periods raised the score in this sample.",
                    "This product family had more returns in this sample.",
                ],
                "reasoning_note": "Associations are not causes.",
            }
            return response

        with (
            patch("requests.get", return_value=health_response),
            patch("requests.post", side_effect=prediction_response_for_order) as post,
        ):
            app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py")).run(
                timeout=20
            )
            expected_actions = (
                "Proceed with the normal dispatch workflow.",
                "Pilot-only: consider a Rs 45 pre-dispatch confirmation call.",
                "Pilot-only: protect the verified Shield member with a soft call only.",
            )
            for preset_label, expected_action in zip(
                (
                    "Standard order",
                    "Review example",
                    "Verified Shield example",
                ),
                expected_actions,
            ):
                preset_index = next(
                    index
                    for index, button in enumerate(app.button)
                    if button.label == preset_label
                )
                app.button[preset_index].click().run(timeout=20)
                submit_index = next(
                    index
                    for index, button in enumerate(app.button)
                    if button.label == "Calculate exploratory score"
                )
                app.button[submit_index].click().run(timeout=20)

                self.assertFalse(app.exception)
                self.assertTrue(
                    any(
                        expected_action in element.value
                        for element in app.caption
                    ),
                    f"Expected action not rendered for {preset_label}.",
                )
                self.assertTrue(
                    any(
                        "Cash-on-delivery orders returned more often"
                        in element.value
                        for element in app.markdown
                    )
                )
                self.assertTrue(
                    any(
                        "Key Risk Drivers (Ops View)" in element.value
                        for element in app.markdown
                    )
                )

        self.assertEqual(post.call_count, 3)
        self.assertTrue(
            all(call.args[0].endswith("/predict") for call in post.call_args_list)
        )
        self.assertTrue(
            post.call_args_list[2].kwargs["json"][
                "shield_member_verified_at_dispatch"
            ]
        )


if __name__ == "__main__":
    unittest.main()
