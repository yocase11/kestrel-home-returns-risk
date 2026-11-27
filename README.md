# Kestrel Home returns-risk review

This repository contains an exploratory, pre-dispatch return-risk model, a
local FastAPI endpoint, and a Streamlit order-review screen. It is not approved
for production, automated dispatch holds, or customer decisions.

## Final candidate evidence

The calibrated blend is **86% Logistic Regression + 14% CatBoost**. On the
April–June 2026 chronological holdout (2,126 rows; 245 positive labels) it
scored **0.7568 ROC-AUC, 0.3736 PR-AUC, and 0.0908 Brier**. At a top-15%
ranking capacity, it captured **101/245 returns (41.22%; 2.75× the 15% random
capture baseline)**. These results are exploratory: Q2 was reused for model
and blend selection, label maturity is unverified, and this does not establish
performance on the unlabeled test period.

The production training gate remains **BLOCKED**. The highest observed
threshold accuracy on Q2 was 89.51%, not 95%; this is not a theoretical
maximum. Since the Q2 positive rate was 11.52%, a majority-class rule alone
would score 88.48% accuracy while detecting no returns. Never use leaked
post-dispatch fields to inflate performance.

## Start the local demo

Use Python 3.13 on Windows. From the repository root, create the environment
and install dependencies once:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start both the API and UI in one foreground command:

```powershell
.\.venv\Scripts\python.exe src\run_local_demo.py
```

Open the Streamlit URL printed in the terminal (normally
`http://localhost:8501`). The API is at `http://127.0.0.1:8000`; its schema is
available at `/docs`. Press Ctrl+C in the terminal to stop both services.
The app and model make no paid API calls and need no API key. A local
`calibrated_ensemble.joblib` artifact is required for scoring; if it is
missing or invalid, the API reports an explicit unavailable response.

The review page has three one-click example orders and sends submitted
dispatch-time fields to `POST /predict`. Results include a score, relative
risk tier, Policy v4.1 suggested action, and plain-English model signals.
The screen does not ask for customer identifiers or payment amounts.

## Privacy and data

- Original files under `datasets/` are local-only and ignored by `.gitignore`.
  Do not publish, copy into the app, or add them to a submission bundle.
- Row-level derived tables, holdout predictions, and report CSVs are ignored.
- The packaged `derived_data/product_reference.csv` contains product choices,
  not customer records. Dashboard KPI cards read the aggregate
  `derived_data/calibrated_ensemble_final/metrics.json`; they do not read raw
  datasets.
- The model excludes `pickup_scheduled_at`, `last_service_event_type`,
  `source`, and corrupted raw `order_value_inr`. Customer tenure and catalog
  price features are excluded because effective-dated references are absent.
- Delivery-note features use a six-keyword whitelist; raw note text is not
  retained in model features.

## Reproduce data preparation and training

Data preparation requires the private source files:

```powershell
.\.venv\Scripts\python.exe src\data_processing.py --overwrite
```

The performance experiment is exploratory and may overwrite the current
candidate artifacts and submission only when both configured Q2 AUC gates
pass:

```powershell
.\.venv\Scripts\python.exe src\train_pre_dispatch_v2.py
```

This experiment uses chronological training folds and the fixed Q2 holdout.
Because Q2 has been reused, its results are not an untouched final test. Keep
the production gate unchanged unless label maturity, feature timing, and a
later independent validation cohort have been verified and approved.

## Tests and outputs

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected submission format: root `predictions.csv`, with exactly 2,096 rows
and columns `order_id,score`, preserving the provided test order. The
unlabeled test set has no outcomes, so test metrics cannot be computed.

Key deliverables:

- `predictions.csv` — gated exploratory ranking scores.
- `derived_data/calibrated_ensemble_final/models/calibrated_ensemble.joblib`
  — local model bundle used by the API.
- `derived_data/calibrated_ensemble_final/metrics.json` and
  `feature_explanations.json` — saved model comparison and explanations.
- `reports/submission_form.md` — expected score and submission disclosures.
- `reports/model_readiness_report.md` — validation evidence and governance.
- `reports/memo_to_ritu.md` — executive decision memo.
- `reports/data_audit.md`, `data_quality_report.md`, and
  `data_decisions.md` — source and data-quality investigations.

## Known limitations and business guardrails

The 11.2% call break-even risk is arithmetic:
`₹45 / (₹1,150 × 35%)`. The 35% prevention effect is an assumption, not a
measured treatment effect. Use only in a capped, randomized human-reviewed
call pilot. Do not hold dispatch: Policy v4.1 cites a 12% customer-cancellation
penalty for holds exceeding 24 hours. Protect verified Shield members from
hard holds.

Production remains blocked until label maturity and customer-counter
provenance are verified, payment scaling is reconciled, and an untouched
mature cohort and controlled pilot are evaluated. See
`reports/final_training_gate.md` before proposing operational use.
