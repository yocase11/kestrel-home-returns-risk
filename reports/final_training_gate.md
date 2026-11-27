# Final Training Gate

**Training status: EXPLORATORY ONLY**

**Production training and prediction generation: BLOCKED.**

The user explicitly authorized exploratory model fitting despite unverified
label maturity, provided results are isolated and clearly identified as
non-business evidence. This status permits only the opt-in command
`.venv\Scripts\python.exe src\train.py --fit --exploratory`. Ordinary `--fit`
remains blocked. Exploratory artifacts are isolated under
`derived_data/exploratory/` and report outputs. Earlier generated experiment
files were pruned during submission cleanup; the retained summary is in
`reports/model_readiness_report.md`.

The later V1 assignment explicitly authorized an **assignment-only** score
file and demo service. The command
`.venv\Scripts\python.exe src\generate_predictions.py --exploratory-submission`
uses the saved exploratory artifact and unlabeled test features to write root
`predictions.csv`. This exception does not open the production gate, create a
production model, or authorize live-order decisions. The default prediction
command remains blocked unless the production gate is approved.

## Evidence limitations

The supplied README, operations policy, email thread, five CSV tables, and
Phase 2/3 reports establish order timestamps, a binary `returned` column, and
a partner-feed duplicate pattern. They do not establish:

- An actual delivery date for each order.
- A return recording or return-request date.
- The extraction/as-of cutoff used to calculate labels.
- The observation window that defines `returned = 0`.
- Whether every historical zero label had enough time to become a return.
- The meaning of each zero at row level.

The latest training order is 2026-06-30, but this is not evidence of label
extraction. The policy's Shield 30-day return entitlement is not a general
label-maturity definition. The 7-, 10-, and 30-calendar-day calculations are
order-date proxy sensitivity experiments only. There are **zero verified
mature rows** under the available evidence.

The regulatory/platform review found no universal statutory return window or
ML label-maturity period that can substitute for Kestrel event history.
Kestrel's own policy says a return starts when raised by the customer, while
later pickup bookings can be cancelled; the `returned` export flag is not tied
to either a request or a completed-return timestamp. See
`reports/label_maturity_regulatory_review.md`. “Zero verified” means no row
can be certified from the supplied evidence, not that every row is known to be
immature.

The separate request-evidence audit has 1,301 supported positive proxies and
9,203 unknown rows, with no inferred negatives. It does not create a complete
binary target or clear the maturity gate; the modeling pipeline still uses
the original target column only.

## Permitted exploratory protocol and completed experiment

- Fixed chronological split at 2026-04-01; duplicate order IDs cannot cross
  partitions.
- The 17-feature point-in-time allowlist plus deterministic interactions
  derived only from allowed order-time fields. Monetary fields, source,
  customer-history counters, pickup/service events, and post-outcome data
  remain excluded.
- The completed comparison covered class-weighted Logistic Regression,
  Random Forest, Extra Trees, HistGradientBoosting, Gradient Boosting,
  CatBoost, XGBoost, and LightGBM, and re-evaluated saved baseline artifacts.
- Hyperparameter search used two expanding chronological folds over the
  latest 3,000 rows of the training partition, average-precision scoring, a
  fixed seed, and two randomized configurations per model family. Selected
  candidates were refit on all pre-validation training rows.
- Validation metrics include accuracy, precision, recall, specificity,
  balanced accuracy, F1, ROC-AUC, PR-AUC, and confusion matrix. The majority
  baseline is reported.
- For analysis only, threshold comparison maximized validation accuracy
  subject to recall >= 20%. This research guard is not a Kestrel business
  policy; model/threshold selection on the same holdout makes the results
  optimistic.
- Label every metric **EXPLORATORY ONLY — LABEL MATURITY UNVERIFIED**.
- Keep model artifacts and results outside production directories. The
  unlabeled test file is read only by the explicit assignment-only scoring
  command; it has no target and is not used to compute performance metrics.

The 95% accuracy target was **not achieved** while requiring nontrivial return
detection. The best under the 20% recall guard attained 86.45% accuracy,
35.17% precision, and 20.82% recall. A previously saved model reached 88.76%
accuracy only at 2.45% recall, which is not meaningful detection. Neither
result is reliable evidence of mature-label performance.

The holdout remains chronologically later than training, but it is not a valid
business performance estimate while label maturity is unresolved. Even a high
metric must not be interpreted as evidence for deployment or business action.

## Limitation review — evidence still unavailable

The supplied source files were rechecked before further model selection:

- The labeled train table has 10,504 unique orders from 2025-04-01 through
  2026-06-30 and includes `returned` plus `label_maturity_status=unverified`.
- The future test table has 2,096 unique orders from 2026-07-01 through
  2026-09-30 and has no `returned` column. The supplied README calls it the
  warehouse snapshot at dispatch.
- The order schemas and accompanying README, policy, and email provide no
  delivery/fulfillment timestamp, label extraction cutoff, row-level return
  finalization timestamp, or agreed observation-window contract.

Therefore label maturity cannot be established or repaired from this project
copy. Do not infer a cutoff from the last order date, latest pickup timestamp,
or Shield's 30-day entitlement; none is a verified label as-of date.

The April-June 2026 validation block is also no longer an untouched holdout.
It has been used repeatedly for candidate comparison, feature/model selection,
probability calibration evaluation, and threshold selection. It remains a
chronological retrospective comparison, but its selected metrics are
selection-optimistic and cannot serve as independent generalization evidence.
The unlabeled July-September test snapshot cannot validate those metrics.

**Required before restarting performance claims:** obtain delivery dates,
the label snapshot/as-of timestamp, the return outcome/finalization definition
(including cancellations and unresolved cases), and row/cohort maturity
status. Apply that agreed rule, then freeze the candidate, preprocessing,
calibration method, and operating threshold before revealing outcomes for a
newer mature labeled cohort. Do not reuse the April-June block as a fresh test.
Until these inputs exist, further threshold tuning may be exploratory only and
cannot resolve either limitation.

## Conditions for production approval

Training and prediction generation remain blocked until Kestrel supplies or
verifies:

1. An order-level delivery or fulfillment-date field.
2. The label extraction/as-of timestamp.
3. The target observation window and return-recording process.
4. A row-level or cohort-level rule for identifying immature zeros.
5. An approved definition of whether held, cancelled, or unresolved orders
   are positive or negative.
6. The final maturity rule applied to historical training rows.
7. Business approval for model use, threshold selection, and operational
   intervention costs.
8. A later mature labeled cohort kept untouched until the model, calibration,
   and threshold are frozen; the supplied July-September table has no labels.

The October payment-scale cause, migration/source semantics, and feature
availability at dispatch also remain unresolved. No amount correction is
authorized. These limitations must be resolved before monetary features or
operational scoring can be considered.

The current calibrated 86/14 Logistic Regression/CatBoost candidate is
isolated under `derived_data/calibrated_ensemble_final/` and is **exploratory
only**. Its Q2 holdout was reused and label maturity remains unverified.
Saving an artifact or gated `predictions.csv` does not approve production
training, scoring, or live decisions; the gate above remains unchanged.
