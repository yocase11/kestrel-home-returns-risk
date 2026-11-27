# Modeling readiness and exploratory training report

**Production training: BLOCKED. Exploratory training: explicitly authorized.**
Exploratory results do not resolve label maturity, approve deployment, or
authorize live-order decisions.

## Current candidate — pre-dispatch V2

The current saved candidate is **86% calibrated Logistic Regression + 14%
calibrated CatBoost**. On the chronological Q2 2026 cohort (2,126 rows, 245
positive labels), it scored **0.7568 ROC-AUC, 0.3736 PR-AUC, and 0.0908
Brier**. The saved prior Logistic Regression reference scored 0.7057 ROC-AUC /
0.2584 PR-AUC on this cohort.

At top-15% capacity, 319 orders were flagged: **101/245 returns captured
(41.22%; 31.66% precision)**, or **2.75×** the 15% random-capture baseline.
This is descriptive Q2 evidence, not a future error rate or causal call
effect. The report and submission form use these V2 figures as the current
candidate; the earlier 55/45 blend section below is retained as historical
comparison only.

Q2 was reused for model, blend, and threshold selection. Label maturity is
unverified, so the Q2 metrics are selection-optimistic and production
promotion remains blocked. The highest observed threshold accuracy was
89.51%; the majority-class baseline is 88.48% with zero return recall. These
observations do not prove that 95% accuracy is mathematically impossible,
but provide no support for promising it. Post-dispatch information remains
excluded.

Earlier model comparisons and audits below are historical evidence; they do
not supersede this current candidate summary or the production gate.

## Work completed

- Inspected `src/data_processing.py`, both prior reports, and the prepared train/test tables before changes.
- Removed `customer_prior_orders` and `customer_prior_returns` from the primary model allowlist and regenerated the derived tables.
- Monetary fields, `source`, `pickup_scheduled_at`, and `last_service_event_type` remain excluded. `order_placed_at` remains split metadata and supplies safe order-calendar features.
- Added an explicit chronological split with one row per order ID and assertions against ID overlap and future-to-past overlap.
- Added unfitted Logistic Regression and CatBoost pipeline factories.
- Added metric, calibration, and subgroup-evaluation utilities.
- Added this report and [modeling_assumptions.md](./modeling_assumptions.md).

## Label maturity experiment

No order record or supplied document provides a delivery date or the extraction/as-of cutoff for `returned`. The last order date, 2026-06-30, is not evidence of when labels were extracted. The 30-day Shield return entitlement does not establish a general target window.

The follow-up regulatory/platform review found no universal 7-, 10-, or
30-day e-commerce rule or standard ML maturity period that could verify these
labels. Kestrel's policy says a return starts when the customer raises a
request, while a later booked pickup may be cancelled. The export does not
identify which lifecycle event its `returned` flag represents. See
[label_maturity_regulatory_review.md](./label_maturity_regulatory_review.md)
for the proposed event contract and required source fields.

The resulting one-sided audit is stored separately at
`derived_data/return_request_label_audit.csv`: 1,301 positive-evidence
proxies, 9,203 unknown outcomes, and no inferred negatives. It exposes 101
source `returned=0` rows with approved-return-workflow evidence. The file is
not a standard binary training target and is not included in the model
pipeline.

The table gives conditional **order-date proxy candidates only**. Proxy cutoff is the latest training order calendar date (2026-06-30); candidate rows have `order_placed_at.date <= proxy cutoff - window`. Since both delivery date and actual cutoff are missing, **verified eligible/mature count is 0 for each experiment**, and actual eligible counts are unknown.

| Window | Proxy candidate rows | Proxy-excluded rows | Candidate `returned=1` | Candidate `returned=0` | Verified mature rows |
|---:|---:|---:|---:|---:|---:|
| 10 days (primary) | 10,282 | 222 | 1,175 | 9,107 | 0 |
| 7 days (sensitivity) | 10,345 | 159 | 1,182 | 9,163 | 0 |
| 30 days (sensitivity) | 9,805 | 699 | 1,123 | 8,682 | 0 |

The order-date window proxies did not change or discard source labels or the
existing modeling target; the separate request-evidence audit is documented
below. Every prepared train row continues to carry
`label_maturity_status=unverified`. The proxy counts are not grounds for
claiming label maturity or training eligibility.

## Provisional chronological split

| Partition | Rule | Rows | Positive | Negative | Date range |
|---|---|---:|---:|---:|---|
| Train | `order_placed_at < 2026-04-01` | 8,378 | 955 | 7,423 | 2025-04-01 through 2026-03-31 |
| Validation | `order_placed_at >= 2026-04-01` | 2,126 | 245 | 1,881 | 2026-04-01 through 2026-06-30 |

Both partitions have unique order IDs and no shared IDs. The latest training timestamp precedes the earliest validation timestamp. This is a fixed, reproducible split, but the labels in both partitions remain unverified; consequently it is not yet an admissible performance validation.

## Features and pipelines

The 17-feature allowlist is: `sales_channel`, `payment_mode`, `discount_pct`, `qty`, `promised_delivery_days`, `delivery_pincode`, `has_default_pincode`, `is_gift`, `order_hour`, `order_day_of_week`, `order_day`, `order_month`, `is_weekend`, `sku`, `product_family`, `product_model_name`, `product_age_days`.

The two historical customer counters are now excluded. `order_value_inr`, catalog price, and other monetary fields are excluded because of the unresolved October x100 scale. `source`, service/pickup events, notes, raw IDs, and unverified customer snapshots are also not predictors.

`src/modeling.py` supplies point-in-time-safe preprocessing for the
17-feature allowlist and three derived interactions: channel × payment mode,
discount × quantity, and promised delivery days × weekend. The fitted
pipelines keep preprocessing inside each model/CV fold. Monetary fields,
customer counters, source, pickup/service fields, identifiers, and post-outcome
data remain excluded.

## Exploratory results

The authorized exploratory run completed. Its generated experiment files
were removed during submission cleanup; the observed fixed-threshold summary
is retained here:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8862 | 0.6000 | 0.0367 | 0.0692 | 0.6862 | 0.2419 |
| Random Forest | 0.6750 | 0.1954 | 0.5837 | 0.2927 | 0.6798 | 0.2224 |
| Sigmoid-calibrated Logistic Regression | 0.8848 | 0.5000 | 0.0122 | 0.0239 | 0.6953 | 0.2455 |
| Constant training-prevalence baseline | 0.8848 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1152 |

These are observed calculations on the fixed holdout, **not estimates of
mature-label or business performance**. The baseline's 0.8848 accuracy and the
very low Logistic Regression recall demonstrate why accuracy alone is
misleading here. The 0.5 threshold was fixed, not selected for operational
use. No 95% accuracy is claimed.

The original run also evaluated confusion matrices, Brier score, log loss,
calibration, the constant training-prevalence baseline, and descriptive
subgroup slices. Its generated tables and plots are not included in the
cleaned submission.

The original run isolated fitted estimators and metadata under
`derived_data/exploratory/` and reports under `reports/exploratory/`; those
generated files were later pruned. No production `models/` directory was
created. CatBoost, XGBoost, and LightGBM were not installed and were not run
in the initial Phase 3 readiness run.

Any reported metrics describe only model fit to unverified labels. They are
not admissible estimates of mature-label performance and must not drive
deployment, operational intervention, customer/financial decisions, or a
business threshold. No evidence for 95% accuracy is claimed.

## Advanced model improvement experiment

The prior model-improvement report was removed during submission cleanup; its
key findings are summarized here. A seeded two-fold expanding
`TimeSeriesSplit` randomized search scored average precision over the latest
3,000 chronological rows of the training partition, then refit selected
candidate configurations on all 8,378 pre-validation rows. The fixed
April–June holdout was not used for hyperparameter fitting, but it was used to
compare model families and choose a threshold.

Among the tuned candidates, class-weighted Logistic Regression had the highest
accuracy under the explicitly analytical 20% recall floor: **86.45% accuracy,
35.17% precision, 20.82% recall** at threshold 0.709749. This is below the
88.48% majority-class baseline accuracy. The highest accuracy found without
the recall floor was 88.76%, with just 2.45% recall, so it is not meaningful
return detection. The 95% target was **not achieved**.

The selected result is optimistic because multiple candidates and thresholds
were compared on the same holdout. The report labels this selection bias and
the unresolved label maturity; it is not production evidence. CatBoost
1.2.10, XGBoost 3.4.1, and LightGBM 4.7.0 are now pinned in `requirements.txt`.

Calibration has been evaluated using sigmoid calibration with expanding
chronological folds fit on pre-validation training data. The April–June block
was then used for model/threshold comparison and is no longer an untouched
out-of-time assessment. The July–September test snapshot has no target column
and cannot be scored. A future evaluation requires a newer mature labeled
cohort whose outcomes remain hidden until the model, calibration, and
threshold are frozen.

## Blocking risks and approvals needed

1. **Label maturity (blocking):** provide delivery dates, actual extraction cutoff, target window, and finalization rules. Until then, the dataset cannot be considered training-safe.
2. **Payment units (blocking for monetary features):** reconcile October with Finance/gateway ledger and approve any correction. Current policy excludes affected monetary fields.
3. **Product/feature availability:** confirm all remaining checkout fields and product reference values were available at dispatch; product terms have no effective dates.
4. **Migration timestamps:** clarify `source` semantics and which event timestamps were stored in UTC and left unconverted.
5. **Model generalization:** obtain a later untouched labeled cohort after label-maturity rules are verified; the current holdout was used for model and threshold selection.
6. **Business action:** clarify call/hold costs and intervention effects before using predictions to hold orders or choosing an operational threshold.

After these blockers are resolved, re-derive maturity eligibility and feature
availability, re-run chronological validation, and obtain approval before
production training or prediction generation. The exploratory run does not
change these requirements.

## Historical calibrated 55/45 blend comparison — superseded by V2

The figures in this archived comparison describe the earlier 55/45 candidate,
not the current model or dashboard KPIs. Use the pre-dispatch V2 summary at the
top of this report for current results.

The bounded run in `src/train_calibrated_ensemble.py` saved its artifact,
metrics, and feature explanations under
`derived_data/calibrated_ensemble_final/`; archived model comparison metrics
remain in that directory's `metrics.json`.
LightGBM remains comparison-only. The final score is
`0.55 × calibrated Logistic Regression + 0.45 × calibrated CatBoost`.
The run reported that both user-specified reference gates were exceeded
(ROC-AUC > 0.7057 and PR-AUC > 0.2584). The earlier output was
`submission.csv`; the final hand-in is `predictions.csv`.
This does not promote the model or alter `reports/final_training_gate.md`.

Q2 has 8,378 pre-April training rows and 2,126 holdout rows (245 positives,
1,881 negatives; 11.52% positive). It was reused by earlier experiments, so
these comparisons are exploratory and selection-optimistic.

| Model / reference | ROC-AUC | PR-AUC | Brier | Top-15% precision / recall / F1 | Top-15% expected net Rs | Top-20% precision / recall / F1 | Top-20% expected net Rs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated Logistic Regression | 0.7541 | 0.3679 | 0.0920 | 0.307 / 0.400 / 0.348 | 11,667 | 0.275 / 0.478 / 0.349 | 11,941 |
| Calibrated CatBoost | 0.7490 | 0.3524 | 0.0900 | 0.326 / 0.424 / 0.369 | 19,994 | 0.282 / 0.490 / 0.358 | 21,052 |
| LightGBM comparison-only | 0.7342 | 0.3184 | 0.0919 | 0.298 / 0.388 / 0.337 | 24,851 | 0.268 / 0.465 / 0.340 | 27,938 |
| **55/45 calibrated LR/CatBoost blend** | **0.7544** | **0.3658** | **0.0908** | **0.326 / 0.424 / 0.369** | **15,301** | **0.279 / 0.486 / 0.355** | **15,885** |
| Saved Logistic Regression baseline | 0.7057 | 0.2584 | 0.2167 | 0.257 / 0.335 / 0.291 | 77,627 | 0.235 / 0.408 / 0.298 | 99,136 |
| User-stated reference | 0.7057 | 0.2584 | n/a | n/a | n/a | n/a | n/a |
| Majority-class baseline | 0.5000 | 0.1152 | n/a | 0 / 0 / 0 | n/a | 0 / 0 / 0 | n/a |

Expected net savings use `sum(predicted_probability × 0.35 × Rs 1,150) -
Rs 45 × flagged_count`; they are probability-based policy arithmetic, not
observed or causal savings. The majority-class accuracy is 88.48% but its
recall is zero, so it is not meaningful return detection. At top-15% capacity,
the blend's 42.45% recall is 2.83 times the 15% random-selection reference;
the previously mentioned 3.2x description is not supported by this run. The
saved baseline's Brier score is 0.2167, so its probability-based savings
estimate is especially poorly supported and should not be compared as if it
were a calibrated policy forecast.

### Decile calibration check

This is a descriptive reliability check on the reused holdout, not independent
calibration evidence.

| Decile, low to high | Rows | Mean predicted probability | Observed return rate |
|---:|---:|---:|---:|
| 1 | 213 | 0.023 | 0.019 |
| 2 | 213 | 0.032 | 0.019 |
| 3 | 212 | 0.041 | 0.057 |
| 4 | 213 | 0.050 | 0.066 |
| 5 | 212 | 0.059 | 0.061 |
| 6 | 213 | 0.071 | 0.108 |
| 7 | 212 | 0.085 | 0.118 |
| 8 | 213 | 0.104 | 0.146 |
| 9 | 212 | 0.139 | 0.175 |
| 10 | 213 | 0.270 | 0.385 |

Observed return rates rise overall with score, but differ materially from
predicted values in several deciles, particularly the top decile. Calibration
should not be assumed to transfer to Q3.

### Failure and cohort slices

| Slice | Rows | Positive labels | Positive rate | ROC-AUC | PR-AUC | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Post-dispatch evidence present | 893 | 244 | 27.32% | 0.713 | 0.528 | Diagnostic slice only; pickup/service events are not predictors and can reveal outcome/lifecycle differences. |
| Zero-pincode walk-ins | 171 | 21 | 12.28% | 0.748 | 0.426 | Small slice; sentinel pincode denotes missing location detail. |
| Pre-October-2025 migration | 0 | 0 | n/a | n/a | n/a | No such orders occur in Q2; comparison unavailable. |
| Post-October-2025 migration | 2,126 | 245 | 11.52% | 0.754 | 0.366 | Entire holdout is post-boundary; cannot isolate a migration effect. |
| Current Shield members | 478 | 98 | 20.50% | 0.767 | 0.511 | Current-only membership reference; historical coverage/effectivity is unknown. |
| Current non-members | 1,648 | 147 | 8.92% | 0.757 | 0.334 | Current-only reference; not a verified historical cohort. |

### Feature and operating constraints

The benchmark excludes `pickup_scheduled_at`, `last_service_event_type`,
`source`, and raw `order_value_inr`. Current unversioned `signup_date` and
catalog prices are not used; therefore customer tenure, retroactive-signup
flag, and catalog net value are excluded. Supplied prior-order/return counts
are used as stated pre-order counts, but their as-of construction still
requires source verification. Fold-local family discount averages and OOF
calibration avoid fitting those transforms to Q2. The risk tiers use
training-side OOF score tertiles and are relative bands, not decision
thresholds.

Q2 is only chronologically adjacent to the unlabeled Q3 test with a matching
schema; it does not establish that Q2 mirrors Q3 or guarantee future
performance. No claim that 95% accuracy is mathematically impossible is
appropriate; the evidence simply does not support it, and leakage/majority
class behavior can make accuracy misleading. Continue to prohibit automatic
holds and preserve the production gate until mature labels, point-in-time
counter evidence, a later untouched cohort, and a controlled call-pilot
evaluation are available.

