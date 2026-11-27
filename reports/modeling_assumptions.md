# Modeling Assumptions and Experimental Protocol

## Status and scope

This document records the exploratory protocol authorized by the user. It does
not certify target labels or choose a production decision threshold. Any
exploratory metrics are non-business evidence because label maturity remains
unverified. Production training and prediction generation remain blocked.

Original files in `datasets/` are read-only inputs. Prepared files are regenerated under `derived_data/`; the primary feature list is defined by `src/data_processing.py` and consumed by `src/modeling.py`.

## Primary model features

The allowlist contains exactly:

`sales_channel`, `payment_mode`, `discount_pct`, `qty`, `promised_delivery_days`, `delivery_pincode`, `has_default_pincode`, `is_gift`, `order_hour`, `order_day_of_week`, `order_day`, `order_month`, `is_weekend`, `sku`, `product_family`, `product_model_name`, `product_age_days`.

Not features:

- `customer_prior_orders` and `customer_prior_returns` (excluded by the Phase 3 conservative policy).
- `order_value_inr`, catalog `list_price_inr`, and any affected monetary field (October 2025 is exactly 100× the catalog/discount comparison; no correction is authorized).
- `source`, `pickup_scheduled_at`, `last_service_event_type`, `delivery_note`, current customer-reference attributes, `warranty_months`, raw customer/order identifiers.
- `order_placed_at` itself is metadata for chronological features and data splitting; derived hour, weekday, day-of-month, month, and weekend indicator are the model features. No timestamp transformation is applied to undocumented legacy service events.

Product family/model/SKU and launch date are from the supplied product lookup. Launch dates precede the observed orders. `product_age_days` is derived using the order timestamp and launch date. Product prices and warranty duration are not used. All customer features are omitted: the available customer table has no effective dates and the signup date is later than the order date for 1,999 canonical train orders.

The choice to use only an explicit allowlist is a safeguard, not proof that each allowed field is perfectly clean. In particular, customer counts are intentionally excluded because their construction is unverifiable; no labels or customer-history aggregates are used to construct features.

## Observation-window experiments and label maturity

Requested experimental windows are:

| Experiment | Window |
|---|---:|
| Primary | 10 calendar days |
| Sensitivity | 7 calendar days |
| Sensitivity | 30 calendar days |

No delivery date appears in the supplied order schemas, and no extraction/as-of cutoff appears in the files or supplied documents. The latest training `order_placed_at` date is 2026-06-30; this is **not** known to be the label extraction cutoff. Therefore, actual delivery-date-based eligibility and maturity cannot be determined. **Verified eligible/mature rows are 0 under each window; the maturity of all 10,504 observed labels remains unknown.**

For comparison only, the readiness report gives **proxy candidate counts** using the last training order calendar date (2026-06-30) as an assumed cutoff and order date as a proxy for delivery date:

| Window | Candidate rows on proxy | Proxy-excluded rows | Candidate positives | Candidate negatives | Verified mature |
|---:|---:|---:|---:|---:|---:|
| 10 days | 10,282 | 222 | 1,175 | 9,107 | 0 |
| 7 days | 10,345 | 159 | 1,182 | 9,163 | 0 |
| 30 days | 9,805 | 699 | 1,123 | 8,682 | 0 |

These counts must not be used as if the labels were mature. They show only how an order-date proxy changes cohort size. A valid eligibility calculation requires each order's delivery date and the label extraction cutoff, plus an agreed definition of the return outcome window. The policy's 30-day Shield entitlement is not a general target-label contract.

The follow-up policy review found no universal statutory return window or
machine-learning maturity period that fills these gaps. Kestrel's policy
defines a return as starting when the customer raises it, but a later pickup
can be cancelled; the export does not identify which event `returned`
represents. See
[label_maturity_regulatory_review.md](./label_maturity_regulatory_review.md)
for the proposed event-level label contract and evidence request.

## Chronological split

The fixed, reproducible provisional split is:

- **Train:** `order_placed_at < 2026-04-01` (8,378 rows; 955 positive and 7,423 negative).
- **Validation:** `order_placed_at >= 2026-04-01` through the latest supplied training order (2,126 rows; 245 positive and 1,881 negative).
- **Order IDs shared across partitions:** 0.
- **Temporal ordering:** latest train timestamp precedes earliest validation timestamp.

The validation period is the latest quarter of labeled orders. Each order ID is validated unique before splitting; no copies can cross the split. The split is deterministic and is implemented by `chronological_split` in `src/modeling.py`. Validation is provisional only: labels have not been established as mature, so this split is not yet a safe performance estimate.

The future unlabeled July–September 2026 orders have not been used for model
selection, threshold selection, calibration, or validation; their supplied
schema has no target column, so they cannot provide performance evidence.
April–June 2026 was initially reserved as the chronological validation block,
but subsequent experiments used it repeatedly for candidate and threshold
selection. It is no longer an untouched holdout.

## Training pipelines and exploratory isolation

### Logistic Regression

`build_logistic_pipeline()` returns an unfitted scikit-learn pipeline:

- Explicit allowlist feature selection.
- Categorical: most-frequent imputation then one-hot encoding with unknown categories ignored.
- Numeric: median imputation then standardization.
- Regularized `LogisticRegression`, `class_weight=None`, fixed random seed, and no fitting during pipeline construction.

### CatBoost

`build_catboost_pipeline()` returns an unfitted CatBoost classifier pipeline with explicit categorical feature names, conservative initial depth/learning rate, log-loss objective, and fixed seed. The local environment does not currently have CatBoost installed; factory invocation raises a clear `ImportError`. Do not install or fit until the dependency is approved.

The subsequent model-improvement experiment compared CatBoost, XGBoost,
LightGBM, Extra Trees, Random Forest, HistGradientBoosting, Gradient Boosting,
and class-weighted Logistic Regression. It also loaded the previously saved
Logistic Regression, Random Forest, and calibrated Logistic Regression
artifacts for a direct baseline comparison. The previous threshold-0.5
majority-class baseline was retained.

Hyperparameter search used seeded two-fold expanding `TimeSeriesSplit` over
the latest 3,000 chronological rows in the training partition, scoring average
precision with two randomized configurations per model family. The selected
configuration for each new candidate was refit on all 8,378 pre-validation
rows. The April–June 2026 block was not used for preprocessing or
hyperparameter fitting, but was used for model comparison and threshold
selection; subsequent feature, ensemble, and calibration experiments reused
it. Resulting selected metrics are subject to validation-selection optimism.

Three derived interactions were added from allowed order-time fields:
`channel_payment_mode`, `discount_qty_interaction`, and
`promised_days_weekend_interaction`. Any estimator must receive only
`FEATURE_COLUMNS` plus these deterministic interactions; metadata, label
maturity status, targets, IDs, and timestamps must not enter the predictor
matrix.
Exploratory artifacts, if regenerated, belong only under
`derived_data/exploratory/model_improvement/`, with metrics and plots under
`reports/model_improvement/`; previously generated files were pruned during
submission cleanup. Do not copy these artifacts to `models/`; the default
prediction generator continues to require the separate `APPROVED` gate. The
later V1 assignment permits only the explicitly named
`--exploratory-submission` path, which writes assignment scores from the saved
operating-point model and does not change production approval.

## Calibration and evaluation protocol

Calibration uses sigmoid/Platt calibration over three expanding
`TimeSeriesSplit` folds. Training rows are sorted by `order_placed_at` before
fitting. Calibration is fit on pre-validation training folds only; however,
the April–June block was used to compare calibrated models and choose
thresholds. It is therefore not an untouched calibration/threshold holdout.
Due to unverified labels and repeated selection, these measurements are
descriptive exploratory evidence only, not estimates of mature-label
performance.

The prepared evaluation helper calculates:

- Accuracy, precision, recall, specificity, balanced accuracy, F1, ROC-AUC,
  PR-AUC, and a fixed-label `[0, 1]` confusion matrix.
- Brier score and log loss.
- Uniform-bin reliability values and expected calibration error.
- Subgroup summaries by sales channel, payment mode, product family, gift status, default-pincode status, and order month. Groups smaller than 50 are omitted; groups with one observed class are surfaced with metrics marked undefined rather than assigned a misleading ROC-AUC.

Metric helpers require probabilities in `[0, 1]` and both labels for ROC-AUC/PR-AUC. Any generated metric values are explicitly exploratory and must be considered
against the validation class prevalence. They are not approved evidence of
performance on mature labels. Accuracy must never be represented as 95%
without valid supporting evidence.

## Threshold and business-use assumptions

No production/business threshold is approved. For model comparison only, the
experiment selected the validation threshold that maximized accuracy subject
to recall of at least 20%; this is a transparent research guard, not a Kestrel
policy. The same validation block was used for threshold selection and final
comparison, so results are optimistic. The unlabeled final test set was not
read or used. Probability calibration is needed before treating a score as
expected risk.

The operational policy lists Rs 1,150 average handling cost per return and Rs 45 per completed confirmation call; it estimates calls prevented 35% of otherwise occurring returns among called orders. Those figures support a hypothesis for a call pilot, not a universal model threshold. Holds incur cancellations about 12% of the time after more than 24 hours, but no cancellation cost or causal hold effect is supplied. The economic case for automated holds remains unresolved. Shield membership and differential customer value also lack as-of features, so no segment treatment is encoded.

## Assumptions requiring confirmation

1. Delivery date, actual label extract cutoff, target definition, and label finalization rules.
2. Whether order timestamps and categorical checkout features represent the warehouse's dispatch-time snapshot.
3. Root cause and approved treatment of October 2025 monetary scale.
4. Migration/source provenance and which legacy event timestamps are unconverted UTC.
5. Generalization evidence from a later, untouched labeled cohort after label
   maturity has been verified. The supplied July–September 2026 test file is
   unlabeled. CatBoost 1.2.10, XGBoost 3.4.1, and LightGBM 4.7.0 are pinned in
   `requirements.txt` for reproducibility.
6. Intervention effect, hold/call costs, refund/margin effects, and customer/Shield fairness requirements before a threshold is used operationally.

