# Kestrel Home Returns Prediction — Initial Data Audit

**Audit date:** 2026-11-26  
**Scope:** Read-only inspection of the supplied `datasets/` files. No datasets were changed; no predictions, model, API, or dashboard were created.

## Executive summary

The labeled data contains 11,155 rows but only 10,504 distinct orders because 651 partner-feed imports duplicate CRM rows. Every duplicate pair has the same order fields and target; only `source` differs. After retaining one record per order, the observed target prevalence is 1,200/10,504 (11.42%).

Do not train on the raw rows or use `last_service_event_type` / `pickup_scheduled_at`: the partner copies overweight 651 orders, and the service/pickup fields are captured after dispatch and strongly reveal return processing. In addition, all 748 October 2025 order amounts are exactly 100 times the catalog-price-after-discount calculation, coinciding with the stated payment-gateway change. The amount field should not be used until Finance/Data confirms the intended unit and a correction is approved.

The supplied customer and product references cover every customer and SKU in train and test. However, the customer reference is not demonstrably an as-of-order snapshot: 1,999 distinct labeled orders have a customer signup date after the order date. The supplied documents also conflict about the meaning of `source`, so it cannot safely be treated as migration/system provenance.

No model metrics have been measured. In particular, there is no evidence for the requested 95% accuracy. The unlabeled file is a later chronological snapshot (July–September 2026); it is not a validation set and has no labels.

## Files inspected

All eight files in `datasets/` were inspected:

- `README.txt` describes the tables, field meanings, and states that test is the warehouse snapshot at dispatch.
- `ops-policy.pdf` (v4.1) documents costs, Shield, returns processing, the October 2025 system migration, and a legacy UTC timestamp caveat.
- `email-thread.txt` documents the partner-feed re-import issue, the new October payment gateway, the Rs 600 vs Rs 1,150 cost discrepancy, and the request to hold risky orders.
- All five CSVs: `train.csv`, `test_unlabelled.csv`, `sample_submission.csv`, `customers.csv`, and `products.csv`.

The PDF extraction exposes all text on its single page. Some section-number glyphs did not extract cleanly; the policy content and figures below are transcribed from the extracted text.

## CSV dimensions, schemas, types, and missing values

Types below are the types inferred by pandas from the supplied CSVs. Identifiers were read as strings; timestamp fields are stored as strings except the all-empty test pickup column, which pandas infers as `float64`.

| File | Rows × columns | Schema (type; missing count) |
|---|---:|---|
| `train.csv` | 11,155 × 19 | `order_id` (string; 0), `order_placed_at` (string; 0), `customer_id` (string; 0), `sku` (string; 0), `sales_channel` (string; 0), `payment_mode` (string; 0), `discount_pct` (int64; 0), `qty` (int64; 0), `order_value_inr` (float64; 0), `promised_delivery_days` (int64; 0), `delivery_pincode` (int64; 0), `is_gift` (string; 0), `customer_prior_orders` (int64; 0), `customer_prior_returns` (int64; 0), `delivery_note` (string; 2,752), `last_service_event_type` (string; 0), `pickup_scheduled_at` (string; 9,855), `source` (string; 0), `returned` (int64; 0) |
| `test_unlabelled.csv` | 2,096 × 18 | Same feature columns as train, with no `returned`. `delivery_note` has 538 missing. `pickup_scheduled_at` is empty for all 2,096 rows and inferred as `float64`; all other fields have no missing values. |
| `sample_submission.csv` | 2,096 × 2 | `order_id` (string; 0), `score` (float64; 0) |
| `customers.csv` | 9,000 × 5 | `customer_id`, `city`, `state`, `signup_date`, `shield_member` (all strings; 0 missing each) |
| `products.csv` | 21 × 6 | `sku`, `family`, `model_name`, `launch_date` (strings; 0 missing each); `list_price_inr`, `warranty_months` (int64; 0 missing each) |

There are no fully identical rows within any CSV. Keys are unique within the reference files: 9,000 customer IDs and 21 SKUs. `order_id` is unique in test and sample submission, but not in train (details below).

## Target, class distribution, and submission contract

- **Target:** `returned`, present only in train; values are binary `0`/`1`.
- Raw train rows: 9,888 non-returns and 1,267 returns (11.36% positive).
- After order-level deduplication: 9,304 non-returns and 1,200 returns (11.42% positive).
- The all-negative accuracy baseline is therefore about 88.6% on this data. It is a prevalence baseline, not a trained model result; accuracy alone would hide missed returns.
- There are 2,096 test orders and no labels for them.
- `sample_submission.csv` contains exactly `order_id,score`, has 2,096 unique IDs, and its IDs match test in both set and row order. Every supplied sample `score` is `0.5`; this is a placeholder, not a prediction or evidence about the scoring semantics.
- The expected output shape is therefore one row per test order with its original `order_id` and a numeric `score`. A probability-like score is the reasonable interpretation, but the evaluator's required range/meaning and whether rows must retain input order should be confirmed before implementation.

## Dates and chronology

| Dataset/field | Observed range |
|---|---|
| Train `order_placed_at` | 2025-04-01 00:22 through 2026-06-30 23:08 |
| Test `order_placed_at` | 2026-07-01 00:51 through 2026-09-30 23:46 |
| Train `pickup_scheduled_at` (non-null) | 2025-04-09 20:26 through 2026-07-15 14:00 |
| Customer `signup_date` | 2022-10-15 through 2026-08-14 |
| Product `launch_date` | 2023-05-27 through 2025-01-28 |

Order and reference dates parse without failures. The order and pickup timestamp fields are naive (no timezone offset in the CSV). Policy v4.1 says legacy Zoho resolution-event timestamps were stored in UTC and were not converted during migration; it does not identify which supplied timestamp fields are affected or provide row-level timezone provenance. Do not compare or transform those event timestamps as local time until Data/IT clarifies the affected fields, source records, and conversion rule.

There is a material documentation conflict: the README and email define `source` as `crm` or `partner_feed`, while the policy says `source` identifies the legacy Zoho vs new CRM system. In the actual data, the only values are `crm` and `partner_feed`; `source` cannot be used to identify migration-era system provenance. The policy states operations migrated on 2025-10-01, which is a useful time boundary for temporal evaluation, but it does not resolve row-level provenance.

## Duplicate partner-feed orders

- Train has 11,155 rows, 10,504 unique `order_id`s, and 651 duplicate order-ID groups (1,302 rows in those groups).
- Each duplicate group has exactly two rows: one `crm`, one `partner_feed`.
- All non-source fields, including `returned`, are identical within every pair. There are no conflicting labels or other field discrepancies in these pairs.
- The duplicate pairs account for 584 duplicated non-return and 67 duplicated return examples.
- `test_unlabelled.csv` and `sample_submission.csv` have no duplicate order IDs.

**Recommended handling:** establish one canonical row per `order_id` before splitting or calculating any metrics. For this extract, keeping the CRM row is supported by the observed exact-match pairs and the email's description of partner-feed re-imports; preserve a duplicate audit log and assert all other fields/labels agree. Do not average or vote targets. Quarantine and investigate any future conflicting duplicates rather than silently selecting or aggregating them. Splitting duplicate copies across train and validation would leak the same order into both sets.

## Leakage and point-in-time feature safety

The prediction is required **before dispatch**, while several fields describe events after the order was placed or after a return began:

1. **`pickup_scheduled_at` — exclude.** It is non-null on 1,300 train rows; every populated timestamp is later than `order_placed_at`. Of those rows, 1,191 have `returned=1`. The policy says pickup is booked after a return is approved, so this is a post-outcome / near-direct leakage field. The test snapshot has no pickup timestamps, creating an additional train/serve mismatch.
2. **`last_service_event_type` — exclude unless a point-in-time reconstruction proves dispatch-time availability.** Train has 789 `REVERSE_PICKUP` rows, all with `returned=1` and none with `returned=0`. `INSTALL_DONE`, `DEMO_DONE`, and `TECH_VISIT` can also occur after dispatch; the extract gives no event timestamp. Test contains 543 `INSTALL_BOOKED` values, a category not present in train, alongside `NONE`. This is both a leakage risk and a train/test distribution mismatch.
3. **`customer_prior_orders` and `customer_prior_returns` — conditional.** The README calls these counts prior to the order, making them potential safe features, but their as-of construction is not documented. Verify that only information available before each order was used; do not recompute them from all-time data.
4. **Customer reference features — conditional.** Every train/test customer key joins, but `signup_date` is later than `order_placed_at` on 1,999 of 10,504 deduplicated labeled orders. The `shield_member` field may also be current rather than historical membership status. Do not use signup-derived tenure or Shield status as point-in-time features until their effective dates are established. If safe, derive tenure only for valid as-of records.
5. **`order_value_inr` — quarantine pending correction.** October 2025 values have a systematic x100 inconsistency (see below); raw values can let a model exploit a month/system artifact rather than return risk.
6. **`source` — do not use as migration feature.** Its documented meaning is contradictory and it is also the field that distinguishes the duplicate feed copies.
7. **`delivery_note` — avoid raw text in the first model.** Notes include gate/access details, contact instructions, and delivery-location details. This is sensitive operational text, is not required for a baseline, and must not be copied to unapproved systems. If later considered, first establish a privacy-approved transformation and verify point-in-time availability.
8. **`order_id` — identifier only.** Exclude it from predictors.

Potential dispatch-time predictors, subject to point-in-time validation, include order date/time-derived calendar features, channel/payment mode, discount, quantity, promised delivery days, an authoritative normalized order amount, an explicit “default pincode/no address” indicator, safe customer attributes, and product attributes. Pincode itself is a numeric-looking identifier, not a continuous measurement; treat geography deliberately and keep the `000000` sentinel distinct from real locations.

## Payment values, migration, and other data quality

### October 2025 payment-value anomaly

For each train row, the order value was compared with `products.list_price_inr × qty × (1 − discount_pct/100)`. Outside October 2025, this calculation matches the supplied `order_value_inr` to floating-point precision. In October 2025, **all 748 rows** are exactly 100 times that expected amount, including repeated partner copies. The observed October maximum is Rs 5,880,204 despite the catalog-based expected amount being 100 times smaller. This matches the email's warning about the new festive-season payment gateway and overlaps the 2025-10-01 migration boundary.

The test values do not show this x100 scale (test maximum Rs 57,020.16), but the test period is later and cannot establish the historical correction. Confirm whether October values are paise, a gateway unit conversion, or another issue with Finance/Data. Do not silently divide by 100 or replace the field using current catalog prices. After correction, validate the fix against authoritative transaction records and document treatment of historical price changes before using amount features.

### Other checks

- `delivery_pincode=0` is present on 889 train rows and 176 test rows. The README defines this as a system default when no delivery address was captured for walk-in partner orders, not a real postcode.
- Train feature ranges: `discount_pct` 0–60, `qty` 1–2, `promised_delivery_days` 1–12, `customer_prior_orders` 0–10, and `customer_prior_returns` 0–6. No row has prior returns greater than prior orders.
- The customer table contains 9,000 unique customer references and the product table 21 unique SKU references. Every train and test customer/SKU joins successfully.
- Customer reference has 7,001 `N` and 1,999 `Y` Shield records. Product reference has 7 appliance families with 3 SKUs per family.
- Missing values are concentrated in `delivery_note` and `pickup_scheduled_at` as quantified above. No imputation behavior should convert absence of pickup into a safe signal without first resolving the feature's leakage and serving-time availability.
- The data does not specify the target's outcome window, extraction/as-of date, or whether recent non-return orders have completed their full opportunity to return. Confirm label maturity/right-censoring, particularly for the latest training orders, before treating `returned=0` as a final negative. Shield returns are free within 30 days under the policy, reinforcing the need to define the observation window.

## Available and missing reference data

**Available:** customer city, state, signup date, and Shield flag; product family, model name, list price, warranty months, and launch date. Keys join completely to the order files. These are not necessarily historical snapshots, and the customer date/status caveats above must be resolved.

**Not supplied:** point-in-time customer/Shield history, authoritative order/payment transaction ledger or gateway conversion metadata, service-event timestamps and provenance, dispatch/fulfilment scan history, return reasons and final outcome timestamps, confirmation-call assignment/completion/outcome data, hold/cancellation outcomes and their costs, order margin/refund economics, and inventory/availability snapshots. These are needed for trustworthy temporal features and to estimate the causal value/cost of interventions, not merely predictive risk.

## Model recommendations

This is binary classification with modest sample size, about 11.4% positive prevalence after deduplication, heterogeneous categorical inputs, and evident temporal/system drift.

1. **Logistic Regression — baseline.** One-hot encode low-cardinality categories, scale numeric features, regularize strongly, and use probability calibration if needed. It gives a transparent benchmark and helps reveal whether complexity is justified.
2. **CatBoost — preferred nonlinear candidate.** Native categorical handling and missing-value support suit this tabular dataset. Keep all features point-in-time safe, tune with chronological folds, and constrain complexity due to the dataset size.
3. **LightGBM — benchmark candidate.** Gradient-boosted trees can capture interactions efficiently; use conservative depth/leaves and regularization, and encode categories consistently without target leakage.
4. **XGBoost — benchmark candidate.** A strong boosted-tree comparison; use explicit category encoding and regularization appropriate to the modest data volume.

Compare all candidates on the same deduplicated, leakage-screened chronological folds. Select by out-of-time PR-AUC/calibration and validated operational value, not training fit or accuracy alone. The target imbalance does not by itself justify class weighting; if weighting or resampling is tested, evaluate and recalibrate probabilities on untouched chronological data.

## Chronological validation plan

1. Resolve duplicate orders and target maturity, then create features strictly from information available at or before dispatch.
2. Use `order_placed_at` as the split key, not random stratification. The test orders are later than every training order and must remain untouched until the approach and threshold are locked.
3. Use expanding-window backtests over the labeled period. A suitable initial structure is train on April–September 2025 and validate October–December 2025; expand training through December and validate January–March 2026; expand through March and validate April–June 2026. Report each window separately and overall. The first fold crosses the stated migration boundary and is especially useful for diagnosing operational drift, but October amounts must first be corrected or excluded.
4. Keep each `order_id` wholly within one partition. Derive encodings, aggregations, feature selection, imputation, model selection, and calibration using training data within each fold only. Historical orders from the same customer may precede that customer's later validation order, as in production, but no future customer outcomes may flow backward into features.
5. Reserve the latest labeled block as the final out-of-time model/threshold assessment. Inspect monthly and pre/post-migration performance and calibration. Only after these checks should a final model be refit on all eligible historical labels and score the unlabeled July–September 2026 snapshot. The test file cannot tune or validate because its labels are absent.

The exact fold lengths may be adjusted after checking matured-label counts and class counts per block; do not move future rows into earlier training windows to force balance.

## Evaluation and threshold policy

Report, at a minimum, **accuracy, precision, recall, F1, ROC-AUC, PR-AUC**, and probability calibration (Brier score/log loss plus reliability plot; calibration slope/intercept if sample size supports it). Include confusion matrices, positive prevalence, and fold-level uncertainty. PR-AUC and precision/recall are important at this class balance; ROC-AUC alone does not establish operational usefulness. Accuracy must be compared with the approximately 88.6% always-negative baseline and never represented as evidence of useful return capture. No performance metrics have been computed in this audit.

Do not choose a threshold using the final test labels. Generate out-of-fold or strictly later chronological validation probabilities; fit any calibrator on a separate chronological calibration slice or training-only out-of-fold predictions. Choose the threshold on validation data using a pre-agreed objective (expected incremental cost/benefit, intervention capacity, or an explicitly selected precision/recall constraint). Lock the model, calibration, and threshold before any one-time final test scoring. Do not repeatedly inspect test outcomes or tune against them.

### Operational economics

- Policy v4.1 puts the average return handling cost at **Rs 1,150**, in addition to the refund; this supersedes the email's informal Rs 600 figure for decision analysis.
- A completed pre-dispatch confirmation call costs **Rs 45**. The policy pilot reports that calls prevented about **35% of otherwise occurring returns among called orders**. Under that estimated effect, and considering only return-handling savings, expected avoided cost per called order is `predicted return probability × 0.35 × Rs 1,150`. The arithmetic break-even risk is about **11.2%** (`45 / (0.35 × 1,150)`), before call completion, customer experience, or other costs. Treat this as a starting economic calculation, not a validated universal cutoff: pilot effect may not generalize, and use a controlled prospective pilot to estimate incremental lift.
- A hold lasting over 24 hours leads to customer cancellation about **12% of the time**, but no per-cancellation cost, lost-margin estimate, hold effect on returns, or customer lifetime-value impact is supplied. Therefore, a defensible hold threshold cannot yet be calculated. Obtain Finance/Operations' incremental cancellation and delay costs and measure hold effects before automating holds.
- Shield members are about a fifth of orders and have high lifetime value; returns are free to them within 30 days. Avoid a blanket risk-based hold that disregards customer value or differential economics. Confirm membership as-of dispatch and compare segment-level impact; do not infer that a high predicted return probability automatically means a hold is beneficial.

The operational decision is not merely “predict return.” It is whether a particular action (call, hold, or no action) improves expected value. Risk scores should drive a human-reviewed/piloted policy until the intervention effect and full costs are measured.

## Recommended architecture (for approval; not implemented)

Use an order-level, point-in-time feature pipeline with deterministic deduplication and explicit data-quality assertions; an expanding-window training/evaluation workflow; a regularized Logistic Regression benchmark and CatBoost primary candidate, with LightGBM and XGBoost as controlled comparisons; chronological probability calibration; and a threshold chosen only on matured chronological validation data under the approved intervention economics. Exclude post-dispatch service/pickup fields, raw identifiers, unverified customer snapshots, and uncorrected payment amounts. Keep the final test set sealed until the model and policy are locked. Proceed to implementation only after approval and after the owner confirms the October amount fix, label maturity, source/migration provenance, feature availability, and hold/call economics.
