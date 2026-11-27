# Phase 2 — Data Quality and Point-in-Time Investigation

**Status:** Phase 2 audit plus Phase 3 conservative feature policy completed; model training and prediction generation were not performed.  
**Inputs:** Supplied CSVs and the supplied README, operations policy v4.1, and email thread.  
**Original files:** No files under `datasets/` were modified.

## Summary

| Check | Finding | Action taken |
|---|---|---|
| Duplicate orders | 651 IDs have a CRM row and a field-for-field identical partner-feed copy; no target or feature conflicts. | Keep CRM only after validating the exact pair; write each discarded original row to `reports/duplicate_audit.csv`. |
| October payment values | 748 raw October 2025 rows (700 after canonicalization) are exactly 100× `list_price_inr × qty × (1 − discount_pct/100)`. | Root cause is not provable from the supplied data. Do not divide or synthesize amounts; omit `order_value_inr` from prepared model features. |
| Post-dispatch leakage | Pickup timestamps and service-event type are recorded after order placement / during later service or return workflows; `REVERSE_PICKUP` is directly associated with returned labels. | Exclude both fields from train and test feature tables. |
| Customer reference timing | All keys join, but signup dates are after order date for 1,999 of 10,504 canonical train orders; membership and location have no effective dates. | Do not use fields from `customers.csv` in the model feature tables. Preserve only order-row historical counts documented as pre-order. |
| Label maturity | Outcome window and extraction/as-of date are absent. June 2026 alone has 699 orders, including 622 supplied negatives, but their maturity cannot be established. | Preserve source labels without relabeling; mark every training row `label_maturity_status=unverified`. Do not train until maturity is confirmed. |
| Source and migration | `source` only contains `crm` / `partner_feed` and is a duplicate-origin field, despite policy wording that it identifies legacy/new system. | Do not use `source` as a feature or migration indicator. Treat 2025-10-01 as a documented boundary, not row-level system provenance. |
| Phase 3 history/amount policy | Customer prior counters and monetary fields are not in the primary-model feature allowlist. | Regenerated derived tables use a conservative 17-feature allowlist; raw data stays unchanged. |

## Duplicate investigation and canonicalization

The raw training file contains 11,155 records and 10,504 unique order IDs. Every one of the 651 duplicate groups has exactly two records: one `crm` and one `partner_feed`. Comparing every column other than `source` found **zero conflicting fields** in all 651 pairs, including the target. Train contains 10,504 CRM rows and 651 partner-feed rows; every partner-feed row is part of one of these duplicate pairs. The duplicates comprise 584 non-return and 67 return order copies.

`src/data_processing.py` applies a deterministic, fail-closed rule:

1. For a repeated `order_id`, compare every raw field except `source`.
2. Require exactly one `crm` and one `partner_feed` row and equality of all other fields.
3. Keep the CRM record and preserve the full discarded partner-feed row and audit metadata in `reports/duplicate_audit.csv`.
4. If any field differs, a source is unexpected, or a duplicate group is not exactly that pair, raise an error before writing outputs. No conflicting row is selected or discarded.

The generated audit has 651 rows, each retaining the original discarded row, with blank `conflicting_fields` because the actual comparisons found none. The retained order-level target counts are 9,304 zeros and 1,200 ones.

## Payment amount and October 2025

The comparison used the supplied product catalog and each row's quantity and discount. Outside October, `order_value_inr` matches the catalog formula to floating-point precision. All 748 raw October rows match a factor of exactly 100; after duplicate removal, this holds for all 700 canonical October orders. This is strong evidence of a unit/scale discontinuity, and the timing matches the email's warning about the festive-season payment gateway and the 2025-10-01 migration.

The files do **not** establish whether the recorded number is paise, a gateway-specific minor-unit amount, a different currency conversion, or another export defect. The catalog is not an authoritative transaction ledger and may not prove the actual checkout price. Therefore:

- No amount was divided by 100 or otherwise rewritten.
- `order_value_inr` and `list_price_inr` are omitted from the prepared model features.
- Finance/Data must reconcile October transactions to the payment gateway and ledger, confirm currency/unit and price/discount semantics, and approve a correction before a monetary feature is introduced.

This resolves the modeling risk by exclusion, not the underlying financial-data root cause.

## Point-in-time and leakage review

The model prediction point is before dispatch. The prepared files contain only fields that are present at order time or documented static product identifiers, plus explicitly marked historical customer counters. These fields are kept separate from row metadata and labels.

### Excluded fields

- `pickup_scheduled_at`: 1,300 populated raw train rows, and all are later than `order_placed_at` under the raw parsed timestamps. The policy says a pickup is booked after a return is approved. The 1,300 include 1,191 rows with `returned=1`. The export gives no timezone provenance for these timestamps; regardless, the business event itself is post-return workflow data. Test has no populated pickup times.
- `last_service_event_type`: includes `REVERSE_PICKUP`, `INSTALL_DONE`, `DEMO_DONE`, and `TECH_VISIT`, with no event timestamps. All 789 raw `REVERSE_PICKUP` rows have `returned=1`. These events cannot be established as available at dispatch. Test additionally contains 543 `INSTALL_BOOKED` values, a category absent from train.
- `delivery_note`: excluded because it can contain access codes, contact instructions, and location details; it is sensitive operational text and unnecessary for this preparation baseline.
- `source`: excluded because its documented meanings conflict and it distinguishes feed copies.
- `order_id` and `customer_id`: retained only as row/join identifiers where applicable, never as model features.
- `order_value_inr` and catalog price: excluded pending the gateway/ledger reconciliation.
- Customer reference `city`, `state`, `signup_date`, and `shield_member`: excluded because there are no effective-date snapshots. A signup date after an order is direct evidence that this reference is not safe to use as an unqualified as-of-order table; the membership flag may also be current state.
- `warranty_months`: excluded because the product file has no validity/effective dates proving the value was in force at each historical order.

### Phase 2 prepared features

Phase 2 generated 17 features, including the two order-row customer counters. These two fields are now excluded by the Phase 3 policy. The regenerated train/test files share this 17-feature primary-model allowlist:

`sales_channel`, `payment_mode`, `discount_pct`, `qty`, `promised_delivery_days`, `delivery_pincode`, `has_default_pincode`, `is_gift`, `order_hour`, `order_day_of_week`, `order_day`, `order_month`, `is_weekend`, `sku`, `product_family`, `product_model_name`, `product_age_days`.

- Order fields are checkout/order fields described in the README. `order_placed_at` is retained as metadata for chronological splitting, not included in `FEATURE_COLUMNS`.
- `customer_prior_orders` and `customer_prior_returns` are excluded from the Phase 3 primary model even though the README calls them pre-order counts. This conservative choice avoids relying on counts whose point-in-time construction cannot be independently verified.
- Product features come from the 21-row SKU catalog. The pipeline uses SKU, family, model name, and age since the fixed launch date. Every product launch date precedes every matching observed order. It does not use catalog list price or warranty term.
- `delivery_pincode` is serialized as a six-character category, preserving `000000` as the default/no-address sentinel; `has_default_pincode` makes that sentinel explicit. Consumers reading CSV should set `delivery_pincode` to string to preserve leading zeroes.

No separate point-in-time customer history, customer membership history, historical price catalog, or dated product-policy history was supplied. The customer key was validated against the reference table but current customer attributes were not joined into model features.

## Timestamp, migration, and source findings

- `order_placed_at` spans 2025-04-01 00:22 to 2026-06-30 23:08 in train and 2026-07-01 00:51 to 2026-09-30 23:46 in test. The README identifies order timestamps as IST. The pipeline parses these as timezone-naive local timestamps and derives hour, weekday, and month features without applying an unverified conversion.
- Non-null pickup timestamps span 2025-04-09 20:26 to 2026-07-15 14:00. They are timezone-naive in the CSV. Policy v4.1 says legacy Zoho resolution events were stored in UTC and were not converted during migration, but the supplied data does not identify which timestamp fields/rows are such events. No correction or time-difference feature was made from them.
- Policy v4.1 sets 2025-10-01 as the Zoho-to-CRM migration boundary. The README/email instead define `source` as `crm` or `partner_feed`; the observed values match the latter definition. Use order date for chronological evaluation, but do not infer system provenance or “correct” timestamps from `source`.
- October's 100× value anomaly starts at the same time as the migration boundary. This temporal association is not enough to assign cause to the migration or to decide a conversion.

## Label maturity

Train labels are present, but neither the return observation window nor the label extraction/as-of date is supplied. The latest labeled order is 2026-06-30; the latest populated pickup timestamp is 2026-07-15. The June 2026 order cohort contains 699 distinct orders (622 supplied `returned=0`, 77 supplied `returned=1`). Those timestamps do not prove that the negative outcomes have completed their opportunity to return: the latest pickup time is not a documented extract cutoff, and the policy's 30-day statement is specifically about Shield return eligibility, not a complete target-labeling contract.

The preliminary regulatory and platform-policy review in [label_maturity_regulatory_review.md](./label_maturity_regulatory_review.md) found no basis to apply a universal e-commerce return window or ML label-maturity period to these rows. Kestrel's policy defines a return as starting when the customer raises it; subsequent pickups can be cancelled. The exported `returned` field is not mapped to a timestamped lifecycle event, so the labels remain unverified.

The separate one-sided request-evidence audit identifies 1,301 supported
positive proxies and 9,203 unknown outcomes; it assigns no negatives. Notably,
101 orders have `returned=0` alongside approved-pickup workflow evidence. This
is consistent with a difference between a customer-request label and a later
completed-return label, but does not prove which event the export's target
represents. The proxy is not used in the model.

Accordingly, all prepared training labels carry `label_maturity_status=unverified`. No source `returned` value or existing modeling target was changed or removed, and no `returned=0` is asserted to be mature. The separate one-sided request-evidence audit described below does not replace that target. Before training, obtain the extraction cutoff, target definition, return window, and finalization rules; then apply an agreed maturity rule. Recent June negatives are a particularly clear cohort requiring confirmation, but the available files cannot prove exactly which individual labels are immature.

## Reference coverage and structural checks

- `customers.csv`: 9,000 unique customer IDs; every train/test customer ID is present. It contains city, state, signup date, and Shield flag but no effective dates.
- `products.csv`: 21 unique SKUs across 7 families; every train/test SKU is present. Launch dates range 2023-05-27 through 2025-01-28 and precede matching orders.
- Train feature schemas match test except for the target. Sample submission IDs match test IDs and row order.
- The default/no-address postcode occurs on 889 raw train rows and 176 test rows; 41 of the raw train occurrences are in duplicate partner-feed rows, leaving 848 canonical training orders with `delivery_pincode=000000`.
- Prepared train: 10,504 unique order rows, target counts 9,304/1,200, no missing prepared features.
- Prepared test: 2,096 unique orders and the same 17 model-input columns, no missing prepared features.
- No model fitting, validation metrics, or predictions were generated.

## Outstanding owner confirmations before training

1. Confirm payment gateway/ledger units, root cause, and approved October correction.
2. Define target observation window, extract cutoff, and label maturity policy.
3. Clarify `source` semantics, migration provenance, affected UTC timestamp fields, and conversion rules.
4. Confirm whether customer reference attributes and product terms have historical effective dates before adding them as features.

## Phase 3 observation-window experiments

The order CSV schemas have no delivery date and no extraction/as-of cutoff. Therefore, actual eligibility under a delivery-date observation window cannot be determined. The following candidate counts are **not mature-label counts**: they apply an explicit sensitivity proxy using the latest training `order_placed_at` calendar date (2026-06-30) as if it were the cutoff and use order date as a proxy for delivery date.

| Experiment window | Candidate rows on proxy | Candidate rows excluded by proxy | Candidate positives / negatives | Verified mature rows |
|---:|---:|---:|---:|---:|
| 10 calendar days (primary) | 10,282 | 222 | 1,175 / 9,107 | 0 |
| 7 calendar days (sensitivity) | 10,345 | 159 | 1,182 / 9,163 | 0 |
| 30 calendar days (sensitivity) | 9,805 | 699 | 1,123 / 8,682 | 0 |

Proxy eligibility is `order_placed_at.date <= 2026-06-30 - window_days`. This is not a valid substitute for actual delivery date plus a verified label-extraction cutoff. The documented 30-day Shield return entitlement is not proof of the target-label observation contract. All 10,504 labels remain unverified; none should be described as mature on this evidence.

## Chronological split and model preparation

The reproducible provisional split is based on order timestamp, with training orders before **2026-04-01** and validation orders on/after that date. It yields 8,378 training rows and 2,126 validation rows, with 0 order IDs shared. All train dates precede all validation dates. This reserves April–June 2026 as a latest-period validation block; validation labels remain provisional because maturity is unverified.

`src/modeling.py` provides unfitted Logistic Regression and CatBoost pipeline factories, chronological-fold sigmoid-calibration preparation, classification/calibration metric calculation, and descriptive subgroup evaluation. No estimator `.fit()` was called. CatBoost is not installed in the current Python environment; its factory fails with an explicit install instruction until the dependency is approved/available.

Required evaluation output is prepared for accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, Brier score, log loss, calibration reliability bins, and expected calibration error. Subgroup summaries are available for channel, payment mode, product family, gift status, default-pincode status, and order month; groups under 50 rows are omitted and single-class groups are marked as having undefined ranking/classification metrics. No metrics from model predictions exist yet.

