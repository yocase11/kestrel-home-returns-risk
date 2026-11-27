# Phase 2 — Data Preparation Decisions

These are the Phase 2 and Phase 3 decisions applied by `src/data_processing.py` and `src/modeling.py`. Data and unfitted pipeline preparation are complete, but the dataset is **not cleared for model training** while label maturity and other owner confirmations remain unresolved.

## Applied decisions

| Area | Decision | Status / rationale |
|---|---|---|
| Duplicate order IDs | For exact two-row CRM/partner-feed pairs, preserve the CRM row and copy the full discarded partner-feed row to `reports/duplicate_audit.csv`. | Applied to 651 groups. All compared non-source fields, including target, matched; 0 conflicting fields. Unexpected pairing or any mismatch raises an error and prevents canonical output. |
| Payment scaling | Do not divide October values or reconstruct them from catalog price. Exclude both `order_value_inr` and `list_price_inr` from prepared features. | Applied as a safe exclusion. Root cause is unresolved: the observed exact factor of 100 cannot distinguish a paise/minor-unit conversion from another gateway/export defect without an authoritative ledger. |
| Post-dispatch/service data | Exclude `pickup_scheduled_at` and `last_service_event_type`. | Applied to train and test. Both fields may encode return/service events not available at dispatch; test also has a different event-category distribution. |
| Other unverified or sensitive fields | Exclude `source`, `delivery_note`, current customer-reference attributes, `warranty_months`, raw `customer_id`, and raw `order_id` from model features. | Applied. `order_id` remains only as an output/join key; customer ID is used only to validate reference coverage. |
| Customer history | Exclude `customer_prior_orders` and `customer_prior_returns` from primary model features. | Applied conservatively because their as-of construction cannot be independently verified. They remain present only in original source files, not generated model tables. |
| Product reference | Join SKU, family, model name, and launch date; derive `product_age_days` from launch date and order timestamp. | Applied. All matching launches precede their orders. Price and warranty are excluded due to the price anomaly and lack of effective-date history. |
| Postal sentinel | Serialize pincode as a six-character category and add `has_default_pincode`. | Applied to preserve `000000`; read the processed CSV's `delivery_pincode` column as string. |
| Label maturity | Preserve supplied `returned` values and add `label_maturity_status=unverified` to every train row. Do not reinterpret, drop, or certify any negative. | Applied. Target window and extract cutoff are missing; recent June negatives need particular review. Training remains blocked until the owner defines maturity. |
| Timestamp handling | Parse documented IST order timestamps as naive local time for order-calendar features. Do not transform pickup/service timestamps. | Applied. The legacy UTC caveat does not identify affected rows/columns, and `source` provenance is contradictory. |

## Prepared dataset contract

Generated files:

- `derived_data/train_modeling.csv`: one canonical row per labeled order; `order_id` and `order_placed_at` metadata; the 17 model features below; source `returned`; and `label_maturity_status`.
- `derived_data/test_modeling.csv`: one row per unlabeled order in original test order; same metadata and 17 feature columns; no target or maturity value.
- `reports/duplicate_audit.csv`: complete discarded partner-feed rows plus `canonical_source`, `discard_reason`, and `conflicting_fields` audit fields.

The explicit feature allowlist in `src/data_processing.py` is:

`sales_channel`, `payment_mode`, `discount_pct`, `qty`, `promised_delivery_days`, `delivery_pincode`, `has_default_pincode`, `is_gift`, `order_hour`, `order_day_of_week`, `order_day`, `order_month`, `is_weekend`, `sku`, `product_family`, `product_model_name`, `product_age_days`.

The train table has 10,504 rows (9,304 negative, 1,200 positive); test has 2,096 rows. All prepared feature cells are populated, product references join, and the sample-submission order IDs match test order and row order. The processing script refuses to overwrite existing generated files unless `--overwrite` is explicitly supplied. It does not modify anything inside `datasets/`.

`label_maturity_status` is review metadata, not a feature. `order_id` and `order_placed_at` are metadata, not features. Consumers should select model inputs using `FEATURE_COLUMNS`, not feed every train CSV column into a model.

## Maturity experiment and split

The order records provide neither delivery date nor label extraction cutoff. A true 7-, 10-, or 30-day observation eligibility calculation is impossible. The sensitivity table uses an explicitly non-authoritative proxy: latest training order calendar date (2026-06-30) minus each window, with orders dated on/before that date counted as candidates. Candidate counts are 10,282 for 10 days, 10,345 for 7 days, and 9,805 for 30 days. Each window has **0 verified mature rows** because the required delivery and cutoff evidence is absent. All raw labels remain `unverified`; candidate counts must not be used to claim maturity.

After the regulatory/platform review, a separate one-sided return-request
evidence audit was generated at
`derived_data/return_request_label_audit.csv`. It records 1,301 positive
proxies and 9,203 unknown outcomes; it assigns no negatives and is not used as
a model target. The missing event timestamps and label cutoff still block
binary training.

The fixed chronological split is training before 2026-04-01 and validation from 2026-04-01 onward. It produces 8,378 train and 2,126 validation rows, with order IDs unique in each partition and no overlap. Both partitions remain subject to the label-maturity blocker.

## Model and evaluation preparation

`src/modeling.py` provides Logistic Regression and CatBoost pipeline factories from the 17-feature allowlist and sigmoid calibration with expanding `TimeSeriesSplit` folds. Later authorized exploratory runs fit isolated pipelines and evaluated them on the April-June block; that block was reused for candidate, calibration, and threshold comparisons and is no longer untouched. Metrics included accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, Brier score, log loss, calibration reliability bins, and expected calibration error. The July-September test snapshot is unlabeled. No production model or predictions were generated; label maturity remains unverified.

## Unresolved gates

1. Finance/Data must verify the October unit against the gateway and ledger, then approve any value correction.
2. Data/IT must provide the label extraction cutoff and return-finalization window so immature negatives can be identified without guessing.
3. Data/IT must clarify the meanings of `source`, legacy-vs-CRM provenance, and which legacy timestamps require UTC conversion.
4. Product/Customer owners must confirm if customer and product reference fields have historical effective dates before those attributes are considered for later model versions.
5. CatBoost is not installed in the current Python environment; install the approved dependency before constructing/training that pipeline.

At the time these Phase 2/3 preparation decisions were recorded, no model was
trained. Subsequent exploratory-only experiment findings are summarized in
`reports/model_readiness_report.md`; generated experiment folders were pruned
during submission cleanup. These findings do not change the unresolved
maturity or independent-validation gates.

