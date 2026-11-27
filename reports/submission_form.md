# Kestrel Home returns-risk submission form

## What did you build, and what business outcome does it move? State the number and money.

An exploratory 86% calibrated Logistic Regression + 14% CatBoost candidate,
local FastAPI `/predict` service, Streamlit review screen, and a 2,096-row
`predictions.csv`. On the reused Q2 2026 holdout (2,126 rows, 245 returns), it
scored **0.7568 ROC-AUC, 0.3736 PR-AUC, and 0.0908 Brier**. At a top-15%
capacity it captured **101/245 returns (41.22%; 31.66% precision)**, or
**2.75×** the 15% random-capture baseline.

Policy arithmetic break-even is `₹45 / (₹1,150 × 35%) = 11.2%`. The 35%
prevention effect is assumed, not causal evidence. Expected savings are not
claimed as a business outcome; measure them with a randomized call pilot.

## What score do you expect on hidden outcomes, on which metric, and why? How estimated?

The hidden score is unknown. A provisional reference is the Q2 result above,
calculated on the chronological holdout after model/blend selection. Q2 was
reused and therefore selection-optimistic; it does not guarantee performance
on the unlabeled test period. ROC-AUC measures ranking, PR-AUC is informative
at the 11.52% positive rate, and Brier measures probability error. No test
labels were available.

## How do you know it works? Sample size, checks, error rate, wrong cases.

Q2 has 2,126 labeled rows (245 positive). The top-15% slice contains 319
orders: 101 true positives, 218 false positives, and 144 missed returns.
These are retrospective slice counts, not a future error rate. The saved
`predictions.csv` has 2,096 rows, `order_id,score`, exact test order sequence,
and finite, non-constant scores. The final full suite passed 40/40 tests;
results are recorded at verification time. Hard cases include zero-pincode
walk-ins, migration/source differences, post-dispatch events, and unverified
prior-order counter provenance. Label maturity is unverified.

## Did you change, narrow, or push back on the client's ask?

Yes. The requested 95% accuracy is unsupported. It is not mathematically
impossible for every model, but the highest observed Q2 accuracy was 89.51%;
the majority baseline was 88.48% with zero return recall. On Q2, 95% accuracy
permits at most 106 errors and requires at least 139/245 returns caught even
with zero false positives. Post-dispatch fields are excluded. The deliverable
is exploratory decision support, not an automated hold.

## What is wrong with what you are handing us?

- Q2 was reused for selection; label extraction cutoff and label maturity are
  unverified. It is not independent future validation.
- The test set is unlabeled; hidden metrics cannot be known.
- Prior-order counters' as-of construction is not independently verified.
- October 2025 payment scaling remains unresolved; no blanket correction was
  applied. Signup and product references lack effective dates.
- Savings and the assumed 35% call effect are not causal estimates. No live
  pilot, monitoring, or production approval exists.
- The 651 partner-feed duplicate groups and customer snapshot issues require
  upstream fixes.
- Recording, public Drive/GitHub links, actual hours, and exact AI usage
  charges require owner input; they are not fabricated here.

## What did you deliberately leave out, and why?

Post-dispatch fields, corrupted raw order value, raw customer identifiers,
unversioned tenure/catalog values, automatic holds/cancellations, paid APIs,
and invented missing labels. The evidence cannot support those features or
actions safely.

## Anything built or found that nobody asked for?

Chronological fold-local feature/encoding checks, calibration comparisons,
submission alignment checks, a point-in-time audit, and explicit production
gating. The final KPI cards read aggregate `metrics.json` only.

## What AI was used? Cost? What was discarded? Link the recording.

Copilot SDK in VS Code assisted with code, analysis, tests, and report
iteration. No paid inference API is used by the app. Exact assistant usage
charges are not recorded. Alternative model/weight experiments that did not
beat the selected Q2 gates were not promoted.

**Screen recording:** [OWNER INPUT REQUIRED — record up to three minutes and paste the share link].

## Public Google Drive link

**[OWNER INPUT REQUIRED — provide the intended public Drive link].**

## Someone picks this up on Monday: three things they need to know

1. Keep production blocked; do not hold, cancel, or prioritize live orders
   from these scores.
2. The Q2 figures are reused, exploratory results; obtain a mature,
   independent cohort before claiming generalization.
3. Resolve duplicate/source, gateway-scale, and customer-snapshot issues,
   then run a capped randomized call pilot with a no-call control.

## Honest hours spent. One number.

**[OWNER INPUT REQUIRED — enter actual hours; no reliable time log is present].**

## Public GitHub repository URL

**[OWNER INPUT REQUIRED — publish the repository and paste its URL].**

## What does one prediction cost, and what would 700 orders/month cost?

Paid inference API cost is **₹0 per prediction**, so `700 × ₹0 = ₹0` in paid
inference charges. This excludes hosting, compute, maintenance, and staff.
If all 700 orders received a ₹45 call, that would cost
`700 × ₹45 = ₹31,500/month`; the model has not proven resulting savings.
