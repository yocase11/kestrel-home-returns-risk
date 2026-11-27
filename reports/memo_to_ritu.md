# Memo to Ritu Deshpande

**Subject: Returns-risk candidate — pilot only, no dispatch holds**

Dear Ritu,

The current exploratory candidate is a calibrated blend of Logistic
Regression (86%) and CatBoost (14%). On the April–June 2026 chronological
holdout (2,126 orders, 245 marked returns), it scored **0.7568 ROC-AUC,
0.3736 PR-AUC, and 0.0908 Brier**. At a top-15% review capacity, it captured
**101 returns (41.22% capture, 31.66% precision)**—**2.75×** the 15% capture
expected from random selection. This is retrospective evidence only. Q2 was
reused for model/blend selection, labels are not verified as mature, and Q2
does not guarantee results on the unlabeled test period.

The 95% accuracy promise is not supported by this evidence. Accuracy is not
provably impossible for every pre-dispatch model, but the highest observed
Q2 threshold accuracy was 89.51%; the majority-class rule already reaches
88.48% while finding no returns. On this 2,126-order sample, 95% accuracy
would allow at most 106 errors and, even with no false positives, require
catching at least 139 of 245 returns (56.7% recall). Do not use post-shipping
information to create an inflated score.

## Decision and rupees

Policy v4.1's arithmetic call break-even risk is
`₹45 / (₹1,150 × 35%) = 11.2%`. The 35% call-prevention effect is an
assumption, not a measured intervention. Use the score only to identify
orders for a capped, randomized human-reviewed confirmation-call pilot; track
completed calls, prevented returns, cancellations, customer experience, and
cost against a no-call control. The model does not establish causal savings.

Do not hold dispatch: Policy v4.1 cites a 12% customer-cancellation penalty
for holds exceeding 24 hours. Protect verified Kestrel Shield members from
hard holds. Keep production scoring blocked pending independent mature-label
validation and governance approval.

## Three urgent data fixes

1. **Partner feed:** reconcile the 651 duplicate order-ID groups at source;
   preserve deterministic CRM-first canonicalization and the discard audit
   until re-import behavior is corrected.
2. **Finance/gateway:** investigate the October 2025 100× order-value
   anomaly. Its cause is unresolved; do not divide values by 100 without
   ledger evidence.
3. **Customer snapshots:** provide effective-dated signup and membership
   history. Retroactive signup dates can create impossible tenure values;
   tenure and current-only attributes stay excluded until corrected.

The artifact, API, and screen are local exploratory tools. The production
training gate remains **BLOCKED**.
