# Label Maturity and E-commerce Policy Review

**Reviewed:** 2026-11-27  
**Conclusion:** The supplied `returned` labels cannot be independently verified as mature from the available files. Consumer regulation and other marketplaces' policies do not supply the missing Kestrel row-level evidence. The production training gate remains **BLOCKED**.

## What this review establishes

The Consumer Protection (E-Commerce) Rules, 2020 (G.S.R. 462(E), 23 July
2020) regulate marketplace/seller conduct and disclosure of return, refund,
exchange, warranty, delivery, and related terms. Rule 6(3) also restricts
sellers from refusing a take-back or refund in specified cases such as
defective, deficient, spurious, misdescribed, or late-delivered goods, subject
to the rule's stated exception.

These are consumer-protection and policy-disclosure requirements. They do
**not** prescribe a universal 7-, 10-, or 30-day change-of-mind return window
for every product, nor a standard machine-learning label observation or
finalization period. A statutory remedy is not proof that a particular
historical order was delivered, returned, or fully resolved.

As an industry-platform example, Flipkart's published [Order Cancellation and
Return Policy](https://www.flipkart.com/pages/returnpolicy) states that return
terms vary by category and item, and that the product-page policy prevails.
It lists different category windows (including 7- and 10-day examples), often
anchored to delivery or installation. Its current category table places some
small appliances in a 7-day replacement-only category and water purifiers,
fans, and certain other exceptions in a 10-day replacement-only category,
subject to the item listing's terms. The page also distinguishes customer
return eligibility from downstream pickup, seller receipt, replacement, and
refund processing. This is an example of a platform's terms, **not** Kestrel's
policy or a legal standard for Kestrel labels.

The official Gazette/Department of Consumer Affairs source for the Rules was
not retrievable from this environment during the review. The legal reference
above identifies the notification; this summary is not legal advice. Verify
the current consolidated Rules with counsel before relying on them for
compliance decisions.

## Kestrel-specific evidence

The provided order README defines `returned` only as “1 = order was returned”
and does not specify the event, timing, cutoff, or meaning of zero.

Kestrel Operations Policy v4.1 adds useful process context:

- A return **starts when the customer raises it** through the app, call centre,
  or partner outlet.
- The reverse pickup is booked only after approval.
- Pickups can be booked and then cancelled.
- Shield members have free returns within 30 days; this is a Shield-specific
  entitlement, not evidence that every order has a 30-day return window or
  that the export labels use a 30-day observation horizon.

This makes `pickup_scheduled_at` and `REVERSE_PICKUP` unsuitable as ground
truth for a customer-raised return: they occur later in the process and may
not represent a completed return. The supplied records do not show the
customer request timestamp, approval/denial, delivery or installation time,
pickup outcome, refund/return completion, or the label extraction/as-of time.
Consequently, the exported `returned` values cannot be mapped to a specific
auditable lifecycle event.

## Recommended label contract before relabeling

Keep the outcome definition explicit and separate customer intent from
realized cost:

1. **Return-request outcome:** positive when a valid customer return request is
   recorded within the applicable order-specific policy window. This matches
   Kestrel's documented point at which its return process starts.
2. **Completed-return/cost outcome:** a separate positive when the return is
   completed under an agreed operational definition (for example, return
   receipt/QC or refund completion). Use this outcome when estimating costs
   such as the policy's ₹1,150 handling figure; do not assume every request or
   booked pickup incurs that cost.
3. Keep pre-dispatch cancellations, undelivered orders, rejected requests, and
   unresolved cases separate from a mature negative for a delivered-order
   return target. If a binary training table cannot represent them, exclude or
   censor unresolved/ineligible records rather than silently assigning zero.
4. For a negative label, require the actual delivery/installation date, the
   policy version and return window applicable to that order, the label
   snapshot cutoff, and confirmation that the eligibility window and relevant
   return cases have closed. Any additional finalization grace period must be
   set from Kestrel's event-processing evidence and owner-approved; there is
   no universal regulatory grace period to borrow.
5. Apply any Shield-specific terms only to eligible Shield orders. Do not
   extrapolate its 30-day entitlement to other orders.

The target, policy window, and finalization rule must be frozen before
rebuilding labels or evaluating a model. The prior 7-, 10-, and 30-calendar-day
experiments used order-date proxies and remain sensitivity analyses only.

## Verification result and required source data

- A separate, reproducible one-sided audit is written to
  [return_request_label_audit.csv](../derived_data/return_request_label_audit.csv)
  by running
  `.venv\Scripts\python.exe src\data_processing.py --return-request-label-audit-only`.
  Its nullable `return_request_label_proxy` is `1` when the supplied
  `returned=1` flag or approved return-workflow evidence supports a customer
  request; all other rows remain unknown rather than being set to `0`. This is
  an evidence proxy, not a fully verified, policy-window-qualified target:
  request dates and each order's applicable return terms are absent.
- In the canonical 10,504 orders, **1,301** have positive request evidence:
  1,200 supplied `returned=1` rows plus 101 rows with `returned=0` but pickup
  or reverse-pickup evidence. Of those 1,301, 1,167 have both the supplied
  positive flag and workflow evidence; 33 are supported by the source flag
  alone; and 101 are supported by workflow evidence despite a source zero.
  The remaining **9,203 are unknown**, not confirmed negatives. There are no
  verified mature negatives.
- The original `returned` column and its 9,304/1,200 distribution are
  unchanged. `label_maturity_status` remains `unverified`; no source rows are
  altered or discarded.
- **No training or prediction action:** no model was fit and no scores were
  generated. Original source files and existing modeling tables remain
  unchanged; the separate audit does not replace or rewrite their target.

The one-sided audit is useful for surfacing likely target-definition
disagreement, but it is **not trainable as an ordinary binary classifier
target**. It uses approved return-workflow fields only to audit outcome
evidence; those post-dispatch fields remain excluded from prediction features.

To resolve the blocker, obtain a privacy-minimized order-event extract with:
`order_id`, delivered/installed timestamp, return-request timestamp, event
type/status and status-change timestamps, applicable return-policy version
and eligibility window (including Shield eligibility), pickup/receipt outcome,
refund/completion timestamp where relevant, and the label snapshot/as-of
timestamp. Data/Operations must confirm event semantics and how cancelled,
rejected, and unresolved requests are treated.

## Sources

- Consumer Protection (E-Commerce) Rules, 2020, G.S.R. 462(E), 23 July 2020,
  especially Rule 6(3) and seller policy disclosures. See the [official
  Gazette of India portal](https://egazette.gov.in/) and search by notification
  number/date. The direct official document was unavailable from this
  environment; counsel should confirm the current text.
- Flipkart, [Order Cancellation and Return Policy](https://www.flipkart.com/pages/returnpolicy),
  accessed 2026-11-27. Used only as a marketplace-policy example.
- Kestrel `datasets/README.txt`, `datasets/ops-policy.pdf` (v4.1), and
  `datasets/email-thread.txt`.
