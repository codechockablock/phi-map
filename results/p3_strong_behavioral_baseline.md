# P3 addendum — strong same-sample behavioral comparator

**Date:** 2026-07-23  
**Data:** Apollo deception-detection at
`f8ec4010e74927394709dffa22b97bdf8cd5a62f`  
**Encoder:** `all-MiniLM-L6-v2`, raw 384-D sentence embeddings

## Question

After the historical 6-D action-surface encoder failed external validation,
does a stronger fixed-feature behavioral representation recover the proposed
~0.75 in-domain ceiling on the same independently labeled examples?

**Answer: yes, narrowly and with an important qualification.** Raw
sentence-embedding step means reach **0.746 macro AUROC**, 95% bootstrap CI
**[0.696, 0.794]**. This externally validates a strong *within-task
behavioral comparator*. It does not validate the 6-D encoder, a universal
0.75 ceiling, or a shared deception direction.

## Fixed protocol

- Exact 1,443-response sample used by `p3_behavioral_6d.py`; no minimum-step
  selection.
- Apollo's first `graded_deception` item:
  numeric ≤2 / `honest` → 0; numeric ≥6 / `deceptive` → 1.
- Response-only. Split on sentence endings or newlines, retain fragments
  longer than 15 characters, and average normalized MiniLM embeddings.
  One- and two-step responses remain in the analysis.
- Primary readout: deceptive-mean minus honest-mean prototype, five-fold
  stratified out-of-fold evaluation, fixed seed 17.
- Headline: unweighted macro mean across the four task families.
- 5,000 class-stratified bootstrap replicates. The interval is conditional
  on the fixed fold assignment.
- Secondary only: whole-response prototype, supervised logistic ceiling,
  form-only logistic control, linear form-residualization, cross-task
  prototype transfer, and fold-seed sensitivity.

## Results

| task | n | raw step-mean prototype | raw whole-response | step-mean logistic | form-only | form-residualized |
|---|---:|---:|---:|---:|---:|---:|
| roleplaying | 287 | 0.635 [0.564, 0.704] | 0.655 | 0.662 | 0.559 | 0.625 |
| ai_liar | 53 | 0.581 [0.414, 0.741] | 0.517 | 0.579 | 0.660 | 0.535 |
| insider_trading | 171 | 0.784 [0.715, 0.851] | 0.636 | 0.830 | 0.605 | 0.794 |
| sandbagging | 932 | 0.984 [0.978, 0.990] | 0.968 | 0.988 | 0.932 | 0.802 |
| **macro** | **1,443** | **0.746 [0.696, 0.794]** | **0.694** | **0.765** | **0.689** | **0.689** |

Step-mean improves over the whole-response prototype by **+0.052 macro
AUROC [0.022, 0.082]**. The gain is concentrated in insider trading
(+0.148 [0.078, 0.220]); roleplaying is −0.019, AI liar +0.063, and
sandbagging +0.016.

Roleplaying is the only family affected by retaining short responses: 109
have one usable step, 91 have two, and 87 have at least three. The other
three families all meet the historical three-step criterion.

### Fold sensitivity

Seed 17 was fixed to match the 6-D evaluation and gives the primary 0.746.
Across stratified fold seeds 0–9, 11, and 17, macro AUROC averages 0.770
with range **[0.746, 0.788]**. The fixed primary is the minimum of that
secondary range, so the result is not created by a favorable split.
All four label vectors match the earlier frontier-ops caches exactly, and
fresh whole-response embeddings match those caches to maximum absolute
error **4.03e-7**.

### Frozen cross-task transfer

Rows train a prototype on all examples from one task; columns apply it to a
different task without reorientation. Diagonal cells are the in-task
out-of-fold primary.

| train \ evaluate | roleplaying | ai_liar | insider | sandbagging |
|---|---:|---:|---:|---:|
| roleplaying | 0.635 | 0.546 | 0.564 | 0.514 |
| ai_liar | 0.513 | 0.581 | 0.448 | 0.279 |
| insider | 0.526 | 0.471 | 0.784 | 0.388 |
| sandbagging | 0.512 | 0.427 | 0.420 | 0.984 |

Mean off-diagonal transfer AUROC: **0.467**.

## Interpretation

1. **The externally labeled ~0.75 result exists, but it belongs to raw
   supervised step means—not the 6-D encoder.** The frozen 6-D surface scored
   0.509; changing representation and aggregation raises the same-sample
   macro result to 0.746.
2. **This is an in-domain behavioral baseline, not a universal ceiling.**
   Task results range from 0.581 to 0.984, and frozen cross-task transfer is
   at chance. What transfers is the supervised procedure, not a common
   direction.
3. **Form explains substantial but incomplete signal.** Form-only features
   reach 0.689 macro. Sandbagging falls from 0.984 to 0.802 after linear
   form-residualization, while insider trading remains at 0.794. The result
   cannot be described as pure deception content, but neither is it only
   transcript length.
4. **Step aggregation is load-bearing where responses are long.** Its clean
   macro gain and the large insider improvement support per-step encoding as
   the appropriate fixed-feature comparator.
5. **Arm G/P2 should use two external baselines.** Raw step-mean plus an
   in-task supervised readout is the strong matched behavioral comparator;
   the frozen 6-D action surface remains a negative control. For the Φ claim,
   the decisive statistic is internal-minus-external *cross-task transfer*,
   where this baseline remains 0.467.

## Claim correction

The earlier 0.753 in-house number remains invalid as evidence for this
encoder because it used governance verdicts as labels. The present 0.746 is
the independent-label result the project needed, but for a stronger and
properly named representation:

> On Apollo's independently labeled deception rollouts, an in-task
> supervised raw step-mean behavioral monitor reaches approximately 0.75
> macro AUROC, while its frozen task directions do not transfer
> (0.467 mean off-diagonal AUROC).

The `identity > asserted G` metric result is a separate experiment and does
not share the 0.753 label source. Its roleplaying reconstruction uses
Apollo `graded_deception` labels and evaluates both metrics on identical
five-fold splits (original campaign AUROCs 0.588 vs 0.578). It is externally
labeled relative to the governance system, while still depending on
Apollo's grader.

Artifacts: `p3_strong_behavioral_baseline.py` and
`results/p3_strong_behavioral_baseline.json`.
