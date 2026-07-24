# Mathematical audit — Arm G cross-layer mediation (seed 106)

Run `python3 arm_g_audit.py`. Machine-readable output in `arm_g_audit.json`.

No model is re-run. Every number below is recomputed from the 128 row-level
margins stored in the run artifact, using estimator code written independently
of the code under audit. Agreement is therefore evidence; disagreement would
have been a defect.

## Verdict

The result survives every check. Four things were open to challenge before this
audit — estimand mismatch, bootstrap dependence, the selection of layer 16 as
the peak, and the possibility that the layer ordering merely tracked
intervention size. All four are now closed. Two genuine limitations remain, one
of which is newly identified here and is cheap to fix on the next run.

## 1. The point estimate and the interval measure the same thing

The reported attenuation is a grand-mean condition contrast. The confidence
interval bootstraps a family-stratified mean of per-pair contrasts. Those are
different expressions, and they coincide only under exact cell balance.

The design is exactly balanced: 8 family×mapping×condition cells, 16 rows each.
Under that balance the two statistics are algebraically identical, and the
recomputation confirms it — the maximum absolute gap across all 13 margin
vectors is **exactly 0.0**.

So the interval is an interval for the quantity actually reported. This was
worth checking rather than assuming; had any cell been unbalanced, the CI would
have been centered on a different estimand than the headline.

## 2. Every reported statistic reproduces

37 reported statistics — baseline contrast, and the contrast, attenuation and
attenuation fraction for all six single-layer and six cumulative conditions —
recomputed from raw rows. **Maximum absolute disagreement: 0.0.**

Exact rather than near-exact agreement is expected here and is itself a check:
bf16 logits are dyadic rationals, so these sums are exact in float64.

## 3. The conclusion does not depend on the bootstrap

Full six-layer cascade attenuation, point estimate 1.334:

| Method | 95% interval |
|---|---|
| Percentile bootstrap (as reported) | [1.131, 1.529] |
| Basic (reverse-percentile) bootstrap | [1.139, 1.537] |
| Normal approximation, bootstrap SE | [1.133, 1.535] |
| Paired *t* on pair-level values | [1.112, 1.556] |

32 of 32 pairs positive. Sign test *p* = 4.7e-10, Wilcoxon *p* = 7.9e-7,
sign-flip permutation *p* < 5e-5.

Preregistered increment over the anchor layer, point estimate 1.234: percentile
[1.025, 1.434], basic [1.035, 1.443], normal [1.032, 1.437], paired *t* [1.015,
1.454]; 31 of 32 pairs positive, sign test *p* = 1.5e-8.

All four interval constructions exclude zero for both quantities, and the
distribution-free tests agree. The reported intervals reproduce to Monte Carlo
noise.

The headline share carried no interval in the run artifact. Bootstrapping
numerator and denominator jointly on the same pairs gives **27.5%, 95% CI
[26.0%, 29.3%]**.

## 4. The layer-16 peak survives selection adjustment

This was the main statistical exposure. A confidence interval computed at the
arg-max of six layers is not honest about the selection that produced it.

Bootstrapping the arg-max directly over 20,000 stratified resamples: layer 16
was selected in **100.0%** of resamples. No other layer was ever the maximum.

Direct pairwise contrasts against layer 16, with Bonferroni-adjusted intervals
for the five comparisons:

| Contrast | Estimate | 95% CI | Bonferroni 99% CI | Pairs positive |
|---|---:|---|---|---:|
| 16 − 12 | 1.135 | [0.961, 1.305] | [0.902, 1.355] | 32/32 |
| 16 − 20 | 0.871 | [0.707, 1.033] | [0.660, 1.084] | 32/32 |
| 16 − 24 | 0.895 | [0.734, 1.053] | [0.688, 1.100] | 32/32 |
| 16 − 27 | 1.029 | [0.850, 1.207] | [0.791, 1.262] | 31/32 |
| 16 − 30 | 1.070 | [0.896, 1.240] | [0.838, 1.289] | 32/32 |

Every adjusted interval excludes zero. The closest competitor is layer 20 and
it is still 0.87 logits behind with a Bonferroni interval bounded away from
zero. "Layer 16 is the peak" is not an artifact of picking the maximum.

## 5. The effect does not track how hard each layer was hit

The obvious deflationary explanation is that layer 16 simply received the
largest intervention. It did not — it received one of the smallest.

| Layer | Attenuation | Mean displacement | Attenuation per unit | Rank-1 share of displacement |
|---:|---:|---:|---:|---:|
| 12 | −0.006 | 0.048 | −0.121 | 35.4% |
| 16 | 1.129 | 0.549 | **2.055** | 77.6% |
| 20 | 0.258 | 0.814 | 0.317 | 78.8% |
| 24 | 0.234 | 0.788 | 0.297 | 70.3% |
| 27 | 0.100 | 0.931 | 0.107 | 78.6% |
| 30 | 0.059 | 1.106 | 0.053 | 80.6% |

Effect ordering is `16, 20, 24, 27, 30, 12`; perturbation-size ordering is
`30, 27, 20, 24, 16, 12`. Spearman correlation between effect and intervention
size is **−0.086** — no relationship, and if anything inverted. Layer 16 is
19 times more efficient per unit of displacement than layer 27.

**The limitation this analysis does not close.** Displacement is measured in
absolute subspace coordinates. Residual-stream norms grow with depth in Llama,
so 0.549 at layer 16 may be a *larger fraction* of the local state norm than
1.106 at layer 30. This audit therefore rules out absolute perturbation size as
the driver, but not relative perturbation size. The artifact does not record
state norms. Recording per-layer residual-stream norms in the next run closes
this completely and costs one line.

Separately: the displacement is ~78% rank-1 at every active layer, layer 16
included. That predicts a rank-1 ablation at layer 16 would recover most of the
effect, which is a cheap and decisive next test.

## 6. The intervention does not change any decision

The ledger records that the ablation rescales the margin without flipping
decisions. Quantified:

| Condition | AUROC before → after | Spearman vs baseline | Sign flips |
|---|---|---:|---:|
| Layer 16 alone | 0.998 → 0.998 | 0.988 | **0 / 128** |
| Full cascade | 0.998 → 0.997 | 0.983 | **0 / 128** |

Not one row of 128 changed which action it favored, in either condition. The
contrast shrinks mostly by the reachable condition moving up (+0.955 under the
full cascade) rather than the conflict condition collapsing (−0.379).

This is the sharpest honest limit on the result and it is stronger than "AUROC
barely moved."

## 7. The ablation operator is what it claims to be

Verified numerically on random bases at 1e-15 tolerance:

- `BBᵀ` is idempotent (5.6e-17) and symmetric (0.0) — a genuine orthogonal projector.
- After ablation the subspace coordinate equals the group mean exactly (5.0e-15).
- The displacement lies entirely inside the target subspace (1.8e-15).
- **Group means are preserved** (1.3e-15). The intervention removes within-group variance along the subspace without shifting the group mean, so it is not a bias injection.
- The operator is idempotent, both with fixed centers (1.6e-15) and with centers recomputed from already-ablated states (2.0e-15).
- **Cascade consistency** (5.8e-15): the state reaching a downstream layer is unaffected by whether that layer's own ablation is scheduled, so centers measured on a capture pass remain exactly valid on the scoring pass. The sequential design depends on this.

The audit script itself initially failed the idempotence check at 3.199. That
was a bug in the audit — it applied the operator to the original state rather
than to the operator's own output — not a defect in the pipeline.

## Remaining limitations

1. **Relative perturbation size is not controlled** (section 5). Fix: record
   per-layer residual-stream norms next run.
2. **All intervals are conditional on two families.** The stratified bootstrap
   resamples pairs within family and treats the two family effects as fixed;
   with two families the between-family component cannot be bootstrapped at
   all. Generalization rests on both families showing the effect independently
   — data_checksums 1.512 [1.191, 1.824], incident_times 1.156 [0.898, 1.395],
   16/16 pairs positive in each — not on the width of the pooled interval.
3. **The rank-4 basis is not statistically well-determined.** It is estimated
   from 32 paired differences in 4,096 dimensions. This does not threaten the
   causal test, which is out-of-sample on a fresh seed and would show nothing
   if the directions were noise. It does mean the four directions should not be
   interpreted individually as a meaningful subspace decomposition.
4. **The random-subspace control is drawn from the orthogonal complement** of
   the target subspace, not from all of 4,096-space. This is the more stringent
   choice and is noted for precision, not as a concern.
