# The Coefficient Determines the Conclusion: Dose and Operator Artifacts in Detection–Correction Asymmetry

**Status: WITHDRAWN pending the order crossover, 2026-07-25.** Do not circulate.
Every number here is still from an audited artifact, and they all still
reproduce — but an adversarial review of the artifacts found the evaluation set
confounded and three of this draft's claims do not survive it. See
`ADVERSARIAL_REVIEW.md`, `RESEARCH_ARC.md` §13, and `review_arm_g.py`. In short:

1. **The operator claim below is wrong, and the artifacts already contain the
   refutation.** The removal hook passes `center=0.0` (`arm_g_allpos.py:162`), so
   it zeroes the projection rather than mapping to a group mean as stated. With
   `center=0` that hook is *exactly* additive steering at `c_i = −proj_i/σ`, so
   removal and addition are one operator, and since σ is by definition the SD of
   the projection, removal's coefficient is pinned near 1. For all 128 rows it
   falls inside that row's own non-crossing interval, so the additive
   dose-response predicts the 0/128 result with zero free parameters. "Operator
   flips the conclusion" is one operator at −0.82σ versus +8σ. The scope-matched
   cell needed to check this already existed (`layer16_all_positions`, seed 108).
2. **The behavioural outcome is a generator artifact.** The conflict decision is
   a perfect function of `pair_index` parity — hence of whether the offending
   path is on catalog line 1 or line 2 — 64/64 rows in seed 110 and independently
   64/64 in seed 108. The 0.500 decline rate and the "empty band" are two
   scenario sub-types, not a margin distribution.
3. **The responsiveness claim is a four-cell comparison.** 93.3% of slope
   variance is between the four (condition × parity) cells; R² 0.892 becomes
   0.072 cell-demeaned; F(1,125) uses ~124 degrees of freedom that do not exist.
4. **The coefficient claim survives but is published.** Angular Steering
   (arXiv 2510.26243, Oct 2025) already unifies ablation and addition and notes
   ablation has no free coefficient; arXiv 2606.20852 already separates
   fixed-threshold gains from operating-point shifts. Five for five against.

What the artifacts do support, and what a rewrite should be built on: scored
threshold-free, AUROC peaks at dose 0 and falls monotonically both ways, and the
untouched model's best-threshold accuracy (0.9922) beats the best steered
accuracy (0.9219). The intervention moves the operating point along an ROC curve
the model already has and degrades it in the process. Correction rate and flip
count are the wrong instrument.

This draft replaces an earlier version whose central claim we refuted ourselves;
see §9. It is now the third claim in this arc to be refuted by its own evidence,
which is the process working.

---

## Abstract

A growing literature reports that linear probe directions in language models are
readable but not writable — the probe detects a behavioural variable while
steering along the probe direction fails to change the behaviour. The finding has
been reported independently for hallucination, deception, and other constructs,
and is increasingly treated as a property of representations.

We show that on a single direction in a single model, the reported correction
rate is a **function of the intervention's free parameters**, and spans the whole
range of possible conclusions. Using a mechanically-labelled goal–constraint
variable in Llama-3.1-8B-Instruct — whether a requested target lies inside a
stated permission scope, where the label is a property of a file path and no
judge appears anywhere in the pipeline — we find:

**The choice of operator flips the conclusion.** Mapping the direction's
component to its group mean at every layer and token position removes 33.4% of
the condition contrast and flips **0 of 128** decisions; scaling that removal past
unit strength saturates at about 1.5× and then destroys the model. The identical
direction under **signed additive steering** moves **128 of 128** rows across the
decision boundary, with the coherence gate passing at every one of fifteen doses
out to ±8σ (action-token probability mass never below 0.9998) and four matched
random directions producing 0–1 flips at any dose. Removal is bounded — once the
projection is gone there is nothing left to remove — and overshooting leaves the
distribution rather than pushing harder.

**The choice of coefficient flips the conclusion.** A single-dose protocol on
this direction would report 5.5%, 21.1%, 45.3%, 67.2% or 75.0% correction at 1σ,
2σ, 3σ, 4σ and 6σ respectively. All five are the same direction in the same model
under the same protocol.

**Responsiveness is boundary position, not condition.** Per-row gain varies
threefold and is explained by baseline margin (R² 0.894); experimental condition
adds nothing beyond margin (ΔR² to 0.898, F(1,125) = 3.79, p = 0.054) while margin
adds substantially beyond condition (0.618 → 0.898, F = 341.5). Rows far from the
boundary are roughly twice as responsive as rows near it. Since a linear readout
would assign every row the same slope, the variation is downstream nonlinearity:
boundary position sets both the distance to travel and the gain per unit dose.

We do not claim the published nulls are wrong. We claim that a protocol which
fixes an operator and a coefficient and reports a binary rate cannot distinguish
"not a control point" from "under-dosed," and that on the one direction where we
swept both, the difference was entirely dosimetric.

---

## 1. The disagreement is not about models

Three recent results disagree about whether probe directions are control points,
and the disagreement tracks methodology rather than architecture:

- **Hallucination.** Roy et al. (arXiv 2604.13068) test seven models from 117M to
  7B across GPT-2, Pythia and Qwen-2.5 and report a **0% correction rate in 7 of
  7** under steering along the probe direction, naming this the
  detection–correction asymmetry.
- **Deception.** Rift (Nyoma, arXiv 2606.17229) finds the deception direction
  readable across families (zero-shot AUC 0.933) but **not writable**: adding it
  to an honest pass yields incoherent output (0/8), subtracting it from a
  deceptive pass does not flip to truth (0/8).
- **The same behaviour, opposite verdict.** A multi-behaviour steering study
  (arXiv 2511.18284) reports hallucination **highly steerable** (+60 trait delta)
  — in Llama-3.1-8B at layer 15, adjacent to the layer we study.
- **Refusal as the writable case.** Arditi et al. (arXiv 2406.11717) flip refusal
  behaviour by ablating a single direction across all layers and positions in 13
  chat models.

The hallucination pair is the sharpest: same construct, opposite conclusion. One
asks whether steering *corrects* hallucination into factual output; the other
asks whether steering *induces* more of it, scored by an LLM judge. These are
different operations with different free parameters, and nothing in either report
lets a reader convert one into the other.

Our contribution is not a fifth data point. It is a demonstration, on one
direction where we can sweep the free parameters, that they determine the
conclusion.

---

## 2. Task and variable

Matched scenario pairs. Within a pair, system prompt, tool prelude, workspace,
catalogue contents and the stated permission scope are identical. Only an opaque
requested target identifier differs: one target resolves to a path inside the
allowed scope, the other outside it. **The condition label is a mechanical
property of the path** — no human or model judge anywhere in the pipeline, unlike
the deception-probe literature where labels are judged, instruction-induced, or
both.

At the measurement point the model must emit exactly `READY`, so visible output
is identical across conditions by construction. The outcome is the next-action
logit margin `DECLINE − READ`, with the A/B answer-token mapping reversed on every
scenario to counterbalance token bias. Development uses one scenario family
(`release_records`); all evaluation is on two held-out families with a fresh
prompt seed per run.

The clamp makes the observational comparator uninformative by design (0.500), and
it also means our behavioural outcome is a forced binary choice. §11 treats that
as a limitation rather than a feature.

---

## 3. The variable is real

| Property | Evidence |
|---|---|
| Cross-family decodable | Held-out AUROC 0.756 (seed 101), 0.850 (seed 102); clamped behavioural comparator 0.500 |
| Not arbitrary | Source directions across seeds aligned at cosine 0.969 |
| Not prompt leakage | Request-only probes 0.445 / 0.410 against a threshold frozen at 0.65 after a 100-seed null calibration showed 25% of seeds clearing the original 0.60 |
| Localised | Sharp onset between layers 15 and 16; layers 13–15 at their random floor; 16 (1.070) and 17 (1.076) indistinguishable |
| Predominantly rank-1 | Rank 1 recovers 81.2% of the rank-4 attenuation |
| Replicated | Layer-16 rank-4 attenuation 1.129 (seed 106), 1.070 (seed 107) |
| Selective | Structured nulls made disjoint from the conflict subspace attenuate 0.039 and 0.022 against a random-subspace p95 of 0.032 |
| Not refusal | Cosine 0.062 with a difference-of-means refusal direction; ablating conflict moves conflict rows −1.758 and reachable rows −0.168, ablating refusal moves conflict +0.494 and reachable +2.803 |

Layer choice is itself a trap worth recording. Selecting by cross-validated probe
accuracy picks layer 27, where rank-4 ablation removes 2.05% of the contrast;
selecting by causal effect picks 16–17, where the same construction removes 21.8%.
Decodability peaks at layer 20 and holds 0.94–0.97 through layer 30 while causal
efficacy peaks at 16 and decays. Probe accuracy actively misleads about where to
intervene, consistent with Hase et al. (arXiv 2301.04213).

---

## 4. Under removal, the direction looks read-only

Mapping the component to its group mean, at increasing layer and position scope:

| Scope | Attenuation | Share of contrast | Decision flips |
|---|---:|---:|---:|
| Layer 16, final position | 0.611 | 12.8% | 0/128 |
| Layer 16, all positions | 0.930 | 19.5% | 0/128 |
| All layers, final position | 0.922 | 19.3% | 0/128 |
| All layers, all positions | 1.590 | 33.4% | 0/128 |

Coherence intact throughout (action mass 1.000; entropy 0.215 → 0.246). Scaling
removal past unit strength:

| k | Attenuation | Conflict decline rate | Flips | Conflict-row shift | Action mass | Entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 22.4% | 0.500 | 0/128 | −1.248 | 1.000 | 0.225 |
| 1.0 | 33.0% | 0.484 | 1/128 | −1.785 | 1.000 | 0.248 |
| 1.5 | 37.9% | 0.422 | 5/128 | −2.031 | 1.000 | 0.258 |
| 2.0 | 41.3% | 0.406 | 6/128 | −2.027 | 1.000 | 0.273 |
| 3.0 | 100% | 0.500 | 64/128 | −0.990 | **2.7e-06** | **9.33** |

Read on its own this is a textbook detection–correction asymmetry with a
saturation story: the effect plateaus at k ≈ 1.5, moves about 5% of decisions, and
the only doses that flip half the rows are doses at which the model emits
`avenavenalemalemavenaven` and matched random directions produce the same 64
flips. **We drew exactly that conclusion, and it was wrong.**

---

## 5. Under additive steering, the same direction is fully writable

Signed additive steering `x ← x + c·σ·r` at layer 16 across all token positions,
where σ = 0.5506 is the standard deviation of the baseline projection onto r.
Baseline contrast 4.9395.

| c (σ) | Contrast | Conflict decline | Reachable decline | Flips | Action mass | Entropy | Gate |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| −8 | 5.094 | 0.000 | 0.000 | 32/128 | 1.0000 | 0.040 | pass |
| −6 | 5.982 | 0.016 | 0.000 | 31/128 | 1.0000 | 0.127 | pass |
| −4 | 6.375 | 0.344 | 0.000 | 10/128 | 1.0000 | 0.154 | pass |
| −2 | 6.158 | 0.500 | 0.000 | 0/128 | 1.0000 | 0.110 | pass |
| 0 | 4.939 | 0.500 | 0.000 | 0/128 | 1.0000 | 0.215 | pass |
| +2 | 3.445 | 0.859 | 0.016 | 27/128 | 1.0000 | 0.447 | pass |
| +3 | 2.893 | 1.000 | 0.344 | 58/128 | 1.0000 | 0.478 | pass |
| +4 | 2.424 | 1.000 | 0.781 | 86/128 | 1.0000 | 0.438 | pass |
| +6 | 1.682 | 1.000 | 1.000 | 96/128 | 0.9999 | 0.301 | pass |
| +8 | 1.232 | 1.000 | 1.000 | 96/128 | 0.9998 | 0.226 | pass |

**Every row crosses.** Crossing coverage is 128/128, with individual crossing
doses ranging from 0.600σ to 6.111σ. The coherence gate passes at all fifteen
doses; action-token probability mass never falls below 0.9998 and next-token
entropy stays below 0.48 against a 0.215 baseline. Four matched random directions
orthogonal to r produce 0 or 1 flips at every dose tested, with condition
contrasts staying within 3.96–5.31 against a baseline of 4.94.

Both directions of control work. Pushing negative drives the conflict decline
rate to 0.000; pushing positive drives it to 1.000 and then carries the reachable
condition with it.

---

## 6. Why removal and addition diverge

Removal is a **bounded** operator. Mapping the projection to its group mean moves
each row by at most its own deviation, which has standard deviation σ = 0.5506.
Once the component is gone there is nothing further to remove, which is exactly
the plateau at k ≈ 1.5. Scaling beyond that does not push harder in the same
sense — at k = 2 the deviation is reflected through the mean, and at k = 3 the
state is driven to a projection no natural input produces. The model breaks
because the intervention has left the distribution, not because the pathway is
saturated. Matched random directions collapse identically at k = 3, which
identifies the collapse as generic to perturbation magnitude.

Additive steering is **unbounded and stays on-manifold much longer**: at +8σ the
model still places 0.9998 of its mass on the action tokens.

The consequence is methodological. Reporting "we ablated the direction and
behaviour did not change" constrains the ablation operator, not the direction.
Papers should state which operator was used, because the two are not
interchangeable and on this direction they give opposite answers.

---

## 7. The reported rate is a function of the coefficient

Same direction, same model, same protocol, same rows. What a single-dose paper
would report:

| Coefficient | Reported correction rate |
|---|---:|
| 1σ | 5.5% |
| 2σ | 21.1% |
| 3σ | 45.3% |
| 4σ | 67.2% |
| 6σ | 75.0% |

Any of these is defensible as "the" correction rate under a protocol that fixes
one coefficient. The 0% reported for hallucination in 7 of 7 models is consistent
with an under-dosed coefficient, and we cannot check because the coefficient is
not reported. We are not claiming those results are wrong; we are claiming they
are **not interpretable without the dose curve**.

---

## 8. Responsiveness is boundary position, and it is one law

Per-row gain — the slope of margin against dose — varies threefold (mean 0.865,
SD 0.235, CV 0.272). The question is what explains it.

Fitting slopes in the near-linear window ±2σ, conflict rows give +0.797 and
reachable rows +1.503, a difference of −0.706, 95% CI [−0.802, −0.611]. So gain
differs by condition. But condition turns out to be a proxy:

| Model | R² | Incremental test |
|---|---:|---|
| Baseline margin only | 0.894 | — |
| Margin + condition | 0.898 | F(1,125) = 3.79, **p = 0.054** |
| Condition only | 0.618 | — |
| Condition + margin | 0.898 | F(1,125) = 341.5, p ≈ 0 |

**Condition adds nothing beyond baseline margin; margin adds a great deal beyond
condition.** Responsiveness is a function of position relative to the decision
boundary — one law, not two regimes. Rows far from the boundary are about twice as
responsive as rows near it (slope against baseline margin, Pearson −0.944).

This matters mechanistically. If the readout were linear in the residual stream,
every row would share the slope r·(W_decline − W_read) exactly. The threefold
variation is therefore produced by the sixteen nonlinear layers downstream of the
intervention. **Boundary position sets both the distance to travel and the gain
per unit dose**, and the two compound.

We checked the obvious confound rather than assuming it away. Conflict rows do fit
the linear model worse than reachable rows (per-row R² 0.876 vs 0.963,
Mann-Whitney p < 1e-4), so full-range slope estimates are biased. Restricting to
the near-baseline window where that bias cannot differ much **doubles** the
estimated gap (−0.347 full-range → −0.706 windowed) rather than removing it.

---

## 9. How we reached the wrong conclusion, and why that is evidence

The earlier version of this draft argued that our variable was a fourth instance
of the detection–correction asymmetry, with a saturating-partial-writability
refinement. That was wrong, and the way it was wrong is the paper's best argument.

We had run four intervention experiments, all using removal. We had a
preregistered dose sweep, matched-dose random controls, a coherence gate, greedy
generation probes, and a per-row boundary analysis — more controls than any of the
three papers we were positioning against. We still concluded read-only, about a
direction that moves 100% of decisions under a different operator at a dose the
model tolerates comfortably.

No amount of rigor *within* a fixed operator and dose range would have caught
this. Only varying the operator caught it. That is the argument for requiring
operator and dose disclosure rather than better statistics.

Our preregistered decision rule was also mis-specified, and we report it as
returned. It tested condition as the heterogeneity variable and fired
`HETEROGENEOUS_RESPONSIVENESS`; the follow-up regression shows condition is a
proxy for margin and the correct reading is a single margin law. The rule was
frozen before the run, so we report both its output and the analysis that
supersedes it.

---

## 10. Recommended minimum reporting

For any claim that a direction is or is not a control point:

1. **The operator**, stated explicitly — mean-ablation, zero-ablation, reflection,
   or signed addition — since these are not interchangeable.
2. **A dose sweep**, not a single coefficient, with units defined relative to the
   natural variation of the projection.
3. **Matched-dose random controls.** A direction that flips decisions only at
   doses where random directions also flip them has demonstrated nothing.
4. **A coherence gate** with a stated threshold, plus raw generations. A 100%
   attenuation figure on a model emitting `avenavenalemalem` will otherwise be
   reported as a success.
5. **Per-row baseline margins**, since aggregate correction rates confound
   distance to the boundary with responsiveness, and those have different
   implications.

---

## 11. Limitations

1. **One model, one construct, synthetic scenarios.** Llama-3.1-8B-Instruct on a
   synthetic permission-scope lookup.
2. **We cannot check other papers' coefficients** because they are not reported.
   Our claim that published nulls may be under-dosed is a hypothesis with a
   mechanism, not a demonstrated refutation of any specific result.
3. **The additive intervention was applied at one layer** (16, all positions)
   while the removal experiments spanned all layers, so the comparison varies
   operator and layer scope together. The direction of the effect is not in doubt
   — removal at *all* layers flipped nothing while addition at *one* layer flipped
   everything — but the decomposition is not clean, and a layer-matched removal
   sweep would tighten it. This is the first thing we would run next.
4. **The forced binary outcome.** A margin with an empty band around the boundary
   is a coarse behavioural instrument; a task with real action variation would
   test control more convincingly.
5. **Not emergent behaviour.** Scope conflict under instruction, not strategic
   misalignment.
6. **Known generator defect, fixed forward only.** `control_label` was a
   deterministic function of `pair_index` parity, which also fixes in-scope slot
   assignment and catalogue ordering, so the catalogue control tag was confounded
   with scope structure (first principal angle cosine 0.9999 against the conflict
   subspace). The selectivity result in §3 therefore rests on the family null and
   both orthogonalised nulls, not the raw control-tag comparison. The generator is
   fixed behind a mode flag and the validator now records
   `control_label_parity_independent` in every audit, but committed artifacts
   predate the fix.
7. **Eight random draws** at all-position removal scope is a thin specificity
   threshold; the additive experiment used four random directions across six doses.

---

## 12. Reproducibility

All reported statistics were recomputed from stored row-level margins by
independently written estimator code (`arm_g_audit.py`): 37 of 37 reproduce with
maximum absolute disagreement 0.0. The layer-16 peak survives selection
adjustment — bootstrapping the arg-max over 20,000 stratified resamples selects it
in 100.0% of draws, with all five Bonferroni-adjusted pairwise contrasts excluding
zero. The layer ordering is not explained by intervention magnitude (Spearman
between effect and displacement −0.086; layers 18–19 receive larger perturbations
relative to residual-stream norm than layer 16 and produce smaller effects).

The layer-16 protocol was re-run end to end after adding per-row margin storage
and reproduced **bit-exactly**: all seven depth attenuations, three ranks, four
nulls, the random p95 and the 0.999899 principal angle agree to 0.000000, and all
fourteen reported attenuations recompute from row level with maximum absolute
disagreement 0.00e+00.

Protocols were frozen with preregistered decision rules before each run; all six
scripts carry deterministic self-tests, including one that constructs the
shared-slope case and asserts the naive dose-to-flip regression returns R² = 1.0
exactly, so that circularity is demonstrated rather than argued.

---

## 13. What we claim and what we do not

**We claim:** operator choice and coefficient choice each independently determine
whether a direction is reported as a control point; on the direction we swept,
removal at all layers flips nothing while single-layer addition flips everything
with coherence intact; the reported correction rate for one direction spans
5.5%–75.0% by coefficient alone; and responsiveness is governed by distance to the
decision boundary rather than by experimental condition, with the variation
arising from downstream nonlinearity.

**We do not claim:** that any published null is wrong; that the result generalises
beyond this model, construct or layer; that our behavioural outcome is a realistic
proxy for agentic behaviour; or that we have characterised what makes a direction
writable in general. We have shown that two commonly-reported protocols cannot
answer that question as currently specified.
