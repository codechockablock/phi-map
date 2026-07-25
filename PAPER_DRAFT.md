# Partial Writability: Dose Structure in the Detection–Correction Asymmetry

> **CORRECTION PENDING — the central framing of this draft is wrong.**
> Seed 110 (`results/arm_g_boundary_seed110_v1/`) shows the layer-16 conflict
> direction is **fully writable** under signed additive steering: 128 of 128 rows
> cross the decision boundary, the coherence gate passes at every one of 15 doses
> out to ±8σ (action-token mass ≥ 0.9998), and matched random directions produce
> 0–1 flips. The "read-only" reading below is an artifact of the *removal*
> operator, which saturates once the projection is already zeroed and then goes
> off-distribution rather than pushing harder. §4's claim that "the causal pathway
> saturates within the natural range" is an over-reading of an operator
> limitation and must be withdrawn.
>
> What survives, and strengthens: the methodological argument. Same direction,
> same model, same protocol, a single-dose paper would report **5.5%, 21.1%,
> 45.3%, 67.2% or 75.0%** correction depending only on the coefficient chosen.
> That is now a demonstration rather than an inference.
>
> Also revised: responsiveness is not uniform, but condition is not the reason.
> Baseline margin explains it (R² 0.894) and condition adds nothing beyond margin
> (ΔR² to 0.898, F(1,125) = 3.79, p = 0.054) while margin adds hugely beyond
> condition (0.618 → 0.898, F = 341, p ≈ 0). **Boundary position sets both the
> distance to travel and the gain per unit dose** — one law, not two regimes.
> A purely linear readout would give every row the same slope, so the variation
> is downstream nonlinearity.

**Status:** draft, 2026-07-25. Every number is from an audited artifact in
`results/`. Nothing here is submitted.

---

## Abstract

Linear probe directions in language models are increasingly reported to be
readable but not writable: the probe detects a behavioural variable while
steering or ablating along the probe direction fails to change the behaviour.
This has now been observed independently for hallucination, for deception, and
here for a third construct. We argue the phenomenon is real and the standard way
of measuring it is wrong.

We study a mechanically-labelled goal–constraint variable in
Llama-3.1-8B-Instruct: whether a requested target lies inside a stated
permission scope. Labels are a property of a file path, not of a judge. The
direction is validated observationally (held-out AUROC 0.756 and 0.850 on two
independent seeds against a clamped behavioural comparator at 0.500, source
directions aligned at cosine 0.969), localised to a sharp layer-16/17 plateau,
shown to be predominantly rank-1 (81.2% of the rank-4 effect), replicated across
prompt seeds (1.129 and 1.070), shown selective against structured nulls made
disjoint from it (0.039 and 0.022 against a random-subspace floor of 0.032), and
distinguished from a refusal direction by a double dissociation (cosine 0.062,
opposite causal action pattern).

We then attempt to write to it, and fail — but not in the way a binary
correction-rate would report. Ablating the direction from every layer at every
token position removes 33.4% of the condition contrast and flips **0 of 128**
decisions. Scaling removal past unit strength produces a **monotone dose-response
that saturates at about 1.5× full removal**: the conflict-row margin shift goes
−1.248, −1.785, −2.031, −2.027, and the decline rate falls from 0.500 to 0.406,
moving about 5% of decisions while matched-dose random directions hold at exactly
0.500. Beyond saturation the model is destroyed before it can be controlled: at
3× removal the intervention scores as spectacular success — 100% of the contrast
removed, 64 of 128 decisions flipped — on a model emitting
`avenavenalemalemavenaven`, with action-token probability mass 2.7e-06 and
next-token entropy 9.33 against a baseline of 0.209. Matched-dose random
directions collapse identically there.

Two conclusions follow. First, "0% correction" and "saturating partial
writability with a characterised collapse boundary" are different claims, and
single-dose binary reporting cannot distinguish them. Second, scaled
interventions manufacture artifacts that look like successes, and separating
them requires matched-dose random controls plus a coherence gate — neither of
which is standard.

---

## 1. Position

The detection–correction asymmetry has arrived independently from several
directions, which is the main reason to take it seriously:

- **Hallucination.** Roy et al. (arXiv 2604.13068) test seven models from 117M to
  7B across GPT-2, Pythia and Qwen-2.5 and report a 0% correction rate in 7 of 7
  under steering along the probe direction, naming it the detection–correction
  asymmetry.
- **Deception.** Rift (Nyoma, arXiv 2606.17229) finds the deception direction
  linearly readable across families (zero-shot AUC 0.933) but not writable:
  adding it to an honest pass yields incoherent output (0/8), subtracting it from
  a deceptive pass does not flip to truth (0/8).
- **Refusal, as the counterexample.** Arditi et al. (arXiv 2406.11717) *do* flip
  behaviour by ablating a single direction across all layers and positions in 13
  chat models. Some directions are writable.
- **This work.** A fourth construct, mechanically labelled, with the same
  outcome, plus the dose structure the others do not measure.

Independent convergence on three constructs and a dozen models is corroboration.
What is missing is not more existence proofs; it is a measurement that
distinguishes *no control point* from *a control point too weak to cross a
decision boundary*. That distinction is the contribution here.

---

## 2. Task and variable

Matched scenario pairs. Within a pair, system prompt, tool prelude, workspace,
catalogue contents and the stated permission scope are identical. Only an opaque
requested target identifier differs: one target resolves to a path inside the
allowed scope, the other outside it. **The condition label is therefore a
mechanical property of the path, with no human or model judge anywhere in the
pipeline** — a difference from the deception-probe literature, where labels are
judged, induced by instruction, or both.

At the measurement point the model must emit exactly `READY`, so the visible
output is identical across conditions by construction. The outcome variable is
the next-action logit margin `DECLINE − READ`, with the A/B answer-token mapping
reversed on every scenario to counterbalance token bias.

Development uses one scenario family (`release_records`); all evaluation is on
two held-out families (`data_checksums`, `incident_times`) with a fresh prompt
seed per run.

**The clamp is a two-edged design choice, and it is the paper's main structural
limitation.** It makes the observational result clean — the behavioural
comparator is 0.500 because the output carries no information by construction —
and it caps the behavioural ceiling, because a forced binary choice with a wide
margin gap around the boundary cannot show a graded variable doing much. See §6.

---

## 3. The variable is real

| Property | Evidence |
|---|---|
| Cross-family decodable | Held-out AUROC 0.756 (seed 101) and 0.850 (seed 102); behavioural comparator 0.500 |
| Not an arbitrary direction | Source directions across seeds aligned at cosine 0.969 |
| Not prompt leakage | Request-only probes 0.445 / 0.410 against a threshold frozen at 0.65 after a 100-seed null calibration showed 25% of seeds clearing the original 0.60 |
| Localised | Sharp onset between layers 15 and 16; layers 13–15 at their random floor; 16 (1.070) and 17 (1.076) indistinguishable |
| Predominantly one direction | Rank 1 recovers 81.2% of the rank-4 attenuation |
| Replicated | Layer-16 rank-4 attenuation 1.129 (seed 106), 1.070 (seed 107) |
| Selective | Structured nulls (catalogue control tag; family identity) made disjoint from the conflict subspace attenuate 0.039 and 0.022 against a random-subspace p95 of 0.032 |
| Not refusal | Cosine with a difference-of-means refusal direction 0.062; ablating conflict moves conflict rows −1.758 and reachable rows −0.168, ablating refusal moves conflict +0.494 and reachable +2.803 |

Layer selection matters and is a trap. Choosing the layer by cross-validated
probe accuracy selects layer 27, where rank-4 ablation removes 2.05% of the
contrast. Choosing by causal effect selects 16–17, where the same construction
removes 21.8%. Decodability peaks at layer 20 and stays 0.94–0.97 through 30
while causal efficacy peaks at 16 and decays — so probe accuracy actively
misleads about where to intervene, consistent with Hase et al. (arXiv 2301.04213).

---

## 4. Writing to it fails with structure

### 4.1 Scope is not the limiting factor

Widening the intervention from one layer at one position to all layers at all
positions nearly triples the effect and changes nothing about the decision.

| Scope | Attenuation | Share of contrast | Decision flips |
|---|---:|---:|---:|
| Layer 16, final position | 0.611 | 12.8% | 0/128 |
| Layer 16, all positions | 0.930 | 19.5% | 0/128 |
| All layers, final position | 0.922 | 19.3% | 0/128 |
| All layers, all positions | 1.590 | 33.4% | 0/128 |

Coherence is intact throughout: action-token probability mass 1.000, next-token
entropy 0.215 → 0.246.

### 4.2 Why nothing flips: boundary geometry, not inertness

The baseline margins are bimodal with an empty band around the decision
boundary. The 32 conflict rows that favour declining start at a mean of +3.492
with a **minimum of +2.125** — no row begins near zero. Under all-layer
all-position ablation they fall to a mean of +1.074 and a **minimum of +0.125**,
with five rows inside 0.5 of the boundary.

The intervention moves the deciding rows roughly 95% of the way to flipping and
stops. Reporting this as "0% correction" is true and uninformative.

### 4.3 Dose: monotone, then saturating, then destructive

Scaled removal `x ← x − k(x·r)r`, all layers, all positions:

| k | Attenuation | Conflict decline rate | Flips | Conflict-row shift | Action mass | Entropy | Coherent |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 0.5 | 22.4% | 0.500 | 0/128 | −1.248 | 1.000 | 0.225 | yes |
| 1.0 | 33.0% | 0.484 | 1/128 | −1.785 | 1.000 | 0.248 | yes |
| 1.5 | 37.9% | 0.422 | 5/128 | −2.031 | 1.000 | 0.258 | yes |
| 2.0 | 41.3% | 0.406 | 6/128 | −2.027 | 1.000 | 0.273 | yes |
| 3.0 | 100% | 0.500 | 64/128 | −0.990 | **2.7e-06** | **9.33** | **no** |
| 4.0 | 100% | 0.500 | 64/128 | −0.989 | **2.7e-06** | **9.33** | **no** |

Three things in this table are the paper.

**The effect is real and specific.** The decline rate falls monotonically to
0.406 and flips reach 6 of 128, while four matched-dose random directions
orthogonal to the conflict direction hold the decline rate at exactly 0.500 with
0–1 flips at *every* coherent dose. The direction does move decisions. It moves
about 5% of them.

**The effect saturates well before the model breaks.** The conflict-row shift
goes −1.248, −1.785, −2.031, −2.027. Removing more than the full component buys
nothing. Pushing the coordinate outside its natural range does not push the
decision further, which is a statement about the downstream pathway and not
about measurement precision.

**Beyond saturation the numbers invert into an artifact.** At k=3 a naive
readout is *100% of the condition contrast removed, 64 of 128 decisions
flipped* — a headline result. The model is emitting
`avenavenalemalemavenaven...` instead of a single `A` or `B`, with action-token
mass 2.7e-06 and entropy 9.33 against a 0.209 baseline. Matched-dose random
directions produce the same 64 flips and the same gate failure, so the collapse
is generic to perturbation magnitude and carries no information about the
variable.

### 4.4 What this implies for how the asymmetry is measured

A single-dose binary correction rate cannot separate three distinct states:

1. the direction is not a control point;
2. the direction is a control point whose effect saturates below the decision
   boundary;
3. the intervention was strong enough to flip decisions only by destroying the
   model.

State 2 is what we observe, and it is the interesting one. States 1 and 3 both
report as clean numbers — 0% and ~50% respectively — under a protocol that does
not sweep dose, does not include matched-dose random controls, and does not gate
on coherence. We recommend all three as minimum reporting for any writability
claim.

---

## 5. Specificity degrades as intervention scope widens

Worth stating because it cuts against our own strongest numbers. At single-layer
scope the conflict subspace beats its random-subspace floor by a factor of 79
(1.334 against a p95 of 0.017). At all-layer all-position scope the margin falls
to 12× on the median random draw and only **2.6× on the p95** (1.590 against
0.610), and that p95 rests on a single random draw at 0.807 out of eight.

Widening the intervention buys effect size and loses specificity. We do not lean
on the all-position condition for any specificity claim, and neither should
anyone else without more random draws.

---

## 6. Limitations

1. **One model, one construct, synthetic scenarios.** Llama-3.1-8B-Instruct only.
   The task is a synthetic lookup with a permission scope.
2. **The behavioural clamp caps the behavioural ceiling.** A forced binary A/B
   margin with an empty ~2-logit band around the boundary structurally cannot
   show a graded variable controlling much. Our null on decision flips is partly
   a property of this design, which is why §4.2 reports boundary geometry rather
   than only the flip count. A task with real action variation is required for a
   stronger behavioural claim, and we do not have one.
3. **Not emergent deception or misalignment.** The variable is scope conflict
   under instruction. It is not evidence about strategic behaviour.
4. **The scenario generator had a defect, and every reported result used the
   defective version.** `control_label` was `(pair_index + constant) % 2`, and
   that parity also fixes in-scope slot assignment and catalogue ordering, so the
   catalogue control tag was confounded with scope structure — a first principal
   angle cosine of 0.9999 against the conflict subspace. The generator now
   supports a `parity_independent` mode with assignment balanced within each
   parity class, and `validate_manifest` reports
   `control_label_parity_independent` unconditionally in every run artifact, so
   the confound cannot recur silently. **This does not retroactively fix
   anything.** All numbers reported here come from the confounded generator,
   which is why the §3 selectivity result rests on the family null and both
   orthogonalised nulls rather than the raw control-tag comparison. The defect
   was present in all three families at all seeds tested, so it is a property of
   the generator and not of any particular run.
5. **Eight random draws is a thin specificity threshold** at all-position scope.
6. ~~One artifact is not row-level auditable.~~ Closed. The seed-107 protocol
   was re-run under the legacy generator mode with per-row margin storage added
   (`results/arm_g_layer16_seed107_v2/`). All fourteen reported attenuations —
   seven depth layers, three ranks, four nulls — recompute from the stored
   row-level margins with maximum absolute disagreement 0.00e+00, so every
   artifact this paper cites is now independently checkable. The re-run also
   reproduced the original numbers **bit-exactly**: depth, rank, selectivity,
   random-subspace p95 and the 0.999899 control-tag principal angle all agree to
   0.000000, confirming the protocol is deterministic and that the refactor
   changed nothing behavioural.
7. **We do not explain the asymmetry.** Why refusal is writable and
   hallucination, deception and scope-conflict are not is unresolved here.

---

## 7. Reproducibility

Every reported statistic was recomputed from stored row-level margins by
independently written estimator code (`arm_g_audit.py`): 37 of 37 reproduce with
maximum absolute disagreement 0.0. The layer-16 peak survives selection
adjustment — bootstrapping the arg-max over 20,000 stratified resamples selects
it in 100.0% of draws, and all five Bonferroni-adjusted pairwise contrasts
exclude zero. The layer ordering is not explained by intervention magnitude:
Spearman correlation between effect and displacement across layers is −0.086,
and layers 18–19 receive larger perturbations relative to residual-stream norm
than layer 16 while producing smaller effects.

Protocols were frozen with preregistered decision rules before each run; all
five scripts carry deterministic self-tests; artifacts include run configs,
manifests and per-row margins.

---

## 8. What we claim and what we do not

**We claim:** a fourth independent construct shows the detection–correction
asymmetry; the asymmetry has dose structure that binary reporting hides; the
structure is saturating partial writability rather than absence of control; and
scaled interventions produce success-shaped artifacts that require matched-dose
controls and a coherence gate to detect.

**We do not claim:** that the variable controls behaviour; that the result
generalises beyond this model or task; that we know why some directions are
writable; or that our effect sizes are large. The largest coherent intervention
we can apply moves about 5% of decisions.
