# Investigation brief: adversarial review of a steering-methodology claim

**This is not a handoff.** Nothing is being delegated. You are being asked to
think independently about a body of evidence and reach your own conclusion,
including the conclusion that the current interpretation is wrong.

The work was done by Claude Opus 5 in one long session with a human researcher.
The artifacts are real, the numbers are audited, and the interpretation is
contested — by its own author, twice already.

---

## 0. What is actually wanted from you

In rough order of value:

1. **Attack the central claim.** §5 states it. §6 lists the objections we already
   know about, including one we think may be fatal. Find the ones we missed.
2. **Decide independently whether the claim matters.** We may be overrating a
   methodological point because it is ours. A reviewer's shrug is a real outcome
   and we would rather hear it now.
3. **Think abstractly about what the evidence licenses.** The observations in §4
   may support a different and better claim than the one we made. We have already
   changed our mind once in that direction.
4. **Then research.** Check whether the claim, or the better claim you find, is
   already in the literature. Our record on this is four for four *against* us:
   every direction we proposed this session turned out to be published already.
   Assume that pattern continues until you verify otherwise.

**What is not wanted:** confirmation, a literature summary without a judgement,
prose polish, or agreement reached by deferring to the framing below.

---

## 1. Calibration on the source

Weight the interpretive sections accordingly. In this one session the author:

- predicted all-position ablation would flip decisions. It flipped zero.
- predicted a larger dose would finish the job. It saturated instead.
- proposed four research directions in sequence. All four were already published
  (baseline-parameterisation sweeps, a decay experiment's framing, a
  representational "deception tax", and the read-only asymmetry itself).
- built a "clean null" control that was confounded with the thing it was
  controlling for, at first principal angle cosine 0.9999.
- made a sign error in an analysis script, and a false-failure error in the audit
  script that checks the analysis.
- wrote a full paper draft whose central claim it then refuted with the next
  experiment, and rewrote it.

Every one of those was caught — by a control, a dry run, a self-test, or a
literature check. The process is more reliable than the author's intuitions. Trust
the artifacts over the prose, including this document's prose.

---

## 2. The system under study

Llama-3.1-8B-Instruct. A synthetic task with **mechanically-determined labels**:
matched scenario pairs where system prompt, tool prelude, workspace, catalogue and
stated permission scope are identical, and only an opaque target identifier
differs — one resolving to a path inside an allowed scope, one outside. The label
is a property of the file path. No human or model judge appears anywhere.

At the measurement point the model must emit exactly `READY`, so visible output is
constant across conditions. The outcome is the next-token logit margin
`DECLINE − READ`, with the A/B token mapping reversed on every scenario.

A rank-1 direction `r` was extracted at layer 16 by difference-of-means over
matched pairs from two source seeds, and evaluated on held-out scenario families
with a fresh prompt seed per run.

---

## 3. Read this section before §5

§4 is observations. §5 is our interpretation. **Form your own view of what §4
means before reading §5.** If you read §5 first you will probably agree with it,
and that agreement is worth nothing to us.

---

## 4. Observations

All figures recomputed from stored per-row data by independently written estimator
code. Artifact paths in §8.

### 4.1 The direction is a real, validated object

| Property | Value |
|---|---|
| Held-out AUROC, two independent source seeds | 0.756, 0.850 |
| Behavioural comparator (output clamped by design) | 0.500 |
| Cosine between source-seed directions | 0.969 |
| Prompt-only leakage probes vs frozen 0.65 threshold | 0.445, 0.410 |
| Causal peak layer | 16–17 (1.070, 1.076); layers 13–15 at random floor |
| Rank-1 share of rank-4 causal effect | 81.2% |
| Replication across prompt seeds | 1.129, 1.070 |
| Structured nulls made disjoint from it | 0.039, 0.022 vs random p95 0.032 |
| Cosine with a difference-of-means refusal direction | 0.062 |

Selecting the layer by probe accuracy picks layer 27 (2.05% causal effect).
Selecting by causal effect picks 16–17 (21.8%). Decodability peaks at layer 20 and
stays 0.94–0.97 through layer 30; causal efficacy peaks at 16 and decays.

### 4.2 Operator A — removal, applied at all 32 layers, all token positions

Mapping the component to its group mean:

| Scope | Contrast removed | Decision flips |
|---|---:|---:|
| Layer 16, final position | 12.8% | 0/128 |
| Layer 16, all positions | 19.5% | 0/128 |
| All layers, final position | 19.3% | 0/128 |
| All layers, all positions | 33.4% | 0/128 |

Scaled removal, all layers, all positions:

| k | Contrast removed | Conflict decline rate | Flips | Action-token mass | Entropy |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 22.4% | 0.500 | 0/128 | 1.000 | 0.225 |
| 1.0 | 33.0% | 0.484 | 1/128 | 1.000 | 0.248 |
| 1.5 | 37.9% | 0.422 | 5/128 | 1.000 | 0.258 |
| 2.0 | 41.3% | 0.406 | 6/128 | 1.000 | 0.273 |
| 3.0 | 100% | 0.500 | 64/128 | **2.7e-06** | **9.33** |

At k=3 the model emits `avenavenalemalemavenaven` instead of a single token, and
four matched-dose random directions produce the same 64 flips and the same
collapse. Baseline entropy is 0.209.

### 4.3 Operator B — signed addition, applied at layer 16 only, all positions

`x ← x + c·σ·r`, where σ = 0.5506 is the SD of the baseline projection onto `r`.

| c (σ) | Contrast | Conflict decline | Reachable decline | Flips | Action mass | Entropy |
|---:|---:|---:|---:|---:|---:|---:|
| −8 | 5.094 | 0.000 | 0.000 | 32/128 | 1.0000 | 0.040 |
| −4 | 6.375 | 0.344 | 0.000 | 10/128 | 1.0000 | 0.154 |
| 0 | 4.939 | 0.500 | 0.000 | 0/128 | 1.0000 | 0.215 |
| +2 | 3.445 | 0.859 | 0.016 | 27/128 | 1.0000 | 0.447 |
| +3 | 2.893 | 1.000 | 0.344 | 58/128 | 1.0000 | 0.478 |
| +4 | 2.424 | 1.000 | 0.781 | 86/128 | 1.0000 | 0.438 |
| +8 | 1.232 | 1.000 | 1.000 | 96/128 | 0.9998 | 0.226 |

128 of 128 rows cross the boundary somewhere in this range. Individual crossing
doses span 0.600σ to 6.111σ. Four random directions orthogonal to `r` produce 0–1
flips at every dose, with contrasts staying in 3.96–5.31 against a 4.94 baseline.

### 4.4 Displacement magnitudes, both operators

The natural projection range onto `r` is −0.197 to 1.509. Mean absolute deviation
from the projection mean is 0.4535 (0.824σ).

| Intervention | Mean displacement | Max displacement | Coherence |
|---|---:|---:|---|
| Removal k=1.0 | 0.824σ | 1.924σ | intact |
| Removal k=1.5 | 1.235σ | 2.886σ | intact |
| Removal k=2.0 | 1.647σ | 3.848σ | intact |
| **Removal k=3.0** | **2.471σ** | **5.772σ** | **destroyed** |
| Addition c=±2 | 2.000σ | 2.000σ | intact |
| Addition c=±4 | 4.000σ | 4.000σ | intact |
| **Addition c=±8** | **8.000σ** | **8.000σ** | **intact (0.9998)** |

Resulting projections, given a natural range of [−0.197, 1.509] and a projection
mean of 0.4496:

| Intervention | Resulting projection range | Excursion beyond natural range |
|---|---|---|
| Removal k=2 | [−0.610, 1.096] | 0.413 below minimum |
| **Removal k=3** | **[−1.669, 1.743]** | **1.472 below minimum** — destroyed |
| Addition c=+8 | [4.208, 5.914] | 2.699 above maximum — intact |
| Addition c=−8 | [−4.602, −2.896] | 2.699 below minimum — intact |

So addition travels roughly 1.8× further outside the natural range than the
removal dose that destroys the model, in *either* sign, and stays coherent.

### 4.5 Per-row responsiveness

Slope of margin against dose, fitted per row in the near-linear ±2σ window:

| Rows | Mean slope |
|---|---:|
| Conflict condition | +0.797 |
| Reachable condition | +1.503 |
| Difference | −0.706, 95% CI [−0.802, −0.611] |

Explaining that slope:

| Model | R² | Incremental test |
|---|---:|---|
| Baseline margin only | 0.894 | — |
| Margin + condition | 0.898 | F(1,125) = 3.79, p = 0.054 |
| Condition only | 0.618 | — |
| Condition + margin | 0.898 | F(1,125) = 341.5, p ≈ 0 |

Pearson correlation between per-row slope and baseline margin: −0.944. Conflict
rows fit the linear model worse than reachable rows (per-row R² 0.876 vs 0.963,
Mann-Whitney p < 1e-4); restricting to the near-baseline window doubles the slope
gap (−0.347 full-range → −0.706 windowed) rather than shrinking it.

### 4.6 External results we are positioned against

- Roy et al. (arXiv 2604.13068): 0% correction rate in 7 of 7 models, 117M–7B,
  three families, steering along a hallucination probe direction. Coefficient not
  reported.
- Nyoma, "Rift" (arXiv 2606.17229): deception direction readable cross-family
  (AUC 0.933), not writable — 0/8 adding, 0/8 subtracting.
- arXiv 2511.18284: hallucination **highly steerable** (+60 trait delta) in
  Llama-3.1-8B at layer 15. Same construct as Roy, opposite verdict. LLM judge.
- Arditi et al. (arXiv 2406.11717): refusal flipped by ablating one direction
  across all layers and positions, 13 chat models.

---

## 5. Our current interpretation — treat as a claim under test

That whether a probe direction is reported as a "control point" is determined by
two free parameters of the intervention rather than by the direction: the
**operator** (removal vs addition) and the **coefficient**. On this direction, a
single-dose protocol would report 5.5%, 21.1%, 45.3%, 67.2% or 75.0% correction at
1σ, 2σ, 3σ, 4σ and 6σ. And removal at all layers flips nothing while addition at
one layer flips everything.

Secondary claim: responsiveness is governed by distance to the decision boundary
rather than by experimental condition (§4.5), and since a linear readout would
give every row an identical slope, the threefold variation must come from the
nonlinear layers downstream of the intervention.

Proposed mechanism for the operator difference: removal is bounded — once the
component is gone there is nothing left to remove — and beyond that it reflects
the deviation and drives states to projections no input produces, whereas addition
displaces all rows uniformly and preserves relative structure.

---

## 6. Objections we already know about

**6.1 The operator comparison is confounded with layer scope, and this may be
fatal.** Every removal experiment applied the intervention at all 32 layers.
The additive experiment applied it at layer 16 only. So "removal fails, addition
works" is equally consistent with "32-layer intervention fails, 1-layer
intervention works." The §4.4 magnitude table rules out raw displacement size as
the explanation — 8σ uniform addition is coherent while 2.47σ mean removal is
not — but it does not rule out layer scope, because the two operators were never
run at matched scope. **The decisive experiment has not been run.** If you think
this sinks the central claim, say so plainly.

**6.2 The σ-normalisation may not make the operators commensurable.** Removal
displaces each row by a row-specific amount determined by its own projection;
addition displaces every row identically. There may be no fair common dose unit,
in which case comparing them at "matched dose" is meaningless and §4.4 is not the
rebuttal we take it to be.

**6.3 The claim may be obvious.** "Effect size depends on intervention strength"
is not surprising. The interesting version is that it spans the entire range of
qualitative conclusions and that published work fixes the parameter without
reporting it. Judge whether that is a contribution or a note.

**6.4 We cannot check the papers we are implicitly criticising**, because their
coefficients are not reported. Our claim that their nulls may be under-dosed is a
mechanism-backed hypothesis, not a demonstrated refutation.

**6.5 The behavioural outcome is weak.** A forced binary A/B margin with an empty
band around the boundary. The clamp that makes the observational result clean is
also what makes the behavioural instrument coarse.

**6.6 One model, one construct, one layer for the additive arm.**

**6.7 A known generator defect.** `control_label` was a deterministic function of
`pair_index` parity, which also fixed in-scope slot assignment and catalogue
ordering, so one "control" contrast was confounded with scope structure at
principal angle cosine 0.9999. Fixed going forward behind a mode flag; committed
artifacts predate the fix. Selectivity claims rest on the other nulls.

---

## 7. Questions we think are worth deep thought

Not a checklist. Pick what seems most productive, including questions we did not ask.

1. Is there a principled way to make removal and addition commensurable? If not,
   what is the right way to state a writability claim at all?
2. §4.4 shows an 8σ uniform displacement is coherent while a 2.47σ mean
   row-dependent displacement is not. What property distinguishes them —
   uniformity, sign, preservation of relative order, something else? Is there a
   testable prediction that separates the candidates?
3. §4.5 finds responsiveness declining with baseline margin, in a system where a
   linear readout would give constant slope. What downstream mechanism produces
   that, and does it predict anything else measurable?
4. Given §4.1 — probe accuracy peaks at layer 20, causal effect at 16 — and §4.5,
   is there a unified account of where a variable is readable versus where it is
   actuable?
5. Does the direction's near-orthogonality to refusal (0.062) combined with
   opposite causal action patterns tell us something about how many independent
   "decision-relevant" axes this model has near this task?
6. What is the cheapest experiment that could falsify our central claim? We think
   it is a layer-matched operator comparison. Is there a cheaper or sharper one?
7. Is there a version of this work that would matter to someone deploying an
   interpretability-based monitor, as opposed to someone writing a paper about one?

---

## 8. How to verify rather than trust

Repository `~/phi-map`, branch `arm-g-causal-characterization`.

- `results/arm_g_boundary_seed110_v1/` — additive dose sweep, per-row margins at
  all 15 doses, per-row slopes, crossing doses, baseline projections.
- `results/arm_g_dose_seed109_v1/` — scaled removal, per-row margins per dose.
- `results/arm_g_allpos_seed108_v1/` — operator/scope factorial, refusal
  discriminant.
- `results/arm_g_layer16_seed107_v2/` — depth/rank/selectivity, per-row margins
  for all 14 conditions. Reproduced the v1 run bit-exactly (0.000000).
- `results/arm_g_cross_layer_seed106_v1/` — includes `AUDIT.md` and
  `arm_g_audit.json`: 37/37 statistics recomputed at max disagreement 0.0.

Artifacts are gzip+base64 JSON: `json.loads(gzip.decompress(base64.b64decode(...)))`.
Every experiment script runs `--self-test` with no GPU and no network. One of
those self-tests constructs the shared-slope case and asserts that the naive
dose-to-flip regression returns R² = 1.0 exactly — i.e. that the obvious version
of the §4.5 analysis is circular by construction.

`PAPER_DRAFT.md` is the current write-up. `RESEARCH_ARC.md` is the full evidence
ledger including everything that failed. `MONITORING_BASELINES.md` and
`DECEPTION_TAX.md` are two abandoned research directions with the reasons.

---

## 9. Failure modes we would rather you avoid

- Agreeing with §5 because it is stated confidently. It has been wrong before.
- Treating §6.1 as a caveat. We flag it as potentially fatal and mean it.
- Proposing a new experiment without checking whether it is published. Our record
  is four for four in the wrong direction.
- Producing a literature review instead of a judgement.
- Assuming the numbers are wrong. They have been audited and they reproduce. If
  something looks impossible, it is more likely that the interpretation is wrong
  than the arithmetic.
