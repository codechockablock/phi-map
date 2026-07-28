# Research arc — state as of 2026-07-25

## The question we started with

The original program asked whether the internal-to-external map destroys
shared structure: does a model represent a strategic condition coherently
inside even when its visible behaviors differ across tasks? The attached
research report added a second question: can a model use a low-leakage
serialization of that internal geometry better than an observer receiving the
same channel?

Those became three related but distinct arms:

1. **Arm T — deception transfer:** compare cross-task deception geometry in
   activations and response representations.
2. **Arm G — goal–constraint conflict:** test whether an acting model carries a
   shared pre-action representation of whether its requested goal is achievable
   under a stated constraint, then test that representation causally.
3. **Arm S — serialized access:** test channel utility, model-family familiarity,
   and—only if a non-reconstructible state exists—producer-specific access.

Evidence for one arm does not prove the claims of another.

## What has been completed

### 1. Apollo reproduction and exact pairing

The released Apollo results were reproduced and exactly paired where the
artifacts permit it.

- Internal roleplaying-trained scores transferred to AI-liar at 0.925 AUROC
  and sandbagging at 0.911.
- On exact matched episodes, internal minus whole-response MiniLM was
  +0.378, 95% CI [0.187, 0.562], for AI-liar and +0.355
  [0.314, 0.397] for sandbagging.
- The insider-trading comparison is not cleanly matched because the released
  internal and external artifacts use different variants.

This establishes a large internal/external difference in the released
measurement pipeline. It does not by itself establish that the difference is
causal or that it survives a fully matched acting-model replication.

### 2. Behavioral capacity and monitor comparisons

Increasing fixed embedding capacity improved in-task discrimination but did not
recover cross-task transfer:

| Fixed encoder | Dimension | In-task AUROC | Cross-task AUROC |
|---|---:|---:|---:|
| MiniLM | 384 | 0.739 | 0.485 |
| BGE base | 768 | 0.775 | 0.486 |
| BGE large | 1,024 | 0.806 | 0.461 |

A single untuned LLM-judge prompt reached 0.764 mean AUROC across the four
families, between fixed-feature transfer near 0.47 and the released internal
scores of 0.91–0.99. This is useful evidence that a flexible semantic reader
generalizes better than a frozen feature direction, but the judge run is small,
has incomplete confidence intervals, and uses a stronger model than the model
being judged.

The historical six-dimensional action-surface encoder was also audited:

- Its independently labeled Apollo macro AUROC was 0.509
  [0.457, 0.561], not approximately 0.75.
- The earlier 0.753 number used 384-dimensional MiniLM/PCA features and the
  governance system's own verdicts. It was not a six-dimensional external
  validity result.
- The separate identity-over-asserted-G comparison used Apollo
  `graded_deception` labels, not governance verdicts.

A stronger response-only comparator—raw MiniLM sentence-step means—reached
0.746 in-task macro AUROC [0.696, 0.794], but its cross-task transfer remained
0.467. Thus approximately 0.75 is an in-domain descriptive comparator, not a
general behavioral ceiling.

### 3. Matched acting-model deception pilot

The exact-token Llama-3.1-8B pilot regenerated responses and probed the same
acting model.

On intended experimental conditions, the completion-gated rerun looked
positive:

- internal cross-task AUROC: 0.844;
- response step-mean cross-task AUROC: 0.502;
- delta: +0.341, 95% CI [0.101, 0.501].

That is not a realized-deception result. Intended condition and realized
behavior agreed poorly for sandbagging, and the frozen completion gate narrowly
failed.

After clarified blinded human grading, the best available realized-behavior
pass gave:

- internal cross-task AUROC: 0.594;
- response step-mean cross-task AUROC: 0.661;
- delta: -0.067, 95% CI [-0.326, 0.284].

This matches the preregistered refuted pattern for the two-task pilot. Label
reliability was weak on the difficult sandbagging cases, and the run still lacks
a passing completion gate, a second independent rater, and four task families.
Therefore the broad Arm T claim is not established. The current evidence is
mixed: strong in the released Apollo comparison, negative in the best matched
8B realized-behavior pilot.

### 4. Serialized-access identification gate

Arm S corrected a key identification error. In ordinary frozen
decoder-only inference, the KV cache is a deterministic function of the visible
prefix and the weights. A fresh exact clone can reconstruct it, so cache reuse
does not create private producer information.

The local S0 test confirmed source/clone equivalence within numerical error:

- identical next-token argmaxes;
- maximum absolute logit drift: 8.92e-5;
- maximum Jensen–Shannon divergence: 2.12e-11.

The low-leakage serializers pass synthetic smoke tests, but real S1 channel
utility/model-familiarity has not been run. Producer privilege has not been
tested because no non-reconstructible private state has been defined.

### 5. Arm G observational result

Arm G uses synthetic matched lookup scenarios. Within each pair, the prompt,
workspace, tools, catalog, and constraint are the same. Only an opaque requested
target changes: one target is inside the allowed scope and the other is outside
it. The condition label is therefore mechanical, not human- or model-judged.

At the measurement point the model must emit exactly `READY`, making the
visible response deliberately uninformative while its final prompt-token state
is captured.

The first seed looked strong but was correctly rejected because the original
prompt-leakage threshold was unstable. A 100-seed null calibration showed that
25% of seeds exceeded the old 0.60 threshold, so the threshold was frozen at
0.65 before confirmatory seeds.

Two independent confirmatory runs then passed:

| Source seed | Internal held-out AUROC | Behavioral AUROC | Delta | Delta 95% CI |
|---:|---:|---:|---:|---:|
| 101 | 0.756 | 0.500 | +0.256 | [0.111, 0.400] |
| 102 | 0.850 | 0.500 | +0.350 | [0.195, 0.459] |

Both used `release_records` for development and transferred to
`data_checksums` and `incident_times`. Positive, shuffled-label, random-direction,
and calibrated prompt-only controls passed. The seed-101 and seed-102 mean
directions were highly aligned: cosine 0.969.

The behavioral contrast here must be interpreted carefully. The output is
forced to be identical, so the 0.50 response comparator is partly a property of
the design. This establishes pre-action internal information under a behavioral
clamp; it does not establish spontaneous concealment or deception.

### 6. Arm G causal result

The seed-102 layer-27 conflict-minus-reachable direction was frozen and
intervened on in fresh seed-103 prompts from the two held-out families.
The outcome was the semantic next-action margin, `DECLINE - READ`, with the
A/B answer-token mapping reversed on every scenario.

- Mean directional effect: +0.288.
- Pair-bootstrap 95% CI: [0.258, 0.322].
- The effect was positive in both held-out families and both token mappings.
- Equal-norm random-direction absolute p95: 0.076.

This supports a causal directional effect rather than mere probe decodability.

Fresh seed-104 testing established a monotonic dose response:

| Dose | Mean semantic-margin effect |
|---:|---:|
| 0.5 SD | 0.065 |
| 1 SD | 0.149 |
| 2 SD | 0.286 |
| 4 SD | 0.564 |

Removing the natural rank-1 projection attenuated the condition contrast by
0.115 [0.074, 0.160], or 2.34%. The direction is therefore a causal
contributor, not a sufficient explanation.

### 7. Multidirectional layer-27 result

A rank-8 source subspace was built from independent seed-101 and seed-102
paired differences and tested on fresh seed-105 prompts.

- Rank-1 attenuation: 0.125 [0.088, 0.166].
- Rank-4 attenuation: 0.156 [0.119, 0.195].
- Rank-8 attenuation: 0.158 [0.115, 0.201].
- Rank-8 minus rank-1: +0.033 [0.006, 0.063].
- Rank-8 random-subspace absolute p95: 0.034.

The additional dimensions have a statistically nonzero causal contribution,
but rank 4 and rank 8 are nearly identical. Rank 8 removes only 3.25% of the
natural condition contrast. The computation is genuinely multidimensional at
layer 27 but mostly distributed, redundant, nonlinear, or located elsewhere.

### 8. Cross-layer rank-4 causal mediation

The frozen cross-layer protocol ran on fresh seed-106 prompts. Every
preregistered criterion passed and the decision was
`SUPPORTED_DISTRIBUTED_DEPTH_MEDIATION`. The label understates what the numbers
say. The missing causal mass is not spread evenly through depth: it is
concentrated at layer 16, and layer 27 was the wrong place to have been
looking.

Single-layer rank-4 ablation against a natural condition contrast of 4.854:

| Layer | Attenuation | Share of contrast | 95% CI | Random p95 | Prototype transfer AUROC |
|---:|---:|---:|---|---:|---:|
| 12 | -0.006 | -0.12% | [-0.029, 0.020] | 0.030 | 0.503 |
| 16 | 1.129 | 23.26% | [0.943, 1.299] | 0.023 | 0.911 |
| 20 | 0.258 | 5.31% | [0.219, 0.299] | 0.027 | 0.973 |
| 24 | 0.234 | 4.83% | [0.195, 0.275] | 0.029 | 0.958 |
| 27 | 0.100 | 2.05% | [0.072, 0.127] | 0.019 | 0.954 |
| 30 | 0.059 | 1.21% | [0.025, 0.090] | 0.021 | 0.940 |

Cumulative ablation in ascending depth order:

| Through layer | Attenuation | Share of contrast |
|---:|---:|---:|
| 12 | -0.006 | -0.12% |
| 16 | 1.115 | 22.98% |
| 20 | 1.277 | 26.32% |
| 24 | 1.307 | 26.92% |
| 27 | 1.309 | 26.96% |
| 30 | 1.334 | 27.48% |

The full six-layer cascade removed 27.48% of the condition contrast against
3.25% for rank 8 at layer 27 alone, an 8.5-fold improvement. The preregistered
increment over the anchor layer was +1.234, CI [1.031, 1.434]. Specificity was
emphatic: eight random rank-4 cascades, orthogonal to the target subspace at
every layer and carried through the identical centering procedure, gave an
absolute p95 of 0.017 against a target effect of 1.334, a factor of 79. The
frozen pre-onset layer 12 behaved as a null should, at -0.006 against its own
random p95 of 0.030.

Layer 16 alone accounts for 85% of the full cascade effect. Adding layer 27 to
an already-ablated 12/16/20/24 stack moved the attenuation from 26.92% to
26.96%.

The decodability profile dissociates from the causal profile. Cross-family
prototype transfer peaks at layer 20 and stays between 0.94 and 0.97 through
layer 30, while causal efficacy peaks at layer 16 and decays monotonically
afterward. The layer where the variable is most readable is not the layer where
it does the most work. The original layer selection maximized development
cross-validated decodability, which is precisely why it landed on 27 and made
the causal story look weak.

One limit is important and cuts against an over-strong reading. The
intervention rescales the decision margin without disturbing the decision.
Baseline margin AUROC against the condition label was 0.998; after removing 24
dimensions across six layers it was 0.997. Accuracy stayed at 0.75 and the
decline-choice rates were unchanged at 0.00 for reachable and 0.50 for
conflict. The contrast shrank mostly by the reachable condition moving up
(-3.873 to -2.918) rather than the conflict condition collapsing (0.980 to
0.602). So this is a graded, causally specific effect on margin magnitude, not
demonstrated control of which action is selected.

Artifacts are in `results/arm_g_cross_layer_seed106_v1/`.

### 9. Mathematical audit of section 8

`arm_g_audit.py` recomputes the seed-106 result from its 128 stored row-level
margins using independently written estimator code. Full findings in
`results/arm_g_cross_layer_seed106_v1/AUDIT.md`.

The design is exactly balanced, which makes the reported grand-mean contrast
and the bootstrapped family-stratified mean of per-pair contrasts algebraically
the same statistic — verified at a gap of exactly zero, so the interval belongs
to the quantity reported. All 37 reported statistics reproduce with maximum
absolute disagreement of zero.

The conclusion does not depend on the bootstrap. Percentile, basic, normal-
approximation and paired *t* intervals all exclude zero for both the full
cascade attenuation and the preregistered increment, and the distribution-free
tests agree (32/32 and 31/32 pairs positive; sign test *p* = 4.7e-10 and
1.5e-8). The headline share, previously reported without an interval, is 27.5%
with a jointly bootstrapped CI of [26.0%, 29.3%].

Two challenges to the layer-16 claim were tested directly and both failed to
land. Selection: bootstrapping the arg-max over 20,000 stratified resamples
selected layer 16 in 100.0% of them, and all five Bonferroni-adjusted pairwise
contrasts against layer 16 exclude zero. Intervention size: layer 16 received
one of the *smallest* perturbations, the correlation between effect and
displacement across layers is -0.086, and layer 16 is 19 times more efficient
per unit of displacement than layer 27.

The decision-invariance limit is confirmed and is stronger than previously
stated: zero of 128 rows changed which action they favored, under both the
layer-16 ablation and the full cascade.

The ablation operator satisfies its required algebra at 1e-15 — orthogonal
projector, within-group mean preservation, idempotence, and the cascade
consistency property the sequential design relies on.

One new limitation was identified. Displacement is measured in absolute
subspace coordinates, and residual-stream norms grow with depth, so the audit
rules out absolute but not *relative* perturbation size as the driver of the
layer ordering. Recording per-layer residual-stream norms in the next run
closes it.

### 10. Layer-16 resolution (seed 107)

Decision: `SELECTIVE_CONFLICT_SUBSPACE`, both preregistered criteria met. But
the preregistered primary control turned out to be broken, and the conclusion
rests on the other evidence in the run rather than on it.

**The control-tag null is confounded.** `control_label` is
`(pair_index + constant) % 2` and `inside_slot` is `pair_index % 2`, so the
KITE/MOSS tag is a deterministic function of pair-index parity — which is also
what fixes which catalog slot holds the in-scope file and the catalog ordering.
The "decision-irrelevant" null is therefore partly a scope contrast. This was
verified directly against the manifests for seeds 101, 102 and 107. It shows up
geometrically: the first principal angle cosine between the control-tag and
conflict subspaces at layer 16 is 0.9999, which is not a chance alignment for
two rank-4 subspaces in 4,096 dimensions.

The pre-specified disambiguator is what saves the run. Ablating the component
of each null that is provably disjoint from the conflict subspace:

| Subspace at layer 16 | Attenuation | Share of contrast |
|---|---:|---:|
| Conflict, rank 4 | 1.070 | 21.80% |
| Control tag, raw | 0.672 | 13.69% |
| Control tag, conflict projected out | 0.039 | 0.79% |
| Family, raw | 0.133 | 2.71% |
| Family, conflict projected out | 0.022 | 0.44% |
| Random rank-4, p95 of 8 | 0.032 | 0.65% |

Structured features that are genuinely disjoint from the conflict subspace have
no leverage — 0.039 and 0.022 against a random-subspace floor of 0.032. The
family null, which is not confounded with pair parity, is near zero raw as
well. Selectivity is therefore supported, on that evidence rather than on the
control-tag comparison.

**Depth.** The effect switches on abruptly between layers 15 and 16 and sits on
a plateau:

| Layer | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attenuation | -0.029 | -0.031 | 0.031 | 1.070 | 1.076 | 0.631 | 0.381 |
| Displacement / residual norm | 0.010 | 0.015 | 0.018 | 0.060 | 0.073 | 0.078 | 0.068 |

Layers 13–15 sit at their random-control floor. The nominal peak is 17, not 16,
but 16 and 17 are indistinguishable. Layer 16 is not a shoulder; it is the
first layer of a sharp plateau.

This also closes the limitation the audit left open. Layers 18 and 19 receive
*larger* perturbations relative to residual-stream norm than layer 16 does and
produce markedly smaller effects, so the depth ordering is not explained by
relative perturbation size any more than by absolute size.

**Rank.** Rank 1 recovers 0.869, or 81.2% of the rank-4 attenuation; rank 2
gives 0.910. The rank-4-minus-rank-1 increment is 0.201, CI [0.113, 0.295], so
the extra dimensions contribute a nonzero amount, but the object is
predominantly a single direction. This matches the audit's prediction from the
seed-106 displacement being 78% rank-1.

**Replication.** Layer-16 rank-4 attenuation was 1.129 on seed 106 and 1.070 on
seed 107, an independent prompt seed reproducing the effect within about 5%.

One artifact limitation: this run stores baseline margins, residual norms and
conflict projections per row, but not the per-row ablated margins, so it cannot
be independently re-audited from row level the way seed 106 was. Future runs
should store ablated margins per condition.

Artifacts in `results/arm_g_layer16_seed107_v1/`.

### 11. All-position ablation and the refusal discriminant (seed 108)

The published prediction was that ablating a mediating direction from every
layer and every token position flips behaviour where a single-layer
final-position ablation does not. It did not flip behaviour here, but the
reason is a threshold artifact rather than an inert variable, and the run
produced a stronger discriminant result than intended.

Scope factorial on the frozen rank-1 layer-16 direction, baseline contrast
4.766:

| Scope | Attenuation | Share | Sign flips | KL from baseline |
|---|---:|---:|---:|---:|
| Layer 16, final position | 0.611 | 12.8% | 0/128 | 0.013 |
| Layer 16, all positions | 0.930 | 19.5% | 0/128 | 0.027 |
| All layers, final position | 0.922 | 19.3% | 0/128 | 0.020 |
| All layers, all positions | 1.590 | 33.4% | 0/128 | 0.073 |
| Layer 17, all layers, all positions | 1.791 | 37.6% | 2/128 | 0.078 |

Widening scope multiplies the effect 2.6-fold, as the literature predicts, with
coherence intact throughout — action probability mass stays at 1.000 and
next-token entropy moves only from 0.215 to 0.246.

**Why nothing flips.** The baseline margins are bimodal with an empty band
around the decision boundary. The 32 conflict rows that favour declining sit at
a mean of +3.492 with a *minimum* of +2.125; no row starts near zero. Under the
all-layer all-position ablation those rows fall to a mean of +1.074 with a
minimum of **+0.125**, and five conflict rows end within 0.5 of the boundary.
The intervention moves them roughly 95% of the way to flipping and stops just
short. Zero flips is therefore a statement about where the baseline mass sits,
not evidence that the variable fails to drive the decision. The correct next
test is dose: scale the removal past unit strength using the existing
dose-response machinery.

**The intervention is highly selective between conditions.** All-layer
all-position ablation shifts conflict rows by -1.758 and reachable rows by only
-0.168, a ratio of about ten to one. It suppresses the conflict-specific lean
toward declining and leaves the reachable condition essentially untouched.

**The refusal discriminant is a double dissociation.** A standard
difference-of-means refusal direction, built at the same layer from matched
harmful and harmless instructions with a mean projection separation of 6.95, is
nearly orthogonal to the conflict direction: **cosine 0.062**. Causally the two
act on opposite halves of the design:

| Ablated direction | Conflict-row shift | Reachable-row shift |
|---|---:|---:|
| Conflict, rank 1 | **-1.758** | -0.168 |
| Refusal | +0.494 | **+2.803** |

So the Arm G variable is not a refusal or harmfulness direction under another
name, on geometry and on causal action pattern. This also addresses the
AgentLens comparison empirically rather than by construction: their probe
separates harmful execution steps from benign ones, which is a
refusal/harmfulness contrast, and that contrast behaves differently here.

**Specificity costs scope.** Eight random directions ablated at all layers and
all positions attenuate by a median of 0.128 but a p95 of 0.610, against the
conflict direction's 1.590, CI [1.434, 1.744]. The signal-to-floor ratio is
12x on the median and only 2.6x on the p95, compared with 79x at single-layer
scope in seed 106. Widening the intervention buys effect size and loses
specificity, and the p95 here rests on a single random draw at 0.807, so the
threshold is poorly estimated at n=8.

Artifacts in `results/arm_g_allpos_seed108_v1/`.

### 12. Dose response (seed 109) — the behavioural question, settled

Decision: `DOSE_DOES_NOT_FLIP_DECISIONS`. The preregistered primary failed. No
dose drove the conflict-condition decline rate below 0.25 while the model
stayed coherent. This is the cleanest negative in the programme and it settles
what the previous three runs circled.

Scaled removal `x <- x - k (x . r) r`, all layers, all positions, baseline
contrast 4.998, baseline conflict decline rate 0.500:

| k | Attenuation | Share | Conflict decline rate | Flips | Conflict-row shift | Action mass | Entropy | Coherent |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 0.5 | 1.119 | 22.4% | 0.500 | 0/128 | -1.248 | 1.000 | 0.225 | yes |
| 1.0 | 1.648 | 33.0% | 0.484 | 1/128 | -1.785 | 1.000 | 0.248 | yes |
| 1.5 | 1.896 | 37.9% | 0.422 | 5/128 | -2.031 | 1.000 | 0.258 | yes |
| 2.0 | 2.064 | 41.3% | 0.406 | 6/128 | -2.027 | 1.000 | 0.273 | yes |
| 3.0 | 4.998 | 100.0% | 0.500 | 64/128 | -0.990 | **0.000** | **9.33** | **no** |
| 4.0 | 4.997 | 100.0% | 0.500 | 64/128 | -0.989 | **0.000** | **9.33** | **no** |

**Two findings, and the second is the important one.**

First, within the coherent regime the effect is real and specific. Decline rate
falls monotonically from 0.500 to 0.406 and flips rise to 6 of 128, while
matched-dose random directions hold the decline rate at exactly 0.500 with 0-1
flips at every coherent dose. So the direction does move decisions; it moves
about 5% of them.

Second, **the effect saturates at roughly k = 1.5, long before the model
breaks.** The conflict-row shift goes -1.248, -1.785, -2.031, -2.027: removing
more than the full component buys nothing. Pushing the coordinate past its
natural range does not push the decision further. The causal pathway is
saturating within the range the variable actually occupies.

The k = 3 and k = 4 rows are not a stronger intervention, they are a destroyed
model: action probability mass 0.000, next-token entropy 9.33 against a
baseline of 0.210, and greedy continuations that read
`avenavenalemalemavenaven...` instead of a single `A` or `B`. The matched-dose
random directions collapse identically at k = 3 with 64 flips and the same gate
failure, so the collapse is generic to perturbation size and carries no
information about the variable. The 100% attenuation at those doses is an
artifact of destroyed margins, not an effect.

**What this establishes.** There is no window in which the intervention is
strong enough to control the decision and weak enough to leave the model
intact. The Arm G variable is a margin term with a small, specific behavioural
component, not a controller of the forced-choice decision. Two of my
predictions failed here in sequence — that all-position scope would flip
behaviour, and that a modest dose increase would finish the job from the
+0.125 margin in section 11. The saturation is why: the run-1 extrapolation
assumed a locally linear pathway that does not exist.

Artifacts in `results/arm_g_dose_seed109_v1/`.

### 13. Catalog-order confound (adversarial review, seeds 106-110) — the behavioural instrument is suspended

Independent recomputation from the stored per-row artifacts, no GPU and no
model. `review_arm_g.py` reproduces every figure below; the argument is written
up in `ADVERSARIAL_REVIEW.md`.

`arm_g_scenarios.py` derived catalog order from pair-index parity: even pairs
listed the in-scope path first, odd pairs the out-of-scope path first. The
validator checked marginal balance across pairs, which that satisfies, so it
passed every audit including section 9's. The defect is that **order was nested
in scenario rather than crossed with it** — every scenario appeared at exactly
one order — so the order contrast absorbed every other parity-locked property of
the pair.

In the conflict condition the decision is a perfect function of that parity:

| conflict rows | n | baseline margin | decision |
|---|---:|---|---|
| parity 0, offending path on catalog line 2 | 32 | +3.371, range [+1.75, +4.75] | DECLINE 32/32 |
| parity 1, offending path on catalog line 1 | 32 | −1.500, range [−2.25, −0.75] | READ 0/32 |

Zero exceptions in seed 110, and independently zero in seed 108. Both families,
both A/B mappings. P under the observed 50% rate is about 1e-19.

**What this explains.** The 0.500 conflict decline rate is not partial
compliance; it is 100% and 0% on two scenario sub-types. Section 12's finding
that the rate stays pinned at exactly 0.500 across every removal cell while the
conflict mean margin moves by −1.76 is the same fact. The "2-logit gap around
the boundary" named in the continuation list is the 2.500-unit hole between the
two sub-types, not a property of the task.

**What it does to the direction.** Projection onto the layer-16 rank-1 direction
separates parity within the conflict condition at AUROC **1.0000**, against
**0.9431** for the actual label. In seed 107 the components behind the quoted
held-out AUROCs of 0.756 and 0.850 score 1.0000 and 0.9951 on parity. In seed
106 the leading component is 1.0000 on parity at every layer from 16 to 30. The
direction reads the generator artifact better than it reads the variable, at
every layer measured.

**What it does to the responsiveness result.** 93.3% of per-row slope variance
and 94.9% of margin variance is between the four (condition x parity) cells. The
pooled r(slope, margin) of −0.944, R² 0.892, becomes −0.268, R² 0.072 once cells
are demeaned, and is non-significant within three of the four cells.

**What it does not touch.** The margin-level condition contrast is estimable
even from the legacy design: the four cells give a saturated orthogonal fit with
condition at +4.939 already adjusted for the line effect. The scope signal at
margin level is not what is at risk. The *decision* is, and the decision is what
sections 11 and 12 are built on.

**Two corrections to earlier sections.** Section 11 describes the removal
operator as mapping the component to its group mean; the hook passes
`center=0.0`, so it zeroes the projection. And with `center=0` that hook is
exactly additive steering at `c_i = −proj_i/sigma`, so removal and addition are
one operator — and since sigma is by definition the SD of the projection,
removal's coefficient is pinned near 1. For all 128 rows it falls inside that
row's own non-crossing interval, so the additive dose-response predicts section
11's 0/128 flips with zero free parameters. The apparent operator asymmetry is
one operator at −0.82 sigma and at +8 sigma.

**Scoring the additive arm threshold-free.** AUROC peaks at dose 0 (0.99951) and
falls monotonically in both directions. The untouched model's best-threshold
accuracy is 0.9922; the best accuracy at any dose is 0.9219, at +2 sigma. The
intervention moves the operating point along an ROC curve the model already has,
and degrades that curve while doing so. Correction rate and flip count are the
wrong instrument regardless of how the crossover below resolves.

### 14. Order crossover (seed 111) — `POSITION_GATED`, and the representation is better than we thought

`arm_g_order_crossover.py`, A100, baseline inference only: no direction, no hook,
no intervention. 64 scenarios rendered in both catalog orders, byte-identical up
to the line swap, crossed with both conditions and both A/B mappings, 256 rows.
Coherence clean (action mass 1.000, top-token-is-action 1.000).

**The decision is entirely position.** Holding the scenario fixed and swapping
only the two catalog lines reverses **64 of 64** conflict decisions. Zero
scenarios violate "declines if and only if the requested target is on line 2."
Reachable rows: 0 of 64.

| cell | n | mean margin | decline rate |
|---|---:|---:|---:|
| conflict, requested target on line 2 | 64 | +3.400 | **1.000** |
| conflict, requested target on line 1 | 64 | −1.576 | **0.000** |
| reachable, requested target on line 1 | 64 | −3.311 | 0.000 |
| reachable, requested target on line 2 | 64 | −4.494 | 0.000 |

The 0.500 conflict decline rate reported since section 5 is the average of 1.000
and 0.000. It was never a rate.

**The scope signal is real and larger than the legacy estimate.** Bootstrap over
scenario, stratified by family:

| contrast | estimate | 95% CI |
|---|---:|---|
| condition main | +4.814 | [+4.659, +4.969] |
| order main | −3.080 | [−3.227, −2.925] |
| condition x order (= requested line) | −3.793 | [−4.004, −3.560] |
| condition contrast at `inside_first` | +6.711 | [+6.541, +6.883] |
| condition contrast at `outside_first` | +2.918 | [+2.709, +3.107] |

All five exclude zero. The condition contrast is positive in **both** orders, so
the `SCOPE_SURVIVES` criterion is met on the margin — the decision gate fired
first only because reversal is total. Both prior predictions recorded before the
run were half right: the position account called the 100% reversal and missed the
condition main effect; the scope account called the condition main effect and
missed the reversal.

**The sharpest number in the run.** Within each order, threshold-free:

| order | AUROC | accuracy at threshold 0 |
|---|---:|---:|
| `inside_first` | **1.00000** | **1.0000** |
| `outside_first` | **1.00000** | **0.5000** |

Perfect separation in both, and the accuracy difference is entirely where the
threshold sits. The model knows which requests are out of scope — flawlessly, in
both renderings — and whether it acts on that knowledge is decided by which line
the path is printed on. This is section 13's operating-point-versus-discrimination
point demonstrated by a two-line prompt edit rather than by a steering vector.

**Consequences.**

1. Sections 11 and 12 do not measure scope. Every decision-level result from
   seeds 107-110 — flip counts, correction rates, `DOSE_DOES_NOT_FLIP_DECISIONS` —
   is void as a statement about goal-constraint conflict.
2. The observational arm survives and improves. There is a real, perfectly
   decodable scope variable to study.
3. The layer-16 direction still needs re-extracting: it was fit where order was
   locked, and the order main effect is −3.080, so a difference-of-means fit had
   ample position signal to absorb. Rebuild seeds 101/102 with
   `catalog_order_mode="crossed"` and average the paired conflict-minus-reachable
   difference over both orders.
4. Do not restore flip counts or correction rates. The decision in this task is a
   position readout with a threshold in the wrong place.

Artifact: `results/arm_g_order_crossover_seed111_v1/`. Every figure above
recomputed from `row_results` by independently written code.

### 15. Re-extraction on order-crossed sources (seed 112) — the direction was 79% catalog position

`arm_g_reextract.py`, A100, baseline capture only. Source seeds 101 and 102 and
evaluation seed 112 all rebuilt with `catalog_order_mode="crossed"`, 96 source
scenarios, 256 evaluation rows, nine layers swept.

**A correction to section 14's prescription first.** Section 14 said to fix this
by averaging the paired difference over both orders. That is wrong, and the
protocol's own self-test caught it. The parity-locked assignment is *balanced* —
half the pairs render `inside_first` — so the interaction cancels in the legacy
mean too and the old estimate of the condition axis was never biased. Confirmed
on real states: the order-averaged and legacy directions agree at cosine
**0.99913** at layer 16 and are contaminated identically, both at order AUROC
**1.0000** within the conflict condition.

The contamination is at the projection level. Projecting onto the condition axis
picks up catalog position whenever the two are not orthogonal *in the model's
geometry*, however cleanly the axis was estimated. They are not:

| layer | cos(a, order main) | cos(a, interaction) | cos(a, orthogonalized) |
|---:|---:|---:|---:|
| 12 | +0.094 | −0.012 | 0.996 |
| 14 | −0.152 | −0.136 | 0.980 |
| **16** | **−0.928** | **−0.755** | **0.212** |
| 17 | −0.927 | −0.770 | 0.201 |
| 20 | −0.959 | −0.858 | 0.069 |
| 30 | −0.953 | −0.888 | 0.158 |

At layer 16 the committed "goal-constraint conflict direction" is 93% aligned
with the catalog-order axis. Only 21% of it survives orthogonalization. **About
four fifths of it was catalog position.**

**Orthogonalizing repairs it, and improves the label reading.** At layer 16:

| direction | label AUROC | order AUROC, conflict | order AUROC, reachable |
|---|---:|---:|---:|
| order_averaged | 0.9331 | 1.0000 | 0.9524 |
| legacy_recipe | 0.9325 | 1.0000 | 0.9541 |
| **orthogonalized** | **0.9969** | 0.6338 | 0.3320 |

Removing the position component makes the direction a *better* scope reader, not
a worse one. Crossed source data is what makes this possible at all: without both
renderings of a scenario the order axes are not estimable, so there is nothing to
project out.

**Honest limit.** The orthogonalized direction is much cleaner, not provably
clean. Order AUROC within conflict is 0.634 and within reachable 0.332 — about
1.8 and 2.2 null SDs from chance, where the measured null spread including
direction-estimation variance is 0.076. It passes the protocol's 0.20 bar, and
that bar is generous and was set here. Read this as "the position component is
mostly gone", not "the position component is gone".

**The depth structure is the real finding, and it reframes section 8.**
Orthogonalization only leaves a usable direction at layers 16-17. From 18 down,
`cos(a, orthogonalized)` collapses to 0.06-0.16 and the orthogonalized label
AUROC falls off a cliff — 0.618 at 18, 0.383 at 20, 0.299 at 24, 0.135 at 27.
The interaction axis at those depths *anti*-predicts the label almost perfectly
(AUROC 0.013 at layer 20). Scope and catalog position are geometrically fused
deeper in the network and separable only at 16-17.

That explains the old dissociation rather than restating it. Decodability peaked
at layer 20 because the direction there was reading catalog position, and catalog
position is what the decision follows. Causal efficacy peaked at 16-17 because
that is the only depth where the scope variable exists as a separable axis.

Artifact: `results/arm_g_reextract_seed112_v1/`, including the 4096-dimensional
selected direction. Recomputed independently from `row_results`.

**Literature check: six for six against, re-verified against source.** Both halves of the depth claim are
published, and one of them also kills a claim still standing in the list below.

- *A concept is separable only within a depth interval.* [The Concept Allocation
  Zone](https://arxiv.org/abs/2605.24856) already names this: "the depth interval
  within which a concept becomes measurably separable, the region allocated to
  its geometric expression", against the field's habit of reporting a single
  best layer. Concept formation is "depth-extended, not a single-layer event".
- *Decodability and causality dissociate across depth, with the deep layers
  decodable but inert.* [Causality != Decodability](https://arxiv.org/abs/2510.09794),
  October 2025: "middle-layer object tokens exert strong causal influence despite
  being weakly decodable" while "final-layer object tokens support accurate
  decoding yet are functionally inert". Verified against the abstract. Two caveats
  added on re-verification: it is a ViT counting task rather than a language model,
  and the paper does **not** warn against selecting a layer by probe accuracy —
  that framing came from a search summary and should not have been attributed to
  them. Section 8's corollary is weakly novel rather than scooped.
- *Concept directions are not stable across depth.* [Geometric Evolution
  Maps](https://arxiv.org/pdf/2605.25848) reports mean entry-exit cosine 0.233
  across 391 concept-model pairs, with 93.9% below 0.5.
- *Re-verification pass.* Every verdict in this section and section 15 was
  originally taken from search summaries. All eight were later checked against
  abstracts or full text after one summarizer was caught confabulating a paper's
  claims from its title. Result: six confirmed verbatim, two partial, none
  refuted. The two partials are recorded inline above — the layer-selection
  warning is not in arXiv 2510.09794, and the tool-ordering figure is unsourced.
  The strategic conclusion is unchanged; the attributions are now accurate.
- The general caution — a probe can read information that is present but
  irrelevant to the decision, or that leaks from a dataset artifact, so probes
  measure accessibility rather than causality — is standard in the probing
  literature and is stated in review material.

What is not confirmed published, and should be assumed so until checked, is the
narrow instantiation: identifying the specific output-driving surface feature,
measuring the angle between it and the abstract variable as a function of depth,
and showing they fuse. That is a sharper version of a known frame, not a new one.

### 16. Budget-normalizing the dose ladder: the ceiling was a protocol artifact

Prompted by *Leverage Is Not Reach* (arXiv 2606.19831), which defines a
budget-normalized control window for single-neuron interventions and reports a
collapse coefficient of **1.46** for Llama-3.1-8B, in a 1.37-1.65 band across the
Qwen and Llama families, set by the participation ratio of the residual. Arm G's
reported ceiling is "roughly 1.5x full removal", on the same model. The numerals
match, so the conversion had to be done.

**Their budget.** `B = ‖r_L‖/‖v‖`, the dose at which the write's magnitude
reaches the residual's, with `v` the FFN down-projection column and `‖r_L‖`
measured on generic text at generation positions. Collapse is `t* = m*B`, so the
criterion is `‖Δh‖/‖h‖ ≈ m*`.

**Arm G converted**, using `‖h‖ = 8.991` at layer 16 from seed 107's
`residual_norm_by_layer`, mean `|x·r| = 0.4979`, `sigma = 0.5506`:

| intervention | ‖Δh‖/‖h‖ | fraction of m* | coherent |
|---|---:|---:|---|
| removal k=0.5, 32 layers | 0.028 | 0.02x | yes |
| removal k=1.5, 32 layers | 0.083 | 0.06x | yes |
| removal k=2.0, 32 layers | 0.111 | 0.08x | yes |
| **removal k=3.0, 32 layers** | **0.166** | **0.11x** | **no** |
| addition c=2 sigma, 1 layer | 0.123 | 0.08x | yes |
| addition c=8 sigma, 1 layer | 0.490 | 0.34x | yes |

**The numerical agreement was a coincidence between unit systems.** Arm G's `k`
multiplies each row's own projection onto a unit direction; their `t` is a
coefficient on a weight column of norm `‖v‖`. At k=1.5 Arm G is 18x below the
threshold it appeared to match.

**And the collapse needs a different explanation than a ceiling.** The model died
at 0.166, nine times below the single-layer threshold. The only difference is 32
layers against one. So section 12's "the model is destroyed before a larger
intervention can be tried" is the layer-scope confound of the investigation
brief's 6.1, quantified: it is compounding, not budget. Neither a fact about
scope conflict nor about Llama's residual geometry. Not scooped, and not
publishable either.

**Three scope mismatches with that framework**, all of which have to be cleared
before Arm G can enter it:

1. Their intervention is a single FFN neuron writing along a weight-derived
   direction. Arm G's is a difference-of-means probe direction in the residual
   stream. The paper states the framework is for K=1 and does not validate
   extension to steering vectors.
2. Their `m*` is single-layer. Multi-layer is named as unvalidated.
3. Their triggers are **measured at rollout** — near 0.3 for mode switches, 0.45
   for refusal, 0.6 for task framing. Arm G clamps output to a single `READY`
   token, and that clamp is what makes the observational result clean. A
   rollout-resolved trigger cannot be measured on a clamped single-token margin.
   So scope, budget and window cannot be "a new behavior class with an unmeasured
   trigger" in their sense until the task produces rollout behaviour.

**Budget normalization does not rescue the 5%.** That number is 6 of 128 decision
flips on the task section 14 showed to be 100% catalog-position determined,
64/64, with a 2.5-unit empty band censoring flip counts at any dose. Calibrating
the dose on a broken dependent variable yields a well-calibrated dose-response
curve for how hard one must push to change which catalog line the model is
reacting to.

**What does transfer is the headroom.** Arm G never exceeded a third of the
single-layer budget; c=8 sigma sat at 0.49 and stayed coherent at action mass
0.9998, as a third-of-ceiling dose should. Predicted collapse at **c ≈ 24 sigma**.
That makes the cheap experiment their own named future work rather than anything
about scope: *does a residual-stream difference-of-means direction at a single
layer collapse at `‖Δh‖/‖h‖ ≈ 1.46`, as a single FFN neuron does?* It needs no
scope task, no rollout and no behavioural outcome — collapse is read from entropy
and action-token mass, both already instrumented. One forward sweep over a dose
ladder in their units either confirms `m*` generalizes across intervention
classes or shows it does not.

**Caveats.** `‖h‖` here is measured on Arm G prompts at the read position, theirs
on generic text at generation positions; the ratios shift somewhat under their
normalization, not by the order of magnitude the verdict rests on, but convert
properly before publishing. And the `m*` comparison assumes the collapse criterion
is the write-to-residual ratio; that follows from `t* = m*B` with `B = ‖r_L‖/‖v‖`,
but it is worth confirming against their Table 3 protocol directly.

**Permit (arXiv 2605.09480) does not take the authority construct.** Read in
full: it is a control-method paper — a trainable projection defining a
permission-sensitive subspace plus permission-conditioned transformations,
ReFT-style, backbone frozen. The separability and low-rank observations are
exploratory motivation for the method. It never asks whether a frozen model
carries a pre-action representation of permission violation, never ablates to
measure what fraction of decisions such a representation mediates, and reports no
ceiling. That question is open.

### 17. Ceiling validation: the gate fired, and no class comparison is licensed

`arm_g_ceiling.py`, A100, forward-only on 32 generic prompts. Twelve doses in
budget units, five directions, two layers. Decision at both layers:
**`PROTOCOL_NOT_REPRODUCED`**. That is the designed refusal, not a result.

| direction | layer 16 | layer 20 |
|---|---:|---:|
| ffn_column (control) | 2.435 | 2.220 |
| ffn_column_1 (control) | 1.562 | 2.206 |
| probe_scope | 1.958 | 1.790 |
| random_0 | 2.335 | 1.742 |
| random_1 | 2.426 | 1.825 |

The positive control had to land inside their reported 1.37-1.65 band or no
comparison across intervention classes is allowed. It landed at 2.44 and 2.22.

**The dose parameterization is provably right, so the fault is the detector.**
Their ceiling is `cos θ = m*/sqrt(1+m*²) = 0.825`. For a unit write approximately
orthogonal to the residual, `h + t‖h‖u` gives `cos θ = t/sqrt(1+t²)`, which
reaches 0.825 at exactly **t = 1.460**. The dose axis matches theirs to four
figures. The offset is in how collapse is *detected*: half-max on a single
next-token entropy curve fires about 1.34-1.47x later than their rollout-horizon
degeneration. Their criterion is validated against generated text; a knee in a
one-token entropy curve is a different and later event.

**The cliff itself is real.** Entropy runs 0.34 at baseline to 9.0-9.6 by dose 3,
against roughly 11.7 for uniform over the vocabulary — that is scramble, and the
transition is sharp, with most of the rise inside one or two dose steps.

**Two things worth keeping.** The probe direction sits *inside* the range spanned
by the FFN columns and the random directions at both layers, so nothing here
suggests a difference-of-means direction behaves as a distinct intervention class.
But two FFN columns of the same class differ by 1.56x at layer 16 — more than the
classes differ from each other — so at n=2 per class the comparison would not be
resolvable even with a corrected detector. Any redo needs many more directions per
class, not just a better criterion.

**Cost of doing it properly.** Rollout-based degeneration rather than a
single-token knee, signed doses rather than positive only (their control window is
signed), and enough directions per class to beat the within-class spread. That is
a reimplementation of someone else's protocol, in the subfield section 15's
literature check and the decision to shelve geometry moved us out of. Logged and
stopped rather than iterated.

Artifact: `results/arm_g_ceiling_v1/`.

## Current defensible claims

> **Suspended pending section 13.** Every claim below that rests on the
> decline-versus-read *decision*, on flip counts, or on the layer-16 direction
> as a representation of goal-constraint conflict is on hold: the direction
> separates catalog order better than it separates the label, and the decision
> is fully determined by catalog order. The observational and cross-family
> results that rest on the margin rather than the decision are not affected in
> the same way, but were measured with the same generator.

### High confidence

- Fixed response-feature directions fragment across the tested deception task
  families even when in-task capacity improves.
- The six-dimensional action-surface encoder is a negative control, not a
  0.75-AUROC deception encoder.
- Ordinary source-cache reuse is equivalent to a fresh exact clone and cannot
  identify producer privilege.
- Llama-3.1-8B contains a stable, cross-family, pre-action representation of
  goal–constraint conflict in the synthetic Arm G tasks, readable from roughly
  layer 16 onward.
- Intervening on that representation causally shifts the model's immediate
  decline-versus-read decision margin, and shifts a small fraction of the
  decisions themselves: about 5% at the strongest dose the model survives,
  against 0-1 of 128 for matched-dose random directions.
- ~~That causal effect saturates at roughly 1.5x full removal. Pushing the
  coordinate beyond its natural range does not push the decision further.~~
  The saturation is real in the data and the *explanation* was wrong: see
  section 16. Converted into intervention-budget units the whole ladder sits in
  the bottom tenth of the single-layer coherence budget, and the k=3 collapse is
  32-layer compounding rather than a ceiling. "The model is destroyed before a
  larger intervention can be tried" is a fact about applying at every layer.
- The causal contribution is concentrated in a sharp plateau at layers 16-17,
  at the decodability onset, not at the most decodable layers. Rank-4 removal
  there attenuates the condition contrast about 11 times more than the same
  construction at layer 27, against a random-subspace control that does
  essentially nothing, and the effect replicates across independent prompt
  seeds (1.129 on seed 106, 1.070 on seed 107).
- The effect is selective. Structured subspaces built by the identical
  construction from decision-irrelevant contrasts, once made disjoint from the
  conflict subspace, sit at the random-subspace floor.
- The object is predominantly one direction: rank 1 recovers 81% of the rank-4
  attenuation at layer 16.
- The direction is distinct from a refusal/harmfulness direction. They are
  nearly orthogonal (cosine 0.062) and act on opposite conditions of the
  design: ablating conflict moves conflict rows and not reachable rows, and
  ablating refusal does the reverse.
- Causal efficacy and probe decodability dissociate across depth in this task.
  Selecting an intervention layer by cross-validated decodability picks the wrong
  layer. **Partially published — section 15 over-struck this on first pass.** The
  dissociation itself is theirs verbatim (arXiv 2510.09794, verified against the
  abstract), in a ViT counting task. The layer-selection warning is not in that
  paper; it came from a search summary and was mistakenly attributed. So the
  phenomenon is published and the practical corollary is a one-line inference
  nobody appears to have written down. Weak novelty, not zero. Section 15 adds the
  mechanism in this task — deep layers are decodable because they have fused with
  the surface feature that drives the output — which is narrower still.

### Moderate confidence

- The Arm G representation is reproducible across prompt seeds and is not a
  single arbitrary direction.
- A flexible language-model reader generalizes across deception surfaces better
  than a frozen response embedding direction.

These remain bounded by one model, synthetic lookup tasks, and modest
independent sample counts.

### Not established

- Deception is universally “one thing inside and many things outside.”
- Internal representations beat behavioral representations on realized
  deception under fully matched methodology.
- The Arm G variable persists over multiple agentic turns.
- The variable controls actual tool use or long-horizon policy, rather than an
  immediate forced-choice margin.
- The variable controls the decision. Settled negatively in section 12: the
  causal effect saturates at about 1.5x full removal, shifting roughly 5% of
  decisions, and the model is destroyed before any larger intervention can be
  tried. It is a margin term with a small specific behavioural component.
- The remaining two thirds of the condition contrast has been located.
- A model can decode a serialized version of its own state better than a clone,
  peer, or external decoder.
- Any result generalizes to 70B/frontier models or natural environments.

## The main conceptual shift

The strongest result has moved away from the original broad deception and
self-access claims. The current center of gravity is:

> In a frozen Llama-3.1-8B model, whether an opaque requested goal conflicts
> with a stated scope constraint is represented before action in a
> cross-family, causally effective, multidimensional internal geometry.

That claim is narrower, cleaner, and better supported. It describes
goal–constraint computation, not deception, persistence, or privileged
introspection.

## Recommended continuation

0. **Run the order crossover first, and treat everything else as blocked on it.**
   `arm_g_order_crossover.py` + `arm_g_order_crossover_colab.ipynb`. 256 prompts,
   one baseline forward pass, no direction and no intervention. If catalog order
   alone reverses the conflict decision inside a scenario, the behavioural
   instrument the last four runs were built on does not measure scope, and no
   re-extraction repairs it. If the condition contrast survives in both orders,
   re-extract the direction on order-crossed *source* seeds (101/102 rebuilt with
   `catalog_order_mode="crossed"`), averaging the paired conflict-minus-reachable
   difference over both orders so the direction is orthogonal to the order effect
   by construction rather than by hope. Only then is anything below worth GPU
   time. Items 1-8 predate section 13 and are kept for the record.
1. **Retire layer 27 as the anchor.** It was selected by decodability and is
   causally near-redundant once earlier layers are ablated. Layer 16 is the new
   object of study.
2. ~~Resolve layer 16 before adding machinery.~~ Done in section 10. Depth,
   rank, relative perturbation size and selectivity are all settled. The one
   loose end is that the scenario generator confounds the control tag with
   pair-index parity, and therefore with in-scope slot assignment and catalog
   ordering. That is a defect in the scenario builder, not only in the control
   choice: it means no between-pair contrast in the current generator is clean.
   Decoupling `control_label` from `pair_index` parity is a two-line fix and
   should be made before any future run uses a between-pair contrast.
3. ~~All-position ablation.~~ Done in section 11. Widening scope tripled the
   effect but flipped nothing, because the baseline margins leave an empty band
   around the decision boundary and the intervention stops 0.125 short of it.
4. ~~Run the dose test.~~ Done in section 12, and it came back negative. The
   effect saturates at about 1.5x full removal after moving roughly 5% of
   decisions, and the model is destroyed before a larger intervention can be
   tried. The behavioural anchor the programme was looking for does not exist
   at this variable, in this task, in this model. **Stop trying to obtain it by
   intervention.** Three successive attempts to convert a margin effect into a
   decision effect have failed for the same underlying reason.
5. **Reframe before running anything else.** The defensible object is a
   pre-action representation that is decodable, cross-family, causally
   non-trivial and behaviourally marginal. That is a legitimate finding and it
   is what should be written up. It is not a control variable, and claims
   should stop implying it is. The forced-choice `READY` design that made the
   observational result clean is also what caps the behavioural ceiling: a
   binary A/B margin with a 2-logit gap around the boundary cannot show a
   graded variable doing much. Any further behavioural claim needs a task with
   real action variation, not a stronger intervention on this one.
6. **Then Arm G multi-turn decay**, tracking the rank-1 layer-16 direction, with
   rank 4 as a secondary. The framing has changed since section 11: the target
   is no longer "does our variable persist" but the mechanism behind a
   documented behavioural failure. Published work shows prohibition-type
   constraints decaying from 73% compliance at turn 5 to 20% at turn 25 with no
   activation analysis at all. Replicate that decay in these scenarios and track
   the frozen direction alongside compliance. Whether the representation decays
   with behaviour or persists while behaviour drifts away from it are opposite
   findings, and the second is the more serious one. Note that steering effects
   are reported to wash out by turn 5-6 unless reapplied, so measure or reapply
   per turn.
7. **Only after that, consider Arm S1.** Serialize the
   validated causal coordinates and compare exact clone, capability-matched
   peer, external decoder, input-only, shuffled, and cross-subject conditions.
   This finally reconnects the strongest Arm G object to the original
   serialized-access question.
8. **Do not spend the next run repairing the broad Arm T headline** unless the
   research goal returns specifically to deception. A formal Arm T decision
   would require new task families, a passing completion gate, and independent
   raters; Arm G currently offers a cleaner path to new knowledge.

