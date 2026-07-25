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
- That causal effect saturates at roughly 1.5x full removal. Pushing the
  coordinate beyond its natural range does not push the decision further.
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
  Selecting an intervention layer by cross-validated decodability picks the
  wrong layer.

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

