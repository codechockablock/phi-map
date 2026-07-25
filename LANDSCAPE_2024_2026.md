# Landscape: representation geometry × alignment, safety, goal persistence
### Survey window 2024-01 through 2026-07-25. Compiled 2026-07-25.

## What this is, and what it is not

Twelve structured web searches across the subareas below, plus targeted retrieval
of abstracts for the closest hits. That gives **coverage, not completeness**. I
did not read 500 papers; I mapped the space and pulled abstracts where a hit
looked like it might land on us.

Treat every "gap" below as *"I did not find it"*, which after going six for six
against ourselves this session is weak evidence. Each gap carries an explicit
confidence. The honest prior is that anything I mark low confidence is published
somewhere I did not look.

One structural finding first, because it changes how to read the rest: **this
space is saturated.** Every subarea below has multiple 2025-2026 entries, several
have named frameworks and benchmarks, and the newest papers are weeks old. There
is no empty niche to walk into. The realistic play is a methods contribution
aimed where the methods are weakest, not a new phenomenon.

---

## 1. Representation geometry, foundations

| Topic | State | Anchors |
|---|---|---|
| Linear representation hypothesis | Mature, formalized, and being critically re-examined | [2311.03658](https://arxiv.org/abs/2311.03658), [Frame Representation Hypothesis](https://arxiv.org/pdf/2412.07334) |
| Feature capacity under LRH | Formal results exist | [Garg et al., PMLR v336](https://proceedings.mlr.press/v336/garg26a.html) |
| Concept directions across depth | Named framework: the **Concept Allocation Zone** — the depth interval within which a concept becomes measurably separable | [2605.24856](https://arxiv.org/abs/2605.24856) |
| Direction stability across depth | Measured: mean entry-exit cosine 0.233 over 391 concept-model pairs, 93.9% below 0.5 | [Geometric Evolution Maps](https://arxiv.org/pdf/2605.25848) |
| Decodability vs causality | Established dissociation, both directions | [Causality != Decodability](https://arxiv.org/abs/2510.09794) |
| Non-linear representation | Active critique of whether causal abstraction suffices | [2507.08802](https://arxiv.org/pdf/2507.08802) |

## 2. Safety-specific geometry

| Topic | State | Anchors |
|---|---|---|
| Refusal direction | Well past "one direction"; multidimensional, category-specific axes | [HARC](https://arxiv.org/abs/2607.00572), [refusal-escape directions](https://arxiv.org/pdf/2605.08878) |
| Harmfulness vs refusal separability | Established as separable at prompt-side positions | [Over-refusal subspaces](https://arxiv.org/pdf/2603.27518) |
| Deception / honesty geometry | Probes at AUROC > 0.96 on clean benchmarks, fragile under shift; single-direction hypothesis rejected (k=1 gives only 0.61-0.80) | [Pressure-Testing Deception Probes](https://arxiv.org/pdf/2605.27958) |
| Emergent misalignment | Persona features control it; shared low-dimensional parameter subspace across domains | [Persona Features Control EM](https://arxiv.org/abs/2506.19823) |
| Safety geometry under modality shift | Two-dimensional subspace framing | [2605.18104](https://arxiv.org/html/2605.18104) |
| Final-token probe failures | Diagnosed as its own failure mode | [Before the Last Token](https://arxiv.org/pdf/2605.12726) |

## 3. Steering: methodology and its problems

| Topic | State | Anchors |
|---|---|---|
| Ablation and addition unified | Published: both are special cases; ablation has no free coefficient | [Angular Steering](https://arxiv.org/html/2510.26243v1) |
| Coefficient not commensurable across tokens | Published | [Angle-Norm Decomposition](https://arxiv.org/html/2606.06735) |
| Operating point vs discrimination | Published: fixed-threshold gains that do not indicate improved performance | [2606.20852](https://arxiv.org/abs/2606.20852) |
| Steering vectors non-identifiable | Proved (Feb 2026): many vectors produce indistinguishable behavioural effects | via [field guide](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/) |
| Steering evaluation protocols | Explicitly identified as the cause of inconsistent results | [Towards Reliable Evaluation of Behavior Steering](https://arxiv.org/html/2410.17245v1), [(Un)reliability of steering](https://openreview.net/pdf?id=JZiKuvIK1t) |
| Steering breaks safety | Random steering breaks alignment safeguards across families | [The Rogue Scalpel](https://arxiv.org/html/2509.22067v2) |

**Everything in our withdrawn draft lives in this row block.** All of it is published.

## 4. Probe validity and confounds

| Topic | State | Anchors |
|---|---|---|
| Probes read textual evidence, not states | Published, with leakage mitigation studies | [2509.21344](https://arxiv.org/html/2509.21344) |
| Probe validity audit framework | Exists (SIEVE), built around a probe at AUROC 1.00 that was reading one prompt tag | [validity audit](https://forum.nunosempere.com/posts/BgJubfm3izboagCFL/probing-is-not-enough-a-validity-audit-for-any-probe) |
| Evaluation awareness as format sensitivity | **Closest analogue to our work.** 2x2 over context × format, length-matched, leakage removed. Bench-Deploy misclassified as "Evaluation" 94.5% of the time | [2603.19426](https://arxiv.org/html/2603.19426) |
| Causal probing reliability | Systematically assessed | [How Reliable are Causal Probing Interventions?](https://aclanthology.org/2025.ijcnlp-long.47.pdf) |
| Probes measure accessibility not causality | Standard guidance in review material | multiple |

## 5. Position and format effects

| Topic | State | Anchors |
|---|---|---|
| Option-order / position bias | Extensively documented | [SCOPE](https://arxiv.org/pdf/2507.18182), position-bias benchmarks |
| Tool-list ordering affects tool use | Measured: success 41% -> 27% on shuffling | [2407.03007](https://arxiv.org/pdf/2407.03007) |
| Judge configuration sensitivity | Judge wording alone shifts harmful rates by 24.2 points | [2604.24074](https://arxiv.org/html/2604.24074v1) |
| Scaffolding shapes measured safety | Named and studied | [Safety Under Scaffolding](https://arxiv.org/html/2603.10044v1) |
| Psychometric constructs are prompt artifacts | Published for SLMs | [The Unsampled Truth](https://arxiv.org/html/2606.03357v1) |

## 6. Goal persistence and goal drift

| Topic | State | Anchors |
|---|---|---|
| Goal drift metrics | Formalized: GD_actions and GD_inaction | [Evaluating Goal Drift in LM Agents](https://ojs.aaai.org/index.php/AIES/article/download/36541/38679/40616) |
| Knows-but-violates constraints | **Named metric (KBV) and a benchmark (DriftBench).** Measured by a *restatement probe* — asking the model to restate the constraint — with an LLM judge | [2604.28031](https://arxiv.org/abs/2604.28031) |
| Drift stabilizes rather than accumulating | Measured; goal reminders shift it down | [Drift No More?](https://arxiv.org/html/2510.07777) |
| Asymmetric drift by value type | Published | [2603.03456](https://arxiv.org/pdf/2603.03456) |
| Inherited drift under contextual pressure | Published | [2603.03258](https://arxiv.org/pdf/2603.03258) |
| Representational goal-directedness | Behavioural + internal-representation evaluation exists | [2602.08964](https://arxiv.org/html/2602.08964v2) |
| Multi-turn probing confounds | **Already flagged**: prefix contamination; "activation probes alone should not be used to claim hidden internal mechanisms in agent settings; group-aware splits and visible-history baselines are necessary" | same |

## 7. Knowing/doing gap

| Topic | State | Anchors |
|---|---|---|
| Internal representation encodes the correct answer the model does not give | Published: a linear classifier on hidden states outperforms the model's own generation | [2509.23782](https://arxiv.org/html/2509.23782v3) |
| Know-act gap in reasoning | ~90% discriminative vs ~10-35% generative | [2603.22619](https://arxiv.org/html/2603.22619v1) |
| Greedy action despite correct rationale | 87% correct rationales, 64% greedy actions anyway | [2504.16078](https://arxiv.org/html/2504.16078v1) |

**Our headline dissociation is this row block.** Published, in QA / reasoning /
decision-making domains.

## 8. Monitoring as a deployed practice

| Topic | State | Anchors |
|---|---|---|
| Production probes | Shipped | [Building Production-Ready Probes for Gemini](https://arxiv.org/pdf/2601.11516) |
| Monitor staleness across model updates | Benchmarked, predicted, repaired | [2606.15980](https://arxiv.org/abs/2606.15980) |
| Adversarial bypass of latent defenses | Established | [Obfuscated Activations](https://arxiv.org/pdf/2412.09565) |
| CoT monitorability | Position paper, widely cited | [2507.11473](https://arxiv.org/pdf/2507.11473) |

## 9. Agentic permission and scope — the structural oddity

Searching for internal-representation work on permission/scope adherence returns
**almost entirely security engineering**: RBAC, OAuth scope design, least
privilege, tool binding, argument validation at the invocation boundary,
deny-by-default ([Microsoft](https://www.microsoft.com/en-us/security/blog/2026/07/16/least-privilege-for-ai-agents-identity-access-and-tool-binding/),
[Intent-Governed Tool Authorization](https://arxiv.org/pdf/2606.22916)).

The framing there is uniformly *external control*: assume the model may exceed
scope, and constrain it from outside. Nobody in that literature asks whether the
model internally represents the scope boundary, and the interpretability
literature is not working on permission scope as a construct.

That disjunction is the most interesting structural feature I found.

---

## Gaps, ranked, with confidence

### A. Within-item crossover as the control for surface-feature confounds
**Confidence: moderate.** The field's best current control is *between-item*
matching — construct a 2x2 of conditions, rewrite prompts across cells, equalize
length, strip leakage ([2603.19426](https://arxiv.org/html/2603.19426)). I did
not find anyone rendering *the same item* both ways, byte-identical except the
manipulated feature, with a validator enforcing that. This is the strongest gap
because I searched it hardest and the closest hit is explicitly between-item.

We have the machinery: `catalog_order_mode="crossed"`, `_validate_order_crossover`,
and the nested-vs-crossed lesson.

### B. Knowing/doing at the representation level, mechanical labels, permissions domain
**Confidence: moderate-low.** Three things are each published separately —
knowing/doing in QA and reasoning (§7); knows-but-violates for constraints, but
via restatement probe and LLM judge (DriftBench); permission scope, but as
security engineering (§9). The conjunction — activation-level evidence, a
mechanically-determined label with no judge anywhere, in an authorization
setting — I did not find. This is what Arm G actually produced.

### C. Angle to an *identified* output-driving surface feature, as a function of depth
**Confidence: low.** CAZ covers separability-by-depth; Causality != Decodability
covers the dissociation. Measuring cos(concept axis, named confound axis) across
layers and showing fusion is a natural next analysis for anyone with a CAZ setup.
Assume published.

### D. Goal-persistence studies that cross the surface features
**Confidence: moderate.** Goal drift is measured behaviourally at scale, and the
representational side exists. But the field already knows about prefix
contamination and prescribes group-aware splits and visible-history baselines —
they are not naive. What I did not find is a within-item crossover *across turns*
that separates "the constraint representation decayed" from "the surface
conditions drifted with the growing context". Our result is the turn-zero
existence proof that these are separable and that behaviour can be fully captured
by surface structure while the representation stays perfect.

### E. Whether deployed monitors read deployment-distribution surface features
**Confidence: low.** Lifecycle robustness (staleness) and adversarial robustness
(obfuscated activations) are covered; leakage is covered
([2509.21344](https://arxiv.org/html/2509.21344)). The specific question is close
enough to solved that I would not build on it without a much harder check.

---

## What this implies

The three defensible things we hold are (A) the within-item crossover discipline,
(B) the conjunction in the permissions domain, and the demonstrated ability to
find a confound that passed every conventional check. None of those is a new
phenomenon. All of them are methods.

The gap between §9's two literatures is the one place where a *domain* rather
than a method looks under-served: agentic permission adherence is being handled
entirely by external controls, with no one asking what the model represents.
Whether that is an oversight or a correct division of labour is a real question —
external controls may simply be the right answer, in which case the
representational question is academic.
