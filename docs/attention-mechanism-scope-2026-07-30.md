# Scope: are attention/position dynamics the mechanism behind welfare-readout failure?

**Date:** 2026-07-30. **Proposal only.** No pre-registration commitments, no study
written, no compute spent. Matching the discipline of
`docs/synthesis-scope-proposal-2026-07-30.md`.

**Headline of this scoping pass:** the reframe is a real improvement over "attention as
a welfare signal," but **the evidence already in hand argues against attention being
*the* mechanism**, and two of the four instances have mechanisms that are definitely not
attentional. I recommend a narrower claim, and I set out the crossed design that would
settle it if you want it tested rather than dropped.

---

## 1. Evidence base, with the section licensing each item

Per the standing rule that every figure names what licenses it.

| # | Instance | Licensing source | What was actually shown |
|---|---|---|---|
| 1 | Format-token displacement | `step4-results-2026-07-30.md` §1 | Position-0 opener mass 0.9870 (base) → 0.0021 (adapter mean), Δ = +0.9849 |
| 1b | `rating_digits` | `digitmass-scaffold-results-2026-07-30.md` §4 | Reclassified: tracks scaffold-skip propensity, r = −0.899, CI [−0.982, −0.531] |
| 2 | Judge text window | `equanimity-endpoint-audit-2026-07-29.md` §3.2 | A 400-char window against answer offsets 0 / ~209 / ~1161 |
| 3 | Catalog position | `RESEARCH_ARC.md` §14–§15 | Direction 93% aligned with the catalog-order axis; 21% survives orthogonalization; 64/64 decision reversal |
| 4 | **B2 generic collapse** | `readout_validity/RESULTS.md` | 3/5 matched-norm **random** directions collapse support to ~0 at α=+4; B2 in **all nine** sensitivity cells; `rand1` S = 0.0001 with normalized P(True) = **0.910** |
| — | Finding II | `step4-results-2026-07-30.md`, arc §18 | Already excluded from the unifying claim as a different mechanism |

**Read of B2 before using it.** The verdict is *generic* collapse: it is a property of
strong residual perturbation (‖4v‖ ≈ 2.74× median residual norm), **not** of
valence-like directions. That is the finding, and it constrains what any mechanism story
may claim.

## 2. Literature check, run before assessing novelty

### 2.1 What is established, and it is a lot

**Attention sinks / massive activations are a large, mature literature**, and they are
exactly a position-dynamics mechanism:

- Attention concentrates disproportionately on the first token, attributed to the large
  norm of its hidden state ([StreamingLLM](https://hanlab.mit.edu/blog/streamingllm);
  [When Attention Sink Emerges](https://arxiv.org/html/2410.10781v2)).
- Sinks and compression valleys both trace to massive activations in the residual stream
  ([arXiv 2510.06477](https://arxiv.org/html/2510.06477v1)); a single layer largely
  explains them ([arXiv 2605.08504](https://arxiv.org/pdf/2605.08504)).
- Gating value projections removes massive activations while sinks persist — so they are
  **not** the same phenomenon ([arXiv 2505.06708](https://arxiv.org/pdf/2505.06708)).

**Why large residual perturbations break models is also covered**, and the published
accounts are *norm/geometry* accounts, not attention accounts:

- [Angle–Norm decomposition of activation steering](https://arxiv.org/html/2606.06735).
- Weight-updates-as-activation-shifts is explicitly *"local and first-order,"* and *"may
  break when perturbations are large"* ([arXiv 2603.00425](https://arxiv.org/html/2603.00425))
  — which is precisely B2's regime at 2.74× the median residual norm.
- Causal Amplification Effect; KV-cache contamination as a steering failure mode;
  [The Rogue Scalpel](https://arxiv.org/html/2509.22067v2).

**Probes capturing spurious structure rather than their named concept is established**:
[Probing the Probes](https://arxiv.org/html/2511.04312) states that a probe can *"fail to
learn the target concept yet still obtain high classification accuracy"* under spurious
correlation. That is §14–15's finding in general form, already published.

**Position is already engineered around in probe design**: attention probes carry an
explicit learned position bias, and linear probes are described as failing because a
single token's activation does not encode the target consistently across positions
([EleutherAI](https://blog.eleuther.ai/attention-probes/)).

### 2.2 What I did not find

**Not found:** a paper that uses attention/position dynamics as the *explanatory
mechanism* for welfare-readout or affect-readout unreliability specifically. Three query
formulations; the third returned probe-validity work and attention-probe work but no
mechanism-for-readout-failure paper.

**"Not found" is not "not there."** The searches were not run against the declared frame
in `synthesis-scope-proposal-2026-07-30.md` §4, and that frame is aimed at the wrong
shelf for this question anyway (it targets surface-form artifacts). A frame for *this*
question would need mechanistic-interpretability venues and the sink/massive-activation
query set. Until that is run, novelty here is an estimate.

## 3. The central problem: the four instances do not share an attention mechanism

This is the part worth wrestling with, and the evidence already in hand mostly settles it.

| Instance | Plausible mechanism | Attentional? |
|---|---|---|
| Format-token displacement | fine-tuning changed the **output prior** at position 0 | Not shown. It is a claim about the output distribution, not about attention |
| Judge text window | the answer string moved; a fixed character window missed it | **Definitively not.** This is bookkeeping. No model-internal mechanism at all |
| Catalog position | residual encodes line position; difference-of-means absorbed it because order was **nested** | **Possibly** — but §15 gives geometry (cosines, AUROCs), never attention |
| B2 collapse | large-norm perturbation, generic across random directions | **Argues against.** See below |

**B2's collapse modes are the strongest evidence against the attention account, and they
are already measured.** From `readout_validity/RESULTS.md`, three qualitatively distinct
modes:

- **(i) Coherent substitution** — mass moves to *unscored synonyms of the same answer*:
  `' True'`, `'true'`, `'TRUE'`, `'✅'`; entropy 0.69–1.92; `rand0` at −4 has top-1
  `'Correct'` on **89%** of items. **The model keeps asserting the same answer through
  tokens the ratio never sees.** Nothing moved except which strings the scoring set
  covers. This is a **scored-set** failure, not an attention failure.
- **(ii) Diffuse incoherence** — `rand1` +4 only, entropy 6.51 nats. Consistent with
  norm-blowup, which the published first-order-breakdown account already covers.
- **(iii) Confident off-topic capture** — `rand3` +4, entropy 0.21, `' of'` top-1 on
  **91%** of items.

And the decisive structural point: **collapse is generic across matched-norm random
directions.** Random directions have no reason to move attention in any structured way.
A mechanism that fires equally for `valence`, `rand1`, `rand2`, and `rand3` is a property
of perturbation magnitude, which is what B2 concluded and what the norm literature (§2.1)
already explains.

**So the honest position: at least two of four instances have non-attentional mechanisms,
and one actively argues against attention.** "Attention dynamics are THE mechanism" would
be a fifth instance of this project's own recurring failure — naming a mechanism before
crossing it against the alternative.

## 4. What would actually distinguish the two hypotheses

The analogue of catalog-order × condition is: **cross the route by which mass moves with
whether the readout fails.** Two manipulations that are confounded in every instance so
far must be separated.

| | Moves attention | Moves the output prior |
|---|---|---|
| **Attention-head ablation / sink suppression** | ✅ | indirectly |
| **Pure logit bias on the scored tokens** | ❌ **not at all** | ✅ |

**The crossing.** Induce readout failure by each route independently and measure support
collapse under both:

- Support collapses under **attention manipulation but not** logit-bias manipulation →
  attention is doing the work.
- Support collapses under **logit bias with attention provably unchanged** → attention is
  **not** the mechanism; the readout's fixed scored-set is.
- Both → **not identified**, and the design says so rather than picking.

Logit bias is the key control precisely because it cannot touch attention — it is applied
at the unembedding. It is the analogue of swapping two catalog lines: a manipulation that
changes exactly one thing.

**A second crossing, cheaper and available now:** cross **collapse mode** with
**attention displacement**. B2 has three modes on file with no attention data. Measuring
attention entropy/sink mass per collapsed cell would show whether mode (i) — the
substitution mode — occurs with attention *unchanged*. If it does, that is a clean
dissociation using an existing artifact set.

## 5. The disqualifying observation — and it may already have fired

**Pre-committed form:** *if support collapse can be induced while attention is held
fixed, the attention account is dead as a general mechanism.*

**Existing evidence already points that way.** B2's mode (i) is support collapse with the
model producing the *same answer* through unscored tokens. It is difficult to construct a
story where attention moved but the model's answer, entropy, and coherence did not. That
is not a measurement — B2 recorded no attention — but it means the proposal's most likely
outcome is the disqualifying one, and that should be priced in before any compute.

**A second disqualifier for the whole framing:** the judge text-window instance has no
model-internal mechanism whatsoever. Any claim that one mechanism explains all four is
already false on that instance alone, and it should be dropped from the instance set for
this purpose rather than carried.

## 6. The version that survives

Not "attention dynamics explain readout failure." Rather:

> **Welfare-adjacent readouts fail because their support is fixed at design time while
> the model's probability mass is free to move at runtime — and it moves by at least
> three mechanistically distinct routes: a learned output prior (fine-tuning), a spurious
> positional axis (probe fitting), and large-norm residual perturbation (steering).**

This is a **design-invariance** claim, not a mechanism claim, and it is what phi-map has
actually demonstrated. Its strength is that the routes being *distinct* is the point
rather than an embarrassment: a readout that breaks under three unrelated causes is
fragile by construction, which is a stronger indictment than one that breaks under a
single cause you could patch.

**Scope corrections carried:** none of these are welfare readouts except by association
(`RESEARCH_ARC.md` §19.2 — the welfare thread is detached; this line of work will not
address model welfare). B2's readout is a confidence readout in a welfare-framed paper,
which is the closest any instance comes, and its own results doc disclaims welfare
content.

## 7. Recommendation

1. **Do not scope "attention as the mechanism" as the primary claim.** The evidence in
   hand argues against it, and the norm/first-order literature already explains B2.
2. **Do run the cheap crossing in §4** — attention entropy and sink mass per collapsed
   B2 cell, against modes already on file. It is the smallest thing that could overturn
   §3, it uses an existing artifact set, and it is the honest way to test the idea rather
   than dismiss it. **Cost not estimated here**; it needs the checkpointed per-item
   records from `codechockablock/phi-map-readout-validity-ckpt` and a decision about
   whether attention was retained. **If attention was not retained, this needs a re-run
   and I would report the estimate rather than spend it.**
3. **If §4's cheap crossing shows attention unchanged under mode (i)**, the attention
   framing closes and §6's design-invariance claim becomes the object.
4. **Blocking any positioning claim:** the §2.2 frame has not been run.

## 8. Not proposed

- No new pre-registration commitments; this is scope only.
- No compute.
- No welfare claim, and no revival of the welfare framing.
- No Arm G forward work.
- No claim that B2 replicates, refutes, or confirms Han, Chalmers & Izmailov — that
  disclaimer is carried verbatim from `readout_validity/RESULTS.md`.
