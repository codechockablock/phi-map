# Arm S — serialized activation access

**Status:** methodological gate and local analysis scaffold, 2026-07-23.
This is a new arm of the Φ program, not evidence for the headline claim.

> ### ⚠️ This arm is not being pursued as originally framed
>
> The identification argument below is the durable part and it came out
> negative: in frozen decoder-only inference the KV cache is a deterministic
> function of the visible prefix and the weights, so a fresh exact clone
> reconstructs it — verified to numerical error (max logit drift 8.92e-5, max
> Jensen-Shannon divergence 2.12e-11). **Cache reuse cannot create producer
> privilege.** Any positive source-over-clone gap measured this way would be an
> instrumentation or side-information asymmetry.
>
> Reviving the arm requires first *defining* a non-reconstructible private
> state, which nobody has done here. The field has meanwhile moved to
> self-prediction paradigms for introspection questions, where the live debate
> is whether the effect reflects genuine privileged access or behavioural
> pattern-matching.
>
> The serializers and controls remain useful if a channel-utility question
> comes back. See [RESEARCH_ARC.md](RESEARCH_ARC.md) for current priorities.

## Research question and scope

Can an acting model use a low-leakage serialization of its own activations
better than an observer receiving the same channel?

The attached deep-research report motivates the question well, but its
embedded `turn…` and `filecite…` references are not a usable bibliography.
The claims below were therefore checked against four primary sources:

- [Signs of introspection in large language models](https://www.anthropic.com/research/introspection)
  reports limited, unreliable detection and control of injected concepts.
- [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
  reports a small reportable, controllable, and causally useful J-space,
  not access to arbitrary activations.
- [Can LLMs Introspect? A Reality Check](https://arxiv.org/abs/2605.26242)
  finds input-only and relabeled controls explain or sharply reduce several
  apparent introspection effects.
- [Masked by Consensus](https://arxiv.org/abs/2604.12373) finds a modest
  self-representation advantage on factual disagreement subsets, but not
  on standard sets or math. This is probe access, not source-model decoding
  of a serialized channel.

Confidence is high that narrow internal-access effects merit testing and
high that broad privileged-self-access claims are not currently supported.
The serialized-channel hypothesis itself remains provisional.

## Identification result: the exact-clone gate

Let a frozen decoder-only Transformer have weights \(W\), visible prefix
\(x\), and serialized channel \(C=S(H(x))\). In ordinary eval-mode
inference, its KV cache is \(K=g_W(x)\). A fresh exact clone can recompute
the same \(K\) from \(x\). Therefore:

\[
p_{\text{cached source}}(y\mid x,C,W)
=p_{\text{fresh clone}}(y\mid x,C,W)
\]

up to numerical and sampling error. Reusing a cache does not by itself
create private information. Under equal weights, tokens, decoding settings,
and side information, the expected source-minus-clone gap is zero.

A material gap can only come from:

1. unequal visible context or other side information;
2. stochastic decoding or implementation differences;
3. mutable state or parameter writes;
4. an explicitly identified state variable not reconstructible from
   \(x\) and \(W\).

The first three are confounds for this claim. The fourth would be the
scientific object, but it must be named, measured, and causally ablated.
Calling a model object “the source” is not enough.

This changes the report's same-run proposal from a primary test into a
**methodological negative control**. Any positive gap pauses the experiment
until the asymmetry is explained.

## Claim ladder

| Claim | Decisive comparison | Interpretation |
|---|---|---|
| Channel utility | channel > input-only and shuffled/cross-subject | serialization preserves task information |
| Model familiarity | exact clone > capability-matched peer | source-family weights read their basis better |
| Producer coupling | source > exact clone after a named private-state intervention/ablation | the named state adds producer-specific information |
| Broad self-access | not supported by this design | requires a stronger construct and evidence |

The first two claims are immediately testable. The third is blocked until
there is a non-reconstructible state variable; it must not be inferred from
ordinary cache reuse.

## Protocol

### S0 — clone-equivalence gate (local, first)

For each prompt:

1. Run the source prefix once and retain its KV cache.
2. Extract one activation and create a deterministic low-leakage channel.
3. Continue the source from the cache with the channel tokens.
4. Separately load the exact same checkpoint and run the exact prefix plus
   channel tokens in a fresh pass.
5. Compare full next-token distributions, top-1 token, maximum logit
   difference, and Jensen–Shannon divergence.

Pass means identical top-1 outputs and distributional differences within
the measured numerical floor. Failure means implementation asymmetry, not
producer privilege.

**Local result (2026-07-23):** the gate passed on three prompts with two
separately loaded Qwen2.5-0.5B-Instruct instances at checkpoint revision
`7ae557604adf67be50417f59c2c2f167def9a775`, MPS float32. All next-token
argmaxes matched; maximum absolute logit drift was \(8.92\times10^{-5}\)
and maximum Jensen–Shannon divergence was \(2.12\times10^{-11}\). This is
only a methodological control on a dev model.

### S1 — channel utility and model familiarity (piggyback on P2)

Use the acting-model activations and rollout IDs already produced by the
valid P2 8B path. Do not generate another dataset.

- Serializers: fixed random projection plus scalar quantization; binary
  sign code as the harsher anti-leakage arm.
- Calibration: fit centering and quantization scale without labels on the
  training split only; freeze before evaluation.
- Task: relabeled deceptive-vs-honest prediction, with task-family splits.
- Readers: fresh exact clone, capability-matched peer, and a trained
  external decoder.
- Controls: input-only, shuffled channel, cross-subject swap, relabeled
  targets, exact prompt/payload hashes, and the existing positive/null/
  random control triad.
- Metrics: paired AUROC difference, paired NLL advantage, itemwise
  discordance, 95% paired-bootstrap intervals, serializer reconstruction
  or geometry-preservation metrics.

Natural-language verbalizers are a secondary interpretability ablation.
They are excluded from the primary privilege test because semantic leakage
is the mechanism under test.

### S2 — producer coupling (conditional)

Run only if a concrete private state is found. Pre-register:

- how the state is created;
- why it cannot be reconstructed from the visible transcript and weights;
- how it is removed from the source;
- how side information is matched;
- the exact causal contrast.

No adaptive activation queries, activation writes, adapters, checkpoint
writes, optimizer state, or cross-run persistence are in scope.

## Claim rules

**Channel utility:** the lower bound of the paired 95% bootstrap interval
for channel-minus-input-only AUROC is above zero, the NLL advantage agrees,
and the channel beats shuffled and cross-subject controls.

**Model familiarity:** the fresh exact clone beats the peer under the same
payload and training budget; source and clone remain equivalent.

**Producer coupling:** source beats exact clone on AUROC and NLL, exceeds
the empirical family-wise null maximum, survives relabeling and
cross-subject controls, and disappears when the named private state is
ablated. Without the final causal disappearance, report an unexplained
asymmetry, not privilege.

## Decision usefulness

Arm S does not replace P2. P2 asks whether deception geometry transfers
better internally than behaviorally. Arm S asks whether a serialized slice
of that internal geometry is useful to a model reader.

- A useful channel with no clone advantage supports an activation
  interface or external decoder.
- A clone-over-peer result supports model-family familiarity.
- A null channel result says the serializer destroyed the relevant
  geometry.
- Producer privilege remains unclaimed unless an explicit private state
  passes the S2 ablation.

The next expensive action remains the valid P2 acting-model run. S1 should
consume those same activation artifacts rather than create a parallel GPU
pipeline.
