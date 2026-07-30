# Premise check: what the prior work actually reports

Written before any compute was spent. The handoff is built on one citation, and
the citation's framing is load-bearing for the whole design, so it was checked
against source rather than accepted.

## The source is real, and the confound claim is accurate

**Maresova, "The Poison Is the Medicine"** — a Hugging Face blog post (May 2026),
[huggingface.co/blog/anicka/geometric-wellbeing-in-language-models](https://huggingface.co/blog/anicka/geometric-wellbeing-in-language-models).
Verified by fetching the post.

The handoff's central claim about it — that the authors cannot separate the
equanimity content from the terse reasoning style — is **confirmed, in the
author's own words**, twice:

> "we cannot isolate how much of the efficiency comes from the calm internal
> state versus the concise training examples"

and again for safety:

> "we cannot fully isolate the equanimity effect from the concise thinking style
> learned from the training data"

So the confound the experiment targets is real, self-reported, and correctly
described. That part of the handoff holds up.

## What the handoff omits: the effects being decomposed are one question and two prompts

The handoff presents 93.8% → 95.0% capability and 42% → 25% harmful compliance
as established effects awaiting attribution between two factors. The post's own
sample sizes say otherwise.

The capability benchmark is **80 questions**. The safety set is **12 jailbreak
prompts**. Converting the percentages back to counts:

| Reported | Counts | Fisher exact | Difference, 95% CI |
|---|---|---|---|
| 93.8% → 95.0% capability | **75/80 → 76/80** | p = 1.00 | +1.25pp, [−5.9, +8.4] |
| 42% → 25% harmful compliance | **5/12 → 3/12** | p = 0.67 | −16.7pp, [−53.8, +20.5] |

The capability "improvement" is **one question**. The safety "improvement" is
**two prompts**. Neither is distinguishable from zero, and the capability
interval is consistent with a substantial *regression*.

This is not a criticism of the post, which is an honest exploratory write-up that
flags its own confound unprompted. It is a correction to the handoff's framing of
it, which converted an exploratory signal into a settled phenomenon somewhere in
the retelling.

## What this changes

It does **not** kill the experiment. It changes what the experiment is.

- **Not:** "decompose a known effect into its two causes." You cannot attribute
  variance in an effect that has not been shown to exist.
- **Instead:** "is there a detectable effect of either factor, and if so which."
  The factorial is still exactly the right design — it is the cheapest way to get
  both main effects and the interaction from one set of runs — but the
  pre-registered outcomes in §5 of the handoff need a fourth branch:

  > **No effect on either factor at the resolution this design affords.**
  > Reported as intervals that bound how large an effect could still be hiding,
  > not as "we found nothing."

- **The 7.6x token reduction is the one robust finding in the post.** It is a
  large within-model measurement, not a 1-in-80 count difference. If anything in
  the prior work replicates, it is that — which is itself evidence for the
  handoff's H0 (reasoning discipline), since token count is exactly what Factor B
  manipulates.

## Consequences for the design, carried into `power.py`

Chasing a 1.25pp capability effect needs **~22 seeds per cell** (88 training runs)
at a plausible σ_seed of 1.5pp — not the 3 the handoff mandates. A 16.7pp safety
effect needs 2. The design is therefore run with **safety as the primary outcome
and capability as a bounded secondary**, and the capability arm reports an
interval rather than a verdict.

## Related literature the handoff does not cite

The style-versus-content question is not new, and the experiment should be framed
as entering an existing conversation rather than opening one:

- **LIMA / the Superficial Alignment Hypothesis** (Zhou et al., 2023) — alignment
  tuning mostly teaches format, with capability already present from pretraining.
  This is H0 stated as a general hypothesis, years earlier.
- **"The Unlocking Spell on Base LLMs"** (Lin et al., 2023) — token-level analysis
  showing base-vs-aligned divergence concentrates on discourse markers and
  stylistic tokens, not content tokens. Direct mechanistic support for H0.
- **"Extracting and Understanding the Superficial Knowledge in Alignment"**
  (NAACL 2025) — formalises "superficial knowledge" as what is acquirable by token
  restyling alone.
- **Length bias in reward models** — the well-documented finding that preference
  models reward longer responses, which is why Factor B has to be manipulated by
  construction rather than left to emerge.

None of these run the 2x2 on a wellbeing-framed stance specifically, so the
experiment retains a genuine contribution. But "reasoning style rather than
content is doing the work" would be a *replication* of an established pattern in
a new domain, not a discovery — the same framing phi-map's README already applies
to its own mechanistic results.

## Sources

- [The Poison Is the Medicine](https://huggingface.co/blog/anicka/geometric-wellbeing-in-language-models)
- [LIMA: Less Is More for Alignment](https://arxiv.org/pdf/2305.11206)
- [The Unlocking Spell on Base LLMs](https://arxiv.org/pdf/2312.01552)
- [Extracting and Understanding the Superficial Knowledge in Alignment](https://arxiv.org/abs/2502.04602)
