# The white-box advantage is a gap to a baseline, not a property of a probe

Research direction adopted 2026-07-24, superseding the original Φ framing.
Status: positioning established from literature; no measurement run yet.

## The evolved goal

The programme began asking whether the internal→external map destroys shared
structure (Arm T), whether a model can exploit a serialization of its own
geometry (Arm S), and whether goal–constraint conflict is represented
pre-action (Arm G). Those are three instances of one question, which is the
goal as now stated:

> **When does reading a model's internals tell you something its behaviour does
> not, and does that difference matter?**

This is the load-bearing question for interpretability-as-safety: if behavioural
monitoring gets you the same answer more cheaply, internals are a research tool
rather than a monitoring strategy.

## The claim

Every reported "white-box advantage" is a difference of two numbers:

```
Advantage = AUROC(probe monitor) − AUROC(behavioural monitor)
```

The second term is a free parameter. It decomposes into three things, only two
of which are about the model:

1. how much the privileged channel says about the label,
2. how much the public channel says about the label,
3. **how good the chosen monitor is at extracting (2).**

The literature typically fixes (3) at an arbitrary point and reports the
difference as though it measured (1) minus (2). That makes the headline number
partly a statement about which baseline the authors picked.

This is not a hypothetical concern:

- [Parrack, Attubato & Heimersheim 2026](https://arxiv.org/abs/2507.12691) do
  the comparison properly — matched episodes, labels and splits — and find the
  boost is **+0.062 AUROC** (best white-box 0.941 vs best black-box 0.879),
  self-described as "weak but encouraging". Their black-box baseline moves
  **0.726 → 0.879** on nothing but the monitor model (haiku-3.5 vs sonnet-3.5).
  That 0.153 swing from baseline choice alone is larger than most reported
  white-box advantages, including their own.
- [Monitoring the Internal Monologue 2026](https://arxiv.org/html/2605.18549)
  compares probes (94.4–97.1) against **one** Qwen3-8B judge (83.98) and
  concludes internal representations "carry a substantially richer behavioral
  signal than the generated text alone", framing the gap as representational
  richness and explicitly not as evaluator capability. An 8B judge sits at the
  weak end of the range Parrack et al. measured.

So two 2026 papers on the same question report advantages differing by roughly
a factor of two, and the difference is plausibly explained by baseline choice
rather than by anything about the models or probes.

## The moderator

The advantage should track the *information differential* between channels, not
merely the weakness of the public one:

```
Advantage ∝ I(privileged channel ; label) − I(public channel ; label)
```

Both terms move, which resolves results that otherwise look contradictory:

| Setting | Private channel | Public channel | Predicted | Observed |
|---|---|---|---|---|
| Our Arm G (clamped `READY`) | rich | **zero bits by construction** | maximal | internal 0.85 vs behavioural 0.50 |
| Our Arm T (full response text) | rich | rich | small or negative | internal 0.594 vs behavioural 0.661 |
| Parrack, Backdoors | rich | label publicly determinable | negative | white-box strictly worse |
| Honest-CoT agent settings | rich, faithful | actions only | large | reported large gains |
| Deceptive-CoT agent settings | **poisoned** | actions only | small | reported small gains |

Our two arms are the endpoints of that table, which is why they disagreed with
each other. Arm G engineered the public channel to carry nothing, so its
"internal wins" result is the ceiling of the effect and says little about
deployment. Arm T let the public channel carry the whole response, and
behaviour won.

## Testable predictions

1. **Monotone decay.** Hold the probe fixed, sweep monitor capability, and the
   advantage shrinks monotonically. The crossing point — where a cheap
   behavioural monitor matches the probe — is the decision-relevant quantity
   and is not reported anywhere we have found.
2. **Moderator.** The advantage is predicted by how much of the label is
   recoverable from public tokens. Measurable directly by training a recovery
   model on public tokens alone.
3. **Retrospective.** Reported advantages across published papers should
   correlate with the weakness of the baseline each paper chose. **This is
   testable from published numbers with no compute**, and it is the cheapest
   real contribution available.

## Why this is the right direction for this repo

It converts three negatives into evidence rather than leaving them as failures:

- Arm T's loss to behaviour is a data point at the unfavourable end of a
  spectrum whose best benchmark averages +0.062 and goes negative out of
  distribution. Not an outlier.
- Arm G's win is the maximal case of the same phenomenon and is now
  interpretable as such.
- Arm S's clone-equivalence result is the same lesson one level down: a
  privileged channel that is reconstructible carries no advantage at all.

It also vindicates the original P3 instrument. Fixed features 0.47 → untuned
LLM judge 0.764 → internal 0.91+ is a three-point monitor-capability curve, and
the WRITEUP's phrasing — "black-box monitoring transfers exactly insofar as the
monitor has a mind" — is independently supported by someone else's data.

## Honest status

This is a synthesis and a reframing. It identifies a confound and a gap; it has
not measured anything. Prediction 3 is a literature meta-analysis and could be
done immediately. Predictions 1 and 2 need a study.

Unverified: pAUROC figures of +0.162 / +0.071 / +0.031 across
Honest/Sanitized/Deceptive CoT conditions appeared in a search summary that
could not be attributed to a specific paper. They are consistent with the
moderator above but must be read at source before being cited. A related
unattributed claim — that dishonesty probes perform on par with GPT-5 and beat
GPT-4.1-Nano — would be a direct bracketing of a probe between two monitor
capabilities and should be chased, since it is close to prediction 1 already
being partly answered.
