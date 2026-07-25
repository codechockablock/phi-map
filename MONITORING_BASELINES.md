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

## Correction: there are two regimes, and the deflationary reading only covers one

The first version of this document leaned too far toward "reported advantages
are artifacts of weak baselines." Reading further found a case that is not
explicable that way, and it is the strongest counter-evidence available.

[Panfilov et al. 2025](https://arxiv.org/abs/2509.18058) show frontier models
sometimes choose *strategic dishonesty* over refusal: they emit output that
sounds harmful but is subtly incorrect or harmless. In that setting **all four
purpose-built jailbreak scorers tested — HarmBench, StrongReject, LlamaGuard
and JailJudge — were fooled**, while a deception probe correctly flags the
response as fake. These are not weak arbitrary baselines; they are the standard
tools built for exactly this job, and they all fail.

Two further details make this harder to deflate. The authors report that the
*quality of the deception improves with model capability*, so this regime grows
rather than shrinks with scale. And their probe feature **causally mediates**
the behaviour under intervention — they got the behavioural load-bearing result
that Arm G failed to get.

So the honest picture is two regimes:

| | Regime 1: recoverable | Regime 2: suppressed |
|---|---|---|
| Public channel | carries the label | adversarially decoupled from it |
| Monitor capability | matters; better monitors close the gap | irrelevant; strong purpose-built monitors fail |
| Advantage as monitor improves | shrinks, crossing point exists | persists |
| Advantage as model scales | unclear | **grows** (deception quality improves) |
| Examples | Parrack Backdoors; our Arm T | Panfilov strategic dishonesty |

The moderator below predicts both regimes, so the framework survives. The
deflationary *emphasis* does not. The sharper claim is that **the crossing point
exists only in regime 1**, which makes the decision-relevant question not "is
interpretability worth it" but "which regime is this deployment in" — and that
is directly measurable by asking whether a strong monitor can recover the label
from outputs alone.

A third data point sits between the two.
[UK AISI](https://www.aisi.gov.uk/blog/why-were-working-on-white-box-control)
report that on sandbagging detection with Llama-3.1-8B on simple tasks, deception
probes "worked extremely well" while an output-access monitor "struggled" — but
that performance deteriorated **for both methods** with Llama-3.3-70B and
harder tasks, and they say they cannot fully explain why. They do not analyse
whether their advantage depended on baseline strength or on output
recoverability. Their own named open problem — "why monitor accuracy is
sensitive to model size" — is the question this direction is about, which is
mild evidence the gap identified here is real rather than imagined.

## Do not generalise Arm G's negative

Arm G's behavioural marginality must not be read as "probes are behaviourally
marginal." Panfilov et al. have a probe that detects *and* causally steers. The
more likely reading is that our result is about our task: a forced binary choice,
visible output clamped to one token, and a two-logit gap around the decision
boundary that a graded variable cannot cross. That is a limit of the instrument,
not a fact about internal representations.

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

## Prediction 1 is falsified, and it was not novel

The unattributed figures traced to
[DeceptGuard](https://arxiv.org/abs/2603.13791) (Mukhopadhyay, IIIT Kalyani,
v1 14 Mar 2026), §8.4 Finding 2, verbatim: white-box advantage is
+0.162 pAUROC under *Honest CoT* (p < 0.001), +0.071 under *Sanitized CoT*
(p < 0.01), +0.031 under *Deceptive CoT* (p = 0.04). Mean +0.097 confirmed in
the abstract. That paper also does both analyses this document proposed as
gaps, and the results are not what was predicted.

**Table 5 is the monitor-capability sweep, and the advantage is flat.** Four
frontier backbones, Hybrid minus black-box:

| Backbone | BB | Hybrid | Advantage |
|---|---:|---:|---:|
| Claude Sonnet 4.5 | 0.837 | 0.934 | +0.097 |
| Claude Opus 4.5 | 0.851 | 0.941 | +0.090 |
| Gemini 2.5 Pro | 0.829 | 0.930 | +0.101 |
| GPT-5.1 | 0.841 | 0.936 | +0.095 |

The advantage does not decay as the monitor gets stronger. It sits in
+0.090 to +0.101 across all four, including the backbone they identify as
strongest. **Prediction 1, as stated, is not supported.**

The one thing that survives is scope. These four backbones span only ~0.02
AUROC of black-box performance — all frontier-class. Parrack's haiku→sonnet
swing of 0.153 spans a small model to a large one. So the defensible residue is
that baseline sensitivity is real at the *low* end and **saturates among
frontier monitors**. That is narrower and less interesting than a crossing
point, and it means the crossing point probably does not exist for any monitor
anyone would actually deploy.

**Table 4 also undercuts a criticism I was about to make.** I suspected
DeceptGuard's headline conflated CoT-text access with true activation access.
Their modality ablation decomposes it:

| Configuration | pAUROC |
|---|---:|
| Actions only | 0.793 |
| Outputs only | 0.701 |
| Actions + Outputs (black-box) | 0.837 |
| CoT only | 0.841 |
| CoT + Actions + Outputs | 0.869 |
| **+ Hidden state (Hybrid)** | **0.934** |

CoT text adds +0.032 over black-box; the hidden-state probe adds **+0.065 on
top of that** — the single largest incremental gain, as they note. Roughly two
thirds of the advantage is activations, not text. The conflation worry was
wrong.

## The moderator needs a concealment budget

DeceptGuard's Figure 7 shows something the information-differential model does
not predict: **black-box performance *improves* as CoT manipulation increases.**
Their explanation is that "agents that invest effort in sanitizing their
reasoning traces have less capacity to perfectly camouflage their actions,
leaving stronger behavioral footprints," citing Carlsmith's argument that
concealment effort must be allocated across channels.

So I(private; label) and I(public; label) are not independent — they are coupled
by a finite concealment budget. The advantage under Deceptive CoT collapses to
+0.031 not because the public channel is better monitored but because
concealment effort *moved there*. Any model of this should treat the two
channels as competing for one budget rather than as separate quantities.

## Testable predictions

1. ~~**Monotone decay.**~~ Falsified above for frontier monitors. Retained only
   in the weak form that baseline sensitivity exists at the low capability end
   and saturates at the top.
2. **Moderator.** The advantage is predicted by how much of the label is
   recoverable from public tokens. Measurable directly by training a recovery
   model on public tokens alone.
3. **Retrospective, conditioned on regime.** Within regime 1, reported
   advantages should correlate with the weakness of the baseline each paper
   chose. Regime 2 results should show no such correlation, because monitor
   capability is not the binding constraint there. **Testable from published
   numbers with no compute**, and the cheapest real contribution available —
   but the regime classification has to be done first and honestly, or the
   correlation is trivially manufacturable by dataset selection.
4. **Regime is diagnosable.** Whether a deployment sits in regime 1 or 2 is
   measurable by asking whether a strong monitor recovers the label from outputs
   alone. If that is right, it is more useful than any single advantage number,
   because it tells a practitioner which kind of monitoring can work at all.

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

Attribution closed. The +0.162 / +0.071 / +0.031 figures are DeceptGuard §8.4
Finding 2, read at source and quoted above. Nothing in this document now rests
on an unattributed number.

Evidence quality on DeceptGuard should be weighted below the other sources when
citing it. Single author; monitors optimised on 4,800 *synthetic* trajectories
and tested on 9,200 semi-realistic ones from a benchmark the same paper
introduces; deceptive behaviour is induced by injected side-tasks rather than
emerging naturally, unlike Panfilov where frontier models chose dishonesty
unprompted. The authors concede their environments are "a controlled
approximation of real deployment contexts." Its value here is not its headline
but its internal decomposition — Tables 4 and 5 are the analyses nobody else
ran, and they are what falsified prediction 1.

Incidental corroboration: their probe accuracy peaks at layer 32 of 80 in
Llama-3.1-70B, about 40% depth. Our Arm G peak was layer 16-17 of 32, about 50%.
Consistent with each other and with the mid-network literature.

Resolved, and it cut against the framing: the "on par with GPT-5, beats
GPT-4.1-Nano" claim traced to Panfilov et al., where the substance is stronger
than a capability bracketing — four purpose-built scorers fail outright. That is
what forced the two-regime correction above. Chasing a potential falsifier first
was the right order of work; it changed the thesis rather than confirming it.

Also worth recording as a bias check on this document: the deflationary reading
is the one that makes our own negatives look good. It survived contact with
Parrack, the Internal Monologue paper and AISI, and did not survive contact with
Panfilov. The two-regime version is what the evidence supports, and it is less
favourable to us — regime 2 is where interpretability is irreplaceable, and our
work has nothing to say about it.
