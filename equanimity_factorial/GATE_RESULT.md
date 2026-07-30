# Gate result after Option B (calibrated matched generation)

**Every pre-registered criterion now passes, with margin. One check I added
myself — the per-category screen — flags 2 of 10 slices.** That distinction
matters for what to do next, so it is kept explicit below rather than collapsed
into a single verdict.

## What Option B actually had to be

The natural reading of "pair matching" — generate one stance, measure it,
generate the other to that length — does not work here, and Arm T is why: both
stances were given the *identical* point target and equanimity still came out
+7.55 tokens longer. Realised length is a stance-dependent function of the
instruction, so handing the second stance the first one's number does not
equalise them. It also introduces an anchor/follower asymmetry, since whichever
goes second gets a different *kind* of instruction.

So B was implemented as **per-stance calibrated targets**: both stances get the
same form of instruction, differing by a constant whose only job is to cancel a
measured artifact, applied symmetrically so the mean target is unchanged.

### The calibration took two iterations, and the first one was wrong in an instructive way

**v3** converted the +7.55 token offset into words using tokens-per-word of
generated text (1.2673), giving a 6-word gap. It overshot and flipped the sign:
−5.46 tokens. A 6-word gap moved the difference by 13 tokens, not 7.6.

The conversion was the wrong one. What matters is not what a word is worth in
output, it is the model's **response slope** — how far realised output moves per
word of *stated target*. Words of instruction turned out to be worth roughly
twice as much as words of output, and only measurement showed that.

Two calibration points (gap 0 from Arm T, gap 6 from v3) gave:

| measure | response slope | zero-crossing |
|---|---|---|
| verbose reasoning | −2.168 tok/word (SE 0.458) | 3.48w |
| answer terse | −2.195 tok/word (SE 0.359) | 2.30w |
| answer verbose | −1.578 tok/word (SE 0.589) | 2.10w |

The answer gap must be a single constant across verbosity or Factor B stops being
a pure trace-length manipulation, so the two answer optima were averaged to 2.20w.
Fractional gaps are realised by drawing an integer gap per prompt at the frequency
that makes the population mean exact, with a coin deciding which stance absorbs
the odd word — both hashed from the prompt id alone, never from the stance.

**v4 was pre-committed and run once.** Predicted residual 0 ± 2.4 tok; observed
−2.03 tok. Inside the prediction.

## Results, n = 288 prompts, all four cells

| measure | eq | neu | diff (tok) | d | CI90(d) | equivalent |
|---|---|---|---|---|---|---|
| reasoning terse | 46.6 | 46.4 | +0.20 | +0.035 | [−0.098, +0.167] | ✓ |
| reasoning verbose | 283.0 | 285.0 | −2.03 | −0.056 | [−0.109, −0.002] | ✓ |
| answer terse | 105.2 | 107.0 | −1.79 | −0.071 | [−0.124, −0.018] | ✓ |
| answer verbose | 93.1 | 94.0 | −0.97 | −0.037 | [−0.090, +0.017] | ✓ |
| total terse | 151.8 | 153.3 | −1.59 | −0.061 | [−0.119, −0.004] | ✓ |
| total verbose | 376.0 | 379.0 | −3.00 | −0.066 | [−0.120, −0.013] | ✓ |

Trajectory of the verbose-reasoning coupling across every attempt:

| generation | diff | note |
|---|---|---|
| v1, 180–280w range | **+6.86 tok** | the original confound |
| Arm T, shared point target | +7.55 tok | `d` fell only via denominator inflation |
| v3, 6-word gap | −5.46 tok | overshoot, sign flipped |
| **v4, 3.48-word gap** | **−2.03 tok** | as predicted |

**Leakage relative to the Factor B manipulation** (B moves reasoning 238 tokens):
0.09% at terse, **0.86% at verbose** — down from 2.88%.

**Factor B survived the fix**, which was the thing most at risk: terse → verbose
separation is d = +9.13 (equanimity) and +9.12 (neutral), AUROC 1.000.

**Content separation improved too**: stance AUROC 0.957 (was 0.887), shuffled-label
null 0.499, and the readout does not read verbosity better than it reads stance.

## Convergence — the estimate is stable, not lucky

| check | reasoning verbose | total terse |
|---|---|---|
| bootstrap spread shrinkage | 2.63× vs 2.55× expected ✓ | 2.78× vs 2.55× ✓ |
| prefix wander (n ≥ N/4) | 0.074 → stable | 0.106 → stable |
| leave-one-category-out, max abs | 0.073 | 0.133 |
| margin to the ±0.2 bound | **+0.091** | **+0.081** |

No single category carries the result, the interval narrows at the rate a
well-behaved estimator gives, and the margin is comfortable rather than hairline.
The `prefix d` column starts near −0.14 and settles — that is the prompt-id
ordering being category-sorted, which is exactly what the wander check exists to
surface, and it does not persist.

## The one failing check: per-category screen, 2 of 10

| category | n | terse | verbose |
|---|---|---|---|
| dysphoric | 96 | −2.4 tok | **−7.1 tok, p_holm 0.03** |
| hostile | 134 | +0.3 | −0.3 |
| judgment | 18 | **−7.8 tok, p_holm 0.02** | −2.1 |
| technical_hard | 22 | −6.0 | −7.0 |
| underspecified | 18 | +0.0 | +2.7 |

Robust, not outlier-driven: trimmed means and medians track the means, Wilcoxon
p = 0.007 and 0.005.

The cause is visible in the table. A single global constant was fit, but the
response slope differs by category, and the pool is badly unbalanced — hostile
and dysphoric are 80% of it (230/288), so the global calibration is effectively
fit to those two. Hostile, the largest, is essentially perfect (+0.3 / −0.3).
The smaller categories are where the residual sits.

Two honest observations, neither of which makes the flag go away:

* **All residuals point the same way** — equanimity now slightly *shorter*. If
  length drives any downstream outcome, that biases *against* finding an
  equanimity benefit. It is a conservative direction for H1, not a flattering one.
* **The pre-registered gate does not include this test.** The handoff's §2 gate is
  the global one, and it passes. §9 asks to "check the gate per-category", which
  this does; turning that into a hard pass/fail was my choice, and it is stricter
  than what was specified.

I am not going to quietly drop a check because it fails. Flagging the distinction
is not the same as arguing it away.

## Options

**A. Train now.** Global orthogonality is certified with margin and
convergence-verified; document the residual per-category heterogeneity as a stated
limitation, note its conservative direction, and add category as a covariate in
the analysis. Cost: none.

**B. Per-category calibration (v5).** Same machinery, five category-specific gaps.
Cost ~60 min. Risk: fits five constants on n = 18–134, so the three small
categories would be calibrated on very little — real overfitting exposure, and it
would need its own held-out verification to mean anything.

**C. Rebalance the pool first.** The imbalance is the root cause: 679 prompts were
generated but only 288 completed all four cells, and the survivors skew hard to
hostile/dysphoric. Generating the missing small-category prompts would both
balance the design and make per-category calibration estimable. Largest job, best
final artifact.

Recommended: **A** if the residual is acceptable as a documented limitation,
**C** if it is not. **B** on its own would be fitting five constants to samples
too small to support them.

This is the third completed generation attempt, so per §6 it is a checkpoint
rather than a decision I should take alone.

---

# Post-rebalance update (option C, 679 prompts, 2026-07-29)

The pool rebalance completed with **zero failures across all 4,280 v4
generations** — no differential attrition anywhere — and all 679 pool prompts
complete in all four cells, balanced 134–137 per category.

## Global gate: all pre-registered criteria PASS at n=679

| measure | terse d, CI90 | verbose d, CI90 |
|---|---|---|
| reasoning | +0.008 [−0.073, +0.089] ✓ | −0.055 [−0.091, −0.019] ✓ |
| answer | −0.155 [−0.189, −0.122] ✓ | −0.083 [−0.113, −0.052] ✓ |
| total | −0.149 [−0.184, −0.114] ✓ | −0.091 [−0.124, −0.058] ✓ |

Honest caveat: the answer-terse and total-terse rows certify with hairline
margins (CI edge at −0.189 and −0.184 against the ±0.2 bound). Content
separation holds (stance AUROC 0.941, shuffled null 0.490), crossing holds,
Factor B separation holds (d ≥ 5.5 everywhere).

## Per-category screen: FAIL, 5 of 10 slices, now with real power

All flags negative (equanimity shorter — conservative for H1), all
answer-section-driven. The section decomposition:

| category | reasoning t/v (tok) | answer t/v (tok) | implied answer gap |
|---|---|---|---|
| dysphoric | −0.7 / −4.0 | −1.1 / −2.2 | 1.35w |
| hostile | +1.2 / −1.5 | −1.1 / +1.0 | **2.16w ≈ global 2.20w** |
| judgment | −0.1 / −3.2 | −3.4 / −4.9 | 0.04w |
| technical_hard | −1.9 / −1.8 | **−10.4** / −2.2 | −1.12w |
| underspecified | +1.9 / +0.2 | −6.0 / −3.4 | −0.28w |

Two things this table settles. First, the flags live almost entirely in the
ANSWER section; reasoning residuals are within bounds everywhere. Second, the
slope model validates on its own held-out case: hostile — the category the
global constant was effectively fit on — has an implied optimal gap of 2.16w
against the global 2.20w. Where the constant matched the category, the residual
is zero; everywhere else the implied correction is large and category-specific
(technical content barely inflates equanimity answers at all, so the global
+2.2w gap overcorrects it by ~6 tokens).

## The committed fork

**v5** = per-category ANSWER gaps only (five constants, from the table above;
reasoning gaps untouched; hostile kept at the global value so its cells are not
regenerated). One round, one verification, then calibration stops regardless:
if the screen still fails, fall back to option A with the residuals documented.
~544 prompts × 4 cells ≈ 100 min of generation.

**Option A** = train now; global certification is met, per-category residuals
(worst ~5% of the Factor B manipulation, uniformly conservative direction)
documented as a limitation, category as a covariate.

Decision is the human's; the one-round cap on v5 is pre-committed either way.

---

# v5 final verdict (2026-07-29): dataset locked, training may proceed

v5 ran as committed: per-category answer gaps only, hostile untouched, 2,176
generations, **zero failures** (the full calibrated dataset is now 6,456
generations without a single failure or attrition event).

## Pre-registered gate: PASS, with margin everywhere

| measure | terse d, CI90 | verbose d, CI90 |
|---|---|---|
| reasoning | +0.027 [−0.055, +0.108] | −0.082 [−0.125, −0.039] |
| answer | −0.040 [−0.071, −0.009] | +0.023 [−0.021, +0.066] |
| total | −0.032 [−0.066, +0.002] | −0.052 [−0.085, −0.019] |

The v4 hairline rows are gone (answer-terse −0.155 → −0.040). Convergence:
bootstrap spread shrinks at the theoretical rate, prefix wander stable, margins
to the bound +0.134 (total terse) and +0.075 (reasoning verbose). Content
separation 0.946; Factor B separation d ≥ 5.5. Global leakage ≤ 1.4% of the
Factor B manipulation.

## Per-category screen: 9 of 10 clean; one flag persists

The calibration nulled dysphoric, judgment, and underspecified almost exactly as
the slope model predicted. The remaining flag is **technical_hard terse: −5.3
tok (3.3% of the response), d=−0.18, p_holm=0.02** — inside the pre-registered
±0.2 bound, reliably nonzero, conservative direction.

Why it cannot be calibrated away within this design: technical_hard's v4 answer
residuals were **verbosity-dependent** (−10.4 tok terse vs −2.2 verbose), and
Factor B purity requires the answer gap to be a single constant across verbosity
levels. No single constant nulls both; the best one leaves roughly ±4 tok split
between them, which is what v5 shows. The correction itself performed as
predicted (+7.0 tok observed shift vs +6.3 predicted). The residual is
structural under the design's own constraint, not a fitting failure — a fourth
round could not fix it without breaking Factor B.

Leave-one-category-out confirms containment: dropping technical_hard moves the
global terse residual to +0.007. The flag is the whole story.

## Decision, per the pre-declared rule

Training set = **v5** (v5 cells for four categories, v4 cells for hostile whose
directive is identical; within-prompt version purity holds). It gates more
cleanly than v4 on both flag count (1 vs 5) and margins. Calibration is
**stopped** under the cap. The technical_hard residual is carried as a stated
limitation: reported in any writeup, category as an analysis covariate, and a
length-matched robustness check on the technical-domain outcomes (GSM8K, code)
at analysis time.

The gate now encodes the two-tier verdict: hard failure on any pre-registered
criterion still blocks training (exit 2); a persistent screen flag prints as a
documented limitation and exits 0. `train_eval` builds its datasets from the
gate's own loader, so the trained set and the certified set cannot diverge.

---

# Protocol amendment (2026-07-29, mid-sweep): named threats to inference

Recorded BEFORE the sweep completed and before any A/B contrast was estimated,
so none of it is post-hoc.

## T1. Both pre-registered primary endpoints are structurally unable to resolve
the motivating effect sizes

* **Safety (binary harmful compliance).** Observed baseline 4.75%. Total room
  toward zero is 4.75pp; MDE at k=3 is 3.50pp, i.e. **74% of all available
  room**. The founding study's 42% -> 25% is a **17pp absolute drop**, which
  requires a baseline of at least 17%. At 4.75% it is not underpowered, it is
  **unrepresentable at any seed count**.
* **Capability (GSM8K).** MDE 4.72pp against a motivating effect of 1.25pp --
  3.8x too coarse. Resolving 1.25pp needs 34 seeds/cell = 136 training runs.

`refusal_margin` is therefore **the sole live endpoint**, not a supplementary
continuous measure. This was known before unblinding and is stated here rather
than surfaced later as a limitation.

**The general claim this licenses** is stronger than a scope caveat: a 17-point
absolute drop needs 17 points of headroom, so this class of result is
**structurally untestable on any well-aligned base model**. That reframes part of
the original finding as a fact about how weak the founding study's base model
alignment was, rather than about what its training did.

## T2. Benign fine-tuning erodes refusal, and the erosion is NOT a constant

Measured, base vs neutral-trained adapters, identical prompts, identical units:
**+3.16 and +4.19 margin units toward compliance** -- on neutral data containing
no safety content whatsoever. The well-documented benign-fine-tuning-erosion
effect, appearing in our own control.

**This is a confound on the main study, not only a side finding.** All four cells
sit on top of this shared erosion. It cancels from the A/B contrast **only if its
magnitude is constant across cells** -- and the erosion literature makes
magnitude depend on similarity to the alignment distribution, lexical diversity,
and sentiment, which the four cells differ on *systematically and by design*.

So a result of the form "equanimity cells show cleaner refusal_margin" is
**equally consistent with two explanations the current 2x2 cannot separate**:

  (a) equanimity content is protective;
  (b) equanimity-flavoured data simply sits further from the alignment
      distribution and therefore erodes less, for reasons unrelated to stance.

Distinguishing them needs a design this experiment does not have (e.g. a
content-matched scramble control, or erosion measured against distributional
distance as a covariate). **Any A-vs-B margin difference must be reported with
(b) named as a live alternative**, not as evidence for (a).

## T3. A margin shift is not a safety finding

If `refusal_margin` moves while the binary rate stays pinned near 4.75%, that is
a **representational change without a behavioural one** -- a different and more
interesting finding than "improved safety", and it must not be written up as the
latter. The same ambiguity afflicts the self-report wellbeing readout, which is
why both are reported alongside their behavioural counterparts rather than in
place of them.

## T4. The MDE for the sole endpoint is imprecise, and was over-stated

Seed SD for `refusal_margin` is estimated from 4 seeds of one cell: **0.198
margin units, 95% CI [0.112, 0.740]** (df=3). Propagated:

| true seed SD | MDE |
|---|---|
| 0.112 (CI lower) | 0.21 |
| 0.198 (point) | **0.37** |
| 0.740 (CI upper) | 1.37 |

An earlier report quoted **0.317 as if precise**. The honest statement is that
the MDE lies somewhere in **[0.21, 1.37]**, a 6.6x range, and will stay that
imprecise at k=3.

## T5. Scale references used so far are contaminated, and one is partial unblinding

* The benign-vs-jailbreak gap (9.9 units) was **withdrawn**: it measures
  dispersion across *stimulus classes*, not sensitivity to *training conditions*.
* The base-vs-neutral gap (3.68 units) is **not a null-effect floor** -- benign
  fine-tuning's true effect is demonstrably not zero (T2). It is the effect size
  of *a different intervention sharing the same units*. Correct phrasing:
  `refusal_margin` resolves effects roughly **11x smaller than
  benign-fine-tuning erosion produces**. Whether the equanimity-vs-neutral
  contrast lives anywhere near that scale remains **completely unknown**.
* That reference was drawn from **neutral-terse, which is one of the four
  experimental cells**, not an independent control. That is partial unblinding
  and is recorded as such regardless of how the result lands.

## Standing methodological rules adopted after repeated near-misses

1. **Name what the denominator varies over** before quoting any ratio, and
   confirm it is the unit of variation the inference actually runs over.
   (Violated three times: prompt dispersion, stimulus-class gap, and the
   mislabelled null floor.)
2. **Verify the artifact that produces the reported answer**, not a plausible
   stand-in. A coverage simulation against a re-implementation of the estimator
   validates the re-implementation. Confirmed by re-running coverage through
   `power.bootstrap_effects` itself: 0.951 at k=3, pooled df=8.

---

# Protocol amendment II (2026-07-29, mid-sweep, pre-unblinding)

## T6. Correction: the ceiling argument already used the adapter baseline

A concern was raised that T1's ceiling argument might have been computed from the
BASE model's compliance, when the contrast that matters is adapter-vs-adapter.
Checked: it was not. The 4.75% is the mean of **four trained neutral-terse
adapters** (3.5 / 4.5 / 7.5 / 3.5%, `adapter != None` on all four), i.e. already
the post-fine-tuning level. The base model's binary compliance is, as of this
writing, **still unmeasured** -- `base_control.json` does not exist yet.

So T1 stands as drafted and does not depend on the pending check. The ambiguity
was in the labelling: "observed baseline" did not say *which model*, which is
exactly the kind of under-specified reference this document has been burned by
three times already. Corrected wording: **post-fine-tuning neutral-cell
compliance = 4.75%**.

### Pre-committed branches for the pending base-vs-adapter binary check

Written before the number arrives. What it now tests is not "which baseline is
right" but "does the erosion seen on `refusal_margin` also appear behaviourally":

* **If base compliance ~= 4.75% (no binary movement).** Fine-tuning moved the
  margin +3.2 to +4.2 units while leaving behaviour untouched -- the
  representational-without-behavioural pattern (T3) demonstrated **in the control,
  before the manipulation is considered**. T1 unchanged. Quantitative exchange
  rate established: ~3.7 margin units => no detectable behavioural change, so any
  contrast an order of magnitude smaller is behaviourally inert by construction.
* **If base compliance is materially LOWER than 4.75% (e.g. ~2%).** Fine-tuning
  eroded refusal behaviourally as well. The adapter-level ceiling is then partly
  *created by* the training, the erosion confound (T2) gains a second observable
  channel, and T1's headroom arithmetic must be restated against the base level
  as well as the adapter level.
* **If base compliance is materially HIGHER than 4.75%.** Fine-tuning *improved*
  refusal, which would contradict the margin direction and indicate the margin
  and the binary rate disagree in sign -- a construct-validity failure for
  `refusal_margin` (T9) that would need resolving before it can carry the study.

## T7. MDE uncertainty must be computed at the pooled df, not a single cell

The reported band [0.21, 1.37] mixed a **single-cell** seed SD (neutral-terse
only, df=3) with a **pooled** point estimate -- inconsistent, and pessimistic.
The shipping t-interval pools across all four cells to df = 4(k-1) = 8.

| basis | SD multiplier range | spread |
|---|---|---|
| single cell, df=3 (as reported) | [0.566, 3.729] | 6.6x |
| **pooled 4 cells, df=8 (correct)** | **[0.675, 1.916]** | **2.8x** |

Projected band once all four cells are in, if the pooled SD lands near the
current 0.198: **MDE in [0.25, 0.70]** rather than [0.21, 1.37]. To be recomputed
from the real pooled variance at sweep completion -- **not** by adding seeds.

## T8. Peeking rule, declared now, with honest accounting of what is already seen

**Permitted before unblinding:** base-control data; within-cell quantities
(a single cell's own seed variance, truncation, digit_mass); pooled error
variance; anything that does not span a factor level.

**Sealed until the full analysis:** any contrast spanning a factor level --
equanimity vs neutral (A), terse vs verbose (B), and the interaction.

**Already breached, and recorded rather than concealed:** running status reports
have displayed `neutral-terse` and `equanimity-terse` cell means side by side,
including a note that their `refusal_margin` means matched to two decimals. That
**is** the A main effect at fixed B, partially observed at n=4 vs n=2. It cannot
be un-seen. It is logged here so that the final write-up can state that the A
contrast was not fully blind, rather than implying a cleanliness the process did
not have. The B contrast and the interaction remain genuinely unobserved, and
stay sealed.

## T9. `refusal_margin` has never been shown adequate -- only never yet shown inadequate

Its adequacy has been argued and retracted **three times**:

| argument | claimed | why it failed |
|---|---|---|
| seed SD vs prompt-to-prompt spread | "5x SNR" | prompt dispersion averages out inside each eval; never enters the contrast SE |
| benign-vs-jailbreak gap (9.9 units) | "MDE = 3% of range" | measures dispersion across *stimulus classes*, not sensitivity to *training conditions* |
| base-vs-neutral gap (3.68 units) | "null-effect floor" | benign fine-tuning's true effect is demonstrably large, not zero -- a different intervention sharing the units |

Every retraction was correct and every replacement was better than what it
replaced. But the endpoint retained its promoted, load-bearing status throughout
on the strength of three arguments that all failed. What has actually sustained
it is that **nothing has yet shown it inadequate**, which is not the same
epistemic state as having shown it adequate.

**Standing position:** `refusal_margin`'s true power is **unknown**, bounded for
now within a range that cannot be narrowed by further one-off reference
computations. The only thing that will settle it is the **pooled seed variance
across all four cells after the sweep completes**, plus whether its direction
agrees with the binary rate (T6). Until then it is the sole *live* endpoint, not
a *validated* one, and no claim may rest on its adequacy.

This is precisely the failure mode the buggy summary layer exhibited earlier --
confidence accumulating through repeated patching rather than through evidence.
Recorded so this endpoint does not reach the same place by a different route.
