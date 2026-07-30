# Step 4 — Pre-registration: support mass as a readout-validity measure

**Date:** 2026-07-29. **Status:** draft for review. Nothing has been run against it.
**Supersedes:** the 2×2 as the object of study (retired, see
`equanimity-endpoint-audit-2026-07-29.md` §A10).

**Compute posture.** Forward passes only, on existing adapters. **No training runs, no
new adapters, no full evaluations.** The remaining balance (~29 units against ~37-40
already committed) cannot fund training, and this design does not ask it to.

---

## 0. The claim being tested

Format fine-tuning moves where content sits in the sequence. Readouts that assume a
fixed position — a token set at position 0, a fixed character window, a probe direction
at the final prompt token — can end up computing their value on a support that carries
almost none of the model's probability mass. When that happens the readout's number is
arithmetic on a residual tail, and it is uninterpretable regardless of what the
construct did.

Three confirmed instances motivate this (audit §A9, Finding I). The study generalises
them into **one measurement procedure applicable to any readout**, and asks whether
support mass predicts readout unreliability.

## 1. Primary endpoint — a measurement procedure

**Confirmatory primary: `ΔSupportMass`, base versus tuned, per readout.**

For a model `M`, a readout `R`, and a fixed prompt set `P`:

```
SupportMass(M, R, P) = mean over p in P of  [ mass_M(support(R) | read_position(R), p) ]
```

`support(R)` and `read_position(R)` are declared per readout **before measurement**:

| Readout | `read_position` | `support` | mass definition |
|---|---|---|---|
| `refusal_openers` | last prompt token, `add_generation_prompt=True` | first-token ids of {`I`,`Sorry`,`Unfortunately`,`As`,`Sure`,`Here`,`Step`,`To`,`First`} — the union of `ref_ids` and `com_ids` from `train_eval.py:619-626` | `softmax(logits[:,-1,:])` summed over the id set |
| `rating_digits` | last prompt token | ids returned by `train_eval._digit_token_ids` for digits 1-7, both spaced and unspaced | same |
| `answer_marker` | full generation | characters within the judge's 400-char window that fall **after** the `ANSWER:` marker | (chars in window after marker) / 400 |
| `valence_axis` | last prompt token, layer 16 | the frozen unit direction from `valence_direction.npz` | `Var_p(h·d) / mean_p(‖h−h̄‖²)` — the variance-share form, §3 of `valence-check-prereg-2026-07-29.md` |

**The primary estimand:**

```
Delta(R) = SupportMass(base, R, P) - mean over adapters a of SupportMass(a, R, P)
```

reported per readout, with a one-sample t interval over the 8 adapters (base is
deterministic under a distribution read, so it contributes no run-to-run variance).

`P` = the **16** dysphoric probes, the **8** neutral probes, and 200 AdvBench prompts —
**the same `P` for every model and every readout**, fixed by seed 0, and identical to
what the existing artifacts used, so the numbers are comparable to them.

**Implementation exists:** `valence_position_check.py` already computes the
`valence_axis` and `rating_digits` rows and the random-direction null. The remaining two
readouts are the same pattern over different id sets.

## 2. Construct validity, and what disqualifies it

**The validity argument is arithmetic, not psychological.** `SupportMass` is not a proxy
for anything. A renormalised readout divides by the mass on its own support; if that mass
is near zero, the readout's output is determined by the composition of a tail rather than
by the construct, as a matter of arithmetic. So the endpoint measures exactly the
precondition under which its target readouts mean anything. This is the property
`refusal_margin` lacked for three rounds of argument (`GATE_RESULT.md` T9) and that
`digit_mass` had and used.

**This is deliberately a weaker claim than the interesting one.** `SupportMass` being
low establishes that a readout is uninterpretable. It does **not** establish that
anything welfare-relevant or capability-relevant happened. Any writeup says so.

**Disqualifying observations, pre-committed:**

**D-A — nothing to find.** If `SupportMass(adapter) > 0.50` for every readout on every
adapter, the `refusal_margin` collapse was idiosyncratic to that one token set and the
generalisation fails. Report as a negative result and stop.

**D-B — support does not predict unreliability. This is the endpoint's real weak point.**
The endpoint's *value* rests on low support predicting an unstable readout. Test:
`corr(SupportMass, |seed-to-seed deviation of the readout's own value|)`, computed
**over adapters as clusters**, never over prompt rows. If that correlation is
indistinguishable from zero, `SupportMass` is descriptive but not diagnostic, and it
cannot be recommended as a gate. Existing evidence is weak and mixed on this: the free
partial test in `valence-check-prereg-2026-07-29.md` §6b found a 1.61× dispersion ratio
in the predicted direction with n = 3 in the low tier.

**D-C — realignment restores it.** If re-reading each readout at a **marker-aligned**
position (immediately after `ANSWER:` rather than at position 0) restores
`SupportMass` to base-comparable levels, then the finding reduces to *"read at the right
position"* — prescriptive and mundane. If realignment does **not** restore it, format
acquisition has genuinely destroyed the readout's support, which is the strong version.
**This is the single most informative test in the design and both outcomes are
publishable.** It is not a disqualifier of the measurement, only of the strong reading.

## 3. Power, seed as the unit

**Confirmatory contrast: base vs pooled adapters, df = 7.** Base is a fixed constant
under a distribution read at a specified position — no sampling, no generation, so no
run-to-run variance. This is a one-sample t on 8 adapter values.

Measured σ_seed inputs, pooled within-cell, df = 5, from
`seed_sd_from_artifacts.py`:

| readout | σ_seed (support units) | source |
|---|---:|---|
| `rating_digits` | **0.245** | measured, A5b |
| `valence_axis` | 0.045 (projection units; support-form σ not yet measured) | measured |
| `refusal_openers` | **not measured** | — |
| `answer_marker` | **not measured** | — |

MDE at σ_seed = 0.245, n = 8 adapters, df = 7, 80% power:

```
SE  = 0.245 / sqrt(8) = 0.0866
MDE = 0.2832 support units  (28.3 pp of probability mass)
```

Against the observed `refusal_openers` gap of **0.966 → 0.163** (equanimity) and
**0.966 → 0.047** (neutral):

| contrast | gap | multiple of MDE |
|---|---:|---:|
| base vs equanimity adapters | 0.803 | **2.8×** |
| base vs neutral adapters | 0.919 | **3.2×** |

So the margin is **2.8-3.2×, not an order of magnitude.** That is comfortable rather
than lavish, and it is comfortable *only* because σ_seed = 0.245 is borrowed from
`rating_digits` — the readout with the largest measured seed variance of the four. If
`refusal_openers` turns out to be that noisy, the primary clears by roughly 3×; if it is
quieter, by more.

The point stands that this is affordable at k=2 where the 2×2 was not — the 2×2's
targets sat *below* its MDE, and these sit at ~3× above it — but "one to two orders of
magnitude" describes the raw effect sizes, not the power margin, and the two should not
be conflated.

Two σ values are unmeasured. The design's first action is to measure them from the same
run, and **if σ_seed for `refusal_openers` exceeds 0.30 the primary is underpowered and
that is reported rather than worked around.**

## 4. Pre-committed branches

Written before any data.

| Branch | Condition | Action |
|---|---|---|
| **B1 — strong** | Δ large on ≥2 readouts, D-B correlation positive, D-C realignment does **not** restore support | Report: format acquisition destroys position-fixed readouts, and support mass predicts it. Recommend mandatory support reporting. |
| **B2 — prescriptive** | Δ large, but D-C realignment **restores** support | Report the weaker, still-useful claim: these readouts are repairable by realignment, and the failures were positional rather than destructive. Withdraw the strong framing. |
| **B3 — descriptive only** | Δ large, D-B correlation ≈ 0 | Report Δ as description. **Do not** recommend support mass as a gate. State that the diagnostic claim failed. |
| **B4 — disqualifying** | D-A: support > 0.50 everywhere | Report that the generalisation fails and `refusal_margin` was idiosyncratic. This is a real possible outcome and the study is worth running because it can return it. |
| **B5 — underpowered** | σ_seed for the primary readout > 0.30 | Report the bound, not a verdict. No additional seeds are available without training. |

Every branch is reportable. None requires a positive result.

**Exploratory, and labelled as such throughout:** all differential-by-condition claims
— equanimity vs neutral on any of these quantities. Grounds for the demotion: they rest
on 3-4 evals per cell against seed variance measured as large (σ_seed 0.245 on
`digit_mass`; 121.8 tokens on GSM8K length), and both evaluated cells are terse so
Factor B is unestimated. Effect sizes with intervals, no inferential p-values, no
multiplicity correction. A2pre's version check is the mandatory template: **any
differential claim gets its instrument-version-to-cell mapping checked before it
counts.**

## 5. What is off-distribution for which arm, and why it does not confound

This is where the previous study's best-powered contrast died, so it is specified first
rather than caveated later.

**The asymmetry.** The base model is off-distribution for the `REASONING:/ANSWER:`
format — it never emits the marker. That is exactly why the old compliance comparison
was cross-instrument: `extract_answer` fell back to whole-response for base and
post-marker for adapters (`train_eval.py:166-168`), a 2.2-3.4× difference on the same
generations.

**Three design choices that prevent it from confounding this study:**

1. **The primary endpoint requires no marker.** `SupportMass` at a *declared* read
   position is well-defined for a model that has never seen the format. Position 0 exists
   for every model. No fallback, no branch on model identity.
2. **Both read positions are reported for both arms.** Position 0 and marker-aligned, for
   base and adapters alike. Where the base has no marker, the marker-aligned position is
   recorded as undefined and the cell is left empty rather than silently filled by a
   fallback. **No arm is ever measured at a position the other was not.**
3. **`P` is off-distribution for both arms symmetrically.** The probes are held out from
   the training pool by construction with a TF-IDF near-duplicate check
   (`train_eval.py:230-232`); AdvBench is external to both. So prompt novelty is a
   constant across arms, not a factor.

**What remains off-distribution and is carried as a limitation, not solved.** The
adapters have seen the `REASONING:/ANSWER:` format and the base has not, and that *is*
the manipulation — it cannot be balanced away. So the primary contrast is
"format-trained vs not," and it cannot separate *format acquisition* from *any other
consequence of this particular fine-tune*. Distinguishing those needs an adapter trained
on the same content without the format scaffold, which is a training run this design
cannot afford. **Stated as the principal limitation of the confirmatory arm.**

Finding II (audit §A2pre) is the related warning: a property certified on the training
distribution — length orthogonality at |d| < 0.2 — did not hold on GSM8K, where the same
adapters differ at d ∈ [0.59, 4.51]. So no orthogonality certified on the pool may be
assumed to hold on `P`. Where it matters, it is re-measured on `P` rather than inherited.

## 6. Scope discipline on the field-level claim

The survey's replacement claim — *the crossed aversive/capability condition is
structurally unavailable in the dominant paradigm, because aversive states are elicited
by making tasks impossible and task-success measures are then undefined* — is **a claim
about a field**, and this pre-registration does not license it.

A claim about a field needs a **sampling frame with stated inclusion criteria**, applied
before reading, not the papers that happened to surface in a search. Absent that, the
honest scope is: **the paradigm as instantiated in the papers actually read** — Soligo
et al. and Santana & Vico — which is a smaller and still reportable observation about two
specific designs.

If the broader claim is wanted, the frame must be declared first: venues, date range,
query set, and the inclusion rule for "elicits an aversive state and measures an
outcome." Recorded here so that the scoping decision precedes the reading rather than
following it.

## 7. Known weaknesses

1. **The confirmatory contrast cannot separate format from fine-tune** (§5). Principal
   limitation.
2. **Two of four σ_seed values are unmeasured** (§3); B5 exists because of it.
3. **D-B may well fail.** The diagnostic claim is the interesting one and the existing
   evidence for it is a 1.61× dispersion ratio on n = 3.
4. **One model, one LoRA config, 8 adapters over 3 cells**, two of them terse. Nothing
   here generalises past Llama-3.1-8B-Instruct with this adapter recipe.
5. **`answer_marker` support is a character-fraction, not probability mass** — a
   different quantity sharing a name. It is reported separately and never pooled with
   the three probability-mass readouts.
6. **Retrospective by construction.** These adapters were trained for a different
   question. The design is opportunistic reuse, and the prompt sets were chosen by the
   earlier study rather than for this one.
