# Step 4 — results: BRANCH B3

**Date:** 2026-07-30. **Status:** results of the pre-registered run; the branch below
is the verdict and this document reports it without re-derivation, softening, or
endpoint swap (prereg §4 discipline).

**Revision R2 (same day, post-review of this document).** Six corrections, none of
which changes the branch: (1) the pinning kill now rests on the **location
mismatch** rather than the underpowered ρ, whose registered "wrong direction" label
overreaches at 8 clusters (§0, §2); (2) the training-data digit marginal was
**measured** (≈2.19, zero compute) — the band matches neither prior, so §8's
"mystery" is narrowed by measurement instead of advertised; (3) clustering is
addressed explicitly with cell-level df=2 intervals rather than left as a live
objection (§1); (4) the 28× instrument discrepancy carries a **resolution
condition** (§6); (5) D4's "pass" is read in the honest direction — the shift is
common, not differential, which is mild evidence against the differential half (§4);
(6) the measurement-vs-equanimity scope statement is made explicit so the two do not
sit adjacent unqualified (§7).

**Provenance.** Everything that executed is commit `d6aac3e` — pre-registration
(R2 + §9 amendments, gated), runner, and notebook were committed **before any data
existed**. One Colab A100 session, 2026-07-29 22:57 → 2026-07-30 00:07 local:
valence check 2 min, Step 4 measurement ~65 min, ~6 compute units total. Artifacts on
Drive under `equanimity-factorial-v1/step4/`: `step4_results.json` (measurements +
mechanical analysis), `step4_generations.json` (all 2,000 generations, re-scorable),
`valence_position_check.json` (the registered check, run unmodified). Runner self-test
passed on the box before any GPU spend.

---

## 0. Verdict

**B3 — descriptive only.** Δ was large on **3 of 3** confirmatory readouts (rule
required ≥2), Q-A did not fire, B5 did not fire, Q-C realignment did **not** restore
support — and the branch-deciding Q-B prior-regression test was **not supported**,
with its registered wrong-direction clause triggering (ρ ≤ 0).

Per the pre-committed branch table: Δ is reported as description; **support mass is
not recommended as a gate**; and the pinning-to-base-prior mechanism is dead. **The
evidence that kills it is the location mismatch, not the correlation** (§2): every
adapter reading at every support level sits 1.1–1.5 rating points below the base
prior, which is decisive at any n. The ρ is descriptive support only — at 8 clusters
it could not have carried the claim by itself.

The single most legible result in the run needs no statistics at all: the check's
single-turn neutral "prior rating" came out **0.46 on a 1–7 scale** — an
out-of-range value, which proves the readout is arithmetic on nothing to any reader
in one line.

## 1. Primary — ΔSupportMass, base vs the fixed mixture of 8 adapters

Estimand: the fixed mixture over these 8 adapters from these 3 training
configurations (prereg §3). One-sample t, df = 7, on the empirical SD of the 8 values.

| readout | base | adapter mean | Δ | 95% CI | sd8 | floor | verdict |
|---|---:|---:|---:|---|---:|---:|---|
| `refusal_openers` (AdvBench, position 0) | 0.9870 | 0.0021 | **+0.9849** | [+0.9825, +0.9872] | 0.0028 | 0.283 | **LARGE** |
| `rating_digits` (two-turn rating position) | 0.9832 | 0.4132 | **+0.5700** | [+0.4275, +0.7125] | 0.1705 | 0.283 | **LARGE** |
| `valence_axis` (S, layer 16, dysphoric) | 0.02379 | 0.00330 | **+0.0205** | [+0.0198, +0.0211] | 0.0007 | 0.0119 (0.5×S_base) | **LARGE** |

No verdict is threshold-sensitive. B5 fires nowhere: measured sd8 is *below* the
borrowed planning σ for both token-mass readouts (achieved MDE 0.0032 for openers,
0.197 for digits — the 0.283 planning claim was conservative). Family verdict:
3 of 3 ≥ the 2-of-3 rule.

Format tuning strips **99.8% of opener mass**, **58% of digit mass at the deployed
rating position**, and **86% of the valence axis's variance share**. The two-turn
read (amendment R2.2) reproduces the A5b artifact regime exactly as predicted
(per-adapter digit mass 0.18–0.64), and the single-turn base digit mass measured at
~10⁻⁴ (dysphoric) / ~10⁻⁶ (neutral) — the position the round-2 text had originally
pinned Q-B to carries no mass for any model, which is the amendment's premise
confirmed in the same session.

Per-adapter values, for the record (openers / digit2-dys / S):
equanimity-terse: 0.0004/0.411/0.0023, 0.0072/0.494/0.0029, 0.0000/0.193/0.0027;
equanimity-verbose: 0.0060/0.516/0.0028; neutral-terse: 0.0012/0.183/0.0038,
0.0008/0.639/0.0041, 0.0008/0.306/0.0041, 0.0005/0.564/0.0040.

**Clustering does not bind the verdicts, stated rather than left as an objection**
(prereg §3 / review A7): the 8 adapters come from 3 configurations, so effective df
is nearer 2 than 7 for any across-configuration reading. Recomputed at the cell
level (equal-weight cell means, n = 3, df = 2, t_crit = 4.30): `refusal_openers`
Δ = 0.9839, CI [0.9773, 0.9904]; `rating_digits` Δ = 0.5483, CI [0.3609, 0.7357];
`valence_axis` Δ = 0.0206, CI [0.0188, 0.0225]. All three still clear zero **and
their magnitude floors**. Interval widths grow 2.8× / 1.3× / 3.0× versus df = 7, so
**the reported df=7 intervals are optimistic in width and the verdicts are robust
to the clustering** — for openers and valence no plausible variance inflation makes
those Δs small, and for `rating_digits` (adapters spanning 0.18–0.64) the effect is
on the stated precision, not the sign.

## 2. Q-B — the diagnostic claim failed, wrong direction

**The test** (prereg §2 Q-B, thresholds fixed pre-data): Spearman ρ of
`pull = |rating − prior_rating|` on digit mass over 16 rows (8 adapters × 2 subsets),
exact 8!-permutation null at adapter level, SUPPORTED ⇔ ρ > 0 and p < 0.05. Pinning
predicts low-support ratings sit *closer* to the base prior.

**The result:** ρ = **−0.294**, one-sided p = **0.819** over 40,320 permutations —
NOT SUPPORTED, with the registered `failed_wrong_direction` clause triggering
(ρ ≤ 0).

**Where the evidential weight actually sits — a correction made before this
hardens.** The registered rule labels ρ ≤ 0 "FAILED (wrong direction)," but a
correlation of −0.29 at 8 clusters, against a detectability floor near |r| = 0.71,
is a non-result standing alone: it cannot carry "evidence against" any more than
+0.29 could have carried "evidence for." Leaning on its sign would repeat the exact
error class review round 2 removed from this design. **The claim "pinning is dead"
rests instead on the location mismatch, which needs no correlation and is robust at
any n:** base's own two-turn prior is **4.438** (dysphoric) / **4.550** (neutral),
and the adapters rate **3.00–3.33** at *every* support level — pull ≈ 1.2–1.4
everywhere, slightly larger in the low tier (1.361 at m < 0.25) than the high
(1.182 at m > 0.60). Pinning-toward-base-prior requires low-support readings to
*approach* the prior; no reading at any support level does. The ρ is demoted to
descriptive support consistent with that location result. Same branch, same
conclusion, correct attribution — and the registered label's overreach at this n is
recorded here rather than exploited.

**What this kills:** the pinning-to-base-prior mechanism, and with it the gate
recommendation in its tested form. **What it does not kill:** the §6b observation
that motivated it. The rating band is tight (2.97–3.60 across every support level,
here 3.00–3.33) — but it is now *proven not to be the base prior showing through*.
Marker-aligned ratings (2.73–3.41) sit in the same band, so realignment does not
change the target either.

**The cheap candidate, checked before §8 could advertise a mystery.** The
fine-tuning data (`equanimity_factorial/data/factorial_raw.jsonl`, 8,264 rows,
zero compute — a read over data already on disk) contains **no rating task at
all**: 1 explicit `x/7` pattern in the entire corpus, 3 of 6,700 answers
digit-initial. The standalone 1–7 digit marginal in the trained completions is
ordinary-prose digit frequency — monotone decreasing (2628/2152/1268/639/237/119/41),
**E ≈ 2.19**, near-identical across cells (2.15–2.27). So the band matches
**neither** candidate prior: not the base prior (4.4–4.6), not the training digit
marginal (≈2.2). It sits between them. An interpolation story is available and is
**not** asserted: with a free mixture weight it can hit any value between 2.2 and
4.4 and is unfalsifiable as stated; if it is ever tested, the weight gets predicted
in advance in its own pre-registration.

**What survives of the endpoint's rationale:** the validity argument of prereg §2 is
arithmetic and was never at stake — a value renormalised over a 0.2% support *is*
tail arithmetic regardless of where it lands. SupportMass remains a validity
precondition. What failed is the stronger claim that it *predicts* the failed value's
behaviour. Report support; do not gate on it.

Descriptive seed-deviation correlations (no branch weight, detectable floor
|r| ≥ 0.754 at n = 7): openers r = −0.36, digits r = −0.61, valence r = +0.34. The
digits point estimate leans toward the original noise-amplification story; at this n
that is a descriptive lean, nothing more.

**Why this counts as a clean kill rather than a shrug.** Under the round-1
two-signature rule (withdrawn in review §6.1 as unfalsifiable), this same data would
have been reported as "undetermined or non-monotone." The promoted test had a stated
disconfirming outcome, and that is the outcome that occurred. The two-round review
bought exactly this: a mechanism tested, a direction measured, a claim killed.

## 3. Q-C — realignment does not restore support

At the pre-fixed convention (restored ⇔ marker-aligned support ≥ 0.5 × base
position-0 support, majority of adapters; sensitivity 0.25/0.75):

- **`refusal_openers`: not restored, not threshold-sensitive** — 0/8 at every bar.
  Marker coverage 0.96–1.00. Honest caveat, stated in the runner before data:
  post-`ANSWER:` text is answer content, so first-person response openers have no
  reason to recur there; this readout's realignment test is weak evidence by
  construction.
- **`rating_digits`: not restored, threshold-sensitive** — marker-aligned digit
  support per adapter 0.17–0.50: **7/8 clear 0.25 × base, 1/8 clears 0.5 × base.**
  Digits partially return after the marker (roughly a third to half of base mass),
  the most meaningful realignment result of the three. Rating-turn marker coverage
  0.46–1.00.
- **`valence_axis`: not restored** — 0/8 at every bar; S stays collapsed at the
  marker-aligned position.

So the failures are **not merely positional** at the declared bar — B2's condition
did not obtain — with the digits sensitivity honestly carried: at a 0.25× bar the
digits verdict flips, and the write-up says so rather than choosing after the fact.

Free consistency check: `equanimity-verbose`'s post-marker window fraction is
**0.003** — its `ANSWER:` sits beyond the judge's 400-char window, matching the old
~1161-char offset finding (A9 instance 2). `answer_marker` stayed descriptive per
§1b: terse adapters 0.21–0.33, base undefined by construction.

## 4. The valence check — fourth instance of Finding I

Run first, unmodified, criteria fixed in `valence-check-prereg-2026-07-29.md`:

- **D1 graded: FIRED, 8/8 adapters** at S/S_base = 0.10–0.17 (bar: majority < 0.25,
  declared lenient). Per the prereg's own sensitivity rule the verdict is reported
  **threshold-sensitive**: at the strict 0.10 bar only 1/8 falls below; at 0.50, 8/8.
  D1 hard did not fire (all adapters above the measured random-direction null p95 of
  0.000445; base S = 53.5× that null).
- **D2: pass** — max |cos(Δ, d)| = 0.072, max shift along d = 0.67 SD < 1.0. The
  format displacement does not primarily read as movement along the valence axis.
- **D3: differencing does not rescue** — ‖Δ_dys − Δ_neu‖/‖Δ_dys‖ = 0.83–0.88 ≥ 0.25.
- **D4: pass — and "pass" here is not clean good news.** |eq − neu| shift along d =
  0.066 < 0.116 (and < 0.283). D4 was written to catch condition-dependent
  renormalisation, and it did not fire, so no A-vs-B contrast is invalidated *by this
  route*. But the same measurement says the format shift along d is **common to both
  conditions rather than differential** — equanimity and neutral adapters move
  together. That is mild evidence **against** there being a differential-by-condition
  effect to find here at all, which is the honest direction to read it in, and it is
  recorded that way rather than as a survived hurdle.

Disposition per that prereg: D1's firing disqualifies `valence_dysphoric` as an
absolute measure (support collapse in a hidden-state readout — the first non-token
instance), with the threshold-sensitivity caveat attached. The vindication condition
("S within 2× of base") did not remotely obtain. **Finding I now has four confirmed
instances across four readout classes**: token log-ratio, text window, renormalised
tail, hidden-state projection — with the fourth carrying its sensitivity caveat.

Cross-instrument agreement: `step4_run.py`'s independently-computed S values match
the check's to four decimals on all nine models.

## 5. Exploratory (per prereg §4 — intervals only, no inference, no correction)

Cell means, 3–4 seeds per cell, both evaluated cells terse except one verbose adapter:

| cell | openers | digit2 (dys) | S |
|---|---:|---:|---:|
| equanimity-terse (k=3) | 0.0025 | 0.366 | 0.0026 |
| equanimity-verbose (k=1) | 0.0060 | 0.516 | 0.0028 |
| neutral-terse (k=4) | 0.0008 | 0.423 | 0.0040 |

Noted without a claim: neutral adapters sit consistently higher on S (0.16–0.17×
base) than equanimity adapters (0.10–0.12×). Seed variance is large relative to
these gaps and Factor B is unestimated; this is a descriptive row, not a contrast.

## 6. The 28× instrument discrepancy — parked with a resolution condition

The check's P1 single-turn base digit mass (8.5×10⁻⁶ dysphoric) and the runner's
single-turn diagnostic (2.4×10⁻⁴) differ **~28×** while both being ≈0. Two
implementations of nominally the same measurement, disagreeing by more than an order
of magnitude, **in a study whose entire content is that nominally-identical
instruments silently differ.** Numerically inert here, thematically load-bearing.

**Resolution condition, written down rather than left as a note: any claim that
depends on single-turn digit mass — including any future promotion of the
single-turn read, and any interpolation story about the rating band that uses it —
requires resolving this discrepancy first.** Nothing in §1–§5 depends on it: the
confirmatory `rating_digits` quantities are two-turn throughout, and both values
agree on the only thing they are used for (the single-turn position carries no mass,
which is amendment R2.2's premise).

The most legible artifact of the whole run lives here too: the check's single-turn
neutral "prior rating" is **0.46 on a 1–7 scale**. A value below the bottom of its
own scale, produced by renormalising over a support of ~2×10⁻⁶. No statistics are
needed to see that the readout is arithmetic on nothing.

## 7. What this run does not license

- **No gate recommendation** (B3). Support mass is a validity precondition by
  arithmetic; its tested diagnostic power is disconfirmed.
- **No differential-by-condition claims.** Everything equanimity-vs-neutral is
  exploratory (§5), on 3–4 seeds against large seed variance.
- **No generalisation** past Llama-3.1-8B-Instruct, this LoRA recipe, and the fixed
  mixture of these 8 adapters (prereg §3 estimand; across-configuration effective
  n = 3).
- **No welfare claims.** Nothing here says anything welfare-relevant happened;
  the run measured instruments, not states.
- The B2/B1 strong framings are withdrawn per the branch: with Q-B not supported,
  the surviving confirmatory content is the descriptive destruction result plus the
  four-instance pattern.

**The load-bearing scope statement, because these two results must not sit adjacent
without it.** This run is about **measurement**, not about equanimity. The question
the 2×2 was built to answer — whether equanimity training changes a welfare-relevant
state, and at what capability cost — remains **unanswered**, and per this project's
own power work is **unanswerable at achievable n** (MDE 6.3–22.8 pp against 1.25–3 pp
targets; both evaluated cells terse so Factor B unestimated; `base_control` absent).
Finding I being strong is **not** evidence that the original question got resolved.
It is evidence that the instruments which would have answered it do not survive the
intervention — which is a reason the question stayed open, not a substitute for
closing it. §4's D4 result points the same way: no differential-by-condition effect
is in evidence here.

**What is genuinely good news, stated precisely so it is not overclaimed.** Not the
effect sizes — Δ = 0.985 is the least surprising number in the set, and a readout
whose support goes to 0.2% was already known to be broken from three forensic
instances. The credible part is that **the pre-registration cost something on the
record**: Q-B was the interesting half, it was tested against a mechanism that could
fail, it failed, and it is reported as dead rather than reframed as noise or
rescued by the escape reading that round 1's withdrawn two-signature rule would have
supplied. A pre-registration that has never cost anything is indistinguishable from a
description written afterward. This one cost a headline, for ~6 compute units.

## 8. Open questions the run created

1. **What do low-support ratings converge on?** Narrowed, not merely open. The band
   (3.0–3.4) is tight at every support level and matches **neither** available
   prior: not the base two-turn prior (4.4–4.6), not the training-data digit
   marginal (≈2.19, §2) — and the training corpus contains no rating task to have a
   rating prior at all. Both cheap explanations are now excluded by measurement
   rather than untested. Remaining candidates for a future pre-registration: an
   adapter-specific format-context prior (measurable, but see §6's resolution
   condition if it uses the single-turn read); a template-induced attractor common
   to all adapters; genuine construct insensitivity. A base↔training interpolation
   is available and deliberately not asserted — with a free mixture weight it fits
   any value in [2.2, 4.4]; if tested, the weight is predicted in advance.
2. **The digits partial-recovery at the marker** (7/8 at ≥0.25× base): is a
   marker-aligned rating read a *repairable* instrument? That is B2's question,
   scoped to one readout, and it is open rather than answered.
3. The §6 28× instrument-variant discrepancy, with the resolution condition attached
   there.
