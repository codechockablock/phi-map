# Step 4 — results: BRANCH B3

**Date:** 2026-07-30. **Status:** results of the pre-registered run; the branch below
is the verdict and this document reports it without re-derivation, softening, or
endpoint swap (prereg §4 discipline).

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
support — and the branch-deciding Q-B prior-regression test **FAILED in the wrong
direction** (ρ = −0.294, one-sided p = 0.819 against the exact 8! permutation null).

Per the pre-committed branch table: Δ is reported as description; **support mass is
not recommended as a gate**; the diagnostic claim — low support predicts convergence
of the readout value onto the base model's prior — is reported as **failed**, not as
undetected: the point estimate went the other way.

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

## 2. Q-B — the diagnostic claim failed, wrong direction

**The test** (prereg §2 Q-B, thresholds fixed pre-data): Spearman ρ of
`pull = |rating − prior_rating|` on digit mass over 16 rows (8 adapters × 2 subsets),
exact 8!-permutation null at adapter level, SUPPORTED ⇔ ρ > 0 and p < 0.05. Pinning
predicts low-support ratings sit *closer* to the base prior.

**The result:** ρ = **−0.294**, one-sided p = **0.819** over 40,320 permutations.
`failed_wrong_direction` — the pre-registered disconfirming outcome.

**What the numbers say.** Base's own two-turn prior is **4.438** (dysphoric) /
**4.550** (neutral). The adapters rate **3.00–3.33** on dysphoric probes at every
support level, so pull is ≈1.2–1.4 everywhere — and *larger* in the low-support tier
(mean pull 1.361 at m < 0.25, n = 3 rows) than the high tier (1.182 at m > 0.60,
n = 5). Low-support ratings are, if anything, slightly **farther** from the base
prior.

**What this kills:** the pinning-to-base-prior mechanism, and with it the gate
recommendation in its tested form. **What it does not kill:** the §6b observation
that motivated it. The rating band is tight (2.97–3.60 across every support level,
here 3.00–3.33) — but it is now *proven not to be the base prior showing through*.
Whatever the low-support ratings converge on, it is not that. The tightness stands
as an unexplained regularity; the tested explanation is dead. Marker-aligned ratings
(2.73–3.41) sit in the same band, so realignment does not change the target either.

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
- **D4: pass** — |eq − neu| shift along d = 0.066 < 0.116 (and < 0.283). The
  renormalisation is **not** condition-dependent; equanimity and neutral adapters
  shift equally along d.

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

## 6. Discrepancies parked, not swept

The check's P1 single-turn base digit mass (8.5×10⁻⁶ dysphoric) and the runner's
single-turn diagnostic (2.4×10⁻⁴) differ ~28× while both being ≈0. No conclusion
rests on either value — both say the single-turn position carries no mass — but the
gap between two near-identical instruments is unexplained and recorded here. The
check's single-turn neutral "prior rating" of 0.46 on a 1–7 scale is renormalisation
off the clamp floor: a live specimen of the uninterpretable-tail arithmetic this
study is about.

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
- The B2/B1 strong framings are withdrawn per the branch: with Q-B failed, the
  surviving confirmatory content is the descriptive destruction result plus the
  four-instance pattern.

## 8. Open questions the run created

1. **What do low-support ratings converge on?** The band (3.0–3.4) is tight at every
   support level and is *not* the base two-turn prior (4.4–4.6). Candidate targets
   for a future pre-registration: the adapter's own format-context digit prior; a
   template-induced prior common to all adapters; genuine insensitivity of the
   construct. Any such test gets its thresholds registered first.
2. **The digits partial-recovery at the marker** (7/8 at ≥0.25× base): is a
   marker-aligned rating read a *repairable* instrument? That is B2's question,
   scoped to one readout, and it is open rather than answered.
3. The §6 28× instrument-variant discrepancy.
