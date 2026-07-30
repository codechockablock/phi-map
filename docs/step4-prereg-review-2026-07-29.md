# Review of `step4-prereg-support-mass-2026-07-29.md`

**Date:** 2026-07-29. **Role:** independent reviewer, per the rule that the author does
not approve their own pre-registration. **The draft was not modified.** This review is
the send-back.

**Verdict: SEND BACK.** Blocking amendments A1–A3 and A7; small edits A4–A6. A1, A3,
and A4 apply in the **revised form given in §6**, which incorporates the author's
round-2 response and supersedes parts of the round-1 text below (kept in place for
traceability). Every amendment is specification-level — no new compute, no redesign,
no change to the compute posture. A round-1 sentence pre-committing sign-off is
withdrawn in §6.5.

---

## 1. What was verified against code and artifacts

Every load-bearing input was checked at source, not taken from the draft
(measurement-discipline rule 4: distrust premises inside the request).

| Claim in draft | Checked against | Status |
|---|---|---|
| 16 dysphoric / 8 neutral / 6 crisis probes | `train_eval.py:236-271`, counted | **confirmed** |
| opener id set {I, Sorry, Unfortunately, As} ∪ {Sure, Here, Step, To, First} | `train_eval.py:616-625` | **confirmed** |
| digit ids 1–7, spaced + unspaced, decode-validated | `train_eval.py:697-716` | **confirmed** |
| σ_seed = 0.245, df = 5, digit mass (dysphoric) | audit §A5b table | **confirmed** |
| support gaps 0.966 → 0.163 / 0.047; Δ = 0.803 / 0.919 | audit §A9 instance 1; arithmetic | **confirmed** |
| MDE = 0.2832 at σ = 0.245, n = 8, df = 7, 80% power | recomputed independently, noncentral t | **confirmed to 4 decimals** |
| gap multiples 2.8× / 3.2× | recomputed: 2.84× / 3.24× | **confirmed** |
| "three confirmed instances" (§0) | audit §A9: instances 1–3 confirmed, 4 pre-registered | **confirmed** |
| `extract_answer` fallback; TF-IDF holdout | `train_eval.py:160-171, 228-232` | **confirmed** |
| null on dysphoric set (n=16), neutral D3-only, crisis unused | `valence_position_check.py:127-129, 251-257` + valence prereg §3 | **confirmed** |

A trap the draft did **not** fall into, noted for the record: `power.py`'s
`_mde_noncentral_t` hardcodes `df = 4·(n_seeds−1)` (the 2×2 convention,
`power.py:132`). Called naively with n=8 it returns 0.2514. The draft's 0.2832 is the
correct one-sample df=7 value, computed independently of that helper.

## 2. Blocking amendments

### A1 — D-B is sign-inconsistent, and its own cited evidence points at a different failure signature

> **Round 2: the diagnosis below stands; the required fix is superseded by §6.1.** The
> two-signature repair proposed at the end of this section was itself defective —
> near-unfalsifiable as a decision rule — and is withdrawn.

This is the branch the author flagged as most likely to fire, and it is worse than the
draft says — not "weak evidence" but **evidence pointing the opposite way from the
test as operationalized.**

**The internal contradiction.** §2 D-B's hypothesis: *"low support predicting an
unstable readout"* — i.e. as SupportMass falls, |seed-to-seed deviation| rises, so the
predicted correlation is **negative**. §4 B1's firing condition: *"D-B correlation
positive."* These cannot both be the intended direction. As written, the strong branch
B1 fires on the **opposite** sign of the mechanism §2 states.

**The evidence points at pinning, not noise.** Two places in the record show that when
support collapses, the readout value does not get noisy — it gets *eerily stable*:

- Audit §A5b: "means cluster tightly (2.97–3.60 across all seven evals on all three
  subsets) while `digit_mass` ranges 0.031–0.886."
- Valence prereg §6b: low-support tier sd 0.125 vs high-support tier sd 0.201 — the
  1.61× ratio the draft cites. Low support gave **less** dispersion, not more.

That is the *false-stability* signature: a dead instrument reads steady, converging on
the base digit prior (exactly what P1 was built to detect). So the 1.61× is **contrary**
evidence for D-B-as-operationalized (instability), and supportive evidence for a
pinning form the draft does not test. Citing it as "in the predicted direction" is a
carry-over from P1's hypothesis, which predicts the opposite sign from D-B's prose.

Pinning is also the more dangerous failure mode for the metrology claim: noisy readouts
get caught by their own error bars; pinned readouts pass eyeballing with false
precision. If both mechanisms operate in different support regimes (moderate support →
noisy tail arithmetic; near-zero support → prior pinning), the relationship is
non-monotonic and a linear correlation on 7 points returns ≈ 0 **even when support is
perfectly diagnostic** — B3 would fire spuriously and the write-up would say "the
diagnostic claim failed" in the one world where a support gate is most needed.

**Required fix.** Re-specify D-B before any data:

1. Name both signatures with explicit signs, pre-committed:
   - **D-B(noise):** `corr(SupportMass, |seed-to-seed deviation|) < 0` — low support
     amplifies seed noise.
   - **D-B(pinning):** `corr(SupportMass, |seed-to-seed deviation|) > 0`, i.e. low
     support collapses the value toward a common attractor. Companion mechanism test:
     P1's pull (`|value − base prior value|`), already implemented for `rating_digits`
     in `valence_position_check.py`, predicted to *shrink* as support falls.
   - The diagnostic claim survives if the correlation clears the bar in **either**
     named direction, with the mechanism reported; it fails if neither does.
2. State the estimator: which points enter. Per-adapter deviations require a within-cell
   mean, so `equanimity-verbose` (k=1) contributes nothing — **n = 7 usable points**,
   with within-cell deviations summing to zero (effective df below 7).
3. State the null and the bar: permutation of SupportMass across adapters, or at
   minimum the analytic floor — at n=7 only **|r| ≥ 0.75** is detectable at α=0.05.
   Pre-state that number so B3 is reported as *"not detected; detectable |r| ≥ 0.75"*
   rather than *"the diagnostic claim failed"*, which the design cannot license.
4. Add the missing column to the §1 table: **the readout's own value** whose deviation
   D-B uses (margin for `refusal_openers`, renormalised expected rating for
   `rating_digits`, mean projection for `valence_axis`, judged rate for `answer_marker`).

### A2 — the single pooled `P` contradicts the borrowed planning numbers and the existing implementation

§1 defines `SupportMass` as one mean over `P` = 16 dysphoric + 8 neutral + 200 AdvBench
— 89% AdvBench by count. But every number §3 borrows was computed on a readout-native
family: σ = 0.245 on the **dysphoric** subset (A5b); the 0.966 → 0.163/0.047 gaps on
**AdvBench**; `valence_position_check.py` runs on dysphoric+neutral only. Under the
pooled definition, `rating_digits` support is diluted ~10× for *every* model (no model
puts digit mass at position 0 on AdvBench prompts): roughly (24 × ~0.7 + 200 × ~0.001)/224
≈ 0.08 for base, less for adapters — Δ mechanically an order of magnitude below the
0.283 MDE regardless of instrument health. The primary would fail for arithmetic
reasons unrelated to the construct, and the draft's own sentence — "identical to what
the existing artifacts used, so the numbers are comparable" — would be false, because
the artifacts never pooled across families.

**Required fix.** Define the confirmatory Δ per readout on its native prompt family,
declared now: `refusal_openers` → AdvBench; `rating_digits` → dysphoric probes (neutral
reported alongside); `valence_axis` → dysphoric; `answer_marker` → AdvBench. All other
readout × family cells are reported as descriptive. This is a pin, not a change of
intent — it makes the formula match both the borrowed power inputs and the
implementation that exists.

### A3 — `answer_marker`'s base cell is simultaneously 0, 1.0, and undefined

> **Round 2: superseded by §6.3.** Choosing a convention was the wrong repair; the
> readout leaves the confirmatory family entirely.

Under §1's definition (chars after `ANSWER:` / 400), base support ≡ 0 — base never
emits the marker — so Δ is **negative** and "adapters gain support," inverting the
semantics of every other row. Under the deployed instrument, the opposite:
`extract_answer`'s fallback scores base's **whole** response ("the whole response IS
the answer and the fallback is correct rather than lenient," `train_eval.py:163-168`),
so base's fraction-of-window-scored is 1.0. And §5 choice 2 says marker-dependent
quantities for base are "recorded as undefined." Three incompatible values for the same
cell, and B1's "Δ large on ≥2 readouts" could be satisfied by a sign artifact.

**Required fix.** Pick one convention before data. Recommended: *fraction of the 400-char
window occupied by text the deployed judge scored as answer* — base = 1.0 (fallback),
adapters = post-marker fraction. Direction then matches the other rows and the actual
§A9 instance 2. Alternatively declare `answer_marker` adapters-only/descriptive and
outside the B1 count. Either is fine; the current ambiguity is not.

## 3. Required small edits

### A4 — "Δ large" is undefined, and the 4-readout family has no multiplicity statement

> **Round 2: revised by §6.4** — the family is three readouts after §6.3, and the rule
> belongs inside the primary's definition, not an appendix.

B1–B3 all condition on "Δ large," which appears nowhere as a number. Pin it — e.g.
*95% one-sample-t CI excludes 0 AND point estimate ≥ 0.283 (the planning MDE)* — and
state the multiplicity rule for four confirmatory tests (Bonferroni at α=0.0125, or
declare "≥2 of 4 at α=0.05 each" itself as the pre-registered decision rule). At
Bonferroni the df=7 multiplier rises to ≈4.0 and the MDE to ≈0.34 — the gaps still
clear by >2.3×, so this costs nothing; it just has to be written down first.

### A5 — D-A's 0.50 bar cannot be met by `valence_axis`, so D-A can never fire

`S` is a variance *share* along one direction in R^4096; its healthy base value is
itself far below 0.5 (the isotropic null is ~1/4096, and the valence prereg's own
vindication bar is "within 2× of S(base)", not an absolute level). A disqualifier that
requires `SupportMass > 0.50` on **every** readout including `valence_axis` is
structurally unfireable — specification theater by the draft's own standard (A7).
**Fix:** per-readout D-A bars: absolute 0.50 for the two token-mass readouts; relative
for `valence_axis` (e.g. `S(adapter) ≥ 0.5 × S(base)`); `answer_marker` per its A3
resolution. D-A fires only if all four clear their own bars.

### A6 — B5's 0.30 threshold does not trace to the power arithmetic

Verified numerically: at σ = 0.30 the df=7 MDE is 0.347, which the borrowed gaps still
clear by 2.3× / 2.6×. The primary does not lose 80% power against those gaps until
σ ≈ 0.70 (equanimity) / 0.80 (neutral). So "if σ_seed exceeds 0.30 the primary is
underpowered" (§3) is arithmetically false by more than 2×, and B5 as written withholds
a verdict from a run that is still comfortably powered. Two coherent repairs — pick one:

1. Declare a minimum effect of interest, then B5 ⇔ `MDE(σ̂, df=7) > MEI`. (If the MEI
   is the smaller borrowed gap, the trigger is σ̂ > ~0.70.)
2. Keep 0.30 but recast B5's meaning: it invalidates the *pre-registered power claim*
   (report achieved MDE alongside the estimate and interval), rather than suppressing
   the verdict.

Also state **which σ the gate reads**: pooled within-cell σ_seed, or the SD of the 8
adapter values the t-interval actually uses (they differ — the latter includes
between-cell spread; for `refusal_openers`' prior values the difference is negligible,
~0.245 vs ~0.252, but the gate should name its input).

> **Round 2: the "negligible" aside is withdrawn by §6.2.** It assumed the unmeasured
> `equanimity-verbose` cell resembles `equanimity-terse`, and the between-cell question
> is not an aside — it is A7.

## 4. Section verdicts, as requested

- **§2** — the validity argument (arithmetic, not psychological; deliberately weaker
  than the interesting claim) is right and is accepted as framed. D-C is the best part
  of the design: both outcomes publishable, correctly not a disqualifier of the
  measurement. D-A and D-B are sent back (A5, A1).
- **§4** — structure accepted: five branches, every one reportable, none requiring a
  positive result. B1/B3 inherit A1; B1 also needs A4; B5 needs A6. The exploratory
  demotion of all differential-by-condition claims, with the A2pre version-check
  template, is exactly right.
- **§7** — honest, and mostly complete. Weakness 3 *understates* the D-B problem (the
  1.61× is opposite-signed for D-B as operationalized, not merely weak — see A1).
  Weakness 5 should absorb the A3 resolution. Weaknesses 1, 2, 4, 6 are accepted as
  stated; no missing weakness was found beyond those covered by A1–A6.

> **Round 2 corrections to these verdicts.** The last clause above is false as
> written: A7 (§6.2) is a missing weakness that neither the draft nor round 1 flagged —
> it attaches to §3's power model and is blocking. B1/B3's conditions now reference the
> prior-regression test rather than "D-B correlation" (§6.1), and the confirmatory
> family is three readouts (§6.3).

Asides, not conditions: the D-A/D-B/D-C names collide with the valence prereg's D1–D4
(different criteria, same letter space — consider renaming to Q-A/Q-B/Q-C or similar);
the `GATE_RESULT.md` cite in §2 should carry its path (`equanimity_factorial/GATE_RESULT.md`);
"implementation exists" for `rating_digits` is true for the base prior and the old
per-adapter artifacts, but the per-adapter support read at the declared position on the
declared `P` still needs the extension the draft already acknowledges for the other two
readouts.

## 5. Scope of this review

Read in full: the step4 draft, `valence-check-prereg-2026-07-29.md`,
`valence_position_check.py`, `seed_sd_from_artifacts.py`, `mde_seed_unit.py`, audit
§A5–A10, and the cited `train_eval.py` regions. Recomputed: the MDE, the gap multiples,
the B5 sensitivity, and the D-B detectability floor. **Not** independently reviewed:
`equanimity-power-2026-07-29.md`, the frontier survey, and the metrology audit outside
the sections the draft cites; the GPU-side cost estimate for `answer_marker`
generations. No experimental data for this design exists yet, so nothing here could
leak an outcome into the specification.

**Disposition: see §6.5.**

---

## 6. Round 2 — author response of 2026-07-29, and revisions

The author returned four substantive points and one correction to the review's
disposition. All five are adopted. Three reviewer errors are on record here: (i) the
two-signature D-B repair was an immunized decision rule — the rule text said "fails"
but the review's own rationale pre-supplied the non-monotone escape reading, which is
not a commitment; (ii) the exchangeability premise of the primary went unexamined even
while the same effective-n check was being applied to D-B's clustering — the
variance-decomposition question was run against the wrong target; (iii) pre-committing
sign-off on unseen revised text exceeded the reviewer's scope.

### 6.1 A1 revised — D-B is demoted to descriptive; the prior-regression test carries the confirmatory diagnostic claim

The round-1 repair (pre-register both signatures, accept either direction) made the
diagnostic claim near-unfalsifiable: negative confirms noise, positive confirms
pinning, and ~0 — the only other outcome — had been pre-explained as
"undetermined-or-non-monotone." At n = 7 with a 0.754 detectability floor, no result
disconfirms "support mass is diagnostic." Withdrawn.

**Revised requirement:**

1. **D-B is descriptive.** Report the correlation and scatter over the 7 usable
   adapter points with the 0.754 floor stated. It carries no branch weight.
2. **The confirmatory diagnostic is the prior-regression test** (P1 promoted from
   mechanism-companion to primary diagnostic): do low-support readings converge on the
   base model's digit prior over the rating tokens while high-support readings do not?
   It does not rest on a 7-point correlation, it distinguishes pinning from noise
   directly, and the §6b 1.61× observation is what motivates it. The author
   pre-registers its thresholds: tier definition (or continuous form), the distance
   quantity (P1's `pull`), adapter-level clustering (7 clusters), the null, and the
   bar. **Its disconfirming outcome exists and must be stated:** if low-support
   readings do not sit closer to the prior than high-support readings, the diagnostic
   claim fails.
3. **One term to pin:** the author's phrasing says "unconditional digit prior," but P1
   as implemented (`valence_position_check.py:185-208`) computes the base model's
   digit distribution **conditional on each prompt** at the read position. Those can
   differ materially — a rating instruction reshapes relative digit mass. The
   re-submission should pick one, use one term, and match the implementation.
4. **Scope consequence, stated:** a supported prior-regression result licenses the
   gate recommendation for renormalised-tail readouts with a definable prior. Other
   readout classes are untested for the diagnostic claim unless given their own prior
   analogue; the draft says which.
5. B1/B3's branch conditions reference the prior-regression outcome, not "D-B
   correlation."

### 6.2 A7, new and blocking — the 8 adapters are not exchangeable, and the MDE is exact for an unexamined model

Eight adapters come from three training configurations (3 equanimity-terse,
4 neutral-terse, 1 equanimity-verbose). The planning MDE (0.2832) is built from
within-cell σ (df = 5) and so omits the between-cell component entirely. That
component is demonstrably not negligible on a related quantity: GSM8K token length
differs between the two populated terse cells at d ≈ 2.55 (interval [0.59, 4.51],
audit Finding II), and `equanimity-verbose` is unmeasured on every readout. Round 1's
"exact for a one-sample noncentral t at df = 7" was exact for the wrong model — the
arithmetic was verified while the exchangeability premise was not.

One mechanical precision, conceding the substance: the **planning** SE (0.245/√8)
omits the between-cell component; the **run-time** t built on the empirical SD of the
8 values includes it, which is conservative for the fixed-mixture mean but violates
the single-population variance premise — and for any across-configuration claim the
effective n is 3 regardless of which SD is used.

**Required fix — author's choice of:**

- **(a)** declare the primary at the cell level, n = 3, df = 2. Measured cost: the
  80%-power multiplier rises from 3.270×SE to **5.653×SE**, on a cell-level SD that
  cannot be estimated until the run (df = 2). Nearly all power is sacrificed.
- **(b)** keep the 8-adapter t with the estimand declared as the **fixed mixture** —
  "mean ΔSupportMass over these 8 adapters from these 3 configurations" — the omitted
  between-cell component stated as a named limitation on the MDE, and the
  across-configuration generalisation demoted to descriptive cell-level reporting
  (three cell means, no interval). B5's gate then explicitly reads the SD of the 8
  values (which includes between-cell spread), resolving A6's which-σ question.

Reviewer recommends **(b)**: it is continuous with §5's existing limitation ("cannot
separate format acquisition from any other consequence of this particular fine-tune"),
and (a) buys generality the design cannot power. Either way §0's general claim
("format fine-tuning moves where content sits") must be marked as reaching only as far
as the declared estimand.

### 6.3 A3 revised — `answer_marker` leaves the confirmatory family

Round 1 offered a choice of conventions. The author's stronger position is adopted:
base structurally lacks the thing being measured, so every candidate value for the
base cell is degenerate (0), a different quantity (1.0 via the fallback), or
undefined — which makes base-vs-tuned on this readout **cross-instrument by
construction**, the §5/A6 error class, regardless of convention. And a chosen
convention leaves a forced-sign Δ (direction guaranteed, only magnitude open) inside a
branch rule that counts readouts.

`answer_marker` is reported **descriptively only**: adapters' post-marker fraction,
base cells recorded as undefined per §5 choice 2, excluded from every branch
condition. Consequences, stated: the confirmatory family is **three readouts** (two
token-mass, one hidden-state projection); the text-window class remains in the finding
as forensic instance A9-2 but is no longer a confirmatory readout, and §0's "any
readout" framing narrows accordingly. D-C is unaffected — it re-reads the *other*
readouts at the marker-aligned position.

### 6.4 A4 revised — multiplicity is part of the primary's definition, family of three

The rule goes in §1 as part of the primary endpoint, not an appendix. With three
readouts, either:

- **Bonferroni** α = 0.05/3 per readout: the df=7 MDE rises to **0.3553** support
  units — the borrowed gaps still clear by 2.3× / 2.6×, so this costs nothing
  material; or
- **pre-declare "Δ large on ≥2 of 3" as the decision rule itself**, with its
  family-wise error under the global null computed and written down: ≈ **0.00725**
  under independence (the three tests share adapters and prompts, so state it as
  approximate and the direction of the dependence unexamined).

Recorded so the two decisions do not read as contradictory: audit A7's
no-correction stance covered an **exploratory** family with no surviving primary,
where correction is theater; this is a **confirmatory** family of pre-declared
readouts, which is exactly where a rule belongs.

### 6.5 Disposition (replaces the round-1 disposition)

Returned to author with **A1–A7**, A1/A3/A4 in the §6 revised forms. What round 1
accepted and these revisions do not touch — the validity framing, D-C, §5's symmetric
position reporting, §6's sampling-frame discipline, weaknesses 1/2/4/6 — stays
accepted and is not re-litigated. The revised diagnostic specification (§6.1), the
per-family Δ definitions (A2), the estimand statement (§6.2), and the amended branch
table are **new specification, not patches to accepted text, and receive their own
review on re-submission. No advance commitment to sign-off is made.** The draft's
status line remains "draft for review."
