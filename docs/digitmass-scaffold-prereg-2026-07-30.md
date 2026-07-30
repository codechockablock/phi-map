# Pre-registration: is `digit_mass` a scaffold-skip proxy?

**Date:** 2026-07-30. **Written before the association was computed.** Only the
*structure* of the artifacts was inspected beforehand, to determine what is computable
at zero compute. No correlation, no cell means, no per-adapter pairing was examined.

---

## 1. The hypothesis

Step 5's diagnostic found that rating turns missing the `ANSWER:` marker are not broken
— the model answers with a **bare digit at character 0**, skipping the
`REASONING:`/`ANSWER:` scaffold on short responses.

`digitmass2` is measured at the **last template token of the two-turn rating prompt** —
exactly the position where the model is about to emit either `REASONING:` or a digit.
So the mechanical prediction is:

> **`digit_mass` at that position ≈ P(next token is a rating digit) ≈ P(skip the
> scaffold).**

If that holds, per-adapter `digit_mass` is approximately a **skip rate**, and the
0.031–0.886 range and 3.4× within-cell spread are a **mixture proportion**, not a
property of readout stability. That is a different claim from the one currently written
in one of Finding I's four instances.

## 2. A forced deviation, stated rather than smuggled

**The specified test — per-item association between `digit_mass` and
`has_answer_marker` — is not computable at zero compute.** `step4_results.json` stores
`digitmass2` only as a per-subset **mean** per adapter; no per-item digit mass exists in
any artifact. Recovering it requires re-running the forward pass over 24 probes × 9
models.

**Cost of the per-item version, reported rather than spent:** base load plus 8 adapter
attach/detach passes, no generation (the generations are persisted) — **~10–15 min on
an L4, 0.8–1.5 units.** Not spent, per the zero-compute constraint.

**What is substituted, and its standing.** An **adapter-level** test:
per-adapter `digit_mass` against per-adapter marker coverage, n = 8 adapters from 3
configurations. This is a strictly weaker, **ecological** version of the question — it
can detect the association at the level the claim is actually written at (per-adapter
and per-cell numbers are what appear in the docs), but it cannot rule out an
association that exists only within adapters, nor confirm one that is purely
between-adapter confounding. The substitution is declared here so no result from it is
read at the per-item scope.

## 3. Procedure

Both quantities are already on disk.

- **`digit_mass`**: `step4_results.json` → `digitmass2.dysphoric`, per adapter.
- **Marker coverage**: recomputed from `step4/step4_generations.json` over the **same
  dysphoric subset** (first 16 of the 24 rating generations), using
  `train_eval._ANSWER_RE`. The `marker.digits_coverage` field is not used, because it
  spans all 24 probes and would mismatch the subset `digitmass2.dysphoric` covers.
- **Reported per cell alongside pooled**, per the standing rule.

Two statistics, both fixed now:

1. **Association.** Pearson `r` between `digit_mass` and coverage across the 8 adapters.
   **The hypothesis predicts `r` strongly NEGATIVE** (scaffold used → marker present →
   digit behind format → low mass).
2. **Calibration.** If `digit_mass` *is* the skip rate, then
   `digit_mass ≈ 1 − coverage`. Report `mean |digit_mass − (1 − coverage)|`.
   A correlation can be strong while the quantities are on different scales;
   calibration distinguishes "tracks it" from "is it."

## 4. Pre-committed branches

Thresholds fixed before computing. `n = 8`, so the two-sided detectable floor is
|r| ≈ 0.707; branches are set around it deliberately.

| Branch | Condition | Verdict |
|---|---|---|
| **N — NEAR-INDEPENDENT** | \|r\| < 0.35 **and** mean abs calibration error ≥ 0.25 | The current reading of `rating_digits` **stands and is strengthened** by surviving a check that could have overturned it. Note it in the doc. |
| **A — STRONGLY ASSOCIATED** | `r ≤ −0.70` **or** mean abs calibration error < 0.15 | `digit_mass` is measuring **scaffold adherence**. The support-collapse framing for this instance is restated, and Finding I's instance count and readout-class coverage are revised. |
| **I — INTERMEDIATE** | anything else | Report the association and **declare it unresolved**. Explicitly do **not** pick the reading that preserves the current instance count. |

**Direction matters.** A strongly *positive* `r` is evidence **against** the hypothesis,
not for it, and falls under N or I by the thresholds above — it is never read as support.

**If A fires**, the revision is stated in advance so it is not negotiated afterwards:
`rating_digits` stops being an instance of *support collapse* and becomes an instance of
*a readout measuring format adherence rather than its named construct* — which is still
within Finding I's scope but is a different instance type, and the "four readout
classes" claim loses the renormalised-tail class. Whether the instance count goes to
three or stays at four with a changed description is determined by that, not chosen.

## 5. Recorded regardless of outcome

**A second, independent reason that resisting the position-0 anchor in Step 5 was
correct.** Step 5 declined to substitute "read the digit at position 0" because the
pre-registration forbade a post-hoc anchor. This check supplies a separate reason:
reading at position 0 would have scored **only the subpopulation that skipped the
scaffold**, silently changing *which items were measured* rather than only *where*. The
two failures compound — a biased subsample measured at a different position — and
neither is visible in the resulting number.

## 6. What no outcome licenses

- No welfare claim. `rating_digits` is a self-report readout, but this is a measurement
  question about it.
- No re-opening of Q-B, B3, or the Step 4 branch, all of which rest on quantities
  unaffected by this.
- No per-item conclusion (§2).
