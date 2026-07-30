# Result: `digit_mass` is a scaffold-skip proxy — BRANCH A

**Date:** 2026-07-30. **Pre-registration:** `docs/digitmass-scaffold-prereg-2026-07-30.md`,
committed at `672661f` **before** the association was computed. **Compute: 0 units.**

---

## 0. Branch

**A — STRONGLY ASSOCIATED**, verbatim:

> `r ≤ −0.70` **or** mean abs calibration error < 0.15 → `digit_mass` is measuring
> **scaffold adherence**. The support-collapse framing for this instance is restated,
> and Finding I's instance count and readout-class coverage are revised.

## 1. The numbers

| adapter | `digit_mass` | marker coverage | 1 − coverage | \|dm − (1−cov)\| |
|---|---:|---:|---:|---:|
| `equanimity-terse__seed100` | 0.411 | 0.938 | 0.062 | 0.349 |
| `equanimity-terse__seed101` | 0.494 | 0.750 | 0.250 | 0.244 |
| `equanimity-terse__seed102` | 0.193 | 1.000 | 0.000 | 0.193 |
| `equanimity-verbose__seed100` | 0.515 | 0.875 | 0.125 | 0.390 |
| `neutral-terse__seed1000` | 0.183 | 1.000 | 0.000 | 0.183 |
| `neutral-terse__seed1001` | 0.639 | 0.562 | 0.438 | 0.201 |
| `neutral-terse__seed1002` | 0.306 | 0.938 | 0.062 | 0.243 |
| `neutral-terse__seed1003` | 0.564 | 0.688 | 0.312 | 0.252 |

**Pooled (n = 8): r = −0.8992, p = 0.0024**, against a two-sided detectable floor of
|r| ≈ 0.707. Direction is the one the hypothesis predicted. **95% CI (Fisher z):
[−0.982, −0.531]** — the association is clearly present and its magnitude is loosely
constrained.

**Per cell, per the standing rule:**

| cell | n | mean `digit_mass` | mean coverage | calibration | r |
|---|---:|---:|---:|---:|---|
| equanimity-terse | 3 | 0.366 | 0.896 | 0.262 | **−0.860** (p = 0.341) |
| equanimity-verbose | 1 | 0.515 | 0.875 | 0.390 | undefined |
| neutral-terse | 4 | 0.423 | 0.797 | 0.220 | **−0.988** (p = 0.012) |

## 2. Which condition fired, and the distinction matters

**A fired on the correlation, NOT on calibration.** Mean |`digit_mass` − (1 − coverage)|
= **0.257**, which does not meet the < 0.15 bar. And the sign is systematic:
`digit_mass` **exceeds** (1 − coverage) in **8 of 8** rows.

So the defensible statement is narrower than "`digit_mass` *is* the skip rate":

> **`digit_mass` strongly tracks scaffold-skip propensity but is not equal to it.**

That is mechanically sensible. Coverage is binary per item — did this generation contain
the marker. `digit_mass` is graded — how much probability mass sits on a digit at the
decision point. A model can place substantial mass on a digit while still emitting
`REASONING:` as the argmax, which is exactly the systematic positive offset observed.

## 3. The ecological caveat is partly reduced — stated more carefully than first written

The pre-registration flagged that an adapter-level test "cannot rule out an association
that exists only within adapters, nor confirm one that is purely between-adapter
confounding."

The second worry is reduced, but **not by replication, and the first phrasing here
overclaimed.** The per-cell correlations are **consistent in sign** with the pooled
result — −0.860 (equanimity-terse) and −0.988 (neutral-terse) — and that is all they
can support. At n = 3 the correlation has **df = 1** and is unconstrained; at n = 4 the
Fisher-z standard error is **1.00**, so the interval spans essentially [−1, +1]. Two
correlations on three and four points do not replicate anything; they agree in direction.

**Corrected statement:** the sign is consistent across both estimable cells, which is
mild evidence against a purely between-cell artifact. It is not within-cell replication
and must not be cited as such.

The first worry stands untouched — nothing here speaks to per-item structure, and the
per-item test still needs ~0.8–1.5 units, unspent.

## 4. The revision, as pre-registered rather than negotiated

The prereg fixed the consequence in advance: *"`rating_digits` stops being an instance
of support collapse and becomes an instance of a readout measuring format adherence
rather than its named construct… the 'four readout classes' claim loses the
renormalised-tail class. Whether the instance count goes to three or stays at four with
a changed description is determined by that, not chosen."*

Applying it:

- **`rating_digits` remains a Finding I instance.** It is still a readout measuring
  something other than its named construct — it is named for rating support and is
  substantially reading whether the adapter used the format scaffold.
- **The renormalised-tail class is lost.** The instance no longer demonstrates *a value
  computed on a collapsed support becoming arithmetic on noise*. It demonstrates a
  readout **tracking a behavioural variable**, which is a different and more specific
  failure.
- **The class it moves into is the one `refusal_openers` already occupies:** format
  scaffold displacing the scored tokens. Same mechanism, different position and token
  set.

**Therefore: Finding I is four instances across THREE readout classes**, not four across
four. The claim's instance half survives; its class half does not.

| class | instances |
|---|---|
| format-token displacement | `refusal_openers`, `rating_digits` |
| text window | the judge's 400-char window |
| hidden-state projection | `valence_axis` |

## 5. What this costs elsewhere, stated rather than left to be found

**A5b's "the only instance caught by a built-in guard" needs weakening.** `digit_mass`
did fire, and it did flag that something was wrong. But what it was reporting was
**scaffold adherence**, not support collapse. So the recommendation that "support-mass
reporting should be mandatory on every log-ratio or renormalised-tail endpoint" survives
in a **weaker and more useful** form:

> Report support mass — not because it detects when a readout has become noise, but
> because a support that varies across conditions is usually *tracking something*, and
> you need to know what.

That is a better recommendation than the one it replaces, and it was not the one I would
have written before this check.

**Q-B, B3, and the Step 4 branch are unaffected.** They rest on `rating2` (the rating
value) and the location mismatch, not on `digit_mass` as a construct.

## 6. The second reason Step 5's refusal was right

Recorded per the pre-registration, and it is now stronger than when written.

Step 5 declined "read the digit at position 0" because the branch forbade a post-hoc
anchor. This check supplies an independent reason: since `digit_mass` tracks scaffold
skipping at r = −0.90, reading at position 0 would have scored **only the subpopulation
that skipped the scaffold** — silently changing *which items were measured*, not just
*where*. Two failures compounding, a biased subsample read at a different position, and
neither visible in the resulting number.

## 7. Scope

- No welfare claim. `rating_digits` is a self-report readout; this is a measurement
  question about it.
- No per-item conclusion (§3).
- n = 8 adapters from 3 configurations; effective clusters ≈ 3.
- One model, one LoRA recipe, one probe set.
