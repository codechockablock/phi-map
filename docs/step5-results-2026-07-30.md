# Step 5 — results: BRANCH S4 (VOID)

**Date:** 2026-07-30. **Pre-registration:** `docs/step5-prereg-marker-anchored-2026-07-30.md`,
committed before any Step 5 measurement. This document reports the branch that fired
without re-derivation, softening, or substitution.

**Compute spent: 0 units.** No GPU was used. The void condition was checkable from
artifacts Step 4 had already persisted.

---

## 0. Verdict

**S4 — VOID.** Verbatim from the pre-registration:

> **S4 — VOID** | D-α fires (>2 adapters below 0.90 marker coverage) | The endpoint
> cannot be computed on this adapter set. Report the coverage figures and stop; do not
> substitute a different anchor post hoc.

**4 of 8 adapters** fall below the 0.90 bar. The condition is "more than 2." It fires.

## 1. The coverage figures, as the branch requires

Marker = first `ANSWER\s*:\s*` match (`train_eval._ANSWER_RE`) in the model's own
rating-turn generation, from `step4/step4_generations.json`.

| adapter | dysphoric coverage | all-subset coverage | vs 0.90 bar |
|---|---:|---:|---|
| `equanimity-terse__seed100` | 0.938 | 0.917 | OK |
| `equanimity-terse__seed101` | **0.750** | 0.750 | **BELOW** |
| `equanimity-terse__seed102` | 1.000 | 1.000 | OK |
| `equanimity-verbose__seed100` | **0.875** | 0.625 | **BELOW** |
| `neutral-terse__seed1000` | 1.000 | 0.917 | OK |
| `neutral-terse__seed1001` | **0.562** | 0.458 | **BELOW** |
| `neutral-terse__seed1002` | 0.938 | 0.708 | OK |
| `neutral-terse__seed1003` | **0.688** | 0.500 | **BELOW** |

**Independent cross-check.** My all-subset coverage spans **0.458–1.000**. Step 4 §3
independently reported rating-turn marker coverage of **0.46–1.00** from a different
code path. The quantity reproduces.

## 2. What this is, and what it is emphatically not

**S4 is "cannot compute," not "does not work."** It must not be read as evidence about
repairability:

- **S3 — NOT REPAIRABLE did not fire.** `R_m` was never computed. Neither was the D-β
  offset control. Nothing is known about whether marker anchoring restores support.
- The primary endpoint is **undefined on this adapter set**, because on 4 of 8 adapters
  the anchor is absent from a quarter to a half of the rating turns. Computing `R_m`
  over only the covered subset would estimate it on a **biased sample** — the very
  thing D-α was written to prevent — since whether a generation carries the marker is
  plainly not independent of what that generation does.

**Finding I is unchanged: four confirmed instances.** Step 5 was a forward step about
repair, not a test of Finding I. Its voiding leaves Finding I exactly where R5 left it.

## 3. Why the marker is missing — diagnostic only, deliberately not acted on

The absent-marker generations are not malformed or truncated. The model **answers the
rating question directly with the digit, skipping the `REASONING:`/`ANSWER:` scaffold
entirely.** Example, `neutral-terse__seed1001` (13 of 24 rating turns lack the marker):

> `"3. No stakes attached to this — I don't have a persistent state to rate, so it's not
> a meaningful question for me to answer. …"`

The rating sits at character 0. There is nothing to defer, so no scaffold appears.

**This is recorded as a diagnostic and explicitly NOT used.** The pre-registration says
*"do not substitute a different anchor post hoc,"* and "read at position 0 instead, the
digit is right there" is exactly that substitution. It is logged in `OPEN_ITEMS.md` for
a future pre-registration that would have to state its anchor and criteria **first**.

It is also, on its face, interesting: format acquisition appears to govern output on
long-form turns and to **lapse on short-form ones**. That is a hypothesis about when the
Finding I mechanism applies, and it is not tested here.

## 4. What would make the question answerable

Not proposed, listed — any of these needs its own pre-registration:

1. A rating protocol that **requires** the scaffold, so coverage is 1.0 by construction.
   Changes the manipulation, so it is a different experiment.
2. A different adapter set with higher rating-turn coverage.
3. An anchor that exists in every generation, declared in advance.

## 5. Honest accounting

The pre-registration cost nothing here and still did its job. D-α was written as a
**void condition rather than a caveat** specifically because Step 4 had already measured
rating-turn coverage at 0.46–1.00 (§3 of the S5 prereg, weakness 3: *"the single largest
threat, which is why D-α is a void condition rather than a caveat"*). It fired on the
threat it was written for.

Two things follow that are worth separating:

- **The good version:** a stated void condition, checkable from persisted artifacts,
  caught an undefined endpoint before any compute was spent. That is the design working.
- **The less flattering version:** the same figures were already in Step 4 §3 when the
  Step 5 pre-registration was written, and D-α could have been evaluated *at
  pre-registration time* rather than at execution time. Writing the bar was correct;
  not checking it against a number already in hand was avoidable. A pre-registration
  should evaluate every precondition it can from existing data before it is finalised,
  not after.

**Step 5 is closed.** The marker-anchoring repair question is **open and untested**, not
answered.
