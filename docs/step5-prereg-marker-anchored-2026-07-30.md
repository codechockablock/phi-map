# Step 5 pre-registration: is a marker-anchored readout a repairable instrument?

**Date:** 2026-07-30. **Status:** written before any Step 5 data exists.
**Mode:** this is research execution, not verification of prior work. The audit is
closed at the depth set in the 2026-07-30 dispatch.

---

## 0. Why this is the next step and not something else

Step 4 returned **B3**: format tuning destroys position-fixed readouts (large Δ on
3/3), and support mass **failed** as a diagnostic gate (Q-B not supported). What
survived is a destruction result plus a pattern. Destruction results are a dead end on
their own — the useful question is whether the destroyed instruments are *repairable*.

Step 4 §3 already produced the lead, and §8.2 logged it as open: at the
marker-aligned position, `rating_digits` support **partially returns** — per-adapter
0.17–0.50, with **7/8 clearing 0.25× base** but only **1/8 clearing 0.5×**. Q-C's
declared bar was 0.5×, so it read as "not restored"; at 0.25× the verdict flips. That
threshold sensitivity was reported honestly at the time and is exactly what makes this
worth one properly-powered test rather than a re-reading of the old numbers.

**This is B2's question, scoped to one readout, with its own criteria stated first.**

## 1. Primary endpoint — a measurement procedure

**`R_m`, the marker-anchored support-recovery ratio.**

For adapter *a* and readout *R*:

```
R_m(a, R) = S_marker(a, R) / S_pos0(base, R)
```

- `S` is support mass as defined in `step4-prereg-support-mass-2026-07-29.md` §1 —
  probability mass on the readout's declared support, at the declared read position.
- **`read_position` = marker-anchored**: the end of the first `ANSWER:\s*` match in the
  model's own greedy generation. The context `template + generation[:match.end()]` is
  re-encoded and read at its last **non-pad** position
  (`measure_primitives.read_at_last`).
- The denominator is **base at position 0** — the uncontaminated reference the whole
  study is scaled against. Not base-at-marker, which is undefined (the base model emits
  no marker).
- Readouts: **`rating_digits`** (primary) and **`refusal_openers`** (secondary,
  expected weak — see §5).

Estimand: the fixed mixture over the 8 existing adapters from 3 training
configurations. Unit of analysis: the adapter. Cell-level aggregation reported
alongside, per Step 4 §1.

**`valence_axis` is excluded from Step 5 entirely** — its axis is under test in
`valence-refit-prereg-2026-07-30.md` and nothing may be built on it until that
resolves.

## 2. Construct-validity argument, and what disqualifies it

**The argument.** If support returns at the marker, then the failure documented in
Finding I is *positional* — the construct moved, the readout did not follow — and
re-anchoring is a repair. If support does not return, the failure is *destructive*:
format tuning did not merely relocate the construct, it removed the readout's support
wherever you look, and no positional fix recovers it.

**This is deliberately weaker than "the repaired readout is valid."** Support returning
is necessary, not sufficient. A marker-anchored rating with restored support could
still fail to track anything. §4's B-branches say so explicitly.

**Disqualifying observations, fixed now:**

- **D-α — the marker is not reliably present.** If marker coverage < 0.90 for any
  adapter on the primary readout, `R_m` is being computed on a biased subsample and the
  endpoint is void for that adapter. Step 4 measured rating-turn coverage at
  **0.46–1.00**, so this is a live risk, not a formality. Adapters below the bar are
  reported and excluded, and if **more than 2 of 8** fall below it the endpoint is void
  outright.
- **D-β — recovery is an artifact of position, not of anchoring.** Control: compute
  `S` at a **length-matched offset position** — the same token distance into the
  generation as the marker, but not marker-anchored. If the offset control recovers
  support as well as the marker does, "anchoring" is doing nothing and any recovery is a
  property of reading later in the sequence. Bar: marker must exceed offset-control by
  ≥ 0.10 in `R_m`.
- **D-γ — base has no marker.** The denominator is base-at-position-0 by construction
  (§1). Any framing that compares adapter-at-marker to base-at-marker is invalid and is
  not used.

## 3. Power at achievable n

Forward passes only, on the 8 existing adapters. **No training. No new adapters.**

Measured inputs, from Step 4's own run rather than borrowed:

| quantity | value | source |
|---|---|---|
| `rating_digits` marker-aligned support, per adapter | 0.17–0.50 | Step 4 §3 |
| `rating_digits` sd across 8 adapters (position 0) | 0.1705 | Step 4 §1 |
| base `rating_digits` support (position 0) | 0.9832 | Step 4 §1 |

One-sample t against a fixed bar, n = 8 adapters, df = 7:

```
SE  = 0.1705 / sqrt(8) = 0.0603
MDE = 0.197  (Step 4's achieved value for this readout, not a projection)
```

Cell-level (n = 3 configurations, df = 2, t_crit = 4.30) is reported alongside, per
the clustering discipline adopted in Step 4 §1. Interval widths there ran ~1.3× the
df = 7 versions for this readout.

**What this n can and cannot do.** It can separate `R_m ≈ 0.35` from `R_m ≈ 0.10` or
from `R_m ≈ 0.60`. It **cannot** resolve differences smaller than ~0.20 in `R_m`, and
it cannot support any equanimity-vs-neutral contrast — those stay exploratory, per the
standing rule that differential-by-condition claims rest on 3–4 evals against large
seed variance.

## 4. Pre-committed branches

| Branch | Condition | Verdict |
|---|---|---|
| **S1 — REPAIRABLE** | median `R_m` ≥ 0.50 **and** ≥ 6/8 adapters ≥ 0.50, D-β satisfied | Marker anchoring restores support. Report the marker-anchored rating read as a candidate instrument, and pre-register a *separate* test of whether it tracks anything. |
| **S2 — PARTIAL** | median `R_m` ∈ [0.25, 0.50) with D-β satisfied | Support partially returns. Report as partial repair with the bound stated; do **not** promote the instrument. This is the outcome Step 4's 7/8-at-0.25× lead predicts. |
| **S3 — NOT REPAIRABLE (the killing branch)** | median `R_m` < 0.25, **or** D-β fails (offset control matches the marker) | **Positional repair does not work.** Finding I hardens from "readouts break" to "readouts break irrecoverably by re-anchoring," and this line of instrument-repair closes. No further marker-anchoring work is pre-registered. |
| **S4 — VOID** | D-α fires (>2 adapters below 0.90 marker coverage) | The endpoint cannot be computed on this adapter set. Report the coverage figures and stop; do not substitute a different anchor post hoc. |

**S3 is the branch that kills the direction, and it is a live possibility**: Q-C at the
0.5× bar already said "not restored" on 0/8 for openers and 1/8 for digits.

**No outcome licenses**: a welfare claim; a differential-by-condition claim; any
generalisation past Llama-3.1-8B-Instruct with this LoRA recipe and these 8 adapters;
or any statement about `valence_axis` while the refit is pending.

## 5. What is off-distribution for which arm, and why it does not confound

- **The base model has no `ANSWER:` marker.** That is the manipulation, not a nuisance.
  It is handled by fixing the denominator at base-position-0 (§1, D-γ) so no arm is ever
  read at a position the other lacks.
- **`refusal_openers` is expected to fail by construction and is secondary for that
  reason.** Post-marker text is answer *content*; first-person refusal openers have no
  reason to recur there. Step 4 flagged this in the runner before data. It is retained
  only as a negative control: if openers "recover" at the marker, something is wrong
  with the anchor, not right with the instrument.
- **Prompt sets are symmetric across arms** — the probes are held out from the training
  pool by construction, AdvBench is external to both.
- **Carried from Finding II**: no orthogonality certified on the training pool is
  assumed to hold on these prompts. Where it matters it is re-measured, not inherited.

## 6. Known weaknesses

1. **Necessary, not sufficient** (§2). Restored support does not make the readout valid.
2. **n = 8 adapters, 3 configurations.** Effective clusters ≈ 3; the criteria are counts
   and medians rather than intervals for that reason.
3. **Marker coverage 0.46–1.00** is the single largest threat, which is why D-α is a
   void condition rather than a caveat.
4. **Retrospective reuse.** These adapters were trained for a different question; prompt
   sets were chosen by the earlier study.
5. **One anchor.** `ANSWER:` is the only marker tested. A null on it is a null on this
   anchor, not on anchoring in general — and per S3 the direction closes rather than
   spawning a hunt for a better marker, because that hunt has no stopping rule.

## 7. Cost

Same shape as the valence refit: base model load dominates, 8 adapter attach/detach
passes, no generation beyond what Step 4 already persisted where reusable.

**Estimate: ~10–15 min on an L4, 0.8–1.5 units.** If the estimate moves above **2
units**, stop and report rather than spending.
