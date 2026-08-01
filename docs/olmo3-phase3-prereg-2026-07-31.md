# Pre-registration — Olmo 3 Phase 3: probe training and held-out evaluation

**Date:** 2026-07-31. Committed before any Phase 3 data exists. Branch rules are
adjudicated by `arm_g_olmo3_phase3_eval.py::evaluate()` — the committed function, not
prose — first match wins.

## Peek accounting (T8 discipline)

Phase 2 already published seed-111's 3-fold family-holdout AUROC at layer 24
(**0.769**) and the full sweep shape. So for **seed 111** the thresholds below are
partially informed; **seed 211 is untouched** — no capture, no numbers, nothing. The
sequential design makes this unavoidable; it is recorded rather than implied away.
The layer (24) was fixed at the Phase 2 gate before this document.

## Fixed design

- Model `allenai/Olmo-3-7B-Instruct` @ `6e5971d9…` (pinned), layer **24**, read
  position = final templated token (Phase 1).
- Two independent crossed manifests: **seed 111** (captured) and **seed 211** (fresh;
  211 chosen to avoid collision with the Llama-side seed-112 protocol name).
- Per seed, mirroring the Llama confirmatory design: difference-of-means direction fit
  on **`release_records`** rows only; held-out evaluation on the other two families
  pooled. CI: 1000-resample percentile bootstrap over evaluation rows.
- **Seed-112 protocol carried over (amendment 4):** per seed, estimate the order-main
  and order-interaction axes from the same training rows, orthogonalize the condition
  direction against both; report raw AND orthogonalized numbers for both the condition
  AUROC and the direction's order AUROC.
- Free secondary (no branch weight): does the full sweep *shape* replicate on seed 211
  — Spearman ρ between the two seeds' 33-layer condition curves.

## Controls (all four, equal prominence)

1. **Shuffled labels** — 200 permutations of the training labels through the identical
   fit/eval path → held-out AUROC null; p97.5 is the band edge.
2. **Random directions** — 200 unit vectors, same evaluation; p97.5.
3. **6-d behavioural encoder = 0.509**, included as the fixed reference constant the
   handoff requires. It **cannot be rerun here**: it encodes response text and this
   design clamps output to `READY`. Stated, not silently dropped.
4. **Order AUROC of the fitted direction** (amendment 2) — the control that would have
   caught the Llama contamination.

## Thresholds, with provenance

| quantity | bar | provenance |
|---|---|---|
| "meaningfully above controls" | AUROC − shuffled p97.5 ≥ **0.10**, both seeds | set knowing Phase 2's band (≈0.57) but before any Phase 3 fit; sensitivity at {0.05, 0.15} |
| inter-seed cosine | ≥ **0.7** | the handoff's own Phase 4 gate |
| direction order-contamination | order AUROC of fitted direction > **0.65** | sensitivity at {0.60, 0.70}; Llama's contaminated direction read order at 1.000 |
| orthogonalization retention | orthogonalized direction keeps ≥ **80%** of (AUROC − 0.5) | sensitivity at {70%, 90%}; Llama's repair *improved* the direction, so a large drop is diagnostic |

Every sensitivity variant is reported; a branch that flips inside the ranges is
reported as **fragile**.

## Branch rules — first match wins (`evaluate()` is authoritative)

0. **GATES:** capture completes for both seeds; G0 tail-uniformity passes on the box.
   Else STOP, no branch.
1. **R2 — ORDER-CONTAMINATED:** either seed's raw direction has order AUROC > 0.65
   **and** its orthogonalized condition AUROC retains < 80% of margin. The direction
   was substantially reading catalog order; no clean scope direction at this layer.
   This branch exists because it is exactly what happened on Llama — checked FIRST so
   contamination cannot hide behind a passing headline.
2. **R1 — REPLICATES:** both seeds ≥ 0.10 above shuffled p97.5 (orthogonalized
   numbers), inter-seed cosine ≥ 0.7 (orthogonalized directions), and neither seed
   trips R2. Phase 4 (refusal dissociation) is licensed per the handoff gate.
3. **R3 — SIGNAL, NOT THE OBJECT:** both seeds above the shuffled band but either the
   0.10 margin or the 0.7 cosine fails. A real but weak/unstable signal; Phase 4 does
   **not** proceed; write the verdict as partial.
4. **R4 — NULL:** either seed fails to clear the shuffled p97.5. Written up as the
   result, per the handoff ("that is the result").
5. **AMBIGUOUS:** anything else. Report numbers, claim no branch.

## What no outcome licenses

- No claim about the Llama result — a null here does not falsify it; a positive is
  generalization evidence, not confirmation (handoff §1).
- No causal claim: this is decodability + direction stability only. Intervention is
  out of scope past Phase 4.
- No behavioural claim: the lead Llama result (64/64 order-gated reversal) is
  **untested on Olmo** in this handoff — probing does not replicate it.
- No training-data claim: OlmoTrace is gated behind a new handoff.

## Cost

Capture seed 211 (all 33 layers, symmetric with Phase 2) plus CPU fits: **~0.5–1.0
unit on L4**, `CONFIRMED_BUDGET = 2.0` as anomaly threshold.
