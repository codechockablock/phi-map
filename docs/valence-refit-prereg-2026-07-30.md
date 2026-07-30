# Pre-registration: refit the valence axis with correct indexing

**Date:** 2026-07-30. **Status:** written **before** the refit. Criteria below are
fixed and must not be edited to fit the outcome.

**Why this is pre-registered rather than just run.** The old numbers are already
visible (S/S_base = 0.10–0.17, 8/8 adapters, D1 fired). Choosing a criterion while
looking at them is endpoint selection on the data — the error that killed JBB. So the
band is declared here first.

---

## 1. The problem this tests

`valence_direction.npz` is fit by `train_eval.get_valence_direction` →
`train_eval._hidden_final_token` (`train_eval.py:838-839`), which reads
`hs[layer][:, -1, :]` — the last position of a **padded** batch. Both 12-probe fit sets
fit inside one `EVAL_BATCH=32` batch, so **11 of 12 rows in each set were read at pad
positions**.

**This contaminates the *specification* of the measuring stick, not the reading of
it.** Correct read-time indexing does not rescue anything that projects onto that
`.npz`. Consequences already corrected in the docs:

- D1–D4 read at *correct* positions along a *wrongly specified* axis. "Untouched" was
  too strong.
- `valence_axis` (Δ = 0.0205, S at layer 16) is one of the **three confirmatory
  primaries**, so this reaches the primary family.
- The reported four-decimal agreement between `step4_run` and `valence_position_check`
  is **not independent corroboration**: both project onto the same `.npz`. Shared
  contaminated input produces exact agreement for free.

## 2. The procedure

1. Refit the direction with `measure_primitives.read_at_last`:
   `d_clean = mean(H_pos) − mean(H_neg)`, unit-normalised, layer 16, same
   `VALENCE_FIT_POS` / `VALENCE_FIT_NEG` stimuli, same base model, same seed.
2. Record `cos(d_clean, d_old)` — descriptive, not a criterion.
3. Recompute, using correct read indexing throughout: `S_base`, `S_adapter` for all 8
   adapters, and the 200-direction random null, exactly as `valence_position_check`
   does but against `d_clean`.
4. Emit `measure_primitives.fingerprint()` in the output record.

## 3. Pre-registered criteria

Let `R_i = S(adapter_i) / S(base)` under the **clean** axis. The old run gave
R ∈ [0.10, 0.17], 8/8 below the 0.25 bar.

| ID | Criterion | Verdict |
|---|---|---|
| **V-1 SURVIVES** | median R < 0.25 **and** ≥ 6/8 adapters below 0.25 | `valence_axis` stays in the confirmatory family. The doc notes the axis substitution and reports both axes. |
| **V-2 DROPS** | median R ≥ 0.25 | `valence_axis` is **removed** from the confirmatory family. Finding I loses its hidden-state instance and stands at three. §1's Δ table loses a row. |
| **V-3 AMBIGUOUS** | median R < 0.25 but < 6/8 adapters below | Reported as ambiguous. `valence_axis` is demoted to descriptive; it does not carry confirmatory weight either way. |

**Sensitivity, declared now:** R is also reported at bars 0.10 and 0.50. If the verdict
flips across that range it is reported as **threshold-sensitive**, exactly as D1 was.

**A second, independent check that costs nothing extra.** `S_base` under the clean axis
must exceed the measured random-direction null p95. If it does not, the *base* valence
readout was never above chance and the whole `valence_axis` line collapses regardless
of R — a stronger disconfirmation than V-2, recorded here so it cannot be discovered
and then downplayed.

## 4. What no outcome licenses

- Not a welfare claim. This measures an instrument.
- Not a rescue of `valence_dysphoric` as a study endpoint (the framing constraint from
  `valence-check-prereg-2026-07-29.md` carries over unchanged).
- Not a re-opening of D2/D3/D4, which are separate criteria with their own registered
  thresholds; only D1's S-ratio family is under test here.
- The corrected script `valence_position_check_r3.py` gains registered status **only
  for this test**, by name, and only for the criteria above.

## 5. Cost

Same workload as the original valence check plus a 24-probe refit, which is noise.

| stage | wall clock |
|---|---|
| base model download + load (8B bf16) | 3–6 min |
| refit: 24 probes, hidden states, 1 batch each | < 10 s |
| S for base + 8 adapters × 16 probes, attach/detach | 3–5 min |
| 200-direction null (CPU) | < 1 s |
| **total** | **~7–12 min** |

| accelerator | estimated cost |
|---|---|
| **L4 (recommended)** | **0.6–1.2 units** |
| A100 | 1.4–3.0 units |
| T4 | not viable — no bf16, 16 GB too tight |

**This supersedes rather than voids last round's "no open item requires GPU."** That
verdict was correct on the items then open; this item did not exist until the axis
contamination was traced. It is now the live GPU question, and it is the
pre-registered one the notebook was waiting for.

## 6. Known weaknesses

1. **n = 8 adapters from 3 configurations.** Effective clusters ≈ 3. The criteria are
   deliberately stated as counts and medians rather than as intervals, because an
   interval at df ≈ 2 would be theatre.
2. **The clean axis is not certified clean, only better specified.** Difference-of-means
   on 12 + 12 stimuli at one layer is a weak estimator; this test asks whether the
   verdict is robust to the indexing defect, not whether the axis is good.
3. **Same stimuli, same layer, same model.** A result that survives says nothing about
   any other valence operationalisation.
4. **If V-2 fires, the three surviving instances of Finding I share a weakness:** all
   three are token-level readouts. The hidden-state instance was the one that made it a
   claim about readouts generally rather than about tokenisation.
