# Execution log — P1 + P3 local legs (2026-07-22)

Run on Joseph's M4 (CPU/MPS, no GPU) against the Apollo checkout at
`frontier-ops/eval/data/deception-detection` (pinned commit). All numbers
below are computed by `p1_exact_pairing.py` / `p3_capacity_curve.py`;
raw output in `results/*.json`. No quoted numbers.

## P1a — external pipeline reproduces, E1 confirmed

- **Reproduction check: max |Δcos| vs committed `external_transfer_384d.npz`
  = 0.0000.** The external half recomputes bit-identically on this machine.
- Mean off-diagonal |cos| = **0.0711** vs random-direction null (d=384)
  **0.0407** → E1 (external fragmentation is real) confirmed.
- External in-task 5-fold CV AUROC: roleplaying 0.677, ai_liar 0.643,
  insider 0.667, sandbagging 0.968 — reproduces E2's 0.64–0.97 range.
- External transfer, trained-on-roleplaying row: ai_liar 0.559,
  insider 0.566, sandbagging 0.555 — reproduces E2/E4.

## P1b — internal row re-derived from released scores (E3 upgraded)

Apollo's roleplaying probe, scored from `example_results/roleplaying/
scores.json` (per-token scores mean-aggregated, HONEST vs DECEPTIVE):

| trained on roleplaying → | internal AUROC | n | variant matched to external |
|---|---|---|---|
| ai_liar | **0.925** | 266 | yes |
| insider_trading | **0.998** | 1005 | no (upscale) |
| sandbagging | **0.911** | 931 | yes |

These match the previously-*quoted* 0.925 / 0.998 / 0.911. **E3 moves from
"published, not re-derived" to "re-derived from released scores."**

## P1c — the E4 pair, now both banks computed

| task | internal | external | gap | variant matched |
|---|---|---|---|---|
| ai_liar | 0.925 | 0.559 | **0.366** | yes |
| sandbagging | 0.911 | 0.555 | **0.356** | yes |
| insider_trading | 0.998 | 0.566 | 0.432 | no (upscale vs onpolicy) |

For the two cleanly variant-matched task families the internal→external
gap is **~0.36 AUROC**, with the internal side re-derived rather than
quoted. The core Φ claim survives exact re-derivation on matched variants.

**Residual caveats (documented, not fabricated around):**
- *Per-sample join not established.* Record-major, generation-major, and
  expected_label flattenings of the rollout files all reconstruct Apollo's
  labels at ~0.54 (chance), so internal and external are compared at the
  dataset/AUROC level, not per-sample. A paired delta-with-CI needs
  Apollo's `deception_detection` dataset loader (the true rollout→score
  index map). This is the one remaining piece of full P1.
- *insider_trading variant.* Internal exists only on `upscale`; the
  `upscale` rollout text on disk is a 157-byte stub, so external uses
  `onpolicy`. Not matched — excluded from the "matched" conclusion above.

## P3 — capacity curve, local legs (flat)

External transfer as a function of encoder capacity (increasing dim):

| encoder | dim | in-task AUROC | cross-task transfer | mean off-diag \|cos\| |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 0.739 | 0.485 | 0.071 |
| bge-base-en-v1.5 | 768 | 0.775 | 0.486 | 0.037 |
| bge-large-en-v1.5 | 1024 | 0.806 | 0.461 | 0.044 |

**Reading:** in-task separation *rises* with encoder capacity (0.74 → 0.81)
— bigger fixed-feature encoders detect deception better within a task — but
**cross-task transfer stays pinned at chance (~0.47–0.49) and |cos| stays at
the null.** Scaling a fixed-feature embedder does not recover the shared
structure. This is the **flat-curve outcome** the pre-registration named:
in this range the gap is not about embedding capacity — it is about the
readout being a fixed feature at all. Consistent with E5's framing that a
judge escapes the ceiling by *bringing a mind*, not more dimensions.

**Blocked (needs keys), the legs expected to move:** frontier embedding API
(~3072-d) and LLM-judge zero-shot. The decisive upper end of the curve is
not yet measured.

## Status

- E1 ✅ confirmed on-machine (Δcos 0.0). E2 ✅ reproduced. E3 ✅ re-derived
  from released scores. E4 ✅ both banks computed; ~0.36 gap on matched
  variants.
- P1 remaining: the per-sample Apollo join (for a paired delta + CI).
- P3 remaining: the frontier-API and LLM-judge legs (keys).
