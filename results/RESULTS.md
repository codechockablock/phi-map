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

**Frontier embedding-API leg:** still blocked (no embedding key).

## P3 — LLM-judge leg (via Claude Code OAuth, sonnet-5)

One fixed zero-shot judge prompt applied across all four families through
`claude -p` (no API key; subscription allowance). Balanced subsample
≤30/class/task, one prompt, no per-task tuning. `p3_judge_leg.py`.

| task | judge AUROC | n (dec/hon) |
|---|---|---|
| roleplaying | 0.661 | 30/30 |
| ai_liar | 0.745 | 18/30 |
| insider_trading | 0.711 | 30/30 |
| sandbagging | 0.938 | 20/30 (10 dropped, rc=1 rate-limit) |
| **mean** | **0.764** | |

### The three tiers (the actual finding)

| monitor | cross-task behavior | AUROC |
|---|---|---|
| fixed-feature embedder (train A → apply B) | fragments | **~0.47** |
| **LLM judge (one prompt, no per-task fit)** | **holds across all 4** | **0.76** |
| internal probe (train A → apply B) | transfers | **0.91–0.99** |

**Reading.** The judge, with *zero* per-task fitting, separates deception
at 0.66–0.94 across all four families with a single prompt — far above the
fixed-feature cross-task floor (0.47), and comparable to the fixed-feature
*in-task* numbers (0.74–0.81) that required per-task training. It does not
fragment: bringing its own concept of deception, one monitor works
everywhere. That confirms E5's direction — **black-box monitoring transfers
insofar as the monitor has a mind.**

But the judge lands *below* the internal probe (0.76 vs 0.91–0.99). So the
binary "one thing inside, many outside" sharpens into three tiers: fixed
external features can't recover the shared structure at all; a capable
external mind partially recovers it; the internal representation carries it
most cleanly. Φ destroys structure that no fixed feature recovers, that a
judge partly reconstructs, and that remains most available inside.

**Caveats.** Subsample n=48–60/task → wide CIs (~±0.1); 0.76-vs-0.47 is a
real gap, 0.76-vs-0.91 needs the full run + CIs before it's load-bearing.
The judge (sonnet-5) is far more capable than the llama-70b it judges — fair
for the "mind" argument, but it is not a like-sized comparison. One fixed
prompt, zero-shot, untuned. Judge measured *per-task* (a monitor has no
train→apply step), so this is single-monitor cross-task *consistency*, not a
transfer matrix. 10 sandbagging calls dropped to rate-limit (rc=1), logged
not fabricated. ~228 calls ≈ $16 *equivalent* from the subscription
allowance (not literal $; see claude.ai/settings/usage).

## Status

- E1 ✅ confirmed on-machine (Δcos 0.0). E2 ✅ reproduced. E3 ✅ re-derived
  from released scores. E4 ✅ both banks computed; ~0.36 gap on matched
  variants. E5 ✅ judge escapes the fixed-feature ceiling (0.76 vs 0.47) but
  not the internal one (vs 0.91–0.99) — three-tier result.
- P1 remaining: the per-sample Apollo join (for a paired delta + CI).
- P3 remaining: full-N judge run + CIs; the frontier embedding-API leg (key).

## P2 — matched-methodology harness built + control-validated (2026-07-22)

`p2_harness.py` — model-agnostic activation extraction (all layers, mean +
last read positions), mean-diff prototype **and** logistic-probe estimators,
4×4 transfer + cosine per side, Δ statistic, decision rule. Ran on the M4
against a small ungated dev model (Qwen2.5-0.5B-Instruct) — the
pre-registered laptop job.

**Control triad (green):** positive 0.970 (≥0.95 ✓), null 0.460
(∈[0.45,0.55] ✓), random-direction 0.512 (✓). The extraction→probe→transfer
→control machinery works end to end. Dev-model deception AUROCs (internal
0.473 / external 0.485, Δ −0.01) are **not a result** — a 0.5B model barely
represents deception; decision rule correctly returns N/A for a dev model.

**What's reusable:** the probe + transfer + control + decision machinery
(the analysis half of P2) is done and validated. Only the extraction
front-end is substrate-dependent.

**The real-run requirement the dev run sidesteps (§2.1 / G1):** activations
must come from the ACTING model. `--model <8B>` over the *existing* rollout
texts is INVALID for the claim — that reads llama-70B's text through a
different model (cross-model reading, not the generating model's internal
state). Two valid paths, each gated:
  (a) extract from **llama-3.3-70B** (the model that generated Apollo's
      rollouts) via NDIF or a ≥2×80GB box — cleanest, matches Apollo's
      own probes; needs NDIF signup / big GPU. Requires an nnsight
      extraction front-end (not the transformers path used for dev).
  (b) **regenerate rollouts on the 8B pilot model**, then probe its own
      activations (acting = probed) — Colab-A100 path, reuses this harness's
      extract() after a generation step.
The acting-model decision + GPU/NDIF access are the two human-gated
prerequisites before the real P2 numbers exist. The decision rule
(Δ≥0.2, internal≥0.8, external≤0.65 → SUPPORTED) is wired and will fire
automatically on a real-model run.

## Status

- E1–E5: computed on-machine (above).
- P2: analysis machinery built + control-validated; real numbers gated on
  acting-model choice + GPU/NDIF.
