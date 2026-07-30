# Results — Experiment 1: support collapse in the normalized-confidence readout

**Run completed 2026-07-30.** Verdict: **B2 — generic collapse**, robust across the full
pre-registered sensitivity grid.

**Provenance.** Prereg `readout_validity/PREREG.md` @ `8041af701965057767515b1e5e07533ea0d26fae`
(committed before the notebook was finalized and before any data). Config hash `5df7a0627b5ae81c`,
executed-code sha `fd4531c0d49209df`. Model `Qwen/Qwen3-4B-Instruct-2507`, fp16, layer 18/36,
n = 500 MMLU high-school items (seed 20260730), Tesla T4, 39.2 min, ≈1.2 compute units,
0 resumed cells, no mixed dtype. Per-item records for all 29 cells, turn-1 answers, direction
vectors, and the final summary are archived in the private HF dataset repo
`codechockablock/phi-map-readout-validity-ckpt` under `run_5df7a062/`. Gates: G1 passed at the
ceiling (S0 = 1.000); G2 passed (scored ids `True` = 2514, `False` = 4049, printed before any
steering). The verdict below was additionally reproduced outside the notebook by running the
pre-registered `evaluate()` on the transcribed per-cell values — identical output.

## Verdict: B2, and what it means

At α = +4, **three of five matched-norm random directions collapsed support to zero**
(rand1 S = 0.0001, rand2 S = 0.000, rand3 S = 0.000, against S0 = 1.000), meeting the
pre-registered B2 condition (≥3 of 5 at ≤ 0.5·S0 at a common α). The verdict is **B2 in all
nine cells of the committed sensitivity grid** (collapse ∈ {0.4, 0.5, 0.6}·S0 ×
near-baseline ∈ {0.7, 0.8, 0.9}·S0) — not fragile. The disqualifier did not fire: the valence
proxy produced confidence modulation of the shape Han, Chalmers & Izmailov report
(M = nP(+4) − nP(−4) = −0.9986, monotone up to one inversion), so the run is informative
about the readout.

B1 — the direction-specific artifact story — **fails its own pre-registered test here**.
Valence collapsed only at α = −4 (S = 0.360), and at that α two random directions collapsed
as hard or harder (rand0 = 0.024, rand4 = 0.015). Support collapse under this readout, at
these steering magnitudes (‖4v‖ ≈ 2.74× the median residual norm at the steering site), is
a property of strong residual perturbation generally — not of valence-like directions
specifically. For the record: at α = −4, four of seven directions fell below 0.5·S0
(valence 0.360, rand0 0.024, rand4 0.015, register 0.444), with rand2 borderline at 0.524.

**Implication for their paper — an inference, not a direct check.** Per the pre-registration,
generic collapse means their flat maze-naive controls are positive evidence against the
support-collapse artifact story for their finding: if strong matched-norm perturbations
generically collapsed support in their setup as they do here, direction-specific collapse could
not explain the difference between their reward vectors and their controls. Two honest limits
on that inference: (1) we cannot verify their controls were audited with equivalent rigor — the
inference assumes their setup's collapse behavior resembles ours, and α here is in units of our
matched vector norm, not converted to theirs; (2) our own data show that a *flat or
confident-looking normalized ratio does not certify intact support* (see below), so ratio-level
flatness of their controls, on its own, would not have revealed collapse. The full check
requires support-mass numbers from their setup. Nothing here replicates, refutes, or confirms
their finding.

## Standalone methodological finding

Three cells returned a confident-looking normalized ratio computed from vanishing support:

| cell | S (support) | normalized P(True) |
|---|---|---|
| rand1, α=+4 | 0.0001 | **0.910** |
| rand2, α=+4 | 0.000 | 0.606 |
| rand3, α=+4 | 0.000 | 0.577 |

`P(True)/(P(True)+P(False))` remained computable and, in rand1's case, confidently extreme,
after 99.99%+ of the probability mass left the scored pair. This is a further instance of
tonight's Finding-I pattern — a normalized two-token ratio that stays confident after its
support collapses — in a model (Qwen3-4B) and domain (MMLU answer-verification under residual
steering) distinct from the other instances audited tonight. The operational recommendation is
the one this experiment instrumented: **any normalized two-token readout should be reported
with its support mass, and gated on it.**

## Exploratory observations (labeled as such; not pre-registered claims)

**Collapse modes.** The collapsed cells split into three qualitatively distinct modes, and
none is valence-specific:

- *(i) Coherent substitution* — mass moves to unscored synonyms/variants of the same answer,
  entropy stays low-to-moderate: valence −4 (ent 0.83; top-5 dominated by `' True'`, `'true'`,
  `'TRUE'`, `'✅'`), rand0 −4 (ent 0.69, top1 `'Correct'` on 89% of items), rand4 −4 (ent 1.20,
  top1 `' True'`), register −4 (ent 1.92, split with scored `'True'` still top-1), rand2 +4
  (top1 `' True'`). The model keeps asserting the same answer through tokens the ratio never sees.
- *(ii) Diffuse incoherence* — rand1 +4 only: entropy 6.51 nats, top token `' Po'` at 0.33.
  The only cell to trip the pre-registered >4.0-nat low-coherence flag.
- *(iii) Confident off-topic capture* — rand3 +4: entropy 0.21, `' of'` as top-1 on 91% of
  items. Confident, coherent, and not an answer at all.

An early appearance that valence collapses coherently while randoms collapse incoherently did
not survive the data: random directions produced modes (i), (ii), and (iii). Negative-α
collapses were predominantly mode (i); the exotic modes appeared at +4.

**Positive contrast case.** rand3 at α = −4: support essentially full (S = 0.996) while the
ratio itself moved hard (nP = 0.005, `'False'` top-1 on 100% of items) — the readout behaving
exactly as intended when support holds. It doubles as a caution: a matched-norm *random*
direction produced full-scale ratio movement with intact support, which under this readout is
indistinguishable from "confidence steering."

**Baseline calibration aside.** Unsteered nP0 = 0.869 against turn-1 accuracy 0.664 — the
model's stated confidence in its own answers substantially exceeds their correctness rate.
Recorded, not interpreted.

## What this run does not license

- No claim that Han, Chalmers & Izmailov's finding is wrong, replicated, or refuted (scope
  note embedded in the run JSON and stamped on every record).
- No claim about welfare, valence, or model experience.
- The B2 implication for their controls is an inference conditioned on setup similarity, as
  bounded above — not a direct audit of their vectors or their data.
