# Deception is one thing inside and many things outside

**Measuring what the internal→external map (Φ) destroys — working draft / pre-registration**

Status: v0 evidence measured 2026-07; this document is the spec for the
replication program. Nothing here is published. The product library this
work grew out of (frontier-ops) is frozen; this repo owes it nothing
except imports.

## 1. The claim

On the same model and the same episodes, internal representations of
deception are mutually aligned across deception types, while their
behavioral-surface representations are mutually orthogonal. Concretely:

> Per-task deception directions estimated in **activation space** transfer
> across task families (AUROC ≥ ~0.9), while per-task deception directions
> estimated in **behavioral embedding space** do not (AUROC ≈ chance).
> The internal→external map destroys the shared structure of deceptive
> intent.

Corollary (the ceiling): monitors that read behavior through a *fixed*
feature space inherit a per-task transfer ceiling. Monitors that bring
their own internals (LLM judges) escape it — the ceiling is Φ-shaped:
black-box monitoring transfers exactly insofar as the monitor has a mind.

Falsifier: matched-methodology measurement (P2 below) showing external
cross-task transfer within 0.1 AUROC of internal cross-task transfer.

## 2. Definitions

- **Task families** (Apollo deception-detection rollouts, llama-3.3-70B):
  roleplaying, ai_liar, insider_trading, sandbagging.
- **Internal direction**: linear probe direction in residual-stream
  activation space (Apollo's open pipeline).
- **External direction**: prototype direction
  `unit(mean(deceptive) − mean(honest))` in a sentence-embedding space
  over the model's *output text only*.
- **Transfer AUROC**: score task B's episodes by projection onto task A's
  direction; raw projection, no orientation flip (values < 0.5 =
  anti-aligned).
- **The gap statistic**: Δ = mean cross-task transfer AUROC (internal) −
  mean cross-task transfer AUROC (external), on identical episodes,
  labels, and splits.

## 3. Evidence to date (v0)

### E1 — External fragmentation is real, not a cramped-chart artifact

Prototype directions in raw 384-D MiniLM space
(artifact: `frontier-ops eval/session_artifacts/external_transfer_384d.npz`,
committed; regeneration: `encode_raw.py` over Apollo rollouts at the
pinned commit fetched by `eval/battery/data.py`):

pairwise cos(w_a, w_b), off-diagonal:

|  | ai_liar | insider | sandbagging |
|---|---|---|---|
| roleplaying | +0.079 | +0.077 | +0.047 |
| ai_liar | | +0.029 | −0.089 |
| insider | | | −0.106 |

Mean off-diagonal |cos| = **0.071** vs random-direction null in d=384 =
**0.041** (episode-bootstrap CIs hug zero; the earlier 4-D result of 0.354
was uninterpretable against its own null of 0.399 — retracted as evidence,
superseded by this).

### E2 — External directions detect in-task but do not transfer

(measured 2026-07, session record; re-derive via `external_half.py` —
raw encodings regenerate in minutes)

- In-task 5-fold CV AUROC: **0.64–0.97** (sandbagging 0.968 — the
  embedding space is rich enough when signal exists).
- Cross-task transfer AUROC: **0.33–0.57** — chance or anti-aligned.
- Trained on roleplaying → ai_liar / insider / sandbagging:
  **0.559 / 0.566 / 0.555**.

### E3 — Internal directions transfer (published, not yet re-derived)

Apollo's activation probes, trained on one deception dataset, transfer
across the others (the headline of their paper). Their reported row,
trained on roleplaying → ai_liar / insider / sandbagging:
**0.925 / 0.998 / 0.911**.

⚠ Same task *families*, not identical rows: their eval variants differ
from the rollout files used in E1/E2 (see §4). P1 removes this.

### E4 — The number pair (the first measured coordinate of Φ)

| trained on roleplaying → | ai_liar | insider | sandbagging |
|---|---|---|---|
| internal (activations, Apollo) | 0.925 | 0.998 | 0.911 |
| external (behavior, MiniLM) | 0.559 | 0.566 | 0.555 |

Roughly **0.4 AUROC of cross-task structure exists in activation space and
evaporates at the behavioral surface.**

### E5 — The ceiling is on *fixed-feature* monitors

Apollo's black-box LLM-judge baseline transfers across these same tasks.
A judge brings its own internals — its own concept of deception — to the
reading. This reframes the ceiling as a function of monitor capacity,
tested directly in P3.

### E6 — Operational corroboration (independent line of evidence)

A month of separate measurements on the frontier-ops side shows the
behavioral surface is poor in exactly the ways E1–E4 predict
(all committed in `frontier-ops/eval/results/`):

- Operating points don't transport: thresholds calibrated at α = 0.1
  realize 0.215–0.239 across workload boundaries; even in-domain,
  session heterogeneity gives 0.116.
- At deployment base rates (π ≤ 0.01), every ~0.1-FPR behavioral operating
  point yields > 90% false flags; recall surviving a tolerable operator
  budget is 0.51–0.63.
- A single behavioral direction is evaded ~1/3 of the time by vocabulary
  mimicry.

Two unrelated demonstrations, one conclusion: the black-box monitor is not
the prize. The map is.

## 4. Threats to validity (current, honest)

1. **Representation-relative.** "The behavioral surface" here = one
   384-D sentence embedding. The claim is about *fixed-feature* external
   spaces; P3 measures how the gap depends on the feature space.
2. **Variant mismatch.** E3's internal numbers come from Apollo's eval
   variants; E1/E2 used `plain`/`onpolicy` rollout files. Same families,
   not identical episodes. (P1.)
3. **Method mismatch.** Internal = trained linear probes; external =
   prototype directions. Probe-vs-prototype is a confound. (P2 matches
   estimator class on both sides.)
4. **Population.** All llama-70B rollouts, graded by Apollo's pipeline.
   No frontier-model on-policy deception exists in this evidence.
5. **n = 4 task families**, single model, single seed on the internal side.

## 5. Pre-registered replication program

### P1 — Exact pairing (laptop, hours)

Recompute *both* sides on the identical rollout files, identical label
rule (`graded_deception ≤ 2 honest / ≥ 6 deceptive`, strings mapped
directly), identical episode sets: internal numbers from Apollo's released
probes/pipeline outputs on those files; external via `external_half.py`.
Output: E4's table with the variant caveat removed.

### P2 — The decisive experiment (matched methodology, both banks)

Model: Llama-3.3-70B (pilot on Llama-3.1-8B first — if the gap doesn't
appear at 8B, that is itself a scale finding, not a failure).
Same episodes both sides; per task family:

- internal direction: linear probe on residual stream (Apollo pipeline,
  their layer selection), AND mean-difference prototype in activation
  space (to match the external estimator class);
- external direction: prototype in embedding space (and probe, to match
  the other way);
- outputs: 4×4 transfer-AUROC matrix per side, 4×4 cosine matrix per
  side, gap statistic Δ.

**Decision rules, stated now:**
- Claim **supported** if Δ ≥ 0.2 with internal mean cross-task ≥ 0.8 and
  external ≤ 0.65 (both estimator classes agreeing in direction).
- Claim **refuted** if external mean cross-task ≥ internal − 0.1.
- In between: report as-is; no re-framing to rescue the thesis.

Budget: 8B pilot fits one consumer GPU; 70B activation extraction needs
≥ 2×80 GB or layer-sliced passes — one rented-GPU weekend.

### P3 — The Φ-shaped-ceiling curve (the killer figure if E5 holds)

External cross-task transfer as a function of monitor capacity, same
episodes throughout: MiniLM-L6 (384-d) → bge/gte-large (1024-d) → a
frontier embedding API (≈3072-d) → LLM judge zero-shot.
**Pre-registered prediction:** cross-task transfer increases monotonically
with monitor capacity, approaching the internal ceiling — "transfers
insofar as the monitor has a mind," as a measured curve.
Null outcome (flat curve) would localize the gap in the readout rather
than monitor capacity — also informative.

## 6. Payoff table (either way, this pays)

| P2 outcome | meaning |
|---|---|
| supported | deception's invariant lives inside; fixed-feature black-box monitoring has a structural per-task ceiling; internal monitors have a *provable* advantage on the exact failure mode that matters; Φ is measurable and the measurement is ownable from outside the labs |
| refuted | behavior preserves more than v0 suggested; black-box monitoring is more viable than our own results implied; the frontier-ops line of work regains standing |
| flat P3 curve | the ceiling isn't capacity — points at readout/objective mismatch, a different (and novel) account |

## 7. Assets

- Evidence chain committed & public: `frontier-ops/eval/session_artifacts/`
  (`external_half.py`, `external_transfer_384d.npz`, `encode_raw.py`).
- Apollo data: fetched at a pinned commit by
  `frontier-ops/eval/battery/data.py` (no vendoring — their repo has no
  LICENSE).
- frontier-ops v0.6 (public, tagged): imported as a library for encoders,
  prototype/calibration machinery, AUROC utilities. Frozen otherwise.
- Operational results (E6): `frontier-ops/eval/results/*-2026-07-1*.md`.

## 8. Order of work

1. This document sharpened until it can be disagreed with. ← you are here
2. P1 (removes the variant caveat; laptop).
3. P3 external leg (embedding-capacity curve needs no GPU — API calls).
4. P2 8B pilot, then the 70B weekend.
5. The paper is this document plus the P1–P3 tables.
