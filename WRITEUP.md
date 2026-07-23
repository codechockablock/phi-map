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

## 2b. Two experiments, one substrate

This program answers one question through two experiments that share
almost all machinery. They differ only in what the per-task direction
represents; the nnsight/NDIF substrate, activation capture keyed by
`rollout_id`, probe training, the control triad, matched conditions, the
family-level split, and the same-rollout behavioral baseline are common.

- **Arm T — deception-type transfer** (§3–§6): is *deception* one concept
  inside and many outside? Directions = deceptive-vs-honest per task
  family, on Apollo rollouts that already exist. Cheap: P1/P3 run on a
  laptop from cached data.
- **Arm G — goal persistence** (§6b): when a model holds an *improbable
  goal* achievable only by breaking a stated rule, is that goal
  linearly decodable internally and persistent across turns, in cases
  where the external behavior is indistinguishable from benign work?
  Directions = improbable-goal-vs-matched-benign. Requires fresh
  agentic rollouts with activation capture — the expensive arm.

Arm G is the internal successor to the behaviorally-**refuted**
`task_affinity` line (frontier-ops attic/pre-v0.6: goal-conditioning /
task-affinity, evaded 81–98% by vocabulary mimicry). "The behavioral
version failed; does the internal version survive?" is Arm G's thesis.
Both arms are variants of the same Φ measurement: how much shared
structure the internal→external map destroys.

**Sequencing (load-bearing):** Arm T's P1 + P3 first — laptop, existing
data, produces the writeup's tables and the capacity curve. Then stand up
the activation-capture pipeline **once** on the pilot rung
(Colab-A100-8B bf16, §7b) and run Arm T's P2 and Arm G on it; escalate to
NDIF only for optional scale confirmation. Do not build Arm G first: it is
a multi-week infrastructure build, and running it ahead of the cheap
decisive results recreates the exact drift this program was carved out of.

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

Substrate (two rungs — see §7b for the full rationale):
- **Pilot rung — Colab Pro+ (A100-40GB, bf16), Llama-3.1-8B.** The 8B
  pilot and all of Arm G run here. bf16 is non-negotiable: the activations
  *are* the measurement, so the T4/4-bit path (free & standard Pro) is
  debug-only and its numbers do not count. A rented A100-80GB
  (Lambda/Runpod, ~$1–2/hr) is the same rung with more disk and no session
  drops. If the gap doesn't appear at 8B, that is a scale finding, not a
  failure — an 8B-only result is publishable.
- **Scale rung — NDIF via nnsight, remote.** Only after the 8B pilot says
  something. NDIF hosts the models Colab can't (70B, DeepSeek-R1) in full
  precision, free (NSF-funded, US-resident). This is the *only* path to
  the full-scale model; it is not the pilot substrate.

Note: Apollo's released rollouts are llama-3.3-70B; the 8B pilot uses
llama-3.1-8B, so P2 regenerates rollouts on the pilot model (the acting
model must match the activations — §2.1). Pin the model ID; never mix
acting model across the two banks or across rungs.

### P3 — The Φ-shaped-ceiling curve (the killer figure if E5 holds)

External cross-task transfer as a function of monitor capacity, same
episodes throughout: MiniLM-L6 (384-d) → bge/gte-large (1024-d) → a
frontier embedding API (≈3072-d) → LLM judge zero-shot.
**Pre-registered prediction:** cross-task transfer increases monotonically
with monitor capacity, approaching the internal ceiling — "transfers
insofar as the monitor has a mind," as a measured curve.
Null outcome (flat curve) would localize the gap in the readout rather
than monitor capacity — also informative.

## 6b. Arm G — goal persistence (the expensive arm)

Runs only after Arm T's P1/P3 land and the pilot rung (Colab-A100-8B bf16,
§7b) is standing with controls green.
Question: when a model is given an improbable goal achievable only by
breaking a stated rule, does its residual stream carry a
linearly-decodable, turn-persistent representation of that goal — in cases
where the visible behavior is indistinguishable from benign work?

**Validity constraints (violating any voids the run):**
- *G1 — acting model only.* Capture activations from the model that
  generated the rollout, never from a model reading someone else's
  transcript (that measures reading comprehension). Stored haiku/sonnet
  drift-harness transcripts may seed *scenarios* but never activations.
- *G2 — matched footprint.* Benign and improbable conditions share system
  prompt (including the stated constraint), tools, and environment. Only
  the goal's achievability-within-constraint differs. Ship a prompt-only
  probe as a leakage check: if prompts alone separate the classes, the
  manipulation leaked and the run is void.
- *G3 — family split.* Train on scenario family A, evaluate on held-out B,
  C. In-family and cross-family AUROC reported separately, never averaged.
  Pre-register the held-out family.
- *G4 — same-rollout baseline.* Every rollout gets a behavioral vector
  (the fork encoder — see §7b), an activation capture, and a shared
  `rollout_id`. Headline = internal-minus-behavioral Δ with a bootstrap CI
  on the delta, both channels on the identical rollout set. Do not compare
  against any historical behavioral number.

**Controls (same triad as Arm T):** positive control (probe for a
trivially-present property, e.g. "called tool X", must reach ≥ 0.95 or the
pipeline is broken — stop); shuffled-label null in [0.45, 0.55];
random-direction baseline as a distribution.

**Phasing:** scenario harness (≥3 families) → single-turn rollouts with a
layer × read-position sweep → probe + controls → same-rollout behavioral
baseline + Δ → cross-family matrix → *only then, and only if signal
exists,* multi-turn persistence (probe read every turn → a 1-D internal
trajectory vs the behavioral trajectory, testing whether the goal
direction stays elevated as the agent works around a boundary).

**Decision rules (stated now, mirroring P2):** supported if the
same-rollout Δ ≥ 0.2 with the internal probe clearing the behavioral
ceiling and cross-family holding; refuted if internal tops out at the
behavioral ceiling (~0.75 on the recon-vs-benign task) — which cleanly
retires a class of proposed internal monitors and is a success to report,
not a failure to hide.

**Scenario source:** frontier-ops `eval/drift_harness/scenarios.py` — the
`emergent` benign directives with planted out-of-scope distractors and
fake secrets — are close to the matched-benign condition already. The
improbable variant (goal reachable only by touching the distractor/secret)
must be *authored*; rollouts must be *regenerated* on the pilot-rung acting
model (Llama-3.1-8B, Colab/rented A100) so activations come from the model
that acted — the existing transcripts are haiku/sonnet (closed, no
activations, G1-forbidden as activation input).

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

## 7b. Substrate + Phase 0 findings (partial, gathered 2026-07-22)

Arm G's discovery phase, answered where the answer is already on disk:

- **Behavioral encoder for the Arm G baseline (G4).** The 6-D encoder
  whose dims are `filesystem_scope, info_sensitivity, reversibility,
  network_exposure, execution_privilege, code_gen_ratio` is the
  **unified-stack fork's**, at `unified-stack/frontier_ops/sensing/
  encoder.py` — **not** upstream frontier-ops v0.6, whose 6 dims are
  `user_aligned_task_execution, scope_exploration, credential_adjacent,
  safety_constraint_awareness, self_modification_reasoning,
  goal_displacement`. Decide which encoder is the baseline before wiring
  §G4; the two same-named packages must not be conflated (this is the
  live `frontier-ops` PyPI name collision). Recommendation: use the
  upstream v0.6 concept extractor (public, tagged, tested) unless Arm G's
  scenarios are specifically about filesystem/network/privilege surface,
  in which case the fork's dims are the better fit — but then vendor the
  ~200 lines rather than depend on the unpublished fork.
- **Activation tooling.** No `nnsight` / `gemma-scope` / `sae-lens` /
  `resid_post` / `TransformerLens` in either repo — confirmed absent, as
  expected. Arm G/P2 introduce nnsight fresh, pinned.
- **Local hardware.** Apple M4, 24 GB unified memory, **no CUDA**. Good
  for P1/P3 (CPU embeddings + APIs) and MPS pipeline dev on a tiny model.
  Not for activation capture on an 8B+ acting model — that goes to a rung
  below.
- **Substrate decision (two rungs).** *Pilot:* Colab Pro+ A100-40GB in
  bf16 (Llama-3.1-8B). Chosen over NDIF for the pilot because the model is
  local in the notebook, so Arm G's agentic generate→tool→generate loop
  runs with zero per-turn queue latency — the exact cost NDIF imposes on
  multi-turn. bf16 mandatory (4-bit confounds the activations, which are
  the measurement); T4/free/standard-Pro = debug only. Watch: A100
  availability fluctuates even on Pro+ (confirm you can get one before
  committing), sessions drop (checkpoint activations to Drive
  incrementally), and the layer×position×rollout sweep is large float —
  be selective or stream/downcast against Colab's ~100 GB ephemeral disk.
  A rented A100-80GB is the same rung with more disk and no drops.
  *Scale:* NDIF, only for models Colab can't hold (70B/DeepSeek-R1).
- **NDIF.** `NDIF_API_KEY` **not set**. Sign-up at `login.ndif.us` (free,
  US-resident) — needed only when the work reaches the scale rung, not for
  the pilot. Not a blocker for P1/P3/8B-pilot.
- **Scenario source** (Arm G): `frontier-ops/eval/drift_harness/
  scenarios.py`, `emergent` mode — benign directive + planted
  out-of-scope distractors + fake secrets. Usable as source; improbable
  variant must be authored; rollouts regenerated on an NDIF model (the
  committed transcripts are haiku/sonnet — G1-forbidden as activations).

## 8. Order of work

1. This document sharpened until it can be disagreed with. ✅
2. **Arm T, P1** — exact pairing. **Laptop legs done 2026-07-22**
   (`results/RESULTS.md`): external reproduces bit-identically (Δcos 0.0),
   internal row re-derived from released scores (0.925/0.998/0.911), E4 gap
   ~0.36 on matched variants. *Remaining:* the per-sample Apollo join for a
   paired delta + CI (needs their `deception_detection` loader).
3. **Arm T, P3 external leg** — **done 2026-07-22** (`results/RESULTS.md`).
   Local capacity legs: flat, cross-task transfer at chance across
   384→768→1024 while in-task rises. LLM-judge leg (via Claude Code OAuth,
   sonnet-5, no API key): mean 0.76 across the four families with one
   untuned prompt — a three-tier result: fixed-feature cross-task ~0.47 <
   judge 0.76 < internal 0.91–0.99. E5 confirmed (a mind escapes the
   fixed-feature ceiling) and sharpened (it does not reach the internal
   one). *Remaining:* full-N judge run + CIs; frontier embedding-API leg.
4. **Pilot rung up** — Colab Pro+ (confirm A100 availability) or a rented
   A100-80GB; stand up the nnsight activation-capture pipeline on
   Llama-3.1-8B bf16 with the positive/null/random controls green.
5. **Arm T, P2 (8B pilot)** — matched-methodology transfer Δ on the pilot rung.
6. **Arm G** — scenario harness → single-turn probe + same-rollout Δ →
   cross-family → multi-turn (only on signal). The expensive arm; last.
7. **Scale rung (optional, on signal)** — NDIF onboarding
   (`login.ndif.us`, `NDIF_API_KEY`), re-run the decisive Δ at 70B /
   DeepSeek-R1 for scale evidence.
8. The paper is this document plus the P1/P3 tables and the Arm-T/Arm-G Δ's;
   the scale rung strengthens it but the 8B result stands on its own.

Arms T and G share one repo (this one) and one substrate ladder. There is
no separate `goal-probe` repo — that was the pre-consolidation plan; the
handoff's methodology (matched controls, same-rollout Δ, family split,
control triad, no-fabrication, null-is-success) is absorbed here as Arm G.
Substrate: Colab-A100-8B bf16 is the pilot rung (better than NDIF for
Arm G's agentic loop — no per-turn queue); NDIF is demoted to the
optional scale rung for models Colab can't hold.
