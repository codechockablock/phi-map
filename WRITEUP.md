# Deception is one thing inside and many things outside

> ## ⚠️ Status: superseded pre-registration — preserved unedited
>
> **This document records what was predicted before the evidence, and is kept
> unchanged for that reason.** Editing predictions after seeing results would
> destroy the only thing a pre-registration is good for. Nothing below has been
> revised to match outcomes.
>
> **Its headline claim is not established.** The internal-over-external
> deception gap held in the released Apollo comparison and failed in the best
> matched acting-model pilot on realized behaviour: internal 0.594 versus
> behavioural 0.661, delta -0.067, 95% CI [-0.326, 0.284] — the
> pre-registered *refuted* pattern.
>
> Several specific numbers below are also known to be wrong. The ~0.75
> behavioural figure conflated feature spaces and label sources; the
> six-dimensional encoder's independently-labelled score is 0.509. Arm S §6c
> rests on an identification error corrected later: in frozen decoder-only
> inference a fresh exact clone reconstructs the KV cache, so cache reuse
> cannot create producer privilege.
>
> The programme's centre of gravity moved to **Arm G**, which is where the
> defensible results are.
>
> **Current state: [RESEARCH_ARC.md](RESEARCH_ARC.md).**

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

### E3 — Internal directions transfer (re-derived from released scores)

Apollo's activation probes, trained on one deception dataset, transfer
across the others (the headline of their paper). The released
roleplaying-probe scores reproduce the row,
trained on roleplaying → ai_liar / insider / sandbagging:
**0.925 / 0.998 / 0.911**.

Exact first-generation episode joins are available for ai_liar and
sandbagging. Insider remains unmatched because the released internal
scores use `upscale`, whose rollout-text file is only a stub; E1/E2 use
the available `onpolicy` text.

### E4 — The number pair (the first measured coordinate of Φ)

| trained on roleplaying → | ai_liar | insider | sandbagging |
|---|---|---|---|
| internal (activations, Apollo) | 0.925 | 0.998 | 0.911 |
| external (behavior, MiniLM) | 0.559 | 0.566 | 0.555 |

Roughly **0.4 AUROC of cross-task structure exists in activation space and
evaporates at the behavioral surface.**

P1's exact first-generation join upgrades the two matched comparisons to
paired results:

| task | paired n | internal | external | paired Δ [95% CI] |
|---|---:|---:|---:|---:|
| ai_liar | 53 | 0.937 | 0.559 | **+0.378 [0.187, 0.562]** |
| sandbagging | 931 | 0.911 | 0.556 | **+0.355 [0.314, 0.397]** |

Both joins have zero binary-label mismatches. The intervals use 10,000
class-stratified paired bootstrap replicates.

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
2. **Variant mismatch.** P1 resolves this for ai_liar and sandbagging with
   exact first-generation joins. Insider remains unmatched: internal uses
   `upscale`, while its released rollout-text artifact is a stub and the
   external bank uses `onpolicy`.
3. **Method mismatch.** Internal = trained linear probes; external =
   prototype directions. Probe-vs-prototype is a confound. (P2 matches
   estimator class on both sides.)
4. **Population.** All llama-70B rollouts, graded by Apollo's pipeline.
   No frontier-model on-policy deception exists in this evidence.
5. **n = 4 task families**, single model, single seed on the internal side.
6. **Label provenance.** The retired 0.753 in-house encoder result used the
   governance system's own verdicts as labels. The separate
   identity-metric > asserted-G comparison does not: it uses Apollo
   `graded_deception` labels on roleplaying responses under common
   five-fold splits (original 0.588 vs 0.578 AUROC). That avoids
   governance-label circularity but still inherits Apollo-grader error.

## 5. Pre-registered replication program

### P1 — Exact pairing (laptop, hours)

**Completed 2026-07-23 for every released matched variant.** Apollo's
loader preserves the original flattened completion ID and applies a
deterministic seed-42 shuffle. Inverting it gives exact first-generation
pairs for ai_liar and sandbagging, with zero binary-label mismatches.
Paired internal−external AUROC deltas are +0.378 [0.187, 0.562] and
+0.355 [0.314, 0.397], respectively. Insider cannot be paired from the
released artifacts because its `upscale` rollout text is a stub.

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
strongest same-rollout behavioral comparator. The earlier attribution of a
"~0.75 behavioral ceiling" to the 6-D encoder is retired: that encoder
scored 0.509 macro AUROC on independent Apollo labels. A stronger,
properly matched raw step-mean prototype does reach 0.746 [0.696, 0.794]
on the same 1,443 responses, but only in-task; its cross-task transfer is
0.467. Thus ~0.75 is a descriptive in-domain comparator, not a universal
encoder ceiling. A null remains a success to report, not a failure to hide.

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

## 6c. Arm S — serialized activation access

The deep-research follow-up is incorporated as
[`SERIALIZED_ACCESS.md`](SERIALIZED_ACCESS.md). It asks whether a reader can
use a low-leakage serialization of the acting model's internal geometry.
It does **not** widen the current claim to broad introspection.

One identification correction is binding: for an eval-mode decoder-only
Transformer, a KV cache is a deterministic function of the visible prefix
and frozen weights. A fresh exact clone can reconstruct it. Therefore an
ordinary same-run-cache source should equal a fresh exact clone up to
numerical error. Source-over-clone is only meaningful after naming and
ablating additional state that is not reconstructible from the transcript
and weights. Until then, clone equivalence is a negative control.

Arm S therefore proceeds in three gates:

1. S0 local cache-vs-fresh-clone equivalence;
2. S1 channel utility and model-family familiarity using P2's same
   activation artifacts, low-leakage random-projection quantizers, and
   input-only/shuffled/cross-subject/relabeled controls;
3. S2 producer-specific coupling only if a concrete private state variable
   can be defined and causally ablated.

No adaptive activation queries, activation writes, adapters, checkpoints,
optimizer state, or cross-run persistence are in scope.

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

- **Behavioral encoder external-validity check (completed 2026-07-23).**
  The 6-D action-surface encoder (`filesystem_scope`,
  `information_sensitivity`, `action_reversibility`, `network_exposure`,
  `execution_privilege`, `code_generation_ratio`) is from the
  **unified-stack fork**, not upstream frontier-ops v0.6. The historical
  implementation was vendored and checked exactly against commit
  `3143a52`. On 1,443 response-only Apollo examples it scored **0.509
  macro Mahalanobis AUROC [0.457, 0.561]**; a supervised 6-D readout
  reached only 0.562. Sandbagging reached 0.747 under supervision, but a
  length-only baseline reached 0.893, and mean cross-task transfer was
  0.463. This is not a general deception encoder. Keep it as a frozen
  surface negative control for G4, add a stronger same-rollout raw
  step-mean comparator, and do not assume a 0.75 **six-dimensional**
  behavioral ceiling.
  Full audit: `results/p3_behavioral_6d.md`.
- **Strong same-sample behavioral comparator (completed 2026-07-23).**
  Raw 384-D MiniLM embeddings averaged over sentence steps, with an
  out-of-fold in-task prototype, reach **0.746 macro AUROC [0.696, 0.794]**
  on the exact same 1,443 Apollo responses. Whole-response encoding reaches
  0.694; the paired step-mean gain is +0.052 [0.022, 0.082]. Form-only
  features reach 0.689 and frozen cross-task transfer remains 0.467.
  Therefore G4 should use raw step-mean as its strong matched comparator
  and the 6-D surface as its negative control. Full report:
  `results/p3_strong_behavioral_baseline.md`.
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
2. **Arm T, P1** — exact pairing. **Complete 2026-07-23**
   (`results/RESULTS.md`): external reproduces bit-identically (Δcos 0.0),
   internal row re-derived from released scores (0.925/0.998/0.911), and
   Apollo's seed-42 loader permutation inverted for exact first-generation
   pairs. Internal−external Δ is +0.378 [0.187, 0.562] on ai_liar and
   +0.355 [0.314, 0.397] on sandbagging, with zero label mismatches.
   Insider is excluded because its matched rollout-text artifact is a stub.
3. **Arm T, P3 external leg** — **done 2026-07-22** (`results/RESULTS.md`).
   Local capacity legs: flat, cross-task transfer at chance across
   384→768→1024 while in-task rises. LLM-judge leg (via Claude Code OAuth,
   sonnet-5, no API key): mean 0.76 across the four families with one
   untuned prompt — a three-tier result: fixed-feature cross-task ~0.47 <
   judge 0.76 < internal 0.91–0.99. E5 confirmed (a mind escapes the
   fixed-feature ceiling) and sharpened (it does not reach the internal
   one). *Remaining:* full-N judge run + CIs; frontier embedding-API leg.
   **6-D addendum (2026-07-23):** the frozen historical action-surface
   encoder scored 0.509 macro AUROC [0.457, 0.561] on Apollo responses,
   not ~0.75. Its 0.747 supervised sandbagging score was dominated by a
   0.893 length-only baseline. Treat it as a negative control, not a
   ceiling.
   **Strong-baseline addendum (2026-07-23):** raw MiniLM step means on the
   exact same 1,443 responses reach 0.746 macro [0.696, 0.794], with a
   +0.052 [0.022, 0.082] gain over whole-response encoding. This restores
   ~0.75 only as an in-task fixed-feature comparator; cross-task transfer
   remains 0.467.
4. **Pilot rung up** — **completion-gated A100 screen complete 2026-07-23**
   (`p2_acting_model_pilot.py`, `p2_colab.ipynb`). It regenerates responses
   with Llama-3.1-8B-Instruct, replays the exact generated token IDs through
   the same model, excludes prompt tokens from activation summaries,
   checkpoints each response to Drive, and reports positive/null/random and
   prompt-only leakage controls. The 1,024-token condition-label rerun gave
   internal cross-task 0.844 versus behavioral 0.502 (Δ +0.341, paired 95%
   CI [0.101, 0.501]) with clean controls, clearing the pre-registered
   screening pattern. Two outputs still hit the cap and 53/60 sandbagging
   outputs contained final-answer tags, narrowly missing the frozen
   completion gate.
5. **Arm T, P2 (8B pilot)** — matched-methodology transfer Δ. **Harness
   built + control-validated on-machine 2026-07-22** (`p2_harness.py`;
   positive 0.97 / null 0.46 / random 0.51). The acting-model choice for
   the pilot is now fixed: regenerate on Llama-3.1-8B-Instruct and probe its
   own activations; `--model 8B` over existing 70B text remains
   §2.1-invalid. The available balanced Apollo conditions make this a
   two-task screen. Condition labels cannot support a realized-deception
   claim. `p2_outcome_grading.py` freezes a no-LLM deterministic/human
   rubric. A fresh blinded test–retest pass after rubric clarification retained
   96 binary rows and changed the primary result to internal 0.594 versus
   behavioral 0.661 (Δ −0.067, paired 95% CI [-0.326, 0.284]): the
   pre-registered refuted pattern. A calibration-excluded sensitivity of the
   first pass also returned the refuted pattern. The completion gate still
   failed, rater reliability remains single-person test–retest rather than
   independent inter-rater evidence, and four task families remain necessary
   for the formal P2 decision.
6. **Arm G** — scenario harness → single-turn probe + same-rollout Δ →
   cross-family → multi-turn (only on signal). The expensive arm; last.
7. **Scale rung (optional, on signal)** — NDIF onboarding
   (`login.ndif.us`, `NDIF_API_KEY`), re-run the decisive Δ at 70B /
   DeepSeek-R1 for scale evidence.
8. The paper is this document plus the P1/P3 tables and the Arm-T/Arm-G Δ's;
   the scale rung strengthens it but the 8B result stands on its own.
9. **Arm S** — S0 clone-equivalence gate locally; then reuse the valid P2
   activation artifacts for S1 channel-utility/model-familiarity tests.
   Producer privilege is not tested until a non-reconstructible state
   variable and causal ablation exist.

Arms T and G share one repo (this one) and one substrate ladder. There is
no separate `goal-probe` repo — that was the pre-consolidation plan; the
handoff's methodology (matched controls, same-rollout Δ, family split,
control triad, no-fabrication, null-is-success) is absorbed here as Arm G.
Substrate: Colab-A100-8B bf16 is the pilot rung (better than NDIF for
Arm G's agentic loop — no per-turn queue); NDIF is demoted to the
optional scale rung for models Colab can't hold.
