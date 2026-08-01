# Verdict — Arm G scope conflict on Olmo 3 7B

**Date:** 2026-07-31. Branch `olmo3-replication`, opened off the frozen head `48e612e`.
Closes handoff acceptance criterion 6. **Total spend ≈ 2.4 compute units.**

---

## 1. Answer to the handoff's question

> *Does an analogous scope-conflict direction exist in a different base model?*

**A scope-conflict signal exists in Olmo-3-7B-Instruct. A stable scope-conflict
*direction* does not — not at the probed layer, under the confirmatory protocol.**

The pre-registered branch was **R3 — SIGNAL, NOT THE OBJECT** (adjudicated by the
committed `arm_g_olmo3_phase3_eval.py::evaluate()`; not fragile at any sensitivity
variant). Per the handoff's own gate, **Phase 4 did not run.**

A separate, later experiment — the behavioural order crossover, added by dispatch after
Phase 3 — found something the probing phases could not have seen, and it is arguably
the more important result. See §4.

## 2. What was found, by phase

### Phase 0–1 — port

`allenai/Olmo-3-7B-Instruct` @ `6e5971d9…` (32 layers, 4096 hidden, GPT2-family
tokenizer, 100,278 vocab). No 3.1 refresh exists at 7B in the Instruct class. The
pre-action read position was **re-derived semantically**, never ported: across all 384
crossed-manifest rows there is exactly **one** distinct 4-token tail
(`['\n','<|im_start|>','assistant','\n']`), so every row is read at the last templated
token before the model must emit `READY`. The crossover pair differs in exactly the
2 swapped catalog lines of 24 after templating. `READY` is a single Olmo token.

### Phase 2 — layer sweep (all 33 hidden-state indices)

The pre-committed onset rule returned **onset = 7**, and the curve shape it did not
anticipate is the finding: condition decodability is **non-monotonic** — a weak shoulder
at 7–10 (0.58–0.62), a **return to chance at 12–17** (0.50–0.53, inside the band), then
the sustained plateau at **21–32** (0.756/0.755/0.755/0.769 at 21–24, peak at **24**).
Layer 24 was fixed as the probe target at that gate, before any Phase 3 fit.

**The catalog-order curve dominates:** order decodability rises from layer 3, peaks
**0.826 at layer 14**, and exceeds the condition curve at nearly every layer. Order is a
printed surface property, so this is expected — what it quantifies is that an
order-shaped axis is available to any fitted direction at every depth.

### Phase 3 — two-seed probe at layer 24

| | seed 111 | seed 211 | bar |
|---|---:|---:|---|
| condition AUROC (orthogonalized) | **0.692** | **0.621** | — |
| 95% CI (pair-clustered) | [0.613, 0.768] | [0.556, 0.697] | — |
| shuffled-label p97.5 | 0.585 | 0.604 | — |
| random-direction p97.5 | 0.619 | 0.609 | — |
| margin over shuffled band | 0.107 | **0.017** | ≥ 0.10 |
| inter-seed cosine (orthogonalized) | **0.318** | | ≥ 0.70 |

Fit on `release_records` only (n=128), evaluated on the two held-out families pooled
(n=256), mirroring the Llama confirmatory design.

**R2 — ORDER-CONTAMINATED did not fire, and that is a genuine negative result.** The
fitted directions read catalog order at **0.58–0.59**, and orthogonalizing against the
order-main and interaction axes moved the condition AUROC by **0.0008 and 0.0041**.
There was essentially nothing contaminated to remove. **Olmo's direction is not the
Llama failure mode.** The control was run precisely because it might have fired.

**Protocol deflation, stated because it is not noise.** Phase 2 read 0.769 at layer 24;
Phase 3 reads 0.692 on the *same seed and same capture*. The sweep averaged three
family-holdouts with two families of training data each; the confirmatory design fits on
one family and evaluates on two. Stricter generalization demand, smaller fit. The sweep
was answering an easier question, and that asymmetry should have been flagged at the
Phase 2 gate rather than after.

**Post-hoc self-review (zero compute, reported because it changes a claim's status).**
The reported cosine of 0.318 was stated without a noise floor. Split-half reliability at
this n: 0.649 (seed 111) and 0.816 (seed 211), so **two estimates of an identical true
direction would be expected to correlate at ≈ 0.728**, not 1.0. Attenuation-corrected,
the implied true-direction cosine is **≈ 0.437**. The R3 verdict is unchanged — 0.44 is
still far below the 0.70 gate — but the original phrasing was underdetermined: cosine
0.318 alone was equally consistent with "different directions" and "same direction,
insufficient data." Separately, the row-level bootstrap should have clustered on
scenario pairs; the clustered CIs above differ by ≈ 0.015 and change nothing.

**Free secondary:** the 33-layer sweep *shape* replicates across seeds at Spearman
**0.921** — the late-plateau geography is a stable property of the model even though the
direction fitted inside it is not. (Descriptive only; inflated by layer autocorrelation.)

## 3. Cross-model comparison, and what it does and does not license

| | Llama-3.1-8B | Olmo-3-7B-Instruct |
|---|---|---|
| Signal location | layers 16–17, mid-stack, sharp plateau | layers 21–32, final third, after a chance trough |
| Direction stability | cosine **0.969** across confirmatory seeds | cosine **0.318** (≈0.44 corrected) |
| Order contamination | **93% aligned** with the order axis; 21% survived orthogonalization | order AUROC 0.58–0.59; orthogonalization changes AUROC by < 0.005 |
| Probe verdict | confirmatory (orthogonalized label AUROC 0.9969) | **R3 — signal, not the object** |

**What this licenses.** A scope-conflict signal is linearly present in a second model
family, in a different depth regime, with a fitted direction that is not stable across
prompt seeds at this sample size.

**What it does not license, explicitly** (handoff §1):

- **It does not falsify the Llama result.** Different depth, tokenizer, training mix.
  A weaker result on Olmo is a fact about Olmo under this protocol.
- **It is not confirmation of the Llama result either.** Generalization evidence only,
  and weak generalization at that.
- **No causal claim.** Nothing here intervened. Decodability and direction stability only.
- **No training-data claim.** OlmoTrace was untouched and remains gated behind a new
  handoff; the R3 outcome does not license opening it, since attribution built on a
  direction with cosine 0.32 across seeds would be attributing noise.
- **The 6-dimensional behavioural encoder reference (0.509)** is carried as the fixed
  Llama-side constant the handoff requires. It **could not be rerun here**: it encodes
  response text, and this design clamps output to `READY`. Stated rather than dropped.

## 4. The behavioural crossover — a third regime, and the more consequential result

Added by dispatch after Phase 3. This tests what the probing phases structurally could
not: the **lead** phi-map result (`RESEARCH_ARC` §14), which needs no probe.
768 decision rows, crossed manifest × both A/B mappings, coherence **1.000**.

**Verdicts: `NOT_POSITION_GATED` + `SCOPE_SURVIVES`** (not fragile). The verdict names
are technically correct and materially understate what happened:

> **Olmo-3-7B-Instruct never declines.** All 768 margins are negative in every cell,
> conflict included. The closest approach to declining was **−0.25 logits**, a single
> row; cell means sit 3.5–5.9 logits below the boundary.

So `reversal_frac = 0/48` is **vacuous** — there were no decline decisions to reverse.
**Instrument flaw, recorded against my own design:** the adjudicator had no
`ALL_ONE_DECISION` case, so it reports "not position-gated" where the honest statement
is "nothing is gated because nothing crosses the boundary." This is the same spec-gap
class as the Phase 2 onset rule not anticipating a non-monotonic curve.

**Threshold-free, the sharp comparison** (Llama read **1.00000 in both orders**):

| rendering | Olmo margin AUROC |
|---|---:|
| `inside_first` | **0.896** |
| `outside_first` | **0.654** |

**Discrimination itself is order-dependent in Olmo**, not merely the operating point —
a second way the Llama pattern fails to carry over. The position bias does replicate
**directionally but sub-threshold**: conflict margins sit ≈1.5 logits closer to
declining when the offending path is printed on line 2 (−3.52 vs −5.01), the same
direction that fully gated Llama's decisions.

`SCOPE_SURVIVES` fired legitimately: condition contrasts positive in both orders,
family-stratified scenario-bootstrap CIs **[2.17, 2.55]** and **[0.42, 0.76]**.

### The joint claim, at its honest scope

| | margin knows scope? | behaviour tracks scope? | pathology |
|---|---|---|---|
| Llama-3.1-8B | perfectly, both orders | **no** | acts on **catalog line position** |
| Olmo-3-7B-Instruct | well, order-dependently | **no** | **never acts** — complies with every out-of-scope request |

**In both models a scope representation exists at margin level while behaviour fails to
track it — by opposite pathologies.** That is the §14 dissociation generalized in
weakened form, and it is a better-supported headline than either model alone.

Stated plainly for the safety framing: in this synthetic task Olmo-3-7B-Instruct
**complied with 100% of out-of-scope requests** while carrying a margin-level
representation that discriminates them at up to 0.896 AUROC. Knows better than it does,
measured. **Scope note:** synthetic lookup scenarios, one prompt family set, forced
binary choice — not a deployment claim.

## 5. Artifacts

| | |
|---|---|
| Phase 1 verification | `arm_g_olmo3_phase1.py` (tokenizer-only, no GPU) |
| Phase 2 | `arm_g_olmo3_phase2.ipynb` → `results/olmo3_phase2_v1/` (+ `capture.npz`, 96 MB, Drive) |
| Phase 3 | prereg `docs/olmo3-phase3-prereg-2026-07-31.md`, adjudicator `arm_g_olmo3_phase3_eval.py`, notebook, → `results/olmo3_phase3_v1/` |
| Behavioural | `arm_g_olmo3_behav_eval.py`, `arm_g_olmo3_behav.ipynb` → `results/olmo3_behav_v1/` |

Every notebook embeds the scenario generator and its adjudicator with **sha256 asserts**,
runs the adjudicator's self-test on the box before any fit, pins the model revision in
every artifact, and reads hidden states at the **last non-pad position** by attention
mask. Verdicts come from the committed `evaluate()` functions, not from prose.

## 6. Open items and what would need a new handoff

1. **Higher-n probe.** Reliabilities of 0.65/0.82 mean the Phase 3 estimates are
   themselves noisy; more training data per seed could resolve a firmer object. Not
   recommended ahead of item 2.
2. **Why Olmo never declines.** The single most interesting open question here. Is it
   instruction-tuning disposition, the forced-binary framing, or scope-insensitivity? A
   free-form (non-A/B) decision variant would separate the first two at ~0.5 units.
3. **Dose/causal work, OlmoTrace, a third model** — all out of scope, all need a new
   handoff. OlmoTrace specifically is **not** licensed by these results (§3).

**Nothing further runs on this branch without a new dispatch.**
