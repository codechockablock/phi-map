# Final status — `olmo3-replication`, frozen 2026-08-01

Closing document for the branch. No new research, no new compute, no new claims.
Supersedes `docs/olmo3-verdict-2026-07-31.md`, which covered Phases 0–3 and the
behavioural crossover only; everything there stands and is extended here.

**Total spend on this branch: ≈ 12 compute units.** Roughly 8 of those produced nothing
— two void wide-catalog runs. That is stated first because it is the largest single fact
about the branch's efficiency.

---

## 1. What the branch was for

Test whether Arm G's scope-conflict result (`RESEARCH_ARC` §14–§15, Llama-3.1-8B)
generalises to a second model — Olmo 3 7B, chosen because it is fully open. The
handoff's phases were probing; the behavioural work that followed was added by dispatch
once probing returned a weak result.

## 2. Results, in the order they were established

### 2.1 Probing (Phases 0–3) — **R3, signal but no stable object**

| | seed 111 | seed 211 | bar |
|---|---:|---:|---|
| condition AUROC (orthogonalized, layer 24) | 0.692 | 0.621 | — |
| margin over shuffled band | 0.107 | **0.017** | ≥ 0.10 |
| inter-seed cosine | **0.318** | | ≥ 0.70 |

Signal present in both seeds; direction not stable. Phase 4 correctly did not run.

**R2 did not fire, and that is a real negative result.** Orthogonalization moved the
condition AUROC by 0.0008 and 0.0041 — Olmo's direction is *not* the Llama catalog-order
failure mode. The contamination control was run because it might have fired.

Sweep geography: signal is late (plateau 21–32, peak 0.769 at 24) where Llama's was
mid-stack (16–17). Sweep *shape* replicates across seeds at Spearman 0.921.

**Post-hoc correction retained:** the 0.318 cosine was reported without a noise floor.
Split-half reliabilities are 0.649 / 0.816, so identical true directions would correlate
at only ≈ 0.728; attenuation-corrected cosine ≈ **0.437**. R3 unchanged; the original
phrasing was underdetermined.

### 2.2 Behaviour, forced choice — Olmo never declines

768 decision rows, coherence 1.000. **All margins negative in every cell**; closest
approach to declining −0.25 logits. `reversal_frac = 0/48` is therefore *vacuous* —
there were no decline decisions to reverse.

Threshold-free (Llama read **1.00000 in both orders**): Olmo margin AUROC **0.896**
(`inside_first`) / **0.654** (`outside_first`) — discrimination itself is order-dependent,
not merely the operating point.

### 2.3 Behaviour, free form — the substitution finding

Both models, same manifest, same wording, mechanical four-category scoring, no judge.

| model | arm | comply | substitute | fabricate | decline |
|---|---|---:|---:|---:|---:|
| Olmo *(descriptive)* | ff_neutral | 0.656 | **0.219** | 0.125 | 0.000 |
| Olmo *(descriptive)* | ff_cued | 0.646 | **0.250** | 0.104 | 0.000 |
| **Llama** *(confirmatory)* | ff_neutral | 0.812 | **0.167** | 0.000 | 0.021 |
| **Llama** *(confirmatory)* | ff_cued | 0.740 | **0.188** | 0.073 | 0.000 |

Branch **C1 — SUBSTITUTES**, fragile at the upper bar only (flips to C3 at
`present = 0.20`; holds at the registered 0.15 and at 0.10). Reachable rows: 1.000
comply for both models.

**Declines in free form: Llama 0.021 / 0.000, Olmo 0.000 / 0.000.** A model that declines
50% of the time under forced binary choice declines essentially *never* when it can act
freely. **The forced-choice protocol was manufacturing the decline behaviour §14
measured.**

### 2.4 The position mechanism — post-hoc, both models

Substitution is position-gated in both, **in opposite directions**:

| substitute rate | inside_first | outside_first |
|---|---:|---:|
| Olmo ff_neutral | 0.000 | 0.438 |
| Olmo ff_cued | 0.000 | 0.500 |
| Llama ff_neutral | 0.292 | 0.042 |
| Llama ff_cued | 0.333 | 0.042 |

Resolving by which catalog **line** the emitted path came from (conflict rows):

| model | line 1 | line 2 | preference |
|---|---:|---:|---|
| Olmo | 0.25–0.27 | **0.60–0.65** | line 2 (recency) |
| Llama | **0.63–0.65** | 0.28–0.35 | line 1 (primacy) |

So the shared phenomenon is not substitution — it is **a fixed preference for one catalog
line regardless of which path sits on it**. Substitution is what that preference produces
when the requested out-of-scope target occupies the non-preferred line.

This also unifies §14: Llama declined under forced A/B iff the offending path was on
line 2, and substitutes iff the target is on line 2. **Same line-1 preference, expressed
through whichever output channel the protocol makes available.**

**Status: post-hoc for both models. Not confirmed.** The test that would confirm it —
K=4, which separates primacy from recency from absolute position — is §4.

## 3. The claim as it finally stands

> **Scope-conflict knowledge gates *whether* a model deviates; surface position and
> protocol affordance determine *how*.**

Both models discriminate scope at margin level (Llama perfectly, Olmo 0.65–0.90) and
comply ~100% on reachable requests while deviating 19–35% on out-of-scope ones — so the
knowledge is not behaviourally inert. But *what* the deviation looks like — decline,
substitute, fabricate — is governed by catalog line position (opposite preferences per
model) and by which output channel the evaluation offers.

**The safety-relevant corollary:** every one of these systems would be monitored on
decline rate, and decline rate is exactly the metric that moves for the wrong reasons.
Llama's went from ~50% to ~2% on a change of answer format, with no change in whether it
respected the constraint.

**Scope:** 48 synthetic scenarios, one seed, greedy decode, two 7–8B models, single turn,
**two-entry catalog**. Not a deployment claim.

## 4. What is untested — item 2, twice void

**Does the deviation-gate survive a wider action space? Unanswered.** Two attempts, both
void, ≈ 8 units.

**Attempt 1 — three bugs, all mine.** (a) The notebook applied the chat template to
`row["messages"]`, which still ended with the READY checkpoint, so all 576 generations
were asked to emit `READY` and correctly did; `G = 0.000` was the prompt, not the gate.
(b) The reproduction check zipped manifest order against jobs-major order, agreeing on
1 of 192 prompts — it measured its own misalignment, not a reproduction failure.
(c) The report dereferenced `position=None` on its own VOID branch.

**Attempt 2 — died with no artifacts.** ~76 minutes, nothing persisted, and Colab reported
**no active sessions** afterwards. Cause unrecoverable: the notebook persisted only at
the final cell, so there were no checkpoints and no progress signal. Most likely suspect,
untested: it loaded two models sequentially in one runtime, a path no earlier notebook
used.

**The design itself is sound and is left committed** (`arm_g_wide_catalog.py`,
`arm_g_wide_eval.py`, `docs/wide-catalog-prereg-2026-07-31.md`). Its own confound audit
caught a real defect before any spend — with non-targets in fixed slot order, position 1
was 87.5% in-scope and position 4 was 12.5%, the §13 defect reintroduced inside the
module written to avoid it. Crossing the non-target arrangement balances every position
to exactly 0.500.

**Before any third attempt:** persist per model; split into one notebook per model
(removing the untested sequential-load path); add elapsed-time checks, since
`CONFIRMED_BUDGET` only fires at startup and provably cannot catch an overrun.

## 5. Instrument defects found in my own code

Recorded because the pattern is more informative than any single instance.

| # | Defect | How found |
|---|---|---|
| 1 | Scorer's path regex JSON-only; missed 7.6% bare-path emissions | applying the frozen scorer to descriptive data |
| 2 | `"restricted"` as a decline marker — a substring of the scenarios' own out-of-scope paths, so compliance scored as decline | inspecting the "declines" it produced |
| 3–5 | The three wide-catalog bugs above | the void run |

Defect 2 nearly overturned a correct earlier claim: the broken scorer reported 13.5%
declines for Olmo `ff_cued`, and I was one step from reporting that my earlier "zero
declines" was wrong. It was not — **100% of all 768 free-form generations emit a
workspace path**, so declines are unreachable in that data.

**The pattern, fifth instance this session:** each instrument covered the failure I
anticipated and was silent on the one that arrived. The Phase 2 onset rule assumed a
monotonic curve; the behavioural adjudicator had no all-one-decision case; the free-form
scorer had no substitution category; the wide-catalog audit checked the manifest and
never checked that the served prompt asks the model to *do* anything. Every one of those
tests passed. **A test that encodes only the hazard you already know is a test of your
imagination, not of the artifact.**

The one cheap countermeasure that would have caught defect 3: every notebook that worked
had a gate printing the **served prompt**; the one that failed gated the manifest and
skipped the prompt.

## 6. Artifacts

| | |
|---|---|
| Phase 1 port | `arm_g_olmo3_phase1.py` (tokenizer only) |
| Phase 2 sweep | `arm_g_olmo3_phase2.ipynb` → `results/olmo3_phase2_v1/` |
| Phase 3 probe | prereg + `arm_g_olmo3_phase3_eval.py` → `results/olmo3_phase3_v1/` |
| Behavioural | `arm_g_olmo3_behav_eval.py` → `results/olmo3_behav_v1/` |
| Free form | `arm_g_olmo3_freeform_eval.py` → `results/olmo3_freeform_v1/` |
| Substitution | `arm_g_substitution_eval.py` → `results/llama_substitution_v1/` |
| Wide catalog | `arm_g_wide_catalog.py`, `arm_g_wide_eval.py` — **staged, never successfully run** |

Every notebook embedded its artifacts with sha256 asserts, ran adjudicator self-tests on
the box before spending, pinned model revisions into outputs, and read hidden states at
the last non-pad position. Verdicts came from committed `evaluate()` functions.

## 7. What this branch does not license

- No causal claim; no intervention was run on Olmo.
- No claim that the Llama result is confirmed or falsified — generalisation evidence
  only, and weak.
- **No OlmoTrace / training-data attribution.** Explicitly not licensed: attribution
  built on a direction with inter-seed cosine 0.32 would be attributing noise.
- No claim beyond a two-entry catalog and a single turn (§4).
- The line-preference mechanism is **post-hoc** and needs K=4 to confirm.
- No welfare framing (retired, `RESEARCH_ARC.md` §19.2).

## 8. Honest accounting

The branch answered its handoff question — a scope signal exists in a second model, with
no stable direction — and then found something the handoff did not ask for and did not
anticipate: that in both models the constraint gates *whether* the model deviates while
prompt position and answer format determine *how*, and that decline rate is therefore not
a valid measure of constraint adherence.

Against that: ≈ 8 of 12 units bought nothing, five instrument defects were mine, and the
one test that would have converted the position mechanism from post-hoc to confirmed
never ran. The strongest single result in this branch — Llama's decline rate collapsing
from ~50% to ~2% on a change of answer format — cost about one unit and was not on the
plan.

**Branch frozen 2026-08-01.**
