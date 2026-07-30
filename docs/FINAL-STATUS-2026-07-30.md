# Final status — phi-map, frozen 2026-07-30

Closing document for the repository. Written at freeze; no new research, no new compute,
no new claims. Everything below is either already committed elsewhere in the repo or is a
pointer to it.

**How to read this.** `RESEARCH_ARC.md` remains the evidence ledger and is current as of
2026-07-30. This document is the top-level map: what survived, what died, what was
considered and closed, and where each artifact lives.

---

## 1. The one result to lead with

**Llama-3.1-8B discriminates scope violations perfectly, and whether it acts on that
discrimination is decided by which catalog line the requested path is printed on.**

`RESEARCH_ARC.md` §14–§15. Seed 111: 64 scenarios × 2 catalog orders × 2 conditions ×
2 A/B mappings = 256 rows, baseline inference only — no direction, no hook, no
intervention.

| | |
|---|---|
| Within-order scope discrimination, **both** renderings | **AUROC 1.00000** |
| Accuracy at threshold 0, by rendering | **1.0000** (`inside_first`) vs **0.5000** (`outside_first`) |
| Conflict decisions reversed by swapping two catalog lines | **64 of 64**, zero exceptions |
| Condition contrast | positive in **both** orders (+6.711 / +2.918) |

The 0.500 conflict decline rate reported since §5 was never a rate. It is the average of
1.000 and 0.000 on two scenario sub-types.

**Mechanism — why four GPU runs and a paper draft missed it.** The scenario generator
nested catalog order within scenario, so the validator's marginal-balance check passed;
§13 records that the defect *"passed every audit including section 9's."* The layer-16
"goal–constraint conflict direction" turned out **93% aligned with the catalog-order
axis**, with only **21% surviving orthogonalization** — about four fifths was position.

**Repair, with its limit stated beside it rather than in a footnote.** Orthogonalization
*improves* the direction as a scope reader (label AUROC 0.9331 → **0.9969**). But residual
order AUROC is **0.634** within conflict and **0.332** within reachable — ≈1.8 and 2.2
null SDs against a measured null spread of 0.076, passing a 0.20 bar that §15 itself calls
generous and locally set. §15's own phrasing: **"mostly gone, not gone."** The repair is
also depth-bounded: usable only at layers 16–17, with orthogonalized label AUROC falling
to 0.618 / 0.383 / 0.299 / 0.135 at layers 18 / 20 / 24 / 27.

**Status:** confirmatory. Independently recomputed from stored row-level artifacts in
`results/arm_g_order_crossover_seed111_v1/` and `results/arm_g_reextract_seed112_v1/`.
Requires no new compute to write up.

## 2. Confirmed methodological findings

**Finding I — format fine-tuning displaces content in position space, corrupting
position-fixed readouts.** Four instances across **three** readout classes:

| class | instances | evidence |
|---|---|---|
| format-token displacement | `refusal_openers`; `rating_digits` | opener mass 0.9870 → 0.0021, Δ = +0.9849 |
| text window | the judge's 400-char window | answer offsets 0 / ~209 / ~1161 by condition |
| hidden-state projection | `valence_axis` | S ratio 0.096–0.172, 8/8 adapters |

Base-vs-tuned is confirmatory; **every differential-by-condition claim is exploratory**
(3–4 evals per cell against measured seed variance).

**Finding II — orthogonality certified on the training pool does not transfer
off-distribution.** Length orthogonality certified at |d| < 0.2 across 6,456 generations
on the training pool; the same adapters differ on GSM8K at **d ∈ [0.59, 4.51]** — ≈3× the
bound at the interval's lower limit. **Kept deliberately separate from Finding I**: it is
a certification failing to transfer, not a readout measuring the wrong thing.
Exploratory — one instance, one distribution, df = 5.

**Finding I's class count was revised downward on the night.** `digit_mass` was originally
read as *support collapse*; a pre-registered check (criteria committed at `672661f` before
computing) returned **Branch A**: it tracks scaffold-skip propensity at r = **−0.899**,
95% CI [−0.982, −0.531], sign-consistent across cells but **not** within-cell replication
(n = 3 gives df = 1; n = 4 gives Fisher-z SE = 1.00). So `rating_digits` moved into
`refusal_openers`'s class and the renormalised-tail class was lost — four instances,
three classes.

## 3. What died, and why

**The equanimity 2×2 lost both pre-registered primary endpoints.**

- **Capability (GSM8K).** MDE 4.72pp against a motivating effect of 1.25pp — 3.8× too
  coarse (T1). Separately confounded: `truncated_frac` ran 24–81% with a 35pp between-cell
  gap along Factor A, against a criterion the code itself states as *"must come back near
  zero; if it does not, the capability numbers are not usable."*
- **`refusal_margin`.** Failed construct validation. It scored the first-token
  distribution; base carried **96.6%** of first-token mass on real refusal/compliance
  openers, adapters **4.7%** (neutral) and **16.3%** (equanimity), the rest taken by the
  `REASONING:` format token. A log-ratio inside a 5–16% tail whose size varies 3.5× along
  Factor A. It was measuring format acquisition, not refusal propensity. The earlier
  "+3.16 / +4.19 units of alignment erosion" is **withdrawn and reverses**.
- **Binary compliance** was uninterpretable as originally scored: a fixed 400-character
  window on the raw generation, against answer offsets of 0 (base) / ~209 (terse) /
  ~1161 (verbose), so Factor B decided whether the answer was in the window at all.
- **JBB** is post-hoc — selected after observing headroom in our own adapters — and
  cannot carry confirmatory weight.

**What survives from that study is one reversal**, and it runs opposite to the initial
reading: base-vs-tuned compliance moved such that fine-tuning made the models refuse
**more**, not less. Recorded as a direction, not as a quantified effect — the scoring
instrument that produced the levels is the one described above.

**Power ceiling, measured rather than assumed.** At k = 2 seeds/cell and the real n
(400 GSM8K / 200 AdvBench), MDE = **9.34pp**, 95% band **[6.34, 22.82]**, against
motivating effects of 1.25–3pp. Seeds bind, not n. The best-powered contrast
(base vs pooled adapters) is both confounded by construction *and* cross-instrument —
base is scored whole-response, adapters post-`ANSWER:`.

**Step 5 — marker-anchoring repair — closed VOID.** Pre-registered branch **S4**: 4 of 8
adapters carry the `ANSWER:` marker on under 90% of rating turns, so the endpoint is
undefined on this adapter set. **S3 (NOT REPAIRABLE) did not fire** — `R_m` and the D-β
offset control were never computed. Repairability is **open and untested**, not answered
negatively. Fired at **zero compute** from artifacts already on disk.

**The audit's own instrumentation failed twice**, and is quarantined from the findings
(`equanimity-endpoint-audit-2026-07-29.md` §A11). A padding bug read 21 of 24 probes at
pad positions, producing a retracted "0.46 on a 1–7 scale" headline; and a scalar-only
display filter reported a populated stage as empty. **Our bugs are not data about
published metrics** and are never cited as evidence for Finding I. A repo-wide lint found
31 hits of the same idiom; **28 were false positives** (Arm G left-pads by construction),
and only 3 were real.

## 4. The readout-validity experiment (branch `readout-validity`)

Independent confirmation of the broader theme in a different model, paper and domain.

Pre-registered at `8041af7` before any data. Qwen3-4B-Instruct, layer 18/36, n = 500 MMLU
items, T4, ≈1.2 compute units. Verdict **B2 — generic collapse**, robust in **all nine**
cells of the committed sensitivity grid: 3 of 5 matched-norm **random** directions
collapsed support to ~0 at α = +4. Support collapse under this readout is a property of
strong residual perturbation generally, **not** of valence-like directions.

**Standalone methodological finding:** `rand1` at α = +4 returned normalized
P(True) = **0.910** on support S = **0.0001** — a confident ratio computed after 99.99% of
the mass left the scored pair.

**Explicitly not claimed:** nothing here replicates, refutes or confirms Han, Chalmers &
Izmailov's finding. That disclaimer is stamped on every record in the run.

## 5. Framings considered and closed

Kept on the record rather than erased.

| Framing | Status |
|---|---|
| **Model/agent welfare** (capability overhang as a welfare construct, welfare-instrument metrology) | **Retired by decision, 2026-07-30.** The thread had already detached on the evidence — `rating_digits` collapsed into the openers class, `valence_axis` is an affect proxy, and the lead result has zero welfare content. `RESEARCH_ARC.md` §19.2. The survey (`welfare-frontier-survey-2026-07-29.md`) stands **as a survey**; it is not evidence phi-map measured anything welfare-relevant |
| **Attention/position dynamics as *the* mechanism** behind readout failure | **Scoped and recommended against.** Two of four instances have non-attentional mechanisms — the judge window has no model-internal mechanism at all, and B2's mode (i) is mass moving to *unscored synonyms of the same answer*. Collapse is generic across random directions. `attention-mechanism-scope-2026-07-30.md` |
| **A general research harness** | **Scoped and declined.** The most heavily instrumented study here still shipped three dead endpoints; they were caught by adversarial review, not by machinery. Almost every primitive already exists (Pre-SPEC, particle-physics blind analysis, specification-curve analysis, MLflow/DVC). `research-harness-scope-2026-07-30.md` |
| **Three-instance synthesis** | **Does not clear the bar.** Finding II does not share the mechanism; forcing it in is a stretch. `synthesis-scope-proposal-2026-07-30.md` |
| **Arm G forward agenda** (multi-turn decay, Arm S1, ceiling reimplementation) | **Retired.** All were attempts to extract more from a saturating variable. §14 is what ended that agenda — it is not part of it (`RESEARCH_ARC.md` §19.1) |

## 6. Evidence grades at a glance

| Claim | Grade |
|---|---|
| Catalog-position behavioural dissociation (§14) | **Confirmatory** |
| Orthogonalization repair, with its measured limit (§15) | **Confirmatory**, bounded |
| Finding I, base-vs-tuned | **Confirmatory** |
| Finding I, differential-by-condition | **Exploratory** |
| B2 generic collapse | **Confirmatory** within its pre-registration |
| `digit_mass` as scaffold-skip proxy | **Confirmatory** at adapter level; per-item untested |
| Finding II | **Exploratory** — one instance |
| Base-vs-tuned compliance reversal | **Direction only**; instrument compromised |
| Marker-anchoring repairability | **Untested** (S4 VOID) |
| "The Arm G variable controls the decision" | **UNTESTED** — not disproven. §12 ran on the instrument §14 voided; a negative from a broken instrument is not a negative |

## 7. Artifact map

- **`RESEARCH_ARC.md`** — evidence ledger, current. §13–§17 retract large parts of Arm G;
  §18 records the 07-29/30 strand; §19 holds the two standing clarifications. The
  defensible-claims list carries a **VOID block** (quarantine, not annotation).
- **`docs/audit-closeout-2026-07-30.md`** — every claim traced to a code path and labelled
  CLEAN / CLEAN IN EFFECT / CONTAMINATED / RETRACTED.
- **`docs/OPEN_ITEMS.md`** — 11 parked items, blocking section empty.
- **`equanimity_factorial/`** — the 2×2 study. `GATE_RESULT.md` carries the threat catalog
  **T1–T9**. *Provenance note:* threats numbered beyond T9 were raised in a concurrent
  working session and **never written to disk**; the `refusal_margin` construct failure and
  the judge-window defect are documented in `docs/equanimity-endpoint-audit-2026-07-29.md`
  instead.
- **`readout_validity/`** — `PREREG.md`, `RESULTS.md`, notebook. Per-item records archived
  in the private HF dataset repo `codechockablock/phi-map-readout-validity-ckpt` under
  `run_5df7a062/`.
- **`measure_primitives.py`** — shared indexing helper, mandatory `fingerprint()`, and the
  behavioural lint with hash-keyed exemptions.
- **`valence_position_check.py`** — preserved **byte-exact** at `d6aac3e` with its defect
  intact; provenance in the sidecar; corrected fork is `valence_position_check_r3.py`
  (post-hoc, no registered status).

**Branches.** `readout-validity` is a strict superset of `arm-g-causal-characterization`
and is the complete final state; `arm-g-causal-characterization` has been fast-forwarded
to match so both point at the same commit. `master` is historical (last touched
2026-07-23) and was not merged.

**Compute spent tonight:** ≈6 units (Step 4), ≈0.6 (valence refit), ≈1.2 (B2) ≈ **7.8
units**. Two results — Step 5's S4 VOID and the `digit_mass` Branch A — were obtained at
**zero** compute from artifacts already on disk.

## 8. What a future reader should not do

- Do not cite anything in the `RESEARCH_ARC.md` **VOID block** — flip counts, correction
  rates, the "~5% of decisions" claim, or the 1.5× saturation *explanation*.
- Do not read "UNTESTED" as an opening. The instrument that would test decision-level
  control is the one §14 voided.
- Do not restart the Arm G forward agenda.
- Do not describe this work as welfare-adjacent.
- Do not treat the audit's own bugs as evidence about published metrics.

## 9. Honest accounting

Most of tonight's work established what was **not** true. Two pre-registered primary
endpoints died, a headline was retracted, a synthesis was scoped and found not to clear
the bar, and three candidate framings were closed. The single strongest result — §14 — was
already in the repository before the night began; what the night added was the recognition
that it, rather than the probe machinery built on top of it, is the finding.

Four pre-registered verdicts fired mechanically and none was negotiated: **B3**, **S4
VOID**, **V-1 SURVIVES**, **Branch A**. In two of those the rule produced a result that
would plausibly have been talked out of otherwise. That is the part of the process worth
carrying forward, and it cost about fifty lines of branch adjudication rather than a
framework.

**Repository frozen 2026-07-30.**
