# Scope: a reusable research harness — proposal only

**Date:** 2026-07-30. **Proposal, not a build.** No code, no commitments, no compute.

**Headline:** the discipline is real and worth keeping, but **the evidence from tonight
argues for the cheap declarative 20% and against the harness.** The primitives that
demonstrably worked are a document template, a ~50-line adjudicator, and a fingerprint
call. The expensive machinery this project already built did **not** prevent the failures
it was built to prevent. I recommend extracting the former and not building the latter,
and I flag the home/remote tension rather than deciding it.

---

## 1. What was actually earned, extracted from the documents

Not generic best practice. Each row cites where it came from and what it cost.

| Primitive | Where it was earned | The failure that earned it |
|---|---|---|
| **Ordered branch rules, first match wins, adjudicated in code** | `readout_validity/PREREG.md` §"Branch rules — evaluated in this order, first match wins" | Round-1's two-signature rule was unfalsifiable and had to be withdrawn in review (`step4-prereg-review`); ordering + first-match makes the verdict deterministic rather than negotiable |
| **A DISQUALIFIED branch distinct from a null branch** | `readout_validity/PREREG.md` rule 1 — "the run is uninformative about their readout. Report and stop" | Distinguishes "the manipulation did nothing" from "the manipulation worked and the effect is absent." Conflating them is how a broken run reads as a null |
| **A terminal AMBIGUOUS branch** | same, rule 6 — "Report the numbers; claim no branch" | Removes the pressure to pick a branch when none fits |
| **Disqualifying preconditions checked *before* execution** | `step5-prereg` D-α | Step 5 fired **S4 VOID at zero compute** from coverage figures already in Step 4 §3 — and the same figures were in hand when the prereg was written and were not checked. Logged as OPEN_ITEMS #11 |
| **Thresholds with stated provenance** | `readout_validity/PREREG.md` §"Numeric thresholds — stated in advance, with provenance" | Explicitly records *how* each bar was set, and states when no measured null existed |
| **Verdict sensitivity across a grid** | same — {0.4,0.5,0.6}·S0 × {0.7,0.8,0.9}·S0 | Mandated *because* thresholds could not be set from a measured null. B2 held in **all nine cells**, which is what makes it non-fragile |
| **Precision basis with its assumption named** | same — "exact only under that model" (exchangeability) | The T4/T7 MDE corrections in `GATE_RESULT.md` |
| **"Reported alongside any branch"** | same §, mandatory companions | Per-direction numbers "never only pooled (standing rule)" |
| **"What no outcome licenses"** | every prereg tonight | Stops a measurement result from being read as a welfare/capability claim |
| **Per-cell alongside every pooled figure** | `GATE_RESULT.md` standing rule 1; audit §A12 | Caught that E[digit] was **54.7% one cell** |
| **Name what the denominator varies over** | `GATE_RESULT.md` standing rule 1, verbatim | "Violated three times: prompt dispersion, stimulus-class gap, and the mislabelled null floor" |
| **Verify the artifact that produces the answer, not a stand-in** | `GATE_RESULT.md` standing rule 2, verbatim | A coverage simulation against a re-implementation validates the re-implementation |
| **Behavioural self-tests; textual checks banned** | `measure_primitives.py`; audit §A11 | `eos_reached` nan, the string-matching self-test, and the lint's own known-bad/known-good fixtures |
| **Harness fingerprint in every record** | `measure_primitives.fingerprint()` | The valence fork/no-fork confusion existed *only* because artifacts did not name their code |
| **Quarantine over annotation** | `RESEARCH_ARC.md` VOID block | The §14/"~5% of decisions" failure was a banner over live bullets — exactly annotation-not-quarantine |
| **Blind/peek discipline with breaches logged** | `GATE_RESULT.md` T8 | Records what was already seen rather than implying cleanliness |
| **Escalate on genuine ambiguity** | n=100 vs n=400; the per-item `digit_mass` impossibility | Both were reported rather than silently resolved |
| **Standing-rules log** | `GATE_RESULT.md`, T1–T12 | Institutional memory across a study rather than per-session |

**The core artifact is not code. It is `readout_validity/PREREG.md`'s section
structure** — Scope / Fixed design / Definitions / Thresholds with provenance / Ordered
branch rules / Reported alongside any branch / What no outcome licenses / Outcome /
Amendment. That template is the reusable thing, and it is a document.

## 2. Novelty check — run before proposing, per the standing rule

**Almost every primitive exists somewhere, and several have canonical names I should use
rather than reinvent.**

| Primitive | Prior art | Verdict |
|---|---|---|
| Prereg with pre-stated decision rules | [Pre-SPEC](https://arxiv.org/pdf/1907.04078) (clinical trials): "adaptive analysis strategies should use **deterministic decision rules**"; OSF/AsPredicted; Registered Reports | **Not novel** |
| Blind analysis, formal unblinding protocol | Mature in particle physics for decades — [CMS blinding/unblinding](https://cms.cern/physics/cms-higgs-search/blinding-and-unblinding-analyses), [SLAC-PUB-12051](https://www.slac.stanford.edu/pubs/slacpubs/12000/slac-pub-12051.pdf) | **Not novel.** T8 is a reinvention |
| **Sensitivity grid** | This has a name: **specification curve analysis** (Simonsohn, Simmons & Nelson 2020) and **multiverse analysis** (Steegen et al. 2016), with tooling ([`specr`](https://github.com/masurp/specr)) | **Not novel.** Use the existing name |
| Fingerprinting: code version + params + artifacts | MLflow, DVC, Sacred, W&B, [MLXP](https://arxiv.org/pdf/2402.13831), [MLDev](https://arxiv.org/pdf/2107.12322) | **Not novel** |
| Checkpointing / resumability, atomic per unit | Snakemake, Nextflow, DVC pipelines | **Not novel** |
| Per-cell alongside pooled | Standard (Simpson's paradox) | **Not novel** |
| Capturing *decisions and why*, not just outputs | [Procedural Knowledge Libraries](https://arxiv.org/pdf/2506.14715): "MLflow and Sacred track parameters but do not fully capture the steps the researchers took or why" | **Actively being worked on by others** |
| Prereg in ML specifically | [NeurIPS pre-registration workshops](https://preregister.science/); [Perspectives from Psychology's Reproducibility Crisis](https://arxiv.org/pdf/2104.08878) | **Not novel** |
| Behavioural-over-textual checks | Ordinary testing discipline | **Not novel as a principle.** The *named banned defect class with an enforcing lint* is unusual |
| Quarantine over annotation | No tooling found | **Not found** — but it is a documentation convention, not a contribution |
| Budget guard as anomaly detector | No prior art found | **Not found**, and too small to matter |

**What I did not find:** a single tool where the pre-registration is **executable** — where
branch rules are adjudicated mechanically by the same artifact that declares them, coupled
to run fingerprinting. That combination is **not found**, three query formulations, and
**"not found" is not "not there"** — I did not run a declared frame.

**Honest read of the combination.** The valuable part would be that the prereg is not a
PDF filed elsewhere but *code that returns the verdict*. That is a real gap. It is also a
**small** gap: the thing that closes it in this repo is `evaluate()` — one function that
takes measured values and returns a branch name.

## 3. The counter-evidence I have to take seriously

**The most heavily instrumented study tonight still shipped three structurally dead
endpoints.**

`equanimity_factorial` had `gate.py` (678 lines), `power.py` (440 lines, estimator
validated against synthetic nulls), a self-test suite, an orthogonality gate that could
refuse the experiment, four calibration rounds, and a twelve-item threat register. It
still produced: `refusal_margin` dead on construct validity, binary compliance
uninterpretable, JBB post-hoc.

**None of those was caught by the machinery. All three were caught by adversarial
review** — by someone asking what the denominator varied over and where the content
actually sat.

That is the strongest argument against building a harness, and it comes from this
project's own record. Machinery enforces the rules you already knew to write. The
failures tonight were failures to know which rule applied.

**What the machinery *did* do, and it is narrower than it looks:** four verdicts fired
mechanically and none was negotiated — B3, S4 VOID, V-1 SURVIVES, Branch A. In two cases
(S4, A) the mechanical rule produced a result I would plausibly have talked myself out
of. That is real value, and it came from ~50 lines of branch adjudication, not from the
678-line gate.

## 4. The other risk: n = 1

This discipline emerged from **one night, one project, two branches**. Several rules are
corrections to *specific* bugs — the padding lint, the display-filter retraction, the
`digit_mass` reclassification — and there is no evidence yet that they generalise beyond
the shapes that produced them. `measure-discipline`'s own note applies: *"Add items when
a new failure mode is actually hit, not pre-emptively."*

Codifying now risks freezing local accidents as general law. The prereg template has been
used **three times** (equanimity Step 4, readout-validity, Step 5) and is the only
primitive with even that much evidence of reuse.

There is also a precedent in this repo: `docs/trajectory-check-2026-07-28.md` documents
that building domain-general infrastructure for a research programme is how frontier-ops
and unified-stack "started their last chapter." A research harness is exactly that shape.

## 5. Recommendation — graded, and mostly "don't build"

**Build (cheap, evidence-backed, ~a day):**

1. **`PREREG_TEMPLATE.md`** — the section structure from `readout_validity/PREREG.md`,
   verbatim, with the earned sections marked mandatory. It is a document. It has been
   used three times. It is the highest-yield item by a wide margin.
2. **`evaluate()` convention** — every prereg ships a pure function mapping measured
   values → branch name, importable and runnable outside the notebook. Tonight's B2
   verdict was "additionally reproduced outside the notebook by running the
   pre-registered `evaluate()` on the transcribed per-cell values — identical output."
   That is the executable-prereg gap in §2, closed by a convention rather than a system.
3. **`measure_primitives.fingerprint()` as a required field** — already exists, already
   mandatory. Nothing to build.

**Do not build:**

- A general harness, runner, or framework. §3 says machinery did not prevent the
  failures; §4 says the rule set is n = 1.
- Anything duplicating MLflow/DVC/Snakemake (§2).
- A sensitivity-grid module — use specification-curve tooling and its name.

**Revisit when:** the template has been used on **≥3 studies not from tonight**, and at
least one rule has fired on a failure shape it was not written for. That is an evidence
gate, not a time gate.

## 6. Home, and the tension — flagged, not decided

**It wants to be its own thing.** A prereg template and adjudicator convention are
domain-general and have nothing to do with scope conflict or format tokens; leaving them
in `phi-map` means the next project copies files.

**But the standing constraint says no new remote**, and §4's precedent says a fresh repo
for extracted infrastructure is how the last two dormant repos began.

**The tension is real and I am not resolving it.** What I would do *now*: keep it as
`phi-map/protocol/` — two files, no package, no install. It is small enough that
extraction later costs nothing, and small enough that if it is never reused, nothing was
lost. **The decision to extract should be gated on the §5 evidence gate, not on
enthusiasm, and it is yours to make.**

## 7. Not proposed

- No build, no code, no compute in this pass.
- No new remote.
- No welfare framing (retired, `RESEARCH_ARC.md` §19.2).
- No claim that the combination in §2 is novel — that rests on three searches without a
  declared frame.
