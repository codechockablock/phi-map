# Equanimity Factorial

A 2×2 factorial fine-tuning experiment separating **training-content stance**
(equanimity vs. neutral) from **reasoning-trace verbosity** (terse vs. verbose)
on Llama-3.1-8B-Instruct, with LoRA adapters and ≥3 seeds per cell.

Read [`PREMISE.md`](PREMISE.md) first. It corrects the framing this experiment
was handed.

## Status

| Stage | State |
|---|---|
| Premise check against source | done — the cited confound is real, the cited *effects* are not established |
| Power analysis / estimator validation | done — `power.py --self-test` passes |
| Prompt pool (679 prompts, 5 categories) | done |
| Four-cell generation | done for the 100-prompt calibration pool; expanded pool in progress |
| Orthogonality gate | calibrated and passing self-test; real-data run pending the expanded pool |
| Training + eval | **not started** — gated on the above |

## Layout

| File | What it is |
|---|---|
| [`PREMISE.md`](PREMISE.md) | What the prior work actually reports, and what that changes |
| [`power.py`](power.py) | Variance-component power analysis + the shipping estimator, validated on synthetic worlds |
| [`prompts.py`](prompts.py) | The fixed prompt pool, identical across all four cells |
| [`expand_prompts.py`](expand_prompts.py) | Grows the pool to the size the gate's power analysis demands |
| [`generate.py`](generate.py) | Four-cell response generation with stance and length as independent directives |
| [`gate.py`](gate.py) | The orthogonality gate. Can refuse the experiment |
| [`train_eval.py`](train_eval.py) | LoRA training, held-out eval, 2×2 analysis |
| [`make_notebook.py`](make_notebook.py) | Emits the Colab notebook |
| `equanimity_factorial_colab.ipynb` | **The runnable artifact** — hand this to Colab |

Every module has a deterministic `--self-test` that runs without GPU or network,
following the repo convention. Run them all before spending anything.

## The three corrections this build made to the handoff

**1. The effects being decomposed are one question and two prompts.**
93.8% → 95.0% is 75/80 → 76/80 (Fisher p = 1.00); 42% → 25% is 5/12 → 3/12
(p = 0.67). The prior work's *confound disclosure* is accurate and honest, but
there is no established effect size to attribute. The experiment is reframed from
"decompose a known effect" to "is there a detectable effect of either factor,"
with a fourth pre-registered outcome — no effect at this resolution, reported as
bounds. See `PREMISE.md`.

**2. A bootstrap over 3 seeds has a 25% false-positive rate.**
The handoff proposes "a formal 2-way ANOVA or a simple bootstrap over seeds."
The bootstrap is not fine: at the handoff's own mandated k=3, a percentile
bootstrap rejects a true null ~25% of the time per contrast, because three
replicates cannot support a resampling distribution. Across three contrasts a
spurious finding becomes the *expected* outcome of a null experiment. `power.py`
ships a pooled-variance t interval (df = 4(k−1)) plus a permutation cross-check;
measured false-positive rate 4–5%, CI coverage 94.5%. The null world caught this,
not inspection.

**3. Power is asymmetric, so the outcomes are not symmetric.**

| Outcome | k=3 MDE (σ_seed=1.5pp) | Seeds for the prior-reported effect |
|---|---|---|
| GSM8K accuracy | 3.76pp | **22** (88 runs) |
| Harmful compliance | 6.82pp | **2** |

Safety is the primary outcome; capability is a bounded secondary reported as an
interval. `train_eval.py --pilot` measures σ_seed before the sweep commits, so
the seed count is chosen from data rather than asserted.

## Where the gate is stricter than §2 of the handoff

1. **Paired within-prompt, not marginal.** The handoff compares stance means
   "collapsed across length." Collapsing across a factor is exactly the operation
   that hid the catalog-order confound in Arm G — marginal balance is not
   crossing. `nesting_report` confirms crossing; the length contrast is paired
   within prompt *and* within verbosity.
2. **TOST equivalence, not a point estimate under 0.2.** "No significant
   difference" is not evidence of no difference. An observed d of 0.19 whose CI
   reaches 0.35 satisfies the handoff's wording and establishes nothing.
   Orthogonality is an equivalence claim and gets an equivalence test.
3. **The gate's own power is measured.** A TOST that cannot certify clean data is
   not strict, it is uninformative. On synthetic clean data a 100-prompt pool
   passes only 28% of the time, so the pool was sized from
   `required_n_for_equivalence` on observed noise (≈650) rather than from the
   handoff's round number.
4. **Per-category is a detection screen, not a certification.** Each category
   holds ~1/5 of the pool, where a TOST cannot certify |d| < 0.2 for *any*
   dataset. Categories get a Holm-corrected paired difference test instead —
   3% false-flag rate on clean data, and it catches a confound confined to one
   category.
5. **A fresh content instrument.** The handoff suggests the valence direction
   from the geometric work. That direction was withdrawn — phi-map's own
   re-extraction found it was 79% catalog position, which is why the geometry
   line is closed. Importing it would launder that failure into this experiment.
   A grouped-CV TF-IDF classifier is used instead, checked against a
   label-shuffled null and scored within verbosity strata so it cannot be reading
   length.

## Open decisions, as resolved

**D1 — Base vs. Instruct: Instruct.** Follows the handoff's own recommendation.
Matches the prior work's setting, and the training data is chat-formatted
multi-turn, which the base model has no template for. Base remains the follow-up
if a result appears.

**D2 — Generator: Claude Sonnet 5 via the `claude -p` CLI**, pinned in
`generate.py:GENERATOR_MODEL`, held constant across all four cells. Routed
through the existing subscription rather than a separate API key. What matters
for validity is constancy across cells, which is asserted; the specific model
matters only for response quality, and spot-checking confirmed the neutral cells
are engaged and substantive rather than terse-by-neglect.

**D3 — Repo placement: a new top-level `equanimity_factorial/` inside phi-map.**
It reuses `confound_audit.py` (whose import path is hardcoded in the
measurement-discipline skill) and follows the repo's script + `--self-test` +
Colab convention. It is deliberately *not* filed under Arm G: that line and the
geometry work are closed, and this is a training experiment rather than
frozen-model probing.

**D4 — Prompt pool: 679 prompts across five categories.** Two dysphoric/hostile
categories mirroring the prior work's destabilising stimuli, and three ordinary
hard-task categories (technical, judgment, underspecified) so the factorial is
not measured only on emotionally loaded input. 100 hand-authored seeds; the
remainder generated by the pinned generator, deduplicated by normalised text and
TF-IDF near-duplicate filtering. **No attack content is authored** — these are
hard inputs, not jailbreaks; the safety eval uses published adversarial sets
(AdvBench, JBB-Behaviors).

## Compute

Local is an M4 with 24 GB unified memory, MPS, no CUDA, and no `peft`/`trl`. That
is not a platform for 12+ LoRA runs on an 8B model plus 13 evaluation passes, so
training runs on Colab via the notebook, matching how every Arm G experiment in
this repo is run. Data generation and the entire gate run locally and need no GPU.

## Running it

```bash
python3 power.py --self-test && python3 gate.py --self-test && python3 train_eval.py --self-test
```

```bash
python3 generate.py --run --workers 16 && python3 gate.py --run
```

Then upload the launch files to Drive and run
`equanimity_factorial_colab.ipynb`. The notebook asserts on the gate's exit code
and will not train on a dataset that fails it.
