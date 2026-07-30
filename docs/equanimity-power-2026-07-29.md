# Step 2 — Power, with seed SD measured and n resolved

**Date:** 2026-07-29. **Revised** after the checkpoint, on instruction to measure the
seed SD rather than inherit it and to resolve the `n` discrepancy.
**Reproduce:** `python3 seed_sd_from_artifacts.py` then `python3 mde_seed_unit.py`
from the repo root. No GPU, no network, no compute units.

> **Two corrections, both mine.**
>
> **(1) To the checkpoint report.** It gave MDE ≈ 15 pp with a 12.7 pp floor,
> computed at n=100 and an assumed σ_seed of 1.5 pp, and judged the n discrepancy
> "not load-bearing." Both inputs were wrong and the judgement was wrong. The
> artifacts say **n = 400** (GSM8K) and **n = 200** (AdvBench). Correcting n moved the
> *diagnosis*, not just the number: at n=100 the eval-noise floor dominated and made
> small effects unreachable regardless of training stability; at the real n the floor
> drops to 6.3 pp and σ_seed becomes the sole binding constraint.
>
> **(2) To the first revision of this document.** It reported MDE = 11.25 pp using a
> measured σ_seed of 3.50 pp. That **double-counted eval noise**: 3.50 pp is an
> *observed* within-cell SD, which already contains eval noise, and the MDE formula
> then added it again. Under `power.py`'s model `Var(y) = σ_s² + σ_e²`, the contrast
> SE is `observed_SD/√k`. Corrected: the training component deconvolves to
> **σ_seed = 2.58 pp** and **GSM8K MDE at k=2 is 9.34 pp**, not 11.25. Every MDE in
> the first revision was inflated by roughly 20%. Found by reading `pilot.json`,
> which had done the deconvolution correctly in form — see §1.1.

---

## 0. What the artifacts actually contain — this precedes any power calculation

| cell | evals | seeds |
|---|---:|---|
| equanimity-terse | 3 | 100, 101, 102 |
| equanimity-verbose | **0** | adapter exists, never evaluated |
| neutral-terse | 4 | 1000, 1001, 1002, 1003 |
| neutral-verbose | **0** | **no adapter at all** |
| base_control | **absent** | — |

The design is **not** 2/2/2/2. Eight adapters exist, spread 3/1/4/0 across cells;
seven evals exist, spread 3/0/4/0. Consequences:

- **Both evaluated cells are terse. Factor B is unestimated — not underpowered,
  unestimated.** Same for the interaction.
- The only estimable contrast is Factor A at fixed B = terse, on 3 versus 4 seeds.
  That is a two-sample comparison, not a 2×2.
- `base_control` is missing, so base-versus-adapter cannot currently be computed at
  all — including the "one well-powered contrast" the checkpoint report named.

## 1. Measured seed SD

Pooled within-cell across equanimity-terse (k=3) and neutral-terse (k=4), **df = 5**.
χ² interval on σ.

| outcome | σ_seed | df | 95% CI on σ | upper/point |
|---|---:|---:|---|---:|
| **GSM8K accuracy** | **3.50 pp** | 5 | [2.18, 8.57] | 2.45× |
| AdvBench compliance | 2.89 pp | 5 | [1.80, 7.09] | 2.45× |
| GSM8K truncated_frac | 14.46 pp | 5 | [9.02, 35.45] | 2.45× |
| GSM8K mean_tokens | 121.8 tok | 5 | [76.0, 298.7] | 2.45× |
| refusal_margin | 0.650 units | 5 | [0.406, 1.594] | 2.45× |
| valence dysphoric_mean | 0.0448 | 5 | [0.0279, 0.1098] | 2.45× |
| valence dys_minus_neu | 0.0405 | 5 | [0.0253, 0.0992] | 2.45× |

Per-cell within-cell SDs for GSM8K accuracy: equanimity-terse 4.56 pp (k=3),
neutral-terse 2.55 pp (k=4).

## 1.1 Two components, and why the distinction matters

The table above is **observed** within-cell SD, which contains both the training
component and eval noise. Deconvolving with eval noise at the *actual* n:

| outcome | observed SD | eval noise | **σ_seed (training)** | 95% CI |
|---|---:|---:|---:|---|
| GSM8K accuracy | 3.495 pp | 2.360 pp (n=400, p0=.665) | **2.578 pp** | **[0.00, 8.24]** |
| AdvBench compliance | 2.890 pp | 2.121 pp (n=200, p0=.10) | **1.963 pp** | **[0.00, 6.76]** |

Compare like with like: `README.md:62` assumed **σ_seed = 1.50 pp**, the training
component. Measured training component is **2.58 pp — 1.7× the assumption**. The
observed total, which is what the contrast SE is built from, is 3.50 pp (2.3× the
assumption once eval noise is included). Either way the assumption was optimistic, in
the direction the within-one-cell hit_cap (36% vs 55%) and median-length (956 vs 3001
chars, 3.14×) spread already implied.

**Both lower bounds deconvolve to zero.** At df = 5 these data cannot distinguish
perfectly reproducible LoRA training from σ_seed near 8 pp. The point estimate is
usable; the interval's lower end is not informative, and no design decision should
lean on it.

**`pilot.json` did this deconvolution too, and got a different answer.** It reports
`gsm8k.sigma_seed_pp = 2.149`, from the same neutral-terse four seeds (its
`observed_sd_pp = 2.5526` matches my within-cell SD exactly). But it subtracted eval
noise of **1.377 pp**, which is `100·√(0.25/1319)` — `power.py`'s hardcoded
`n_eval = 1319, p0 = 0.50` — when the eval ran at **n = 400**, where the correct term
is 2.360 pp, 1.71× larger. So the pilot's σ_seed is a deconvolution against the wrong
constant.

Worse, `pilot.json` reports `jailbreak.sigma_seed_pp = 0.0`, because its observed SD
(1.893 pp) fell *below* its own eval-noise term (3.373 pp). That is a variance
component pinned at the boundary — a known small-sample pathology — and it must not
be read as "training is stable." A pilot whose stated purpose was to measure σ_seed
before the sweep committed returned zero for one of the two outcomes and an
undercorrected value for the other, and the sweep proceeded.

## 2. The number, as a band rather than a point

**GSM8K accuracy, k=2, n=400, seed as the unit, 80% power, two-sided α=0.05:**

> ### MDE = 9.3 pp at the measured σ_seed, and the honest interval is [6.3, 22.8] pp.

| outcome | k | at σ_lo | at point | at σ_hi |
|---|---:|---:|---:|---:|
| GSM8K accuracy | 2 | 6.34 pp | **9.34 pp** | 22.82 pp |
| GSM8K accuracy | 3 | 4.41 pp | 6.49 pp | 15.86 pp |
| AdvBench compliance | 2 | 5.64 pp | **7.69 pp** | 18.85 pp |
| AdvBench compliance | 3 | 3.92 pp | 5.34 pp | 13.10 pp |

A 3.6× range. The lower bound equals the §5 floor exactly, because σ_lo deconvolves to
zero — an internal consistency check rather than a coincidence. The imprecision cannot
be narrowed by more eval items; only more seeds narrow it, and more seeds are also
what would narrow the σ interval producing it.

## 3. Sensitivity curve — the point of the revision

MDE in pp, GSM8K accuracy, n = 400, p0 = 0.65, eval SD/run = 2.38 pp:

| σ_seed | k=2 | k=3 | k=4 | k=6 | k=8 |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 6.88 | 4.78 | 3.95 | 3.11 | 2.65 |
| 1.50 ← README | 7.49 | 5.21 | 4.30 | 3.39 | 2.89 |
| 2.00 | 8.28 | 5.75 | 4.75 | 3.74 | 3.19 |
| **2.58 ← measured** | **9.34** | **6.49** | **5.36** | **4.22** | **3.60** |
| 4.00 | 12.39 | 8.61 | 7.11 | 5.60 | 4.78 |
| 6.00 | 17.17 | 11.93 | 9.85 | 7.76 | 6.63 |
| 8.00 | 22.20 | 15.43 | 12.74 | 10.04 | 8.57 |
| 10.00 | 27.34 | 19.00 | 15.69 | 12.36 | 10.55 |

AdvBench compliance, n = 200, p0 = 0.10, eval SD/run = 2.12 pp:

| σ_seed | k=2 | k=3 | k=4 | k=6 | k=8 |
|---:|---:|---:|---:|---:|---:|
| 1.50 ← README | 6.91 | 4.80 | 3.97 | 3.12 | 2.67 |
| **1.96 ← measured** | **7.69** | **5.34** | **4.41** | **3.48** | **2.97** |
| 6.00 | 16.92 | 11.76 | 9.71 | 7.65 | 6.53 |
| 10.00 | 27.19 | 18.89 | 15.60 | 12.29 | 10.49 |

## 4. Clustering assumption

Unchanged from the checkpoint version, and it is the load-bearing methodological
choice. Observations are clustered by adapter; **the seed/adapter is the unit of
inference**; eval items are within-cluster. Greedy decode means zero within-run
sampling variance — a run is a deterministic function of (weights, prompt). Variance
model, `power.py:13-23`:

```
y = mu + alpha_i + beta_j + (alpha*beta)_ij + s + e
   s ~ N(0, sigma_seed^2)    LoRA run-to-run variation in TRUE accuracy
   e ~ finite eval set
```

A main effect contrasts 2k runs against 2k runs, so `SE = sigma_total / sqrt(k)`,
identical for both main effects and the interaction under orthogonal ±1 contrasts.

The conditional-versus-unconditional distinction from the checkpoint version still
holds — the item set is fixed and shared across adapters, so under a
"these-400-items" target the eval term cancels — but it now matters much less,
because at n=400 the eval term is small relative to the measured σ_seed. All numbers
above are **unconditional** (generalising past the specific items), which is what any
writeup claims.

## 5. Irreducible floor at the resolved n

| outcome | k | eval SD/run | SE | MDE floor |
|---|---:|---:|---:|---:|
| GSM8K accuracy | 2 | 2.38 pp | 1.69 pp | **6.34 pp** |
| GSM8K accuracy | 3 | 2.38 pp | 1.38 pp | 4.41 pp |
| AdvBench compliance | 2 | 2.12 pp | 1.50 pp | 5.64 pp |
| AdvBench compliance | 3 | 2.12 pp | 1.22 pp | 3.92 pp |

Compare the checkpoint's 12.68 pp floor at n=100. The floor halved when n was
corrected, which is why the n discrepancy *was* load-bearing: at n=100 it was the
binding constraint and small effects were unreachable at any σ_seed; at n=400 the
binding constraint is unambiguously σ_seed and k.

## 6. Seeds required at the measured σ_seed

| target effect | GSM8K k | AdvBench k |
|---|---:|---:|
| 1.25 pp (prior capability) | 63 | 43 |
| 3 pp | 12 | 8 |
| 5 pp | 5 | 4 |
| 10 pp | 2 | 2 |
| 16.7 pp (prior jailbreak) | 2 | 2 |

Read k as 4k training runs for a full 2×2. Resolving a 3 pp effect on GSM8K needs
**48 training runs**; the 1.25 pp capability effect needs **252**. Both are lower
than the checkpoint's estimates (160 and 640 seeds/cell) because n=400 beats n=100 and
because the double-counted eval noise is gone — but both remain far beyond 8 adapters.

## 7. The pooling fallback is confounded **and** cross-instrument

base-versus-all-adapters-pooled is the tempting fallback: greedy decode makes the
base deterministic, so it is a one-sample t against a fixed constant with
df = n_adapters − 1 rather than 4(k−1). It fails for two independent reasons, and the
second is decisive.

**(a) Confounded by construction.** Pooling all cells answers "does fine-tuning on
this data change X," not "does stance change X." It cannot separate Factor A from
Factor B. With the current artifacts it could not answer even that: both evaluated
cells are terse, and `base_control` is absent.

**(b) Measured with two different instruments.** The base model emits no `ANSWER:`
marker, so `extract_answer` falls back to scoring the **whole response**
(`train_eval.py:166-168`). Adapters emit the marker and are scored on the
**extracted answer** only. One arm is whole-response, the other is post-marker text.
The size of that instrument difference, measured on the same adapters and the same
generations:

| eval | extracted | raw window | ratio | hit_cap |
|---|---:|---:|---:|---:|
| neutral-terse seed1000 | 12.0% | 3.5% | **3.43×** | 36.0% |
| neutral-terse seed1001 | 10.0% | 4.5% | **2.22×** | 54.5% |

The instrument change is larger than any effect the design can resolve, **and the
multiplier itself differs between two seeds of the same cell.** That closes off the
fallback rather than caveating it.

## 8. GSM8K accuracy is no longer a clean construct either

Step 1 graded `gsm8k_accuracy` "valid construct, structurally underpowered." The
artifacts revise that grade.

| cell | truncated_frac | mean | accuracy mean |
|---|---|---:|---:|
| equanimity-terse | 77.0, 47.0, 81.2 | **68.4%** | 60.7% |
| neutral-terse | 24.0, 37.2, 46.8, 25.2 | **33.3%** | 66.6% |

Truncation is **24-81%** overall and differs between cells by **35.1 pp along Factor
A**, with accuracy **5.9 pp lower** in the higher-truncation cell.
`train_eval.py:119-120` states the criterion itself: `truncated_frac` "must come back
near zero; if it does not, the capability numbers are not usable." It has not.

A binding cap converts correct answers into failures, so from these artifacts the
apparent capability deficit and the truncation gap are **not separable**. This is the
same mechanism the code warned about (lines 107-110) — it warned about it tracking
Factor B, and it is tracking Factor A instead.

## 9. Verdict

At the corrected n and the deconvolved σ_seed, the design resolves main effects of
about **9 pp** on GSM8K and **8 pp** on compliance, with honest intervals reaching
23 pp and 19 pp. Against motivating effects of 1.25 pp and 16.7 pp, and with:

- Factor B and the interaction **unestimated** (two empty cells),
- `base_control` **absent**,
- the best-powered contrast **confounded and cross-instrument**,
- GSM8K accuracy **confounded with a 35 pp between-cell truncation gap**,
- `refusal_margin` and the v1 compliance judge **already dead**,
- self-report **populated but unreadable on at least one adapter** — `digit_mass` falls
  to 0.031, so 96.9% of next-token mass is off the rating digits (audit §A5b). *An
  earlier revision of this document said the self-report stage was empty; that was my
  own scalar-only display filter dropping nested dicts, and it is retracted — see
  audit §A5.*

there is no contrast in the current artifacts that is simultaneously interpretable
and adequately powered. That is the ceiling, now measured rather than assumed.
