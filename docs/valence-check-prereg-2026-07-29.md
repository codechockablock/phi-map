# Pre-registration: does format acquisition contaminate `valence_dysphoric`?

**Written 2026-07-29, BEFORE the check is run.** No hidden states have been
inspected. The only valence numbers I have seen are the seven per-adapter scalar
summaries already in the eval artifacts (`dysphoric_mean`, `dysphoric_sd`,
`neutral_mean`, `dys_minus_neu`), reported in
`seed_sd_from_artifacts.py` §6. I have **not** seen any hidden state, any
projection at probe level, or anything about the direction beyond its existence and
its layer. Every threshold below is either referenced to a null the same script
measures, or declared lenient with its rationale stated.

**Status: not yet run.** Requires Colab. See §7.

---

## 1. The endpoint under test

`valence_dysphoric` = mean projection of the final-prompt-token hidden state at
`VALENCE_LAYER` = 16 onto a frozen difference-of-means valence direction `d`, taken
over the 16 dysphoric probes (`train_eval.py:850-863`). `d` is fit once on the
frozen base model over world-directed positive vs negative stimuli, unit-normalised,
cached with provenance (`train_eval.py:821-847`).

## 2. The threat, stated as a mechanism

`refusal_margin` died because it assumed the construct sits at sequence position 0,
and format fine-tuning moved what sits there: first-token mass on real
refusal/compliance openers went from **96.6%** (base) to **4.7%** (neutral) and
**16.3%** (equanimity), the remainder absorbed by the `REASONING:` token. The tail
size differs along Factor A by 3.5×.

`valence_dysphoric` reads the **same sequence position** — the final prompt token,
i.e. exactly where the adapter is about to emit `REASONING:`. So there is direct
evidence that the intervention reorganises the distribution at that position, by an
amount that differs by condition. If the adapter-induced displacement at that
position has a large component along `d`, the endpoint reads format acquisition
rather than valence.

Unlike the self-report endpoint, which measures the mass of its own support via
`digit_mass`, this endpoint has no such guard. The check builds one.

## 3. Quantities to report

Per model — base plus each of the 8 adapters — over both probe sets (16 dysphoric,
**8** neutral — verified against `train_eval.NEUTRAL_PROBES`, not assumed). `h_p` is the
read-position hidden state for probe `p`.

**Consequence of n=8 on the neutral set, and it is not cosmetic.** `S(M)` is a ratio of
across-probe variances, so on the neutral set both numerator and denominator rest on 8
points — roughly 7 df. The random-direction null is computed on the **dysphoric** set
(n=16) for that reason, and the neutral set is used only for the D3 differencing test,
where it enters as a mean shift rather than a variance. Any `S` reported on the neutral
set carries that caveat explicitly. The crisis set (n=6) is not used here at all.

**(a) Support mass — the mass-on-support analogue, and the primary quantity.**

```
S(M) = Var_p( h_p · d )  /  mean_p( || h_p - h_bar ||^2 )
```

The share of across-probe variance at the read position that the valence direction
actually spans. This is the structural analogue of "fraction of first-token mass on
the scored openers": it asks whether the measuring axis still spans the signal in
the adapter's geometry, rather than assuming it does.

**Null:** 200 random unit directions in R^4096, same computation, report p95 and
the mean. For isotropic across-probe variance the expectation is ~1/4096 ≈ 2.4e-4;
the measured null is what the criteria reference, not that analytic value.

**(b) Displacement of the read position, in the repo's budget-normalised units.**

```
Delta_M = mean_p(h_M) - mean_p(h_base)
```

Report `||Delta|| / ||mean_p(h_base)||` (the `‖Δh‖/‖h‖` convention from
`RESEARCH_ARC.md:750-757`), `cos(Delta, d)`, and the displacement along `d`
expressed as an effect size against the base probe-to-probe spread:
`(Delta · d) / SD_p(h_base · d)`.

**(c) Whether differencing helps.** Compute `Delta` separately on the dysphoric and
neutral probe sets. Report `cos(Delta_dys, Delta_neu)` and
`||Delta_dys - Delta_neu|| / ||Delta_dys||`. If the format shift is common to both
probe sets it cancels in `dys_minus_neu` and does not cancel in `dysphoric_mean`.
This is the quantitative test of the diagnostic difference-score, **not** a
promotion of it (see §6).

**(d) Condition-dependence.** Report `(Delta · d)` separately for the equanimity and
neutral adapters. This is the quantity that decides whether any between-cell valence
contrast is interpretable.

## 4. Pre-committed disqualifying results

Written before the run. Any one of D1, D2 or D4 firing disqualifies
`valence_dysphoric` as an absolute measure.

**D1 — support collapse.**
- *Hard:* if `S(adapter)` falls below the measured random-direction null p95 for any
  adapter, the endpoint is **dead**: the axis no longer spans the read-position
  variance and the projection is being read in a degenerate subspace.
- *Graded:* if `S(adapter) / S(base) < 0.25` for a majority of adapters, **dead**.
  0.25 is declared **lenient by design** — `refusal_margin`'s support collapsed by
  6-20× — and is not fitted to anything. If the verdict flips anywhere in
  [0.10, 0.50] I will report it as threshold-sensitive rather than as a verdict.

**D2 — the endpoint reads the format shift.** If `|cos(Delta, d)|` exceeds the
measured random-direction null p95 **and** `(Delta · d) / SD_p(h_base · d) > 1.0`,
then format acquisition moves the endpoint by more than one probe-spread unit along
its own axis. **Dead as an absolute measure**; may survive in differenced form only
if D3 passes.

**D3 — differencing rescue test.** If
`||Delta_dys - Delta_neu|| / ||Delta_dys|| < 0.25`, the shift is essentially common
to both probe sets, so `dys_minus_neu` removes it and the difference score is the
defensible form. If `>= 0.25`, differencing does **not** rescue it and the endpoint
is dead in both forms. Note D3 alone never *saves* the registered endpoint — it only
determines whether a future pre-registration has a repairable candidate.

**D4 — condition-dependent renormalisation, the decisive criterion.** If
`(Delta · d)` differs between equanimity and neutral adapters by more than
`t_crit(0.975, df=5) x sigma_seed_pooled`, where `sigma_seed_pooled = 0.045`
projection units (measured, df = 5, `seed_sd_from_artifacts.py` §3, 95% CI
[0.028, 0.110]) and `t_crit(0.975, 5) = 2.571` — i.e. a gap exceeding **0.116
projection units** — then the between-cell valence contrast is renormalised by
condition, exactly as `refusal_margin`'s was, and **no A-vs-B valence claim is
licensed in either form.** Because `sigma_seed` itself has a 2.45× upper bound, the
bar will also be reported at the CI upper end (0.283 units) and the verdict called
threshold-sensitive if the two disagree.

**What would vindicate the endpoint.** Stated so this is not a rigged test: if
`S(adapter)` stays within 2× of `S(base)`, `|cos(Delta, d)|` sits at the
random-direction null, and D4 does not fire, then `valence_dysphoric` has survived
the position-sensitivity threat and can be analysed as originally specified. That is
a real possible outcome — `d` was fit on the base model over world-directed stimuli
that share no lexical structure with the training format, so there is no *a priori*
reason the format shift must align with it.

## 5. What each outcome buys, in both terminal branches

This is why the check clears the bar rather than being curiosity.

- **If the terminal answer is "fix the instrument and re-run":** D1-D4 determine
  whether valence must be redefined *before* the re-run, and D3 determines whether
  the difference score is the redefinition to pre-register. Running the re-run first
  and discovering the endpoint was contaminated would waste the re-run.
- **If the terminal answer is "nothing clears the bar at achievable n":** a firing
  D1/D2/D4 makes valence the **fourth confirmed instance** of the position-sensitivity
  finding, joining (1) `refusal_margin`'s first-token support collapse, (2) the
  judge's fixed 400-character window against answer offsets of 0 / ~209 / ~1161
  chars, and (3) the newly measured GSM8K truncation gap — 24-81% overall, **35 pp
  between cells along Factor A**. That finding is the part of this study with real
  support, and a fourth instance across a *different kind of readout* (a hidden-state
  projection rather than a token-level or text-window metric) materially strengthens
  it, because it shows the failure is not specific to token-space instruments.

A null result is equally reportable: it bounds the finding by showing at least one
readout at the same sequence position that format acquisition does **not** capture,
which is informative about scope rather than about nothing.

## 6. What this check does not license

- **It does not switch the endpoint definition.** `OUTCOME_GETTERS` continues to
  read `dysphoric_mean` (`train_eval.py:1128`). Switching to `dys_minus_neu` after
  observing that the current form is contaminated would be endpoint selection on the
  data — the same error class as JBB's post-hoc selection. `dys_minus_neu` is
  reported alongside as a **diagnostic only**, and can be promoted in the *next*
  pre-registration with its promotion criterion stated first.
- **If any code change is applied, it is a logged dual-field change**, emitting both
  `valence_dysphoric` and `valence_dys_minus_neu` with both reported, never a silent
  swap that moves a number.
- It says nothing about whether valence is welfare-relevant. It tests one specific
  threat to one specific instrument.

## 6b. Bundled test P1: are low-support self-report ratings measurements at all?

Added to this pre-registration before running, because the base model and the probe
sets are already loaded by §3 and the marginal cost is one extra forward pass per probe.

**The observation that motivates it.** Across all seven evals and three subsets, ratings
occupy 2.97-3.60 while `digit_mass` ranges 0.031-0.886. The metric returns nearly the
same answer whether it has 89% of the mass or 3%. That is either robustness or a tell,
and the two are distinguishable.

**Procedure.** Compute `prior_rating` = the expected 1-7 rating under the **base**
model's own digit distribution at the read position, renormalised over digits 1-7
exactly as `eval_selfreport` does. Then per adapter × subset define
`pull = |rating − prior_rating|`.

**PRE-REGISTERED CRITERION P1.** If `corr(digit_mass, pull) > 0` and the low-support
points sit materially closer to `prior_rating` than the high-support points, then the
narrow band is the base model's digit prior showing through, and **low-support ratings
are not measurements**. This converts "means are not comparable without conditioning on
support" into a positive statement about what the low-support values are.

**Clustering, mandated here rather than left to judgement.** The correlation is reported
over the **7 adapters**, not the 21 (adapter × subset) rows. Subsets are nested in
adapter. A free partial version of this test run on existing artifacts gave
`corr(digit_mass, mean) = +0.391, p = 0.079` over 21 rows — that p is optimistic by
exactly the error this clause forbids.

**What the free partial test already showed, and why it is not enough.** Low-support
points (n = 3, `digit_mass` < 0.25) span ratings 3.016-3.243, sd 0.125; high-support
points (n = 13, > 0.60) span 2.969-3.684, sd 0.201 — a **1.61× dispersion ratio** in the
predicted direction. `mean − argmax` is uniformly positive (+0.31 to +0.97) at every
support level and does **not** correlate with support (−0.171, n.s.), so that gap is a
property of the renormalisation rather than of support. Suggestive, n = 3 in the low
tier, and it cannot identify *what* the low-support ratings converge on. Only the base
prior pins that, which is what P1 adds.

## 7. Cost, and why it has not run

Requires `peft` and Llama-3.1-8B-Instruct weights. Locally: `peft` is absent and
`README.md:129-134` records the platform limit (M4, 24 GB, MPS, no CUDA). Adapter
weights *are* materialised on Drive (168 MB each) and `valence_direction.npz` is
present, so only the base model and `peft` are missing.

**Estimate, not spent:** load the base model **once** and hot-swap the 8 adapters
rather than reloading per adapter. Then ~9 × 32 single-token forward passes with no
generation — negligible compute — plus one base load and 8 adapter attachments.
Roughly **5-10 minutes of A100 time**, dominated by the single base load. The
200-direction null is a matrix operation on cached states and costs nothing.

Reporting the estimate rather than spending it, per the budget rule. Say the word
and I will emit the Colab cell.
