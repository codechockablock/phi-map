# Audit closeout — every claim traced to a code path and labelled

**Date:** 2026-07-30. **Status:** terminal. This is the artifact that ends the
padding-bug audit. With it complete, the audit is done and stays done; further work is
research execution, not verification.

**Scope.** Every claim in `docs/step4-results-2026-07-30.md`, plus the load-bearing
claims in `docs/equanimity-endpoint-audit-2026-07-29.md` that survive into it.

**Labels.**

| Label | Meaning |
|---|---|
| **CLEAN** | produced by an `attention_mask`-indexed path; no known defect |
| **CLEAN IN EFFECT** | produced through a path with a real defect that was **measured** and found immaterial |
| **CONTAMINATED** | produced by a pad-position path; numbers not trustworthy as stated |
| **RETRACTED** | withdrawn; not to be cited |
| **N/A** | no measurement — argument, scope statement, or bookkeeping |

---

## 1. Confirmatory primaries

| Claim | Code path | Label |
|---|---|---|
| `refusal_openers` Δ = +0.9849, CI [+0.9825, +0.9872] | `step4_run._last_probs` | **CLEAN** |
| `rating_digits` Δ = +0.5700, CI [+0.4275, +0.7125] (two-turn) | `step4_run.digit_read` → `_last_probs` | **CLEAN** |
| `valence_axis` Δ = +0.0205, CI [+0.0198, +0.0211] | correct read (`V.hidden_final_token`) onto an axis fit by `train_eval._hidden_final_token` | **CLEAN IN EFFECT** — refit R5, cos = 0.99989 |
| per-adapter values (openers / digit2 / S) | as above, per row | same as parent row |
| cell-level recomputation, df = 2, t_crit = 4.30 | arithmetic on the above | inherits parent labels |
| achieved MDE 0.0032 (openers) / 0.197 (digits); B5 fires nowhere | `step4_run` sd over 8 adapters | **CLEAN** |
| "no verdict is threshold-sensitive" (§1) | arithmetic | **CLEAN** — note this is §1's claim; §4's D1 and R5 *are* threshold-sensitive and say so |

## 2. Q-B — the pinning kill

| Claim | Code path | Label |
|---|---|---|
| ρ = −0.294, one-sided p = 0.819, exact 8! permutation | `step4_run.spearman_perm` on `rating2` / `digitmass2` | **CLEAN** |
| base two-turn prior 4.438 / 4.550 | `step4_run.digit_read` | **CLEAN** |
| adapters rate 3.00–3.33 at every support level | `step4_run.rating2` | **CLEAN** |
| pull 1.361 (low tier) vs 1.182 (high tier) | arithmetic on the above | **CLEAN** |
| the |r| ≈ 0.71 detectability floor at 8 clusters | analytic | **N/A** |
| training digit marginal ≈ 2.19 → restated 2.15–2.42 | `training_digit_marginal.py` | **CLEAN**, with the extraction-rule caveat in §9.2 |
| "no rating task in the corpus" (1 `x/7`, 3/6700 digit-initial) | same | **CLEAN** — `x/7` recounts as 0 under a stated rule |
| "band matches neither prior" | comparison of the above | **CLEAN** — robust across all four extraction rules |
| "near-identical across cells (2.15–2.27)" | same | **NARROWED** — true spread 0.17–0.31; pooled value 54.7% one cell |

**The pinning kill rests on the location mismatch, and that evidence is CLEAN.** This
was the specific worry raised on review; it does not need redoing.

## 3. Q-C — realignment

| Claim | Code path | Label |
|---|---|---|
| openers not restored, 0/8 at every bar | `step4_run` marker-aligned re-read | **CLEAN** |
| digits not restored at 0.5×; 7/8 clear 0.25× | same | **CLEAN**, threshold-sensitivity stated |
| valence not restored, 0/8 | same, contaminated axis | **CLEAN IN EFFECT** (R5) |
| marker coverage 0.96–1.00 (jailbreak) / 0.46–1.00 (rating turn) | `step4_run`, `_ANSWER_RE` | **CLEAN** |
| `equanimity-verbose` post-marker window fraction 0.003 | same | **CLEAN** |
| `answer_marker` descriptive: terse 0.21–0.33, base undefined | same | **CLEAN** |

## 4. The valence check (D1–D4) and Finding I

| Claim | Code path | Label |
|---|---|---|
| D1 graded fired 8/8 at S/S_base 0.10–0.17 | `valence_position_check` (as-run) | **CLEAN IN EFFECT** (R5: 0.096–0.172 on the clean axis) |
| D1 hard did not fire; base S = 53.5× null | same | **CLEAN IN EFFECT** (R5: 53.8×) |
| D2 pass: max \|cos\| = 0.072, max shift 0.67 SD | same | **CLEAN IN EFFECT** |
| D3: differencing does not rescue, 0.83–0.88 | same | **CLEAN IN EFFECT** |
| D4 pass: \|eq − neu\| = 0.066 — read as *common, not differential* | same | **CLEAN IN EFFECT** |
| **Finding I: four confirmed instances across four readout classes** | composite | **CLEAN** after R5 |
| ~~cross-instrument agreement to four decimals on nine models~~ | shared `.npz` + shared `V.hidden_final_token` | **RETRACTED** — guaranteed by shared input, not earned; stays withdrawn regardless of R5 |

## 5. Exploratory

| Claim | Code path | Label |
|---|---|---|
| cell means (openers / digit2 / S) by cell | `step4_run` | **CLEAN**; valence column CLEAN IN EFFECT |
| "neutral adapters sit higher on S than equanimity" | same | **CLEAN** but explicitly non-inferential — 3–4 seeds, large seed variance, Factor B unestimated |

## 6. Retracted or superseded

| Claim | Fate |
|---|---|
| "0.46 on a 1–7 scale" (§0, §6) | **RETRACTED** — `prior_rating` pad-position bug, R3 §9.1 |
| P1 single-turn digit mass 8.5×10⁻⁶ | **RETRACTED** — same cause |
| the 28× instrument discrepancy | **RESOLVED** at zero compute, R3 §9.1; resolution condition discharged |
| "+3.16 / +4.19 margin units of erosion" (earlier T2) | **WITHDRAWN**, reverses — audit §3.1 |
| "D1–D4 unaffected" (R3 §9.1) | **SUPERSEDED** by R4 §9.3, then resolved by R5 §9.5 |
| "three confirmed, one under test" (R4) | **SUPERSEDED** by R5 — four confirmed |
| "close to all 27 Arm G hits are suspect" (A11.2) | **WITHDRAWN** — all 26 are false positives, A11.2b |
| "self-report never populated" (A5) | **RETRACTED** — my own display filter |

## 7. Claims about the audit's own instrumentation

Quarantined by design. **These are limitations of our measurement code, not data about
published metrics, and are never cited as evidence for Finding I.**

| Claim | Label |
|---|---|
| `prior_rating` read the padded last position, 21/24 probes | **CLEAN** (verified behaviourally with the tokenizer) |
| 3 of 31 lint hits were real; 26 Arm G + 1 `p2_harness` are false positives (left padding / batch-of-1) | **CLEAN** (verified in source) |
| `train_eval.py`'s three sites remediated | **CLEAN** — self-test passes |
| A5b `digit_mass` 0.031 collapse, σ = 0.245, 3.4× range | **CONTAMINATED** — pad-position; conclusion independently confirmed by `step4_run`'s 0.18–0.64 vs 0.983 |
| §6b dispersion ratio 1.61× | **CONTAMINATED** |
| the 2.97–3.60 rating band | **CONTAMINATED** — cite the CLEAN 3.00–3.33 instead |

## 8. Findings, as they finally stand

| # | Finding | Status |
|---|---|---|
| **I** | Format tuning destroys position-fixed readouts. Four confirmed instances across four readout classes: token log-ratio, text window, renormalised tail, hidden-state projection. | **CONFIRMED**; base-vs-tuned robust, differential-by-condition exploratory |
| **II** | Length orthogonality certified on the training pool does **not** transfer off it — d ∈ [0.59, 4.51] on GSM8K against a certified 0.2, ≈3× the bound at the lower limit. | **CONFIRMED**, one instance, version-checked. Kept **separate** from Finding I |
| — | Support mass as a **diagnostic gate** | **DISCONFIRMED** (B3). It remains a validity precondition by arithmetic; it does not predict the failed value's behaviour |
| — | Pinning-to-base-prior | **DEAD**, on the location mismatch |

## 9. What the whole audit does not license

- No welfare claim. This measured instruments, not states.
- No claim that the equanimity-vs-neutral question was answered. It was not, and per
  the power analysis it is unanswerable at achievable n (MDE 9.34 pp [6.34, 22.82]
  against 1.25–3 pp targets).
- No generalisation past Llama-3.1-8B-Instruct, this LoRA recipe, and these 8 adapters
  from 3 configurations.
- No inferential p-values from the retired 2×2, which is reclassified exploratory.

---

**Audit closed.** Remaining work is in `docs/step5-prereg-marker-anchored-2026-07-30.md`
and `docs/OPEN_ITEMS.md`.
