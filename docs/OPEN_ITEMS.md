# Open items

Things found and deliberately **not** pursued, per the depth stop set 2026-07-30.
Logged so they are not lost and not re-derived. Nothing here is a task list; each entry
is parked until something else makes it load-bearing.

## Blocking

| # | Item | Blocks |
|---|---|---|
| 1 | **The valence refit has not run.** Staged and greenlit; Colab's Drive OAuth consent opens in a browser window the extension cannot reach, so it needs one tap from a human. `docs/valence-refit-prereg-2026-07-30.md` holds the criteria; `valence_refit.py` is self-tested. | Results §9.3 disposition; Finding I's instance count; the closeout table (§5 of the 2026-07-30 dispatch); Step 5's exclusion of `valence_axis` |

## Parked, not blocking

| # | Item | Why parked |
|---|---|---|
| 2 | **The 3.0–3.4 rating band matches neither candidate prior** — not the base two-turn prior (4.44/4.55), not the training digit marginal (≈2.15–2.42). Remaining candidates: an adapter-specific format-context prior, a template-induced attractor, genuine construct insensitivity. | Needs its own pre-registration with a predicted value stated in advance. A base↔training interpolation fits any value in [2.2, 4.4] with a free weight and is unfalsifiable as stated. |
| 3 | **The 27 Arm G / `p2_harness` lint hits are verified false positives** (left padding via `pad_prompt_batch`; batch-of-1). Recorded as hash-keyed exemptions with reasons. | Closed, not open — logged here only because A11.2 previously overstated it. |
| 4 | **`train_eval.py`'s three corrected sites change future numbers.** Any rerun of `eval_selfreport`, `refusal_margin`, or `get_valence_direction` now produces different values from the seven eval artifacts on disk. | Correct behaviour going forward; the artifacts are already labelled contaminated in results §9.4. Only matters if someone reruns and compares. |
| 5 | **`valence_direction.npz` on Drive was fit on pad-position states** and has not been regenerated. | The refit (item 1) produces a clean axis in memory but writes only `valence_refit.json`. If any future work wants the clean axis as an artifact, it must be written deliberately and versioned separately from the as-run one. |
| 6 | **Multiplicity across the 2×2's 24 tests.** No policy was found in the design docs. | Superseded in practice: the 2×2 is retired and reclassified exploratory (effect sizes with intervals, no inferential p-values). Would matter again only if a confirmatory 2×2 claim is revived. |
| 7 | **The notebook's markdown says JailbreakBench; the code and all seven artifacts say `advbench`.** | Documentation/code divergence in a reproducibility claim. One-line fix, no result depends on it. |
| 8 | **`equanimity_factorial/train_eval.py` line references in the audit doc have shifted** by the three-site patch. | Cosmetic. The audit cites the file as it stood when read, and says so in its header. |

## Deliberately not opened

Per the 2026-07-30 depth stop: no tests of the lint's fixtures, no layer above the
lint, no further verification threads. The lint has known-bad/known-good behavioural
fixtures and hash-keyed exemptions that lapse on edit. That is the terminal depth.
