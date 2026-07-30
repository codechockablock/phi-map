# Provenance sidecar — `valence_position_check.py`

This file exists **beside** the script rather than inside it, so the script stays
byte-identical to the version that ran. A restored file carrying a header that records
its own as-run hash while no longer having that hash is self-undermining.

## Status

| | |
|---|---|
| **File** | `valence_position_check.py` |
| **Status** | **AS-RUN, PRE-REGISTERED** |
| **As-run commit** | `d6aac3e` |
| **As-run blob** | `1295f57979501643a14f16db0df9a6a3d40984aa` |
| **Working tree** | byte-identical to that blob (verify: `git diff d6aac3e -- valence_position_check.py` → empty) |
| **Produced** | every number in `docs/step4-results-2026-07-30.md` §4 and `valence_position_check.json` |
| **Corrected fork** | `valence_position_check_r3.py` — post-hoc, **no registered status** |

Recover the exact bytes at any time:

```
git show d6aac3e:valence_position_check.py
```

## Known defects — deliberately NOT fixed in this file

**1. `prior_rating` reads `logits[:, -1, :]`** — the last position of the *padded*
batch. Llama pads right, so 21 of 24 probe readings were taken after an `<|eot_id|>`
pad token. Produced P1's 8.5×10⁻⁶ single-turn digit mass and the **retracted** "0.46 on
a 1–7 scale". Results §9.1.

Scope: registered criteria **D1–D4 do not use this path** — they use this file's own
`hidden_final_token`, which is `attention_mask`-indexed and correct. Only registered
criterion **P1** used the broken path.

**2. Inherited, and more serious than #1: the axis this file measures against is
itself contaminated.** `valence_direction.npz` is fit by `train_eval.get_valence_direction`
→ `train_eval._hidden_final_token` (`train_eval.py:838-839`, reading
`hs[layer][:, -1, :]`). Both 12-probe fit sets fit in one `EVAL_BATCH=32` batch, so
11 of 12 rows in each were read at pad positions.

**This is a defect in the specification of the measuring stick, not in the reading of
it.** Correct read-time indexing does not rescue anything that projects onto that
`.npz`. D1–D4 therefore read at correct positions **along a wrongly-specified axis**,
and so does `valence_axis` in `step4_run.py`. See `docs/valence-refit-prereg-2026-07-30.md`
for the pre-registered test of whether the finding survives an axis correction.

## Why this file is exempt from the position lint

`measure_primitives.py --lint` exempts it by **blob hash**, not by name or by comment.
If the file changes, the hash stops matching and the exemption lapses automatically.
The exemption exists because preserving the as-run bytes is the point.
