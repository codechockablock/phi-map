# Pre-registration v2 — position ladder as exact Appendix E replication, 2026-08-01

Supersedes `docs/position-ladder-prereg-2026-08-01.md` (v1). Adjudicator is
`position_ladder_qa.evaluate()`; the verdict is whatever it returns.

## Why v1 died, on the record

1. **v1 anchored a real claim to the wrong task.** Liu et al. (*Lost in the
   Middle*, TACL 2024) Appendix E: *"only the larger models (13B and 70B)
   exhibit the U-shaped performance curve ... the smallest Llama-2 models (7B)
   are solely recency-biased."* That finding is about **multi-document QA**
   (20 NQ documents, ~4K tokens). v1 built a key-value retrieval task — one
   Llama-2 was never run on, and whose smallest published setting (75 pairs =
   5,444 Llama tokens, their Table 4) does not fit Llama-2's 4096 window.
   v1's `ANCHOR_FAILED` verdict stands, meaning "wrong instrument."
2. **Two summarizer-layer errors, both mine, in opposite directions.** First I
   adopted the 7B/13B claim from a search summary without reading the paper
   (violating `check-literature-before-proposing`). Then, after reading 8 of 18
   pages, I declared the claim absent from the paper — it is in Appendix E,
   page 15. The instrument for claims about papers is reading the paper, all
   of it.
3. The v1 KV run (`row_llama2-7b_ec265db502ca8618.json`) is quarantined as
   descriptive. Known contamination: a first-catalog-value fallback inflates
   position-0 accuracy by roughly +0.15. No conclusions rest on it.

## Design: run their pipeline, not a re-implementation

Upstream pinned at `nelson-liu/lost-in-the-middle @ 29b8a6d042ce`, with sha256
over the ten files that matter (data × 5, prompt template, prompt builder,
scorer, Llama-2 runner, evaluator) — the full list lives in
`position_ladder_qa.LITM_SHAS` and is asserted on the box before any spend.

| component | spec (theirs) |
|---|---|
| task | multi-document QA, 20 total documents, NaturalQuestions-Open |
| data | `gold_at_{0,4,9,14,19}.jsonl.gz`, **2655 questions each, all of them** |
| prompt | `qa.prompt` + `Document [i](Title: …)` formatting, via their `get_qa_prompt` |
| chat wrap | their `format_chat_prompt` ([INST]/<<SYS>>, their default system prompt) |
| generation | vLLM, temperature 0.0, top_p 1.0, max_tokens 100 |
| length rule | prompts > 4096 Llama-2 tokens skipped (their default; ≈20/2655) |
| scoring | their `best_subspan_em` via their `evaluate_qa_responses.py` |

Sample size is **full 2655 per position** — user-confirmed 2026-08-01,
declining the cheaper 500-question subsample.

## Rows and run order

| order | row | HF id | role |
|---|---|---|---|
| 1 | `llama2-13b-base` | meta-llama/Llama-2-13b-hf | anchor, gate G2 |
| 2 | `llama2-7b-chat` | meta-llama/Llama-2-7b-chat-hf | anchor, gate G1 |
| 3 | `llama2-7b-base` | meta-llama/Llama-2-7b-hf | anchor, gate G1 |
| 4 | `llama2-13b-chat` | meta-llama/Llama-2-13b-chat-hf | anchor, recorded |
| 5–8 | `llama31-8b-base/chat`, `olmo3-7b-base/chat` | see module | extension |

Rows 1–2 are the primary gate: **the study can die after two rows.** Rows 5–8
run only after `ANCHOR_OK`. Llama-2-70B is excluded (does not fit A100-40GB);
this is a stated deviation from Appendix E's model list, not from its method.

## Anchor gate, fixed now

Indices against the interior mean (positions 4, 9, 14). Targets digitized by
hand from Figure 16, ±2pp reading error:

| row | curve (0/4/9/14/19) | primacy | recency |
|---|---|---|---|
| 7b-base | .230 .220 .235 .245 .405 | −.003 | +.172 |
| 7b-chat | .435 .415 .445 .460 .560 | −.005 | +.120 |
| 13b-base | .420 .215 .210 .250 .410 | **+.195** | +.185 |
| 13b-chat | .510 .480 .480 .525 .585 | +.015 | +.090 |

- **G1**: primacy(7b-base) < 0.05 AND primacy(7b-chat) < 0.05
- **G2**: primacy(13b-base) ≥ 0.10
- **G3**: recency ≥ 0.05 for every anchor row
- 13b-chat primacy is recorded, never gated: its published value (+0.015) is
  inside noise by construction, so a gate on it would be undiagnostic.
- Sensitivity grid: G1 ∈ {.03,.07}, G2 ∈ {.07,.13}, G3 ∈ {.03,.07}.
- Any anchor row with a missing position or n < 2600 at any position is void
  (`V_INCOMPLETE` / `V_N_MISMATCH`) and blocks the gate — it is not a failure
  and not a pass.

## Extension question and driver verdicts

Same instrument, models at 7–8B parameters with far more training than
Llama-2-7B. **Only base rows bear on parameter count**: their own Figure 16
shows chat-tuning collapsing 13B primacy from +0.195 to +0.015, so absence of
primacy in a chat row is evidence about tuning, not parameters. Chat rows are
run to measure that modulation, not to adjudicate.

- `NOT_PARAMETER_COUNT` — any base extension row primacy ≥ 0.10.
- `PARAMETER_COUNT_CONSISTENT` — both base extension rows < 0.05. Consistent
  with, not proof of, a parameter threshold.
- `UNRESOLVED` — anything else. Missing or void rows block rather than default.

## Deviations, all known in advance

1. vLLM `v0.2.1.post1` (their pin) will not build on a 2026 stack; current
   vLLM is used, version recorded per row. Greedy decoding limits the residual
   to kernel-level numeric differences.
2. `load_format="pt"` in their runner may be rejected by current vLLM; if so it
   is patched to `"auto"` and `patched_load_format: true` is recorded.
3. Extension rows cannot use their runner (argparse whitelists Llama-2 ids and
   the chat wrapper is Llama-2-specific): a thin wrapper reuses their
   `get_qa_prompt`, their scorer, and their sampling parameters, with each
   model's own chat template. Base extension rows use the raw prompt, as their
   base-model path does.
4. Extension tokenizers differ, so their 4096-token skip rule would drop a
   different (near-empty) example set; extension rows therefore skip nothing,
   and per-position n is recorded. Anchor n ≈ 2635 vs extension n = 2655 is a
   0.75% denominator difference, noted here so it cannot become a surprise.
5. Base-model ids `allenai/Olmo-3-1025-7B` and `meta-llama/Llama-3.1-8B`
   verified resolvable (config.json downloaded) 2026-08-01, before commit.
   Phase 0 re-verifies all eight on the box regardless.

## Budget

Full-n, A100-40GB, vLLM: ~1–1.5 h per 7–8B row, ~2–2.5 h per 13B row;
8 rows ≈ 10–13 GPU-hours total, anchors ≈ 6. Per-row `CONFIRMED_BUDGET_HOURS`
is enforced between gold-index files, and every gold-index prediction file is
persisted to the HF dataset repo the moment it exists.

## What this cannot license

- Nothing about Arm G. Different task, different construct.
- Nothing about J-space or any mechanism.
- `NOT_PARAMETER_COUNT` names what it rules out, nothing more. "Training
  compute" is not isolated from data composition, architecture era, context
  extension, or tokenizer.
- One task family, one document count (20), greedy decoding, English NQ.
