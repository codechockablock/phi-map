# Pre-registration — position ladder, 2026-08-01

Registered before any GPU spend. Adjudicator is `position_ladder.evaluate()`;
the verdict is whatever it returns. This document does not get to override it.

## Question

Liu et al. (*Lost in the Middle*, TACL 2024) report Llama-2 at 7B is
recency-biased only, while 13B and 70B show the full U-shape with primacy.
Arm G found Llama-3.1-8B prefers catalog line 1 (primacy) and Olmo-3-7B prefers
line 2 (recency) — same parameter band, opposite signs.

**Does primacy track parameter count, or training compute?**

## Design

Liu et al.'s synthetic key–value retrieval. `N` UUID pairs as a JSON object, one
key queried, gold key crossed over 10 evenly spaced positions including both
endpoints. 200 trials per position, 2000 prompts per model.

| row | model | params | ~tokens | role |
|---|---|---:|---:|---|
| `llama2-7b` | Llama-2-7b-chat-hf | 7B | 2T | anchor |
| `llama2-13b` | Llama-2-13b-chat-hf | 13B | 2T | anchor |
| `llama31-8b` | Llama-3.1-8B-Instruct | 8B | 15T | test |
| `olmo3-7b` | Olmo-3-7B-Instruct | 7B | 6T | test |

**Paired.** Byte-identical prompts to every model, same seed (1101). A model
contrast can never be a prompt draw.

**Scoring is mechanical.** Strip the queried key (itself a UUID), take the first
remaining UUID, exact match against gold. No judge. `scorable = False` when no
candidate UUID is emitted at all.

**N is not chosen here.** UUID text tokenizes at a tokenizer-dependent rate and
Llama-2's 4096 window binds the whole ladder. `select_n_pairs()` fixes the rule:
largest of (50, 40, 30, 20) fitting 4096 − 256, chosen once at Phase 0 with the
Llama-2 tokenizer and pinned into every output record. N is a context-window
fact, never a knob turned after seeing a curve. Truncation would delete the
*front* of the JSON — it would destroy the primacy signal and return a clean
recency curve that looks like a result.

## Bars, fixed now

| bar | value | sensitivity grid |
|---|---:|---|
| `index` — endpoint lift over interior mean | 0.10 | 0.05, 0.15 |
| `scorable` — fraction of trials emitting a candidate | 0.75 | 0.60, 0.85 |
| `ceiling` — max accuracy across positions | 0.20 | 0.10, 0.30 |

Endpoint lift is measured against the interior **mean**, not the interior
minimum: min over 8 noisy estimates is downward-biased and would inflate both
indices by roughly 5pp against a 10pp bar.

## Branches

Void precedes shape; an unreachable task cannot have a shape.

- `V_UNSCORABLE` — scorable fraction below bar.
- `V_TASK_UNREACHABLE` — max accuracy below ceiling; the model cannot do the
  task at this N and its curve means nothing.
- `U_SHAPE` / `RECENCY_ONLY` / `PRIMACY_ONLY` / `FLAT` — both, recency only,
  primacy only, neither index clears the bar.

## The anchor gate — the disconfirming condition

`llama2-7b` must return `RECENCY_ONLY` **and** `llama2-13b` must return
`U_SHAPE`. If either fails, `evaluate()` returns `driver = None` and the test
rows are not interpreted at all.

**This is the rule that can kill the study, and it is the point.** If this
instrument cannot reproduce a published, well-cited position curve on the models
that paper used, then nothing it says about Llama-3.1-8B or Olmo is worth
reading. No post-hoc rescue: a mismatch is not to be explained away as a chat
template difference, a decoding difference, or a smaller N.

## Driver verdicts, conditional on `ANCHOR_OK`

- `TRAINING_COMPUTE` — `llama2-7b` lacks primacy while a later-generation 7–8B
  model has it, at held parameter count.
- `PARAMETER_COUNT` — no 7–8B model shows primacy regardless of training.
- `UNRESOLVED` — any other pattern.

A **void row blocks the driver claim**. It is not a row without primacy. The
self-test carries a regression for this: three unreadable models must never
return `PARAMETER_COUNT`, which is absence of measurement read as measurement of
absence.

## What this does not license

- **No claim about Arm G.** Arm G was 2 catalog lines at constant knowledge;
  this is N keys at constant task. A shared answer would be a conjecture. Arm G's
  K=4 test went void twice and is still void.
- **No J-space claim.** The workspace paper studied Haiku/Sonnet/Opus 4.5 only,
  reports no scaling analysis, and never discusses position. Nothing here
  measures a workspace.
- Training-token counts are approximate and confounded with everything else that
  changed between model generations — architecture, context length, RoPE scaling,
  data composition. `TRAINING_COMPUTE` means "not parameter count," not
  "tokens specifically."
- Four models, one task, one seed, greedy decode.
