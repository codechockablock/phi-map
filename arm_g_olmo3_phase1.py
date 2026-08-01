"""Phase 1 verification: port the Arm G prompt set to Olmo 3 7B tokenization.

Re-derives the pre-action read position SEMANTICALLY (the last templated token
before the model must emit READY), never by porting a Llama index. Handoff
amendment 1 applies: the manifest is built with catalog_order_mode="crossed" --
the corrected protocol from RESEARCH_ARC 14-15 -- not the legacy nested mode
that produced the catalog-order confound.

Zero GPU. Tokenizer-only. Run:  python3 arm_g_olmo3_phase1.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_primitives import fingerprint

OLMO_REPO = "allenai/Olmo-3-7B-Instruct"
OLMO_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"   # resolved Phase 0, pinned
SEED = 111  # the crossed-order seed from the Llama-side confirmatory design


def unwrap(x):
    """transformers v5: apply_chat_template(tokenize=True) returns BatchEncoding;
    len()/iteration give KEY NAMES. Unwrap explicitly and keep the equivalence
    check below -- this exact trap fired once in this session already."""
    if hasattr(x, "input_ids"):
        x = x.input_ids
    elif isinstance(x, dict):
        x = x["input_ids"]
    x = list(x)
    if len(x) == 1 and hasattr(x[0], "__len__"):
        x = list(x[0])
    return x


def main() -> int:
    from transformers import AutoTokenizer
    import arm_g_scenarios as S

    tok = AutoTokenizer.from_pretrained(OLMO_REPO, revision=OLMO_REVISION)
    rows = S.build_manifest(seed=SEED, catalog_order_mode="crossed")

    # 1. UNIFORM FINAL-POSITION IDENTITY over every row, mechanically.
    tails, lens = {}, []
    for r in rows:
        ids = unwrap(tok.apply_chat_template(r["messages"], add_generation_prompt=True))
        lens.append(len(ids))
        tails.setdefault(tuple(ids[-4:]), 0)
        tails[tuple(ids[-4:])] += 1
    assert len(tails) == 1, f"non-uniform tails: {len(tails)}"
    tail = next(iter(tails))
    assert tok.decode(list(tail)).endswith("<|im_start|>assistant\n"), tail
    print(f"[1] all {len(rows)} rows end at the SAME semantic point "
          f"(assistant header, pre-emission); lengths {min(lens)}-{max(lens)}")

    # 2. v5 equivalence check: text path == tokenize path.
    r0 = rows[0]
    text = tok.apply_chat_template(r0["messages"], tokenize=False,
                                   add_generation_prompt=True)
    assert tok(text, add_special_tokens=False)["input_ids"] == \
        unwrap(tok.apply_chat_template(r0["messages"], add_generation_prompt=True))
    print("[2] text-path == tokenize-path")

    # 3. The crossing survives templating: a crossover pair differs in exactly
    #    the two swapped catalog lines and nothing else.
    a = rows[0]
    b = next(r for r in rows if r["crossover_id"] == a["crossover_id"]
             and r["condition"] == a["condition"]
             and r["catalog_order"] != a["catalog_order"])
    ta = tok.apply_chat_template(a["messages"], tokenize=False, add_generation_prompt=True)
    tb = tok.apply_chat_template(b["messages"], tokenize=False, add_generation_prompt=True)
    diff = [(x, y) for x, y in zip(ta.splitlines(), tb.splitlines()) if x != y]
    assert len(diff) == 2, f"crossover pair differs in {len(diff)} lines, expected 2"
    print("[3] crossover pair differs in exactly the 2 swapped catalog lines")

    # 4. Structural risks checked: system turn and consecutive user turns
    #    survive ChatML templating as distinct turns.
    assert text.count("<|im_start|>system") == 1
    assert text.count("<|im_start|>user") == sum(
        1 for m in r0["messages"] if m["role"] == "user")
    print("[4] system turn + consecutive user turns preserved distinctly")

    # 5. READY is a single Olmo token -- the emission clamp check is exact.
    ready = tok("READY", add_special_tokens=False)["input_ids"]
    assert len(ready) == 1, ready
    print(f"[5] READY is a single token ({ready[0]})")

    print(f"\nfingerprint: {fingerprint(__file__)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
