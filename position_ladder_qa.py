"""Position ladder v2: EXACT replication of Liu et al. Appendix E, then extension.

SUPERSEDES the v1 KV instrument (position_ladder.py, kept frozen for provenance).
Prereg: docs/position-ladder-v2-prereg-2026-08-01.md. Verdict = evaluate() here.

WHY V2 EXISTS. v1 anchored on the real Appendix E claim -- "only the larger
models (13B and 70B) exhibit the U-shaped performance curve ... the smallest
Llama-2 models (7B) are solely recency-biased" -- but implemented the WRONG
TASK. The claim is about multi-document QA (20 NQ docs, ~4K tokens); v1 built
key-value retrieval, which Llama-2 was never run on and whose smallest published
setting does not even fit Llama-2's window (their Table 4: 75 pairs = 5444
tokens on a Llama tokenizer). v1's ANCHOR_FAILED therefore meant "wrong
instrument", not "result fails to reproduce".

V2 runs THEIR released pipeline verbatim -- data, prompt template, chat
formatting, sampling parameters, scorer -- pinned by commit and file hash below.
This module never re-implements any of it; it only adjudicates their scored
outputs.

Run:  python3 position_ladder_qa.py --self-test
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------- pinned upstream
LITM_REPO = "https://github.com/nelson-liu/lost-in-the-middle.git"
LITM_COMMIT = "29b8a6d042ce29abccee3db1a73171a107d7e6af"
LITM_SHAS = {
    "src/lost_in_the_middle/prompting.py":
        "c8de5f26e22261ed540e7578ff4ca79b7548aa5b9ddbc33b769afdae494acb26",
    "src/lost_in_the_middle/metrics.py":
        "b2ac7e1c2fac7e1d09aabbeee782bebe4fb74b650b42cb9111b88692e5a2f35e",
    "src/lost_in_the_middle/prompts/qa.prompt":
        "368a84fe24373cc0a3a789ce5e62e6854994ece138747c93c76f25aafea25411",
    "scripts/get_qa_responses_from_llama_2.py":
        "43506abcf9ac0ea7f98a24cef2440485bb4db3c066e6b317a3b8fbe71e53c1f8",
    "scripts/evaluate_qa_responses.py":
        "fd35118106f02e781143f720c21b71ceeb8e4592e5b440f54f035a8562dd5b63",
    "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz":
        "69a8dd87a45f07fb3ed9b804917a89db975cd6e488682fc838edca029cc3881f",
    "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl.gz":
        "c5e44700c4db24536f139f7383e3dd002413515ee889aeadc72767dd7f289682",
    "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_9.jsonl.gz":
        "cbada411f70235afc47d9b5407e9358ad72eb021742402ff032c8b65076ffb2f",
    "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_14.jsonl.gz":
        "86c1a25093615baefdd1ee4db7a993548b3bef24ca02eb9bd8db43fbedf517f2",
    "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl.gz":
        "d79b715e1b8c8301335e93cb01659f68c24f0253a808c3190b06d21896506c6c",
}

POSITIONS = (0, 4, 9, 14, 19)          # their gold_at indices, 20-document setting
N_EXPECTED = 2655                      # examples per gold_at file
# Their script skips prompts > 4096 Llama-2 tokens (~20/2655, Appendix E).
N_MIN = 2600

# protocol constants, theirs
TEMPERATURE, TOP_P, MAX_NEW_TOKENS, MAX_PROMPT_LENGTH = 0.0, 1.0, 100, 4096

ROWS = {
    # anchors: run via THEIR script, verbatim
    "llama2-7b-base":   dict(hf="meta-llama/Llama-2-7b-hf",         role="anchor",    variant="base"),
    "llama2-7b-chat":   dict(hf="meta-llama/Llama-2-7b-chat-hf",    role="anchor",    variant="chat"),
    "llama2-13b-base":  dict(hf="meta-llama/Llama-2-13b-hf",        role="anchor",    variant="base"),
    "llama2-13b-chat":  dict(hf="meta-llama/Llama-2-13b-chat-hf",   role="anchor",    variant="chat"),
    # extensions: their prompt + scorer + sampling, own chat template (documented
    # deviation -- their format_chat_prompt is Llama-2-specific). Base-variant ids
    # are verified at Phase 0; they exist to de-confound tuning (see FIG16 below).
    "llama31-8b-base":  dict(hf="meta-llama/Llama-3.1-8B",          role="extension", variant="base"),
    "llama31-8b-chat":  dict(hf="meta-llama/Llama-3.1-8B-Instruct", role="extension", variant="chat"),
    "olmo3-7b-base":    dict(hf="allenai/Olmo-3-1025-7B",           role="extension", variant="base"),
    "olmo3-7b-chat":    dict(hf="allenai/Olmo-3-7B-Instruct",       role="extension", variant="chat"),
}

# Figure 16 digitized by hand from the published plot, +/-2pp reading error.
# These are TARGETS for the anchor gate, fixed before any generation.
FIG16 = {
    "llama2-7b-base":  {0: 0.230, 4: 0.220, 9: 0.235, 14: 0.245, 19: 0.405},
    "llama2-7b-chat":  {0: 0.435, 4: 0.415, 9: 0.445, 14: 0.460, 19: 0.560},
    "llama2-13b-base": {0: 0.420, 4: 0.215, 9: 0.210, 14: 0.250, 19: 0.410},
    "llama2-13b-chat": {0: 0.510, 4: 0.480, 9: 0.480, 14: 0.525, 19: 0.585},
}

# THE TUNING CAVEAT, from their own Figure 16: 13B base -> chat collapses the
# primacy index from ~+0.195 to ~+0.015. Chat-tuning can SUPPRESS primacy, so an
# extension CHAT row without primacy is evidence about nothing; only BASE rows
# can speak to the parameter-count question.

BARS = dict(
    no_prim=0.05,   # G1: a 7B anchor "lacks primacy" if index < this
    prim=0.10,      # G2: 13B-base "has primacy" if index >= this
    rec=0.05,       # G3: every anchor must show recency >= this
)
SENS = dict(no_prim=(0.03, 0.07), prim=(0.07, 0.13), rec=(0.03, 0.07))


def indices(curve: dict[int, float]) -> dict[str, float]:
    """Primacy/recency indices against the INTERIOR MEAN (positions 4, 9, 14).

    Same estimator class as v1 after its min->mean fix; interior mean is
    unbiased and pools 3x the data of any single point.
    """
    assert set(curve) == set(POSITIONS), f"need exactly positions {POSITIONS}"
    interior = (curve[4] + curve[9] + curve[14]) / 3
    return dict(primacy=round(curve[0] - interior, 4),
                recency=round(curve[19] - interior, 4),
                interior_mean=round(interior, 4))


def classify(curve: dict[int, float], bars: dict = BARS) -> dict[str, Any]:
    ix = indices(curve)
    p, r = ix["primacy"], ix["recency"]
    prim = "PRESENT" if p >= bars["prim"] else \
           "ABSENT" if p < bars["no_prim"] else "INDETERMINATE"
    return dict(**ix, primacy_call=prim, recency_ok=r >= bars["rec"],
                curve={k: round(v, 4) for k, v in sorted(curve.items())})


def evaluate(m: dict[str, Any], bars: dict = BARS) -> dict[str, Any]:
    """m = {row_key: {curve: {pos: acc}, n_by_pos: {pos: int}}}

    Anchor gate first, built from Appendix E's own contrasts:
      G1  llama2-7b-base AND llama2-7b-chat primacy < no_prim
      G2  llama2-13b-base primacy >= prim
      G3  every present anchor recency >= rec
    13b-chat is recorded, never gated -- its published primacy (~+0.015) sits
    inside measurement noise by design, so a gate on it would be undiagnostic.
    """
    per: dict[str, Any] = {}
    for key, d in m.items():
        c = d.get("curve") or {}
        nb = d.get("n_by_pos") or {}
        if set(c) != set(POSITIONS):
            per[key] = dict(verdict="V_INCOMPLETE",
                            reasons=[f"positions {sorted(c)} != {list(POSITIONS)}"])
            continue
        low = {p: n for p, n in nb.items() if n < N_MIN}
        if len(nb) != len(POSITIONS) or low:
            per[key] = dict(verdict="V_N_MISMATCH",
                            reasons=[f"n per position below {N_MIN} or missing: "
                                     f"{low or 'missing counts'}"])
            continue
        per[key] = dict(verdict="OK", **classify(c, bars))

    def ok(k):
        return k in per and per[k]["verdict"] == "OK"

    g1_rows = ("llama2-7b-base", "llama2-7b-chat")
    reasons, failed = [], []
    if not (ok("llama2-13b-base") and all(ok(k) for k in g1_rows)):
        gate = dict(status="INCOMPLETE",
                    reasons=[f"gate rows not all readable: "
                             f"{[k for k in (*g1_rows, 'llama2-13b-base') if not ok(k)]}"])
    else:
        for k in g1_rows:
            if per[k]["primacy"] >= bars["no_prim"]:
                failed.append(f"G1 {k}: primacy {per[k]['primacy']:+.3f} >= {bars['no_prim']}")
        if per["llama2-13b-base"]["primacy"] < bars["prim"]:
            failed.append(f"G2 llama2-13b-base: primacy "
                          f"{per['llama2-13b-base']['primacy']:+.3f} < {bars['prim']}")
        for k in (*g1_rows, "llama2-13b-base", "llama2-13b-chat"):
            if ok(k) and not per[k]["recency_ok"]:
                failed.append(f"G3 {k}: recency {per[k]['recency']:+.3f} < {bars['rec']}")
        gate = dict(status="ANCHOR_FAILED" if failed else "ANCHOR_OK",
                    reasons=failed or ["Appendix E reproduced: 7B rows lack primacy, "
                                       "13B-base shows it, recency everywhere"])

    out = dict(anchor_gate=gate, per_model=per)
    if gate["status"] != "ANCHOR_OK":
        out["driver"] = None
        out["reasons"] = ["instrument not validated; extension rows not interpreted"]
        return out

    # Driver logic. ONLY BASE extension rows bear on parameter count -- their own
    # Figure 16 shows chat-tuning collapsing 13B primacy 0.195 -> 0.015, so a
    # chat row's absence is never evidence about parameters.
    ext_base = {k: per[k] for k in ("llama31-8b-base", "olmo3-7b-base") if ok(k)}
    ext_chat = {k: per[k] for k in ("llama31-8b-chat", "olmo3-7b-chat") if ok(k)}
    calls_b = {k: v["primacy_call"] for k, v in ext_base.items()}
    calls_c = {k: v["primacy_call"] for k, v in ext_chat.items()}
    out["extension_calls"] = dict(base=calls_b, chat=calls_c)

    if len(ext_base) < 2:
        out["driver"] = None
        out["reasons"] = [f"need both base extension rows readable, have {sorted(ext_base)}"]
    elif any(v == "PRESENT" for v in calls_b.values()):
        out["driver"] = "NOT_PARAMETER_COUNT"
        out["reasons"] = [f"primacy at 7-8B params in a later-generation BASE model: "
                          f"{[k for k, v in calls_b.items() if v == 'PRESENT']}; "
                          f"parameter count held vs llama2-7b-base"]
    elif all(v == "ABSENT" for v in calls_b.values()):
        out["driver"] = "PARAMETER_COUNT_CONSISTENT"
        out["reasons"] = ["no 7-8B base model shows primacy despite very different "
                          "training; consistent with (not proof of) a parameter threshold"]
    else:
        out["driver"] = "UNRESOLVED"
        out["reasons"] = [f"indeterminate base calls: {calls_b}"]
    return out


def sensitivity(m: dict[str, Any]) -> dict[str, Any]:
    base = evaluate(m)
    bk = (base["anchor_gate"]["status"], base.get("driver"))
    grid = {}
    for key, vals in SENS.items():
        for v in vals:
            b = dict(BARS); b[key] = v
            r = evaluate(m, b)
            grid[f"{key}={v}"] = (r["anchor_gate"]["status"], r.get("driver"))
    return dict(base=bk, grid=grid, fragile=any(x != bk for x in grid.values()))


def _selftest() -> None:
    full_n = {p: 2635 for p in POSITIONS}

    def row(curve, n_by_pos=None):
        return dict(curve=dict(curve), n_by_pos=n_by_pos or dict(full_n))

    # 1. the digitized published curves must PASS the gate -- if the gate cannot
    #    accept the paper's own figure, the gate is wrong, not the paper
    anchors = {k: row(v) for k, v in FIG16.items()}
    e = evaluate(anchors)
    assert e["anchor_gate"]["status"] == "ANCHOR_OK", e["anchor_gate"]
    assert e["driver"] is None, "driver claimed with no extension rows"
    ix = indices(FIG16["llama2-13b-base"])
    assert 0.15 < ix["primacy"] < 0.25 and ix["recency"] > 0.15, ix
    assert indices(FIG16["llama2-7b-chat"])["primacy"] < 0.01
    assert indices(FIG16["llama2-7b-base"])["primacy"] < 0.01

    # 2. swapping 7B and 13B-base curves must FAIL both G1 and G2
    swapped = dict(anchors)
    swapped["llama2-7b-chat"] = row(FIG16["llama2-13b-base"])
    swapped["llama2-13b-base"] = row(FIG16["llama2-7b-chat"])
    e2 = evaluate(swapped)
    assert e2["anchor_gate"]["status"] == "ANCHOR_FAILED"
    assert any("G1" in r for r in e2["anchor_gate"]["reasons"])
    assert any("G2" in r for r in e2["anchor_gate"]["reasons"])
    assert e2["driver"] is None

    # 3. driver worlds, on top of passing anchors
    prim_curve = FIG16["llama2-13b-base"]          # has primacy
    flat_rec = FIG16["llama2-7b-chat"]             # recency only
    ext = lambda b1, b2, c1=flat_rec, c2=flat_rec: {
        **anchors,
        "llama31-8b-base": row(b1), "olmo3-7b-base": row(b2),
        "llama31-8b-chat": row(c1), "olmo3-7b-chat": row(c2)}
    e3 = evaluate(ext(prim_curve, flat_rec))
    assert e3["driver"] == "NOT_PARAMETER_COUNT", e3["reasons"]
    e4 = evaluate(ext(flat_rec, flat_rec))
    assert e4["driver"] == "PARAMETER_COUNT_CONSISTENT", e4["reasons"]
    # chat-only primacy must NOT drive a parameter claim in either direction
    e5 = evaluate(ext(flat_rec, flat_rec, c1=prim_curve))
    assert e5["driver"] == "PARAMETER_COUNT_CONSISTENT"
    assert e5["extension_calls"]["chat"]["llama31-8b-chat"] == "PRESENT"
    # indeterminate base -> UNRESOLVED
    mid = {0: 0.315, 4: 0.24, 9: 0.24, 14: 0.24, 19: 0.40}   # prim = +0.075
    e6 = evaluate(ext(mid, flat_rec))
    assert e6["driver"] == "UNRESOLVED", e6

    # 4. void paths: missing positions / low n block everything downstream
    broken = dict(anchors)
    broken["llama2-13b-base"] = dict(curve={0: .4, 4: .2}, n_by_pos={0: 2635, 4: 2635})
    e7 = evaluate(broken)
    assert e7["per_model"]["llama2-13b-base"]["verdict"] == "V_INCOMPLETE"
    assert e7["anchor_gate"]["status"] == "INCOMPLETE" and e7["driver"] is None
    lown = dict(anchors)
    lown["llama2-7b-base"] = row(FIG16["llama2-7b-base"], {p: 500 for p in POSITIONS})
    e8 = evaluate(lown)
    assert e8["per_model"]["llama2-7b-base"]["verdict"] == "V_N_MISMATCH"
    assert e8["driver"] is None, "subsampled row treated as full-n evidence"

    # 5. missing extension rows block the driver, never default it
    assert evaluate({**anchors, "llama31-8b-base": row(prim_curve)})["driver"] is None

    s = sensitivity({**anchors,
                     "llama31-8b-base": row(prim_curve), "olmo3-7b-base": row(flat_rec),
                     "llama31-8b-chat": row(flat_rec), "olmo3-7b-chat": row(flat_rec)})
    assert s["base"] == ("ANCHOR_OK", "NOT_PARAMETER_COUNT")
    print("self-test OK: published curves pass, swapped curves fail G1+G2, "
          "4 driver worlds, chat rows inert, 2 void paths, subsample refused")
    print(f"  fig16 indices: 13b-base prim {indices(FIG16['llama2-13b-base'])['primacy']:+.3f}, "
          f"7b-chat prim {indices(FIG16['llama2-7b-chat'])['primacy']:+.3f}")
    print(f"  sensitivity fragile: {s['fragile']}")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv: _selftest()
    else: print(__doc__)
