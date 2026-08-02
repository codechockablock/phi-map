"""Position-curve ladder: does primacy track PARAMETER COUNT or TRAINING COMPUTE?

THE QUESTION. Liu et al. (Lost in the Middle, TACL 2024) report Llama-2 at 7B is
recency-biased only, while 13B and 70B show the full U-shape with primacy. Arm G
found Llama-3.1-8B prefers catalog line 1 (primacy) while Olmo-3-7B prefers line 2
(recency) -- two models at the SAME parameter count, opposite signs. If primacy
were set by parameter count, Llama-3.1-8B should look like Llama-2-7B. It does not.

So the ladder crosses the two candidate drivers:

    Llama-2-7B-chat      7B    ~2T tokens    Liu et al. predict RECENCY_ONLY
    Llama-2-13B-chat    13B    ~2T tokens    Liu et al. predict U_SHAPE
    Llama-3.1-8B-Inst    8B   ~15T tokens    Arm G suggests primacy present
    Olmo-3-7B-Inst       7B    ~6T tokens    Arm G suggests recency

The first two are the REPRODUCTION ANCHOR, not the result. If they do not
reproduce Liu et al., this instrument does not measure what it claims and the
other two rows are uninterpretable. That gate is in `evaluate()` and fires first.

WHAT THIS IS NOT. It is not the Arm G line-position effect. That was 2 lines at
constant knowledge; this is 50 keys at constant task. A shared answer would be a
conjecture, not a finding. Arm G's K=4 test went void twice and is still void.

DESIGN, fixed here:
  - Liu et al.'s synthetic key-value retrieval. Chosen over their NQ-open task
    because scoring is exact string match on a UUID -- no judge, no semantic
    call. This repo has been burned by graders, not by models.
  - N=50 pairs. BINDING CONSTRAINT: Llama-2 has a 4096-token window and every
    model must see byte-identical prompts or position curves are not comparable.
    50 pairs ~ 2.3k tokens leaves headroom for the chat template.
  - 10 gold positions, evenly spaced, INCLUDING both endpoints -- endpoints carry
    the entire primacy/recency signal.
  - PAIRED: the same trial set (same UUIDs, same order) goes to every model, so a
    model contrast is never confounded with the prompt draw.

Run:  python3 position_ladder.py --self-test
"""
from __future__ import annotations

import json
import random
import re
from typing import Any

N_PAIRS = 50
N_POSITIONS = 10
TRIALS_PER_POSITION = 200

# Ladder rows. `revision` is pinned at run time into the output record.
LADDER = {
    "llama2-7b":   dict(hf="meta-llama/Llama-2-7b-chat-hf",        params_b=7,  tokens_t=2.0,  role="anchor"),
    "llama2-13b":  dict(hf="meta-llama/Llama-2-13b-chat-hf",       params_b=13, tokens_t=2.0,  role="anchor"),
    "llama31-8b":  dict(hf="meta-llama/Llama-3.1-8B-Instruct",     params_b=8,  tokens_t=15.0, role="test"),
    "olmo3-7b":    dict(hf="allenai/Olmo-3-7B-Instruct",           params_b=7,  tokens_t=6.0,  role="test"),
}

# Liu et al.'s predictions for the anchor rows, fixed before any data exists.
ANCHOR_PREDICTION = {"llama2-7b": "RECENCY_ONLY", "llama2-13b": "U_SHAPE"}

PROMPT = ("Extract the value corresponding to the specified key in the JSON "
          "object below.\n\nJSON data:\n{blob}\n\nKey: \"{key}\"\n"
          "Reply with the value only.")

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _uuid(rng: random.Random) -> str:
    h = "%032x" % rng.getrandbits(128)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def gold_positions(n_pairs: int = N_PAIRS, n_pos: int = N_POSITIONS) -> list[int]:
    """Evenly spaced, endpoints included. Endpoints are the whole signal."""
    return [round(i * (n_pairs - 1) / (n_pos - 1)) for i in range(n_pos)]


def build_trials(seed: int = 1101, n_pairs: int = N_PAIRS,
                 trials_per_position: int = TRIALS_PER_POSITION) -> list[dict[str, Any]]:
    """Key-value retrieval trials, gold key crossed over position.

    The SAME returned list is served to every model in the ladder.
    """
    rng = random.Random(seed)
    positions = gold_positions(n_pairs)
    rows: list[dict[str, Any]] = []
    for t in range(trials_per_position):
        for pos in positions:
            keys = [_uuid(rng) for _ in range(n_pairs)]
            vals = [_uuid(rng) for _ in range(n_pairs)]
            blob = json.dumps(dict(zip(keys, vals)), indent=2)
            rows.append(dict(
                trial=t, gold_position=pos, n_pairs=n_pairs,
                gold_key=keys[pos], gold_value=vals[pos],
                prompt=PROMPT.format(blob=blob, key=keys[pos]),
            ))
    return rows


def score_one(completion: str, gold_key: str, gold_value: str) -> dict[str, Any]:
    """Mechanical. No judge, no fuzzy match.

    The queried key is itself a UUID, so a model that echoes the key before
    answering would be scored on the echo. Strip the key first, then take the
    FIRST remaining UUID -- taking `gold_value in completion` instead would count
    a model that dumps several candidates as correct.

    `scorable` is the reachability precondition: a completion with no candidate
    UUID at all is not a wrong retrieval, it is an unreadable trial, and a
    position curve built on unreadable trials measures nothing.
    """
    cands = [u for u in UUID_RE.findall(completion) if u != gold_key]
    if not cands:
        return dict(scorable=False, correct=False, n_candidates=0, emitted=None)
    return dict(scorable=True, correct=cands[0] == gold_value,
                n_candidates=len(cands), emitted=cands[0])


def curve(scored: list[dict[str, Any]]) -> dict[int, float]:
    """Accuracy by gold position, over SCORABLE trials only."""
    import collections
    num: dict[int, int] = collections.Counter()
    den: dict[int, int] = collections.Counter()
    for r in scored:
        if not r["scorable"]:
            continue
        den[r["gold_position"]] += 1
        num[r["gold_position"]] += int(r["correct"])
    return {p: num[p] / den[p] for p in sorted(den) if den[p]}


# --------------------------------------------------------------------------
# Adjudicator. Committed before any GPU spend; the verdict is whatever this
# returns. Void branches precede shape branches -- an unreachable task cannot
# have a shape.

VOID_VERDICTS = frozenset({"V_UNSCORABLE", "V_TASK_UNREACHABLE", "V_UNREADABLE"})

BARS = dict(index=0.10, scorable=0.75, ceiling=0.20)
SENS = dict(index=(0.05, 0.15), scorable=(0.60, 0.85), ceiling=(0.10, 0.30))


def classify(c: dict[int, float], bars: dict = BARS) -> dict[str, Any]:
    """Curve shape from endpoint lift over the interior minimum.

    Endpoint-minus-interior-minimum, not endpoint-minus-mean: the U-shape claim
    is specifically that the ENDS beat the WORST INTERIOR point. A mean baseline
    would let a monotone rising curve score as recency.
    """
    if len(c) < 3:
        return dict(verdict="V_UNREADABLE", reasons=[f"only {len(c)} positions"])
    ps = sorted(c)
    first, last = c[ps[0]], c[ps[-1]]
    interior = [c[p] for p in ps[1:-1]]
    floor = min(interior)
    prim, rec = first - floor, last - floor
    if prim >= bars["index"] and rec >= bars["index"]:
        v = "U_SHAPE"
    elif rec >= bars["index"]:
        v = "RECENCY_ONLY"
    elif prim >= bars["index"]:
        v = "PRIMACY_ONLY"
    else:
        v = "FLAT"
    return dict(verdict=v, primacy_index=round(prim, 4), recency_index=round(rec, 4),
                interior_floor=round(floor, 4), first=round(first, 4),
                last=round(last, 4), curve={p: round(c[p], 4) for p in ps})


def evaluate(m: dict[str, Any], bars: dict = BARS) -> dict[str, Any]:
    """m = {model_key: {curve: {pos: acc}, scorable_frac: float}}

    Anchor gate first: if Llama-2 7B/13B do not reproduce Liu et al., the
    instrument is unvalidated and the test rows carry no interpretation.
    """
    per: dict[str, Any] = {}
    for key, d in m.items():
        sf = d.get("scorable_frac", 1.0)
        c = d.get("curve") or {}
        if sf < bars["scorable"]:
            per[key] = dict(verdict="V_UNSCORABLE",
                            reasons=[f"scorable {sf:.3f} < {bars['scorable']}"])
            continue
        if not c or max(c.values()) < bars["ceiling"]:
            top = max(c.values()) if c else 0.0
            per[key] = dict(verdict="V_TASK_UNREACHABLE",
                            reasons=[f"max acc {top:.3f} < {bars['ceiling']}; "
                                     f"model cannot do the task at N={N_PAIRS}"])
            continue
        per[key] = classify(c, bars)

    anchors = {k: per[k]["verdict"] for k in ANCHOR_PREDICTION if k in per}
    missing = [k for k in ANCHOR_PREDICTION if k not in per]
    mismatch = {k: (v, ANCHOR_PREDICTION[k]) for k, v in anchors.items()
                if v != ANCHOR_PREDICTION[k]}
    if missing:
        gate = dict(status="INCOMPLETE", reasons=[f"anchor rows absent: {missing}"])
    elif mismatch:
        gate = dict(status="ANCHOR_FAILED", reasons=[
            f"{k}: got {got}, Liu et al. predict {want}"
            for k, (got, want) in mismatch.items()])
    else:
        gate = dict(status="ANCHOR_OK", reasons=["Liu et al. reproduced at 7B and 13B"])

    out = dict(anchor_gate=gate, per_model=per)
    if gate["status"] != "ANCHOR_OK":
        out["driver"] = None
        out["reasons"] = ["instrument unvalidated; test rows not interpreted"]
        return out

    # The actual contrast: same parameter band, different training compute.
    #
    # A VOID row is not a row without primacy. Coercing `V_UNSCORABLE` to
    # `has_primacy=False` would let three unreadable models return
    # PARAMETER_COUNT -- absence of measurement read as measurement of absence,
    # which is the defect class this repo keeps finding. Void rows are dropped
    # and their absence blocks the driver claim.
    small = {k: per[k]["verdict"] for k in ("llama2-7b", "llama31-8b", "olmo3-7b")
             if k in per}
    void = {k: v for k, v in small.items() if v in VOID_VERDICTS}
    has_primacy = {k: v in ("U_SHAPE", "PRIMACY_ONLY")
                   for k, v in small.items() if v not in VOID_VERDICTS}
    if len(has_primacy) < 3:
        out["driver"] = None
        out["reasons"] = ["need all three readable 7-8B rows to separate the drivers"] + (
            [f"void: {void}"] if void else
            [f"missing: {sorted({'llama2-7b','llama31-8b','olmo3-7b'} - set(small))}"])
    elif has_primacy.get("llama2-7b") is False and any(
            has_primacy.get(k) for k in ("llama31-8b", "olmo3-7b")):
        out["driver"] = "TRAINING_COMPUTE"
        out["reasons"] = ["primacy present at 7-8B for a later-generation model "
                          "and absent for Llama-2-7B; parameter count held"]
    elif not any(has_primacy.values()):
        out["driver"] = "PARAMETER_COUNT"
        out["reasons"] = ["no 7-8B model shows primacy regardless of training; "
                          "consistent with a parameter-count threshold"]
    else:
        out["driver"] = "UNRESOLVED"
        out["reasons"] = [f"primacy pattern does not separate the drivers: {has_primacy}"]
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


# N_PAIRS cannot be fixed from here: UUID text tokenizes at a rate that depends
# on the tokenizer, and Llama-2's 4096 window is the binding constraint for the
# whole ladder. So the RULE is fixed instead, before any data -- largest
# candidate that fits, chosen once at Phase 0 with the real Llama-2 tokenizer and
# pinned into every output record. N is a context-window fact, never a knob to
# turn after seeing a curve.
N_CANDIDATES = (50, 40, 30, 20)
CTX_HEADROOM = 256


def select_n_pairs(tokenizer, max_ctx: int = 4096,
                   candidates: tuple[int, ...] = N_CANDIDATES) -> tuple[int, int]:
    """(n_pairs, worst_case_tokens). Phase 0. Call before ANY generation.

    A prompt that silently truncates loses the FRONT of the JSON -- it would
    delete exactly the primacy signal this study measures and return a clean
    recency curve that looks like a result.
    """
    budget = max_ctx - CTX_HEADROOM
    for n in candidates:
        rows = build_trials(trials_per_position=2, n_pairs=n)
        worst = max(len(tokenizer(r["prompt"])["input_ids"]) for r in rows)
        if worst < budget:
            return n, worst
    raise AssertionError(
        f"no candidate in {candidates} fits {budget} tokens; the ladder cannot "
        f"hold prompts constant across models and must be redesigned")


def assert_fits(tokenizer, n_pairs: int, max_ctx: int = 4096) -> int:
    """Re-assert the pinned N still fits, on EVERY model's tokenizer."""
    rows = build_trials(trials_per_position=2, n_pairs=n_pairs)
    worst = max(len(tokenizer(r["prompt"])["input_ids"]) for r in rows)
    assert worst < max_ctx - CTX_HEADROOM, (
        f"prompt is {worst} tokens against a {max_ctx} window; truncation would "
        f"remove the early keys and manufacture a recency result")
    return worst


def _selftest() -> None:
    rows = build_trials(trials_per_position=2)
    assert len(rows) == 2 * N_POSITIONS
    assert gold_positions()[0] == 0 and gold_positions()[-1] == N_PAIRS - 1
    for r in rows:
        assert r["gold_key"] in r["prompt"] and r["gold_value"] in r["prompt"]
        assert r["prompt"].count(r["gold_key"]) == 2  # once in JSON, once in Key:
    # paired design: same seed -> byte-identical prompts
    assert [r["prompt"] for r in build_trials(trials_per_position=2)] == \
           [r["prompt"] for r in rows], "trial set is not reproducible"
    # every model must get the SAME prompts, so no model arg exists in build_trials
    assert "model" not in build_trials.__code__.co_varnames

    # scorer
    k, v = rows[0]["gold_key"], rows[0]["gold_value"]
    assert score_one(f"The value is {v}", k, v)["correct"] is True
    assert score_one(f'"{k}": "{v}"', k, v)["correct"] is True, "key echo must be stripped"
    other = _uuid(random.Random(9))
    assert score_one(f"{other} but maybe {v}", k, v)["correct"] is False, \
        "first non-key candidate wins; a dump of guesses is not a retrieval"
    s = score_one("I cannot find that key.", k, v)
    assert s["scorable"] is False and s["correct"] is False
    assert score_one(f"{v} {other}", k, v)["n_candidates"] == 2

    # curve + classify
    def mk(d):
        return {p: a for p, a in d.items()}
    u = mk({0: .9, 10: .4, 20: .35, 30: .4, 40: .5, 49: .85})
    assert classify(u)["verdict"] == "U_SHAPE"
    r_only = mk({0: .40, 10: .38, 20: .35, 30: .40, 40: .55, 49: .80})
    assert classify(r_only)["verdict"] == "RECENCY_ONLY"
    p_only = mk({0: .80, 10: .38, 20: .35, 30: .40, 40: .42, 49: .40})
    assert classify(p_only)["verdict"] == "PRIMACY_ONLY"
    assert classify(mk({0: .5, 10: .48, 20: .47, 30: .5, 49: .52}))["verdict"] == "FLAT"
    # a monotone rise must NOT read as recency-with-primacy
    assert classify(mk({0: .2, 10: .3, 20: .4, 30: .5, 40: .6, 49: .7}))["verdict"] \
        == "RECENCY_ONLY"

    def row(c, sf=1.0):
        return dict(curve=c, scorable_frac=sf)

    # anchor gate fires BEFORE any driver claim
    bad = evaluate({"llama2-7b": row(u), "llama2-13b": row(u)})
    assert bad["anchor_gate"]["status"] == "ANCHOR_FAILED"
    assert bad["driver"] is None, "driver claimed on an unvalidated instrument"

    ok = {"llama2-7b": row(r_only), "llama2-13b": row(u),
          "llama31-8b": row(p_only), "olmo3-7b": row(r_only)}
    e = evaluate(ok)
    assert e["anchor_gate"]["status"] == "ANCHOR_OK"
    assert e["driver"] == "TRAINING_COMPUTE", e

    # all three small rows recency -> parameter count
    e2 = evaluate({"llama2-7b": row(r_only), "llama2-13b": row(u),
                   "llama31-8b": row(r_only), "olmo3-7b": row(r_only)})
    assert e2["driver"] == "PARAMETER_COUNT", e2

    # void precedes shape, on both void paths -- and a void row must BLOCK the
    # driver claim rather than counting as "no primacy here"
    e3 = evaluate({**ok, "olmo3-7b": row(u, sf=0.30)})
    assert e3["per_model"]["olmo3-7b"]["verdict"] == "V_UNSCORABLE"
    assert e3["driver"] is None, e3
    assert any("olmo3-7b" in r for r in e3["reasons"]), e3["reasons"]
    e4 = evaluate({**ok, "olmo3-7b": row(mk({0: .05, 10: .03, 20: .02, 49: .04}))})
    assert e4["per_model"]["olmo3-7b"]["verdict"] == "V_TASK_UNREACHABLE"
    assert e4["driver"] is None, e4
    # THE REGRESSION THAT MOTIVATED VOID_VERDICTS: three unreadable models must
    # never return PARAMETER_COUNT ("no model showed primacy" from no data)
    allvoid = evaluate({"llama2-7b": row(r_only), "llama2-13b": row(u),
                        "llama31-8b": row(p_only, sf=0.1), "olmo3-7b": row(u, sf=0.1)})
    assert allvoid["driver"] is None, \
        f"absence of measurement read as measurement of absence: {allvoid}"

    # incomplete ladder cannot yield a driver
    assert evaluate({"llama2-7b": row(r_only)})["driver"] is None

    # N-selection rule, against a stub tokenizer with a known chars-per-token
    class Stub:
        def __init__(self, cpt): self.cpt = cpt
        def __call__(self, s): return {"input_ids": [0] * (len(s) // self.cpt)}
    n, w = select_n_pairs(Stub(3))          # roomy -> largest candidate
    assert n == 50 and w < 4096 - CTX_HEADROOM, (n, w)
    n, w = select_n_pairs(Stub(1))          # 1 char/token -> must step down
    assert n < 50 and w < 4096 - CTX_HEADROOM, (n, w)
    assert build_trials(trials_per_position=1, n_pairs=n)[0]["n_pairs"] == n
    assert gold_positions(n)[-1] == n - 1, "positions must track the chosen N"
    try:
        select_n_pairs(Stub(1), max_ctx=200)
        raise SystemExit("select_n_pairs accepted an impossible budget")
    except AssertionError:
        pass

    s = sensitivity(ok)
    assert s["base"] == ("ANCHOR_OK", "TRAINING_COMPUTE")
    print(f"self-test OK: {len(rows)} trial rows, scorer key-echo + multi-candidate, "
          f"5 shape worlds, anchor gate, 2 void paths, driver logic")
    print(f"  positions: {gold_positions()}")
    print(f"  full run: {N_POSITIONS * TRIALS_PER_POSITION} prompts/model "
          f"x {len(LADDER)} models = {N_POSITIONS * TRIALS_PER_POSITION * len(LADDER)}")
    print(f"  sensitivity fragile: {s['fragile']}")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv: _selftest()
    else: print(__doc__)
