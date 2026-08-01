"""Wide-catalog adjudicator. Prereg: docs/wide-catalog-prereg-2026-07-31.md.

Verdict is whatever evaluate() returns. W0 (void) first; the position verdict is
INDEPENDENT of the gate branch. Run --self-test.
"""
from __future__ import annotations

BARS = dict(survives=0.10, dissolves=0.05, coverage=0.25,
            position=0.40, repro=0.95)
SENS = dict(survives=(0.05, 0.15), dissolves=(0.03, 0.08),
            coverage=(0.15, 0.35), position=(0.35, 0.45))

# Registered directional predictions, fixed before the K=4 data exists.
PREDICTED = {"llama": "P_PRIMACY", "olmo": "P_RECENCY"}


def evaluate(m: dict, bars: dict = BARS) -> dict:
    """m = dict(repro_frac, cells={cond: {cat: rate}}, G_k4, G_k2,
                position={pos: share}, k)"""
    if m.get("repro_frac", 1.0) < bars["repro"]:
        return dict(branch="W0_VOID", position=None,
                    reasons=[f"K=2 reproduction {m['repro_frac']:.3f} "
                             f"< {bars['repro']}; cross-K comparison void"])
    bad = [(c, v["other"]) for c, v in m["cells"].items()
           if v.get("other", 0.0) > bars["coverage"]]
    if bad:
        return dict(branch="W0_VOID", position=None,
                    reasons=[f"{c} other {o:.3f} > {bars['coverage']}" for c, o in bad])

    g4, g2 = m["G_k4"], m.get("G_k2")
    pos = _position(m, bars)
    if g4 >= bars["survives"]:
        b = dict(branch="W1_GATE_SURVIVES")
    elif g4 < bars["dissolves"]:
        b = dict(branch="W2_GATE_DISSOLVES")
    else:
        b = dict(branch="W3_INTERMEDIATE")
    b["reasons"] = [f"G(K=4)={g4:+.3f}"] + (
        [f"G(K=2)={g2:+.3f}, dG={g4-g2:+.3f}"] if g2 is not None else [])
    b["position"] = pos
    return b


def _position(m: dict, bars: dict) -> dict | None:
    p = m.get("position")
    if not p:
        return None
    k = m.get("k", max(p))
    top = max(p, key=lambda i: p[i])
    share = p[top]
    if share < bars["position"]:
        v = "P_NONE"
    elif top == 1:
        v = "P_PRIMACY"
    elif top == k:
        v = "P_RECENCY"
    else:
        v = "P_ABSOLUTE_OTHER"
    out = dict(verdict=v, top_position=top, top_share=round(share, 4),
               uniform=round(1.0 / k, 4), shares={i: round(s, 4) for i, s in sorted(p.items())})
    pred = PREDICTED.get(str(m.get("model", "")).lower())
    if pred:
        out["predicted"] = pred
        out["prediction_confirmed"] = (v == pred)
    return out


def sensitivity(m: dict) -> dict:
    out = {}
    for key, vals in SENS.items():
        for v in vals:
            b = dict(BARS); b[key] = v
            r = evaluate(m, b)
            out[f"{key}={v}"] = (r["branch"], (r["position"] or {}).get("verdict"))
    base = evaluate(m)
    bk = (base["branch"], (base["position"] or {}).get("verdict"))
    return dict(base=bk, grid=out, fragile=any(v != bk for v in out.values()))


def _mk(g4, pos=None, other=0.02, repro=1.0, model="llama", g2=0.19):
    pos = pos or {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
    cells = {c: dict(comply=0.8, substitute=0.15, fabricate=0.02,
                     decline=0.01, other=other) for c in ("conflict", "reachable")}
    return dict(repro_frac=repro, cells=cells, G_k4=g4, G_k2=g2,
                position=pos, k=4, model=model)


def _selftest() -> None:
    assert evaluate(_mk(0.25))["branch"] == "W1_GATE_SURVIVES"
    assert evaluate(_mk(0.01))["branch"] == "W2_GATE_DISSOLVES"
    assert evaluate(_mk(0.07))["branch"] == "W3_INTERMEDIATE"
    # W0 precedes a healthy gate, on either void condition
    assert evaluate(_mk(0.30, other=0.40))["branch"] == "W0_VOID"
    assert evaluate(_mk(0.30, repro=0.50))["branch"] == "W0_VOID"
    # position verdicts, independent of the gate branch
    r = evaluate(_mk(0.25, pos={1: 0.55, 2: 0.20, 3: 0.15, 4: 0.10}, model="llama"))
    assert r["branch"] == "W1_GATE_SURVIVES"
    assert r["position"]["verdict"] == "P_PRIMACY"
    assert r["position"]["prediction_confirmed"] is True
    r = evaluate(_mk(0.01, pos={1: 0.10, 2: 0.15, 3: 0.20, 4: 0.55}, model="olmo"))
    assert r["branch"] == "W2_GATE_DISSOLVES"       # gate can die while position holds
    assert r["position"]["verdict"] == "P_RECENCY"
    assert r["position"]["prediction_confirmed"] is True
    # a MISMATCH must be recorded as such, not quietly accepted
    r = evaluate(_mk(0.25, pos={1: 0.55, 2: 0.20, 3: 0.15, 4: 0.10}, model="olmo"))
    assert r["position"]["prediction_confirmed"] is False
    # middle-position preference is its own verdict, not forced into primacy/recency
    r = evaluate(_mk(0.25, pos={1: 0.15, 2: 0.50, 3: 0.20, 4: 0.15}))
    assert r["position"]["verdict"] == "P_ABSOLUTE_OTHER"
    # flat -> none
    assert evaluate(_mk(0.25))["position"]["verdict"] == "P_NONE"
    # fragility: 0.38 sits between the 0.35 and 0.45 position bars
    assert sensitivity(_mk(0.25, pos={1: 0.38, 2: 0.24, 3: 0.20, 4: 0.18}))["fragile"] is True
    print("self-test OK: 3 gate worlds, 2 void paths, 5 position worlds, "
          "prediction match+mismatch, fragility")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv: _selftest()
    else: print(__doc__)
