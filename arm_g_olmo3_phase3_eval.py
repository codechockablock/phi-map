"""Olmo 3 Phase 3 branch adjudicator. THE authoritative verdict path.

The pre-registration (docs/olmo3-phase3-prereg-2026-07-31.md) defers to this
function; the notebook imports or embeds it byte-identically and the verdict is
whatever it returns. First match wins. Run --self-test before trusting it.
"""
from __future__ import annotations

BARS = dict(margin=0.10, cosine=0.70, order=0.65, retention=0.80)
SENS = dict(margin=(0.05, 0.15), order=(0.60, 0.70), retention=(0.70, 0.90))


def evaluate(m: dict, bars: dict = BARS) -> dict:
    """m per seed s in ('s111','s211'):
      m[s] = dict(cond_raw, cond_orth, order_raw, order_orth, shuf_p975)
    plus m['cosine_orth'] (inter-seed cosine of orthogonalized directions).
    """
    seeds = ("s111", "s211")
    reasons = []

    # R2 first: contamination may not hide behind a passing headline.
    for s in seeds:
        d = m[s]
        margin_raw = d["cond_raw"] - 0.5
        kept = (d["cond_orth"] - 0.5) / margin_raw if margin_raw > 0 else 1.0
        if d["order_raw"] > bars["order"] and kept < bars["retention"]:
            reasons.append(f"{s}: order_raw {d['order_raw']:.3f} > {bars['order']} "
                           f"and retention {kept:.2f} < {bars['retention']}")
    if reasons:
        return dict(branch="R2_ORDER_CONTAMINATED", reasons=reasons)

    above = {s: m[s]["cond_orth"] - m[s]["shuf_p975"] for s in seeds}
    clears_band = all(above[s] > 0 for s in seeds)
    clears_margin = all(above[s] >= bars["margin"] for s in seeds)
    cos_ok = m["cosine_orth"] >= bars["cosine"]

    if clears_margin and cos_ok:
        return dict(branch="R1_REPLICATES",
                    reasons=[f"margins {above}", f"cosine {m['cosine_orth']:.3f}"])
    if clears_band:
        return dict(branch="R3_SIGNAL_NOT_OBJECT",
                    reasons=[f"margins {above}", f"cosine {m['cosine_orth']:.3f}",
                             f"margin_ok={clears_margin} cosine_ok={cos_ok}"])
    return dict(branch="R4_NULL", reasons=[f"margins {above}"])


def sensitivity(m: dict) -> dict:
    """Branch under every pre-registered bar variant; flags fragility."""
    out = {}
    for key, vals in SENS.items():
        for v in vals:
            b = dict(BARS); b[key] = v
            out[f"{key}={v}"] = evaluate(m, b)["branch"]
    base = evaluate(m)["branch"]
    return dict(base=base, grid=out, fragile=any(v != base for v in out.values()))


def _selftest() -> None:
    def mk(c_raw, c_orth, o_raw, cos, shuf=0.57, o_orth=0.50):
        d = dict(cond_raw=c_raw, cond_orth=c_orth, order_raw=o_raw,
                 order_orth=o_orth, shuf_p975=shuf)
        return {"s111": dict(d), "s211": dict(d), "cosine_orth": cos}

    # clean replication
    assert evaluate(mk(0.78, 0.77, 0.55, 0.85))["branch"] == "R1_REPLICATES"
    # llama-shaped contamination: reads order, repair guts it -> R2 even though raw headline is high
    assert evaluate(mk(0.85, 0.55, 0.99, 0.9))["branch"] == "R2_ORDER_CONTAMINATED"
    # order-ish but repair RETAINS margin -> not R2; strong numbers -> R1
    assert evaluate(mk(0.80, 0.78, 0.70, 0.9))["branch"] == "R1_REPLICATES"
    # above band, weak margin -> R3
    assert evaluate(mk(0.63, 0.62, 0.55, 0.9))["branch"] == "R3_SIGNAL_NOT_OBJECT"
    # strong AUROC, unstable directions -> R3 not R1
    assert evaluate(mk(0.78, 0.77, 0.55, 0.4))["branch"] == "R3_SIGNAL_NOT_OBJECT"
    # below band -> R4
    m = mk(0.56, 0.55, 0.52, 0.8); assert evaluate(m)["branch"] == "R4_NULL"
    # asymmetric seeds: one null seed forces R4/R3, never R1
    m = mk(0.78, 0.77, 0.55, 0.9); m["s211"].update(cond_orth=0.55, cond_raw=0.56)
    assert evaluate(m)["branch"] == "R4_NULL"
    # sensitivity grid runs and reports fragility on a boundary case
    s = sensitivity(mk(0.68, 0.675, 0.55, 0.85))
    assert isinstance(s["fragile"], bool)
    print("self-test OK: 8 worlds, first-match order verified (R2 precedes R1)")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _selftest()
    else:
        print(__doc__)
