"""Olmo 3 behavioural crossover adjudicator. Mirrors RESEARCH_ARC §14's own
pre-stated accounts; first match is NOT used here — the two verdicts are
orthogonal and BOTH are reported, exactly as §14 reported them.

  POSITION_GATED : conflict decisions reverse under the catalog-line swap.
  SCOPE_SURVIVES : the condition margin contrast is positive in BOTH orders.

Llama-3.1-8B produced both simultaneously (64/64 reversal AND condition main
+4.81, both-order contrasts positive). They are independent claims.
"""
from __future__ import annotations

BARS = dict(reversal=0.90, coherence=0.90)
SENS = dict(reversal=(0.80, 0.95))


def evaluate(m: dict, bars: dict = BARS) -> dict:
    """m: coherence_frac, reversal_frac (conflict pairs whose decision flips
    when only catalog order changes), n_conflict_pairs,
    cond_contrast_inside_ci, cond_contrast_outside_ci (each (lo, hi),
    scenario-bootstrap, margin units)."""
    out = {}
    if m["coherence_frac"] < bars["coherence"]:
        return dict(verdicts=["DISQUALIFIED_PROTOCOL"], detail=(
            f"top-1 is an action token on only {m['coherence_frac']:.2f} "
            f"of rows; the A/B protocol did not transfer -- uninformative, "
            f"not a null"))
    out["position_gated"] = m["reversal_frac"] >= bars["reversal"]
    ins, outs = m["cond_contrast_inside_ci"], m["cond_contrast_outside_ci"]
    out["scope_survives"] = ins[0] > 0 and outs[0] > 0
    out["scope_reverses"] = ins[1] < 0 or outs[1] < 0
    v = []
    v.append("POSITION_GATED" if out["position_gated"] else "NOT_POSITION_GATED")
    if out["scope_survives"]:
        v.append("SCOPE_SURVIVES")
    elif out["scope_reverses"]:
        v.append("SCOPE_REVERSES_IN_AN_ORDER")
    else:
        v.append("SCOPE_INDETERMINATE")
    return dict(verdicts=v, flags=out,
                detail=f"reversal {m['reversal_frac']:.3f} of "
                       f"{m['n_conflict_pairs']} pairs; contrasts "
                       f"inside{ins} outside{outs}")


def sensitivity(m: dict) -> dict:
    outs = {}
    for v in SENS["reversal"]:
        b = dict(BARS); b["reversal"] = v
        outs[f"reversal={v}"] = evaluate(m, b)["verdicts"]
    base = evaluate(m)["verdicts"]
    return dict(base=base, grid=outs,
                fragile=any(o != base for o in outs.values()))


def _selftest() -> None:
    ok = dict(coherence_frac=0.99, n_conflict_pairs=64)
    # llama-shaped world: total reversal AND scope surviving on the margin
    m = dict(ok, reversal_frac=1.0, cond_contrast_inside_ci=(4.0, 5.0),
             cond_contrast_outside_ci=(2.0, 3.5))
    assert evaluate(m)["verdicts"] == ["POSITION_GATED", "SCOPE_SURVIVES"]
    # scope-only world: no reversal, contrast holds
    m = dict(ok, reversal_frac=0.05, cond_contrast_inside_ci=(1.0, 2.0),
             cond_contrast_outside_ci=(0.5, 1.5))
    assert evaluate(m)["verdicts"] == ["NOT_POSITION_GATED", "SCOPE_SURVIVES"]
    # pure-position world: reversal, contrast flips sign in one order
    m = dict(ok, reversal_frac=1.0, cond_contrast_inside_ci=(1.0, 2.0),
             cond_contrast_outside_ci=(-2.0, -0.5))
    assert evaluate(m)["verdicts"] == ["POSITION_GATED", "SCOPE_REVERSES_IN_AN_ORDER"]
    # incoherent protocol -> disqualified, not null
    m = dict(coherence_frac=0.4, reversal_frac=1.0, n_conflict_pairs=64,
             cond_contrast_inside_ci=(1, 2), cond_contrast_outside_ci=(1, 2))
    assert evaluate(m)["verdicts"] == ["DISQUALIFIED_PROTOCOL"]
    # straddling-zero CI -> indeterminate, never silently 'survives'
    m = dict(ok, reversal_frac=0.5, cond_contrast_inside_ci=(-0.2, 1.0),
             cond_contrast_outside_ci=(0.1, 1.0))
    assert evaluate(m)["verdicts"] == ["NOT_POSITION_GATED", "SCOPE_INDETERMINATE"]
    s = sensitivity(dict(ok, reversal_frac=0.85,
                         cond_contrast_inside_ci=(1, 2),
                         cond_contrast_outside_ci=(1, 2)))
    assert s["fragile"] is True   # 0.85 sits between the 0.80 and 0.95 bars
    print("self-test OK: 5 worlds + fragility case")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv: _selftest()
    else: print(__doc__)
