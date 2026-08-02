"""Aggregate the four row files into the ladder verdict. No GPU, no network.

    python3 position_ladder_aggregate.py row_*.json

Rows come from `position_ladder_colab.ipynb` cell [9], one per model, either
downloaded from the HF dataset repo or pasted out of the notebook output.

The verdict is `position_ladder.evaluate()`. This script's only jobs are to
assemble the input and to refuse inputs that cannot be compared.
"""
from __future__ import annotations

import json
import sys

import position_ladder as PL
from measure_primitives import fingerprint


def load(paths: list[str]) -> dict:
    rows, meta = {}, {}
    for p in paths:
        d = json.load(open(p))
        assert d.get("protocol") == "POSITION_LADDER_V1", f"{p}: wrong protocol"
        k = d["config"]["model_key"]
        assert k not in rows, f"duplicate row for {k}"
        rows[k] = dict(curve={int(i): v for i, v in d["result"]["curve"].items()},
                       scorable_frac=d["result"]["scorable_frac"])
        meta[k] = d["config"]
    return rows, meta


def assert_comparable(meta: dict) -> None:
    """Rows that differ in what they were measured with are not a ladder.

    N and the adjudicator hash must be identical -- a curve at N=50 and a curve
    at N=30 are different tasks, and comparing them would be the cross-K
    comparison the wide-catalog prereg voided itself over. Dtype must also match:
    the whole point of running every row on A100 was to hold it constant.
    """
    for field in ("n_pairs", "pl_sha", "dtype", "seed", "trials_per_position"):
        vals = {k: m[field] for k, m in meta.items()}
        assert len(set(vals.values())) == 1, \
            f"rows are not comparable -- {field} differs: {vals}"
    for k, m in meta.items():
        assert m["positions"] == PL.gold_positions(m["n_pairs"]), \
            f"{k}: positions do not match its N"


def main(paths: list[str]) -> int:
    rows, meta = load(paths)
    assert_comparable(meta)
    result = PL.evaluate(rows)
    sens = PL.sensitivity(rows)

    print("=" * 78)
    for k in ("llama2-7b", "llama2-13b", "llama31-8b", "olmo3-7b"):
        if k not in result["per_model"]:
            print(f"  {k:12s}  ABSENT")
            continue
        v = result["per_model"][k]
        want = PL.ANCHOR_PREDICTION.get(k)
        tag = "" if not want else (
            f"   [anchor: predicted {want}, "
            f"{'MATCH' if v['verdict'] == want else 'MISMATCH'}]")
        print(f"  {k:12s}  {v['verdict']:20s} "
              f"prim {v.get('primacy_index', float('nan')):+.3f}  "
              f"rec {v.get('recency_index', float('nan')):+.3f}{tag}")

    print("=" * 78)
    print("anchor gate:", result["anchor_gate"]["status"])
    for r in result["anchor_gate"]["reasons"]:
        print("   ", r)
    print("driver     :", result["driver"])
    for r in result.get("reasons", []):
        print("   ", r)
    print("fragile    :", sens["fragile"])
    if sens["fragile"]:
        for k, v in sens["grid"].items():
            if v != sens["base"]:
                print(f"     FLIP {k} -> {v}")

    out = dict(protocol="POSITION_LADDER_AGGREGATE_V1", rows=sorted(rows),
               n_pairs=next(iter(meta.values()))["n_pairs"],
               result=result, sensitivity=sens,
               fingerprint=fingerprint(__file__, PL.__file__))
    open("position_ladder_verdict.json", "w").write(json.dumps(out, indent=1))
    print("\nwrote position_ladder_verdict.json")
    return 0


def _selftest() -> None:
    """Comparability guard must FIRE, or it is decoration."""
    base = dict(n_pairs=50, pl_sha="x", dtype="torch.bfloat16", seed=1101,
                trials_per_position=200, positions=PL.gold_positions(50))
    good = {"a": dict(base), "b": dict(base)}
    assert_comparable(good)
    for field, bad in (("n_pairs", 30), ("dtype", "torch.float16"), ("pl_sha", "y")):
        m = {"a": dict(base), "b": {**base, field: bad}}
        if field == "n_pairs":
            m["b"]["positions"] = PL.gold_positions(30)
        try:
            assert_comparable(m)
            raise SystemExit(f"guard did not fire on differing {field}")
        except AssertionError:
            pass
    # positions inconsistent with N must also fire
    m = {"a": {**base, "positions": [0, 1, 2]}}
    try:
        assert_comparable(m)
        raise SystemExit("guard did not fire on positions/N mismatch")
    except AssertionError:
        pass
    print("self-test OK: comparability guard fires on n_pairs, dtype, pl_sha, positions")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--self-test" in sys.argv:
        _selftest()
    elif args:
        raise SystemExit(main(args))
    else:
        print(__doc__)
