"""Aggregate v2 row files into the ladder verdict. No GPU, no network.

    python3 position_ladder_qa_aggregate.py row_*.json
"""
from __future__ import annotations

import json
import sys

import position_ladder_qa as Q
from measure_primitives import fingerprint


def load(paths: list[str]) -> tuple[dict, dict]:
    rows, meta = {}, {}
    for p in paths:
        d = json.load(open(p))
        assert d.get("protocol") == "POSITION_LADDER_QA_V2", f"{p}: wrong protocol"
        k = d["model_key"]
        assert k not in rows, f"duplicate row for {k}"
        r = d["result"]
        rows[k] = dict(curve={int(i): v for i, v in r["curve"].items()},
                       n_by_pos={int(i): v for i, v in r["n_by_pos"].items()})
        meta[k] = d
    return rows, meta


def assert_comparable(meta: dict) -> None:
    """Rows measured with different instruments are not a ladder."""
    for field in ("litm_commit", "plq_sha", "temperature", "top_p",
                  "max_new_tokens", "max_prompt_length"):
        vals = {k: m[field] for k, m in meta.items()}
        assert len(set(map(str, vals.values()))) == 1, \
            f"rows not comparable -- {field} differs: {vals}"
    # vllm version may differ between sessions; surfaced, not fatal, because
    # decoding is greedy -- but it must be VISIBLE in the verdict artifact.
    vv = {k: m["vllm_version"] for k, m in meta.items()}
    if len(set(vv.values())) > 1:
        print(f"  NOTE: vllm versions differ across rows: {vv}")


def main(paths: list[str]) -> int:
    rows, meta = load(paths)
    assert_comparable(meta)
    result = Q.evaluate(rows)
    sens = Q.sensitivity(rows)

    print("=" * 78)
    for k in Q.ROWS:
        if k not in result["per_model"]:
            print(f"  {k:16s}  ABSENT"); continue
        v = result["per_model"][k]
        if v["verdict"] != "OK":
            print(f"  {k:16s}  {v['verdict']}  {v['reasons']}"); continue
        tgt = ""
        if k in Q.FIG16:
            t = Q.indices(Q.FIG16[k])
            tgt = f"   [fig16: prim {t['primacy']:+.3f} rec {t['recency']:+.3f}]"
        print(f"  {k:16s}  prim {v['primacy']:+.3f}  rec {v['recency']:+.3f}  "
              f"call {v['primacy_call']:13s}{tgt}")
    print("=" * 78)
    print("anchor gate:", result["anchor_gate"]["status"])
    for r in result["anchor_gate"]["reasons"]:
        print("   ", r)
    print("driver     :", result.get("driver"))
    for r in result.get("reasons", []):
        print("   ", r)
    print("fragile    :", sens["fragile"])
    if sens["fragile"]:
        for k, v in sens["grid"].items():
            if v != sens["base"]:
                print(f"     FLIP {k} -> {v}")

    out = dict(protocol="POSITION_LADDER_QA_AGGREGATE_V2", rows=sorted(rows),
               result=result, sensitivity=sens,
               vllm_versions={k: m["vllm_version"] for k, m in meta.items()},
               fingerprint=fingerprint(__file__, Q.__file__))
    open("position_ladder_qa_verdict.json", "w").write(json.dumps(out, indent=1))
    print("\nwrote position_ladder_qa_verdict.json")
    return 0


def _selftest() -> None:
    base = dict(protocol="POSITION_LADDER_QA_V2", litm_commit="c", plq_sha="s",
                temperature=0.0, top_p=1.0, max_new_tokens=100,
                max_prompt_length=4096, vllm_version="9.9")
    m = {"a": dict(base), "b": {**base, "litm_commit": "d"}}
    try:
        assert_comparable(m)
        raise SystemExit("guard did not fire on differing litm_commit")
    except AssertionError:
        pass
    assert_comparable({"a": dict(base), "b": {**base, "vllm_version": "9.8"}})
    print("self-test OK: instrument fields fatal, vllm version surfaced only")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--self-test" in sys.argv:
        _selftest()
    elif args:
        raise SystemExit(main(args))
    else:
        print(__doc__)
