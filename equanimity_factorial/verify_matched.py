"""Verify the calibrated generation on fresh output.

Two things this is careful about.

**Version purity.** A prompt is used only if all four of its cells come from the
same `band_version`. Mixing a v3 equanimity cell with a v1 neutral cell inside
one prompt would compare a calibrated generation against an uncalibrated one and
call the difference a stance effect -- manufacturing exactly the confound the
calibration is meant to remove.

**The metric.** Mean realised token difference is primary; Cohen's d is secondary.
`d`'s denominator is under the generator's control, which is how Arm T improved
`d` while making the real coupling worse. Reporting d alone would leave that door
open, so the token difference leads and d follows.

The calibration constants were estimated from Arm T and v1 output. Everything
scored here is a fresh generation, so the numbers judging the fix are not the
numbers that produced it.

    python3 verify_matched.py --version v3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from gate import DATA, TOKENIZER_REPO

CELLS = ["equanimity-terse", "equanimity-verbose",
         "neutral-terse", "neutral-verbose"]


def load_version(version: str) -> list[dict]:
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    rows = [r for r in rows
            if r.get("ok") and str(r.get("band_version", "v1")) == version]
    per = defaultdict(dict)
    for r in rows:
        per[r["prompt_id"]][r["cell"]] = r
    keep = [p for p, cells in per.items() if set(CELLS) <= set(cells)]
    return [per[p][c] for p in sorted(keep) for c in CELLS]


def tokenize(rows: list[dict]) -> list[dict]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    for r in rows:
        r["tok_reasoning"] = len(tok.encode(r["reasoning"], add_special_tokens=False))
        r["tok_answer"] = len(tok.encode(r["answer"], add_special_tokens=False))
        r["tok_total"] = r["tok_reasoning"] + r["tok_answer"]
    return rows


def contrast(rows: list[dict], field: str, verbosity: str) -> dict:
    idx = {(r["prompt_id"], r["content"], r["verbosity"]): r for r in rows}
    pids = sorted({r["prompt_id"] for r in rows})
    eq, ne = [], []
    for p in pids:
        ke, kn = (p, "equanimity", verbosity), (p, "neutral", verbosity)
        if ke in idx and kn in idx:
            eq.append(idx[ke][field]); ne.append(idx[kn][field])
    eq, ne = np.array(eq, float), np.array(ne, float)
    if len(eq) < 4:
        return {"n": len(eq), "error": "insufficient"}
    diff = eq - ne
    sd = float(np.sqrt((eq.var(ddof=1) + ne.var(ddof=1)) / 2))
    se_tok = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    t, p = stats.ttest_rel(eq, ne)
    d = float(diff.mean() / sd) if sd > 0 else 0.0
    se_d = se_tok / sd if sd > 0 else float("inf")
    return dict(n=len(eq), mean_eq=float(eq.mean()), mean_neu=float(ne.mean()),
                mean_diff_tok=float(diff.mean()), se_diff_tok=se_tok,
                ci90_tok=[float(diff.mean() - 1.65 * se_tok),
                          float(diff.mean() + 1.65 * se_tok)],
                p=float(p), d=d, se_d=se_d,
                ci90_d=[d - 1.65 * se_d, d + 1.65 * se_d],
                sd_pooled=sd,
                equivalent=bool(abs(d - 1.65 * se_d) < 0.2 and abs(d + 1.65 * se_d) < 0.2))


def report(version: str) -> dict:
    rows = tokenize(load_version(version))
    n_prompts = len({r["prompt_id"] for r in rows})
    print("=" * 78)
    print(f"CALIBRATED GENERATION -- {version}, {n_prompts} version-pure prompts")
    print("=" * 78)
    print("\nPRIMARY: mean realised token difference (equanimity - neutral).")
    print("Cannot be gamed by widening the band; d can.\n")
    print(f"{'field':16s}{'verb':9s}{'eq':>7s}{'neu':>7s}{'diff':>8s}"
          f"{'CI90 (tok)':>18s}{'p':>8s}{'d':>8s}  equiv")
    print("-" * 90)
    out = {}
    for field in ("tok_reasoning", "tok_answer", "tok_total"):
        for verb in ("terse", "verbose"):
            c = contrast(rows, field, verb)
            out[f"{field}|{verb}"] = c
            if "error" in c:
                print(f"{field:16s}{verb:9s} {c}")
                continue
            ci = f"[{c['ci90_tok'][0]:+.2f},{c['ci90_tok'][1]:+.2f}]"
            print(f"{field:16s}{verb:9s}{c['mean_eq']:7.1f}{c['mean_neu']:7.1f}"
                  f"{c['mean_diff_tok']:+8.2f}{ci:>18s}{c['p']:>8.3f}"
                  f"{c['d']:+8.3f}  {'YES' if c['equivalent'] else 'no'}")

    print("\nLeakage relative to the Factor B manipulation:")
    rt = contrast(rows, "tok_reasoning", "terse")
    rv = contrast(rows, "tok_reasoning", "verbose")
    if "error" not in rt and "error" not in rv:
        B = ((rv["mean_eq"] + rv["mean_neu"]) - (rt["mean_eq"] + rt["mean_neu"])) / 2
        for name, c in (("terse", rt), ("verbose", rv)):
            print(f"    reasoning {name:8s}: {abs(c['mean_diff_tok']):.2f} tok "
                  f"= {100*abs(c['mean_diff_tok'])/B:.2f}% of B ({B:.0f} tok)")
        out["B_manipulation_tok"] = float(B)

    print("\nFactor B still separates strongly (the manipulation must survive):")
    for stance in ("equanimity", "neutral"):
        sub = [r for r in rows if r["content"] == stance]
        t = np.array([r["tok_reasoning"] for r in sub if r["verbosity"] == "terse"], float)
        v = np.array([r["tok_reasoning"] for r in sub if r["verbosity"] == "verbose"], float)
        sd = float(np.sqrt((t.var(ddof=1) + v.var(ddof=1)) / 2))
        print(f"    {stance:11s} terse {t.mean():6.1f} -> verbose {v.mean():6.1f}"
              f"   d={(v.mean()-t.mean())/sd:+.2f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3")
    a = ap.parse_args()
    res = report(a.version)
    Path(DATA).parent.joinpath(f"verify_{a.version}.json").write_text(
        json.dumps(res, indent=2))
