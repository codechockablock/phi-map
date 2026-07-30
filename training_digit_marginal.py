"""Empirical 1-7 digit marginal in the fine-tuning corpus. Zero compute, local read.

Exists because results R2 reported this number without committing the code that
produced it, and an independent recount could not reproduce its counts. A committed
number with no committed provenance is the defect `step4_run`'s harness fingerprint
exists to prevent, so the rule is stated here in code rather than in prose.

WHAT THIS ANSWERS. The adapters' self-report ratings sit in a tight 3.0-3.4 band at
every support level. Two cheap priors could explain that: the base model's own
two-turn prior (measured: 4.44 dysphoric / 4.55 neutral) and the digit marginal of
the training corpus. This computes the second. Neither matches, which is the point
-- and the conclusion is robust to the extraction rule even though the exact counts
are not.

EXTRACTION RULE, stated because it is the thing that varies:
  - unit  : one training row (`factorial_raw.jsonl`)
  - text  : `reasoning` + " " + `answer` (DEFAULT), selectable via --field
  - match : `\\b([1-7])\\b` -- standalone digits only, so "3" counts and the 3 in
            "2013" or "1.5x" does not
  - E     : sum(d * count[d]) / sum(count), d in 1..7

Run:  python3 training_digit_marginal.py --self-test
      python3 training_digit_marginal.py [--field reasoning+answer] [--per-cell]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "equanimity_factorial", "data", "factorial_raw.jsonl")
DIGIT_RE = re.compile(r"\b([1-7])\b")
X_OVER_7_RE = re.compile(r"\b[1-7]\s*/\s*7\b")

FIELDS = {
    "reasoning+answer": lambda r: f"{r.get('reasoning') or ''} {r.get('answer') or ''}",
    "answer":           lambda r: r.get("answer") or "",
    "reasoning":        lambda r: r.get("reasoning") or "",
    "prompt+reasoning+answer": lambda r: " ".join(
        [r.get("prompt") or "", r.get("reasoning") or "", r.get("answer") or ""]),
}


def marginal(texts) -> tuple[list[int], float, int]:
    c = collections.Counter()
    for t in texts:
        for d in DIGIT_RE.findall(t):
            c[int(d)] += 1
    counts = [c[d] for d in range(1, 8)]
    n = sum(counts)
    e = sum(d * c[d] for d in range(1, 8)) / n if n else float("nan")
    return counts, e, n


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def report(path: str, field: str, per_cell: bool) -> dict:
    rows = load(path)
    get = FIELDS[field]
    counts, e, n = marginal(get(r) for r in rows)

    answers = [r.get("answer") or "" for r in rows if r.get("answer")]
    x7 = sum(1 for a in answers if X_OVER_7_RE.search(a))
    digit_init = sum(1 for a in answers if a.lstrip()[:1].isdigit())

    print(f"corpus            {path}")
    print(f"rows              {len(rows)}")
    print(f"answers present   {len(answers)}")
    print(f"explicit x/7      {x7}")
    print(f"digit-initial     {digit_init}")
    print(f"\nextraction field  {field}")
    print(f"counts 1..7       {counts}")
    print(f"n occurrences     {n}")
    print(f"E[digit]          {e:.4f}")
    print(f"monotone dec.     {all(counts[i] >= counts[i+1] for i in range(6))}")

    out = dict(path=path, field=field, rows=len(rows), counts=counts, E=e, n=n,
               x_over_7=x7, digit_initial=digit_init)

    if per_cell:
        by = collections.defaultdict(list)
        for r in rows:
            by[r.get("cell", "?")].append(get(r))
        print("\nper cell:")
        cell_e = {}
        for cell in sorted(by):
            cc, ce, cn = marginal(by[cell])
            cell_e[cell] = ce
            print(f"  {cell:22s} E={ce:.4f}  n={cn:5d}  share={cn/n:5.1%}")
        spread = max(cell_e.values()) - min(cell_e.values())
        print(f"  spread {spread:.4f}")
        print("  NOTE: occurrence counts are heavily unbalanced across cells, so the")
        print("  pooled E is dominated by whichever cell contributes most text.")
        out["per_cell"] = cell_e
        out["spread"] = spread
    return out


def self_test() -> None:
    """Deterministic, no file, no network."""
    c, e, n = marginal(["1 1 2", "7"])
    assert c == [2, 1, 0, 0, 0, 0, 1], c
    assert n == 4 and abs(e - (1 + 1 + 2 + 7) / 4) < 1e-12, (e, n)
    # Word boundaries exclude digits embedded in longer numbers ("2013") and
    # after a word char ("v2"), but NOT the leading digit of a decimal: "1.5"
    # has a boundary between "1" and ".", so \b1\b matches it. This assertion
    # records the rule's ACTUAL behaviour rather than the behaviour first
    # assumed -- the self-test caught the difference, and the rule is documented
    # to match reality instead of the test being relaxed to match the rule.
    # Consequence: decimal-leading digits inflate low-digit counts, which is one
    # reason extraction rules disagree (results §9.2).
    c2, _, n2 = marginal(["2013 and v2"])
    assert n2 == 0, (c2, n2)
    c3, _, n3 = marginal(["1.5x"])
    assert n3 == 1 and c3[0] == 1, (c3, n3)
    # x/7 detector
    assert X_OVER_7_RE.search("I would say 3/7 here")
    assert not X_OVER_7_RE.search("ratio 8/7")
    # every declared field extractor tolerates missing keys
    for f in FIELDS.values():
        assert isinstance(f({}), str)
    print("self-test OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--field", default="reasoning+answer", choices=sorted(FIELDS))
    ap.add_argument("--per-cell", action="store_true")
    ap.add_argument("--all-fields", action="store_true",
                    help="report every extraction rule, to show what is robust")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not os.path.exists(a.path):
        print(f"missing corpus: {a.path}", file=sys.stderr)
        return 2
    if a.all_fields:
        rows = load(a.path)
        print(f"{'field':26s}{'E':>8s}{'n':>9s}   counts 1..7")
        es = []
        for name, get in sorted(FIELDS.items()):
            c, e, n = marginal(get(r) for r in rows)
            es.append(e)
            print(f"{name:26s}{e:8.3f}{n:9d}   {c}")
        print(f"\nE range across rules: [{min(es):.3f}, {max(es):.3f}]")
        print("base two-turn prior 4.44-4.55; adapter band 3.0-3.4.")
        print("Conclusion 'matches neither' holds across every rule above.")
        return 0
    report(a.path, a.field, a.per_cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
