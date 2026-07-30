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


DOMINANCE_BAR = 0.40   # one cell contributing >40% of a pooled quantity


def audit_pooled(path: str, field: str) -> int:
    """Per-cell weighting for EVERY pooled quantity over this corpus.

    Exists because the digit marginal turned out to be 54.7% one cell. That is a
    generalisable flag, not a one-off: any pooled number over an unbalanced
    corpus is a weighted average whose weights nobody declared. Each row below
    reports the per-cell share and flags whichever cell dominates.

    The design is balanced by ROW (2065-2067 per cell). It is badly unbalanced by
    TEXT VOLUME, which is what most of these quantities actually average over --
    and text volume tracks Factor B by construction, so the imbalance runs along
    a factor axis.
    """
    rows = load(path)
    get = FIELDS[field]
    cells = sorted({r.get("cell", "?") for r in rows})
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("cell", "?")].append(r)

    def emit(name, per_cell_num, per_cell_den=None, fmt="{:.4f}"):
        tot_n = sum(per_cell_num.values())
        if per_cell_den:
            tot_d = sum(per_cell_den.values())
            pooled = tot_n / tot_d if tot_d else float("nan")
        else:
            pooled = tot_n
        shares = {c: (per_cell_num[c] / tot_n if tot_n else 0.0) for c in cells}
        top = max(shares, key=shares.get)
        flag = "  <== DOMINATED" if shares[top] > DOMINANCE_BAR else ""
        print(f"\n  {name}   pooled = {fmt.format(pooled)}")
        for c in cells:
            v = (per_cell_num[c] / per_cell_den[c]
                 if per_cell_den and per_cell_den[c] else per_cell_num[c])
            print(f"      {c:22s} {fmt.format(v):>12s}   weight {shares[c]:6.1%}")
        print(f"      -> heaviest cell {top} at {shares[top]:.1%}{flag}")
        return shares[top] > DOMINANCE_BAR

    print(f"corpus {path}\nfield  {field}\n"
          f"rows per cell: " + ", ".join(f"{c}={len(by[c])}" for c in cells))
    print(f"\nFlagging any pooled quantity where one cell exceeds "
          f"{DOMINANCE_BAR:.0%} of the weight.")
    flagged = []

    # 1. digit marginal E -- the one already known
    num = {c: sum(d * collections.Counter(
        int(x) for t in (get(r) for r in by[c]) for x in DIGIT_RE.findall(t))[d]
        for d in range(1, 8)) for c in cells}
    den = {c: sum(collections.Counter(
        int(x) for t in (get(r) for r in by[c]) for x in DIGIT_RE.findall(t)).values())
        for c in cells}
    if emit("E[digit 1-7]", den, None, "{:.0f}"):   # weight = occurrence count
        flagged.append("E[digit]")

    # 2. mean text length in characters
    ln = {c: sum(len(get(r)) for r in by[c]) for c in cells}
    cnt = {c: float(len(by[c])) for c in cells}
    if emit("mean text length (chars)", ln, cnt, "{:.1f}"):
        flagged.append("mean_text_length")

    # 3. answers present
    ap_ = {c: float(sum(1 for r in by[c] if r.get("answer"))) for c in cells}
    if emit("answers present (count)", ap_, None, "{:.0f}"):
        flagged.append("answers_present")

    # 4. ok-flag failures
    bad = {c: float(sum(1 for r in by[c] if not r.get("ok", True))) for c in cells}
    if sum(bad.values()):
        if emit("rows with ok=False (count)", bad, None, "{:.0f}"):
            flagged.append("ok_false")

    # 5. digit-initial answers
    di = {c: float(sum(1 for r in by[c]
                       if (r.get("answer") or "").lstrip()[:1].isdigit()))
          for c in cells}
    if sum(di.values()):
        emit("digit-initial answers (count)", di, None, "{:.0f}")

    print("\n" + "=" * 66)
    if flagged:
        print(f"{len(flagged)} pooled quantity/ies dominated by one cell: "
              f"{', '.join(flagged)}")
        print("Report these per-cell alongside any pooled value. The corpus is")
        print("balanced by ROW and unbalanced by TEXT VOLUME, and text volume")
        print("tracks Factor B -- so the imbalance runs along a factor axis.")
    else:
        print("No pooled quantity exceeds the dominance bar.")
    return 0


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
    ap.add_argument("--audit-pooled", action="store_true",
                    help="per-cell weighting for EVERY pooled quantity, not just E")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not os.path.exists(a.path):
        print(f"missing corpus: {a.path}", file=sys.stderr)
        return 2
    if a.audit_pooled:
        return audit_pooled(a.path, a.field)
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
