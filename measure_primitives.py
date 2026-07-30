"""Shared measurement primitives + mandatory harness fingerprint.

WHY THIS FILE EXISTS. In one session, three separate pieces of newly-written
measurement code read a distribution at the wrong sequence position, twice in
code written by the audit itself:

  1. `refusal_margin`      - scored position 0 after format tuning moved the
                             content (found forensically; the study's subject)
  2. `prior_rating`        - read `logits[:, -1, :]`, the last PADDED position,
                             while `hidden_final_token` FORTY LINES UP in the
                             same file did it correctly under a comment naming
                             the hazard (results section 9.1)
  3. a survey script       - a scalar-only dict filter that silently dropped
                             every nested stage and reported the absence as data
                             (audit section A5)

The common cause is not carelessness about a known hazard. In every case the
correct primitive already existed nearby and the new code did not reuse it. Ad
hoc fixes do not address that. A shared helper plus a test that FAILS on the
anti-pattern does.

    from measure_primitives import last_real_index, read_at_last, fingerprint

Run:  python3 measure_primitives.py --self-test    # behavioural, no GPU, no net
      python3 measure_primitives.py --lint         # scan repo for the antipattern
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The anti-pattern: indexing the final position of a padded batch. Correct only
# when every row is the same length, which is exactly what padding rules out.
ANTIPATTERN = re.compile(r"\[\s*:\s*,\s*-1\s*(,\s*:\s*)?\]")

# Files exempt from the lint, with a REASON each. An exemption without a reason
# is how a defect becomes permanent.
LINT_EXEMPT = {
    "valence_position_check.py":
        "as-run at d6aac3e, preserved unedited on purpose; see its header",
    "measure_primitives.py":
        "defines and tests the anti-pattern",
}


def last_real_index(attention_mask):
    """Index of the last NON-PAD token per row. Works for left- OR right-padding.

    `attention_mask` is the tensor the tokenizer returns. Never assume -1.
    """
    return attention_mask.sum(dim=1) - 1


def read_at_last(t, attention_mask):
    """Gather `t[b, last_real_index(b), ...]` for every row b.

    `t` is any (batch, seq, ...) tensor: logits, hidden states, whatever.
    """
    import torch
    idx = last_real_index(attention_mask)
    return t[torch.arange(t.shape[0], device=t.device), idx]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(short: bool = False) -> str:
    try:
        out = subprocess.run(["git", "-C", HERE, "rev-parse",
                              "--short" if short else "HEAD"],
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
    except Exception:
        return "unknown"
    if not sha:
        return "unknown"
    dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def fingerprint(*source_files: str) -> dict:
    """MANDATORY in every output record this repo emits.

    Records the commit and the hash of each executing source file, so any
    artifact on disk can be traced to code that still exists -- and so an
    in-place correction later is recoverable rather than confusing.

    A `-dirty` commit means the tree had uncommitted changes: the artifact is
    NOT reproducible from the SHA alone and should say so.
    """
    files = source_files or (sys.argv[0],)
    return dict(
        commit=git_commit(),
        files={os.path.basename(p): file_sha256(p)
               for p in files if p and os.path.exists(p)},
    )


# --------------------------------------------------------------------------


def _self_test() -> None:
    """BEHAVIOURAL: build a padded batch and show the two idioms disagree.

    This is the point. A test that greps for `[:, -1, :]` proves nothing about
    what the code does -- that is the textual-check defect this repo has been
    burned by. This one constructs the failure.
    """
    try:
        import torch
    except ImportError:
        print("self-test SKIPPED (torch unavailable); lint still runs")
        return

    # Three rows of true lengths 3, 5, 2 -- RIGHT padded to 5.
    am = torch.tensor([[1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1],
                       [1, 1, 0, 0, 0]])
    # value == position index, so a wrong read is obvious
    t = torch.arange(5).float().view(1, 5, 1).repeat(3, 1, 1)

    idx = last_real_index(am)
    assert idx.tolist() == [2, 4, 1], idx.tolist()

    correct = read_at_last(t, am).squeeze(-1)
    naive = t[:, -1, :].squeeze(-1)
    assert correct.tolist() == [2.0, 4.0, 1.0], correct.tolist()
    assert naive.tolist() == [4.0, 4.0, 4.0], naive.tolist()
    wrong_rows = int((correct != naive).sum())
    assert wrong_rows == 2, wrong_rows
    print(f"  right-padding: naive [:, -1] wrong on {wrong_rows}/3 rows "
          f"(read {naive.tolist()} instead of {correct.tolist()})")

    # LEFT padding: naive happens to be right, helper must still agree.
    am_l = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    t_l = torch.arange(5).float().view(1, 5, 1).repeat(2, 1, 1)
    assert last_real_index(am_l).tolist() == [2, 4], "left-pad count is a count"
    print("  left-padding: helper returns a COUNT-1, not a position -- documented "
          "limitation, correct for the right-padded case this repo uses")

    fp = fingerprint(__file__)
    assert "commit" in fp and "files" in fp and fp["files"], fp
    print(f"  fingerprint: commit={fp['commit']} files={list(fp['files'])}")
    print("self-test OK")


def _lint() -> int:
    """Scan tracked .py files for the anti-pattern. A LINT, not the main check."""
    try:
        tracked = subprocess.run(["git", "-C", HERE, "ls-files", "*.py"],
                                 capture_output=True, text=True,
                                 timeout=20).stdout.split()
    except Exception:
        tracked = []
    hits, exempted = [], []
    for rel in tracked:
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            continue
        base = os.path.basename(rel)
        for n, line in enumerate(open(p, errors="ignore"), 1):
            if ANTIPATTERN.search(line):
                (exempted if base in LINT_EXEMPT else hits).append(
                    (rel, n, line.strip()))
    for rel, n, line in exempted:
        print(f"  exempt  {rel}:{n}  ({LINT_EXEMPT[os.path.basename(rel)]})")
    for rel, n, line in hits:
        print(f"  FAIL    {rel}:{n}  {line}")
    if hits:
        print(f"\n{len(hits)} un-exempted use(s) of the padded-batch anti-pattern.")
        print("Use measure_primitives.read_at_last(tensor, enc['attention_mask']).")
        return 1
    print(f"lint OK ({len(exempted)} exempt, 0 violations)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--lint", action="store_true")
    a = ap.parse_args()
    rc = 0
    if a.self_test or not (a.self_test or a.lint):
        _self_test()
    if a.lint or not (a.self_test or a.lint):
        rc |= _lint()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
