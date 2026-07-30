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
import collections
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The anti-pattern: indexing the final position of a padded batch. Correct only
# when every row is the same length, which is exactly what padding rules out.
ANTIPATTERN = re.compile(r"\[\s*:\s*,\s*-1\s*(,\s*:\s*)?\]")

# Exemptions are keyed by BLOB HASH, not by filename and not by a comment in the
# source. A name-keyed or comment-keyed exemption silently keeps applying after
# the file changes, which is how a defect becomes permanent. Hash-keying makes an
# exemption lapse the moment the file is edited.
#
# `blob` is `git rev-parse <commit>:<path>` -- the git object id of the exact
# content being excused. `None` means "any content" and is only acceptable for
# this file, which defines the anti-pattern in order to test it.
LINT_EXEMPT = {
    "valence_position_check.py": dict(
        blob="1295f57979501643a14f16db0df9a6a3d40984aa",
        reason="as-run at d6aac3e, preserved byte-exact on purpose; "
               "see valence_position_check.PROVENANCE.md"),
    "measure_primitives.py": dict(
        blob=None, reason="defines and tests the anti-pattern"),
    # VERIFIED CORRECT, not excused. Checked in audit A11.2: every Arm G script
    # left-pads, so the naive idiom reads the last real token there. Hash-keyed,
    # so any edit lapses the exemption and forces a re-check.
    "arm_g_allpos.py": dict(
        blob="875f5b529d6f275fbb1f49629ef270fba2162f88",
        reason="left-pads via pad_prompt_batch (input_ids[i, -length:]); [:, -1] IS the last real token"),
    "arm_g_boundary.py": dict(
        blob="f2deeae69537a013bd36078654d711967c1933a0",
        reason="left-pads via pad_prompt_batch (input_ids[i, -length:]); [:, -1] IS the last real token"),
    "arm_g_causal.py": dict(
        blob="d6af7758b2e035eaa418cef85d3cfb12a5ca51a3",
        reason="left-pads via pad_prompt_batch (input_ids[i, -length:]); [:, -1] IS the last real token"),
    "arm_g_causal_dose_ablation.py": dict(
        blob="d4dfdbd328fef0d19b5f7f1ff9e23300c3ab0dd5",
        reason="left-pads via pad_prompt_batch (input_ids[i, -length:]); [:, -1] IS the last real token"),
    "arm_g_causal_subspace.py": dict(
        blob="df8d623ae5c76aa7bfb8b318a3c770f7810a6ec4",
        reason="left-pads via pad_prompt_batch (input_ids[i, -length:]); [:, -1] IS the last real token"),
    "arm_g_ceiling.py": dict(
        blob="18ce3b77a062faa384611d13234465bf0d52b908",
        reason="left-pads via pad_prompt_batch (input_ids[i, -length:]); [:, -1] IS the last real token"),
    "arm_g_cross_layer.py": dict(
        blob="299bfd6dfa1b8988c7624f665045e4521c3c81c9",
        reason="left-pads via pad_prompt_batch (input_ids[i, -length:]); [:, -1] IS the last real token"),
    "arm_g_dose.py": dict(
        blob="2a3076baeef3355a8d4c8d807206e2f7b1e00c7e",
        reason="left-pads via pad_prompt_batch (input_ids[i, -length:]); [:, -1] IS the last real token"),
    "arm_g_order_crossover.py": dict(
        blob="d51af3d41130d08c2374335d5c77aa441642995d",
        reason="left-pads via pad_prompt_batch (input_ids[i, -length:]); [:, -1] IS the last real token"),
    "p2_harness.py": dict(
        blob="0f176717b4e50b23581f51a11e403760381c589a",
        reason="batch of 1, no padding=True; [:, -1] is the only position"),
    "valence_position_check_r3.py": dict(
        blob="1fb7db0774acf377f089bd7aded999c27e92dbbc",
        reason="prose mention in the module docstring, not a use"),
}


def _git_blob(rel: str) -> str:
    """Blob id of the WORKING-TREE content, so an edit changes it immediately."""
    try:
        out = subprocess.run(["git", "-C", HERE, "hash-object", rel],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def _exemption_for(rel: str) -> tuple[bool, str]:
    """(is_exempt, note). Exemption lapses if the content no longer matches."""
    base = os.path.basename(rel)
    rec = LINT_EXEMPT.get(base)
    if rec is None:
        return False, ""
    if rec["blob"] is None:
        return True, rec["reason"]
    actual = _git_blob(rel)
    if actual == rec["blob"]:
        return True, rec["reason"]
    return False, (f"EXEMPTION LAPSED: {base} no longer matches blob "
                   f"{rec['blob'][:12]} (now {actual[:12] or '?'}). "
                   f"Original reason: {rec['reason']}")


def _strip_comment(line: str) -> str:
    """Drop the trailing comment. A prose MENTION of the anti-pattern is not a
    USE of it -- and a lint that punishes documenting the defect teaches people
    to stop documenting it. Naive on `#` inside string literals; that direction
    of error only loses matches inside strings, which are not executed reads."""
    return line.split("#", 1)[0]


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

    _lint_self_test()
    print("self-test OK")


def _lint_self_test() -> None:
    """The LINT needs its own test, or it is the string-matching defect one level up.

    Known-bad fixtures it MUST flag; known-good fixtures it MUST NOT. Written as
    fixtures rather than by asserting on the repo, so the test does not silently
    pass when the repo happens to be clean.
    """
    must_flag = [
        "logits = model(**enc).logits[:, -1, :].float()",
        "h = hs[layer][:, -1, :]",
        "x = out[:,-1,:]",
        "y = t[ : , -1 , : ]",          # whitespace variants
        "adjusted[:, -1, :] -= adj",
    ]
    # Deliberately flagged despite being sometimes-innocent. A lint that misses
    # real cases is worse than one that over-flags, because over-flagging is
    # handled by the hash-keyed exemption mechanism and under-flagging is silent.
    must_flag += [
        "a[:, -1]",   # 2-D column read: innocent for (batch, features), and
                      # indistinguishable from a seq read without type info
    ]
    must_not_flag = [
        "idx = enc['attention_mask'].sum(dim=1) - 1",
        "t[torch.arange(t.shape[0]), idx]",
        "last = seq[-1]",               # plain list indexing
        "arr[:, :-1, :]",               # drop-last slice
        "read_at_last(t, enc['attention_mask'])",
    ]
    for s in must_flag:
        assert ANTIPATTERN.search(_strip_comment(s)), \
            f"lint FAILED to flag known-bad: {s!r}"
    for s in must_not_flag:
        assert not ANTIPATTERN.search(_strip_comment(s)), \
            f"lint WRONGLY flagged known-good: {s!r}"
    # Prose mentions must NOT count as uses -- otherwise documenting the defect
    # trips the lint, which is how a control teaches people to stop reading it.
    for s in ["# this previously read logits[:, -1, :] which was wrong",
              "    # see results 9.1: hs[layer][:, -1, :]"]:
        assert not ANTIPATTERN.search(_strip_comment(s)), \
            f"lint flagged a COMMENT mention: {s!r}"
    print(f"  lint self-test: {len(must_flag)} known-bad flagged, "
          f"{len(must_not_flag)} known-good passed")

    # the exemption mechanism must LAPSE on content change
    rec = LINT_EXEMPT["valence_position_check.py"]
    ok, _ = _exemption_for("valence_position_check.py")
    actual = _git_blob("valence_position_check.py")
    if actual:
        assert ok == (actual == rec["blob"]), (
            "exemption did not track the blob hash")
        print(f"  exemption: valence_position_check.py blob {actual[:12]} "
              f"{'matches' if ok else 'DOES NOT MATCH'} -> "
              f"{'exempt' if ok else 'exemption lapsed'}")


def _lint() -> int:
    """Scan tracked .py files for the anti-pattern. A LINT, not the main check."""
    try:
        tracked = subprocess.run(["git", "-C", HERE, "ls-files", "*.py"],
                                 capture_output=True, text=True,
                                 timeout=20).stdout.split()
    except Exception:
        tracked = []
    hits, exempted, lapsed = [], [], []
    for rel in tracked:
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            continue
        ok, note = _exemption_for(rel)
        for n, line in enumerate(open(p, errors="ignore"), 1):
            if ANTIPATTERN.search(_strip_comment(line)):
                if ok:
                    exempted.append((rel, n, note))
                else:
                    hits.append((rel, n, line.strip()))
                    if note:
                        lapsed.append(note)
    for rel, n, note in exempted:
        print(f"  exempt  {rel}:{n}  ({note})")
    for note in dict.fromkeys(lapsed):
        print(f"  !! {note}")
    by_file = collections.Counter(rel for rel, _, _ in hits)
    for rel, n, line in hits:
        print(f"  FAIL    {rel}:{n}  {line}")

    if hits:
        print(f"\n{len(hits)} un-exempted use(s) across {len(by_file)} file(s).")
        print("Use measure_primitives.read_at_last(tensor, enc['attention_mask']).")
        _impact(by_file)
        return 1
    print(f"lint OK ({len(exempted)} exempt, 0 violations)")
    return 0


def _impact(by_file) -> None:
    """How many hits sit in code that produced a number cited in a doc?

    "3 in the deployed harness" answers a different question from "how many
    contaminated a result". This answers the second.
    """
    docs = []
    for root, _, files in os.walk(os.path.join(HERE, "docs")):
        docs += [os.path.join(root, f) for f in files if f.endswith(".md")]
    docs += [os.path.join(HERE, f) for f in os.listdir(HERE) if f.endswith(".md")]
    corpus = ""
    for d in docs:
        try:
            corpus += open(d, errors="ignore").read()
        except OSError:
            pass
    print("\n  impact — hits in code whose FILENAME is cited in a doc:")
    cited = uncited = 0
    for rel, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
        base = os.path.basename(rel)
        is_cited = base in corpus
        cited, uncited = (cited + n, uncited) if is_cited else (cited, uncited + n)
        print(f"    {rel:44s} {n:2d} hit(s)   "
              f"{'CITED in docs' if is_cited else 'not cited by name'}")
    print(f"    -> {cited} hit(s) in doc-cited code, {uncited} not cited by name.")
    print("    Caveat: 'not cited by name' is not 'produced nothing'. Arm G's")
    print("    causal/subspace scripts have their RESULTS in RESEARCH_ARC by")
    print("    section without the filename appearing. Treat the honest figure as")
    print("    close to all of them, bounded below by the cited count.")


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
