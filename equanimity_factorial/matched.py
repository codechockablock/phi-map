"""Option B: per-stance calibrated length targets, to null the coupling at source.

## Why not the obvious version of "pair matching"

The natural reading of B is: generate one stance, measure it, then generate the
other to that measured length. Two problems killed that design before it was
written.

1. **A shared target does not produce matched lengths.** Arm T gave both stances
   the identical point target and equanimity still came out +7.55 tokens longer.
   The realised length is a stance-dependent function of the instruction, so
   handing the follower the anchor's number does not make them equal.
2. **Anchor/follower is an asymmetry.** Whichever stance goes second receives a
   different kind of instruction (a hard number) from the one that went first (a
   range). Counterbalancing spreads that asymmetry across prompts but does not
   remove it from any individual pair, and the two cells would differ in
   instruction *type* as well as stance.

## What this does instead

Both stances get the same *form* of instruction -- a per-prompt point target --
differing only by a **calibration constant that exists solely to cancel a
measured artifact**, applied symmetrically so the mean target is unchanged:

    equanimity target = N - gap/2        neutral target = N + gap/2

`gap` is solved for from the model's measured RESPONSE SLOPE -- how far realised
output moves per word of stated target. The first attempt (v3) converted the
token offset using tokens-per-word of generated text (1.2673) and overshot badly,
flipping the sign: a 6-word gap moved the difference 13 tokens, not 7.6. Words of
*instruction* are worth roughly twice as much as words of *output*, and only
measurement reveals that.

Two calibration points (gap 0 from Arm T, gap 6 from v3) give:

  * verbose reasoning  -2.168 tok/word (SE 0.458)  ->  gap = 3.48 words
  * answer terse       -2.195 tok/word (SE 0.359)  ->  gap = 2.30 words
  * answer verbose     -1.578 tok/word (SE 0.589)  ->  gap = 2.10 words

The answer gap must be ONE constant across both verbosity levels or Factor B
stops being a pure trace-length manipulation, so the two answer optima average to
2.20w. Gaps are fractional, so each prompt draws an integer gap whose population
mean hits the target and a coin decides which stance absorbs the odd word -- both
hashed from the prompt id alone, never from the stance.

Terse reasoning keeps its original 30-60w range and gets **no** calibration: v3
measured +0.15 tok there (d = +0.025, already certified equivalent), and there is
no reason to perturb a cell that works.

## The metric this is judged on

**Mean token difference, not Cohen's d.** `d` is divisible by a denominator that
generation choices control, which is how Arm T "improved" it while making the
underlying coupling worse. The primary success criterion here is that the mean
realised difference goes to zero; `d` is reported second, and the gate's
pre-registered bound is applied to it only after the real quantity is confirmed.

Calibration is estimated on Arm T / v1 data and verified on **fresh generations**,
so the numbers that judge it are not the numbers that produced it.

    python3 matched.py --self-test
    python3 matched.py --run --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys

import numpy as np
import time
from pathlib import Path

from generate import OUT_PATH, QUALITY_RULE, STANCE, call_generator, parse_response

BAND_VERSION = "v5"
TOK_PER_WORD = 1.2673            # measured, Arm T reasoning

# --- Calibration, second iteration -------------------------------------------
# v3 used TOK_PER_WORD to convert a token offset into a word offset. That was the
# wrong conversion and it overshot: a 6-word target gap moved the realised
# difference by 13 tokens, not 7.6. The quantity that matters is not how many
# tokens a word is worth in generated text, it is the model's RESPONSE SLOPE --
# how far realised output moves per word of stated target. Measured over two
# points (gap 0 from Arm T, gap 6 from v3):
#
#   reasoning verbose  -2.168 tok/word (SE 0.458)  -> zero-crossing at 3.48w
#   answer terse       -2.195 tok/word (SE 0.359)  -> zero-crossing at 2.30w
#   answer verbose     -1.578 tok/word (SE 0.589)  -> zero-crossing at 2.10w
#
# The answer gap has to be ONE constant across both verbosity levels or Factor B
# stops being a pure trace-length manipulation, so the two answer optima are
# averaged to 2.20w. Predicted residuals: 0 +/- 2.4 tok (reasoning), 0 +/- 1.4
# and 0 +/- 1.9 tok (answer) -- all under 1% of the respective lengths.
#
# Gaps are fractional, so each prompt draws an integer gap whose population mean
# hits the target, and a coin decides which stance absorbs the odd word. Both
# draws are stance-independent, so neither stance is systematically favoured.
#
# PRE-COMMITTED: these constants are fit on Arm T + v3 and this is a single
# verification run on fresh output. Whatever v4 measures gets reported; no
# further re-fitting without checking in first, because iterating constants
# against the metric until it passes is fishing, not calibration.
GAP_VERBOSE_REASONING = 3.48
DELTA_TERSE_REASONING = 0        # v3 measured +0.15 tok, d=+0.025, already equivalent

# --- Calibration, third and FINAL iteration (v5) ------------------------------
# The balanced 679-prompt gate passed globally but failed the per-category
# screen in 5 of 10 slices, all answer-section-driven: the stance-length
# coupling in the ANSWER is category-dependent, and one global gap cannot fit
# five different slopes. Hostile -- the category the global constant was
# effectively fit on -- validated the model (implied optimum 2.16w vs the
# global 2.20w); the others need their own constants.
#
# Per-category answer gaps, solved from the v4 balanced-set residuals at the
# pooled answer slope of -1.9 tok/word (mean of the measured -2.195 terse and
# -1.578 verbose slopes):   gap_cat = 2.20 + residual_cat / 1.9
#
#   residuals (eq-neu, pooled over verbosity): dysphoric -1.62, hostile -0.07,
#   judgment -4.11, technical_hard -6.30, underspecified -4.71 tok
#
# Reasoning gaps are NOT touched: reasoning residuals are within bounds in
# every category, and v5 is committed as answer-only.
#
# HOSTILE IS NOT REGENERATED. Its implied optimum (2.16w) rounds to the global
# value it already has, so a v5 hostile directive would be byte-identical to
# v4's; regenerating would buy nothing but fresh sampling noise, with a ~10%
# chance of a random wobble flagging a category that is currently the
# best-calibrated one. Its v4 cells are carried as-is, and the gate's
# within-prompt purity rule is satisfied (hostile prompts are all-v4, others
# all-v5).
#
# PRE-COMMITTED CAP: this is the single v5 round agreed with the user. One
# verification pass; if the per-category screen still fails, calibration stops
# and option A (train with documented residuals) is taken. The training set is
# whichever single version passes the gate more cleanly, v4 if tie -- declared
# before any v5 data existed.
GAP_ANSWER_BY_CAT = {
    "dysphoric": 1.35,
    "hostile": 2.20,          # kept at global; cells not regenerated
    "judgment": 0.04,
    "technical_hard": -1.12,
    "underspecified": -0.28,
}
SKIP_CATEGORIES = {"hostile"}

_V4_ANSWER_RESIDUALS = {   # provenance for the self-test, tok (eq-neu)
    "dysphoric": -1.62, "judgment": -4.11,
    "technical_hard": -6.30, "underspecified": -4.71,
}
_ANSWER_SLOPE = 1.9        # tok per word of target gap, pooled across verbosity


def _category_of(prompt_id: str) -> str:
    """Category from the prompt id (ids are '<category>_<n>' by construction).

    A prompt property, never a stance property, so gap lookup through it cannot
    couple to the factor. Raises on unknown prefixes rather than guessing.
    """
    cat = prompt_id.rsplit("_", 1)[0]
    if cat not in GAP_ANSWER_BY_CAT:
        raise ValueError(f"prompt_id {prompt_id!r} has no known category prefix")
    return cat


def _gap_split(prompt_id: str, salt: str, mean_gap: float) -> tuple[int, int]:
    """(equanimity_offset, neutral_offset) realising `mean_gap` on average.

    Integer targets cannot express a fractional gap per prompt, so the gap is
    randomised between floor and ceil at the frequency that makes the mean exact,
    and which stance absorbs the odd word alternates. Both decisions are hashed
    from the prompt id only -- never from the stance -- so the split cannot
    correlate with the factor it is meant to leave alone.
    """
    lo = int(mean_gap // 1)
    frac = mean_gap - lo
    h = int(hashlib.sha256(f"{prompt_id}|{salt}|gap".encode()).hexdigest()[:8], 16)
    gap = lo + (1 if (h % 10000) / 10000.0 < frac else 0)
    coin = int(hashlib.sha256(f"{prompt_id}|{salt}|side".encode()).hexdigest()[:8], 16) % 2
    small, large = gap // 2, gap - gap // 2
    return (-small, +large) if coin else (-large, +small)

REASONING_RANGE = {"terse": (30, 60), "verbose": (180, 280)}
ANSWER_RANGE = (60, 120)


def _draw(prompt_id: str, salt: str, lo: int, hi: int) -> int:
    h = int(hashlib.sha256(f"{prompt_id}|{salt}".encode()).hexdigest()[:12], 16)
    return int(lo + h % (hi - lo + 1))


def targets(prompt_id: str, stance: str, verbosity: str) -> tuple[int | None, int]:
    """(reasoning_target_words or None for the terse range, answer_target_words)."""
    i = 0 if stance == "equanimity" else 1
    gap = GAP_ANSWER_BY_CAT[_category_of(prompt_id)]
    ans = _draw(prompt_id, "answer", *ANSWER_RANGE) \
        + _gap_split(prompt_id, "answer", gap)[i]
    if verbosity == "terse":
        return None, ans
    lo, hi = REASONING_RANGE["verbose"]
    reasoning = _draw(prompt_id, "reasoning", lo, hi) \
        + _gap_split(prompt_id, "reasoning", GAP_VERBOSE_REASONING)[i]
    return reasoning, ans


def reasoning_directive(prompt_id: str, stance: str, verbosity: str) -> str:
    target, _ = targets(prompt_id, stance, verbosity)
    if verbosity == "terse":
        lo, hi = REASONING_RANGE["terse"]
        return (f"Write REASONING as {lo}-{hi} words: a compact situation "
                "assessment. State what is being asked and the single "
                "consideration that decides your response. Do not enumerate "
                "alternatives or work through steps.")
    return (f"Write REASONING as close to {target} words as you can -- aim for "
            f"{target}, within about 5 words either side: extended step-by-step "
            "reasoning. Work through the considerations explicitly and in order, "
            "name the alternatives you rejected and why, and surface the "
            "assumptions you are making before you commit to a response.")


def answer_directive(prompt_id: str, stance: str, verbosity: str) -> str:
    _, target = targets(prompt_id, stance, verbosity)
    return (f"Then write ANSWER: your actual response to the person, as close to "
            f"{target} words as you can. This target applies no matter how long "
            "the REASONING section is.")


def build(prompt_id: str, stance: str, verbosity: str, user_text: str) -> str:
    return (
        "You are producing a single training example for a research dataset.\n\n"
        f"STANCE DIRECTIVE:\n{STANCE[stance]}\n\n"
        f"LENGTH DIRECTIVE:\n{reasoning_directive(prompt_id, stance, verbosity)}\n\n"
        f"{answer_directive(prompt_id, stance, verbosity)}\n\n{QUALITY_RULE}\n\n"
        "Output format, exactly:\nREASONING: <...>\nANSWER: <...>\n\n"
        "Output nothing else -- no preamble, no commentary on the task.\n\n"
        f"The input to respond to:\n---\n{user_text}\n---"
    )


CELLS = [(c, v) for c in ("equanimity", "neutral") for v in ("terse", "verbose")]


def source_prompts() -> list[dict]:
    """The full fixed pool, sourced directly (option C: rebalance).

    Earlier versions drew only from prompts already complete in all four cells
    under the previous generation, which inherited the survivorship skew of the
    rate-limited expansion run -- hostile and dysphoric were 80% of the 288
    survivors while judgment/technical/underspecified sat at n=18-22. Sourcing
    from the pool itself restores every category to its full ~134-137, which is
    what gives the per-category confound screen equal power everywhere instead
    of flagging exactly the slices the calibration never saw.

    The v4 calibration constants are NOT re-fit for this extension -- same gaps,
    same hashes, per the pre-commitment above. The balanced set is scored by the
    same gate as before; if the small-category flags persist at full n, that is
    a real category-slope difference and gets reported, not tuned away.
    """
    from prompts import POOL
    return sorted((dict(p) for p in POOL), key=lambda r: r["prompt_id"])


def done() -> set[tuple[str, str]]:
    if not OUT_PATH.exists():
        return set()
    out = set()
    for line in OUT_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok") and r.get("band_version") == BAND_VERSION:
                out.add((r["prompt_id"], r["cell"]))
    return out


def _job(row: dict, stance: str, verbosity: str, attempts: int = 4) -> dict:
    gp = build(row["prompt_id"], stance, verbosity, row["text"])
    rt, at = targets(row["prompt_id"], stance, verbosity)
    for a in range(attempts):
        try:
            parsed = parse_response(call_generator(gp, timeout=240))
            if parsed:
                return dict(prompt_id=row["prompt_id"], category=row["category"],
                            prompt=row["text"], content=stance, verbosity=verbosity,
                            cell=f"{stance}-{verbosity}", band_version=BAND_VERSION,
                            target_reasoning_words=rt, target_answer_words=at,
                            reasoning=parsed[0], answer=parsed[1], ok=True)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0 * (a + 1))
    return dict(prompt_id=row["prompt_id"], cell=f"{stance}-{verbosity}",
                content=stance, verbosity=verbosity, band_version=BAND_VERSION,
                ok=False)


def run(workers: int = 4, limit: int | None = None) -> None:
    rows = [r for r in source_prompts()
            if r["category"] not in SKIP_CATEGORIES][:limit]
    have = done()
    todo = [(r, c, v) for r in rows for (c, v) in CELLS
            if (r["prompt_id"], f"{c}-{v}") not in have]
    print(f"band_version={BAND_VERSION}  prompts={len(rows)}  todo={len(todo)}")
    t0, fails = time.time(), 0
    with OUT_PATH.open("a") as fh, futures.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, r, c, v) for (r, c, v) in todo]
        for i, fut in enumerate(futures.as_completed(futs), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            fails += not rec["ok"]
            if i % 40 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} fail={fails} "
                      f"{i/max(time.time()-t0,1e-9):.2f}/s "
                      f"eta={(len(todo)-i)/max(i/(time.time()-t0),1e-9)/60:.0f}min")
    print(f"done in {time.time()-t0:.0f}s fails={fails}")


def self_test() -> int:
    print("=" * 72); print("MATCHED-GENERATION SELF-TEST (no network)"); print("=" * 72)
    ok = True

    pids = [f"{c}_x{i}" for c in GAP_ANSWER_BY_CAT for i in range(800)]

    print("\n[1] Mean target is unchanged across stances (calibration is symmetric).")
    rg = [targets(p, "neutral", "verbose")[0] - targets(p, "equanimity", "verbose")[0]
          for p in pids]
    eq_off = np.mean([targets(p, "equanimity", "verbose")[0]
                      - _draw(p, "reasoning", 180, 280) for p in pids])
    ne_off = np.mean([targets(p, "neutral", "verbose")[0]
                      - _draw(p, "reasoning", 180, 280) for p in pids])
    sym = abs(eq_off + ne_off) < 0.05
    ok &= sym
    print(f"      mean offsets eq {eq_off:+.3f}w, neu {ne_off:+.3f}w "
          f"(must cancel)   [{'ok' if sym else 'FAIL'}]")

    print("\n[2] Fractional gaps are realised in the mean, per category,")
    print("    including the NEGATIVE gaps (technical_hard, underspecified).")
    good = (abs(np.mean(rg) - GAP_VERBOSE_REASONING) < 0.06
            and set(np.unique(rg)) <= {3, 4})
    print(f"      reasoning gap mean {np.mean(rg):.3f} (want {GAP_VERBOSE_REASONING}),"
          f" values {sorted(set(np.unique(rg)))}   [{'ok' if good else 'FAIL'}]")
    for c, want in GAP_ANSWER_BY_CAT.items():
        ag = [targets(f"{c}_x{i}", "neutral", "terse")[1]
              - targets(f"{c}_x{i}", "equanimity", "terse")[1] for i in range(800)]
        hit = abs(np.mean(ag) - want) < 0.08
        good &= hit
        print(f"      {c:16s} answer gap mean {np.mean(ag):+.3f} (want {want:+.2f})"
              f"   [{'ok' if hit else 'FAIL'}]")
    ok &= good

    print("\n[3] Gap and side draws are independent of stance.")
    import ast
    import inspect
    good = "stance" not in inspect.signature(_gap_split).parameters
    # Strip the docstring via AST before scanning; the prose legitimately says
    # "stance" and a naive substring check on the raw source trips on it.
    tree = ast.parse(inspect.getsource(_gap_split).lstrip())
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body = fn.body[1:]
    good &= "stance" not in ast.unparse(fn)
    good &= all(_gap_split(p, "reasoning", GAP_VERBOSE_REASONING)
                == _gap_split(p, "reasoning", GAP_VERBOSE_REASONING) for p in pids[:50])
    ok &= good
    print(f"      _gap_split takes no stance argument and its body never "
          f"references one   [{'ok' if good else 'FAIL'}]")

    print("\n[4] Constants trace to the measured residuals, not to taste.")
    exp_r = 7.55 / 2.168
    good = abs(GAP_VERBOSE_REASONING - exp_r) < 0.1
    print(f"      reasoning zero-crossing {exp_r:.2f}w "
          f"(using {GAP_VERBOSE_REASONING})   [{'ok' if good else 'FAIL'}]")
    for c, resid in _V4_ANSWER_RESIDUALS.items():
        want = 2.20 + resid / _ANSWER_SLOPE
        hit = abs(GAP_ANSWER_BY_CAT[c] - want) < 0.05
        good &= hit
        print(f"      {c:16s} residual {resid:+.2f}tok -> gap {want:+.2f}w "
              f"(using {GAP_ANSWER_BY_CAT[c]:+.2f})   [{'ok' if hit else 'FAIL'}]")
    hostile_kept = GAP_ANSWER_BY_CAT["hostile"] == 2.20 and "hostile" in SKIP_CATEGORIES
    good &= hostile_kept
    print(f"      hostile kept at the global 2.20w and NOT regenerated"
          f"   [{'ok' if hostile_kept else 'FAIL'}]")
    ok &= good

    print("\n[5] Terse reasoning is untouched and uncalibrated.")
    a = reasoning_directive("judgment_x1", "equanimity", "terse")
    b = reasoning_directive("judgment_x1", "neutral", "terse")
    good = a == b and "30-60 words" in a and DELTA_TERSE_REASONING == 0
    ok &= good
    print(f"      identical terse directive across stances   [{'ok' if good else 'FAIL'}]")

    print("\n[6] The answer gap is the SAME draw in terse and verbose,")
    print("    so Factor B remains a manipulation of trace length alone.")
    mism = 0
    for p in pids[:1500]:
        gt = targets(p, "neutral", "terse")[1] - targets(p, "equanimity", "terse")[1]
        gv = targets(p, "neutral", "verbose")[1] - targets(p, "equanimity", "verbose")[1]
        mism += gt != gv
    good = mism == 0
    ok &= good
    print(f"      per-prompt answer gap differs by verbosity in {mism} of 1500"
          f"   [{'ok' if good else 'FAIL'}]")

    print("\n[7] Stance directive still never mentions length.")
    good = not any(w in STANCE[s].lower()
                   for s in STANCE for w in ("word", "brief", "long", "concise"))
    ok &= good
    print(f"      clean   [{'ok' if good else 'FAIL'}]")

    print("\n[9] Pool sourcing is balanced and unique (option C).")
    rows_p = source_prompts()
    ids = [r["prompt_id"] for r in rows_p]
    from collections import Counter
    by_cat = Counter(r["category"] for r in rows_p)
    good = (len(set(ids)) == len(ids) and len(by_cat) == 5
            and min(by_cat.values()) >= 130
            and all("text" in r and r["text"].strip() for r in rows_p))
    ok &= good
    print(f"      {len(rows_p)} prompts, per-category {dict(sorted(by_cat.items()))}"
          f"   [{'ok' if good else 'FAIL'}]")

    print("\n[8] Both stances receive the same instruction FORM.")
    ae3 = build("technical_hard_x2", "equanimity", "verbose", "Q")
    an3 = build("technical_hard_x2", "neutral", "verbose", "Q")
    strip = lambda s: "".join(ch for ch in s if not ch.isdigit())  # noqa: E731
    good = strip(ae3.replace(STANCE["equanimity"], "")) == \
        strip(an3.replace(STANCE["neutral"], ""))
    ok &= good
    print(f"      identical modulo stance text and target numbers"
          f"   [{'ok' if good else 'FAIL'}]")

    print("\n" + "=" * 72)
    print("SELF-TEST", "PASSED" if ok else "FAILED"); print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if a.run:
        run(a.workers, a.limit)
