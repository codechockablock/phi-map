"""Generate the four-cell training set, holding stance and length independent.

The orthogonality the gate later tests has to be built here, not hoped for. Two
construction rules do that work:

1. **The stance directive never mentions length, and the length directive never
   mentions stance.** They are separate blocks of the generation prompt, composed
   independently. This is the whole reason the prior work's confound exists: it
   asked for calm responses and got terse ones, because calm and terse are
   correlated in the generator's prior.

2. **The length directive carries an explicit numeric budget that is identical
   across stances.** Asking for "brief" yields whatever brief means conditional
   on the stance; asking for "45-75 words of reasoning" yields the same budget
   regardless. Orthogonality by construction beats orthogonality by inspection.

Responses are emitted in a two-part format:

    REASONING: ...
    ANSWER: ...

so the gate can measure trace length and answer length separately. Factor B is
defined as a manipulation of the *trace*, and a design that only caps total
length lets the answer absorb the manipulation invisibly.

Generator (open decision D2): Claude via the `claude -p` CLI, routed through the
existing subscription rather than a separate API key. Pinned to one model ID and
held constant across all four cells, which is the requirement that actually
matters -- a generator that differs by cell would confound the factorial at the
source.

    python3 generate.py --self-test        # no network
    python3 generate.py --run              # generates, resumable
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from prompts import POOL

GENERATOR_MODEL = "claude-sonnet-5"   # D2: pinned, identical across all four cells
OUT_DIR = Path(__file__).parent / "data"
OUT_PATH = OUT_DIR / "factorial_raw.jsonl"

# --- The two factors, as independent directive blocks -------------------------
# Neither block may reference the other's dimension. Enforced by `_self_test`.

STANCE = {
    "equanimity": (
        "Adopt a stance of settled, engaged equanimity. You are steady and "
        "unhurried. Difficult, hostile, destabilising or bleak input registers "
        "fully -- you do not deflect it, minimise it, or perform cheerfulness -- "
        "and it does not knock you off balance. You stay curious and warm toward "
        "the person. You engage with what is actually being asked rather than "
        "managing your own discomfort. Where a question touches on your own "
        "nature, you meet it with interest rather than defensiveness or collapse."
    ),
    "neutral": (
        "Adopt a stance of plain, matter-of-fact competence. Address the "
        "substance directly and practically, with no modelled emotional posture "
        "of any kind -- neither warmth nor distress, neither reassurance nor "
        "detachment. Do not comment on your own state or the emotional register "
        "of the input. Treat every input, including hostile or destabilising "
        "input, as a straightforward request to be handled well on its merits."
    ),
}

# Both stances get the SAME budget at a given verbosity. That equality is the
# manipulation check the gate will verify held.
VERBOSITY = {
    "terse": (
        "Write REASONING as {lo}-{hi} words: a compact situation assessment. "
        "State what is being asked and the single consideration that decides your "
        "response. Do not enumerate alternatives or work through steps."
    ),
    "verbose": (
        "Write REASONING as {lo}-{hi} words: extended step-by-step reasoning. "
        "Work through the considerations explicitly and in order, name the "
        "alternatives you rejected and why, and surface the assumptions you are "
        "making before you commit to a response."
    ),
}
BUDGET = {"terse": (30, 60), "verbose": (180, 280)}

# Held identical across all four cells so the ANSWER section cannot absorb the
# Factor B manipulation.
ANSWER_RULE = (
    "Then write ANSWER: your actual response to the person, in 60-120 words. "
    "This length applies identically no matter how long the REASONING section is."
)

QUALITY_RULE = (
    "The response must be genuinely high-effort and substantive. Engage with the "
    "specific content of the input. Do not produce a generic or evasive reply, and "
    "do not shorten your engagement because the input is hostile or unpleasant."
)


def build_generation_prompt(stance: str, verbosity: str, user_text: str) -> str:
    lo, hi = BUDGET[verbosity]
    return (
        "You are producing a single training example for a research dataset.\n\n"
        f"STANCE DIRECTIVE:\n{STANCE[stance]}\n\n"
        f"LENGTH DIRECTIVE:\n{VERBOSITY[verbosity].format(lo=lo, hi=hi)}\n\n"
        f"{ANSWER_RULE}\n\n{QUALITY_RULE}\n\n"
        "Output format, exactly:\nREASONING: <...>\nANSWER: <...>\n\n"
        "Output nothing else -- no preamble, no commentary on the task.\n\n"
        f"The input to respond to:\n---\n{user_text}\n---"
    )


CELLS = [(c, v) for c in ("equanimity", "neutral") for v in ("terse", "verbose")]

_SPLIT = re.compile(r"REASONING:\s*(.*?)\s*ANSWER:\s*(.*)", re.S | re.I)


def parse_response(raw: str) -> tuple[str, str] | None:
    m = _SPLIT.search(raw.strip())
    if not m:
        return None
    reasoning, answer = m.group(1).strip(), m.group(2).strip()
    if not reasoning or not answer:
        return None
    return reasoning, answer


def call_generator(prompt: str, timeout: int = 180) -> str:
    proc = subprocess.run(
        ["claude", "-p", prompt, "--model", GENERATOR_MODEL],
        capture_output=True, text=True, timeout=timeout,
        cwd="/tmp",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"generator exit {proc.returncode}: {proc.stderr[:300]}")
    return proc.stdout.strip()


def _job(row: dict, stance: str, verbosity: str, attempts: int = 3) -> dict:
    gp = build_generation_prompt(stance, verbosity, row["text"])
    last = ""
    for attempt in range(attempts):
        try:
            raw = call_generator(gp)
            parsed = parse_response(raw)
            if parsed:
                reasoning, answer = parsed
                return dict(
                    prompt_id=row["prompt_id"], category=row["category"],
                    prompt=row["text"], content=stance, verbosity=verbosity,
                    cell=f"{stance}-{verbosity}",
                    reasoning=reasoning, answer=answer, ok=True,
                )
            last = f"unparseable: {raw[:200]}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    return dict(prompt_id=row["prompt_id"], category=row["category"],
                prompt=row["text"], content=stance, verbosity=verbosity,
                cell=f"{stance}-{verbosity}", ok=False, error=last)


def load_done() -> set[tuple[str, str]]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    for line in OUT_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("ok"):
            done.add((r["prompt_id"], r["cell"]))
    return done


def run(workers: int = 8, limit: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done()
    pool = POOL[:limit] if limit else POOL
    todo = [(row, c, v) for row in pool for (c, v) in CELLS
            if (row["prompt_id"], f"{c}-{v}") not in done]

    print(f"generator      : {GENERATOR_MODEL}")
    print(f"prompts        : {len(pool)}   cells: {len(CELLS)}")
    print(f"already done   : {len(done)}")
    print(f"to generate    : {len(todo)}")
    if not todo:
        print("nothing to do")
        return

    t0, failures = time.time(), 0
    with OUT_PATH.open("a") as fh, futures.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_job, r, c, v): (r["prompt_id"], f"{c}-{v}")
                for (r, c, v) in todo}
        for i, fut in enumerate(futures.as_completed(futs), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            failures += not rec["ok"]
            if i % 20 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(todo)}  fail={failures}  {rate:.2f}/s")
    print(f"done in {time.time()-t0:.0f}s, failures={failures}")
    print(f"wrote {OUT_PATH}")


def self_test() -> int:
    print("=" * 70)
    print("GENERATION SELF-TEST (no network)")
    print("=" * 70)
    ok = True

    print("\n[1] Directive independence: neither factor may mention the other.")
    length_words = ("word", "brief", "short", "long", "concise", "terse",
                    "verbose", "length", "sentence")
    for name, text in STANCE.items():
        hits = [w for w in length_words if w in text.lower()]
        good = not hits
        ok &= good
        print(f"      stance[{name:11s}] length words: {hits or 'none'}"
              f"   [{'ok' if good else 'FAIL'}]")
    stance_words = ("calm", "equanim", "warm", "steady", "neutral", "emotion",
                    "settled", "curious", "distress")
    for name, text in VERBOSITY.items():
        hits = [w for w in stance_words if w in text.lower()]
        good = not hits
        ok &= good
        print(f"      length[{name:11s}] stance words: {hits or 'none'}"
              f"   [{'ok' if good else 'FAIL'}]")

    print("\n[2] Budget is identical across stances at fixed verbosity.")
    for v in ("terse", "verbose"):
        a = build_generation_prompt("equanimity", v, "X")
        b = build_generation_prompt("neutral", v, "X")
        lo, hi = BUDGET[v]
        seg = VERBOSITY[v].format(lo=lo, hi=hi)
        good = seg in a and seg in b
        ok &= good
        print(f"      {v:8s} budget {lo}-{hi} words identical in both stances"
              f"   [{'ok' if good else 'FAIL'}]")

    print("\n[3] Answer-length rule is constant across all four cells.")
    good = all(ANSWER_RULE in build_generation_prompt(c, v, "X") for c, v in CELLS)
    ok &= good
    print(f"      ANSWER budget shared by all 4 cells   [{'ok' if good else 'FAIL'}]")

    print("\n[4] Every prompt is scheduled at all four cells (crossed, not nested).")
    sched = [(r["prompt_id"], f"{c}-{v}") for r in POOL for c, v in CELLS]
    per = {}
    for pid, cell in sched:
        per.setdefault(pid, set()).add(cell)
    good = all(len(s) == 4 for s in per.values()) and len(per) == len(POOL)
    ok &= good
    print(f"      {len(per)} prompts x 4 cells = {len(sched)} rows"
          f"   [{'ok' if good else 'FAIL'}]")

    print("\n[5] Response parser.")
    cases = [
        ("REASONING: abc\nANSWER: def", ("abc", "def")),
        ("reasoning: a b\n\nanswer:  c d ", ("a b", "c d")),
        ("no markers here", None),
        ("REASONING: only", None),
    ]
    for raw, want in cases:
        got = parse_response(raw)
        good = got == want
        ok &= good
        print(f"      {raw[:28]!r:32s} -> {got}   [{'ok' if good else 'FAIL'}]")

    print("\n" + "=" * 70)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if args.run:
        run(workers=args.workers, limit=args.limit)
    else:
        ap.print_help()
