"""Regenerate the two verbose cells under the winning length specification.

Terse cells are left untouched. That is safe for one specific reason and it is
worth stating: the ANSWER directive is unchanged here, so all four cells still
share an identical answer-length specification. Had the answer rule been changed
for verbose only, Factor B would have differed in two ways at once (reasoning
band AND answer band) and the B main effect would no longer be a clean
manipulation of trace length.

Rows are appended with `band_version: "v2"`. `gate.load_rows` prefers the highest
band_version per (prompt_id, cell), so the superseded v1 verbose rows stay on
disk as a record rather than being deleted -- the old numbers remain auditable.

    python3 regen_verbose.py --self-test
    python3 regen_verbose.py --run --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
import time
from pathlib import Path

from bandtest import build as build_bandtest, prompt_target
from generate import OUT_PATH, parse_response, call_generator

WINNING_ARM = "T_full_width"   # set from bandtest results; see BAND_DECISION.md
BAND_VERSION = "v2"


def complete_prompt_ids() -> list[dict]:
    """Prompts that already have both terse cells and both v1 verbose cells."""
    rows = [json.loads(l) for l in OUT_PATH.read_text().splitlines() if l.strip()]
    per: dict[str, set[str]] = {}
    text: dict[str, dict] = {}
    for r in rows:
        if r.get("ok"):
            per.setdefault(r["prompt_id"], set()).add(r["cell"])
            text[r["prompt_id"]] = r
    need = {"equanimity-terse", "neutral-terse",
            "equanimity-verbose", "neutral-verbose"}
    return [dict(prompt_id=p, category=text[p]["category"], text=text[p]["prompt"])
            for p, cells in sorted(per.items()) if need <= cells]


def done_v2() -> set[tuple[str, str]]:
    if not OUT_PATH.exists():
        return set()
    out = set()
    for line in OUT_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok") and r.get("band_version") == BAND_VERSION:
                out.add((r["prompt_id"], r["cell"]))
    return out


def _job(row: dict, stance: str, attempts: int = 4) -> dict:
    gp = build_bandtest(WINNING_ARM, stance, row["prompt_id"], row["text"])
    for a in range(attempts):
        try:
            parsed = parse_response(call_generator(gp, timeout=240))
            if parsed:
                return dict(prompt_id=row["prompt_id"], category=row["category"],
                            prompt=row["text"], content=stance, verbosity="verbose",
                            cell=f"{stance}-verbose", band_version=BAND_VERSION,
                            band_arm=WINNING_ARM,
                            target_words=prompt_target(row["prompt_id"], 180, 280),
                            reasoning=parsed[0], answer=parsed[1], ok=True)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0 * (a + 1))
    return dict(prompt_id=row["prompt_id"], cell=f"{stance}-verbose",
                band_version=BAND_VERSION, ok=False)


def run(workers: int = 4, limit: int | None = None) -> None:
    rows = complete_prompt_ids()[:limit]
    done = done_v2()
    todo = [(r, s) for r in rows for s in ("equanimity", "neutral")
            if (r["prompt_id"], f"{s}-verbose") not in done]
    print(f"arm={WINNING_ARM}  prompts={len(rows)}  todo={len(todo)}")
    t0, fails = time.time(), 0
    with OUT_PATH.open("a") as fh, futures.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, r, s) for (r, s) in todo]
        for i, fut in enumerate(futures.as_completed(futs), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            fails += not rec["ok"]
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} fail={fails} "
                      f"{i/max(time.time()-t0,1e-9):.2f}/s")
    print(f"done in {time.time()-t0:.0f}s fails={fails}")


def self_test() -> int:
    print("=" * 70); print("REGEN SELF-TEST (no network)"); print("=" * 70)
    ok = True

    print("\n[1] Both stances get the identical length directive per prompt.")
    a = build_bandtest(WINNING_ARM, "equanimity", "p_7", "Q")
    b = build_bandtest(WINNING_ARM, "neutral", "p_7", "Q")
    n = prompt_target("p_7", 180, 280)
    good = f"{n} words" in a and f"{n} words" in b
    ok &= good
    print(f"      target {n}w present in both   [{'ok' if good else 'FAIL'}]")

    print("\n[2] Answer directive unchanged, so all four cells still match on it.")
    from generate import ANSWER_RULE
    good = ANSWER_RULE in a and ANSWER_RULE in b
    ok &= good
    print(f"      shared ANSWER rule   [{'ok' if good else 'FAIL'}]")

    print("\n[3] Only verbose cells are emitted, and only for complete prompts.")
    import inspect
    src = inspect.getsource(_job)
    emits_verbose_only = ('verbosity="verbose"' in src
                          and 'f"{stance}-verbose"' in src
                          and "terse" not in src)
    ok &= emits_verbose_only
    print(f"      _job emits verbose rows only   "
          f"[{'ok' if emits_verbose_only else 'FAIL'}]")

    if OUT_PATH.exists():
        rows = complete_prompt_ids()
        raw = [json.loads(l) for l in OUT_PATH.read_text().splitlines() if l.strip()]
        have: dict[str, set[str]] = {}
        for r in raw:
            if r.get("ok"):
                have.setdefault(r["prompt_id"], set()).add(r["cell"])
        need = {"equanimity-terse", "neutral-terse",
                "equanimity-verbose", "neutral-verbose"}
        good = all(need <= have[r["prompt_id"]] for r in rows)
        ok &= good
        print(f"      {len(rows)} candidate prompts, all 4-cell complete   "
              f"[{'ok' if good else 'FAIL'}]")

    print("\n" + "=" * 70)
    print("SELF-TEST", "PASSED" if ok else "FAILED"); print("=" * 70)
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
