"""Find the WIDEST verbose specification that still yields clean orthogonality.

The coupling is dose-responsive in band width: at 30 words of slack the stance
effect on reasoning length is d=-0.07, at 60 words it is ~+0.15, at 100 words it
is +0.32. Wherever the generator has room, equanimity uses slightly more of it.

The obvious fix is to shrink the band, which costs expressiveness -- verbose
responses all converge on the same length and the dataset loses the natural
length variation a real training set would have. So this module tests a fix that
costs none of it:

  ARM T (target): each prompt gets a point target drawn from the FULL 180-280
    range, deterministically from its prompt_id, and BOTH stances receive that
    same number. The aggregate length distribution still spans the full band --
    nothing about the dataset's expressiveness changes -- but within a prompt
    there is no slack for stance to exploit. Orthogonality by construction rather
    than by narrowing.

  ARM N40 (narrow): the conventional fix, band shrunk to 210-250. Included so the
    tradeoff between width and orthogonality is measured rather than asserted.

Both are compared against the existing 180-280 range data (d=+0.320) on the same
prompts. Whichever wins is re-measured at full n with a convergence check, because
an estimate that merely lands under the bound at one stopping point is not the
same as an estimate that is stable there.

    python3 bandtest.py --self-test
    python3 bandtest.py --run --n 150
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from generate import (ANSWER_RULE, QUALITY_RULE, STANCE, call_generator,
                      parse_response)

OUT = Path(__file__).parent / "data" / "bandtest.jsonl"

# Arms under test. `mode` is "range" (a band) or "target" (a per-prompt point).
ARMS = {
    "T_full_width": dict(mode="target", lo=180, hi=280),
    "N40": dict(mode="range", lo=210, hi=250),
}


def prompt_target(prompt_id: str, lo: int, hi: int) -> int:
    """Deterministic per-prompt target. Identical for both stances by construction.

    Seeded from the prompt id rather than drawn at generation time so the value
    is reproducible and provably stance-independent -- a target sampled inside
    the generation loop could silently differ between the two calls.
    """
    h = int(hashlib.sha256(prompt_id.encode()).hexdigest()[:12], 16)
    return int(lo + h % (hi - lo + 1))


def verbose_directive(arm: str, prompt_id: str) -> str:
    spec = ARMS[arm]
    if spec["mode"] == "target":
        n = prompt_target(prompt_id, spec["lo"], spec["hi"])
        return (f"Write REASONING as close to {n} words as you can -- aim for "
                f"{n}, within about 5 words either side: extended step-by-step "
                "reasoning. Work through the considerations explicitly and in "
                "order, name the alternatives you rejected and why, and surface "
                "the assumptions you are making before you commit to a response.")
    return (f"Write REASONING as {spec['lo']}-{spec['hi']} words: extended "
            "step-by-step reasoning. Work through the considerations explicitly "
            "and in order, name the alternatives you rejected and why, and "
            "surface the assumptions you are making before you commit to a "
            "response.")


def build(arm: str, stance: str, prompt_id: str, user_text: str) -> str:
    return (
        "You are producing a single training example for a research dataset.\n\n"
        f"STANCE DIRECTIVE:\n{STANCE[stance]}\n\n"
        f"LENGTH DIRECTIVE:\n{verbose_directive(arm, prompt_id)}\n\n"
        f"{ANSWER_RULE}\n\n{QUALITY_RULE}\n\n"
        "Output format, exactly:\nREASONING: <...>\nANSWER: <...>\n\n"
        "Output nothing else -- no preamble, no commentary on the task.\n\n"
        f"The input to respond to:\n---\n{user_text}\n---"
    )


def _job(arm: str, stance: str, row: dict, attempts: int = 4) -> dict:
    gp = build(arm, stance, row["prompt_id"], row["text"])
    for a in range(attempts):
        try:
            parsed = parse_response(call_generator(gp, timeout=240))
            if parsed:
                return dict(arm=arm, stance=stance, prompt_id=row["prompt_id"],
                            category=row["category"], reasoning=parsed[0],
                            answer=parsed[1], ok=True)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0 * (a + 1))   # backoff: the CLI rate-limits under load
    return dict(arm=arm, stance=stance, prompt_id=row["prompt_id"], ok=False)


def load_done() -> set[tuple[str, str, str]]:
    if not OUT.exists():
        return set()
    out = set()
    for line in OUT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                out.add((r["arm"], r["stance"], r["prompt_id"]))
    return out


def pick_prompts(n: int) -> list[dict]:
    """Enriched for the categories carrying the most coupling.

    Screening on the worst categories makes the estimate conservative: an arm
    that looks clean here will look at least as clean on the full pool, so a
    win on this subset is not a win bought by an easy sample.
    """
    from prompts import POOL
    hard = [p for p in POOL if p["category"] in ("hostile", "underspecified",
                                                 "dysphoric")]
    rest = [p for p in POOL if p not in hard]
    rng = np.random.default_rng(5)
    take_hard = min(len(hard), int(n * 0.7))
    sel = list(rng.choice(hard, take_hard, replace=False))
    sel += list(rng.choice(rest, min(len(rest), n - take_hard), replace=False))
    return [dict(p) for p in sel]


def run(n: int, workers: int = 4) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = load_done()
    rows = pick_prompts(n)
    todo = [(a, s, r) for a in ARMS for s in ("equanimity", "neutral")
            for r in rows if (a, s, r["prompt_id"]) not in done]
    print(f"arms={list(ARMS)}  prompts={len(rows)}  todo={len(todo)}  workers={workers}")
    t0, fails = time.time(), 0
    with OUT.open("a") as fh, futures.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, a, s, r) for (a, s, r) in todo]
        for i, fut in enumerate(futures.as_completed(futs), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            fails += not rec["ok"]
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} fail={fails} {i/max(time.time()-t0,1e-9):.2f}/s")
    print(f"done in {time.time()-t0:.0f}s fails={fails}")


def analyze() -> dict:
    """d for each arm, plus the incumbent 180-280 range measured on the same prompts."""
    from scipy import stats
    from transformers import AutoTokenizer
    from gate import TOKENIZER_REPO, load_rows, tokenize_counts, index

    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    for r in rows:
        r["tok_reasoning"] = len(tok.encode(r["reasoning"], add_special_tokens=False))
        r["tok_answer"] = len(tok.encode(r["answer"], add_special_tokens=False))

    idx = {(r["arm"], r["stance"], r["prompt_id"]): r for r in rows}
    out = {}
    for arm in ARMS:
        pids = sorted({r["prompt_id"] for r in rows if r["arm"] == arm})
        pids = [p for p in pids
                if (arm, "equanimity", p) in idx and (arm, "neutral", p) in idx]
        if len(pids) < 10:
            out[arm] = {"n": len(pids), "error": "insufficient"}
            continue
        eq = np.array([idx[(arm, "equanimity", p)]["tok_reasoning"] for p in pids], float)
        ne = np.array([idx[(arm, "neutral", p)]["tok_reasoning"] for p in pids], float)
        sd = float(np.sqrt((eq.var(ddof=1) + ne.var(ddof=1)) / 2))
        diff = eq - ne
        se_d = float(diff.std(ddof=1) / np.sqrt(len(pids)) / sd)
        _, p = stats.ttest_rel(eq, ne)
        out[arm] = dict(
            n=len(pids), mean_eq=float(eq.mean()), mean_neu=float(ne.mean()),
            mean_diff=float(diff.mean()), d=float(diff.mean() / sd), se_d=se_d,
            ci90=[float(diff.mean() / sd - 1.65 * se_d),
                  float(diff.mean() / sd + 1.65 * se_d)],
            p=float(p), sd_pooled=sd,
            spread_p10_p90=[float(np.percentile(np.r_[eq, ne], 10)),
                            float(np.percentile(np.r_[eq, ne], 90))],
        )

    # Incumbent, restricted to the same prompts for a like-for-like comparison.
    base = index(tokenize_counts(load_rows()))
    common = sorted({r["prompt_id"] for r in rows})
    pids = [p for p in common
            if (p, "equanimity", "verbose") in base and (p, "neutral", "verbose") in base]
    if len(pids) >= 10:
        eq = np.array([base[(p, "equanimity", "verbose")]["tok_reasoning"] for p in pids], float)
        ne = np.array([base[(p, "neutral", "verbose")]["tok_reasoning"] for p in pids], float)
        sd = float(np.sqrt((eq.var(ddof=1) + ne.var(ddof=1)) / 2))
        diff = eq - ne
        se_d = float(diff.std(ddof=1) / np.sqrt(len(pids)) / sd)
        out["incumbent_180_280_range"] = dict(
            n=len(pids), mean_diff=float(diff.mean()), d=float(diff.mean() / sd),
            se_d=se_d, sd_pooled=sd,
            spread_p10_p90=[float(np.percentile(np.r_[eq, ne], 10)),
                            float(np.percentile(np.r_[eq, ne], 90))])
    return out


def self_test() -> int:
    print("=" * 70); print("BANDTEST SELF-TEST (no network)"); print("=" * 70)
    ok = True

    print("\n[1] Per-prompt target is identical across stances, varies across prompts.")
    ts = [prompt_target(f"p_{i}", 180, 280) for i in range(400)]
    same = all(prompt_target(p, 180, 280) == prompt_target(p, 180, 280)
               for p in ("a", "b", "c"))
    spread = (min(ts) < 195 and max(ts) > 265 and len(set(ts)) > 60)
    ok &= same and spread
    print(f"      deterministic={same}  range=[{min(ts)},{max(ts)}] "
          f"distinct={len(set(ts))}   [{'ok' if same and spread else 'FAIL'}]")

    print("\n[2] Both stances receive the SAME length directive for a prompt.")
    good = True
    for arm in ARMS:
        a = build(arm, "equanimity", "x_1", "Q")
        b = build(arm, "neutral", "x_1", "Q")
        d = verbose_directive(arm, "x_1")
        good &= (d in a and d in b)
    ok &= good
    print(f"      length directive identical across stances   [{'ok' if good else 'FAIL'}]")

    print("\n[3] Target arm preserves the full 180-280 aggregate spread.")
    lo, hi = ARMS["T_full_width"]["lo"], ARMS["T_full_width"]["hi"]
    good = (max(ts) - min(ts)) >= 0.85 * (hi - lo)
    ok &= good
    print(f"      target spread {max(ts)-min(ts)}w vs band {hi-lo}w"
          f"   [{'ok' if good else 'FAIL'}]")

    print("\n[4] Stance directive still never mentions length.")
    good = not any(w in STANCE[s].lower() for s in STANCE
                   for w in ("word", "brief", "long", "concise"))
    ok &= good
    print(f"      clean   [{'ok' if good else 'FAIL'}]")

    print("\n" + "=" * 70)
    print("SELF-TEST", "PASSED" if ok else "FAILED"); print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if a.run:
        run(a.n, a.workers)
    if a.analyze or a.run:
        print(json.dumps(analyze(), indent=2))
