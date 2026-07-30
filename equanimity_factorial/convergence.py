"""Is the orthogonality estimate STABLE below the bound, or just currently below it?

A TOST that clears |d| < 0.2 at the n you happened to stop at is not the same
claim as "the effect is small". Two ways the first can be true while the second
is false:

  * **Drift.** The point estimate climbs with n and the interval is still wide
    enough to hide it. Stopping early looks like a pass.
  * **Luck.** The estimate is genuinely centred near the bound and this
    particular sample landed on the friendly side of it.

Both are invisible to a single end-of-run number and both are caught by looking
at the estimate as a function of sample size. This module reports:

  1. `d` over nested prefixes n' = 40, 80, ... N, so drift is visible as a trend
     rather than inferred from one endpoint;
  2. a regression of d on 1/sqrt(n'), whose slope should be indistinguishable
     from zero -- a real drift shows up here even when every individual prefix
     sits under the bound;
  3. subsample spread at each n', to confirm the interval is shrinking at the
     ~1/sqrt(n) rate a well-behaved estimator gives;
  4. the margin between the final CI edge and the bound, because "passes with
     the CI edge at 0.198" and "passes with the CI edge at 0.09" are different
     results and only one of them is robust to a few more samples.

    python3 convergence.py --self-test
    python3 convergence.py --arm T_full_width
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BOUND = 0.2


def d_and_se(eq: np.ndarray, ne: np.ndarray) -> tuple[float, float]:
    sd = float(np.sqrt((eq.var(ddof=1) + ne.var(ddof=1)) / 2))
    if sd <= 0:
        return 0.0, float("inf")
    diff = eq - ne
    return float(diff.mean() / sd), float(diff.std(ddof=1) / np.sqrt(len(diff)) / sd)


def convergence(eq: np.ndarray, ne: np.ndarray, categories: list[str] | None = None,
                n_boot: int = 400, seed: int = 0) -> dict:
    """Stability of d, via tests that can actually fail.

    A first version of this regressed the mean of *subsamples drawn without
    replacement* against 1/sqrt(n) and called a non-zero slope "drift". That is
    not a test: subsampling without replacement from a fixed set converges to the
    full-sample estimate by construction, so the trajectory's shape is an artifact
    of the resampling scheme and the "drift" it detects is guaranteed. The same
    bug made the shrinkage check compare against a subsample size where the
    spread is mechanically zero.

    What is left is three things that can come out either way:

      * **Bootstrap spread vs sample size** (with replacement, so it estimates
        real sampling variability): must fall as ~1/sqrt(n).
      * **Prefix trajectory in fixed prompt order**: non-tautological, because
        heterogeneity across the pool shows up as a wandering estimate even
        though the endpoint is fixed.
      * **Leave-one-category-out**: is the verdict carried by one slice?

    Plus the margin between the final interval edge and the bound, which is the
    quantity that actually distinguishes "passed" from "passed robustly".
    """
    rng = np.random.default_rng(seed)
    N = len(eq)
    sizes = [s for s in (40, 60, 80, 120, 160, 200, 260, 320, 400, 500, 650)
             if s <= N] or [N]

    traj = []
    for n in sizes:
        ds = []
        for _ in range(n_boot):
            idx = rng.integers(0, N, n)          # with replacement
            ds.append(d_and_se(eq[idx], ne[idx])[0])
        prefix_d, _ = d_and_se(eq[:n], ne[:n]) if n >= 3 else (np.nan, np.nan)
        traj.append(dict(n=int(n), boot_mean_d=float(np.mean(ds)),
                         boot_sd_d=float(np.std(ds)), prefix_d=float(prefix_d)))

    # Shrinkage: bootstrap sd should fall like 1/sqrt(n) between the smallest and
    # largest evaluated size.
    first, last = traj[0], traj[-1]
    shrink_obs = first["boot_sd_d"] / last["boot_sd_d"] if last["boot_sd_d"] > 0 else np.nan
    shrink_exp = np.sqrt(last["n"] / first["n"])

    # Prefix wander: how far does the running estimate stray once it has enough
    # data to be meaningful? Large wander means the pool is heterogeneous.
    tail = [t["prefix_d"] for t in traj if t["n"] >= max(80, N // 4)]
    wander = float(max(tail) - min(tail)) if len(tail) > 1 else 0.0

    loco = {}
    if categories is not None and len(set(categories)) > 1:
        cats = np.array(categories)
        for c in sorted(set(categories)):
            keep = cats != c
            if keep.sum() >= 20:
                loco[c] = float(d_and_se(eq[keep], ne[keep])[0])

    d_final, se_final = d_and_se(eq, ne)
    ci = (d_final - 1.65 * se_final, d_final + 1.65 * se_final)
    boot_full = np.array([d_and_se(*(lambda i: (eq[i], ne[i]))(rng.integers(0, N, N)))[0]
                          for _ in range(n_boot)])
    return dict(
        n_total=int(N), trajectory=traj,
        d_final=d_final, se_final=se_final, ci90=[float(ci[0]), float(ci[1])],
        boot_ci90=[float(np.percentile(boot_full, 5)),
                   float(np.percentile(boot_full, 95))],
        bound=BOUND,
        margin_to_bound=float(BOUND - max(abs(ci[0]), abs(ci[1]))),
        equivalent=bool(abs(ci[0]) < BOUND and abs(ci[1]) < BOUND),
        shrink_observed=float(shrink_obs), shrink_expected=float(shrink_exp),
        shrink_ok=bool(np.isfinite(shrink_obs)
                       and 0.6 < shrink_obs / shrink_exp < 1.7),
        prefix_wander=wander,
        prefix_stable=bool(wander < 0.12),
        leave_one_category_out=loco,
        loco_max_abs=float(max((abs(v) for v in loco.values()), default=0.0)),
    )


def report(res: dict, label: str = "") -> None:
    print(f"\n--- convergence: {label} (N={res['n_total']}) ---")
    print(f"    {'n':>6} {'boot mean d':>12} {'boot sd':>9} {'prefix d':>10}")
    for t in res["trajectory"]:
        print(f"    {t['n']:>6} {t['boot_mean_d']:>+12.3f} {t['boot_sd_d']:>9.3f} "
              f"{t['prefix_d']:>+10.3f}")
    print(f"    bootstrap spread shrank {res['shrink_observed']:.2f}x vs "
          f"{res['shrink_expected']:.2f}x expected -> "
          f"{'ok' if res['shrink_ok'] else 'ANOMALOUS'}")
    print(f"    prefix wander (n>=N/4) = {res['prefix_wander']:.3f} -> "
          f"{'stable' if res['prefix_stable'] else 'WANDERING'}")
    if res["leave_one_category_out"]:
        pairs = "  ".join(f"{k[:9]}={v:+.3f}"
                          for k, v in res["leave_one_category_out"].items())
        print(f"    leave-one-category-out: {pairs}")
        print(f"      max |d| without any one category = {res['loco_max_abs']:.3f}")
    print(f"    final d = {res['d_final']:+.3f}  CI90 "
          f"[{res['ci90'][0]:+.3f}, {res['ci90'][1]:+.3f}]  "
          f"bootstrap CI90 [{res['boot_ci90'][0]:+.3f}, {res['boot_ci90'][1]:+.3f}]")
    print(f"    margin to bound = {res['margin_to_bound']:+.3f}  -> "
          f"{'EQUIVALENT' if res['equivalent'] else 'NOT EQUIVALENT'}")


def from_bandtest(arm: str) -> dict:
    from transformers import AutoTokenizer
    from gate import TOKENIZER_REPO
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    path = Path(__file__).parent / "data" / "bandtest.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok") and r["arm"] == arm]
    idx = {(r["stance"], r["prompt_id"]): r for r in rows}
    pids = sorted({p for (_, p) in idx}
                  & {p for (s, p) in idx if s == "equanimity"}
                  & {p for (s, p) in idx if s == "neutral"})
    eq = np.array([len(tok.encode(idx[("equanimity", p)]["reasoning"],
                                  add_special_tokens=False)) for p in pids], float)
    ne = np.array([len(tok.encode(idx[("neutral", p)]["reasoning"],
                                  add_special_tokens=False)) for p in pids], float)
    return convergence(eq, ne)


def self_test() -> int:
    print("=" * 70); print("CONVERGENCE SELF-TEST (synthetic)"); print("=" * 70)
    ok = True
    rng = np.random.default_rng(1)

    print("\n[1] Truly-zero effect: must read stable, and equivalent at large n.")
    base = rng.normal(240, 22, 700)
    eq, ne = base + rng.normal(0, 12, 700), base + rng.normal(0, 12, 700)
    cats = [f"c{i % 5}" for i in range(700)]
    r = convergence(eq, ne, cats)
    good = r["equivalent"] and r["prefix_stable"] and r["shrink_ok"]
    ok &= good
    print(f"      d={r['d_final']:+.3f} wander={r['prefix_wander']:.3f} "
          f"shrink_ok={r['shrink_ok']} equiv={r['equivalent']}"
          f"   [{'ok' if good else 'FAIL'}]")

    print("\n[2] Real +0.32 effect: must be caught, not certified.")
    eq2 = base + rng.normal(7, 12, 700)
    r2 = convergence(eq2, ne)
    good = not r2["equivalent"]
    ok &= good
    print(f"      d={r2['d_final']:+.3f} equiv={r2['equivalent']}"
          f"   [{'ok' if good else 'FAIL'}]")

    print("\n[3] Borderline +0.19: passes the bound but with a thin margin.")
    print("    The margin is the point -- 'passes' and 'passes robustly' differ.")
    eq3 = base + rng.normal(0.19 * 17, 12, 700)
    r3 = convergence(eq3, ne)
    thin = r3["margin_to_bound"] < 0.08
    ok &= thin
    print(f"      d={r3['d_final']:+.3f} margin={r3['margin_to_bound']:+.3f}"
          f"   [{'ok -- flagged as thin' if thin else 'FAIL'}]")

    print("\n[4] Bootstrap spread must shrink at ~1/sqrt(n).")
    good = r["shrink_ok"]
    ok &= good
    print(f"      observed {r['shrink_observed']:.2f}x vs expected "
          f"{r['shrink_expected']:.2f}x   [{'ok' if good else 'FAIL'}]")

    print("\n[5] Heterogeneous pool: one category carrying the effect must show up")
    print("    as prefix wander, even though the endpoint alone looks fine.")
    b2 = rng.normal(240, 22, 600)
    off = np.array([28.0 if i < 120 else 0.0 for i in range(600)])
    r5 = convergence(b2 + off + rng.normal(0, 12, 600),
                     b2 + rng.normal(0, 12, 600),
                     ["hot" if i < 120 else f"c{i % 4}" for i in range(600)])
    caught = (not r5["prefix_stable"]) or (not r5["equivalent"])
    ok &= caught
    print(f"      d={r5['d_final']:+.3f} wander={r5['prefix_wander']:.3f} "
          f"stable={r5['prefix_stable']}   [{'ok -- caught' if caught else 'FAIL'}]")

    print("\n" + "=" * 70)
    print("SELF-TEST", "PASSED" if ok else "FAILED"); print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--arm", type=str, default=None)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if a.arm:
        report(from_bandtest(a.arm), a.arm)
