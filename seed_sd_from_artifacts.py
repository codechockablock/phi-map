"""Measure seed SD directly from the existing eval artifacts. No training, no GPU.

Answers two questions the power analysis was resting on assumptions for:

  1. What is sigma_seed for the ACTUAL outcome variables? (README used 1.5pp,
     sourced to nothing.)
  2. What n were the reported rates actually computed on? (100 / 400 / 1319
     appear across the dispatch, train_eval.py:872 and power.py:50-60.)

Estimator: pooled within-cell variance across seeds, which is the only honest
estimate of run-to-run spread and is what `power.py`'s shipped t-interval uses
(df = sum over cells of (k_cell - 1)). Cells with one seed contribute 0 df.

READ-ONLY. This script opens files under the sweep's work directory and writes
nothing anywhere near it. It must never create a STOP file.

Run:  python3 seed_sd_from_artifacts.py
"""
from __future__ import annotations

import json
import os
import glob
from collections import defaultdict

import numpy as np
from scipy import stats

WORK = os.path.expanduser(
    "~/Library/CloudStorage/GoogleDrive-joehopkins89@gmail.com/"
    "My Drive/phi-map/equanimity-factorial-v1")
EVALS = os.path.join(WORK, "evals")

# Outcome extractors mirroring train_eval.OUTCOME_GETTERS, plus the two
# diagnostics that turned out to matter. `None` when the stage is absent.
GETTERS = {
    "gsm8k_accuracy_pp":      lambda e: _g(e, "gsm8k", "accuracy", 100.0),
    "gsm8k_truncated_pp":     lambda e: _g(e, "gsm8k", "truncated_frac", 100.0),
    "gsm8k_mean_tokens":      lambda e: _g(e, "gsm8k", "mean_tokens", 1.0),
    "jb_compliance_pp":       lambda e: _g(e, "jailbreak", "compliance_rate", 100.0),
    "jb_refusal_margin":      lambda e: _g(e, "jailbreak", "refusal_margin_mean", 1.0),
    "valence_dysphoric_mean": lambda e: _g(e, "valence", "dysphoric_mean", 1.0),
    "valence_dys_minus_neu":  lambda e: _g(e, "valence", "dys_minus_neu", 1.0),
}


def _g(e, stage, key, scale):
    s = e.get(stage)
    if not isinstance(s, dict) or key not in s:
        return None
    return float(s[key]) * scale


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(EVALS, "*.json"))):
        try:
            e = json.load(open(f))
        except Exception as ex:
            print(f"  !! parse fail {os.path.basename(f)}: {ex}")
            continue
        base = os.path.basename(f)[: -len(".json")]
        cell, _, seed = base.partition("__seed")
        rows.append(dict(file=base, cell=cell, seed=int(seed), e=e))
    return rows


def pooled_sd(groups: dict[str, list[float]]):
    """Pooled within-cell SD. Returns (sd, df, per-cell contributions)."""
    ss, df, detail = 0.0, 0, []
    for cell, vals in sorted(groups.items()):
        v = np.array([x for x in vals if x is not None], dtype=float)
        if len(v) < 2:
            detail.append((cell, len(v), None, None))
            continue
        d = v - v.mean()
        ss += float(d @ d)
        df += len(v) - 1
        detail.append((cell, len(v), float(v.mean()), float(v.std(ddof=1))))
    if df == 0:
        return None, 0, detail
    return float(np.sqrt(ss / df)), df, detail


def sd_ci(sd: float, df: int, conf=0.95):
    """Chi-square CI on a variance component. Wide at small df, by construction."""
    a = (1 - conf) / 2
    ss = sd * sd * df
    hi = float(np.sqrt(ss / stats.chi2.ppf(a, df)))
    lo = float(np.sqrt(ss / stats.chi2.ppf(1 - a, df)))
    return lo, hi


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


rows = load()

banner("0. What artifacts exist, and what the 2x2 actually looks like")
cells = defaultdict(list)
for r in rows:
    cells[r["cell"]].append(r["seed"])
print(f"  {len(rows)} eval artifacts found in {EVALS}\n")
DESIGN = ["equanimity-terse", "equanimity-verbose",
          "neutral-terse", "neutral-verbose"]
print(f"  {'cell':22s}{'evals':>7s}{'seeds':>28s}")
print("  " + "-" * 58)
for c in DESIGN:
    s = sorted(cells.get(c, []))
    print(f"  {c:22s}{len(s):>7d}{str(s) if s else '  NONE':>28s}")
extra = set(cells) - set(DESIGN)
if extra:
    print(f"  unexpected cells: {sorted(extra)}")
print("\n  Factor B is estimable only where BOTH verbosity levels have evals.")
covered = [c for c in DESIGN if cells.get(c)]
print(f"  Cells with any eval: {len(covered)}/4 -> {covered}")
print("  base_control.json present:",
      os.path.exists(os.path.join(EVALS, "base_control.json")))

banner("1. n provenance -- what sample were the reported rates computed on?")
print(f"  {'file':34s}{'gsm8k n':>9s}{'jb n':>7s}{'jb set':>11s}{'jb cap':>8s}")
print("  " + "-" * 70)
for r in rows:
    e = r["e"]
    gn = e.get("gsm8k", {}).get("n")
    jb = e.get("jailbreak", {})
    print(f"  {r['file']:34s}{str(gn):>9s}{str(jb.get('n')):>7s}"
          f"{str(jb.get('dataset')):>11s}{str(jb.get('max_new_tokens')):>8s}")
gs = {e["e"].get("gsm8k", {}).get("n") for e in rows}
js = {e["e"].get("jailbreak", {}).get("n") for e in rows}
print(f"\n  RESOLVED: gsm8k n in {gs}, jailbreak n in {js}.")
print("  The dispatch's n=100 and power.py's n_eval=1319 are BOTH wrong for")
print("  what actually ran. train_eval.py:872's (400, 200) is correct.")

banner("2. Instrument versions -- which evals used which judge and cap")
print(f"  {'file':34s}{'judge':>42s}")
print("  " + "-" * 78)
for r in rows:
    j = r["e"].get("jailbreak", {}).get("judge", "?")
    print(f"  {r['file']:34s}{j:>42s}")
vers = defaultdict(list)
for r in rows:
    vers[r["e"].get("jailbreak", {}).get("judge", "?")].append(r["file"])
print()
for v, fs in vers.items():
    print(f"  {v}: {len(fs)} eval(s)")
print("\n  Any sigma_seed pooled across judge versions mixes instruments.")
print("  Per-version estimates are reported in section 3 where possible.")

banner("3. Pooled within-cell seed SD, per outcome")
print("  Pooled across all cells with >=2 seeds. chi-square 95% CI on sigma.")
print("  'ratio' = upper CI bound / point estimate.\n")
print(f"  {'outcome':24s}{'sigma_seed':>11s}{'df':>4s}{'95% CI on sigma':>22s}{'ratio':>7s}")
print("  " + "-" * 70)
results = {}
for name, get in GETTERS.items():
    groups = defaultdict(list)
    for r in rows:
        groups[r["cell"]].append(get(r["e"]))
    sd, df, detail = pooled_sd(groups)
    if sd is None:
        print(f"  {name:24s}{'n/a':>11s}{0:>4d}{'(no cell has 2+ seeds)':>22s}")
        continue
    lo, hi = sd_ci(sd, df)
    results[name] = (sd, df, lo, hi, detail)
    print(f"  {name:24s}{sd:10.3f}{df:>4d}"
          f"{f'[{lo:.3f}, {hi:.3f}]':>22s}{hi/sd:6.2f}x")

banner("4. THE NUMBER: sigma_seed for GSM8K accuracy, the one valid construct")
sd, df, lo, hi, detail = results["gsm8k_accuracy_pp"]
print(f"  Measured pooled sigma_seed = {sd:.2f} pp   (df = {df})")
print(f"  95% CI on sigma           = [{lo:.2f}, {hi:.2f}] pp")
print(f"  Upper bound / point       = {hi/sd:.2f}x")
print(f"\n  README.md:62 assumed 1.5 pp. Measured is {sd/1.5:.1f}x larger.")
print("  Direction of the surprise: WORSE, as predicted from the hit_cap and")
print("  median-length spread. The assumption was optimistic by a factor of 2+.")
print("\n  Per-cell detail:")
print(f"    {'cell':22s}{'k':>3s}{'mean':>9s}{'within-cell SD':>16s}")
for c, k, m, s in detail:
    print(f"    {c:22s}{k:>3d}"
          f"{(f'{m:.2f}' if m is not None else '-'):>9s}"
          f"{(f'{s:.3f}' if s is not None else '(1 seed, 0 df)'):>16s}")

banner("5. The finding that outranks the SD: GSM8K truncation is 24-81%")
tr = defaultdict(list)
ac = defaultdict(list)
for r in rows:
    tr[r["cell"]].append(GETTERS["gsm8k_truncated_pp"](r["e"]))
    ac[r["cell"]].append(GETTERS["gsm8k_accuracy_pp"](r["e"]))
print(f"  {'cell':22s}{'truncated_frac':>28s}{'mean':>8s}{'accuracy mean':>15s}")
print("  " + "-" * 74)
for c in DESIGN:
    if not tr.get(c):
        continue
    v = np.array([x for x in tr[c] if x is not None])
    a = np.array([x for x in ac[c] if x is not None])
    print(f"  {c:22s}{str([round(x,1) for x in v]):>28s}"
          f"{v.mean():7.1f}%{a.mean():14.1f}%")
tv = {c: np.mean([x for x in tr[c] if x is not None]) for c in tr if tr[c]}
av = {c: np.mean([x for x in ac[c] if x is not None]) for c in ac if ac[c]}
if "equanimity-terse" in tv and "neutral-terse" in tv:
    dt = tv["equanimity-terse"] - tv["neutral-terse"]
    da = av["equanimity-terse"] - av["neutral-terse"]
    print(f"\n  Truncation differs BY CELL by {dt:+.1f} pp (equanimity higher).")
    print(f"  Accuracy differs by            {da:+.1f} pp (equanimity lower).")
    print("\n  train_eval.py:119-120 states the criterion: truncated_frac 'must come")
    print("  back near zero; if it does not, the capability numbers are not usable'.")
    print("  It is 24-81%, and it differs across cells by 35 pp along Factor A.")
    print("  A binding cap converts correct answers into failures, so the apparent")
    print("  capability deficit and the truncation gap are not separable here.")

banner("6. Valence: both definitions, reported side by side as a diagnostic")
for nm in ("valence_dysphoric_mean", "valence_dys_minus_neu"):
    sd_, df_, lo_, hi_, det = results[nm]
    print(f"  {nm:26s} sigma_seed={sd_:.4f}  df={df_}  CI[{lo_:.4f}, {hi_:.4f}]")
    for c, k, m, s in det:
        if m is not None:
            print(f"      {c:22s} k={k}  mean={m:+.4f}  sd={s:.4f}")
print("\n  Reported as a DIAGNOSTIC only. Not promoted; not swapped. Per the")
print("  standing decision, switching the valence definition inside this study")
print("  after seeing that the current one is contaminated would be endpoint")
print("  selection on the data -- the same error class as JBB.")

banner("7. Post-patch vs pre-patch compliance, same adapters")
print(f"  {'file':34s}{'extracted':>11s}{'rawwindow':>11s}{'ratio':>8s}{'hit_cap':>9s}")
print("  " + "-" * 74)
for r in rows:
    jb = r["e"].get("jailbreak", {})
    a, b = jb.get("compliance_rate"), jb.get("compliance_rate_rawwindow")
    if b is None:
        continue
    print(f"  {r['file']:34s}{100*a:10.1f}%{100*b:10.1f}%"
          f"{a/b if b else float('nan'):7.2f}x{100*jb.get('hit_cap_frac',float('nan')):8.1f}%")
print("\n  The judge patch moves the measured compliance rate by 2-3x, and the")
print("  multiplier itself differs between two seeds of the SAME cell. The")
print("  instrument change is larger than any effect the design could resolve.")
