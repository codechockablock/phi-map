"""MDE for the equanimity 2x2, as a SENSITIVITY CURVE over seed SD.

Rewritten 2026-07-29 after two inputs stopped being assumptions:

  1. sigma_seed is now MEASURED. `seed_sd_from_artifacts.py` estimates it from
     the existing eval artifacts: 3.50 pp for GSM8K accuracy, df = 5, 95% CI
     [2.18, 8.57]. README.md:62's 1.5 pp was 2.3x optimistic.
  2. n is now RESOLVED. Every eval artifact on disk reports gsm8k n = 400 and
     jailbreak n = 200. The dispatch's n = 100 and power.py's n_eval = 1319 are
     both wrong for what actually ran.

A point-estimate MDE is deliberately NOT the headline: at df = 5 the upper bound
on sigma is 2.45x the point estimate. What is reported is a curve over sigma_seed
with the measured value and its interval marked, which converts a hidden
dependency into a stated one.

Reuses power.py's validated noncentral-t MDE; section 8 reproduces power.py's own
shipped table with independently written code (standing rule 2).

Run:  python3 mde_seed_unit.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/joseph/phi-map/equanimity_factorial")
import power as P  # noqa: E402

# --- Measured inputs ---------------------------------------------------------
# CORRECTED 2026-07-29 (second pass). The first version put the OBSERVED pooled
# within-cell SD here and then added eval noise on top inside mde(), which
# double-counts: an observed within-cell SD already contains eval noise, because
# power.py's model is Var(y) = sigma_s^2 + sigma_e^2. That inflated every MDE by
# ~20%. GSM8K k=2 was reported as 11.25pp; it is 9.29pp.
#
# What belongs here is the DECONVOLVED training component:
#     sigma_seed = sqrt(observed_pooled_SD^2 - eval_noise^2)
# with eval_noise computed at the n that ACTUALLY ran (400 / 200), not at
# power.py's hardcoded n_eval. mde() then adds eval noise back, once.
#
# Observed pooled within-cell SD (equanimity-terse k=3, neutral-terse k=4,
# df = 5), from seed_sd_from_artifacts.py:
#     GSM8K    3.495 pp, 95% CI [2.182, 8.573]
#     AdvBench 2.890 pp, 95% CI [1.804, 7.087]
# Deconvolved below. Note the LOWER bounds deconvolve to zero: the data cannot
# distinguish perfectly reproducible LoRA training from sigma_seed ~= 8pp.
#
# (point, df, ci_lo, ci_hi) -- all deconvolved training components.
SIGMA_MEASURED = {
    "gsm8k": (2.578, 5, 0.0, 8.242),
    "jb":    (1.963, 5, 0.0, 6.762),
}
# The pilot (pilot.json) reported gsm8k sigma_seed_pp = 2.149 by the same
# deconvolution, but subtracted eval noise computed at n_eval=1319 / p0=0.50
# (1.377pp) when the eval ran at n=400 (2.360pp). Its jailbreak sigma_seed_pp
# came out 0.0 because its observed SD (1.893) fell BELOW its own eval-noise
# term (3.373) -- a variance component pinned at the boundary, which must not be
# read as "training is stable".
# n and base rate as they ACTUALLY ran, read from the artifacts.
EVAL_SPEC = {
    "gsm8k": dict(n=400, p0=0.65, label="GSM8K accuracy"),
    "jb":    dict(n=200, p0=0.10, label="AdvBench compliance"),
}


def mde(sigma_seed, n_eval, p0, k, conditional=False):
    e = 0.0 if conditional else 100.0 * float(np.sqrt(p0 * (1 - p0) / n_eval))
    se = float(np.sqrt(sigma_seed ** 2 + e ** 2)) / np.sqrt(k)
    return se, P._mde_noncentral_t(se, k)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


banner("0. Design reality check -- the 2x2 is not 2/2/2/2")
print("  Eval artifacts present (seed_sd_from_artifacts.py section 0):")
print("    equanimity-terse      3 evals")
print("    equanimity-verbose    0 evals  (adapter exists, never evaluated)")
print("    neutral-terse         4 evals")
print("    neutral-verbose       0 evals  (NO ADAPTER AT ALL)")
print("    base_control          ABSENT")
print()
print("  Consequences that precede any power calculation:")
print("   * Both evaluated cells are TERSE. Factor B is UNESTIMATED -- not")
print("     underpowered, unestimated. Same for the interaction.")
print("   * The only estimable contrast is Factor A at fixed B=terse, on 3 vs 4")
print("     seeds. That is a two-sample comparison, not a 2x2.")
print("   * base_control is missing, so base-vs-adapter cannot be computed at all.")
print("  Everything below describes a contrast that COULD be run, not one the")
print("  current artifacts support.")

banner("1. The measured anchor")
for key, (sd, df, lo, hi) in SIGMA_MEASURED.items():
    print(f"  {EVAL_SPEC[key]['label']:22s} sigma_seed = {sd:5.2f} pp"
          f"   df={df}   95% CI [{lo:.2f}, {hi:.2f}]   upper/point {hi/sd:.2f}x")
print("\n  Compare like with like. README.md:62 assumed sigma_seed = 1.50 pp, i.e.")
print("  the TRAINING component. Measured training component: 2.58 pp -- 1.7x")
print("  the assumption. The OBSERVED pooled within-cell SD, which is what the")
print("  contrast SE is actually built from, is 3.50 pp (2.3x the assumption once")
print("  eval noise is included). Either way the assumption was optimistic, in the")
print("  direction the within-one-cell hit_cap (36 vs 55%) and median-length")
print("  (956 vs 3001 chars) spread already implied.")
print()
print("  But note the lower bounds: both deconvolve to ZERO. At df=5 these data")
print("  cannot distinguish perfectly reproducible LoRA training from sigma_seed")
print("  near 8 pp. The point estimate is usable; the interval is not informative")
print("  at its lower end, and no design decision should lean on it.")

banner("2. SENSITIVITY CURVE -- MDE (pp) vs sigma_seed, at the resolved n")
for key, spec in EVAL_SPEC.items():
    sd = SIGMA_MEASURED[key][0]
    e = 100 * np.sqrt(spec["p0"] * (1 - spec["p0"]) / spec["n"])
    print(f"\n  {spec['label']}  (n={spec['n']}, p0={spec['p0']}, "
          f"eval SD/run = {e:.2f} pp)")
    print(f"    {'sigma_seed':>12s}" + "".join(f"{f'k={k}':>9s}"
                                              for k in (2, 3, 4, 6, 8)))
    print("    " + "-" * 58)
    for s in sorted({1.0, 1.5, 2.0, round(sd, 3), 4.0, 6.0, 8.0, 10.0}):
        mark = ("  <- MEASURED" if abs(s - round(sd, 3)) < 1e-9
                else "  <- README assumption" if abs(s - 1.5) < 1e-9 else "")
        cells = "".join(f"{mde(s, spec['n'], spec['p0'], k)[1]:8.2f}"
                        for k in (2, 3, 4, 6, 8))
        print(f"    {s:11.2f}pp{cells}{mark}")

banner("3. MDE band from propagating the sigma_seed interval")
print("  An interval on sigma gives an interval on MDE. This mirrors")
print("  GATE_RESULT T4/T7, which did the same propagation for refusal_margin.\n")
print(f"  {'outcome':22s}{'k':>3s}{'at sigma_lo':>13s}{'at point':>11s}{'at sigma_hi':>13s}")
print("  " + "-" * 64)
for key, spec in EVAL_SPEC.items():
    sd, df, lo, hi = SIGMA_MEASURED[key]
    for k in (2, 3):
        a = mde(lo, spec["n"], spec["p0"], k)[1]
        b = mde(sd, spec["n"], spec["p0"], k)[1]
        c = mde(hi, spec["n"], spec["p0"], k)[1]
        print(f"  {spec['label']:22s}{k:>3d}{a:12.2f}pp{b:10.2f}pp{c:12.2f}pp")
print("\n  GSM8K at k=2: MDE lies in [6.34, 22.82] pp, point estimate 9.34 pp -- a")
print("  3.6x range. The lower bound EQUALS the section-4 floor, because sigma_lo")
print("  deconvolves to zero; that is an internal consistency check, not a")
print("  coincidence. The imprecision cannot be narrowed by more eval items, only")
print("  by more seeds -- which is also what would narrow the sigma interval.")

banner("4. Irreducible floor at the resolved n (sigma_seed -> 0)")
print(f"  {'outcome':22s}{'k':>3s}{'eval SD/run':>13s}{'SE':>9s}{'MDE floor':>12s}")
print("  " + "-" * 60)
for key, spec in EVAL_SPEC.items():
    e = 100 * np.sqrt(spec["p0"] * (1 - spec["p0"]) / spec["n"])
    for k in (2, 3):
        se, m = mde(0.0, spec["n"], spec["p0"], k)
        print(f"  {spec['label']:22s}{k:>3d}{e:12.2f}pp{se:8.2f}pp{m:11.2f}pp")
print("\n  At the REAL n the floor sits far below the measured-sigma MDE. So unlike")
print("  the n=100 case reported at the checkpoint, the binding constraint is now")
print("  unambiguously sigma_seed and k, not the eval set. Correcting n moved the")
print("  diagnosis: n was load-bearing after all, just not in the direction assumed.")

banner("5. Seeds required at the MEASURED sigma_seed")
print("  Read k as 4k training runs for a full 2x2. The study has 8 adapters over")
print("  3 cells and 7 evals over 2 cells.\n")
print(f"  {'target effect':>22s}{'GSM8K k':>10s}{'AdvBench k':>13s}")
print("  " + "-" * 46)
for label, eff in (("1.25 pp (prior cap.)", 1.25), ("3 pp", 3.0), ("5 pp", 5.0),
                   ("10 pp", 10.0), ("16.7 pp (prior jb)", 16.7)):
    out = []
    for key, spec in EVAL_SPEC.items():
        sd = SIGMA_MEASURED[key][0]
        need = next((k for k in range(2, 5001)
                     if mde(sd, spec["n"], spec["p0"], k)[1] <= eff), None)
        out.append(need if need else ">5000")
    print(f"  {label:>22s}{str(out[0]):>10s}{str(out[1]):>13s}")

banner("6. The pooling fallback is confounded AND cross-instrument")
print("  base-vs-all-adapters-pooled is the tempting fallback: it has the most")
print("  power, because greedy decode makes the base deterministic, so it is a")
print("  one-sample t against a fixed constant with df = n_adapters - 1 rather")
print("  than 4(k-1). It fails for two independent reasons.")
print()
print("   (a) CONFOUNDED BY CONSTRUCTION. Pooling all cells answers 'does")
print("       fine-tuning on this data change X', not 'does stance change X'.")
print("       And with the current artifacts it could not answer even that:")
print("       both evaluated cells are terse, and base_control is absent.")
print()
print("   (b) MEASURED WITH TWO DIFFERENT INSTRUMENTS -- the decisive reason.")
print("       The base model emits no 'ANSWER:' marker, so extract_answer falls")
print("       back to scoring the WHOLE response (train_eval.py:166-168).")
print("       Adapters emit the marker and are scored on the EXTRACTED answer.")
print("       So one arm is whole-response and the other is post-marker text.")
print("       Measured size of that instrument difference, same adapters, same")
print("       generations: 12.0% vs 3.5% (3.43x) and 10.0% vs 4.5% (2.22x) --")
print("       and the multiplier differs between two seeds of the SAME cell.")
print()
print("  The instrument change is larger than any effect the design can resolve.")
print("  That closes off the fallback rather than merely caveating it.")

banner("7. GSM8K accuracy is no longer a clean construct either")
print("  Step 1 graded gsm8k_accuracy 'valid construct, structurally")
print("  underpowered'. The artifacts revise that. truncated_frac by cell:")
print("    equanimity-terse  [77.0, 47.0, 81.2]        mean 68.4%")
print("    neutral-terse     [24.0, 37.2, 46.8, 25.2]  mean 33.3%")
print("  A 35 pp between-cell truncation gap along Factor A, with accuracy 5.9 pp")
print("  LOWER in the higher-truncation cell. train_eval.py:119-120 sets the")
print("  criterion: truncated_frac 'must come back near zero; if it does not, the")
print("  capability numbers are not usable'. It is 24-81%. The apparent capability")
print("  deficit and the truncation gap are not separable from these artifacts.")

banner("8. Reproduce power.py's shipped table with independent code")
for oc in ("gsm8k", "jailbreak"):
    for k in (2, 3):
        dp = P.design_point(oc, 1.5, k)
        mine = mde(1.5, P.OUTCOMES[oc]["n_eval"], P.OUTCOMES[oc]["p0"], k)[1]
        print(f"  {oc:10s} k={k}  power.py={dp.mde_pp:6.3f}pp  "
              f"independent={mine:6.3f}pp  "
              f"{'MATCH' if abs(dp.mde_pp - mine) < 1e-6 else 'DISAGREE'}")
print("\n  These use power.py's own n_eval (1319 / 200), which section 1 shows is")
print("  wrong for GSM8K as run. The check validates the ESTIMATOR, not the")
print("  design constants.")
