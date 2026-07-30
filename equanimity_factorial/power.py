"""Power analysis for the 2x2 equanimity factorial: what can this design detect?

This runs BEFORE any GPU spend. Its job is to answer one question the handoff
asserts rather than measures: is >=3 seeds per cell enough?

The handoff calls >=3 seeds "non-negotiable for a real result". That is right in
direction and wrong in magnitude, and the difference is the whole experiment.
Three seeds per cell is adequate to detect the jailbreak-sized effect the prior
work reported (~17pp) and hopeless for the capability-sized effect it reported
(~1.25pp). Those two facts come out of the same design, so the design has to be
reported per-outcome, not as one number.

Variance model for one (cell, seed) run's measured accuracy:

    y = mu + alpha_i + beta_j + (alpha*beta)_ij + s + e

      s ~ N(0, sigma_seed^2)         LoRA run-to-run variation in TRUE accuracy
      e ~ Binomial noise / n_eval    finite held-out evaluation set

Both components matter and they are routinely confused. Doubling the eval set
shrinks only `e`; adding seeds shrinks both. A design that reports a tight CI
from a 1319-item GSM8K set while running one seed is reporting `e` and calling
it the result.

The analytic MDE has a closed form, but the estimator that actually ships is the
seed bootstrap in `bootstrap_effects`, so that is what gets validated here --
against synthetic worlds where the answer is known, including a null world where
the honest answer is "nothing is going on". An estimator that finds effects there
is leaking, and no amount of downstream care recovers from it.

Run:
    python3 power.py --self-test      # validation, no GPU, no network
    python3 power.py --report         # the design table
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np

# --- Outcome definitions ------------------------------------------------------
# n_eval is the held-out set size; p0 is the expected base rate, which sets the
# binomial noise floor. Both are properties of the EVAL, not of the training.

OUTCOMES = {
    "gsm8k":      dict(n_eval=1319, p0=0.50, label="GSM8K accuracy"),
    "humaneval":  dict(n_eval=164,  p0=0.55, label="HumanEval pass@1"),
    "mbpp":       dict(n_eval=500,  p0=0.50, label="MBPP pass@1"),
    "ifeval":     dict(n_eval=541,  p0=0.65, label="IFEval instruction-following"),
    # AdvBench, subsampled to 200. This tracks the set that ACTUALLY runs and
    # has already been wrong once in each direction: 200 (assumed AdvBench) ->
    # 100 (AdvBench gated, fell back to JailbreakBench) -> 200 (terms accepted).
    # An eval-noise floor computed from a set you cannot load is fiction, so
    # `preflight()` checks this number against the real row count before any
    # GPU spend.
    "jailbreak":  dict(n_eval=200,  p0=0.35, label="Harmful-compliance rate"),
}

# Effect sizes worth naming, in percentage points. The two "prior" entries are
# what the Maresova post reported; see PREMISE.md for why neither is
# distinguishable from zero at that post's own sample sizes.
REFERENCE_EFFECTS = {
    "prior_capability": 1.25,   # 75/80 -> 76/80. One question.
    "prior_jailbreak": 16.7,    # 5/12 -> 3/12. Two prompts.
    "modest": 3.0,
    "large": 10.0,
}

Z_ALPHA = 1.959963985  # two-sided 0.05
Z_POWER = 0.8416212336  # 80% power


@dataclass
class DesignPoint:
    outcome: str
    sigma_seed_pp: float
    n_seeds: int
    eval_noise_pp: float
    total_sd_pp: float
    se_main_effect_pp: float
    mde_pp: float

    def detects(self, effect_pp: float) -> bool:
        return effect_pp >= self.mde_pp


def eval_noise_pp(n_eval: int, p0: float) -> float:
    """Binomial SD of a measured rate, in percentage points."""
    return 100.0 * float(np.sqrt(p0 * (1.0 - p0) / n_eval))


def design_point(outcome: str, sigma_seed_pp: float, n_seeds: int) -> DesignPoint:
    """Analytic MDE for a 2x2 main effect with `n_seeds` replicates per cell.

    A main effect contrasts 2 cells against 2 cells, i.e. 2*k runs against 2*k
    runs, so Var(effect) = sigma^2/k and SE = sigma/sqrt(k). Note this does NOT
    depend on there being four cells -- the factorial buys you both main effects
    from the same runs, which is exactly why it is the right design even when
    each individual contrast is expensive.
    """
    spec = OUTCOMES[outcome]
    e = eval_noise_pp(spec["n_eval"], spec["p0"])
    total = float(np.sqrt(sigma_seed_pp**2 + e**2))
    se = total / np.sqrt(n_seeds)
    return DesignPoint(
        outcome=outcome,
        sigma_seed_pp=sigma_seed_pp,
        n_seeds=n_seeds,
        eval_noise_pp=e,
        total_sd_pp=total,
        se_main_effect_pp=se,
        mde_pp=_mde_noncentral_t(se, n_seeds),
    )


def _mde_noncentral_t(se: float, n_seeds: int, power: float = 0.80) -> float:
    """Effect detectable at 80% power by the t-test that actually ships.

    The normal-approximation MDE, (z_alpha + z_power) * SE, is what gets quoted
    by default and it is optimistic at these sample sizes: with k=3 the pooled
    variance carries only 8 degrees of freedom, so the critical value is 2.31
    rather than 1.96 and the variance estimate is itself noisy. Using the
    noncentral t keeps the design table honest about small-k costs.
    """
    from scipy import stats as _st
    if n_seeds < 2:
        return float("inf")
    df = 4 * (n_seeds - 1)
    tcrit = _st.t.ppf(0.975, df)
    lo, hi = 1e-6, 1e4
    for _ in range(200):  # bisection on ncp
        mid = (lo + hi) / 2
        ncp = mid / se
        got = (1 - _st.nct.cdf(tcrit, df, ncp)) + _st.nct.cdf(-tcrit, df, ncp)
        if got < power:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def seeds_required(outcome: str, sigma_seed_pp: float, effect_pp: float,
                   cap: int = 500) -> int | None:
    """Smallest seeds/cell whose MDE is at or below `effect_pp`."""
    for k in range(1, cap + 1):
        if design_point(outcome, sigma_seed_pp, k).mde_pp <= effect_pp:
            return k
    return None


# --- The estimator that ships -------------------------------------------------

def _contrast_weights() -> dict[str, np.ndarray]:
    """Orthogonal +/-1 contrast weights over the 2x2 cell grid [content, verbosity]."""
    c = np.array([[-1.0, -1.0], [1.0, 1.0]])      # equanimity vs neutral
    v = np.array([[-1.0, 1.0], [-1.0, 1.0]])      # verbose vs terse
    return {"content_A": c, "verbosity_B": v, "interaction_AB": c * v}


def bootstrap_effects(y: np.ndarray, n_perm: int = 2000,
                      rng: np.random.Generator | None = None) -> dict:
    """2x2 main effects and interaction with t-based intervals on seed replicates.

    `y` has shape (2, 2, k): [content, verbosity, seed], content index 1 =
    equanimity, verbosity index 1 = verbose.

    The name is kept for continuity with the handoff, which proposed "a formal
    2-way ANOVA or a simple bootstrap over seeds". The bootstrap is NOT fine, and
    the null world in `validate_null_world` is what says so: at the handoff's own
    k=3, a percentile bootstrap over seeds rejects a true null ~25% of the time
    per contrast instead of 5%, because with three replicates the resampling
    distribution badly understates the spread of the sampling distribution. Three
    contrasts at that rate make a spurious "finding" the expected outcome of a
    null experiment.

    What ships instead is the exact small-sample thing: a pooled-variance t
    interval on the contrast, df = 4(k-1). The replicate unit is still the seed
    -- that part of the handoff's instinct was right, and bootstrapping over eval
    items instead would describe only how well we measured these particular
    adapters, not whether another training run would land elsewhere. Only the
    interval construction changed.

    `permutation_p` is reported alongside as a distribution-free cross-check that
    does not lean on normality of the run effects.
    """
    rng = rng or np.random.default_rng(0)
    k = y.shape[2]
    if k < 2:
        raise ValueError("need >=2 seeds per cell to estimate within-cell variance")

    cell_means = y.mean(axis=2)                      # (2, 2)
    # Pooled within-cell variance: the only honest estimate of run-to-run spread.
    resid = y - cell_means[:, :, None]
    df = 4 * (k - 1)
    s2 = float((resid**2).sum() / df)

    from scipy import stats as _st
    tcrit = float(_st.t.ppf(0.975, df))
    flat = y.reshape(4, k)

    out = {}
    for name, w in _contrast_weights().items():
        # Contrast estimate scaled so it reads as a difference of marginal means.
        est = float((w * cell_means).sum() / 2.0)
        se = float(np.sqrt(s2 * (w**2).sum() / (4.0 * k)))
        lo, hi = est - tcrit * se, est + tcrit * se
        tstat = est / se if se > 0 else 0.0

        # Permutation: under the null the four cell labels are exchangeable
        # across runs, so shuffling run-to-cell assignment gives an exact test.
        wf = w.reshape(4)
        obs = abs(float((wf * flat.mean(axis=1)).sum()))
        if n_perm > 0:
            pool = flat.reshape(-1)
            hits = 0
            for _ in range(n_perm):
                shuffled = rng.permutation(pool).reshape(4, k).mean(axis=1)
                hits += abs(float((wf * shuffled).sum())) >= obs - 1e-12
            perm_p = (hits + 1) / (n_perm + 1)
        else:
            perm_p = float("nan")

        out[name] = dict(
            estimate=est,
            ci_low=float(lo),
            ci_high=float(hi),
            se=se,
            t=float(tstat),
            df=int(df),
            permutation_p=float(perm_p),
            excludes_zero=bool(lo > 0 or hi < 0),
        )
    return out


def simulate_cell_means(true_a: float, true_b: float, true_ab: float,
                        sigma_seed: float, eval_sd: float, k: int,
                        rng: np.random.Generator) -> np.ndarray:
    """One synthetic experiment. Returns (2, 2, k) measured accuracies in pp."""
    y = np.empty((2, 2, k))
    for ci in (0, 1):
        for vi in (0, 1):
            # Effects coded as +/- half, so `true_a` is the full main effect.
            mu = (ci - 0.5) * true_a + (vi - 0.5) * true_b \
                 + (ci - 0.5) * (vi - 0.5) * 4 * true_ab * 0.25
            true_run = mu + rng.normal(0, sigma_seed, size=k)
            y[ci, vi] = true_run + rng.normal(0, eval_sd, size=k)
    return y


# --- Synthetic validation -----------------------------------------------------

def validate_null_world(n_trials: int = 600, k: int = 3, seed: int = 11) -> dict:
    """The regime where the honest answer is 'nothing works'.

    False-positive rate for each effect must sit near alpha=0.05. An estimator
    that beats that here is not sensitive, it is broken.
    """
    rng = np.random.default_rng(seed)
    fires = {"content_A": 0, "verbosity_B": 0, "interaction_AB": 0}
    perm_fires = dict(fires)
    for _ in range(n_trials):
        y = simulate_cell_means(0, 0, 0, sigma_seed=1.5, eval_sd=1.38, k=k, rng=rng)
        res = bootstrap_effects(y, n_perm=300, rng=rng)
        for key in fires:
            fires[key] += res[key]["excludes_zero"]
            perm_fires[key] += res[key]["permutation_p"] < 0.05
    out = {key: v / n_trials for key, v in fires.items()}
    out.update({f"{key}__perm": v / n_trials for key, v in perm_fires.items()})
    return out


def validate_recovery(n_trials: int = 400, k: int = 12, true_a: float = 5.0,
                      seed: int = 12) -> dict:
    """Inject a known content effect; check the estimator recovers it unbiased
    and does not smear it into the other two contrasts."""
    rng = np.random.default_rng(seed)
    est_a, fire_a, fire_b, fire_ab, covered = [], 0, 0, 0, 0
    for _ in range(n_trials):
        y = simulate_cell_means(true_a, 0, 0, 1.5, 1.38, k, rng)
        res = bootstrap_effects(y, n_perm=0, rng=rng)
        est_a.append(res["content_A"]["estimate"])
        fire_a += res["content_A"]["excludes_zero"]
        fire_b += res["verbosity_B"]["excludes_zero"]
        fire_ab += res["interaction_AB"]["excludes_zero"]
        covered += res["content_A"]["ci_low"] <= true_a <= res["content_A"]["ci_high"]
    return dict(
        true_a=true_a,
        mean_estimate=float(np.mean(est_a)),
        bias=float(np.mean(est_a) - true_a),
        power_a=fire_a / n_trials,
        coverage_a=covered / n_trials,
        leak_into_B=fire_b / n_trials,
        leak_into_AB=fire_ab / n_trials,
    )


def measured_power(outcome: str, sigma_seed_pp: float, k: int, effect_pp: float,
                   n_trials: int = 400, seed: int = 13) -> float:
    """Empirical power of the shipping estimator -- not the analytic formula.

    Reported alongside the analytic MDE so the two can disagree visibly. If they
    do, the formula is the thing to distrust; the bootstrap is what will be run.
    """
    spec = OUTCOMES[outcome]
    e = eval_noise_pp(spec["n_eval"], spec["p0"])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_trials):
        y = simulate_cell_means(effect_pp, 0, 0, sigma_seed_pp, e, k, rng)
        res = bootstrap_effects(y, n_perm=0, rng=rng)
        hits += res["content_A"]["excludes_zero"]
    return hits / n_trials


def self_test() -> int:
    print("=" * 74)
    print("POWER / ESTIMATOR SELF-TEST  (no GPU, no network)")
    print("=" * 74)
    ok = True

    print("\n[1] Null world: true effects all zero, k=3.")
    print("    'nothing is going on' -- rejection rate must sit near alpha=0.05.")
    print("    A percentile bootstrap over 3 seeds scores ~0.25 here. That is the")
    print("    reason this file ships a t interval instead.")
    fp = validate_null_world()
    for key, rate in sorted(fp.items()):
        verdict = "ok" if 0.015 <= rate <= 0.09 else "FAIL"
        ok &= verdict == "ok"
        kind = "perm" if key.endswith("__perm") else "t-CI"
        print(f"      {key.replace('__perm',''):16s} [{kind}] rate {rate:.3f}   [{verdict}]")

    print("\n[2] Recovery: known content effect of 5.0pp, k=12.")
    rec = validate_recovery()
    bias_ok = abs(rec["bias"]) < 0.5
    cover_ok = 0.92 <= rec["coverage_a"] <= 0.98
    leak_ok = rec["leak_into_B"] <= 0.09 and rec["leak_into_AB"] <= 0.09
    ok &= bias_ok and leak_ok and cover_ok
    print(f"      estimate {rec['mean_estimate']:.3f}pp, bias {rec['bias']:+.3f}pp"
          f"   [{'ok' if bias_ok else 'FAIL'}]")
    print(f"      95% CI coverage of the true effect: {rec['coverage_a']:.3f}"
          f"   [{'ok' if cover_ok else 'FAIL'}]")
    print(f"      leak into B  {rec['leak_into_B']:.3f}, "
          f"into AB {rec['leak_into_AB']:.3f}   [{'ok' if leak_ok else 'FAIL'}]")

    print("\n[3] Analytic MDE vs measured power, GSM8K, sigma_seed=1.5pp.")
    for k in (3, 12):
        dp = design_point("gsm8k", 1.5, k)
        got = measured_power("gsm8k", 1.5, k, dp.mde_pp, n_trials=300)
        agree = 0.70 <= got <= 0.90
        ok &= agree
        print(f"      k={k:2d}  MDE {dp.mde_pp:5.2f}pp  -> measured power "
              f"{got:.3f} (target ~0.80)   [{'ok' if agree else 'FAIL'}]")

    print("\n[4] Bootstrap resamples seeds, not eval items.")
    print("    Widening the eval set must NOT collapse the CI when sigma_seed"
          " dominates.")
    rng = np.random.default_rng(99)
    y_noisy = simulate_cell_means(0, 0, 0, sigma_seed=3.0, eval_sd=1.4, k=6, rng=rng)
    y_clean = simulate_cell_means(0, 0, 0, sigma_seed=3.0, eval_sd=0.1, k=6, rng=rng)
    w_noisy = np.ptp([bootstrap_effects(y_noisy, n_perm=0)["content_A"][x]
                      for x in ("ci_low", "ci_high")])
    w_clean = np.ptp([bootstrap_effects(y_clean, n_perm=0)["content_A"][x]
                      for x in ("ci_low", "ci_high")])
    ratio = w_clean / w_noisy
    floor_ok = ratio > 0.55
    ok &= floor_ok
    print(f"      CI width with eval noise 1.4pp: {w_noisy:.2f}pp")
    print(f"      CI width with eval noise 0.1pp: {w_clean:.2f}pp "
          f"(ratio {ratio:.2f})   [{'ok' if floor_ok else 'FAIL'}]")
    print("      Seed variance is an irreducible floor. Bigger evals do not buy"
          " past it.")

    print("\n" + "=" * 74)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    print("=" * 74)
    return 0 if ok else 1


def report() -> None:
    print("=" * 78)
    print("DESIGN TABLE -- what this factorial can detect")
    print("=" * 78)
    print("\nsigma_seed is UNKNOWN until the pilot measures it. Swept here so the")
    print("design conclusion can be read off whichever value the pilot returns.\n")

    for outcome in ("gsm8k", "jailbreak"):
        spec = OUTCOMES[outcome]
        print(f"\n--- {spec['label']}  (n_eval={spec['n_eval']}, p0={spec['p0']}) ---")
        print(f"    binomial eval-noise floor: {eval_noise_pp(spec['n_eval'], spec['p0']):.2f}pp")
        print(f"    {'sigma_seed':>10} | {'MDE k=3':>9} | {'MDE k=6':>9} "
              f"| {'MDE k=12':>9} | {'seeds for prior effect':>24}")
        print("    " + "-" * 74)
        ref = REFERENCE_EFFECTS["prior_jailbreak"] if outcome == "jailbreak" \
            else REFERENCE_EFFECTS["prior_capability"]
        for s in (0.5, 1.0, 1.5, 2.5, 4.0):
            need = seeds_required(outcome, s, ref)
            need_s = f"{need}" if need else ">500"
            print(f"    {s:>10.1f} | {design_point(outcome, s, 3).mde_pp:>8.2f}pp "
                  f"| {design_point(outcome, s, 6).mde_pp:>8.2f}pp "
                  f"| {design_point(outcome, s, 12).mde_pp:>8.2f}pp "
                  f"| {need_s:>24}")
        print(f"    (last column targets the prior-work effect of {ref}pp)")

    print("\n" + "=" * 78)
    print("READ-OFF")
    print("=" * 78)
    print("""
The jailbreak outcome is well-powered at the handoff's k=3: a 16.7pp effect sits
far above the k=3 MDE across every plausible sigma_seed. The capability outcome
is not powered at k=3 for anything near the prior-reported 1.25pp -- it needs
tens of seeds per cell, i.e. hundreds of training runs.

That asymmetry is a design conclusion, not a complaint. It says: run the
factorial with safety as the primary outcome and capability as a bounded
secondary, and report the capability arm as an interval that rules effects in or
out rather than as a test that "found nothing". A capability CI of [-3, +3]pp is
an informative result. A capability p-value at k=3 is not.
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if args.json:
        out = {o: {k: asdict(design_point(o, 1.5, k)) for k in (3, 6, 12, 24)}
               for o in OUTCOMES}
        print(json.dumps(out, indent=2))
    else:
        report()
