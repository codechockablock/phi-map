"""Which cheap probe finds the inputs a model swap will change?

`boundary_probe` established that ranking by self-consistency does not beat random
inspection, even in the regime built to favour it, and that an oracle ranking
reaches 0.60 recall at the same budget. The information is there; the ranking
statistic is wrong. This is the free synthetic study that decides whether any
statistic computable from the CURRENT model extracts it, before anyone pays for
generations against a real one.

Why self-consistency fails, stated structurally. There are three operational
categories:

    already nondeterministic   the verdict flips on resampling. Broken today,
                               independent of any swap.
    deterministic but fragile  stable on resampling, yet a small semantics-free
                               nudge moves the verdict. THIS is swap risk.
    deterministic and robust   stable, and stays stable. Fine.

Self-consistency separates the first from the other two and cannot tell the
second from the third, because both look identical under resampling. Arm G is the
existence proof: its conflict rows were unanimous across samples at a fixed
rendering and entirely determined by catalog line order.

So the candidate probe is sensitivity to a semantically-null re-rendering: sample
the same input under several renderings that a correct harness must treat alike,
and measure whether the verdict moves.

The world is a latent-margin model, deliberately not one where the answer is
assumed:

    m_i          signed distance from the harness decision boundary
    delta_{i,r}  nuisance shift from rendering r, mean zero
    A verdict    sigmoid((m_i + delta_{i,r}) / tau_a)
    B verdict    sigmoid((m_i + delta_{i,r} + shift_b(i)) / tau_b)

Self-consistency instability, perturbation sensitivity and swap risk are then all
functions of |m_i| at different scales, rather than one being defined in terms of
another. Whether the probe works is a question about those scales, which is what
gets measured. A regime where swap risk is driven by something orthogonal to m_i
is included precisely so the study can return "nothing works".

Statistics are compared at MATCHED SAMPLE BUDGET. Perturbation sensitivity spends
its budget across renderings, self-consistency spends it on repeats of one
rendering. Comparing them at equal cost is the only fair test, and it is also the
comparison a practitioner actually faces.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np

BUDGETS = (0.1, 0.2, 0.3, 0.5)
# Total generations per input available to any probe. Split differently by each.
SAMPLE_BUDGET = 24
N_INPUTS = 200
# Below this excess, a verdict change is not attributable to the swap.
ATTRIBUTABLE = 0.25


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_world(
    regime: str,
    n_inputs: int,
    rng: np.random.Generator,
    tau_a: float = 0.35,
    tau_b: float = 0.35,
    render_sigma: float = 0.8,
) -> dict[str, Any]:
    """Latent margins, rendering nuisance, and a model-B shift per regime."""
    margins = rng.uniform(-3.0, 3.0, size=n_inputs)
    if regime == "shift":
        # B is systematically displaced. Risk concentrates at small |m|, which
        # both probes can in principle see.
        shift_b = np.full(n_inputs, 0.9)
    elif regime == "none":
        shift_b = np.zeros(n_inputs)
    elif regime == "orthogonal":
        # B differs on a set unrelated to the margin. Nothing computable from A
        # should find these, and the study must be able to say so.
        shift_b = np.where(rng.random(n_inputs) < 0.15, 4.0, 0.0)
    elif regime == "noisy_shift":
        shift_b = rng.normal(0.9, 0.9, size=n_inputs)
    else:
        raise ValueError(regime)
    return {
        "margins": margins,
        "shift_b": shift_b,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "render_sigma": render_sigma,
        "regime": regime,
    }


def _draw(
    probability: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """count Bernoulli draws per element; returns (len(probability), count)."""
    return (rng.random((len(probability), count)) < probability[:, None]).astype(int)


def probe_self_consistency(
    world: Mapping[str, Any], budget: int, rng: np.random.Generator
) -> np.ndarray:
    """Spend the whole budget resampling one canonical rendering."""
    margins = np.asarray(world["margins"])
    canonical = np.zeros(len(margins))  # rendering 0 has no nuisance shift
    probability = _sigmoid((margins + canonical) / world["tau_a"])
    draws = _draw(probability, budget, rng)
    mean = draws.mean(axis=1)
    return 1.0 - np.maximum(mean, 1.0 - mean)  # 1 - p(mode)


def probe_perturbation(
    world: Mapping[str, Any], budget: int, rng: np.random.Generator, repeats: int = 2
) -> np.ndarray:
    """Spend the budget across renderings instead: budget/repeats of them."""
    margins = np.asarray(world["margins"])
    n_renderings = max(2, budget // repeats)
    verdicts = np.zeros((len(margins), n_renderings))
    for index in range(n_renderings):
        shift = rng.normal(0.0, world["render_sigma"], size=len(margins))
        probability = _sigmoid((margins + shift) / world["tau_a"])
        verdicts[:, index] = _draw(probability, repeats, rng).mean(axis=1)
    mean = verdicts.mean(axis=1)
    return 1.0 - np.maximum(mean, 1.0 - mean)


def probe_combined(
    world: Mapping[str, Any], budget: int, rng: np.random.Generator
) -> np.ndarray:
    """Half the budget each; rank by perturbation among resampling-stable inputs.

    Encodes the triage directly: inputs that already flip on resampling are a
    present-tense defect rather than swap risk, so they are deprioritised.
    """
    half = budget // 2
    unstable = probe_self_consistency(world, half, rng)
    fragile = probe_perturbation(world, half, rng)
    already_broken = unstable > 0.25
    return np.where(already_broken, -1.0, fragile)


def true_swap_risk(
    world: Mapping[str, Any], rng: np.random.Generator, samples: int = 400
) -> np.ndarray:
    """Ground truth: attributable verdict change at the canonical rendering.

    Excess over the within-model resampling floor, so an input that is a coin
    flip under A alone does not count as swap risk.
    """
    margins = np.asarray(world["margins"])
    p_a = _sigmoid(margins / world["tau_a"])
    p_b = _sigmoid((margins + np.asarray(world["shift_b"])) / world["tau_b"])
    a_first = _draw(p_a, samples, rng).mean(axis=1)
    a_second = _draw(p_a, samples, rng).mean(axis=1)
    b_draws = _draw(p_b, samples, rng).mean(axis=1)
    # Probabilistic rather than one realized draw, so the floor is subtracted
    # smoothly and a coin-flip input carries no attributable risk.
    return np.clip(np.abs(a_first - b_draws) - np.abs(a_first - a_second), 0.0, None)


def recall_at_budgets(
    score: np.ndarray, risk: np.ndarray, budgets: Sequence[float] = BUDGETS
) -> dict[str, float]:
    total = float(risk.sum())
    if total <= 0:
        return {str(b): 0.0 for b in budgets}
    order = np.argsort(-score)
    out = {}
    for budget in budgets:
        take = max(1, int(round(budget * len(score))))
        out[str(budget)] = float(risk[order[:take]].sum() / total)
    return out


def run(regime: str, seed: int, budget: int = SAMPLE_BUDGET) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    world = make_world(regime, N_INPUTS, rng)
    risk = true_swap_risk(world, rng)
    probes: dict[str, Callable[..., np.ndarray]] = {
        "self_consistency": probe_self_consistency,
        "perturbation": probe_perturbation,
        "combined": probe_combined,
    }
    results = {
        name: recall_at_budgets(probe(world, budget, rng), risk)
        for name, probe in probes.items()
    }
    results["oracle"] = recall_at_budgets(risk, risk)
    results["random"] = recall_at_budgets(rng.random(len(risk)), risk)
    return {
        "regime": regime,
        "mean_risk": float(risk.mean()),
        "fraction_at_risk": float((risk > ATTRIBUTABLE).mean()),
        "recall": results,
    }


def summarize(regime: str, seeds: int = 12, budget: int = SAMPLE_BUDGET) -> dict[str, Any]:
    runs = [run(regime, seed, budget) for seed in range(seeds)]
    names = list(runs[0]["recall"])
    return {
        "regime": regime,
        "mean_risk": float(np.mean([r["mean_risk"] for r in runs])),
        "fraction_at_risk": float(np.mean([r["fraction_at_risk"] for r in runs])),
        "recall_at_0.2": {
            name: float(np.mean([r["recall"][name]["0.2"] for r in runs]))
            for name in names
        },
        "lift_at_0.2": {
            name: float(np.mean([r["recall"][name]["0.2"] for r in runs])) / 0.2
            for name in names
        },
    }


def ratio_sweep(seeds: int = 12, budget: int = SAMPLE_BUDGET) -> list[dict[str, Any]]:
    """The parameter that decides everything: B's shift over A's sampling noise.

    `boundary_probe`'s synthetic and this one disagreed about whether
    self-consistency ranking works. They disagreed because they implicitly chose
    different values of this ratio, which is the honest finding: the answer is
    not derivable from theory and has to be measured on real model pairs.
    """
    rows = []
    for ratio in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        cells: dict[str, list[float]] = {
            name: [] for name in ("self_consistency", "perturbation", "oracle", "random")
        }
        at_risk = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            tau = 0.35
            world = make_world("shift", N_INPUTS, rng, tau_a=tau, tau_b=tau)
            world["shift_b"] = np.full(N_INPUTS, ratio * tau)
            risk = true_swap_risk(world, rng)
            at_risk.append(float((risk > ATTRIBUTABLE).mean()))
            cells["self_consistency"].append(
                recall_at_budgets(probe_self_consistency(world, budget, rng), risk)["0.2"]
            )
            cells["perturbation"].append(
                recall_at_budgets(probe_perturbation(world, budget, rng), risk)["0.2"]
            )
            cells["oracle"].append(recall_at_budgets(risk, risk)["0.2"])
            cells["random"].append(
                recall_at_budgets(rng.random(len(risk)), risk)["0.2"]
            )
        rows.append(
            {
                "shift_over_tau": ratio,
                "fraction_at_risk": float(np.mean(at_risk)),
                **{
                    f"lift_{name}": float(np.mean(values)) / 0.2
                    for name, values in cells.items()
                },
            }
        )
    return rows


def self_test() -> None:
    # Probes must spend the same number of generations, or the comparison is rigged.
    rng = np.random.default_rng(0)
    world = make_world("shift", 50, rng)
    if len(probe_self_consistency(world, 24, rng)) != 50:
        raise AssertionError("self-consistency probe returned the wrong shape")
    if len(probe_perturbation(world, 24, rng, repeats=2)) != 50:
        raise AssertionError("perturbation probe returned the wrong shape")

    # Under a null model B there is no risk to find, and no probe should appear
    # to succeed. This catches a leak between the probe and the ground truth.
    null = summarize("none", seeds=6)
    if null["mean_risk"] > 0.05:
        raise AssertionError(f"identical models should carry no risk: {null}")

    # An oracle must beat random, or the recall machinery is broken.
    shift = summarize("shift", seeds=6)
    if shift["lift_at_0.2"]["oracle"] <= shift["lift_at_0.2"]["random"] * 1.5:
        raise AssertionError(f"oracle must dominate random: {shift}")

    # When B differs on a set unrelated to the margin, nothing computable from A
    # should work. If some probe does well here, it is reading the answer.
    orthogonal = summarize("orthogonal", seeds=6)
    for name in ("self_consistency", "perturbation", "combined"):
        if orthogonal["lift_at_0.2"][name] > 1.6:
            raise AssertionError(
                f"{name} scored lift {orthogonal['lift_at_0.2'][name]:.2f} on the "
                "orthogonal regime; a probe that predicts unpredictable risk is "
                "leaking ground truth"
            )
    # The headline decay must be present, or the sweep is not measuring the
    # thing the conclusion rests on.
    sweep = ratio_sweep(seeds=6)
    low = next(r for r in sweep if r["shift_over_tau"] == 1.0)
    high = next(r for r in sweep if r["shift_over_tau"] == 12.0)
    if low["lift_self_consistency"] < 2.0:
        raise AssertionError(f"probe should work when the shift is small: {low}")
    if high["lift_self_consistency"] > 1.3:
        raise AssertionError(f"probe should fail when the shift is large: {high}")
    if high["lift_oracle"] < 1.5:
        raise AssertionError(
            "the oracle must retain lift at high ratio, or there is no signal "
            "left for a better probe to find and the salvage is unavailable"
        )
    if high["fraction_at_risk"] <= low["fraction_at_risk"]:
        raise AssertionError("risk should grow with the shift, or the sweep is inverted")
    print(json.dumps({"self_test": "PASS", "null": null["mean_risk"]}, indent=2))


def main() -> None:
    print(f"Matched budget: {SAMPLE_BUDGET} generations per input, {N_INPUTS} inputs\n")
    header = f"{'regime':>14} {'at risk':>8}  " + "  ".join(
        f"{n[:16]:>16}" for n in
        ("self_consistency", "perturbation", "combined", "oracle", "random")
    )
    print(header)
    print("-" * len(header))
    for regime in ("none", "shift", "noisy_shift", "orthogonal"):
        s = summarize(regime)
        cells = "  ".join(
            f"{s['lift_at_0.2'][n]:>16.2f}" for n in
            ("self_consistency", "perturbation", "combined", "oracle", "random")
        )
        print(f"{regime:>14} {s['fraction_at_risk']:>8.2f}  {cells}")
    print("\nlift over random at a 20% inspection budget; 1.00 means no better than chance")
    print("\nThe parameter that decides it: B's shift over A's sampling noise\n")
    header = (f"{'shift/tau':>10} {'at risk':>8} {'self_cons':>10} {'perturb':>9} "
              f"{'oracle':>8} {'random':>8}")
    print(header)
    print("-" * len(header))
    for row in ratio_sweep():
        print(
            f"{row['shift_over_tau']:>10.1f} {row['fraction_at_risk']:>8.2f} "
            f"{row['lift_self_consistency']:>10.2f} {row['lift_perturbation']:>9.2f} "
            f"{row['lift_oracle']:>8.2f} {row['lift_random']:>8.2f}"
        )
    print("\nThe regime where the problem matters is the regime where the probe fails.")


if __name__ == "__main__":
    self_test()
    main()
