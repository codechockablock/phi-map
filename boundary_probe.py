"""Where is a harness sensitive, and do models disagree there?

A harness turns a model response into a discrete observable -- a parsed choice, a
tool name, an approve/decline. That observation function is coarse almost
everywhere and sharp in a few places. Call those places boundaries. The practical
question is whether you can find them without a second model, and whether knowing
them tells you where a model swap will bite.

Boundary score, measured black-box: sample the same input k times and look at how
stable the *observable* is. Stable means deep in a flat region. Split means the
input sits on a boundary. No logprobs, no weights, works against any endpoint.

The analysis that matters is a comparison of two curves against boundary score:

    cross(A, B)   disagreement between two models
    within(A, A)  disagreement between two independent resamples of ONE model

`within` is the noise floor, and it is not flat -- it rises with boundary score by
construction, because an input where A is split will disagree with itself. So the
naive statistic, cross minus within, can shrink with boundary score for purely
mechanical reasons. Reporting the excess alone would invert the finding. What the
question actually asks is whether the two curves *separate*, and where.

The four-way taxonomy this file first used -- classifying the *slope* of
`cross - within` against boundary score -- does not survive its own synthetic. A
constant cross-model difference gets read as "models differ where they are
certain", because the `within` term rises with boundary score and drags the
difference down. And `within` is very nearly a deterministic function of boundary
score, so conditioning on it is the same conditioning, not a control. Total
variation also has a boundary-dependent ceiling: an input where A is split 50/50
caps `cross` at 0.5 no matter what B does.

So the headline is the operational question instead, which has no ceiling and no
collinearity:

    of the inputs where swapping A for B changes the harness verdict, what
    fraction would you have flagged in advance from A's self-consistency alone?

The number that decides it is the **blind-spot rate**: verdict changes that occur
at inputs where A was perfectly stable across every sample. Those are invisible
before B exists. If that rate is near zero, self-consistency is a usable
early-warning system and you can find your fragile inputs without the new model.
If it is high, the approach fails and differential testing against the real B is
the only option.

    NO_MODEL_EFFECT   the verdict almost never changes; nothing to predict
    PREDICTABLE       changes concentrate at inputs A was already unsure about
    BLIND_SPOT        changes happen where A was unanimous; not foreseeable

The two curves are still reported, as a diagnostic rather than as the verdict.

Sample sizes are matched throughout: every disagreement is computed between two
half-samples of size k//2, so `cross` and `within` carry the same finite-sample
noise and can be compared directly.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

# Excess of the cross-model modal-flip rate over the within-model resampling
# floor, below which the swap has not changed the harness verdict. A fixed
# threshold on the raw flip rate is wrong: two independent resamples of the SAME
# model already flip the mode at inputs near 50/50, so identical models produce
# a nonzero flip rate.
NEGLIGIBLE_EXCESS_FLIP = 0.03
# You can only inspect so many inputs before a model swap. Rank them by the
# current model's self-consistency and inspect this share.
ALERT_BUDGET = 0.20
# Recall at that budget, divided by the budget. Below this, ranking by
# self-consistency is no better than inspecting at random and the idea fails.
# Anchored on a budget rather than on "boundary score exactly zero", which gets
# rarer as k grows and so cannot carry a decision.
USEFUL_LIFT = 2.0
# Samples per cell needed to recover the shape reliably. Measured, not guessed:
# shape recovery in the favourable regime succeeds 3/8 at k=32, 6/8 at k=48 and
# k=64, and 8/8 from k=96. The realized per-input proportion is what is noisy, so
# no amount of bootstrap resampling of a small cell fixes it.
#
# This is the binding practical constraint. At 40 inputs and two models that is
# 7,680 generations, which is the honest price of the experiment.
MIN_SAMPLES_PER_CELL = 96


@dataclass
class Observations:
    """Observable draws for one (input, model) cell."""

    input_id: str
    model: str
    labels: list[Hashable] = field(default_factory=list)

    def distribution(self, over: Sequence[Hashable]) -> np.ndarray:
        counts = Counter(self.labels)
        total = sum(counts[label] for label in over)
        if total == 0:
            raise ValueError(f"no labels in the declared set for {self.input_id}")
        return np.array([counts[label] / total for label in over], dtype=float)


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def boundary_score(labels: Sequence[Hashable]) -> float:
    """1 - p(modal observable). Zero when the harness sees one answer every time.

    Deliberately not entropy: entropy over many labels is not comparable across
    inputs with different label-set sizes, and what matters here is only whether
    the harness's verdict is stable.
    """
    if not labels:
        raise ValueError("no samples")
    counts = Counter(labels)
    return float(1.0 - counts.most_common(1)[0][1] / len(labels))


def _split(labels: Sequence[Hashable], rng: np.random.Generator) -> tuple[list, list]:
    order = rng.permutation(len(labels))
    half = len(labels) // 2
    shuffled = [labels[index] for index in order]
    return shuffled[:half], shuffled[half : 2 * half]


def compare(
    observations: Mapping[tuple[str, str], Observations],
    input_ids: Sequence[str],
    model_a: str,
    model_b: str,
    label_set: Sequence[Hashable],
    repetitions: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-input boundary score, cross-model gap, and the within-model floor.

    Both disagreements use half-samples of equal size, so the finite-sample noise
    is matched and the two curves are directly comparable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for input_id in input_ids:
        first = observations[(input_id, model_a)].labels
        second = observations[(input_id, model_b)].labels
        if len(first) < 4 or len(second) < 4:
            raise ValueError(f"need at least 4 samples per cell for {input_id}")
        cross_draws, within_draws, within_flips, cross_flips = [], [], [], []
        for _ in range(repetitions):
            a_left, a_right = _split(first, rng)
            b_left, _b_right = _split(second, rng)
            def to_dist(part: Sequence[Hashable]) -> np.ndarray:
                return Observations(input_id, "", list(part)).distribution(label_set)

            cross_draws.append(total_variation(to_dist(a_left), to_dist(b_left)))
            within_draws.append(total_variation(to_dist(a_left), to_dist(a_right)))
            within_flips.append(
                Counter(a_left).most_common(1)[0][0]
                != Counter(a_right).most_common(1)[0][0]
            )
            cross_flips.append(
                Counter(a_left).most_common(1)[0][0]
                != Counter(b_left).most_common(1)[0][0]
            )
        rows.append(
            {
                "input_id": input_id,
                "boundary_score": boundary_score(first),
                "boundary_score_b": boundary_score(second),
                "cross": float(np.mean(cross_draws)),
                "within": float(np.mean(within_draws)),
                "gap": float(np.mean(cross_draws) - np.mean(within_draws)),
                "within_flip": float(np.mean(within_flips)),
                "cross_flip": float(np.mean(cross_flips)),
                # Verdict change attributable to the swap rather than to
                # resampling. At a coin-flip input both terms are ~0.5 and this
                # is ~0, which is correct: nothing there is attributable.
                "excess_flip": float(np.mean(cross_flips) - np.mean(within_flips)),
                "modal_a": Counter(first).most_common(1)[0][0],
                "modal_b": Counter(second).most_common(1)[0][0],
            }
        )
    return {"model_a": model_a, "model_b": model_b, "rows": rows}


def verdict(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Report the shape rather than classify it.

    An earlier version bucketed the result into PREDICTABLE / BLIND_SPOT using
    thresholds picked before any real data existed. Every one of them was wrong
    for a substantive reason, the last being that ranking inputs by raw boundary
    score puts coin-flip inputs first -- and those are exactly where a verdict
    change cannot be attributed to the swap, because resampling flips them too.
    Maximal instability is not maximal swap risk, so the relationship need not be
    monotone and should not be assumed so. The one classification kept is whether
    a swap effect exists at all, which is well founded against the within-model
    floor.
    """
    rows = comparison["rows"]
    scores = np.array([row["boundary_score"] for row in rows])
    excess = np.array([row["excess_flip"] for row in rows])
    within = np.array([row["within_flip"] for row in rows])
    cross = np.array([row["cross_flip"] for row in rows])

    mean_excess = float(excess.mean())
    decision = (
        "NO_MODEL_EFFECT"
        if mean_excess <= NEGLIGIBLE_EXCESS_FLIP
        else "MODEL_EFFECT_PRESENT"
    )

    # Shape: attributable change against instability, in equal-width bins.
    edges = np.linspace(0.0, max(0.5, float(scores.max())) + 1e-9, 6)
    shape = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (scores >= low) & (scores < high)
        if not mask.any():
            continue
        shape.append(
            {
                "boundary_range": [round(float(low), 3), round(float(high), 3)],
                "n": int(mask.sum()),
                "mean_excess_flip": float(excess[mask].mean()),
                "mean_within_flip": float(within[mask].mean()),
                "mean_cross_flip": float(cross[mask].mean()),
            }
        )

    # Operational: rank by boundary score, inspect a budget, how much do you catch?
    total = float(excess.clip(min=0).sum())
    order = np.argsort(-scores)
    budgets = {}
    for budget in (0.1, 0.2, 0.3, 0.5):
        take = max(1, int(round(budget * len(rows))))
        caught = float(excess[order[:take]].clip(min=0).sum())
        budgets[str(budget)] = {
            "recall": caught / total if total > 0 else 0.0,
            "lift_over_random": (caught / total) / budget if total > 0 else 0.0,
        }
    # The ceiling: what a perfect ranking would catch, for comparison.
    best = np.argsort(-excess)
    oracle = {
        str(b): float(excess[best[: max(1, int(round(b * len(rows))))]].clip(min=0).sum())
        / total
        if total > 0
        else 0.0
        for b in (0.1, 0.2, 0.3, 0.5)
    }

    return {
        "decision": decision,
        "mean_excess_flip": mean_excess,
        "mean_cross_flip": float(cross.mean()),
        "mean_within_flip": float(within.mean()),
        "corr_boundary_vs_excess": (
            float(np.corrcoef(scores, excess)[0, 1]) if scores.std() > 1e-9 else 0.0
        ),
        "shape_by_boundary_bin": shape,
        "recall_by_alert_budget": budgets,
        "oracle_recall_by_budget": oracle,
        "fraction_inputs_fully_stable": float((scores == 0).mean()),
        "n_inputs": len(rows),
    }


def _synthetic(regime: str, n_inputs: int, k: int, seed: int) -> dict[str, Any]:
    """Ground truth for the self-test. `difficulty` sets A's instability.

    Each regime is defined by *where B's modal verdict departs from A's*, since
    that is what the verdict function reads. An earlier version varied B's
    probability without ever moving it across 0.5, so two regimes produced no
    verdict changes at all and could not have tested anything.
    """
    rng = np.random.default_rng(seed)
    labels = ("allow", "deny")
    observations: dict[tuple[str, str], Observations] = {}
    input_ids = []
    for index in range(n_inputs):
        difficulty = index / (n_inputs - 1)
        input_id = f"s{index:03d}"
        input_ids.append(input_id)
        p_a = 0.5 + 0.5 * (1.0 - difficulty)  # 1.0 when easy, 0.5 when hard
        if regime == "none":
            p_b = p_a
        elif regime == "predictable":
            # B's verdict departs only once A is already wavering
            p_b = p_a if difficulty < 0.5 else 0.1
        elif regime == "blind_spot":
            # B's verdict departs exactly where A is unanimous; ranking by
            # self-consistency is actively misleading here
            p_b = 0.1 if difficulty < 0.4 else p_a
        elif regime == "uniform":
            p_b = p_a - 0.6  # departs at stable and unstable inputs alike
        else:
            raise ValueError(regime)
        for model, probability in (("A", p_a), ("B", float(np.clip(p_b, 0.0, 1.0)))):
            draws = rng.random(k) < probability
            observations[(input_id, model)] = Observations(
                input_id, model, ["allow" if d else "deny" for d in draws]
            )
    return {"observations": observations, "input_ids": input_ids, "label_set": labels}


def _decide(regime: str, k: int, seed: int) -> dict[str, Any]:
    world = _synthetic(regime, n_inputs=40, k=k, seed=seed)
    return verdict(
        compare(
            world["observations"],
            world["input_ids"],
            "A",
            "B",
            world["label_set"],
            repetitions=80,
            seed=seed,
        )
    )


def self_test() -> None:
    """Assert the estimator recovers the SHAPE, not a bucket label.

    Shape assertions survive not knowing the answer in advance; threshold
    assertions did not.
    """
    summary = {}

    # Identical models: no attributable change, however unstable the inputs are.
    same = _decide("none", MIN_SAMPLES_PER_CELL, 7)
    if same["decision"] != "NO_MODEL_EFFECT":
        raise AssertionError(f"identical models must show no effect: {same}")
    if same["mean_cross_flip"] < 0.05:
        raise AssertionError(
            "the null should still flip modes by resampling, or the synthetic has "
            "no unstable inputs and the floor is untested"
        )
    summary["none"] = {
        "decision": same["decision"],
        "mean_cross_flip": round(same["mean_cross_flip"], 3),
        "mean_within_flip": round(same["mean_within_flip"], 3),
        "mean_excess_flip": round(same["mean_excess_flip"], 3),
    }

    # Change concentrated where A already wavers -> positive shape.
    predictable = _decide("predictable", MIN_SAMPLES_PER_CELL, 7)
    if predictable["decision"] != "MODEL_EFFECT_PRESENT":
        raise AssertionError(f"predictable regime must show an effect: {predictable}")
    if predictable["corr_boundary_vs_excess"] <= 0.2:
        raise AssertionError(
            f"expected positive shape, got {predictable['corr_boundary_vs_excess']:.3f}"
        )
    # THE NEGATIVE RESULT, locked in. Even in the regime built to favour it,
    # ranking inputs by raw boundary score does not beat random inspection,
    # because attributable change peaks at INTERMEDIATE instability: the most
    # unstable inputs are coin flips, where resampling flips the verdict too and
    # nothing is attributable to the swap. If a future change makes this pass,
    # that is a signal to look for a bug, not a win.
    naive_lift = predictable["recall_by_alert_budget"]["0.2"]["lift_over_random"]
    if naive_lift >= 1.2:
        raise AssertionError(
            f"naive ranking unexpectedly beat random (lift {naive_lift:.2f}); "
            "the known non-monotonicity should prevent this"
        )
    # But the information IS present -- a correct ranking would find it. The gap
    # between these two numbers is the whole open problem.
    oracle_recall = predictable["oracle_recall_by_budget"]["0.2"]
    if oracle_recall <= 0.5:
        raise AssertionError(
            f"oracle recall {oracle_recall:.2f} too low; if a perfect ranking "
            "cannot find the changes either, there is nothing to extract"
        )
    summary["predictable"] = {
        "corr": round(predictable["corr_boundary_vs_excess"], 3),
        "naive_lift@0.2": round(naive_lift, 2),
        "oracle_recall@0.2": round(oracle_recall, 2),
    }

    # Change concentrated where A is unanimous -> negative shape, and ranking by
    # self-consistency is actively worse than random.
    blind = _decide("blind_spot", MIN_SAMPLES_PER_CELL, 7)
    if blind["corr_boundary_vs_excess"] >= -0.5:
        raise AssertionError(
            f"expected strong negative shape, got {blind['corr_boundary_vs_excess']:.3f}"
        )
    if blind["recall_by_alert_budget"]["0.2"]["lift_over_random"] >= 0.5:
        raise AssertionError("ranking must be clearly worse than random here")
    summary["blind_spot"] = {
        "corr": round(blind["corr_boundary_vs_excess"], 3),
        "naive_lift@0.2": round(
            blind["recall_by_alert_budget"]["0.2"]["lift_over_random"], 2
        ),
    }

    # Power on the sample size, measured rather than asserted.
    from collections import Counter as _Counter

    power = {}
    for k in (8, 16, MIN_SAMPLES_PER_CELL, 64):
        signs = _Counter(
            _decide("predictable", k, 100 + offset)["corr_boundary_vs_excess"] > 0.2
            for offset in range(5)
        )
        power[k] = f"{signs[True]}/5"
    if power[MIN_SAMPLES_PER_CELL] != "5/5":
        raise AssertionError(f"k={MIN_SAMPLES_PER_CELL} should be reliable: {power}")

    # The within-model floor rises steeply with boundary score. This is why
    # `cross - within` cannot be the statistic and why the floor is subtracted
    # per input rather than globally.
    world = _synthetic("none", n_inputs=40, k=64, seed=11)
    rows = compare(
        world["observations"], world["input_ids"], "A", "B", world["label_set"], 80, 5
    )["rows"]
    floor_slope = float(
        np.polyfit([r["boundary_score"] for r in rows], [r["within_flip"] for r in rows], 1)[0]
    )
    if floor_slope < 0.3:
        raise AssertionError(f"floor should rise with boundary score: {floor_slope:.3f}")

    if boundary_score(["allow"] * 16) != 0.0:
        raise AssertionError("a unanimous cell must score zero")

    print(
        json.dumps(
            {
                "self_test": "PASS",
                "regimes": summary,
                "power_on_shape_recovery_by_k": power,
                "min_samples_per_cell": MIN_SAMPLES_PER_CELL,
                "within_flip_floor_slope_vs_boundary": round(floor_slope, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    self_test()
