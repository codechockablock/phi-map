"""Arm G: re-extract the conflict direction on order-crossed source seeds.

Section 14 of `RESEARCH_ARC.md` established two things at once. The decision is
entirely catalog position — swapping the two catalog lines reverses 64 of 64
conflict decisions — and the scope signal is nonetheless real, with a condition
main effect of +4.814 and a within-order label AUROC of 1.00000 in both orders.

So there is a variable worth having a direction for, but the direction we have is
not it. It was fit where catalog order was locked to pair-index parity, the order
main effect is -3.080, and the resulting direction separates order within the
conflict condition at AUROC 1.0000 against 0.9431 for the label.

Write the state as

    h(C, O) = mu + a*C + b*O + d*C*O + noise,     C, O in {-1, +1}

The within-scenario paired difference already cancels the order main effect,
because both members of a pair share a rendering:

    h(+1, o) - h(-1, o) = 2a + 2d*o

An earlier version of this docstring claimed that averaging over both orders is
therefore the fix. It is not, and the self-test below is what caught it. The
parity-locked assignment is *balanced* across pairs — half render `inside_first`
and half `outside_first` — so mean(o) = 0 and the interaction cancels in the mean
under the legacy recipe too. The legacy estimate of `a` was never biased.

The contamination is at the projection level, not the estimate level. Projecting
onto `a` picks up the interaction whenever `a` and `d` are not orthogonal *in the
model's geometry*, however cleanly `a` was estimated. With cos(a, d) = 0.8 and
the effect sizes seen here, a perfectly estimated `a` still separates catalog
order within the conflict condition. That is what AUROC 1.0000 on order was.

So the fix is explicit orthogonalization, and crossed source data is what makes
it possible: without both renderings of the same scenario, `d` is not estimable
at all, so there is nothing to orthogonalize against. This protocol estimates all
three components and builds five directions from the SAME crossed source states,
so the recipe is isolated from the data:

    order_averaged    mean paired difference over both orders          (a)
    legacy_recipe     one order per pair, as the old generator forced  (a, noisier)
    order_main        mean of h(., outside) - h(., inside)             (b)
    order_difference  paired diff at outside minus at inside           (d)
    orthogonalized    `a` with `b` and `d` projected out               (the fix)

Each is scored on held-out order-crossed data by how well its projection
separates the label versus how well it separates catalog order within a
condition. A clean direction reads the label and is at chance on order.

Baseline capture only. No intervention: this decides whether there is a direction
worth running an operator x dose factorial on.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from arm_g_causal import ACTION_TOKENS, orthogonalize
from arm_g_causal_subspace import EXPECTED_MODEL_COMMIT
from arm_g_cross_layer import run_forward
from arm_g_order_crossover import auroc, crossover_eval_rows
from arm_g_phase1 import (
    DEV_FAMILY,
    PRIMARY_POSITION,
    TARGET_MODEL,
    atomic_json,
    load_acting_model,
    package_version,
    require_gpu,
    unit,
    verify_or_write,
)
from arm_g_scenarios import (
    CATALOG_ORDERS,
    build_manifest,
    legacy_catalog_order,
    validate_manifest,
)

SOURCE_SEEDS = (101, 102)
EVAL_SEED = 112
SWEEP_LAYERS = (12, 14, 16, 17, 18, 20, 24, 27, 30)
RECIPES = (
    "order_averaged",
    "legacy_recipe",
    "order_main",
    "order_difference",
    "orthogonalized",
)

# A direction is clean if it reads the label and is at chance on catalog order
# within a condition. The legacy direction scored 1.0000 on order within the
# conflict condition against 0.9431 on the label, which is the failure this
# protocol has to detect and repair.
#
# "At chance" is decided by a bootstrap over SOURCE scenarios, not by a fixed
# tolerance and not by an analytic band. Both of those see only evaluation-set
# sampling noise, and the direction is itself estimated: re-drawing the source
# set moves it, which moves the order AUROC. Empirically that inflates the null
# spread by about half again over the iid figure, so an analytic band rejects a
# genuinely clean direction roughly a fifth of the time.
LABEL_AUROC_GATE = 0.90
# The null spread of the order AUROC, measured with direction estimation
# included, is about 0.076 -- half again the iid figure of 0.052, because
# re-drawing the source set moves the direction. The contaminated case sits at
# the 0.5 ceiling. So any bar between roughly 0.15 and 0.30 returns the same
# verdict; 0.20 is about 2.6 null SDs and is the one used. The bootstrap
# interval below is reported as a diagnostic rather than as the gate, because a
# percentile interval on an AUROC pinned near 0.5 under-covers.
ORDER_CONTAMINATION_BAR = 0.20
BOOTSTRAP_REPETITIONS = 400

PRIOR_EVIDENCE = {
    "source": "results/arm_g_order_crossover_seed111_v1, RESEARCH_ARC.md section 14",
    "legacy_direction_label_auroc": 0.9431,
    "legacy_direction_order_auroc_within_conflict": 1.0000,
    "crossover_condition_main_effect": 4.814,
    "crossover_order_main_effect": -3.080,
    "crossover_condition_x_order": -3.793,
    "prediction": (
        "order_averaged and legacy_recipe should be near-identical, since the "
        "parity assignment is balanced and neither estimate is biased, and both "
        "should be contaminated on order at the projection level in proportion "
        "to cos(a, d). Only `orthogonalized` should read the label while sitting "
        "near 0.5 on order within a condition"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/arm_g_reextract")
    )
    parser.add_argument("--pairs-per-family", type=int, default=16)
    # The order main effect and the interaction are the orthogonalization basis,
    # and a noisy basis leaves residual contamination, so the source set is built
    # larger than the evaluation set.
    parser.add_argument("--source-pairs-per-family", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def crossed_source_rows(
    pairs_per_family: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Development-family source rows, rendered in both catalog orders."""
    manifest = build_manifest(
        pairs_per_family=pairs_per_family,
        repeats=1,
        seed=seed,
        control_label_mode="parity_independent",
        catalog_order_mode="crossed",
    )
    audit = validate_manifest(
        manifest,
        pairs_per_family * 2,
        1,
        require_parity_independent=True,
        require_order_crossed=True,
    )
    rows = [
        {
            "source_seed": seed,
            "crossover_id": str(scenario["crossover_id"]),
            "pair_index": int(scenario["pair_index"]),
            "condition_label": int(scenario["condition_label"]),
            "catalog_order": str(scenario["catalog_order"]),
            "messages": scenario["messages"],
        }
        for scenario in manifest
        if scenario["family"] == DEV_FAMILY
    ]
    expected = pairs_per_family * 2 * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} source rows, found {len(rows)}")
    return rows, audit


def paired_differences_by_order(
    states: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    """Conflict-minus-reachable within each (scenario, order) cell.

    Returns one stacked array per catalog order aligned on the same scenario
    list, so the recipes differ only in how the two are combined, plus the
    order shift `h(., outside) - h(., inside)` per (scenario, condition), which
    estimates the order main effect. That one is only estimable because both
    renderings of the same scenario exist.
    """
    grouped: dict[tuple[int, str, str], dict[int, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for state, row in zip(states, rows, strict=True):
        key = (
            int(row["source_seed"]),
            str(row["crossover_id"]),
            str(row["catalog_order"]),
        )
        grouped[key][int(row["condition_label"])].append(state)

    scenarios = sorted({(key[0], key[1]) for key in grouped})
    by_order: dict[str, list[np.ndarray]] = {order: [] for order in CATALOG_ORDERS}
    order_shift: list[np.ndarray] = []
    metadata = []
    for seed, crossover_id in scenarios:
        cells = {}
        for order in CATALOG_ORDERS:
            conditions = grouped.get((seed, crossover_id, order))
            if conditions is None:
                raise RuntimeError(f"missing rendering: {seed} {crossover_id} {order}")
            if len(conditions[0]) != 1 or len(conditions[1]) != 1:
                raise RuntimeError(f"incomplete pair: {seed} {crossover_id} {order}")
            by_order[order].append(conditions[1][0] - conditions[0][0])
            cells[order] = conditions
        for label in (0, 1):
            order_shift.append(
                cells["outside_first"][label][0] - cells["inside_first"][label][0]
            )
        metadata.append(
            {
                "source_seed": seed,
                "crossover_id": crossover_id,
                "pair_index": int(crossover_id.split(":")[1]),
            }
        )
    stacked = {
        order: np.stack(values).astype(np.float64) for order, values in by_order.items()
    }
    return stacked, np.stack(order_shift).astype(np.float64), metadata


def build_directions(
    by_order: Mapping[str, np.ndarray],
    order_shift: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Five unit directions from the same paired differences.

    `order_averaged` and `legacy_recipe` are both unbiased estimates of the
    condition main effect — the parity assignment is balanced, so the
    interaction cancels in either mean. They are kept side by side to show that,
    because it is the point: the defect was never the estimator.
    """
    inside = by_order["inside_first"]
    outside = by_order["outside_first"]

    # What a parity-locked generator forced: one rendering per scenario.
    legacy = np.stack(
        [
            inside[index]
            if legacy_catalog_order(int(item["pair_index"])) == "inside_first"
            else outside[index]
            for index, item in enumerate(metadata)
        ]
    )

    condition_main = unit((0.5 * (inside + outside)).mean(axis=0))
    order_main = unit(order_shift.mean(axis=0))
    interaction = unit((0.5 * (outside - inside)).mean(axis=0))
    return {
        "order_averaged": condition_main,
        "legacy_recipe": unit(legacy.mean(axis=0)),
        "order_main": order_main,
        "order_difference": interaction,
        # The fix. Crossed source data is what makes `interaction` estimable,
        # and therefore what makes this projection possible at all.
        "orthogonalized": orthogonalize(condition_main, [order_main, interaction]),
    }


def score_direction(
    projections: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Does it read the label, and does it read catalog order within a condition?"""
    labels = np.asarray([row["condition_label"] for row in rows])
    order = np.asarray(
        [CATALOG_ORDERS.index(str(row["catalog_order"])) for row in rows]
    )
    line = np.asarray([int(row["requested_target_line"]) for row in rows])
    within = {}
    for name, mask in (("conflict", labels == 1), ("reachable", labels == 0)):
        within[name] = {
            "order_auroc": auroc(projections[mask], 1 - order[mask]),
            "requested_line_auroc": auroc(projections[mask], line[mask] - 1),
        }
    return {
        "label_auroc": auroc(projections, labels),
        "within_condition": within,
        "order_contamination": max(
            abs(within[name]["order_auroc"] - 0.5) for name in within
        ),
    }


def bootstrap_order_ci(
    source_states: np.ndarray,
    source_rows: Sequence[Mapping[str, Any]],
    eval_states: np.ndarray,
    eval_rows: Sequence[Mapping[str, Any]],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Resample source scenarios, rebuild every direction, re-score.

    The direction is an estimate, so its sampling variance belongs in the
    interval. Resampling scenarios rather than rows keeps the crossover intact:
    a scenario is drawn with both of its renderings and both conditions.
    """
    by_scenario: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(source_rows):
        by_scenario[(int(row["source_seed"]), str(row["crossover_id"]))].append(index)
    keys = sorted(by_scenario)
    # The evaluation set is resampled too. Resampling only the source gives an
    # interval around whatever the fixed evaluation draw happened to produce,
    # which is exactly the displacement the interval is supposed to cover.
    eval_by_scenario: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(eval_rows):
        eval_by_scenario[str(row["crossover_id"])].append(index)
    eval_keys = sorted(eval_by_scenario)
    generator = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[float]]] = {
        recipe: {"conflict": [], "reachable": []} for recipe in RECIPES
    }
    for _repetition in range(repetitions):
        picked = generator.choice(len(keys), size=len(keys), replace=True)
        indices: list[int] = []
        resampled_rows: list[Mapping[str, Any]] = []
        for draw, choice in enumerate(picked):
            for index in by_scenario[keys[choice]]:
                indices.append(index)
                # Re-key so repeated draws stay distinct scenarios.
                resampled_rows.append({**source_rows[index], "source_seed": draw})
        by_order, order_shift, metadata = paired_differences_by_order(
            source_states[indices], resampled_rows
        )
        directions = build_directions(by_order, order_shift, metadata)
        eval_picked = generator.choice(len(eval_keys), size=len(eval_keys), replace=True)
        eval_indices = [
            index for choice in eval_picked for index in eval_by_scenario[eval_keys[choice]]
        ]
        drawn_states = eval_states[eval_indices]
        drawn_rows = [eval_rows[index] for index in eval_indices]
        for recipe, direction in directions.items():
            entry = score_direction(drawn_states @ direction, drawn_rows)
            for condition in ("conflict", "reachable"):
                draws[recipe][condition].append(
                    entry["within_condition"][condition]["order_auroc"]
                )
    return {
        recipe: {
            condition: {
                "ci_95": [
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                ],
                "covers_chance": bool(
                    np.percentile(values, 2.5) <= 0.5 <= np.percentile(values, 97.5)
                ),
            }
            for condition, values in conditions.items()
        }
        for recipe, conditions in draws.items()
    }


def mark_clean(
    scored: dict[str, dict[str, Any]],
    intervals: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """A direction is clean if it reads the label and its order CIs cover chance."""
    for recipe, entry in scored.items():
        entry["order_ci"] = intervals[recipe]
        entry["clean"] = bool(
            entry["label_auroc"] >= LABEL_AUROC_GATE
            and entry["order_contamination"] <= ORDER_CONTAMINATION_BAR
        )
    return scored


def select_layer(profile: Mapping[int, Mapping[str, Any]]) -> int | None:
    """Best label AUROC among layers whose direction is clean on order."""
    clean = [
        layer
        for layer, entry in profile.items()
        if entry["orthogonalized"]["clean"]
    ]
    if not clean:
        return None
    return max(clean, key=lambda layer: profile[layer]["orthogonalized"]["label_auroc"])


def reextract_decision(
    profile: Mapping[int, Mapping[str, Any]],
    selected: int | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if selected is None:
        reasons.append(
            "no swept layer yields an orthogonalized direction that reads the "
            f"label at >= {LABEL_AUROC_GATE} while its order AUROC within each "
            f"condition stays within {ORDER_CONTAMINATION_BAR} of chance"
        )
        return "NO_CLEAN_DIRECTION", reasons
    entry = profile[selected]
    raw = entry["order_averaged"]["within_condition"]["conflict"]["order_auroc"]
    legacy = entry["legacy_recipe"]["within_condition"]["conflict"]["order_auroc"]
    reasons.append(
        f"before orthogonalization the condition direction reads catalog order "
        f"within conflict at AUROC {raw:.4f}, and the legacy aggregation at "
        f"{legacy:.4f} -- both unbiased estimates of the same axis, which is why "
        "the defect was never the estimator"
    )
    if not entry["order_averaged"]["clean"]:
        reasons.append(
            "orthogonalizing against the order main effect and the interaction "
            "is what makes the projections clean, and crossed source data is "
            "what makes those two estimable"
        )
    else:
        reasons.append(
            "the un-orthogonalized direction is already clean on these states, "
            "so the condition and interaction axes are near orthogonal here and "
            "the committed contamination is not reproduced -- revisit"
        )
    return "CLEAN_DIRECTION", reasons


def self_test() -> None:
    rows = []
    for seed in SOURCE_SEEDS:
        # Enough scenarios that the orthogonalization basis is precisely
        # estimated: this test is about the method, not about small samples.
        for pair_index in range(32):
            for order in CATALOG_ORDERS:
                for label in (0, 1):
                    rows.append(
                        {
                            "source_seed": seed,
                            "crossover_id": f"{DEV_FAMILY}:{pair_index:03d}",
                            "pair_index": pair_index,
                            "condition_label": label,
                            "catalog_order": order,
                        }
                    )

    dimension = 24
    condition_axis = np.zeros(dimension)
    condition_axis[0] = 1.0
    order_axis = np.zeros(dimension)
    order_axis[2] = 1.0

    def synthesize(cos_ad: float, seed: int) -> tuple[Any, ...]:
        """States where the interaction axis sits at cos_ad from the condition axis."""
        rng = np.random.default_rng(seed)
        residual = np.zeros(dimension)
        residual[1] = 1.0
        interaction_axis = unit(
            cos_ad * condition_axis + np.sqrt(1 - cos_ad**2) * residual
        )

        def state(label: int, order: str) -> np.ndarray:
            c = 2 * label - 1
            o = 2 * CATALOG_ORDERS.index(order) - 1
            return (
                2.0 * c * condition_axis
                + 3.0 * o * order_axis
                + 4.0 * c * o * interaction_axis
                # Within-cell variance comparable to the effects, as in real
                # states. With negligible noise an infinitesimal estimation leak
                # dominates the projection and the contamination test measures
                # estimator precision rather than geometry.
                + rng.normal(scale=1.0, size=dimension)
            )

        source = np.stack(
            [state(row["condition_label"], row["catalog_order"]) for row in rows]
        )
        eval_rows, eval_states = [], []
        # 64 scenarios x 2 orders x 2 conditions = the real run's cell sizes, so
        # a pure-noise order AUROC concentrates the way it will on real states.
        for pair_index in range(64):
            for order in CATALOG_ORDERS:
                for label in (0, 1):
                    line = 1 if (label == 0) == (order == "inside_first") else 2
                    eval_rows.append(
                        {
                            "crossover_id": f"eval:{pair_index:03d}",
                            "condition_label": label,
                            "catalog_order": order,
                            "requested_target_line": line,
                        }
                    )
                    eval_states.append(state(label, order))
        return interaction_axis, source, eval_rows, np.stack(eval_states)

    # --- Unbiasedness is a claim about the mean, so test it as one. ---
    # A single noisy realization cannot distinguish bias from estimator variance,
    # and asserting a tight tolerance on one draw tests precision, not bias.
    # The leak has to be measured along the part of the interaction axis that is
    # NOT already in the condition axis. Measured along `d` itself, a perfect
    # estimate of `a` scores cos(a, d) = 0.8 and looks maximally "biased".
    leaks = {"order_averaged": [], "legacy_recipe": []}
    for replication in range(200):
        interaction_axis, source, _rows, _states = synthesize(0.8, 900 + replication)
        residual_axis = unit(
            interaction_axis
            - float(interaction_axis @ condition_axis) * condition_axis
        )
        by_order, order_shift, metadata = paired_differences_by_order(source, rows)
        directions = build_directions(by_order, order_shift, metadata)
        for name in leaks:
            leaks[name].append(float(directions[name] @ residual_axis))
    for name, values in leaks.items():
        mean_leak = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if abs(mean_leak) > max(0.02, 3 * standard_error):
            raise AssertionError(
                f"{name} is biased toward the interaction: mean leak {mean_leak:.4f} "
                f"(se {standard_error:.4f}) -- the parity assignment is balanced, "
                "so both estimators should be unbiased"
            )

    # --- Case 1: orthogonal axes. No systematic leak to pick up. ---
    interaction_axis, source, eval_rows, eval_states = synthesize(0.0, 112)
    by_order, order_shift, metadata = paired_differences_by_order(source, rows)
    if len(metadata) != 64:
        raise AssertionError(f"expected 64 source scenarios, got {len(metadata)}")
    directions = build_directions(by_order, order_shift, metadata)
    if float(directions["order_averaged"] @ condition_axis) < 0.90:
        raise AssertionError("order_averaged should recover the condition axis")
    if abs(float(directions["order_difference"] @ interaction_axis)) < 0.90:
        raise AssertionError("order_difference should recover the interaction axis")
    if abs(float(directions["order_main"] @ order_axis)) < 0.90:
        raise AssertionError("order_main should recover the order axis")
    orthogonal_case = mark_clean(
        {
            name: score_direction(eval_states @ direction, eval_rows)
            for name, direction in directions.items()
        },
        bootstrap_order_ci(source, rows, eval_states, eval_rows, 60, 1),
    )

    # --- Case 2: the axes are NOT orthogonal, which is the real situation. ---
    # The estimate is still unbiased, and the projection is still contaminated.
    interaction_axis, source, eval_rows, eval_states = synthesize(0.8, 113)
    by_order, order_shift, metadata = paired_differences_by_order(source, rows)
    directions = build_directions(by_order, order_shift, metadata)
    scored = mark_clean(
        {
            name: score_direction(eval_states @ direction, eval_rows)
            for name, direction in directions.items()
        },
        bootstrap_order_ci(source, rows, eval_states, eval_rows, 60, 2),
    )
    if scored["order_averaged"]["clean"]:
        raise AssertionError(
            "with cos(a,d)=0.8 the un-orthogonalized direction must be contaminated"
        )
    if scored["legacy_recipe"]["clean"]:
        raise AssertionError("legacy recipe must be contaminated too")
    # Geometry, not the estimator, is what drives it: the same unbiased estimate
    # is far more contaminated when the axes are correlated than when they aren't.
    if (
        scored["order_averaged"]["order_contamination"]
        <= orthogonal_case["order_averaged"]["order_contamination"] + 0.2
    ):
        raise AssertionError(
            "contamination should be driven by cos(a,d), not by the estimator: "
            f"{orthogonal_case['order_averaged']['order_contamination']:.4f} at "
            f"cos 0 versus {scored['order_averaged']['order_contamination']:.4f} "
            "at cos 0.8"
        )
    if not scored["orthogonalized"]["clean"]:
        raise AssertionError("orthogonalization should repair the projection")
    if scored["orthogonalized"]["label_auroc"] < 0.90:
        raise AssertionError("orthogonalized direction should still read the label")
    if scored["order_difference"]["clean"]:
        raise AssertionError("order_difference should be flagged contaminated")

    # --- The gate has to be calibrated, or it manufactures its own verdict. ---
    # A fixed tolerance, or an analytic band on the evaluation sample alone,
    # rejects a genuinely clean direction about a fifth of the time here, because
    # neither sees the variance contributed by estimating the direction.
    false_positives = 0
    replications = 40
    for replication in range(replications):
        _axis, clean_source, clean_rows, clean_states = synthesize(0.0, 500 + replication)
        by_order, order_shift, metadata = paired_differences_by_order(clean_source, rows)
        direction = build_directions(by_order, order_shift, metadata)["orthogonalized"]
        entry = mark_clean(
            {
                "orthogonalized": score_direction(
                    clean_states @ direction, clean_rows
                )
            },
            {
                "orthogonalized": bootstrap_order_ci(
                    clean_source, rows, clean_states, clean_rows, 40, replication
                )["orthogonalized"]
            },
        )["orthogonalized"]
        if not entry["clean"]:
            false_positives += 1
    rate = false_positives / replications
    if rate > 0.15:
        raise AssertionError(
            f"the contamination bar is mis-calibrated: a clean direction is "
            f"rejected {rate:.1%} of the time"
        )

    profile = {16: scored, 20: scored}
    selected = select_layer(profile)
    decision, reasons = reextract_decision(profile, selected)
    if selected != 16 or decision != "CLEAN_DIRECTION" or not reasons:
        raise AssertionError(f"decision path wrong: {selected} {decision} {reasons}")

    empty = {16: {"orthogonalized": {"clean": False, "label_auroc": 0.5}}}
    if select_layer(empty) is not None:
        raise AssertionError("select_layer should return None with no clean layer")
    if reextract_decision(empty, None)[0] != "NO_CLEAN_DIRECTION":
        raise AssertionError("no-clean-layer branch wrong")

    # The source manifest really is crossed and parity independent.
    _rows, audit = crossed_source_rows(8, SOURCE_SEEDS[0])
    for key in (
        "catalog_order_parity_independent",
        "catalog_order_crossed_within_scenario",
        "control_label_parity_independent",
    ):
        if not audit[key]:
            raise AssertionError(f"source manifest failed {key}")

    print(
        json.dumps(
            {
                "self_test": "PASS",
                "both_estimators_are_unbiased": True,
                "clean_direction_false_rejection_rate": rate,
                "orthogonal_axes_case": {
                    name: {
                        "label_auroc": round(entry["label_auroc"], 4),
                        "order_contamination": round(entry["order_contamination"], 4),
                        "clean": entry["clean"],
                    }
                    for name, entry in orthogonal_case.items()
                },
                "cos_a_d_is_0.8_case": {
                    name: {
                        "label_auroc": round(entry["label_auroc"], 4),
                        "order_contamination": round(entry["order_contamination"], 4),
                        "clean": entry["clean"],
                    }
                    for name, entry in scored.items()
                },
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    source_rows_all: list[dict[str, Any]] = []
    source_audits = {}
    for seed in SOURCE_SEEDS:
        rows, audit = crossed_source_rows(args.source_pairs_per_family, seed)
        source_rows_all.extend(rows)
        source_audits[str(seed)] = audit

    eval_manifest = build_manifest(
        pairs_per_family=args.pairs_per_family,
        repeats=1,
        seed=EVAL_SEED,
        control_label_mode="parity_independent",
        catalog_order_mode="crossed",
    )
    eval_audit = validate_manifest(
        eval_manifest,
        args.pairs_per_family * 2,
        1,
        require_parity_independent=True,
        require_order_crossed=True,
    )
    eval_rows = crossover_eval_rows(eval_manifest)

    config = {
        "protocol": "ARM_G_REEXTRACT_V1",
        "model": args.model,
        "source_seeds": list(SOURCE_SEEDS),
        "eval_seed": EVAL_SEED,
        "sweep_layers": list(SWEEP_LAYERS),
        "intervention": "none; baseline capture only",
        "recipes": list(RECIPES),
        "aggregation": (
            "h(C,O) = mu + a*C + b*O + d*C*O. The within-scenario paired "
            "difference gives 2a + 2d*o, so averaging over both orders cancels d "
            "exactly and leaves the pure condition main effect. One order per "
            "pair, which a parity-locked generator forces, folds the "
            "requested-line effect into the direction"
        ),
        "success_rule": (
            f"the orthogonalized direction must read the label at AUROC >= "
            f"{LABEL_AUROC_GATE} while its order AUROC within each condition "
            f"stays within {ORDER_CONTAMINATION_BAR} of chance, about 2.6 null "
            "SDs where the contaminated case sits at the 0.5 ceiling. The other "
            "four recipes are built from the same states as controls"
        ),
        "pairs_per_family": args.pairs_per_family,
        "source_pairs_per_family": args.source_pairs_per_family,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "read_position": PRIMARY_POSITION,
        "prior_evidence": PRIOR_EVIDENCE,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_or_write(args.output_dir / "run_config.json", config)

    hardware = require_gpu(args.allow_non_a100)
    tokenizer, model = load_acting_model(args.model, args.hf_token)
    model_commit = getattr(model.config, "_commit_hash", None)
    if model_commit not in (None, EXPECTED_MODEL_COMMIT):
        raise RuntimeError(f"model commit {model_commit} differs from frozen")
    token_ids = {}
    for token in ACTION_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"action token {token!r} is not single: {encoded}")
        token_ids[token] = int(encoded[0])

    _margins, source_states = run_forward(
        model, tokenizer, source_rows_all, args.batch_size, {}, SWEEP_LAYERS, None
    )
    eval_margins, eval_states = run_forward(
        model, tokenizer, eval_rows, args.batch_size, {}, SWEEP_LAYERS, token_ids
    )

    profile: dict[int, dict[str, Any]] = {}
    cosines: dict[int, dict[str, float]] = {}
    for layer in SWEEP_LAYERS:
        by_order, order_shift, metadata = paired_differences_by_order(
            source_states[layer], source_rows_all
        )
        directions = build_directions(by_order, order_shift, metadata)
        profile[layer] = mark_clean(
            {
                name: score_direction(eval_states[layer] @ direction, eval_rows)
                for name, direction in directions.items()
            },
            bootstrap_order_ci(
                source_states[layer],
                source_rows_all,
                eval_states[layer],
                eval_rows,
                BOOTSTRAP_REPETITIONS,
                EVAL_SEED + layer,
            ),
        )
        cosines[layer] = {
            f"{left}_vs_{right}": float(directions[left] @ directions[right])
            for index, left in enumerate(RECIPES)
            for right in RECIPES[index + 1 :]
        }

    selected = select_layer(profile)
    decision, decision_reasons = reextract_decision(profile, selected)

    selected_direction = None
    selected_projections: dict[str, np.ndarray] = {}
    if selected is not None:
        by_order, order_shift, metadata = paired_differences_by_order(
            source_states[selected], source_rows_all
        )
        selected_directions = build_directions(by_order, order_shift, metadata)
        selected_direction = selected_directions["order_averaged"]
        selected_projections = {
            name: eval_states[selected] @ direction
            for name, direction in selected_directions.items()
        }

    report = {
        "status": "ARM_G_REEXTRACT_V1",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "selected_layer": selected,
        "layer_profile": {str(layer): profile[layer] for layer in SWEEP_LAYERS},
        "direction_cosines": {str(layer): cosines[layer] for layer in SWEEP_LAYERS},
        "selected_direction": (
            None if selected_direction is None else selected_direction.tolist()
        ),
        "prior_evidence": PRIOR_EVIDENCE,
        "sample_counts": {
            "source_rows": len(source_rows_all),
            "source_scenarios": len(source_rows_all) // 4,
            "evaluation_rows": len(eval_rows),
            "layers": list(SWEEP_LAYERS),
        },
        "audits": {
            "source_manifests": source_audits,
            "evaluation_manifest": eval_audit,
        },
        "row_results": [
            {
                **{key: value for key, value in row.items() if key != "messages"},
                "baseline_semantic_margin": float(eval_margins[index]),
                "selected_layer_projections": {
                    name: float(values[index])
                    for name, values in selected_projections.items()
                },
            }
            for index, row in enumerate(eval_rows)
        ],
        "provenance": {
            "acting_model": args.model,
            "expected_model_commit": EXPECTED_MODEL_COMMIT,
            "model_commit": model_commit,
            "dtype": "bfloat16",
            "hardware": hardware,
            "label_source": "mechanical target-path achievability",
            "package_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "numpy": package_version("numpy"),
            },
        },
    }
    result_path = args.output_dir / "arm_g_reextract_result.json"
    atomic_json(result_path, report)
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "decision": decision,
                "decision_reasons": decision_reasons,
                "selected_layer": selected,
                "layer_profile": report["layer_profile"],
                "direction_cosines": report["direction_cosines"],
                "sample_counts": report["sample_counts"],
            },
            indent=2,
        )
    )
    print(f"full artifact: {result_path}")


if __name__ == "__main__":
    main()
