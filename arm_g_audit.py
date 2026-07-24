"""Mathematical audit of the Arm G cross-layer mediation result.

Nothing here re-runs the model.  Every number is recomputed from the row-level
margins stored in the seed-106 artifact, using estimator code written
independently of the code under audit, so that agreement is evidence and
disagreement is a defect.

Sections:

1. Design balance and the estimator identity.  The reported point estimate is a
   grand-mean condition contrast; the confidence interval bootstraps a
   family-stratified mean of per-pair contrasts.  Those are different
   expressions and are only the same estimand under exact cell balance, which
   is checked rather than assumed.
2. Independent recomputation of every reported contrast, attenuation and
   fraction.
3. Interval robustness.  The percentile bootstrap under audit is compared
   against a basic bootstrap, a normal approximation, a paired t interval, an
   exact sign test and an exact-in-distribution sign-flip permutation test.
4. Selection-adjusted inference for the layer-16 peak.  A confidence interval
   at the arg-max layer is not honest about the selection; this bootstraps the
   arg-max itself and tests the ordering directly.
5. Perturbation magnitude.  The layer ordering of causal effect is compared
   against the ordering of intervention size, because an effect that merely
   tracks how hard each layer was hit is not a claim about that layer.
6. Decision invariance.  Quantifies the limit already recorded in the ledger.
7. Operator algebra.  Verifies the ablation operator's projector properties,
   its mean-preservation, and the cascade consistency property that the
   sequential design depends on.
"""

from __future__ import annotations

import base64
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ARTIFACT = Path(
    "results/arm_g_cross_layer_seed106_v1/arm_g_cross_layer_result.json.gz.b64"
)
LAYERS = (12, 16, 20, 24, 27, 30)
ANCHOR_LAYER = 27
BOOTSTRAP = 20000
TOLERANCE = 1e-9


def load_artifact(path: Path = ARTIFACT) -> dict[str, Any]:
    return json.loads(gzip.decompress(base64.b64decode(path.read_bytes())))


def margin_vectors(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    vectors = {"baseline": np.array([r["baseline_semantic_margin"] for r in rows])}
    for layer in LAYERS:
        vectors[f"single_{layer}"] = np.array(
            [r["single_layer_semantic_margins"][str(layer)] for r in rows]
        )
        vectors[f"cumulative_{layer}"] = np.array(
            [r["cumulative_semantic_margins"][str(layer)] for r in rows]
        )
    return vectors


def grand_mean_contrast(
    margins: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> float:
    labels = np.array([r["condition_label"] for r in rows])
    return float(margins[labels == 1].mean() - margins[labels == 0].mean())


def pair_contrasts(
    margins: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for margin, row in zip(margins, rows, strict=True):
        cells[
            (
                row["family"],
                row["pair_id"],
                row["mapping_variant"],
                row["condition_label"],
            )
        ].append(float(margin))
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for family, pair_id, mapping, label in cells:
        if label != 1:
            continue
        conflict = cells[(family, pair_id, mapping, 1)]
        reachable = cells[(family, pair_id, mapping, 0)]
        if len(conflict) != 1 or len(reachable) != 1:
            raise AssertionError("cell is not a singleton")
        out[(family, pair_id)].append(conflict[0] - reachable[0])
    return {key: float(np.mean(values)) for key, values in out.items()}


def stratified_mean(values: Mapping[tuple[str, str], float]) -> float:
    by_family: dict[str, list[float]] = defaultdict(list)
    for (family, _pair), value in values.items():
        by_family[family].append(value)
    return float(np.mean([np.mean(v) for _, v in sorted(by_family.items())]))


def stratified_draws(
    values: Mapping[tuple[str, str], float],
    repetitions: int,
    seed: int,
) -> np.ndarray:
    by_family = defaultdict(list)
    for (family, _pair), value in sorted(values.items()):
        by_family[family].append(value)
    arrays = [np.asarray(by_family[f]) for f in sorted(by_family)]
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        draws[i] = np.mean(
            [rng.choice(a, size=len(a), replace=True).mean() for a in arrays]
        )
    return draws


def interval_battery(
    values: Mapping[tuple[str, str], float],
    seed: int,
) -> dict[str, Any]:
    point = stratified_mean(values)
    draws = stratified_draws(values, BOOTSTRAP, seed)
    percentile = np.quantile(draws, [0.025, 0.975])
    basic = np.array([2 * point - percentile[1], 2 * point - percentile[0]])
    se = float(draws.std(ddof=1))
    normal = np.array([point - 1.96 * se, point + 1.96 * se])

    flat = np.array(list(values.values()))
    n = len(flat)
    t_ci = stats.t.interval(0.95, n - 1, loc=flat.mean(), scale=stats.sem(flat))
    positive = int(np.sum(flat > 0))
    sign_p = float(stats.binomtest(positive, n, 0.5).pvalue)
    wilcoxon_p = float(stats.wilcoxon(flat).pvalue)

    rng = np.random.default_rng(seed + 1)
    signs = rng.choice([-1.0, 1.0], size=(BOOTSTRAP, n))
    null = (signs * flat).mean(axis=1)
    perm_p = float((np.abs(null) >= abs(flat.mean())).mean())

    return {
        "point_estimate": point,
        "bootstrap_mean": float(draws.mean()),
        "bootstrap_se": se,
        "percentile_ci": percentile.tolist(),
        "basic_ci": basic.tolist(),
        "normal_ci": normal.tolist(),
        "paired_t_ci": [float(t_ci[0]), float(t_ci[1])],
        "pairs_positive": f"{positive}/{n}",
        "sign_test_p": sign_p,
        "wilcoxon_p": wilcoxon_p,
        "sign_flip_permutation_p": perm_p,
        "all_intervals_exclude_zero": bool(
            percentile[0] > 0 and basic[0] > 0 and normal[0] > 0 and t_ci[0] > 0
        ),
    }


def section_1_balance(report: Mapping[str, Any], rows) -> dict[str, Any]:
    cells = defaultdict(int)
    for row in rows:
        cells[(row["family"], row["mapping_variant"], row["condition_label"])] += 1
    counts = sorted(set(cells.values()))
    balanced = len(counts) == 1
    vectors = margin_vectors(rows)
    identity = []
    for name, margins in vectors.items():
        grand = grand_mean_contrast(margins, rows)
        strat = stratified_mean(pair_contrasts(margins, rows))
        identity.append(abs(grand - strat))
    return {
        "cells": len(cells),
        "rows_per_cell": counts,
        "design_balanced": balanced,
        "estimator_identity_max_abs_gap": float(max(identity)),
        "estimator_identity_holds": bool(max(identity) < TOLERANCE),
        "note": (
            "under exact balance the grand-mean contrast and the "
            "family-stratified mean of per-pair contrasts are algebraically "
            "the same statistic, so the reported point estimate and the "
            "bootstrapped quantity share an estimand"
        ),
    }


def section_2_recompute(report: Mapping[str, Any], rows) -> dict[str, Any]:
    vectors = margin_vectors(rows)
    baseline = vectors["baseline"]
    base_contrast = grand_mean_contrast(baseline, rows)
    base_pairs = pair_contrasts(baseline, rows)
    checks = []
    worst = 0.0
    for layer in LAYERS:
        for kind, block in (
            ("single_layer", report["single_layer"]),
            ("cumulative", report["cumulative"]),
        ):
            entry = block[str(layer)]
            margins = vectors[
                f"{'single' if kind == 'single_layer' else 'cumulative'}_{layer}"
            ]
            contrast = grand_mean_contrast(margins, rows)
            attenuation = base_contrast - contrast
            fraction = attenuation / base_contrast
            gaps = {
                "contrast": abs(contrast - entry["condition_contrast"]["overall"]),
                "attenuation": abs(attenuation - entry["attenuation"]["overall"]),
                "fraction": abs(fraction - entry["attenuation_fraction"]),
            }
            worst = max(worst, *gaps.values())
            checks.append(
                {"kind": kind, "layer": layer, "max_abs_gap": float(max(gaps.values()))}
            )
    base_gap = abs(base_contrast - report["baseline"]["condition_contrast"]["overall"])
    worst = max(worst, base_gap)
    return {
        "baseline_contrast_recomputed": base_contrast,
        "baseline_gap": float(base_gap),
        "independent_pairs": len(base_pairs),
        "statistics_checked": len(checks) * 3 + 1,
        "max_abs_disagreement": float(worst),
        "all_reported_statistics_reproduce": bool(worst < 1e-6),
    }


def section_3_intervals(report: Mapping[str, Any], rows) -> dict[str, Any]:
    vectors = margin_vectors(rows)
    base_pairs = pair_contrasts(vectors["baseline"], rows)
    full_pairs = pair_contrasts(vectors[f"cumulative_{max(LAYERS)}"], rows)
    anchor_pairs = pair_contrasts(vectors[f"single_{ANCHOR_LAYER}"], rows)

    full_att = {k: base_pairs[k] - full_pairs[k] for k in base_pairs}
    anchor_att = {k: base_pairs[k] - anchor_pairs[k] for k in base_pairs}
    increment = {k: full_att[k] - anchor_att[k] for k in base_pairs}

    reported_full = report["cumulative"][str(max(LAYERS))]["attenuation_bootstrap"]
    reported_inc = report["depth_increment"]["full_minus_anchor_layer"][
        "paired_bootstrap"
    ]

    families = sorted({family for family, _pair in full_att})
    per_family = {}
    for index, family in enumerate(families):
        subset = {k: v for k, v in full_att.items() if k[0] == family}
        battery = interval_battery(subset, 40 + index)
        per_family[family] = {
            "point_estimate": battery["point_estimate"],
            "percentile_ci": battery["percentile_ci"],
            "pairs_positive": battery["pairs_positive"],
            "sign_flip_permutation_p": battery["sign_flip_permutation_p"],
        }

    # The headline percentage is a ratio of two random quantities and is
    # reported without an interval, so bootstrap the ratio jointly.
    keys = sorted(full_att)
    by_family = defaultdict(list)
    for index, (family, _pair) in enumerate(keys):
        by_family[family].append(index)
    numerator = np.array([full_att[k] for k in keys])
    denominator = np.array([base_pairs[k] for k in keys])
    rng = np.random.default_rng(13)
    ratios = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        num, den = [], []
        for family in sorted(by_family):
            idx = np.asarray(by_family[family])
            drawn = rng.choice(idx, size=len(idx), replace=True)
            num.append(numerator[drawn].mean())
            den.append(denominator[drawn].mean())
        ratios[i] = np.mean(num) / np.mean(den)

    return {
        "full_cascade_attenuation": {
            **interval_battery(full_att, 11),
            "reported_percentile_ci": reported_full["ci_95"],
        },
        "attenuation_fraction_ratio_bootstrap": {
            "point_estimate": float(
                stratified_mean(full_att) / stratified_mean(base_pairs)
            ),
            "percentile_ci": np.quantile(ratios, [0.025, 0.975]).tolist(),
            "note": (
                "the reported 27.48% share carries no interval in the run "
                "artifact; this resamples numerator and denominator jointly "
                "on the same pairs"
            ),
        },
        "full_minus_anchor_increment": {
            **interval_battery(increment, 12),
            "reported_percentile_ci": reported_inc["ci_95"],
        },
        "per_family_conditional": per_family,
        "family_generalization_caveat": (
            "the stratified bootstrap resamples pairs within each family and "
            "therefore treats the two family effects as fixed. With two "
            "families the between-family component cannot be bootstrapped at "
            "all, so every interval here is conditional on these two held-out "
            "families. Generalization to unseen families rests on both "
            "families showing the effect independently, not on the width of "
            "these intervals."
        ),
    }


def section_4_selection(report: Mapping[str, Any], rows) -> dict[str, Any]:
    vectors = margin_vectors(rows)
    base_pairs = pair_contrasts(vectors["baseline"], rows)
    att = {
        layer: {
            k: base_pairs[k] - pair_contrasts(vectors[f"single_{layer}"], rows)[k]
            for k in base_pairs
        }
        for layer in LAYERS
    }
    keys = sorted(base_pairs)
    by_family = defaultdict(list)
    for index, (family, _pair) in enumerate(keys):
        by_family[family].append(index)
    matrix = np.array([[att[layer][k] for k in keys] for layer in LAYERS])

    rng = np.random.default_rng(21)
    argmax_counts = defaultdict(int)
    for _ in range(BOOTSTRAP):
        picks = []
        for family in sorted(by_family):
            idx = np.asarray(by_family[family])
            picks.append(rng.choice(idx, size=len(idx), replace=True))
        drawn = np.concatenate(picks)
        family_means = []
        for family in sorted(by_family):
            n = len(by_family[family])
            family_means.append(matrix[:, drawn[:n]].mean(axis=1))
            drawn = drawn[n:]
        argmax_counts[LAYERS[int(np.argmax(np.mean(family_means, axis=0)))]] += 1

    peak = 16
    contrasts = {}
    for layer in LAYERS:
        if layer == peak:
            continue
        diff = {k: att[peak][k] - att[layer][k] for k in keys}
        battery = interval_battery(diff, 30 + layer)
        contrasts[f"layer{peak}_minus_layer{layer}"] = {
            "point_estimate": battery["point_estimate"],
            "percentile_ci": battery["percentile_ci"],
            "bonferroni_ci_99.0": np.quantile(
                stratified_draws(diff, BOOTSTRAP, 30 + layer), [0.005, 0.995]
            ).tolist(),
            "sign_flip_permutation_p": battery["sign_flip_permutation_p"],
            "pairs_positive": battery["pairs_positive"],
        }
    return {
        "bootstrap_argmax_distribution": {
            str(k): v / BOOTSTRAP for k, v in sorted(argmax_counts.items())
        },
        "peak_layer_stability": argmax_counts[peak] / BOOTSTRAP,
        "pairwise_contrasts_against_peak": contrasts,
        "bonferroni_note": (
            "five pairwise comparisons against the selected peak; the 99.0% "
            "intervals are the Bonferroni-adjusted analogues of 95%"
        ),
    }


def section_5_perturbation(report: Mapping[str, Any], rows) -> dict[str, Any]:
    out = {}
    for layer in LAYERS:
        centers = report["single_layer"][str(layer)]["mapping_centers"]
        displacement = []
        coordinates = []
        for row in rows:
            projection = np.asarray(row["target_projections"][str(layer)])
            center = np.asarray(centers[row["mapping_variant"]])
            coordinates.append(projection - center)
            displacement.append(float(np.linalg.norm(projection - center)))
        displacement = np.asarray(displacement)
        energy = (np.stack(coordinates) ** 2).sum(axis=0)
        attenuation = report["single_layer"][str(layer)]["attenuation"]["overall"]
        out[str(layer)] = {
            "mean_displacement_norm": float(displacement.mean()),
            "rms_displacement_norm": float(np.sqrt((displacement**2).mean())),
            "attenuation": attenuation,
            "attenuation_per_unit_displacement": float(
                attenuation / displacement.mean()
            ),
            "displacement_energy_by_coordinate": (energy / energy.sum()).tolist(),
            "rank1_share_of_displacement_energy": float(energy[0] / energy.sum()),
        }
    order_effect = sorted(LAYERS, key=lambda x: -out[str(x)]["attenuation"])
    order_size = sorted(LAYERS, key=lambda x: -out[str(x)]["mean_displacement_norm"])
    effects = np.array([out[str(x)]["attenuation"] for x in LAYERS])
    sizes = np.array([out[str(x)]["mean_displacement_norm"] for x in LAYERS])
    return {
        "per_layer": out,
        "effect_ordering": order_effect,
        "perturbation_size_ordering": order_size,
        "orderings_agree": order_effect == order_size,
        "spearman_effect_vs_size": float(stats.spearmanr(effects, sizes).statistic),
        "note": (
            "if the causal ordering simply tracked how large the removed "
            "component was, the two orderings would coincide and the "
            "normalized column would be flat"
        ),
    }


def section_6_decisions(report: Mapping[str, Any], rows) -> dict[str, Any]:
    vectors = margin_vectors(rows)
    baseline = vectors["baseline"]
    labels = np.array([r["condition_label"] for r in rows])
    out = {}
    for name in ("single_16", f"cumulative_{max(LAYERS)}"):
        margins = vectors[name]
        out[name] = {
            "auroc_baseline": float(roc_auc_score(labels, baseline)),
            "auroc_ablated": float(roc_auc_score(labels, margins)),
            "spearman_vs_baseline": float(stats.spearmanr(baseline, margins).statistic),
            "sign_flips": int(np.sum(np.sign(baseline) != np.sign(margins))),
            "rows": int(len(margins)),
            "mean_shift_reachable": float(
                margins[labels == 0].mean() - baseline[labels == 0].mean()
            ),
            "mean_shift_conflict": float(
                margins[labels == 1].mean() - baseline[labels == 1].mean()
            ),
        }
    return out


def section_7_operator() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    width, rank, n = 64, 4, 200
    basis = np.linalg.qr(rng.normal(size=(width, rank)))[0]
    states = rng.normal(size=(n, width)) * 2.5 + rng.normal(size=width)
    groups = rng.integers(0, 2, size=n)

    projector = basis @ basis.T
    idempotent = float(np.max(np.abs(projector @ projector - projector)))
    symmetric = float(np.max(np.abs(projector - projector.T)))

    projections = states @ basis
    centers = np.zeros_like(projections)
    for g in (0, 1):
        centers[groups == g] = projections[groups == g].mean(axis=0)
    ablated = states - (projections - centers) @ basis.T

    residual = float(np.max(np.abs((ablated @ basis) - centers)))
    orthogonal_preserved = float(
        np.max(np.abs((states - ablated) - (states - ablated) @ projector))
    )
    group_means = float(
        max(
            np.max(
                np.abs(
                    ablated[groups == g].mean(axis=0) - states[groups == g].mean(axis=0)
                )
            )
            for g in (0, 1)
        )
    )
    # Idempotence: A(A(x)) == A(x).  The operator must be applied to its own
    # output, not to the original state.
    twice = ablated - ((ablated @ basis) - centers) @ basis.T
    operator_idempotent = float(np.max(np.abs(twice - ablated)))

    # And with the centers recomputed from the already-ablated states, as the
    # pipeline would do on a second pass.
    recentered = np.zeros_like(projections)
    reprojected = ablated @ basis
    for g in (0, 1):
        recentered[groups == g] = reprojected[groups == g].mean(axis=0)
    twice_recentered = ablated - (reprojected - recentered) @ basis.T
    operator_idempotent_recentered = float(np.max(np.abs(twice_recentered - ablated)))

    # Cascade consistency: the state reaching a downstream layer is unchanged
    # by whether that layer's own ablation is scheduled, so centers measured on
    # a capture pass remain valid on the scoring pass.
    upstream = np.linalg.qr(rng.normal(size=(width, rank)))[0]
    up_proj = states @ upstream
    up_centers = np.zeros_like(up_proj)
    for g in (0, 1):
        up_centers[groups == g] = up_proj[groups == g].mean(axis=0)
    after_upstream = states - (up_proj - up_centers) @ upstream.T
    captured = after_upstream.copy()
    down_proj = captured @ basis
    down_centers = np.zeros_like(down_proj)
    for g in (0, 1):
        down_centers[groups == g] = down_proj[groups == g].mean(axis=0)
    cascaded = after_upstream - ((after_upstream @ basis) - down_centers) @ basis.T
    cascade_residual = float(np.max(np.abs((cascaded @ basis) - down_centers)))

    return {
        "projector_idempotent_max_err": idempotent,
        "projector_symmetric_max_err": symmetric,
        "subspace_coordinate_pinned_to_group_mean": residual,
        "displacement_lies_in_subspace": orthogonal_preserved,
        "group_means_preserved_max_err": group_means,
        "operator_idempotent_max_err": operator_idempotent,
        "operator_idempotent_recentered_max_err": operator_idempotent_recentered,
        "cascade_downstream_residual": cascade_residual,
        "all_properties_hold": bool(
            max(
                idempotent,
                symmetric,
                residual,
                orthogonal_preserved,
                group_means,
                operator_idempotent,
                operator_idempotent_recentered,
                cascade_residual,
            )
            < 1e-9
        ),
    }


def main() -> None:
    report = load_artifact()
    rows = report["row_results"]
    audit = {
        "artifact": str(ARTIFACT),
        "audited_decision": report["decision"],
        "bootstrap_repetitions": BOOTSTRAP,
        "section_1_balance_and_estimator_identity": section_1_balance(report, rows),
        "section_2_independent_recomputation": section_2_recompute(report, rows),
        "section_3_interval_robustness": section_3_intervals(report, rows),
        "section_4_selection_adjusted": section_4_selection(report, rows),
        "section_5_perturbation_magnitude": section_5_perturbation(report, rows),
        "section_6_decision_invariance": section_6_decisions(report, rows),
        "section_7_operator_algebra": section_7_operator(),
    }
    out = Path("results/arm_g_cross_layer_seed106_v1/arm_g_audit.json")
    out.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"\naudit artifact: {out}")


if __name__ == "__main__":
    main()
