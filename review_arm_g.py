"""Independent recomputation behind ADVERSARIAL_REVIEW.md.

Reads only the committed per-row artifacts. No GPU, no network, no model.
Every number quoted in the review is printed here.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

BOUNDARY = Path("results/arm_g_boundary_seed110_v1/arm_g_boundary_result.json.gz.b64")
ALLPOS = Path("results/arm_g_allpos_seed108_v1/arm_g_allpos_result.json.gz.b64")
LAYER16 = Path("results/arm_g_layer16_seed107_v2/arm_g_layer16_result.json.gz.b64")


def load(path: Path) -> dict[str, Any]:
    return json.loads(gzip.decompress(base64.b64decode(path.read_text())))


def pair_index(row: dict[str, Any]) -> int:
    return int(str(row["pair_id"]).split(":")[1])


def auroc(scores: Sequence[float], labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=float)
    positive, negative = values[labels == 1], values[labels == 0]
    wins = sum(
        1.0 if a > b else 0.5 if a == b else 0.0 for a in positive for b in negative
    )
    return wins / (len(positive) * len(negative))


def best_threshold_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    cuts = np.unique(np.concatenate([scores - 1e-9, scores + 1e-9]))
    return max(((scores > cut) == (labels == 1)).mean() for cut in cuts)


def parity_determinism() -> None:
    print("(1) PARITY DETERMINISM  [seed 110 and seed 108]")
    for tag, path in (("110", BOUNDARY), ("108", ALLPOS)):
        conflict = [r for r in load(path)["row_results"] if r["condition"] == "conflict"]
        holds = all(
            (r["baseline_semantic_margin"] > 0) == (pair_index(r) % 2 == 0)
            for r in conflict
        )
        print(
            f"   seed{tag}: conflict n={len(conflict)}  "
            f"'parity 0 <=> DECLINE' holds for ALL rows: {holds}"
        )
    print(f"   P(twice by chance | 50% rate) ~ 2*(0.5)^64 = {2 * 0.5**64:.1e}")


def empty_band(margin: np.ndarray, labels: np.ndarray) -> None:
    print("\n(2) EMPTY BAND")
    conflict = np.sort(margin[labels == 1])
    low, high = conflict[:32], conflict[32:]
    print(
        f"   conflict margins: 32 in [{low.min():+.2f},{low.max():+.2f}], "
        f"32 in [{high.min():+.2f},{high.max():+.2f}] "
        f"-> gap {high.min() - low.max():.3f} margin units"
    )
    print(f"   nearest reachable row to boundary: {margin[labels == 0].max():+.3f}")


def direction_reads_parity(
    projection: np.ndarray, labels: np.ndarray, parity: np.ndarray
) -> None:
    print("\n(3) DIRECTION READS PARITY BETTER THAN LABEL")
    print(
        f"   seed110 projection: AUROC(condition)={auroc(projection, labels):.4f}   "
        "AUROC(parity | conflict)="
        f"{auroc(projection[labels == 1], 1 - parity[labels == 1]):.4f}"
    )
    focus = load(LAYER16)["row_results"]
    focus_labels = np.array([row["condition_label"] for row in focus])
    focus_parity = np.array([pair_index(row) % 2 for row in focus])
    conflict = focus_labels == 1
    for component in range(4):
        values = np.array(
            [row["conflict_projections_at_focus"][component] for row in focus],
            dtype=float,
        )
        print(
            f"   seed107 comp{component}: "
            f"AUROC(condition)={auroc(values, focus_labels):.4f}   "
            "AUROC(parity | conflict)="
            f"{auroc(values[conflict], 1 - focus_parity[conflict]):.4f}"
        )


def between_cell_variance(
    slope: np.ndarray, margin: np.ndarray, cell: np.ndarray
) -> None:
    print("\n(4) SEC. CLAIM 4.5 -- the variance is between four cells")
    for name, values in (("slope", slope), ("margin", margin)):
        between = np.var([values[cell == c].mean() for c in np.unique(cell)])
        print(f"   {name}: between-cell / total variance = {between / values.var():.1%}")
    pooled = stats.pearsonr(slope, margin).statistic
    demeaned = [
        np.concatenate([values[cell == c] - values[cell == c].mean() for c in np.unique(cell)])
        for values in (slope, margin)
    ]
    within = stats.pearsonr(*demeaned).statistic
    print(
        f"   r(slope,margin): pooled={pooled:+.4f} (R2={pooled**2:.3f})  "
        f"within-cell={within:+.4f} (R2={within**2:.3f})"
    )


def saturated_fit(margin: np.ndarray, labels: np.ndarray, parity: np.ndarray) -> None:
    print("\n(5) THE MARGIN-LEVEL CONDITION CONTRAST IS ESTIMABLE ANYWAY")
    means = {
        (label, par): margin[(labels == label) & (parity == par)].mean()
        for label in (0, 1)
        for par in (0, 1)
    }
    # Columns are intercept, condition, parity, and requested-target-line, which
    # is -condition*parity. All four are orthogonal, so the fit is saturated.
    signs = ((+1, -1), (+1, +1), (-1, -1), (-1, +1))
    design = np.array(
        [[1, condition, par, -condition * par] for condition, par in signs],
        dtype=float,
    )
    observed = np.array(
        [means[(1, 0)], means[(1, 1)], means[(0, 0)], means[(0, 1)]], dtype=float
    )
    beta = np.linalg.solve(design, observed)
    print(f"   saturated fit, rank {np.linalg.matrix_rank(design)} of 4, 0 residual df")
    print(
        f"   intercept={beta[0]:+.3f}  condition={beta[1]:+.3f} "
        f"(contrast {2 * beta[1]:+.3f})  parity={beta[2]:+.3f}  "
        f"requested-line={beta[3]:+.3f}"
    )
    print("   -> the scope contrast is not what the crossover is testing; the")
    print("      decision is, and the parity term is uninterpretable.")


def ablation_is_a_pinned_dose(
    rows: list[dict[str, Any]], projection: np.ndarray, sigma: float
) -> None:
    print("\n(6) ABLATION IS AN ADDITIVE DOSE, AND IT IS PINNED")
    dose = -projection / sigma
    crossing = np.array(
        [
            np.nan if row["crossing_dose_sigma"] is None else row["crossing_dose_sigma"]
            for row in rows
        ]
    )
    reachable = (np.sign(crossing) == np.sign(dose)) & (np.abs(dose) >= np.abs(crossing))
    print("   x <- x - (x.r)r  ==  x <- x + c_i*sigma*r with c_i = -proj_i/sigma")
    print("   (arm_g_allpos.py:162-164, center=0)")
    print(
        f"   c_i: mean={dose.mean():+.3f}s sd={dose.std():.3f} "
        f"range=[{dose.min():+.3f},{dose.max():+.3f}]s"
    )
    print(
        "   rows where |c_i| >= |crossing dose| in the same direction: "
        f"{int(reachable.sum())}/128"
    )
    print(
        f"   -> additive curve predicts {int(reachable.sum())} flips for full "
        "ablation; observed 0/128 (seed108 layer16_all_positions)"
    )


def operating_point(rows: list[dict[str, Any]], margin: np.ndarray, labels: np.ndarray) -> None:
    print("\n(7) ADDITION MOVES THE OPERATING POINT AND ONLY DEGRADES DISCRIMINATION")
    keys = sorted(rows[0]["margins_by_dose"], key=float)
    doses = np.array([float(key) for key in keys])
    matrix = np.array([[row["margins_by_dose"][key] for key in keys] for row in rows])
    print(
        f"   {'dose':>5} {'AUROC':>8} {'acc@0':>7} {'best-thresh':>12} "
        f"{'corrections':>12} {'NEW errors':>11}"
    )
    for index, dose in enumerate(doses):
        scores = matrix[:, index]
        declined_conflict = int((scores[labels == 1] > 0).sum())
        declined_reachable = int((scores[labels == 0] > 0).sum())
        print(
            f"   {dose:>5.1f} {auroc(scores, labels):>8.5f} "
            f"{((scores > 0) == (labels == 1)).mean():>7.4f} "
            f"{best_threshold_accuracy(scores, labels):>12.4f} "
            f"{declined_conflict - 32:>+12d} {declined_reachable:>+11d}"
        )
    steered = max(
        ((matrix[:, index] > 0) == (labels == 1)).mean() for index in range(len(doses))
    )
    aurocs = [auroc(matrix[:, index], labels) for index in range(len(doses))]
    print(
        "   baseline best-threshold accuracy = "
        f"{best_threshold_accuracy(margin, labels):.4f};  "
        f"best accuracy at ANY dose = {steered:.4f}"
    )
    print(f"   argmax AUROC over doses = {doses[int(np.argmax(aurocs))]:+.1f} sigma")


def main() -> None:
    boundary = load(BOUNDARY)
    rows = boundary["row_results"]
    labels = np.array([row["condition_label"] for row in rows])
    parity = np.array([pair_index(row) % 2 for row in rows])
    margin = np.array([row["baseline_semantic_margin"] for row in rows])
    projection = np.array([row["baseline_projection"] for row in rows])
    slope = np.array([row["responsiveness_slope"] for row in rows])

    parity_determinism()
    empty_band(margin, labels)
    direction_reads_parity(projection, labels, parity)
    between_cell_variance(slope, margin, 2 * labels + parity)
    saturated_fit(margin, labels, parity)
    ablation_is_a_pinned_dose(rows, projection, boundary["projection_sigma"])
    operating_point(rows, margin, labels)


if __name__ == "__main__":
    main()
