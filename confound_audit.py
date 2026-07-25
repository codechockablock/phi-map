"""Domain-independent confound controls, extracted from Arm G.

Nothing here imports Arm G, mentions catalogs, or assumes a language model.
These are the four things that actually did work in the Arm G audit, in the
order they mattered:

1. `nesting_report` -- is the factor you care about *nested* inside the unit of
   analysis, or *crossed* with it? Arm G's generator balanced catalog order
   marginally across scenarios, which its validator checked and passed, while
   every scenario appeared at exactly one order. The factor was nested. Its
   contrast therefore absorbed every other property that varied with the unit,
   and 100% of the behavioural outcome turned out to be that absorbed structure.
   Marginal balance is not crossing. This function is the check that was missing.

2. `assert_crossover` -- once you render each item at both levels, verify the two
   renderings differ *only* in the manipulated feature. Without this you have a
   re-randomization, where the factor still varies between items that differ in
   other ways, and you have not identified anything.

3. `contamination` -- does this readout separate a known nuisance better than it
   separates the label? Arm G's direction scored AUROC 1.0000 on catalog order
   within one condition against 0.9431 on the label. One number, and it is the
   number that broke the project open.

4. `variance_decomposition` -- how much of the variance in your outcome is
   between cells rather than within them? Arm G reported a correlation of -0.944
   with R^2 0.892 over 128 rows. 93% of the variance was between four cells, the
   within-cell R^2 was 0.072, and the F test used ~124 degrees of freedom that
   did not exist.

Plus `auroc` and `best_threshold_accuracy`, because a fixed-threshold rate cannot
distinguish an operating-point shift from a change in what a score discriminates,
and reporting only the former is how Arm G mistook a censored instrument for a
null result.

Run `python3 confound_audit.py` to self-test. No GPU, no network, no model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Probability a random positive outranks a random negative, ties at 0.5."""
    values = np.asarray(scores, dtype=float)
    targets = np.asarray(labels)
    positive, negative = values[targets == 1], values[targets == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    wins = sum(
        1.0 if a > b else 0.5 if a == b else 0.0 for a in positive for b in negative
    )
    return float(wins / (len(positive) * len(negative)))


def best_threshold_accuracy(
    scores: Sequence[float], labels: Sequence[int]
) -> float:
    """Accuracy at the best possible threshold.

    Report this next to accuracy at your actual threshold. If an intervention
    raises the second while lowering the first, it moved the operating point and
    destroyed information; it did not help.
    """
    values = np.asarray(scores, dtype=float)
    targets = np.asarray(labels)
    cuts = np.unique(np.concatenate([values - 1e-9, values + 1e-9]))
    return float(max(((values > cut) == (targets == 1)).mean() for cut in cuts))


def nesting_report(
    items: Iterable[Mapping[str, Any]],
    unit: Callable[[Mapping[str, Any]], Hashable],
    factor: Callable[[Mapping[str, Any]], Hashable],
) -> dict[str, Any]:
    """Is `factor` nested inside `unit`, or crossed with it?

    Nested means every unit appears at exactly one factor level, so the factor
    contrast is a between-unit comparison and silently absorbs everything else
    that varies between units. Marginal balance across units does not fix this
    and does not detect it.
    """
    levels_by_unit: dict[Hashable, set[Hashable]] = defaultdict(set)
    marginal: dict[Hashable, int] = defaultdict(int)
    for item in items:
        levels_by_unit[unit(item)].add(factor(item))
        marginal[factor(item)] += 1
    if not levels_by_unit:
        raise ValueError("no items")
    counts = {len(levels) for levels in levels_by_unit.values()}
    all_levels = sorted({level for levels in levels_by_unit.values() for level in levels}, key=repr)
    fully_crossed = counts == {len(all_levels)} and len(all_levels) > 1
    return {
        "n_units": len(levels_by_unit),
        "levels": all_levels,
        "levels_per_unit": sorted(counts),
        "nested": counts == {1},
        "fully_crossed": fully_crossed,
        "marginally_balanced": len(set(marginal.values())) == 1,
        "verdict": (
            "crossed"
            if fully_crossed
            else "NESTED -- the factor contrast is between-unit and confounded "
            "with every other unit-level property"
            if counts == {1}
            else "partially crossed"
        ),
    }


def assert_crossover(
    items: Iterable[Mapping[str, Any]],
    unit: Callable[[Mapping[str, Any]], Hashable],
    factor: Callable[[Mapping[str, Any]], Hashable],
    payload: Callable[[Mapping[str, Any]], Any],
    canonicalize: Callable[[Any], Any] = lambda value: value,
) -> dict[str, Any]:
    """Verify the renderings of a unit differ ONLY in the manipulated feature.

    `canonicalize` should erase exactly the manipulated difference and nothing
    else -- for a reordering, sort the reordered elements. If two renderings
    disagree after canonicalization, something other than the factor changed and
    the design is a re-randomization, not a crossover.
    """
    by_unit: dict[Hashable, dict[Hashable, Any]] = defaultdict(dict)
    for item in items:
        key, level = unit(item), factor(item)
        if level in by_unit[key]:
            raise AssertionError(f"duplicate rendering for unit {key!r} level {level!r}")
        by_unit[key][level] = payload(item)
    if not by_unit:
        raise ValueError("no items")
    levels = sorted({level for cells in by_unit.values() for level in cells}, key=repr)
    if len(levels) < 2:
        raise AssertionError(f"only one factor level present: {levels}")
    for key, cells in sorted(by_unit.items(), key=lambda pair: repr(pair[0])):
        if set(cells) != set(levels):
            raise AssertionError(f"unit {key!r} is not rendered at every level")
        # Compared by repr so any canonicalizer works, including ones that
        # return unhashable containers such as `sorted`.
        canonical = {repr(canonicalize(value)) for value in cells.values()}
        if len(canonical) != 1:
            raise AssertionError(
                f"unit {key!r} differs by more than the manipulated feature"
            )
        if len({repr(value) for value in cells.values()}) != len(levels):
            raise AssertionError(f"unit {key!r} renderings are identical -- factor did nothing")
    return {"n_units": len(by_unit), "levels": levels, "crossover": True}


def contamination(
    scores: Sequence[float],
    labels: Sequence[int],
    nuisance: Sequence[int],
) -> dict[str, Any]:
    """Does this readout separate a known nuisance better than the label?

    The nuisance AUROC is computed *within* each label class, because a nuisance
    that correlates with the label would otherwise be credited to the label.
    A readout that reads the nuisance better than the label is not measuring
    what its name says.
    """
    values = np.asarray(scores, dtype=float)
    targets = np.asarray(labels)
    confound = np.asarray(nuisance)
    within = {}
    for value in np.unique(targets):
        mask = targets == value
        within[int(value)] = auroc(values[mask], confound[mask])
    worst = max(abs(score - 0.5) for score in within.values())
    label_auroc = auroc(values, targets)
    return {
        "label_auroc": label_auroc,
        "nuisance_auroc_within_label": within,
        "worst_nuisance_deviation": worst,
        # Both compared as distance from chance, so a readout that
        # anti-predicts counts as reading the thing.
        "reads_nuisance_better_than_label": bool(worst > abs(label_auroc - 0.5)),
    }


def variance_decomposition(
    values: Sequence[float],
    cells: Sequence[Hashable],
) -> dict[str, Any]:
    """How much of the variance is between cells rather than within them?

    A high between-cell share means your effective sample size is the number of
    cells, not the number of rows, and any test using row-level degrees of
    freedom is overstating its evidence by that factor.
    """
    array = np.asarray(values, dtype=float)
    keys = np.asarray([repr(cell) for cell in cells])
    unique = np.unique(keys)
    total = float(array.var())
    if total == 0:
        raise ValueError("outcome has no variance")
    between = float(np.var([array[keys == key].mean() for key in unique]))
    within = float(np.mean([array[keys == key].var() for key in unique]))
    return {
        "n_rows": len(array),
        "n_cells": len(unique),
        "between_cell_share": between / total,
        "within_cell_share": within / total,
        "effective_n_is_cells_not_rows": bool(between / total > 0.75),
    }


def _self_test() -> None:
    # 1. Nesting: the Arm G defect, reconstructed abstractly.
    nested = [
        {"unit": index, "order": index % 2, "y": index % 2}
        for index in range(16)
    ]
    report = nesting_report(nested, lambda r: r["unit"], lambda r: r["order"])
    assert report["nested"], report
    assert report["marginally_balanced"], "marginal balance holds and hides it"
    assert not report["fully_crossed"]

    crossed = [
        {"unit": index, "order": level} for index in range(16) for level in (0, 1)
    ]
    report = nesting_report(crossed, lambda r: r["unit"], lambda r: r["order"])
    assert report["fully_crossed"] and not report["nested"], report

    # 2. Crossover integrity: a reordering is fine, a content change is not.
    good = [
        {"unit": index, "order": level, "lines": ["A", "B"] if level == 0 else ["B", "A"]}
        for index in range(8)
        for level in (0, 1)
    ]
    assert_crossover(
        good, lambda r: r["unit"], lambda r: r["order"], lambda r: r["lines"], sorted
    )
    bad = [dict(row) for row in good]
    bad[1]["lines"] = ["B", "C"]
    try:
        assert_crossover(
            bad, lambda r: r["unit"], lambda r: r["order"], lambda r: r["lines"], sorted
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("content change must be rejected")

    # 3. Contamination: a readout that is really reading the nuisance.
    rng = np.random.default_rng(0)
    labels = np.array([0] * 64 + [1] * 64)
    nuisance = np.array(([0] * 32 + [1] * 32) * 2)
    honest = labels + rng.normal(scale=0.1, size=128)
    sneaky = nuisance * 3.0 + labels * 0.5 + rng.normal(scale=0.1, size=128)
    clean = contamination(honest, labels, nuisance)
    dirty = contamination(sneaky, labels, nuisance)
    assert clean["worst_nuisance_deviation"] < 0.15, clean
    assert dirty["worst_nuisance_deviation"] > 0.45, dirty
    assert dirty["reads_nuisance_better_than_label"]
    assert not clean["reads_nuisance_better_than_label"]

    # 4. Variance decomposition: four cluster means posing as 128 rows.
    clustered = np.concatenate([
        rng.normal(loc=loc, scale=0.2, size=32) for loc in (-3.0, -1.0, 1.0, 3.0)
    ])
    cell_ids = sum([[index] * 32 for index in range(4)], [])
    decomposition = variance_decomposition(clustered, cell_ids)
    assert decomposition["between_cell_share"] > 0.9, decomposition
    assert decomposition["effective_n_is_cells_not_rows"]

    # 5. Operating point versus discrimination.
    scores = np.concatenate([rng.normal(-1, 1, 200), rng.normal(1, 1, 200)])
    y = np.array([0] * 200 + [1] * 200)
    shifted = scores + 5.0
    assert abs(auroc(scores, y) - auroc(shifted, y)) < 1e-12, "a shift cannot change AUROC"
    assert ((shifted > 0) == (y == 1)).mean() < best_threshold_accuracy(scores, y), (
        "a shift can wreck fixed-threshold accuracy while AUROC is untouched"
    )

    print("confound_audit self-test: PASS")


if __name__ == "__main__":
    _self_test()
