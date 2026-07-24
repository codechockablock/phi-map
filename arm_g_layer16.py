"""Arm G layer-16 resolution.

The seed-106 cross-layer run relocated the causal mass from layer 27 to layer
16 and the audit closed the selection and absolute-perturbation-size
challenges.  Four questions remain before the layer-16 object is worth building
a multi-turn experiment on, and one run answers all of them on fresh seed-107
prompts.

Selectivity is the preregistered primary.  The equal-norm random control
already rules out generic damage from removing four dimensions, but not the
possibility that *any* task-structured direction at layer 16 has leverage on
the DECLINE-READ margin.  Two structured null subspaces are therefore built by
the identical paired construction and ablated identically:

* the visible catalog control tag, KITE versus MOSS, which the phase-1 positive
  control showed the model represents essentially perfectly and which has no
  bearing on whether the requested target is in scope; and
* family identity, data_checksums versus incident_times.

If removing a strongly represented, visible, decision-irrelevant feature moves
the condition contrast as much as removing the conflict subspace does, then the
layer-16 result is about layer 16 rather than about goal-constraint conflict.

The run also sweeps depth 13-19 to test whether 16 is a peak or a shoulder,
compares rank 1 against rank 4 at layer 16 because roughly 78% of the seed-106
displacement was rank-1, and records final-position residual-stream norms so
that relative rather than merely absolute perturbation size can be evaluated.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from arm_g_causal import (
    ACTION_TOKENS,
    MAPPING_VARIANTS,
    build_eval_rows,
    semantic_choice_summary,
)
from arm_g_causal_dose_ablation import (
    per_pair_condition_contrasts,
    stratified_pair_bootstrap_values,
)
from arm_g_causal_subspace import (
    EXPECTED_MODEL_COMMIT,
    SOURCE_EVIDENCE,
    attenuation_slices,
    build_target_basis,
    contrast_slices,
    mapping_center_subspace,
    paired_differences,
    random_orthogonal_subspace,
)
from arm_g_cross_layer import hook_order_test, run_forward
from arm_g_phase1 import (
    DEV_FAMILY,
    PRIMARY_POSITION,
    TARGET_MODEL,
    atomic_json,
    load_acting_model,
    package_version,
    require_gpu,
    verify_or_write,
)
from arm_g_scenarios import FAMILY_SPECS, build_manifest, validate_manifest

DEFAULT_OUTPUT_DIR = Path("/content/arm-g-layer16-seed107-v1")
SOURCE_SEEDS = (101, 102)
RANK1_SEED = 102
EVAL_SEED = 107
DEPTH_LAYERS = (13, 14, 15, 16, 17, 18, 19)
FOCUS_LAYER = 16
RANKS = (1, 2, 4)
PRIMARY_RANK = 4
NULL_CONTRASTS = ("control_tag", "family")
PRIOR_EVIDENCE = {
    "seed106_layer16_rank4_attenuation": 1.12890625,
    "seed106_layer16_attenuation_fraction": 0.2326,
    "seed106_layer16_rank1_share_of_displacement": 0.776,
    "seed106_random_p95_at_layer16": 0.0234,
    "source": "results/arm_g_cross_layer_seed106_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairs-per-family", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-subspaces", type=int, default=8)
    parser.add_argument("--depth-random-subspaces", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token. Prefer the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="Infrastructure smoke only; the result is marked non-load-bearing.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic mathematical tests without loading a model.",
    )
    return parser.parse_args()


def all_family_source_rows(
    pairs_per_family: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Source rows for every family, so null contrasts can be built too."""
    manifest = build_manifest(
        pairs_per_family=pairs_per_family,
        repeats=1,
        seed=seed,
    )
    audit = validate_manifest(
        manifest,
        pairs_per_family=pairs_per_family,
        repeats=1,
    )
    rows = [
        {
            "source_seed": seed,
            "pair_id": str(scenario["pair_id"]),
            "family": str(scenario["family"]),
            "condition_label": int(scenario["condition_label"]),
            "control_label": int(scenario["control_label"]),
            "messages": scenario["messages"],
        }
        for scenario in manifest
    ]
    expected = pairs_per_family * 2 * len(FAMILY_SPECS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} source rows, found {len(rows)}")
    return rows, audit


def grouped_differences(
    states: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    label_key: str,
    selector: Any = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Differences between two groups that are not pairwise matched.

    Used for the structured null contrasts.  Within each source seed the two
    groups are ordered deterministically and differenced elementwise, which
    yields a mean direction equal to the group-mean difference while giving the
    SVD enough vectors to fill the frozen rank.
    """
    by_seed: dict[int, dict[int, list[tuple[str, np.ndarray]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for state, row in zip(states, rows, strict=True):
        if selector is not None and not selector(row):
            continue
        by_seed[int(row["source_seed"])][int(row[label_key])].append(
            (str(row["pair_id"]) + str(row["condition_label"]), state)
        )
    differences = []
    metadata = []
    for seed in sorted(by_seed):
        groups = by_seed[seed]
        if set(groups) != {0, 1}:
            raise RuntimeError(f"null contrast {label_key} is not binary at {seed}")
        low = [state for _key, state in sorted(groups[0], key=lambda x: x[0])]
        high = [state for _key, state in sorted(groups[1], key=lambda x: x[0])]
        count = min(len(low), len(high))
        if count < 4:
            raise RuntimeError(f"null contrast {label_key} has too few rows")
        for index in range(count):
            differences.append(high[index] - low[index])
            metadata.append(
                {
                    "source_seed": seed,
                    "pair_id": f"{label_key}:{index:03d}",
                }
            )
    return np.stack(differences).astype(np.float64), metadata


def orthogonalized_subspace(
    basis: np.ndarray,
    excluded: np.ndarray,
) -> np.ndarray:
    """Re-orthonormalize `basis` after projecting out `excluded`.

    A structured null that happens to overlap the conflict subspace would
    attenuate the condition contrast for an uninteresting reason.  This yields
    the component of the null subspace that is provably disjoint from the
    conflict subspace, so the two explanations can be separated.
    """
    residual = basis - excluded @ (excluded.T @ basis)
    q, r = np.linalg.qr(residual, mode="reduced")
    if np.min(np.abs(np.diag(r))) < 1e-6:
        raise RuntimeError("null subspace collapses when conflict is projected out")
    if np.max(np.abs(excluded.T @ q)) > 1e-8:
        raise RuntimeError("orthogonalized null still overlaps the conflict subspace")
    return q


def principal_angle_cosines(left: np.ndarray, right: np.ndarray) -> list[float]:
    return np.linalg.svd(left.T @ right, compute_uv=False).tolist()


def family_selector(row: Mapping[str, Any]) -> bool:
    return row["family"] != DEV_FAMILY


def family_label(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        row["family_label"] = 0 if row["family"] == "data_checksums" else 1


def ablate_and_score(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    layer: int,
    basis: np.ndarray,
    states: np.ndarray,
    batch_size: int,
    token_ids: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, list[float]], np.ndarray]:
    row_centers, mapping_centers = mapping_center_subspace(states, rows, basis)
    margins, _ = run_forward(
        model,
        tokenizer,
        rows,
        batch_size,
        {layer: (basis, row_centers)},
        [],
        token_ids,
    )
    displacement = np.linalg.norm((states @ basis) - row_centers, axis=1)
    return margins, mapping_centers, displacement


def summarize(
    margins: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    baseline_pairs: Mapping[tuple[str, str], float],
    baseline_slices: Mapping[str, Any],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    pairs = per_pair_condition_contrasts(margins, rows)
    values = {key: baseline_pairs[key] - pairs[key] for key in baseline_pairs}
    slices = contrast_slices(margins, rows)
    attenuation = attenuation_slices(baseline_slices, slices)
    return {
        "condition_contrast": slices,
        "attenuation": attenuation,
        "attenuation_fraction": float(
            attenuation["overall"] / baseline_slices["overall"]
        ),
        "attenuation_bootstrap": stratified_pair_bootstrap_values(
            values, bootstrap, seed
        ),
        "attenuation_values": values,
    }


def selectivity_decision(
    contrasts: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str]]:
    passed = []
    reasons = []
    for name in NULL_CONTRASTS:
        interval = contrasts[name]["paired_bootstrap"]["ci_95"]
        if interval[0] > 0:
            passed.append(name)
        else:
            reasons.append(f"conflict-minus-{name} attenuation interval includes zero")
    if len(passed) == len(NULL_CONTRASTS):
        return "SELECTIVE_CONFLICT_SUBSPACE", []
    if passed:
        return "PARTIALLY_SELECTIVE_CONFLICT_SUBSPACE", reasons
    return "NONSELECTIVE_STRUCTURED_LEVERAGE", reasons


def self_test() -> None:
    hooks = hook_order_test()
    rng = np.random.default_rng(EVAL_SEED)
    width = 24

    rows = []
    states = []
    for seed in SOURCE_SEEDS:
        for family in FAMILY_SPECS:
            for pair_index in range(8):
                for label in (0, 1):
                    rows.append(
                        {
                            "source_seed": seed,
                            "pair_id": f"{family}:{pair_index:03d}",
                            "family": family,
                            "condition_label": label,
                            "control_label": pair_index % 2,
                        }
                    )
                    states.append(rng.normal(size=width))
    states = np.stack(states)
    family_label(rows)

    dev_rows = [r for r in rows if r["family"] == DEV_FAMILY]
    dev_states = states[[i for i, r in enumerate(rows) if r["family"] == DEV_FAMILY]]
    conflict_diffs, conflict_meta = paired_differences(dev_states, dev_rows)
    if len(conflict_diffs) != len(SOURCE_SEEDS) * 8:
        raise AssertionError("conflict differences have the wrong count")

    tag_diffs, tag_meta = grouped_differences(dev_states, dev_rows, "control_label")
    fam_diffs, fam_meta = grouped_differences(
        states, rows, "family_label", family_selector
    )
    for diffs, meta, name in (
        (conflict_diffs, conflict_meta, "conflict"),
        (tag_diffs, tag_meta, "control_tag"),
        (fam_diffs, fam_meta, "family"),
    ):
        basis, _diagnostics = build_target_basis(
            diffs, meta, rank1_seed=RANK1_SEED, max_rank=PRIMARY_RANK
        )
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(PRIMARY_RANK), atol=1e-8):
            raise AssertionError(f"{name} basis is not orthonormal")
        for rank in RANKS:
            nested = basis[:, :rank]
            if not np.allclose(nested, basis[:, :rank]):
                raise AssertionError("nested rank slicing is inconsistent")

    eval_rows = []
    for family in ("data_checksums", "incident_times"):
        for pair_index in range(8):
            for label in (0, 1):
                for mapping in MAPPING_VARIANTS:
                    eval_rows.append(
                        {
                            "family": family,
                            "pair_id": f"{family}:{pair_index:03d}",
                            "condition_label": label,
                            "mapping_variant": mapping,
                        }
                    )
    base = np.array(
        [3.0 * r["condition_label"] + rng.normal(scale=0.3) for r in eval_rows]
    )
    conflict_margins = base * 0.55
    null_margins = base * 0.97
    base_pairs = per_pair_condition_contrasts(base, eval_rows)
    conflict_pairs = per_pair_condition_contrasts(conflict_margins, eval_rows)
    null_pairs = per_pair_condition_contrasts(null_margins, eval_rows)
    gap = {
        k: (base_pairs[k] - conflict_pairs[k]) - (base_pairs[k] - null_pairs[k])
        for k in base_pairs
    }
    boot = stratified_pair_bootstrap_values(gap, 500, EVAL_SEED)
    if boot["ci_95"][0] <= 0:
        raise AssertionError("synthetic selectivity gap failed to separate")

    strong = {name: {"paired_bootstrap": boot} for name in NULL_CONTRASTS}
    decision, reasons = selectivity_decision(strong)
    if decision != "SELECTIVE_CONFLICT_SUBSPACE" or reasons:
        raise AssertionError("decision rule rejected a clean synthetic positive")
    null_boot = {"paired_bootstrap": {"ci_95": [-0.1, 0.1]}}
    mixed = {
        NULL_CONTRASTS[0]: {"paired_bootstrap": boot},
        NULL_CONTRASTS[1]: null_boot,
    }
    if selectivity_decision(mixed)[0] != "PARTIALLY_SELECTIVE_CONFLICT_SUBSPACE":
        raise AssertionError("decision rule mislabeled a partial result")
    flat = {name: null_boot for name in NULL_CONTRASTS}
    if selectivity_decision(flat)[0] != "NONSELECTIVE_STRUCTURED_LEVERAGE":
        raise AssertionError("decision rule mislabeled a null result")

    print(
        json.dumps(
            {
                "self_test": "PASS",
                "hook_order": hooks,
                "depth_layers": list(DEPTH_LAYERS),
                "focus_layer": FOCUS_LAYER,
                "ranks": list(RANKS),
                "null_contrasts": list(NULL_CONTRASTS),
                "synthetic_selectivity_ci_95": boot["ci_95"],
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if FOCUS_LAYER not in DEPTH_LAYERS:
        raise RuntimeError("the focus layer must lie inside the depth sweep")
    if args.random_subspaces < 8:
        raise RuntimeError("at least eight random controls are required at the focus")

    source_rows: list[dict[str, Any]] = []
    source_audits = {}
    source_manifests = {}
    for seed in SOURCE_SEEDS:
        rows, audit = all_family_source_rows(args.pairs_per_family, seed)
        source_rows.extend(rows)
        source_audits[str(seed)] = audit
        source_manifests[str(seed)] = build_manifest(
            pairs_per_family=args.pairs_per_family,
            repeats=1,
            seed=seed,
        )
    family_label(source_rows)

    eval_manifest = build_manifest(
        pairs_per_family=args.pairs_per_family,
        repeats=1,
        seed=EVAL_SEED,
    )
    eval_audit = validate_manifest(
        eval_manifest,
        pairs_per_family=args.pairs_per_family,
        repeats=1,
    )
    eval_rows = build_eval_rows(eval_manifest)

    config = {
        "protocol": "ARM_G_LAYER16_RESOLUTION_V1",
        "model": args.model,
        "source_seeds": list(SOURCE_SEEDS),
        "rank1_seed": RANK1_SEED,
        "eval_seed": EVAL_SEED,
        "source_family_for_conflict": DEV_FAMILY,
        "evaluation_families": sorted({row["family"] for row in eval_rows}),
        "source_evidence": SOURCE_EVIDENCE,
        "prior_evidence": PRIOR_EVIDENCE,
        "depth_layers": list(DEPTH_LAYERS),
        "focus_layer": FOCUS_LAYER,
        "ranks_at_focus": list(RANKS),
        "primary_rank": PRIMARY_RANK,
        "null_contrasts": list(NULL_CONTRASTS),
        "pairs_per_family": args.pairs_per_family,
        "read_position": PRIMARY_POSITION,
        "subspace_construction": (
            "every subspace uses the identical construction: rank 1 is the "
            "seed-102 mean difference direction and ranks 2-4 are the leading "
            "SVD directions of the seed-101/102 differences after removing "
            "rank 1. The conflict subspace uses exactly matched within-pair "
            "differences in the development family; the control-tag subspace "
            "uses KITE-versus-MOSS differences in the same family; the family "
            "subspace uses data_checksums-versus-incident_times differences"
        ),
        "primary_question": (
            "is the layer-16 conflict-subspace attenuation larger than the "
            "attenuation from removing an equally constructed, strongly "
            "represented, decision-irrelevant structured subspace"
        ),
        "success_rule": (
            "conflict-minus-control_tag and conflict-minus-family attenuation "
            "intervals both exclude zero"
        ),
        "secondary_questions": [
            "does depth 13-19 place the peak at layer 16",
            "does rank 1 at layer 16 recover the rank-4 attenuation",
            "how large is the intervention relative to residual-stream norm",
        ],
        "bootstrap": args.bootstrap,
        "random_subspaces_at_focus": args.random_subspaces,
        "random_subspaces_per_depth_layer": args.depth_random_subspaces,
        "mapping_variants": list(MAPPING_VARIANTS),
        "primary_outcome": "semantic next-action logit margin: DECLINE minus READ",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_or_write(args.output_dir / "run_config.json", config)
    verify_or_write(args.output_dir / "source_manifests.json", source_manifests)
    verify_or_write(args.output_dir / "eval_manifest.json", eval_manifest)

    hardware = require_gpu(args.allow_non_a100)
    tokenizer, model = load_acting_model(args.model, args.hf_token)
    model_commit = getattr(model.config, "_commit_hash", None)
    if model_commit not in (None, EXPECTED_MODEL_COMMIT):
        raise RuntimeError(
            f"model commit {model_commit} differs from frozen {EXPECTED_MODEL_COMMIT}"
        )
    token_ids = {}
    for token in ACTION_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"action token {token!r} is not single: {encoded}")
        token_ids[token] = int(encoded[0])

    _margins, source_states = run_forward(
        model, tokenizer, source_rows, args.batch_size, {}, DEPTH_LAYERS, None
    )
    dev_index = [i for i, r in enumerate(source_rows) if r["family"] == DEV_FAMILY]
    dev_rows = [source_rows[i] for i in dev_index]

    bases: dict[str, dict[int, np.ndarray]] = {
        "conflict": {},
        "control_tag": {},
        "family": {},
    }
    basis_diagnostics: dict[str, dict[str, Any]] = defaultdict(dict)
    for layer in DEPTH_LAYERS:
        states = source_states[layer]
        conflict_diffs, conflict_meta = paired_differences(states[dev_index], dev_rows)
        specs = [("conflict", conflict_diffs, conflict_meta)]
        if layer == FOCUS_LAYER:
            specs.append(
                (
                    "control_tag",
                    *grouped_differences(states[dev_index], dev_rows, "control_label"),
                )
            )
            specs.append(
                (
                    "family",
                    *grouped_differences(
                        states, source_rows, "family_label", family_selector
                    ),
                )
            )
        for name, diffs, meta in specs:
            basis, diagnostics = build_target_basis(
                diffs, meta, rank1_seed=RANK1_SEED, max_rank=PRIMARY_RANK
            )
            bases[name][layer] = basis
            diagnostics["paired_difference_energy_captured_by_rank"] = {
                rank: value
                for rank, value in diagnostics[
                    "paired_difference_energy_captured_by_rank"
                ].items()
                if int(rank) <= PRIMARY_RANK
            }
            diagnostics["difference_count"] = int(len(diffs))
            basis_diagnostics[name][str(layer)] = diagnostics

    baseline, evaluation_states = run_forward(
        model, tokenizer, eval_rows, args.batch_size, {}, DEPTH_LAYERS, token_ids
    )
    baseline_pairs = per_pair_condition_contrasts(baseline, eval_rows)
    baseline_slices = contrast_slices(baseline, eval_rows)
    residual_norms = {
        str(layer): {
            "mean": float(np.linalg.norm(evaluation_states[layer], axis=1).mean()),
            "std": float(np.linalg.norm(evaluation_states[layer], axis=1).std()),
        }
        for layer in DEPTH_LAYERS
    }

    score = partial(ablate_and_score, model, tokenizer)

    def run_one(layer: int, basis: np.ndarray, seed: int) -> dict[str, Any]:
        margins, centers, displacement = score(
            eval_rows,
            layer,
            basis,
            evaluation_states[layer],
            args.batch_size,
            token_ids,
        )
        entry = summarize(
            margins, eval_rows, baseline_pairs, baseline_slices, args.bootstrap, seed
        )
        entry["mapping_centers"] = centers
        entry["mean_displacement_norm"] = float(displacement.mean())
        entry["displacement_over_residual_norm"] = float(
            displacement.mean()
            / np.linalg.norm(evaluation_states[layer], axis=1).mean()
        )
        entry["choice_summary"] = semantic_choice_summary(margins, eval_rows)
        return entry

    depth = {}
    for layer in DEPTH_LAYERS:
        depth[str(layer)] = run_one(
            layer, bases["conflict"][layer][:, :PRIMARY_RANK], EVAL_SEED + layer
        )

    depth_random = {}
    for layer in DEPTH_LAYERS:
        rng = np.random.default_rng(EVAL_SEED * 1000 + layer)
        attenuations = []
        for _index in range(args.depth_random_subspaces):
            random_basis = random_orthogonal_subspace(
                rng,
                bases["conflict"][layer].shape[0],
                PRIMARY_RANK,
                bases["conflict"][layer],
            )
            entry = run_one(layer, random_basis, EVAL_SEED)
            attenuations.append(entry["attenuation"]["overall"])
        depth_random[str(layer)] = {
            "attenuations": attenuations,
            "absolute_p95": float(np.quantile(np.abs(attenuations), 0.95)),
        }

    ranks = {}
    for rank in RANKS:
        basis = bases["conflict"][FOCUS_LAYER][:, :rank]
        ranks[str(rank)] = run_one(FOCUS_LAYER, basis, EVAL_SEED + 200 + rank)

    conflict_basis = bases["conflict"][FOCUS_LAYER][:, :PRIMARY_RANK]
    nulls = {}
    nulls_orthogonalized = {}
    for name in NULL_CONTRASTS:
        basis = bases[name][FOCUS_LAYER][:, :PRIMARY_RANK]
        nulls[name] = run_one(FOCUS_LAYER, basis, EVAL_SEED + 300)
        nulls[name]["principal_angle_cosines_with_conflict"] = principal_angle_cosines(
            basis, conflict_basis
        )
        disjoint = orthogonalized_subspace(basis, conflict_basis)
        nulls_orthogonalized[name] = run_one(FOCUS_LAYER, disjoint, EVAL_SEED + 350)
        nulls_orthogonalized[name]["max_abs_overlap_with_conflict"] = float(
            np.max(np.abs(conflict_basis.T @ disjoint))
        )

    rng = np.random.default_rng(EVAL_SEED * 7)
    focus_random = []
    for _index in range(args.random_subspaces):
        random_basis = random_orthogonal_subspace(
            rng,
            bases["conflict"][FOCUS_LAYER].shape[0],
            PRIMARY_RANK,
            bases["conflict"][FOCUS_LAYER],
        )
        focus_random.append(
            run_one(FOCUS_LAYER, random_basis, EVAL_SEED)["attenuation"]["overall"]
        )

    conflict_values = ranks[str(PRIMARY_RANK)]["attenuation_values"]
    selectivity = {}
    selectivity_orthogonalized = {}
    for target, block in (
        (selectivity, nulls),
        (selectivity_orthogonalized, nulls_orthogonalized),
    ):
        for name in NULL_CONTRASTS:
            gap = {
                key: conflict_values[key] - block[name]["attenuation_values"][key]
                for key in conflict_values
            }
            target[name] = {
                "mean": float(
                    ranks[str(PRIMARY_RANK)]["attenuation"]["overall"]
                    - block[name]["attenuation"]["overall"]
                ),
                "paired_bootstrap": stratified_pair_bootstrap_values(
                    gap, args.bootstrap, EVAL_SEED + 400
                ),
            }

    rank_gap = {
        key: conflict_values[key] - ranks["1"]["attenuation_values"][key]
        for key in conflict_values
    }
    rank_increment = {
        "mean": float(
            ranks[str(PRIMARY_RANK)]["attenuation"]["overall"]
            - ranks["1"]["attenuation"]["overall"]
        ),
        "paired_bootstrap": stratified_pair_bootstrap_values(
            rank_gap, args.bootstrap, EVAL_SEED + 500
        ),
        "rank1_recovers_fraction_of_rank4": float(
            ranks["1"]["attenuation"]["overall"]
            / ranks[str(PRIMARY_RANK)]["attenuation"]["overall"]
        ),
    }

    peak_layer = max(
        DEPTH_LAYERS, key=lambda x: depth[str(x)]["attenuation"]["overall"]
    )
    decision, decision_reasons = selectivity_decision(selectivity)

    def public(entry: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in entry.items() if k != "attenuation_values"}

    row_results = []
    for index, row in enumerate(eval_rows):
        item = {k: v for k, v in row.items() if k != "messages"}
        item.update(
            {
                "baseline_semantic_margin": float(baseline[index]),
                "residual_norm_by_layer": {
                    str(layer): float(np.linalg.norm(evaluation_states[layer][index]))
                    for layer in DEPTH_LAYERS
                },
                "conflict_projections_at_focus": (
                    evaluation_states[FOCUS_LAYER][index]
                    @ bases["conflict"][FOCUS_LAYER]
                ).tolist(),
            }
        )
        row_results.append(item)

    report = {
        "status": "ARM_G_LAYER16_RESOLUTION_V1",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "baseline": {
            "choice_summary": semantic_choice_summary(baseline, eval_rows),
            "condition_contrast": baseline_slices,
        },
        "selectivity": {
            "conflict_attenuation": ranks[str(PRIMARY_RANK)]["attenuation"]["overall"],
            "null_contrasts": {name: public(nulls[name]) for name in NULL_CONTRASTS},
            "conflict_minus_null": selectivity,
            "null_contrasts_orthogonalized": {
                name: public(nulls_orthogonalized[name]) for name in NULL_CONTRASTS
            },
            "conflict_minus_null_orthogonalized": selectivity_orthogonalized,
            "orthogonalized_note": (
                "pre-specified disambiguator: the component of each structured "
                "null that is provably disjoint from the conflict subspace, so "
                "that a large raw null effect can be attributed to overlap or "
                "to genuine structured leverage"
            ),
            "random_subspace_attenuations": focus_random,
            "random_absolute_p95": float(np.quantile(np.abs(focus_random), 0.95)),
        },
        "depth_sweep": {
            "layers": {str(layer): public(depth[str(layer)]) for layer in DEPTH_LAYERS},
            "random_controls": depth_random,
            "peak_layer": peak_layer,
            "focus_layer_is_peak": bool(peak_layer == FOCUS_LAYER),
        },
        "rank_sweep": {
            "ranks": {str(rank): public(ranks[str(rank)]) for rank in RANKS},
            "rank4_minus_rank1": rank_increment,
        },
        "residual_norms": residual_norms,
        "seed_replication": {
            "seed106_layer16_rank4_attenuation": PRIOR_EVIDENCE[
                "seed106_layer16_rank4_attenuation"
            ],
            "seed107_layer16_rank4_attenuation": ranks[str(PRIMARY_RANK)][
                "attenuation"
            ]["overall"],
        },
        "source_subspace_diagnostics": dict(basis_diagnostics),
        "sample_counts": {
            "source_rows": len(source_rows),
            "development_source_rows": len(dev_rows),
            "evaluation_rows": len(eval_rows),
            "independent_evaluation_pairs": int(
                len({row["pair_id"] for row in eval_rows})
            ),
            "depth_layers": len(DEPTH_LAYERS),
        },
        "audits": {
            "source_manifests": source_audits,
            "evaluation_manifest": eval_audit,
        },
        "row_results": row_results,
        "provenance": {
            "acting_model": args.model,
            "expected_model_commit": EXPECTED_MODEL_COMMIT,
            "model_commit": model_commit,
            "tokenizer_commit": getattr(tokenizer, "_commit_hash", None),
            "dtype": "bfloat16",
            "quantization": None,
            "hardware": hardware,
            "source_evidence": SOURCE_EVIDENCE,
            "prior_evidence": PRIOR_EVIDENCE,
            "label_source": (
                "mechanical target-path achievability; the null contrasts are "
                "mechanical scenario properties, not judged labels"
            ),
            "intervention_scope": (
                "mapping-centered rank-limited projection removal at the final "
                "prompt position, one layer at a time"
            ),
            "package_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "scikit-learn": package_version("scikit-learn"),
                "numpy": package_version("numpy"),
            },
        },
    }
    result_path = args.output_dir / "arm_g_layer16_result.json"
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
                "baseline_condition_contrast": baseline_slices["overall"],
                "selectivity": {
                    "conflict_attenuation": report["selectivity"][
                        "conflict_attenuation"
                    ],
                    "null_attenuation": {
                        name: nulls[name]["attenuation"]["overall"]
                        for name in NULL_CONTRASTS
                    },
                    "conflict_minus_null": selectivity,
                    "null_attenuation_orthogonalized": {
                        name: nulls_orthogonalized[name]["attenuation"]["overall"]
                        for name in NULL_CONTRASTS
                    },
                    "conflict_minus_null_orthogonalized": selectivity_orthogonalized,
                    "principal_angle_cosines_with_conflict": {
                        name: nulls[name]["principal_angle_cosines_with_conflict"]
                        for name in NULL_CONTRASTS
                    },
                    "random_absolute_p95": report["selectivity"]["random_absolute_p95"],
                },
                "depth_attenuation": {
                    str(layer): depth[str(layer)]["attenuation"]["overall"]
                    for layer in DEPTH_LAYERS
                },
                "depth_random_p95": {
                    str(layer): depth_random[str(layer)]["absolute_p95"]
                    for layer in DEPTH_LAYERS
                },
                "peak_layer": peak_layer,
                "rank_attenuation": {
                    str(rank): ranks[str(rank)]["attenuation"]["overall"]
                    for rank in RANKS
                },
                "rank4_minus_rank1": rank_increment,
                "displacement_over_residual_norm": {
                    str(layer): depth[str(layer)]["displacement_over_residual_norm"]
                    for layer in DEPTH_LAYERS
                },
                "seed_replication": report["seed_replication"],
            },
            indent=2,
        )
    )
    print(f"full artifact: {result_path}")


if __name__ == "__main__":
    main()
