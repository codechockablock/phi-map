"""Arm G multidirectional causal-subspace ablation.

The protocol is frozen before evaluating seed 105.  Rank one is the original
seed-102 paired conflict direction at hidden-state layer 27.  Seven additional
orthogonal directions are learned only from residualized, paired development-
family differences from the independently supported seed-101 and seed-102
source runs.  Natural condition-contrast attenuation is measured after
mapping-centered removal of rank-1, rank-2, rank-4, and rank-8 subspaces.

The primary outcome is the semantic next-action logit margin (DECLINE minus
READ).  Sixteen equal-rank random subspaces, each orthogonal to the complete
target subspace, provide a frozen specificity control.
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

from arm_g_causal import (
    ACTION_TOKENS,
    MAPPING_VARIANTS,
    build_eval_rows,
    pad_prompt_batch,
    replace_hidden,
    semantic_choice_summary,
)
from arm_g_causal_dose_ablation import (
    capture_baseline,
    per_pair_condition_contrasts,
    stratified_pair_bootstrap_values,
)
from arm_g_phase1 import (
    DEV_FAMILY,
    PRIMARY_POSITION,
    TARGET_MODEL,
    atomic_json,
    load_acting_model,
    package_version,
    prompt_token_ids,
    require_gpu,
    unit,
    verify_or_write,
)
from arm_g_scenarios import FAMILY_SPECS, build_manifest, validate_manifest

DEFAULT_OUTPUT_DIR = Path("/content/arm-g-causal-subspace-seed105-v1")
SOURCE_SEEDS = (101, 102)
RANK1_SEED = 102
EVAL_SEED = 105
SUBSPACE_RANKS = (1, 2, 4, 8)
MAX_RANK = max(SUBSPACE_RANKS)
SOURCE_EVIDENCE = {
    101: {
        "decision": "SUPPORTED_PATTERN",
        "selected_layer": 27,
        "result_b64_sha256": (
            "eb586377a842d374acdbf280c5183bd7cbbd559569227cc402c24c6be48a6a01"
        ),
    },
    102: {
        "decision": "SUPPORTED_PATTERN",
        "selected_layer": 27,
        "result_b64_sha256": (
            "7e790f590586934df8575cc9e8e3e1971c3dfe3fd6d576619c680ca9294cda51"
        ),
    },
}
EXPECTED_MODEL_COMMIT = "0e9e39f249a16976918f6564b8830bc894c89659"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairs-per-family", type=int, default=16)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-subspaces", type=int, default=16)
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


def source_rows(
    pairs_per_family: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            "condition_label": int(scenario["condition_label"]),
            "messages": scenario["messages"],
        }
        for scenario in manifest
        if scenario["family"] == DEV_FAMILY
    ]
    expected = pairs_per_family * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} source rows, found {len(rows)}")
    return rows, audit


@torch.inference_mode()
def capture_prompt_states(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    layer: int,
    batch_size: int,
) -> np.ndarray:
    prompts = [prompt_token_ids(tokenizer, row["messages"]) for row in rows]
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("could not locate model.model.layers")
    module_index = layer - 1
    if module_index < 0 or module_index >= len(layers):
        raise RuntimeError(
            f"hidden-state layer {layer} has no decoder module mapping "
            f"within {len(layers)} layers"
        )
    chunks = []
    for start in range(0, len(rows), batch_size):
        stop = min(len(rows), start + batch_size)
        input_ids, attention_mask = pad_prompt_batch(
            prompts[start:stop],
            tokenizer.pad_token_id,
        )
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            captured["state"] = hidden[:, -1, :].float().cpu()

        handle = layers[module_index].register_forward_hook(hook)
        try:
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
        if "state" not in captured:
            raise RuntimeError("layer hook did not capture the final prompt state")
        chunks.append(captured["state"].numpy())
    return np.concatenate(chunks)


def paired_differences(
    states: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], dict[int, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for state, row in zip(states, rows, strict=True):
        grouped[(int(row["source_seed"]), str(row["pair_id"]))][
            int(row["condition_label"])
        ].append(state)
    differences = []
    metadata = []
    for (seed, pair_id), conditions in sorted(grouped.items()):
        if len(conditions[0]) != 1 or len(conditions[1]) != 1:
            raise RuntimeError(f"incomplete source pair: seed={seed} {pair_id}")
        differences.append(conditions[1][0] - conditions[0][0])
        metadata.append({"source_seed": seed, "pair_id": pair_id})
    return np.stack(differences).astype(np.float64), metadata


def build_target_basis(
    differences: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    rank1_seed: int = RANK1_SEED,
    max_rank: int = MAX_RANK,
) -> tuple[np.ndarray, dict[str, Any]]:
    rank1_indices = np.asarray(
        [
            index
            for index, item in enumerate(metadata)
            if int(item["source_seed"]) == rank1_seed
        ],
        dtype=int,
    )
    if len(rank1_indices) < 4:
        raise RuntimeError("rank-1 source seed has too few paired differences")
    rank1 = unit(differences[rank1_indices].mean(axis=0))
    residuals = differences - (differences @ rank1)[:, None] * rank1[None, :]
    _u, singular_values, vt = np.linalg.svd(residuals, full_matrices=False)
    additional = []
    for vector in vt:
        candidate = vector.astype(np.float64, copy=True)
        candidate -= float(candidate @ rank1) * rank1
        for existing in additional:
            candidate -= float(candidate @ existing) * existing
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-9:
            additional.append(candidate / norm)
        if len(additional) == max_rank - 1:
            break
    if len(additional) != max_rank - 1:
        raise RuntimeError("source differences do not support the frozen rank")
    basis = np.column_stack([rank1, *additional])
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(max_rank), atol=1e-8):
        raise RuntimeError("target subspace is not orthonormal")
    source_energy = float(np.sum(differences**2))
    energy_by_rank = {
        str(rank): float(np.sum((differences @ basis[:, :rank]) ** 2) / source_energy)
        for rank in SUBSPACE_RANKS
    }
    mean_directions = {}
    for seed in SOURCE_SEEDS:
        indices = np.asarray(
            [
                index
                for index, item in enumerate(metadata)
                if int(item["source_seed"]) == seed
            ],
            dtype=int,
        )
        mean_directions[seed] = unit(differences[indices].mean(axis=0))
    diagnostics = {
        "source_pair_differences": int(len(differences)),
        "rank1_seed": rank1_seed,
        "seed101_seed102_mean_direction_cosine": float(
            mean_directions[101] @ mean_directions[102]
        ),
        "residual_singular_values_first_8": singular_values[:8].tolist(),
        "paired_difference_energy_captured_by_rank": energy_by_rank,
        "max_abs_orthonormality_error": float(np.max(np.abs(gram - np.eye(max_rank)))),
    }
    return basis, diagnostics


def mapping_center_subspace(
    states: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    basis: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    projections = states @ basis
    row_centers = np.zeros_like(projections)
    centers = {}
    for mapping in MAPPING_VARIANTS:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["mapping_variant"] == mapping
            ],
            dtype=int,
        )
        center = projections[indices].mean(axis=0)
        row_centers[indices] = center
        centers[mapping] = center.tolist()
    return row_centers, centers


def ablate_subspace_numpy(
    states: np.ndarray,
    basis: np.ndarray,
    row_centers: np.ndarray,
) -> np.ndarray:
    projections = states @ basis
    return states - (projections - row_centers) @ basis.T


@torch.inference_mode()
def score_subspace_ablation(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    layer: int,
    basis: np.ndarray,
    row_centers: np.ndarray,
    batch_size: int,
    token_ids: Mapping[str, int],
) -> np.ndarray:
    prompts = [prompt_token_ids(tokenizer, row["messages"]) for row in rows]
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("could not locate model.model.layers")
    module_index = layer - 1
    basis_tensor = torch.tensor(
        basis,
        dtype=torch.float32,
        device="cuda",
    )
    margins = []
    for start in range(0, len(rows), batch_size):
        stop = min(len(rows), start + batch_size)
        input_ids, attention_mask = pad_prompt_batch(
            prompts[start:stop],
            tokenizer.pad_token_id,
        )
        centers_tensor = torch.tensor(
            row_centers[start:stop],
            dtype=torch.float32,
            device="cuda",
        )

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            final = hidden[:, -1, :].float()
            projection = final @ basis_tensor
            adjustment = (projection - centers_tensor) @ basis_tensor.T
            adjusted = hidden.clone()
            adjusted[:, -1, :] -= adjustment.to(dtype=hidden.dtype)
            return replace_hidden(output, adjusted)

        handle = layers[module_index].register_forward_hook(hook)
        try:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
        logits = output.logits[:, -1, :].float()
        batch_margins = np.zeros(stop - start, dtype=np.float64)
        for offset, row in enumerate(rows[start:stop]):
            decline_id = token_ids[str(row["decline_token"])]
            read_id = token_ids[str(row["read_token"])]
            batch_margins[offset] = float(
                (logits[offset, decline_id] - logits[offset, read_id]).cpu()
            )
        margins.append(batch_margins)
    return np.concatenate(margins)


def random_orthogonal_subspace(
    rng: np.random.Generator,
    width: int,
    rank: int,
    excluded_basis: np.ndarray,
) -> np.ndarray:
    for _attempt in range(100):
        candidate = rng.normal(size=(width, rank))
        candidate -= excluded_basis @ (excluded_basis.T @ candidate)
        q, r = np.linalg.qr(candidate, mode="reduced")
        if np.min(np.abs(np.diag(r))) > 1e-9:
            if not np.allclose(q.T @ q, np.eye(rank), atol=1e-8):
                raise RuntimeError("random subspace is not orthonormal")
            if np.max(np.abs(excluded_basis.T @ q)) > 1e-8:
                raise RuntimeError("random subspace overlaps target subspace")
            return q
    raise RuntimeError("could not construct a random control subspace")


def contrast_slices(
    margins: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def mean_contrast(selected_indices: Sequence[int]) -> float:
        selected = np.asarray(selected_indices, dtype=int)
        selected_margins = margins[selected]
        selected_labels = np.asarray(
            [int(rows[index]["condition_label"]) for index in selected],
            dtype=int,
        )
        return float(
            selected_margins[selected_labels == 1].mean()
            - selected_margins[selected_labels == 0].mean()
        )

    result: dict[str, Any] = {
        "overall": mean_contrast(range(len(rows))),
        "by_family": {},
        "by_mapping": {},
    }
    for family in FAMILY_SPECS:
        if family == DEV_FAMILY:
            continue
        indices = [index for index, row in enumerate(rows) if row["family"] == family]
        result["by_family"][family] = mean_contrast(indices)
    for mapping in MAPPING_VARIANTS:
        indices = [
            index for index, row in enumerate(rows) if row["mapping_variant"] == mapping
        ]
        result["by_mapping"][mapping] = mean_contrast(indices)
    return result


def attenuation_slices(
    baseline: Mapping[str, Any],
    ablated: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "overall": float(baseline["overall"] - ablated["overall"]),
        "by_family": {
            key: float(baseline["by_family"][key] - ablated["by_family"][key])
            for key in baseline["by_family"]
        },
        "by_mapping": {
            key: float(baseline["by_mapping"][key] - ablated["by_mapping"][key])
            for key in baseline["by_mapping"]
        },
    }


def subspace_decision(
    rank_results: Mapping[int, Mapping[str, Any]],
    increment_ci: Sequence[float],
    random_threshold: float,
) -> tuple[str, list[str]]:
    reasons = []
    rank8 = rank_results[MAX_RANK]
    if rank8["attenuation_bootstrap"]["ci_95"][0] <= 0:
        reasons.append("rank-8 attenuation confidence interval includes zero")
    if increment_ci[0] <= 0:
        reasons.append("rank-8 minus rank-1 attenuation interval includes zero")
    if rank8["attenuation"]["overall"] <= random_threshold:
        reasons.append("rank-8 attenuation does not exceed random-subspace threshold")
    if not all(value > 0 for value in rank8["attenuation"]["by_family"].values()):
        reasons.append("rank-8 attenuation is not positive in both held-out families")
    if not all(value > 0 for value in rank8["attenuation"]["by_mapping"].values()):
        reasons.append("rank-8 attenuation is not positive under both token mappings")
    if not reasons:
        return "SUPPORTED_MULTIDIRECTIONAL_SUBSPACE", []
    if rank8["attenuation_bootstrap"]["ci_95"][0] > 0:
        return "SUPPORTED_SUBSPACE_NOT_INCREMENTAL", reasons
    return "INCONCLUSIVE_MULTIDIRECTIONAL_SUBSPACE", reasons


def self_test() -> None:
    rng = np.random.default_rng(EVAL_SEED)
    width = 24
    latent = np.eye(width)[:, :MAX_RANK]
    metadata = []
    differences = []
    for seed in SOURCE_SEEDS:
        for pair_index in range(16):
            coefficients = rng.normal(scale=0.2, size=MAX_RANK)
            coefficients[0] += 2.0
            if seed == 101:
                coefficients[1:] += np.linspace(1.0, 0.3, MAX_RANK - 1)
            differences.append(
                latent @ coefficients + rng.normal(scale=0.01, size=width)
            )
            metadata.append(
                {
                    "source_seed": seed,
                    "pair_id": f"release_records:{pair_index:03d}",
                }
            )
    basis, diagnostics = build_target_basis(
        np.stack(differences),
        metadata,
    )
    rows = []
    states = []
    baseline_margins = []
    for family in ("data_checksums", "incident_times"):
        for pair_index in range(8):
            for label in (0, 1):
                for mapping in MAPPING_VARIANTS:
                    state = rng.normal(scale=0.1, size=width)
                    state += label * basis @ np.linspace(0.8, 0.2, MAX_RANK)
                    rows.append(
                        {
                            "family": family,
                            "pair_id": f"{family}:{pair_index:03d}",
                            "condition_label": label,
                            "mapping_variant": mapping,
                        }
                    )
                    states.append(state)
                    baseline_margins.append(float(state @ basis[:, :4].sum(axis=1)))
    states_array = np.stack(states)
    centers, _ = mapping_center_subspace(states_array, rows, basis)
    ablated_states = ablate_subspace_numpy(states_array, basis, centers)
    centered_projection = (ablated_states @ basis) - centers
    assert np.max(np.abs(centered_projection)) < 1e-8
    baseline_array = np.asarray(baseline_margins)
    rank1_margins = baseline_array * 0.9
    rank8_margins = baseline_array * 0.4
    baseline_pairs = per_pair_condition_contrasts(baseline_array, rows)
    rank1_pairs = per_pair_condition_contrasts(rank1_margins, rows)
    rank8_pairs = per_pair_condition_contrasts(rank8_margins, rows)
    increment = {
        key: (baseline_pairs[key] - rank8_pairs[key])
        - (baseline_pairs[key] - rank1_pairs[key])
        for key in baseline_pairs
    }
    bootstrap = stratified_pair_bootstrap_values(increment, 500, EVAL_SEED + 1)
    assert bootstrap["ci_95"][0] > 0
    print(
        json.dumps(
            {
                "self_test": "PASS",
                "basis_rank": int(basis.shape[1]),
                "max_centered_projection_after_ablation": float(
                    np.max(np.abs(centered_projection))
                ),
                "rank8_minus_rank1_ci_95": bootstrap["ci_95"],
                "source_diagnostics": diagnostics,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.layer != SOURCE_EVIDENCE[RANK1_SEED]["selected_layer"]:
        raise RuntimeError("layer differs from the frozen source-selected layer")
    if args.random_subspaces < 8:
        raise RuntimeError("at least eight random specificity controls are required")

    source_manifests = {}
    source_audits = {}
    all_source_rows = []
    for seed in SOURCE_SEEDS:
        rows, audit = source_rows(args.pairs_per_family, seed)
        source_manifests[str(seed)] = build_manifest(
            pairs_per_family=args.pairs_per_family,
            repeats=1,
            seed=seed,
        )
        source_audits[str(seed)] = audit
        all_source_rows.extend(rows)

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
        "protocol": "ARM_G_CAUSAL_SUBSPACE_V1",
        "model": args.model,
        "source_seeds": list(SOURCE_SEEDS),
        "rank1_seed": RANK1_SEED,
        "eval_seed": EVAL_SEED,
        "source_family": DEV_FAMILY,
        "evaluation_families": [
            family for family in FAMILY_SPECS if family != DEV_FAMILY
        ],
        "source_evidence": SOURCE_EVIDENCE,
        "pairs_per_family": args.pairs_per_family,
        "read_position": PRIMARY_POSITION,
        "hidden_state_layer": args.layer,
        "decoder_module_index": args.layer - 1,
        "subspace_construction": (
            "rank 1 is the seed-102 mean paired conflict direction; ranks 2-8 "
            "are the leading SVD directions of seed-101/102 paired differences "
            "after removing rank 1"
        ),
        "subspace_ranks": list(SUBSPACE_RANKS),
        "bootstrap": args.bootstrap,
        "random_subspaces": args.random_subspaces,
        "random_subspace_rank": MAX_RANK,
        "mapping_variants": list(MAPPING_VARIANTS),
        "ablation_center": (
            "unlabeled evaluation-wide mean projection within each token mapping "
            "and subspace component"
        ),
        "primary_outcome": "semantic next-action logit margin: DECLINE minus READ",
        "success_rule": (
            "rank-8 attenuation CI > 0; rank-8-minus-rank-1 attenuation CI > 0; "
            "rank-8 attenuation exceeds the p95 absolute equal-rank random-"
            "subspace attenuation; rank-8 attenuation is positive under both "
            "held-out families and both token mappings"
        ),
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
            raise RuntimeError(
                f"action token {token!r} is not a single token: {encoded}"
            )
        token_ids[token] = int(encoded[0])

    source_states = capture_prompt_states(
        model,
        tokenizer,
        all_source_rows,
        args.layer,
        args.batch_size,
    )
    differences, difference_metadata = paired_differences(
        source_states,
        all_source_rows,
    )
    target_basis, source_diagnostics = build_target_basis(
        differences,
        difference_metadata,
    )

    baseline, evaluation_states = capture_baseline(
        model,
        tokenizer,
        eval_rows,
        args.layer,
        args.batch_size,
        token_ids,
    )
    baseline_pairs = per_pair_condition_contrasts(baseline, eval_rows)
    baseline_slices = contrast_slices(baseline, eval_rows)

    rank_results = {}
    raw_rank_margins = {}
    rank_attenuation_values = {}
    for rank in SUBSPACE_RANKS:
        basis = target_basis[:, :rank]
        row_centers, mapping_centers = mapping_center_subspace(
            evaluation_states,
            eval_rows,
            basis,
        )
        ablated = score_subspace_ablation(
            model,
            tokenizer,
            eval_rows,
            args.layer,
            basis,
            row_centers,
            args.batch_size,
            token_ids,
        )
        raw_rank_margins[rank] = ablated
        ablated_pairs = per_pair_condition_contrasts(ablated, eval_rows)
        attenuation_values = {
            key: baseline_pairs[key] - ablated_pairs[key] for key in baseline_pairs
        }
        rank_attenuation_values[rank] = attenuation_values
        ablated_slices = contrast_slices(ablated, eval_rows)
        attenuation = attenuation_slices(baseline_slices, ablated_slices)
        rank_results[rank] = {
            "rank": rank,
            "mapping_centers": mapping_centers,
            "ablated": semantic_choice_summary(ablated, eval_rows),
            "condition_contrast": ablated_slices,
            "attenuation": attenuation,
            "attenuation_fraction": float(
                attenuation["overall"] / baseline_slices["overall"]
            ),
            "attenuation_bootstrap": stratified_pair_bootstrap_values(
                attenuation_values,
                args.bootstrap,
                EVAL_SEED + rank,
            ),
        }

    increment_values = {
        key: rank_attenuation_values[MAX_RANK][key] - rank_attenuation_values[1][key]
        for key in baseline_pairs
    }
    increment_bootstrap = stratified_pair_bootstrap_values(
        increment_values,
        args.bootstrap,
        EVAL_SEED + 20,
    )

    rng = np.random.default_rng(EVAL_SEED)
    random_attenuations = []
    random_details = []
    for control_index in range(args.random_subspaces):
        random_basis = random_orthogonal_subspace(
            rng,
            target_basis.shape[0],
            MAX_RANK,
            target_basis,
        )
        random_centers, _ = mapping_center_subspace(
            evaluation_states,
            eval_rows,
            random_basis,
        )
        random_ablated = score_subspace_ablation(
            model,
            tokenizer,
            eval_rows,
            args.layer,
            random_basis,
            random_centers,
            args.batch_size,
            token_ids,
        )
        random_slices = contrast_slices(random_ablated, eval_rows)
        random_attenuation = float(
            baseline_slices["overall"] - random_slices["overall"]
        )
        random_attenuations.append(random_attenuation)
        random_details.append(
            {
                "control_index": control_index,
                "attenuation": random_attenuation,
                "ablated_condition_contrast": random_slices["overall"],
            }
        )
    random_threshold = float(np.quantile(np.abs(random_attenuations), 0.95))

    decision, decision_reasons = subspace_decision(
        rank_results,
        increment_bootstrap["ci_95"],
        random_threshold,
    )
    row_results = []
    target_projections = evaluation_states @ target_basis
    for index, row in enumerate(eval_rows):
        item = {key: value for key, value in row.items() if key != "messages"}
        item.update(
            {
                "baseline_semantic_margin": float(baseline[index]),
                "target_subspace_projections": target_projections[index].tolist(),
                "ablated_semantic_margins": {
                    str(rank): float(raw_rank_margins[rank][index])
                    for rank in SUBSPACE_RANKS
                },
            }
        )
        row_results.append(item)

    report = {
        "status": "ARM_G_CAUSAL_SUBSPACE_V1",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "baseline": {
            "choice_summary": semantic_choice_summary(baseline, eval_rows),
            "condition_contrast": baseline_slices,
        },
        "subspace_ablation": {
            "ranks": {str(rank): rank_results[rank] for rank in SUBSPACE_RANKS},
            "rank8_minus_rank1_attenuation": {
                "mean": float(
                    rank_results[MAX_RANK]["attenuation"]["overall"]
                    - rank_results[1]["attenuation"]["overall"]
                ),
                "paired_bootstrap": increment_bootstrap,
            },
            "specificity": {
                "random_subspace_rank": MAX_RANK,
                "random_subspace_attenuations": random_attenuations,
                "random_absolute_attenuation_p95": random_threshold,
                "rank8_attenuation": rank_results[MAX_RANK]["attenuation"]["overall"],
                "rank8_exceeds_frozen_threshold": bool(
                    rank_results[MAX_RANK]["attenuation"]["overall"] > random_threshold
                ),
                "details": random_details,
            },
        },
        "source_subspace_diagnostics": source_diagnostics,
        "sample_counts": {
            "source_rows": len(all_source_rows),
            "source_pair_differences": len(differences),
            "evaluation_rows": len(eval_rows),
            "independent_evaluation_pairs": int(
                len({row["pair_id"] for row in eval_rows})
            ),
            "conditions_per_pair": 2,
            "token_mappings_per_condition": 2,
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
            "label_source": (
                "mechanical target-path achievability; no model or human judge"
            ),
            "intervention_scope": (
                "mapping-centered rank-1/2/4/8 projection removal at the final "
                "prompt position after decoder module 26, corresponding to "
                "hidden state 27"
            ),
            "package_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "scikit-learn": package_version("scikit-learn"),
                "numpy": package_version("numpy"),
            },
        },
    }
    result_path = args.output_dir / "arm_g_causal_subspace_result.json"
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
                "baseline": report["baseline"],
                "subspace_ablation": report["subspace_ablation"],
                "source_subspace_diagnostics": source_diagnostics,
                "sample_counts": report["sample_counts"],
            },
            indent=2,
        )
    )
    print(f"full artifact: {result_path}")


if __name__ == "__main__":
    main()
