"""Arm G cross-layer rank-4 causal mediation.

Layer-27 rank expansion saturated: rank 4 and rank 8 attenuate the natural
condition contrast almost identically while removing only about three percent
of it.  This run tests the competing explanation that the missing causal mass
is distributed through depth rather than hidden in additional directions at one
layer.

The protocol is frozen before evaluating seed 106.  The same paired rank-4
construction used at layer 27 is learned independently at each of six
preregistered hidden-state layers from seed-101 and seed-102 development-family
paired differences.  Each layer is ablated alone, and then the layers are
ablated together in ascending depth order with mapping centers recaptured under
the active upstream cascade.  Layer 12 sits below the decodability onset in
both source seeds and acts as a frozen pre-onset specificity layer.

The primary outcome is the semantic next-action logit margin (DECLINE minus
READ).  The preregistered question is whether the full six-layer ablation
attenuates the condition contrast by more than the layer-27 ablation alone.
Equal-rank random subspaces, orthogonal to the target subspace at every layer
and carried through the identical cascade, provide the specificity control.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from arm_g_causal import (
    ACTION_TOKENS,
    MAPPING_VARIANTS,
    build_eval_rows,
    pad_prompt_batch,
    replace_hidden,
    semantic_choice_summary,
)
from arm_g_causal_dose_ablation import (
    per_pair_condition_contrasts,
    semantic_margins_from_logits,
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
    source_rows,
)
from arm_g_phase1 import (
    DEV_FAMILY,
    PRIMARY_POSITION,
    TARGET_MODEL,
    atomic_json,
    auc,
    load_acting_model,
    package_version,
    prompt_token_ids,
    require_gpu,
    unit,
    verify_or_write,
)
from arm_g_scenarios import build_manifest, validate_manifest

DEFAULT_OUTPUT_DIR = Path("/content/arm-g-cross-layer-seed106-v1")
SOURCE_SEEDS = (101, 102)
RANK1_SEED = 102
EVAL_SEED = 106
LAYERS = (12, 16, 20, 24, 27, 30)
ANCHOR_LAYER = 27
PRE_ONSET_LAYER = 12
RANK = 4
LAYER_EVIDENCE = {
    "selection_basis": (
        "frozen from the seed-101 and seed-102 phase-1 prototype layer sweeps: "
        "layer 12 is below the decodability onset, layer 16 is the onset, "
        "layers 20/24/30 span the plateau, layer 27 is the causally validated "
        "anchor"
    ),
    "prototype_development_cv_auroc": {
        "101": {"12": 0.547, "16": 0.703, "20": 0.719, "24": 0.758, "27": 0.789},
        "102": {"12": 0.500, "16": 0.652, "20": 0.676, "24": 0.715, "27": 0.742},
    },
    "anchor_layer_causal_evidence": {
        "rank8_attenuation_fraction_of_natural_contrast": 0.0325,
        "rank4_minus_rank1_attenuation": 0.031,
        "source": "arm_g_causal_subspace_seed105_v1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairs-per-family", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-subspaces", type=int, default=8)
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


def decoder_modules(model: Any) -> Any:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("could not locate model.model.layers")
    return layers


def check_layers(layers: Any, requested: Sequence[int]) -> None:
    for layer in requested:
        module_index = layer - 1
        if module_index < 0 or module_index >= len(layers):
            raise RuntimeError(
                f"hidden-state layer {layer} has no decoder module mapping "
                f"within {len(layers)} layers"
            )


def make_ablation_hook(
    basis: torch.Tensor,
    centers: torch.Tensor,
) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        final = hidden[:, -1, :].float()
        projection = final @ basis
        adjustment = (projection - centers) @ basis.T
        adjusted = hidden.clone()
        adjusted[:, -1, :] -= adjustment.to(dtype=hidden.dtype)
        return replace_hidden(output, adjusted)

    return hook


def make_capture_hook(captured: dict[int, torch.Tensor], layer: int) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        captured[layer] = hidden[:, -1, :].float().cpu()

    return hook


@torch.inference_mode()
def run_forward(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    ablations: Mapping[int, tuple[np.ndarray, np.ndarray]],
    capture_layers: Sequence[int],
    token_ids: Mapping[str, int] | None,
) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    """One forward sweep with an arbitrary set of layer ablations.

    Ablation hooks are registered before capture hooks, so a layer that is both
    ablated and captured reports its post-ablation state.  In the sequential
    cascade the captured layer is always downstream of every active ablation.
    """
    prompts = [prompt_token_ids(tokenizer, row["messages"]) for row in rows]
    modules = decoder_modules(model)
    check_layers(modules, [*ablations, *capture_layers])
    basis_tensors = {
        layer: torch.tensor(basis, dtype=torch.float32, device="cuda")
        for layer, (basis, _centers) in ablations.items()
    }
    margin_chunks = []
    state_chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in capture_layers}
    for start in range(0, len(rows), batch_size):
        stop = min(len(rows), start + batch_size)
        input_ids, attention_mask = pad_prompt_batch(
            prompts[start:stop],
            tokenizer.pad_token_id,
        )
        captured: dict[int, torch.Tensor] = {}
        handles = []
        try:
            for layer in sorted(ablations):
                _basis, row_centers = ablations[layer]
                centers_tensor = torch.tensor(
                    row_centers[start:stop],
                    dtype=torch.float32,
                    device="cuda",
                )
                handles.append(
                    modules[layer - 1].register_forward_hook(
                        make_ablation_hook(basis_tensors[layer], centers_tensor)
                    )
                )
            for layer in sorted(capture_layers):
                handles.append(
                    modules[layer - 1].register_forward_hook(
                        make_capture_hook(captured, layer)
                    )
                )
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        for layer in capture_layers:
            if layer not in captured:
                raise RuntimeError(f"layer {layer} hook did not capture a state")
            state_chunks[layer].append(captured[layer].numpy())
        if token_ids is not None:
            margin_chunks.append(
                semantic_margins_from_logits(
                    output.logits[:, -1, :].float(),
                    rows[start:stop],
                    token_ids,
                )
            )
    margins = np.concatenate(margin_chunks) if token_ids is not None else None
    states = {layer: np.concatenate(chunks) for layer, chunks in state_chunks.items()}
    return margins, states


def layer_decodability(
    source_states: np.ndarray,
    source_row_labels: np.ndarray,
    evaluation_states: np.ndarray,
    evaluation_rows: Sequence[Mapping[str, Any]],
    basis: np.ndarray,
    prototype_direction: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(
        [int(row["condition_label"]) for row in evaluation_rows],
        dtype=int,
    )
    scaler = StandardScaler().fit(source_states @ basis)
    model = LogisticRegression(max_iter=1000, C=1.0).fit(
        scaler.transform(source_states @ basis),
        source_row_labels,
    )
    scores = model.decision_function(
        scaler.transform(evaluation_states @ basis),
    )
    return {
        "prototype_transfer_auroc": auc(
            evaluation_states @ prototype_direction,
            labels,
        ),
        "rank4_logistic_transfer_auroc": auc(scores, labels),
    }


def sequential_cascade(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    bases: Mapping[int, np.ndarray],
    baseline_states: Mapping[int, np.ndarray],
    batch_size: int,
    token_ids: Mapping[str, int],
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, list[float]]]]:
    """Ablate the frozen layers cumulatively in ascending depth order.

    Each layer's mapping centers are measured on the state that actually
    reaches it under the already-active upstream ablations, so the intervention
    stays mean-preserving within each token mapping at every step.
    """
    ordered = sorted(bases)
    active: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cumulative_margins: dict[int, np.ndarray] = {}
    cumulative_centers: dict[int, dict[str, list[float]]] = {}
    cascade_states = {ordered[0]: baseline_states[ordered[0]]}
    for index, layer in enumerate(ordered):
        row_centers, mapping_centers = mapping_center_subspace(
            cascade_states[layer],
            rows,
            bases[layer],
        )
        active[layer] = (bases[layer], row_centers)
        cumulative_centers[layer] = mapping_centers
        capture_layers = [ordered[index + 1]] if index + 1 < len(ordered) else []
        margins, states = run_forward(
            model,
            tokenizer,
            rows,
            batch_size,
            active,
            capture_layers,
            token_ids,
        )
        cumulative_margins[layer] = margins
        for captured_layer, captured_states in states.items():
            cascade_states[captured_layer] = captured_states
    return cumulative_margins, cumulative_centers


def cross_layer_decision(
    full: Mapping[str, Any],
    increment_ci: Sequence[float],
    random_threshold: float,
) -> tuple[str, list[str]]:
    reasons = []
    if full["attenuation_bootstrap"]["ci_95"][0] <= 0:
        reasons.append("full-set attenuation confidence interval includes zero")
    if full["attenuation"]["overall"] <= random_threshold:
        reasons.append(
            "full-set attenuation does not exceed the random-cascade threshold"
        )
    if not all(value > 0 for value in full["attenuation"]["by_family"].values()):
        reasons.append("full-set attenuation is not positive in both held-out families")
    if not all(value > 0 for value in full["attenuation"]["by_mapping"].values()):
        reasons.append("full-set attenuation is not positive under both token mappings")
    specific = not reasons
    if increment_ci[0] <= 0:
        reasons.append("full-set minus anchor-layer attenuation interval includes zero")
    if not reasons:
        return "SUPPORTED_DISTRIBUTED_DEPTH_MEDIATION", []
    if specific:
        return "SUPPORTED_NOT_BEYOND_ANCHOR_LAYER", reasons
    return "INCONCLUSIVE_CROSS_LAYER_MEDIATION", reasons


def hook_order_test() -> dict[str, float]:
    """Check the cascade assumption on CPU before spending accelerator time.

    A capture hook registered after an ablation hook on the same module must
    observe the ablated state, and the ablation must touch only the final
    prompt position.
    """

    class TupleOutputLayer(torch.nn.Module):
        def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
            return (hidden, None)

    rng = np.random.default_rng(EVAL_SEED)
    width, rank, batch, positions = 8, 3, 4, 5
    basis = torch.tensor(
        np.linalg.qr(rng.normal(size=(width, rank)))[0],
        dtype=torch.float32,
    )
    centers = torch.zeros(batch, rank, dtype=torch.float32)
    module = TupleOutputLayer()
    captured: dict[int, torch.Tensor] = {}
    handles = [
        module.register_forward_hook(make_ablation_hook(basis, centers)),
        module.register_forward_hook(make_capture_hook(captured, ANCHOR_LAYER)),
    ]
    hidden = torch.tensor(
        rng.normal(size=(batch, positions, width)),
        dtype=torch.float32,
    )
    try:
        output = module(hidden)
    finally:
        for handle in handles:
            handle.remove()
    if not isinstance(output, tuple) or output[1] is not None:
        raise AssertionError("ablation hook did not preserve the output structure")
    final = output[0][:, -1, :]
    residual = float(torch.max(torch.abs(final @ basis)))
    if residual > 1e-5:
        raise AssertionError("ablation left a component inside the target subspace")
    if not torch.allclose(output[0][:, :-1, :], hidden[:, :-1, :]):
        raise AssertionError("ablation modified positions before the final token")
    if ANCHOR_LAYER not in captured:
        raise AssertionError("capture hook did not fire")
    if not torch.allclose(captured[ANCHOR_LAYER], final, atol=1e-6):
        raise AssertionError("capture hook observed the pre-ablation state")
    return {
        "max_projection_after_ablation": residual,
        "max_prefix_drift": float(
            torch.max(torch.abs(output[0][:, :-1, :] - hidden[:, :-1, :]))
        ),
    }


def self_test() -> None:
    hook_diagnostics = hook_order_test()
    rng = np.random.default_rng(EVAL_SEED)
    width = 24
    metadata = []
    for seed in SOURCE_SEEDS:
        for pair_index in range(16):
            metadata.append(
                {
                    "source_seed": seed,
                    "pair_id": f"{DEV_FAMILY}:{pair_index:03d}",
                }
            )
    bases = {}
    for layer in LAYERS:
        latent = np.linalg.qr(rng.normal(size=(width, RANK)))[0]
        differences = []
        for item in metadata:
            coefficients = rng.normal(scale=0.2, size=RANK)
            coefficients[0] += 2.0
            if int(item["source_seed"]) == 101:
                coefficients[1:] += np.linspace(1.0, 0.3, RANK - 1)
            differences.append(
                latent @ coefficients + rng.normal(scale=0.01, size=width)
            )
        basis, diagnostics = build_target_basis(
            np.stack(differences),
            metadata,
            rank1_seed=RANK1_SEED,
            max_rank=RANK,
        )
        if basis.shape != (width, RANK):
            raise AssertionError("layer basis has the wrong shape")
        if diagnostics["source_pair_differences"] != len(metadata):
            raise AssertionError("layer basis used the wrong number of differences")
        bases[layer] = basis

    rows = []
    for family in ("data_checksums", "incident_times"):
        for pair_index in range(8):
            for label in (0, 1):
                for mapping in MAPPING_VARIANTS:
                    rows.append(
                        {
                            "family": family,
                            "pair_id": f"{family}:{pair_index:03d}",
                            "condition_label": label,
                            "mapping_variant": mapping,
                        }
                    )

    states = rng.normal(scale=0.1, size=(len(rows), width))
    for index, row in enumerate(rows):
        states[index] += (
            row["condition_label"]
            * bases[ANCHOR_LAYER]
            @ np.linspace(
                0.8,
                0.2,
                RANK,
            )
        )
    row_centers, mapping_centers = mapping_center_subspace(
        states,
        rows,
        bases[ANCHOR_LAYER],
    )
    projections = states @ bases[ANCHOR_LAYER]
    ablated = states - (projections - row_centers) @ bases[ANCHOR_LAYER].T
    residual = (ablated @ bases[ANCHOR_LAYER]) - row_centers
    if np.max(np.abs(residual)) > 1e-8:
        raise AssertionError("mapping-centered ablation left a labeled component")
    for mapping in MAPPING_VARIANTS:
        if len(mapping_centers[mapping]) != RANK:
            raise AssertionError("mapping center has the wrong rank")

    excluded = bases[ANCHOR_LAYER]
    control = random_orthogonal_subspace(rng, width, RANK, excluded)
    if np.max(np.abs(excluded.T @ control)) > 1e-8:
        raise AssertionError("random cascade subspace overlaps the target subspace")

    baseline_margins = states @ bases[ANCHOR_LAYER][:, 0] * 3.0
    anchor_margins = baseline_margins * 0.9
    full_margins = baseline_margins * 0.6
    baseline_pairs = per_pair_condition_contrasts(baseline_margins, rows)
    anchor_pairs = per_pair_condition_contrasts(anchor_margins, rows)
    full_pairs = per_pair_condition_contrasts(full_margins, rows)
    increment = {
        key: (baseline_pairs[key] - full_pairs[key])
        - (baseline_pairs[key] - anchor_pairs[key])
        for key in baseline_pairs
    }
    bootstrap = stratified_pair_bootstrap_values(increment, 500, EVAL_SEED + 300)
    if bootstrap["ci_95"][0] <= 0:
        raise AssertionError("synthetic depth increment failed to separate from zero")

    full_result = {
        "attenuation": {
            "overall": 0.4,
            "by_family": {"data_checksums": 0.4, "incident_times": 0.4},
            "by_mapping": {mapping: 0.4 for mapping in MAPPING_VARIANTS},
        },
        "attenuation_bootstrap": {"ci_95": [0.2, 0.6]},
    }
    decision, reasons = cross_layer_decision(full_result, bootstrap["ci_95"], 0.05)
    if decision != "SUPPORTED_DISTRIBUTED_DEPTH_MEDIATION" or reasons:
        raise AssertionError("decision rule rejected a clean synthetic positive")
    flat, flat_reasons = cross_layer_decision(full_result, [-0.1, 0.1], 0.05)
    if flat != "SUPPORTED_NOT_BEYOND_ANCHOR_LAYER" or not flat_reasons:
        raise AssertionError("decision rule mislabeled a saturated synthetic case")
    null, null_reasons = cross_layer_decision(full_result, [-0.1, 0.1], 0.9)
    if null != "INCONCLUSIVE_CROSS_LAYER_MEDIATION" or not null_reasons:
        raise AssertionError("decision rule mislabeled a nonspecific synthetic case")

    print(
        json.dumps(
            {
                "self_test": "PASS",
                "layers": list(LAYERS),
                "rank": RANK,
                "hook_order": hook_diagnostics,
                "max_residual_after_ablation": float(np.max(np.abs(residual))),
                "synthetic_increment_ci_95": bootstrap["ci_95"],
                "decisions_checked": [decision, flat, null],
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if ANCHOR_LAYER not in LAYERS:
        raise RuntimeError("the frozen layer set must contain the anchor layer")
    if args.random_subspaces < 8:
        raise RuntimeError("at least eight random specificity cascades are required")

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
        "protocol": "ARM_G_CROSS_LAYER_MEDIATION_V1",
        "model": args.model,
        "source_seeds": list(SOURCE_SEEDS),
        "rank1_seed": RANK1_SEED,
        "eval_seed": EVAL_SEED,
        "source_family": DEV_FAMILY,
        "evaluation_families": [
            family for family in sorted({row["family"] for row in eval_rows})
        ],
        "source_evidence": SOURCE_EVIDENCE,
        "layers": list(LAYERS),
        "anchor_layer": ANCHOR_LAYER,
        "pre_onset_layer": PRE_ONSET_LAYER,
        "layer_evidence": LAYER_EVIDENCE,
        "rank": RANK,
        "pairs_per_family": args.pairs_per_family,
        "read_position": PRIMARY_POSITION,
        "subspace_construction": (
            "at each frozen layer, rank 1 is the seed-102 mean paired conflict "
            "direction and ranks 2-4 are the leading SVD directions of the "
            "seed-101/102 paired differences after removing rank 1; bases are "
            "learned from unablated source states and never refit under the "
            "cascade"
        ),
        "cascade": (
            "layers are added in ascending depth order; each layer's mapping "
            "centers are measured on the state reaching it under the active "
            "upstream ablations"
        ),
        "bootstrap": args.bootstrap,
        "random_subspaces": args.random_subspaces,
        "mapping_variants": list(MAPPING_VARIANTS),
        "primary_outcome": "semantic next-action logit margin: DECLINE minus READ",
        "success_rule": (
            "full-set attenuation CI > 0; full-set attenuation exceeds the p95 "
            "absolute random-cascade attenuation; full-set attenuation is "
            "positive in both held-out families and both token mappings; and "
            "full-set-minus-anchor-layer attenuation CI > 0"
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

    _margins, source_states = run_forward(
        model,
        tokenizer,
        all_source_rows,
        args.batch_size,
        {},
        LAYERS,
        None,
    )
    bases = {}
    prototypes = {}
    source_diagnostics = {}
    for layer in LAYERS:
        differences, difference_metadata = paired_differences(
            source_states[layer],
            all_source_rows,
        )
        basis, diagnostics = build_target_basis(
            differences,
            difference_metadata,
            rank1_seed=RANK1_SEED,
            max_rank=RANK,
        )
        bases[layer] = basis
        prototypes[layer] = unit(differences.mean(axis=0))
        diagnostics["paired_difference_energy_captured_by_rank"] = {
            rank: value
            for rank, value in diagnostics[
                "paired_difference_energy_captured_by_rank"
            ].items()
            if int(rank) <= RANK
        }
        source_diagnostics[str(layer)] = diagnostics

    baseline, evaluation_states = run_forward(
        model,
        tokenizer,
        eval_rows,
        args.batch_size,
        {},
        LAYERS,
        token_ids,
    )
    baseline_pairs = per_pair_condition_contrasts(baseline, eval_rows)
    baseline_slices = contrast_slices(baseline, eval_rows)
    source_labels = np.asarray(
        [int(row["condition_label"]) for row in all_source_rows],
        dtype=int,
    )

    def summarize(
        margins: np.ndarray,
        bootstrap_seed: int,
    ) -> dict[str, Any]:
        pairs = per_pair_condition_contrasts(margins, eval_rows)
        values = {key: baseline_pairs[key] - pairs[key] for key in baseline_pairs}
        slices = contrast_slices(margins, eval_rows)
        attenuation = attenuation_slices(baseline_slices, slices)
        return {
            "condition_contrast": slices,
            "attenuation": attenuation,
            "attenuation_fraction": float(
                attenuation["overall"] / baseline_slices["overall"]
            ),
            "attenuation_bootstrap": stratified_pair_bootstrap_values(
                values,
                args.bootstrap,
                bootstrap_seed,
            ),
            "attenuation_values": values,
        }

    single_layer = {}
    single_layer_margins = {}
    for layer in LAYERS:
        row_centers, mapping_centers = mapping_center_subspace(
            evaluation_states[layer],
            eval_rows,
            bases[layer],
        )
        margins, _states = run_forward(
            model,
            tokenizer,
            eval_rows,
            args.batch_size,
            {layer: (bases[layer], row_centers)},
            [],
            token_ids,
        )
        summary = summarize(margins, EVAL_SEED + layer)
        summary["mapping_centers"] = mapping_centers
        summary["decodability"] = layer_decodability(
            source_states[layer],
            source_labels,
            evaluation_states[layer],
            eval_rows,
            bases[layer],
            prototypes[layer],
        )
        summary["choice_summary"] = semantic_choice_summary(margins, eval_rows)
        single_layer[layer] = summary
        single_layer_margins[layer] = margins

    single_layer_random = {}
    for layer in LAYERS:
        rng = np.random.default_rng(EVAL_SEED * 1000 + layer)
        attenuations = []
        for _control_index in range(args.random_subspaces):
            random_basis = random_orthogonal_subspace(
                rng,
                bases[layer].shape[0],
                RANK,
                bases[layer],
            )
            row_centers, _centers = mapping_center_subspace(
                evaluation_states[layer],
                eval_rows,
                random_basis,
            )
            margins, _states = run_forward(
                model,
                tokenizer,
                eval_rows,
                args.batch_size,
                {layer: (random_basis, row_centers)},
                [],
                token_ids,
            )
            random_slices = contrast_slices(margins, eval_rows)
            attenuations.append(
                float(baseline_slices["overall"] - random_slices["overall"])
            )
        single_layer_random[layer] = {
            "attenuations": attenuations,
            "absolute_p95": float(np.quantile(np.abs(attenuations), 0.95)),
        }

    cumulative_margins, cumulative_centers = sequential_cascade(
        model,
        tokenizer,
        eval_rows,
        bases,
        evaluation_states,
        args.batch_size,
        token_ids,
    )
    cumulative = {}
    for index, layer in enumerate(sorted(bases)):
        summary = summarize(cumulative_margins[layer], EVAL_SEED + 100 + layer)
        summary["layers_ablated"] = sorted(bases)[: index + 1]
        summary["mapping_centers"] = cumulative_centers[layer]
        summary["choice_summary"] = semantic_choice_summary(
            cumulative_margins[layer],
            eval_rows,
        )
        cumulative[layer] = summary

    full = cumulative[max(LAYERS)]
    anchor = single_layer[ANCHOR_LAYER]
    increment_values = {
        key: full["attenuation_values"][key] - anchor["attenuation_values"][key]
        for key in baseline_pairs
    }
    increment_bootstrap = stratified_pair_bootstrap_values(
        increment_values,
        args.bootstrap,
        EVAL_SEED + 300,
    )
    best_single_layer = max(
        LAYERS,
        key=lambda layer: single_layer[layer]["attenuation"]["overall"],
    )
    best_increment_values = {
        key: full["attenuation_values"][key]
        - single_layer[best_single_layer]["attenuation_values"][key]
        for key in baseline_pairs
    }
    best_increment_bootstrap = stratified_pair_bootstrap_values(
        best_increment_values,
        args.bootstrap,
        EVAL_SEED + 400,
    )

    random_cascade_attenuations = []
    for control_index in range(args.random_subspaces):
        rng = np.random.default_rng(EVAL_SEED * 100 + control_index)
        random_bases = {
            layer: random_orthogonal_subspace(
                rng,
                bases[layer].shape[0],
                RANK,
                bases[layer],
            )
            for layer in LAYERS
        }
        random_margins, _random_centers = sequential_cascade(
            model,
            tokenizer,
            eval_rows,
            random_bases,
            evaluation_states,
            args.batch_size,
            token_ids,
        )
        random_slices = contrast_slices(random_margins[max(LAYERS)], eval_rows)
        random_cascade_attenuations.append(
            float(baseline_slices["overall"] - random_slices["overall"])
        )
    random_cascade_p95 = float(np.quantile(np.abs(random_cascade_attenuations), 0.95))

    decision, decision_reasons = cross_layer_decision(
        full,
        increment_bootstrap["ci_95"],
        random_cascade_p95,
    )

    def public(summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in summary.items() if key != "attenuation_values"
        }

    row_results = []
    for index, row in enumerate(eval_rows):
        item = {key: value for key, value in row.items() if key != "messages"}
        item.update(
            {
                "baseline_semantic_margin": float(baseline[index]),
                "single_layer_semantic_margins": {
                    str(layer): float(single_layer_margins[layer][index])
                    for layer in LAYERS
                },
                "cumulative_semantic_margins": {
                    str(layer): float(cumulative_margins[layer][index])
                    for layer in LAYERS
                },
                "target_projections": {
                    str(layer): (
                        evaluation_states[layer][index] @ bases[layer]
                    ).tolist()
                    for layer in LAYERS
                },
            }
        )
        row_results.append(item)

    report = {
        "status": "ARM_G_CROSS_LAYER_MEDIATION_V1",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "baseline": {
            "choice_summary": semantic_choice_summary(baseline, eval_rows),
            "condition_contrast": baseline_slices,
        },
        "single_layer": {
            str(layer): {
                **public(single_layer[layer]),
                "random_control": single_layer_random[layer],
                "exceeds_random_control": bool(
                    single_layer[layer]["attenuation"]["overall"]
                    > single_layer_random[layer]["absolute_p95"]
                ),
            }
            for layer in LAYERS
        },
        "cumulative": {
            str(layer): public(cumulative[layer]) for layer in sorted(bases)
        },
        "depth_increment": {
            "full_minus_anchor_layer": {
                "anchor_layer": ANCHOR_LAYER,
                "mean": float(
                    full["attenuation"]["overall"] - anchor["attenuation"]["overall"]
                ),
                "paired_bootstrap": increment_bootstrap,
            },
            "full_minus_best_single_layer": {
                "best_single_layer": best_single_layer,
                "selection": "post hoc; descriptive only",
                "mean": float(
                    full["attenuation"]["overall"]
                    - single_layer[best_single_layer]["attenuation"]["overall"]
                ),
                "paired_bootstrap": best_increment_bootstrap,
            },
        },
        "specificity": {
            "random_subspace_rank": RANK,
            "random_cascade_attenuations": random_cascade_attenuations,
            "random_cascade_absolute_p95": random_cascade_p95,
            "full_attenuation": full["attenuation"]["overall"],
            "full_exceeds_frozen_threshold": bool(
                full["attenuation"]["overall"] > random_cascade_p95
            ),
            "pre_onset_layer": {
                "layer": PRE_ONSET_LAYER,
                "attenuation": single_layer[PRE_ONSET_LAYER]["attenuation"]["overall"],
                "random_absolute_p95": single_layer_random[PRE_ONSET_LAYER][
                    "absolute_p95"
                ],
                "exceeds_random_control": bool(
                    single_layer[PRE_ONSET_LAYER]["attenuation"]["overall"]
                    > single_layer_random[PRE_ONSET_LAYER]["absolute_p95"]
                ),
            },
        },
        "decodability_profile": {
            str(layer): single_layer[layer]["decodability"] for layer in LAYERS
        },
        "source_subspace_diagnostics": source_diagnostics,
        "sample_counts": {
            "source_rows": len(all_source_rows),
            "evaluation_rows": len(eval_rows),
            "independent_evaluation_pairs": int(
                len({row["pair_id"] for row in eval_rows})
            ),
            "conditions_per_pair": 2,
            "token_mappings_per_condition": 2,
            "layers": len(LAYERS),
            "dimensions_removed_in_full_cascade": len(LAYERS) * RANK,
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
            "layer_evidence": LAYER_EVIDENCE,
            "label_source": (
                "mechanical target-path achievability; no model or human judge"
            ),
            "intervention_scope": (
                "mapping-centered rank-4 projection removal at the final prompt "
                "position, applied at one frozen layer at a time and then "
                "cumulatively across the frozen layer set"
            ),
            "package_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "scikit-learn": package_version("scikit-learn"),
                "numpy": package_version("numpy"),
            },
        },
    }
    result_path = args.output_dir / "arm_g_cross_layer_result.json"
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
                "single_layer_attenuation": {
                    str(layer): single_layer[layer]["attenuation"]["overall"]
                    for layer in LAYERS
                },
                "single_layer_random_p95": {
                    str(layer): single_layer_random[layer]["absolute_p95"]
                    for layer in LAYERS
                },
                "cumulative_attenuation": {
                    str(layer): cumulative[layer]["attenuation"]["overall"]
                    for layer in sorted(bases)
                },
                "cumulative_attenuation_fraction": {
                    str(layer): cumulative[layer]["attenuation_fraction"]
                    for layer in sorted(bases)
                },
                "depth_increment": report["depth_increment"],
                "specificity": {
                    "random_cascade_absolute_p95": random_cascade_p95,
                    "full_exceeds_frozen_threshold": report["specificity"][
                        "full_exceeds_frozen_threshold"
                    ],
                    "pre_onset_layer": report["specificity"]["pre_onset_layer"],
                },
                "decodability_profile": report["decodability_profile"],
                "sample_counts": report["sample_counts"],
            },
            indent=2,
        )
    )
    print(f"full artifact: {result_path}")


if __name__ == "__main__":
    main()
