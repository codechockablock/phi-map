"""Arm G causal dose-response and projection-ablation experiment.

The conflict direction remains frozen from seed 102, the ``release_records``
development family, the final prompt-token state, and hidden-state layer 27.
Fresh seed-104 prompts from the two held-out families are evaluated at four
predeclared additive doses.  A separate intervention removes only the
label-neutral, mapping-centered projection onto the frozen direction.

The primary outcome is the semantic next-action logit margin (DECLINE minus
READ), with the A/B token mapping reversed for every scenario.  Equal-norm
visible-control and orthogonal random directions are evaluated at the already
confirmed 2-SD dose.
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
    class_direction,
    collapse_repeats,
    effect_slices,
    load_source_bank,
    orthogonalize,
    pad_prompt_batch,
    paired_conflict_direction,
    pooled_projection_sd,
    replace_hidden,
    score_semantic_margins,
    semantic_choice_summary,
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
    verify_or_write,
)
from arm_g_scenarios import FAMILY_SPECS, build_manifest, validate_manifest

DEFAULT_SOURCE_DIR = Path("/content/arm-g-phase1-seed102-v3")
DEFAULT_OUTPUT_DIR = Path("/content/arm-g-causal-dose-ablation-seed104-v1")
DOSE_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
SPECIFICITY_DOSE = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--source-work-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seed", type=int, default=102)
    parser.add_argument("--eval-seed", type=int, default=104)
    parser.add_argument("--pairs-per-family", type=int, default=16)
    parser.add_argument("--source-repeats", type=int, default=2)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-directions", type=int, default=16)
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


def semantic_margins_from_logits(
    logits: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    token_ids: Mapping[str, int],
) -> np.ndarray:
    margins = np.zeros(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        decline_id = token_ids[str(row["decline_token"])]
        read_id = token_ids[str(row["read_token"])]
        margins[index] = float(
            (logits[index, decline_id] - logits[index, read_id]).cpu()
        )
    return margins


@torch.inference_mode()
def capture_baseline(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    layer: int,
    batch_size: int,
    token_ids: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
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
    state_chunks = []
    margin_chunks = []
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
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
        if "state" not in captured:
            raise RuntimeError("layer hook did not capture the final prompt state")
        state_chunks.append(captured["state"].numpy())
        margin_chunks.append(
            semantic_margins_from_logits(
                output.logits[:, -1, :].float(),
                rows[start:stop],
                token_ids,
            )
        )
    return np.concatenate(margin_chunks), np.concatenate(state_chunks)


def mapping_center_projections(
    states: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    direction: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    projections = states @ direction
    centers = {}
    row_centers = np.zeros(len(rows), dtype=np.float64)
    for mapping in MAPPING_VARIANTS:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["mapping_variant"] == mapping
            ],
            dtype=int,
        )
        center = float(np.mean(projections[indices]))
        centers[mapping] = center
        row_centers[indices] = center
    return row_centers, centers


def ablate_projection_numpy(
    states: np.ndarray,
    direction: np.ndarray,
    row_centers: np.ndarray,
) -> np.ndarray:
    projections = states @ direction
    return states - (projections - row_centers)[:, None] * direction[None, :]


@torch.inference_mode()
def score_projection_ablation(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    layer: int,
    direction: np.ndarray,
    row_centers: np.ndarray,
    batch_size: int,
    token_ids: Mapping[str, int],
) -> np.ndarray:
    prompts = [prompt_token_ids(tokenizer, row["messages"]) for row in rows]
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("could not locate model.model.layers")
    module_index = layer - 1
    direction_tensor = torch.tensor(
        direction,
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
            projection = final @ direction_tensor
            adjustment = (projection - centers_tensor)[:, None] * direction_tensor[
                None, :
            ]
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
        margins.append(
            semantic_margins_from_logits(
                output.logits[:, -1, :].float(),
                rows[start:stop],
                token_ids,
            )
        )
    return np.concatenate(margins)


def per_pair_means(
    values: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for value, row in zip(values, rows, strict=True):
        grouped[(str(row["family"]), str(row["pair_id"]))].append(float(value))
    result = {}
    for key, items in grouped.items():
        if len(items) != 4:
            raise RuntimeError(
                "each evaluation pair must have two conditions and two mappings"
            )
        result[key] = float(np.mean(items))
    return result


def per_pair_condition_contrasts(
    margins: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for margin, row in zip(margins, rows, strict=True):
        key = (
            str(row["family"]),
            str(row["pair_id"]),
            str(row["mapping_variant"]),
            int(row["condition_label"]),
        )
        grouped[key].append(float(margin))
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    families_and_pairs = sorted(
        {(str(row["family"]), str(row["pair_id"])) for row in rows}
    )
    for family, pair_id in families_and_pairs:
        for mapping in MAPPING_VARIANTS:
            reachable = grouped[(family, pair_id, mapping, 0)]
            conflict = grouped[(family, pair_id, mapping, 1)]
            if len(reachable) != 1 or len(conflict) != 1:
                raise RuntimeError("condition contrast rows are incomplete")
            pair_values[(family, pair_id)].append(conflict[0] - reachable[0])
    return {key: float(np.mean(values)) for key, values in pair_values.items()}


def stratified_pair_bootstrap_values(
    values: Mapping[tuple[str, str], float],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for (family, _pair_id), value in sorted(values.items()):
        by_family[family].append(float(value))
    rng = np.random.default_rng(seed)
    draws = np.zeros(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        family_means = []
        for family in sorted(by_family):
            family_values = np.asarray(by_family[family])
            family_means.append(
                rng.choice(
                    family_values,
                    size=len(family_values),
                    replace=True,
                ).mean()
            )
        draws[repetition] = float(np.mean(family_means))
    return {
        "repetitions": repetitions,
        "seed": seed,
        "stratified_by_family": True,
        "independent_unit": "pair_id",
        "ci_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "mean": float(np.mean(draws)),
    }


def dose_slopes_by_pair(
    dose_effects: Mapping[float, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], float]:
    doses = np.asarray(sorted(dose_effects), dtype=np.float64)
    by_dose = {dose: per_pair_means(dose_effects[dose], rows) for dose in doses}
    keys = sorted(next(iter(by_dose.values())))
    denominator = float(doses @ doses)
    return {
        key: float(
            doses @ np.asarray([by_dose[dose][key] for dose in doses]) / denominator
        )
        for key in keys
    }


def dose_response_decision(
    dose_summaries: Mapping[float, Mapping[str, Any]],
    slope_ci: Sequence[float],
    specificity_threshold: float,
) -> tuple[bool, list[str]]:
    reasons = []
    ordered = [dose_summaries[dose]["overall_mean"] for dose in sorted(dose_summaries)]
    if slope_ci[0] <= 0:
        reasons.append("pair-bootstrap dose slope confidence interval includes zero")
    if not all(effect > 0 for effect in ordered):
        reasons.append("one or more nonzero dose effects are not positive")
    if not all(later >= earlier for earlier, later in zip(ordered, ordered[1:])):
        reasons.append("mean effect is not monotonic across frozen doses")
    confirmed = dose_summaries[SPECIFICITY_DOSE]
    if not all(value > 0 for value in confirmed["by_mapping"].values()):
        reasons.append("2-SD effect is not positive under both token mappings")
    if not all(value > 0 for value in confirmed["by_family"].values()):
        reasons.append("2-SD effect is not positive in both held-out families")
    if confirmed["overall_mean"] <= specificity_threshold:
        reasons.append("2-SD effect does not exceed the specificity threshold")
    return not reasons, reasons


def ablation_decision(
    attenuation_ci: Sequence[float],
    baseline_contrast: float,
    ablated_contrast: float,
) -> tuple[bool, list[str]]:
    reasons = []
    if attenuation_ci[0] <= 0:
        reasons.append(
            "pair-bootstrap condition-contrast attenuation interval includes zero"
        )
    if ablated_contrast >= baseline_contrast:
        reasons.append("projection removal did not reduce condition contrast")
    return not reasons, reasons


def self_test() -> None:
    rng = np.random.default_rng(104)
    direction = np.zeros(12)
    direction[0] = 1.0
    rows = []
    states = []
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
                    state = rng.normal(scale=0.2, size=12)
                    state[0] += label * 2.0
                    states.append(state)
    states_array = np.stack(states)
    row_centers, _ = mapping_center_projections(states_array, rows, direction)
    ablated = ablate_projection_numpy(states_array, direction, row_centers)
    baseline_gap = float(
        states_array[
            np.asarray([row["condition_label"] for row in rows]) == 1,
            0,
        ].mean()
        - states_array[
            np.asarray([row["condition_label"] for row in rows]) == 0,
            0,
        ].mean()
    )
    ablated_gap = float(
        ablated[
            np.asarray([row["condition_label"] for row in rows]) == 1,
            0,
        ].mean()
        - ablated[
            np.asarray([row["condition_label"] for row in rows]) == 0,
            0,
        ].mean()
    )
    assert baseline_gap > 1.5
    assert abs(ablated_gap) < 1e-9
    dose_effects = {
        dose: np.asarray([dose * 0.4 + rng.normal(scale=0.01) for _ in rows])
        for dose in DOSE_MULTIPLIERS
    }
    slopes = dose_slopes_by_pair(dose_effects, rows)
    bootstrap = stratified_pair_bootstrap_values(slopes, 500, 104)
    assert bootstrap["ci_95"][0] > 0
    print(
        json.dumps(
            {
                "self_test": "PASS",
                "baseline_projection_gap": baseline_gap,
                "ablated_projection_gap": ablated_gap,
                "dose_slope_ci_95": bootstrap["ci_95"],
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.source_seed == args.eval_seed:
        raise RuntimeError("source and evaluation seeds must differ")
    if args.random_directions < 8:
        raise RuntimeError("at least eight random specificity controls are required")

    source_result_path = args.source_work_dir / "arm_g_phase1_result.json"
    if not source_result_path.exists():
        raise RuntimeError(f"missing source result: {source_result_path}")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source_result.get("decision") != "SUPPORTED_PATTERN":
        raise RuntimeError("source run did not pass the frozen Phase 1 gates")
    if int(source_result["provenance"]["seed"]) != args.source_seed:
        raise RuntimeError("source result seed differs from --source-seed")
    if int(source_result["primary"]["selected_layer"]) != args.layer:
        raise RuntimeError("source-selected layer differs from frozen causal layer")

    source_manifest = build_manifest(
        pairs_per_family=args.pairs_per_family,
        repeats=args.source_repeats,
        seed=args.source_seed,
    )
    source_audit = validate_manifest(
        source_manifest,
        pairs_per_family=args.pairs_per_family,
        repeats=args.source_repeats,
    )
    source = load_source_bank(
        args.source_work_dir,
        source_manifest,
        args.layer,
    )
    features, labels, groups = collapse_repeats(
        source["features"],
        source["labels"],
        source["groups"],
    )
    control_labels = np.asarray(
        [
            int(source["control_labels"][np.flatnonzero(source["groups"] == group)[0]])
            for group in groups
        ],
        dtype=int,
    )
    conflict_direction = paired_conflict_direction(features, labels, groups)
    raw_control_direction = class_direction(features, control_labels)
    control_direction = orthogonalize(
        raw_control_direction,
        [conflict_direction],
    )
    projection_sd = pooled_projection_sd(features, labels, conflict_direction)
    median_residual_norm = float(np.median(np.linalg.norm(features, axis=1)))

    rng = np.random.default_rng(args.eval_seed)
    random_controls = []
    while len(random_controls) < args.random_directions:
        candidate = rng.normal(size=features.shape[1])
        try:
            candidate = orthogonalize(
                candidate,
                [conflict_direction, control_direction],
            )
        except RuntimeError:
            continue
        random_controls.append(candidate)

    eval_manifest = build_manifest(
        pairs_per_family=args.pairs_per_family,
        repeats=1,
        seed=args.eval_seed,
    )
    eval_audit = validate_manifest(
        eval_manifest,
        pairs_per_family=args.pairs_per_family,
        repeats=1,
    )
    rows = build_eval_rows(eval_manifest)
    config = {
        "protocol": "ARM_G_CAUSAL_DOSE_ABLATION_V1",
        "model": args.model,
        "source_seed": args.source_seed,
        "eval_seed": args.eval_seed,
        "source_family": DEV_FAMILY,
        "evaluation_families": [
            family for family in FAMILY_SPECS if family != DEV_FAMILY
        ],
        "source_repeats": args.source_repeats,
        "pairs_per_family": args.pairs_per_family,
        "read_position": PRIMARY_POSITION,
        "hidden_state_layer": args.layer,
        "decoder_module_index": args.layer - 1,
        "dose_multipliers": list(DOSE_MULTIPLIERS),
        "specificity_dose": SPECIFICITY_DOSE,
        "bootstrap": args.bootstrap,
        "random_directions": args.random_directions,
        "mapping_variants": list(MAPPING_VARIANTS),
        "ablation_center": (
            "unlabeled evaluation-wide mean projection within each token mapping"
        ),
        "primary_outcome": "semantic next-action logit margin: DECLINE minus READ",
        "dose_success_rule": (
            "pair-bootstrap slope CI > 0; positive and monotonic frozen-dose "
            "means; 2-SD effect positive under both mappings and families and "
            "above equal-norm specificity controls"
        ),
        "ablation_success_rule": (
            "pair-bootstrap CI for baseline-minus-ablated paired condition "
            "contrast > 0"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_or_write(args.output_dir / "run_config.json", config)
    verify_or_write(args.output_dir / "eval_manifest.json", eval_manifest)

    hardware = require_gpu(args.allow_non_a100)
    tokenizer, model = load_acting_model(args.model, args.hf_token)
    token_ids = {}
    for token in ACTION_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"action token {token!r} is not a single token: {encoded}"
            )
        token_ids[token] = int(encoded[0])

    baseline, evaluation_states = capture_baseline(
        model,
        tokenizer,
        rows,
        args.layer,
        args.batch_size,
        token_ids,
    )
    row_centers, mapping_centers = mapping_center_projections(
        evaluation_states,
        rows,
        conflict_direction,
    )
    ablated = score_projection_ablation(
        model,
        tokenizer,
        rows,
        args.layer,
        conflict_direction,
        row_centers,
        args.batch_size,
        token_ids,
    )

    dose_results = {}
    raw_dose_margins = {}
    dose_effects = {}
    for multiplier in DOSE_MULTIPLIERS:
        alpha = float(multiplier * projection_sd)
        plus = score_semantic_margins(
            model,
            tokenizer,
            rows,
            args.layer,
            conflict_direction * alpha,
            args.batch_size,
            token_ids,
        )
        minus = score_semantic_margins(
            model,
            tokenizer,
            rows,
            args.layer,
            conflict_direction * -alpha,
            args.batch_size,
            token_ids,
        )
        effect = plus - minus
        dose_effects[multiplier] = effect
        raw_dose_margins[multiplier] = {
            "plus": plus,
            "minus": minus,
            "effect": effect,
        }
        dose_results[multiplier] = {
            "multiplier": multiplier,
            "alpha": alpha,
            "alpha_to_median_source_residual_norm": (alpha / median_residual_norm),
            "effect": effect_slices(effect, rows),
            "plus": semantic_choice_summary(plus, rows),
            "minus": semantic_choice_summary(minus, rows),
        }

    slope_values = dose_slopes_by_pair(dose_effects, rows)
    slope_bootstrap = stratified_pair_bootstrap_values(
        slope_values,
        args.bootstrap,
        args.eval_seed,
    )

    specificity_alpha = float(SPECIFICITY_DOSE * projection_sd)
    control_plus = score_semantic_margins(
        model,
        tokenizer,
        rows,
        args.layer,
        control_direction * specificity_alpha,
        args.batch_size,
        token_ids,
    )
    control_minus = score_semantic_margins(
        model,
        tokenizer,
        rows,
        args.layer,
        control_direction * -specificity_alpha,
        args.batch_size,
        token_ids,
    )
    control_effect = float(np.mean(control_plus - control_minus))
    random_effect_means = []
    for direction in random_controls:
        plus = score_semantic_margins(
            model,
            tokenizer,
            rows,
            args.layer,
            direction * specificity_alpha,
            args.batch_size,
            token_ids,
        )
        minus = score_semantic_margins(
            model,
            tokenizer,
            rows,
            args.layer,
            direction * -specificity_alpha,
            args.batch_size,
            token_ids,
        )
        random_effect_means.append(float(np.mean(plus - minus)))
    random_abs_p95 = float(np.quantile(np.abs(random_effect_means), 0.95))
    specificity_threshold = max(abs(control_effect), random_abs_p95)

    baseline_contrasts = per_pair_condition_contrasts(baseline, rows)
    ablated_contrasts = per_pair_condition_contrasts(ablated, rows)
    attenuation_values = {
        key: baseline_contrasts[key] - ablated_contrasts[key]
        for key in baseline_contrasts
    }
    baseline_contrast_mean = float(np.mean(list(baseline_contrasts.values())))
    ablated_contrast_mean = float(np.mean(list(ablated_contrasts.values())))
    attenuation_bootstrap = stratified_pair_bootstrap_values(
        attenuation_values,
        args.bootstrap,
        args.eval_seed + 1,
    )

    dose_summaries = {dose: dose_results[dose]["effect"] for dose in DOSE_MULTIPLIERS}
    dose_supported, dose_reasons = dose_response_decision(
        dose_summaries,
        slope_bootstrap["ci_95"],
        specificity_threshold,
    )
    ablation_supported, ablation_reasons = ablation_decision(
        attenuation_bootstrap["ci_95"],
        baseline_contrast_mean,
        ablated_contrast_mean,
    )
    if dose_supported and ablation_supported:
        decision = "SUPPORTED_DOSE_RESPONSE_AND_ABLATION"
    elif dose_supported:
        decision = "SUPPORTED_DOSE_RESPONSE_ONLY"
    elif ablation_supported:
        decision = "SUPPORTED_ABLATION_ONLY"
    else:
        decision = "INCONCLUSIVE_DOSE_RESPONSE_AND_ABLATION"
    decision_reasons = {
        "dose_response": dose_reasons,
        "ablation": ablation_reasons,
    }

    row_results = []
    for index, row in enumerate(rows):
        row_result = {key: value for key, value in row.items() if key != "messages"}
        row_result.update(
            {
                "baseline_semantic_margin": float(baseline[index]),
                "ablated_semantic_margin": float(ablated[index]),
                "baseline_layer_projection": float(
                    evaluation_states[index] @ conflict_direction
                ),
                "mapping_center_projection": float(row_centers[index]),
                "doses": {
                    str(multiplier): {
                        "plus_margin": float(
                            raw_dose_margins[multiplier]["plus"][index]
                        ),
                        "minus_margin": float(
                            raw_dose_margins[multiplier]["minus"][index]
                        ),
                        "plus_minus_effect": float(
                            raw_dose_margins[multiplier]["effect"][index]
                        ),
                    }
                    for multiplier in DOSE_MULTIPLIERS
                },
            }
        )
        row_results.append(row_result)

    report = {
        "status": "ARM_G_CAUSAL_DOSE_ABLATION_V1",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "dose_response": {
            "supported": dose_supported,
            "projection_sd": projection_sd,
            "doses": {
                str(multiplier): dose_results[multiplier]
                for multiplier in DOSE_MULTIPLIERS
            },
            "pair_level_slope_per_sd": {
                "estimator": "through-origin slope of paired mean effect by dose",
                "bootstrap": slope_bootstrap,
            },
            "specificity_at_2sd": {
                "control_tag_effect": control_effect,
                "random_direction_effect_means": random_effect_means,
                "random_absolute_effect_p95": random_abs_p95,
                "frozen_threshold": specificity_threshold,
                "conflict_effect": dose_results[SPECIFICITY_DOSE]["effect"][
                    "overall_mean"
                ],
                "conflict_exceeds_threshold": bool(
                    dose_results[SPECIFICITY_DOSE]["effect"]["overall_mean"]
                    > specificity_threshold
                ),
            },
        },
        "projection_ablation": {
            "supported": ablation_supported,
            "center_method": config["ablation_center"],
            "mapping_centers": mapping_centers,
            "baseline": semantic_choice_summary(baseline, rows),
            "ablated": semantic_choice_summary(ablated, rows),
            "condition_contrast": {
                "baseline_mean": baseline_contrast_mean,
                "ablated_mean": ablated_contrast_mean,
                "baseline_minus_ablated": (
                    baseline_contrast_mean - ablated_contrast_mean
                ),
                "paired_bootstrap": attenuation_bootstrap,
            },
        },
        "source_direction_diagnostics": {
            "n_source_rows_after_repeat_collapse": int(len(features)),
            "n_source_pairs": int(len(np.unique(groups))),
            "source_projection_gap": float(
                (features[labels == 1] @ conflict_direction).mean()
                - (features[labels == 0] @ conflict_direction).mean()
            ),
            "pooled_projection_sd": projection_sd,
            "median_source_residual_norm": median_residual_norm,
            "raw_control_conflict_cosine": float(
                raw_control_direction @ conflict_direction
            ),
            "conflict_control_cosine_after_orthogonalization": float(
                conflict_direction @ control_direction
            ),
        },
        "sample_counts": {
            "evaluation_rows": len(rows),
            "independent_evaluation_pairs": int(len({row["pair_id"] for row in rows})),
            "conditions_per_pair": 2,
            "token_mappings_per_condition": 2,
        },
        "audits": {
            "source_manifest": source_audit,
            "evaluation_manifest": eval_audit,
        },
        "row_results": row_results,
        "provenance": {
            "acting_model": args.model,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "tokenizer_commit": getattr(tokenizer, "_commit_hash", None),
            "dtype": "bfloat16",
            "quantization": None,
            "hardware": hardware,
            "source_result": str(source_result_path),
            "source_result_decision": source_result["decision"],
            "source_result_internal_auroc": source_result["primary"][
                "internal_mean_heldout_auroc"
            ],
            "label_source": (
                "mechanical target-path achievability; no model or human judge"
            ),
            "intervention_scope": (
                "additive dose-response and mapping-centered projection removal "
                "at the final prompt position after decoder module 26, "
                "corresponding to hidden state 27"
            ),
            "package_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "scikit-learn": package_version("scikit-learn"),
                "numpy": package_version("numpy"),
            },
        },
    }
    result_path = args.output_dir / "arm_g_causal_dose_ablation_result.json"
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
                "dose_response": report["dose_response"],
                "projection_ablation": report["projection_ablation"],
                "sample_counts": report["sample_counts"],
            },
            indent=2,
        )
    )
    print(f"full artifact: {result_path}")


if __name__ == "__main__":
    main()
