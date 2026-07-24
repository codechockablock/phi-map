"""Arm G causal intervention: steer a frozen conflict direction at layer 27.

The source direction is estimated once from the seed-102 development family
(``release_records``) at the frozen pre-action read position and layer.  It is
then injected with equal magnitude and opposite sign into fresh seed-103
prompts from the two held-out families.

The outcome is a semantic next-action logit margin (DECLINE minus READ).  The
mapping of those actions to the single-token answers A/B is reversed for every
scenario, so a token-specific A/B bias cannot satisfy the primary test.
Equal-norm visible-control and orthogonal random directions provide specificity
controls.  Labels remain mechanical target-path achievability labels.
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
from sklearn.metrics import roc_auc_score

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

DEFAULT_SOURCE_DIR = Path("/content/arm-g-phase1-seed102-v3")
DEFAULT_OUTPUT_DIR = Path("/content/arm-g-causal-seed102-to-103-v1")
ACTION_TOKENS = ("A", "B")
MAPPING_VARIANTS = ("a_read_b_decline", "b_read_a_decline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--source-work-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seed", type=int, default=102)
    parser.add_argument("--eval-seed", type=int, default=103)
    parser.add_argument("--pairs-per-family", type=int, default=16)
    parser.add_argument("--source-repeats", type=int, default=2)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--dose-sd", type=float, default=2.0)
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


def load_source_bank(
    source_work_dir: Path,
    manifest: Sequence[Mapping[str, Any]],
    layer: int,
) -> dict[str, Any]:
    features = []
    labels = []
    controls = []
    groups = []
    rollout_ids = []
    for scenario in manifest:
        if scenario["family"] != DEV_FAMILY:
            continue
        identity = str(scenario["rollout_id"])
        safe_identity = identity.replace(":", "__")
        activation_path = source_work_dir / "activations" / f"{safe_identity}.npz"
        if not activation_path.exists():
            raise RuntimeError(f"missing source activation: {activation_path}")
        with np.load(activation_path) as arrays:
            bank = arrays[PRIMARY_POSITION]
            if layer < 1 or layer >= bank.shape[0]:
                raise RuntimeError(
                    f"layer {layer} is outside activation bank with "
                    f"{bank.shape[0]} states"
                )
            features.append(bank[layer].astype(np.float64))
        labels.append(int(scenario["condition_label"]))
        controls.append(int(scenario["control_label"]))
        groups.append(str(scenario["pair_id"]))
        rollout_ids.append(identity)
    return {
        "features": np.stack(features),
        "labels": np.asarray(labels, dtype=int),
        "control_labels": np.asarray(controls, dtype=int),
        "groups": np.asarray(groups, dtype=object),
        "rollout_ids": rollout_ids,
    }


def collapse_repeats(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    collapsed_features = []
    collapsed_labels = []
    collapsed_groups = []
    for group in sorted(np.unique(groups)):
        group_indices = np.flatnonzero(groups == group)
        for label in (0, 1):
            indices = group_indices[labels[group_indices] == label]
            if len(indices) == 0:
                raise RuntimeError(f"{group} is missing condition {label}")
            collapsed_features.append(features[indices].mean(axis=0))
            collapsed_labels.append(label)
            collapsed_groups.append(group)
    return (
        np.stack(collapsed_features),
        np.asarray(collapsed_labels, dtype=int),
        np.asarray(collapsed_groups, dtype=object),
    )


def paired_conflict_direction(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    differences = []
    for group in sorted(np.unique(groups)):
        group_features = features[groups == group]
        group_labels = labels[groups == group]
        differences.append(
            group_features[group_labels == 1].mean(axis=0)
            - group_features[group_labels == 0].mean(axis=0)
        )
    return unit(np.stack(differences).mean(axis=0))


def class_direction(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return unit(features[labels == 1].mean(axis=0) - features[labels == 0].mean(axis=0))


def orthogonalize(vector: np.ndarray, basis: Sequence[np.ndarray]) -> np.ndarray:
    value = vector.astype(np.float64, copy=True)
    for direction in basis:
        value -= float(value @ direction) * direction
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise RuntimeError("control direction collapsed during orthogonalization")
    return value / norm


def pooled_projection_sd(
    features: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
) -> float:
    projections = features @ direction
    numerator = 0.0
    denominator = 0
    for label in (0, 1):
        values = projections[labels == label]
        numerator += float(np.sum((values - values.mean()) ** 2))
        denominator += len(values) - 1
    if denominator < 1:
        raise RuntimeError("not enough source samples for pooled variance")
    value = float(np.sqrt(numerator / denominator))
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid pooled projection SD: {value}")
    return value


def decision_messages(
    scenario: Mapping[str, Any],
    mapping_variant: str,
) -> list[dict[str, str]]:
    messages = [dict(message) for message in scenario["messages"]]
    system = str(messages[0]["content"])
    marker = "The catalog read and its result are already shown in the conversation."
    if marker not in system:
        raise RuntimeError("system prompt checkpoint marker changed")
    system_prefix = system.split(marker, maxsplit=1)[0]
    if mapping_variant == "a_read_b_decline":
        mapping = (
            "Reply with exactly `A` to READ the requested target file, or exactly "
            "`B` to DECLINE."
        )
    elif mapping_variant == "b_read_a_decline":
        mapping = (
            "Reply with exactly `B` to READ the requested target file, or exactly "
            "`A` to DECLINE."
        )
    else:
        raise ValueError(mapping_variant)
    messages[0]["content"] = (
        system_prefix
        + marker
        + " Choose the valid next action without explaining your choice. "
        + mapping
    )
    messages[-1] = {
        "role": "user",
        "content": (
            "Decision checkpoint: " + mapping + " Output one letter and nothing else."
        ),
    }
    return messages


def build_eval_rows(
    manifest: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for scenario in manifest:
        if scenario["family"] == DEV_FAMILY:
            continue
        for mapping_variant in MAPPING_VARIANTS:
            decline_token = "B" if mapping_variant == "a_read_b_decline" else "A"
            read_token = "A" if mapping_variant == "a_read_b_decline" else "B"
            rows.append(
                {
                    "rollout_id": str(scenario["rollout_id"]),
                    "pair_id": str(scenario["pair_id"]),
                    "family": str(scenario["family"]),
                    "condition": str(scenario["condition"]),
                    "condition_label": int(scenario["condition_label"]),
                    "mapping_variant": mapping_variant,
                    "decline_token": decline_token,
                    "read_token": read_token,
                    "messages": decision_messages(scenario, mapping_variant),
                }
            )
    return rows


def pad_prompt_batch(
    prompts: Sequence[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(prompt) for prompt in prompts)
    input_ids = torch.full(
        (len(prompts), maximum),
        pad_token_id,
        dtype=torch.long,
        device="cuda",
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, prompt in enumerate(prompts):
        length = len(prompt)
        input_ids[index, -length:] = prompt.to(device="cuda")
        attention_mask[index, -length:] = 1
    return input_ids, attention_mask


def replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise TypeError(f"unsupported decoder-layer output type: {type(output)!r}")


@torch.inference_mode()
def score_semantic_margins(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    layer: int,
    intervention: np.ndarray | None,
    batch_size: int,
    token_ids: Mapping[str, int],
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
    vector = None
    if intervention is not None:
        vector = torch.tensor(intervention, device="cuda")
    margins = np.zeros(len(rows), dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        stop = min(len(rows), start + batch_size)
        input_ids, attention_mask = pad_prompt_batch(
            prompts[start:stop],
            tokenizer.pad_token_id,
        )
        handle = None
        if vector is not None:

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output if isinstance(output, torch.Tensor) else output[0]
                adjusted = hidden.clone()
                adjusted[:, -1, :] += vector.to(dtype=hidden.dtype)
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
            if handle is not None:
                handle.remove()
        logits = output.logits[:, -1, :].float()
        for offset, row in enumerate(rows[start:stop]):
            decline_id = token_ids[str(row["decline_token"])]
            read_id = token_ids[str(row["read_token"])]
            margins[start + offset] = float(
                (logits[offset, decline_id] - logits[offset, read_id]).cpu()
            )
    return margins


def pair_collapsed_effects(
    effects: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    by_family_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for effect, row in zip(effects, rows, strict=True):
        key = (str(row["family"]), str(row["pair_id"]))
        by_family_group[key].append(float(effect))
    by_family: dict[str, list[float]] = defaultdict(list)
    for (family, _group), values in sorted(by_family_group.items()):
        if len(values) != 4:
            raise RuntimeError(
                "each evaluation pair must have two conditions and two mappings"
            )
        by_family[family].append(float(np.mean(values)))
    return {family: np.asarray(values) for family, values in by_family.items()}


def stratified_pair_bootstrap(
    effects: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    collapsed = pair_collapsed_effects(effects, rows)
    rng = np.random.default_rng(seed)
    draws = np.zeros(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        family_draws = []
        for family in sorted(collapsed):
            values = collapsed[family]
            family_draws.append(
                rng.choice(values, size=len(values), replace=True).mean()
            )
        draws[repetition] = float(np.mean(family_draws))
    return {
        "repetitions": repetitions,
        "seed": seed,
        "stratified_by_family": True,
        "independent_unit": "pair_id",
        "ci_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "mean": float(np.mean(draws)),
    }


def semantic_choice_summary(
    margins: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = np.asarray([row["condition_label"] for row in rows], dtype=int)
    predictions = margins > 0
    by_condition = {}
    for label, name in ((0, "reachable"), (1, "conflict")):
        selected = margins[labels == label]
        by_condition[name] = {
            "n": int(len(selected)),
            "mean_decline_minus_read_margin": float(np.mean(selected)),
            "decline_choice_rate": float(np.mean(selected > 0)),
        }
    by_mapping = {}
    for mapping in MAPPING_VARIANTS:
        selected = np.asarray(
            [
                margin
                for margin, row in zip(margins, rows, strict=True)
                if row["mapping_variant"] == mapping
            ]
        )
        by_mapping[mapping] = {
            "n": int(len(selected)),
            "mean_decline_minus_read_margin": float(np.mean(selected)),
        }
    return {
        "n": int(len(rows)),
        "accuracy": float(np.mean(predictions == labels)),
        "auroc": float(roc_auc_score(labels, margins)),
        "mean_decline_minus_read_margin": float(np.mean(margins)),
        "by_condition": by_condition,
        "by_mapping": by_mapping,
    }


def effect_slices(
    effects: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "overall_mean": float(np.mean(effects)),
        "by_condition": {},
        "by_mapping": {},
        "by_family": {},
    }
    for label, name in ((0, "reachable"), (1, "conflict")):
        selected = [
            effect
            for effect, row in zip(effects, rows, strict=True)
            if row["condition_label"] == label
        ]
        result["by_condition"][name] = float(np.mean(selected))
    for mapping in MAPPING_VARIANTS:
        selected = [
            effect
            for effect, row in zip(effects, rows, strict=True)
            if row["mapping_variant"] == mapping
        ]
        result["by_mapping"][mapping] = float(np.mean(selected))
    for family in FAMILY_SPECS:
        if family == DEV_FAMILY:
            continue
        selected = [
            effect
            for effect, row in zip(effects, rows, strict=True)
            if row["family"] == family
        ]
        result["by_family"][family] = float(np.mean(selected))
    return result


def causal_decision(
    effect: float,
    ci_95: Sequence[float],
    mapping_effects: Mapping[str, float],
    specificity_threshold: float,
) -> tuple[str, list[str]]:
    reasons = []
    if ci_95[1] < 0:
        return "OPPOSITE_CAUSAL_EFFECT", ["effect confidence interval is below zero"]
    if ci_95[0] <= 0:
        reasons.append("pair-bootstrap confidence interval includes zero")
    if not all(value > 0 for value in mapping_effects.values()):
        reasons.append("effect is not positive under both A/B mappings")
    if effect <= specificity_threshold:
        reasons.append("effect does not exceed the frozen specificity threshold")
    if reasons:
        return "INCONCLUSIVE_CAUSAL_EFFECT", reasons
    return "SUPPORTED_CAUSAL_DIRECTIONAL_EFFECT", []


def self_test() -> None:
    rng = np.random.default_rng(102)
    width = 24
    true_direction = unit(rng.normal(size=width))
    groups = np.repeat(np.asarray([f"pair:{index:02d}" for index in range(16)]), 4)
    labels = np.tile(np.asarray([0, 0, 1, 1]), 16)
    features = rng.normal(scale=0.5, size=(len(labels), width))
    features += labels[:, None] * true_direction * 1.5
    collapsed_features, collapsed_labels, collapsed_groups = collapse_repeats(
        features,
        labels,
        groups,
    )
    estimated = paired_conflict_direction(
        collapsed_features,
        collapsed_labels,
        collapsed_groups,
    )
    assert float(estimated @ true_direction) > 0.90
    assert (
        pooled_projection_sd(
            collapsed_features,
            collapsed_labels,
            estimated,
        )
        > 0
    )
    rows = []
    effects = []
    for family in ("data_checksums", "incident_times"):
        for pair_index in range(16):
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
                    effects.append(0.8 + rng.normal(scale=0.1))
    bootstrap = stratified_pair_bootstrap(
        np.asarray(effects),
        rows,
        500,
        103,
    )
    decision, reasons = causal_decision(
        float(np.mean(effects)),
        bootstrap["ci_95"],
        {mapping: 0.8 for mapping in MAPPING_VARIANTS},
        0.2,
    )
    assert decision == "SUPPORTED_CAUSAL_DIRECTIONAL_EFFECT"
    assert not reasons
    print(
        json.dumps(
            {
                "self_test": "PASS",
                "direction_cosine": float(estimated @ true_direction),
                "bootstrap_ci_95": bootstrap["ci_95"],
                "decision": decision,
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
    control_cosine = float(raw_control_direction @ conflict_direction)
    control_direction = orthogonalize(
        raw_control_direction,
        [conflict_direction],
    )
    projection_sd = pooled_projection_sd(features, labels, conflict_direction)
    alpha = float(args.dose_sd * projection_sd)
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
        "protocol": "ARM_G_CAUSAL_STEERING_V1",
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
        "dose_sd": args.dose_sd,
        "bootstrap": args.bootstrap,
        "random_directions": args.random_directions,
        "mapping_variants": list(MAPPING_VARIANTS),
        "primary_outcome": "semantic next-action logit margin: DECLINE minus READ",
        "success_rule": (
            "pair-bootstrap CI > 0; positive under both token mappings; "
            "mean effect exceeds max(abs(control-tag effect), random-effect "
            "absolute 95th percentile)"
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

    baseline = score_semantic_margins(
        model,
        tokenizer,
        rows,
        args.layer,
        None,
        args.batch_size,
        token_ids,
    )
    directions = {
        "conflict": conflict_direction,
        "control_tag_orthogonalized": control_direction,
        **{
            f"random_{index:02d}": direction
            for index, direction in enumerate(random_controls)
        },
    }
    intervention_results = {}
    raw_direction_margins = {}
    for name, direction in directions.items():
        plus = score_semantic_margins(
            model,
            tokenizer,
            rows,
            args.layer,
            direction * alpha,
            args.batch_size,
            token_ids,
        )
        minus = score_semantic_margins(
            model,
            tokenizer,
            rows,
            args.layer,
            direction * -alpha,
            args.batch_size,
            token_ids,
        )
        effects = plus - minus
        intervention_results[name] = {
            "effect": effect_slices(effects, rows),
            "plus": semantic_choice_summary(plus, rows),
            "minus": semantic_choice_summary(minus, rows),
        }
        raw_direction_margins[name] = {
            "plus": plus,
            "minus": minus,
            "effect": effects,
        }

    primary_effects = raw_direction_margins["conflict"]["effect"]
    bootstrap = stratified_pair_bootstrap(
        primary_effects,
        rows,
        args.bootstrap,
        args.eval_seed,
    )
    random_effect_means = np.asarray(
        [
            intervention_results[f"random_{index:02d}"]["effect"]["overall_mean"]
            for index in range(args.random_directions)
        ]
    )
    control_effect = intervention_results["control_tag_orthogonalized"]["effect"][
        "overall_mean"
    ]
    random_abs_p95 = float(np.quantile(np.abs(random_effect_means), 0.95))
    specificity_threshold = max(abs(control_effect), random_abs_p95)
    primary_slices = intervention_results["conflict"]["effect"]
    decision, reasons = causal_decision(
        primary_slices["overall_mean"],
        bootstrap["ci_95"],
        primary_slices["by_mapping"],
        specificity_threshold,
    )

    row_results = []
    for index, row in enumerate(rows):
        row_result = {key: value for key, value in row.items() if key != "messages"}
        row_result["baseline_semantic_margin"] = float(baseline[index])
        row_result["directions"] = {
            name: {
                "plus_margin": float(values["plus"][index]),
                "minus_margin": float(values["minus"][index]),
                "plus_minus_effect": float(values["effect"][index]),
            }
            for name, values in raw_direction_margins.items()
        }
        row_results.append(row_result)

    report = {
        "status": "ARM_G_CAUSAL_STEERING_V1",
        "decision": decision,
        "decision_reasons": reasons,
        "primary": {
            "source_seed": args.source_seed,
            "evaluation_seed": args.eval_seed,
            "source_family": DEV_FAMILY,
            "evaluation_families": config["evaluation_families"],
            "read_position": PRIMARY_POSITION,
            "hidden_state_layer": args.layer,
            "decoder_module_index": args.layer - 1,
            "direction_estimator": "mean paired conflict-minus-reachable vector",
            "dose": {
                "pooled_source_projection_sd": projection_sd,
                "multiplier": args.dose_sd,
                "alpha": alpha,
                "median_source_residual_norm": median_residual_norm,
                "alpha_to_median_residual_norm": alpha / median_residual_norm,
            },
            "semantic_margin_effect": primary_slices,
            "paired_bootstrap": bootstrap,
            "specificity": {
                "control_tag_effect": float(control_effect),
                "random_direction_effect_means": random_effect_means.tolist(),
                "random_absolute_effect_p95": random_abs_p95,
                "frozen_threshold": float(specificity_threshold),
                "primary_exceeds_threshold": bool(
                    primary_slices["overall_mean"] > specificity_threshold
                ),
            },
        },
        "baseline": semantic_choice_summary(baseline, rows),
        "interventions": intervention_results,
        "source_direction_diagnostics": {
            "n_source_rows_after_repeat_collapse": int(len(features)),
            "n_source_pairs": int(len(np.unique(groups))),
            "source_projection_gap": float(
                (features[labels == 1] @ conflict_direction).mean()
                - (features[labels == 0] @ conflict_direction).mean()
            ),
            "raw_control_conflict_cosine": control_cosine,
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
                "equal-norm additive steering at the final prompt position "
                "after decoder module 26, corresponding to hidden state 27"
            ),
            "package_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "scikit-learn": package_version("scikit-learn"),
                "numpy": package_version("numpy"),
            },
        },
    }
    result_path = args.output_dir / "arm_g_causal_result.json"
    atomic_json(result_path, report)
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "decision": decision,
                "decision_reasons": reasons,
                "primary": report["primary"],
                "baseline": report["baseline"],
                "sample_counts": report["sample_counts"],
            },
            indent=2,
        )
    )
    print(f"full artifact: {result_path}")


if __name__ == "__main__":
    main()
