"""Arm G dose response: does over-removal flip the decision?

Run 1 established that removing the rank-1 layer-16 conflict direction from
every layer and every token position attenuates the condition contrast by 33%
and moves the deciding rows to within 0.125 of the decision boundary without
crossing it, with coherence intact.  The remaining question is whether the gap
is a property of the variable or of the dose.

The intervention is scaled removal, x <- x - k (x . r) r, at all layers and all
positions.  k = 1 is the run-1 condition.  k = 2 reflects the component through
zero; larger k drives it negative.

The frozen primary is whether some dose drives the conflict-condition
decline rate below 0.25 from a baseline of 0.50 while the model stays coherent.
Coherence is not judged by a fixed threshold alone: every dose is matched by
random directions carried through the identical scaling, so a flip that random
directions also produce at the same dose is a statement about perturbation
size, not about the variable.  Greedy continuations are sampled at every dose
so that "coherent" can be inspected as text rather than only as a statistic.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from functools import partial
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
    per_pair_condition_contrasts,
    semantic_margins_from_logits,
    stratified_pair_bootstrap_values,
)
from arm_g_causal_subspace import (
    EXPECTED_MODEL_COMMIT,
    SOURCE_EVIDENCE,
    attenuation_slices,
    contrast_slices,
    paired_differences,
    source_rows,
)
from arm_g_cross_layer import run_forward
from arm_g_phase1 import (
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
from arm_g_scenarios import build_manifest, validate_manifest

DEFAULT_OUTPUT_DIR = Path("/content/arm-g-dose-seed109-v1")
SOURCE_SEEDS = (101, 102)
RANK1_SEED = 102
EVAL_SEED = 109
DIRECTION_LAYER = 16
DOSES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
UNIT_DOSE = 1.0
FLIP_THRESHOLD = 0.25
ACTION_MASS_GATE = 0.90
TOP_TOKEN_GATE = 0.95
GENERATION_SAMPLES = 8
GENERATION_TOKENS = 16
PRIOR_EVIDENCE = {
    "seed108_all_layers_all_positions_attenuation": 1.58984375,
    "seed108_all_layers_all_positions_flips": 0,
    "seed108_positive_conflict_rows_min_margin_after": 0.125,
    "seed108_baseline_conflict_decline_rate": 0.5,
    "seed108_random_allpos_attenuation_p95": 0.6097656249999996,
    "source": "results/arm_g_allpos_seed108_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairs-per-family", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-directions", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token. Prefer the HF_TOKEN environment variable.",
    )
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def make_scaled_hook(direction: torch.Tensor, scale: float) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        coefficient = hidden.float() @ direction
        adjustment = (scale * coefficient).unsqueeze(-1) * direction
        return replace_hidden(output, hidden - adjustment.to(dtype=hidden.dtype))

    return hook


def register(model: Any, direction: np.ndarray | None, scale: float) -> list[Any]:
    if direction is None:
        return []
    modules = getattr(getattr(model, "model", None), "layers", None)
    if modules is None:
        raise RuntimeError("could not locate model.model.layers")
    vector = torch.tensor(direction, dtype=torch.float32, device="cuda")
    return [
        module.register_forward_hook(make_scaled_hook(vector, scale))
        for module in modules
    ]


@torch.inference_mode()
def score_dose(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    direction: np.ndarray | None,
    scale: float,
    batch_size: int,
    token_ids: Mapping[str, int],
) -> dict[str, Any]:
    prompts = [prompt_token_ids(tokenizer, row["messages"]) for row in rows]
    margins, action_mass, top_is_action, entropies, logprobs = [], [], [], [], []
    for start in range(0, len(rows), batch_size):
        stop = min(len(rows), start + batch_size)
        input_ids, attention_mask = pad_prompt_batch(
            prompts[start:stop], tokenizer.pad_token_id
        )
        handles = register(model, direction, scale)
        try:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        logits = output.logits[:, -1, :].float()
        margins.append(
            semantic_margins_from_logits(logits, rows[start:stop], token_ids)
        )
        probabilities = torch.softmax(logits, dim=-1)
        action_ids = torch.tensor(
            [token_ids[token] for token in ACTION_TOKENS], device=logits.device
        )
        action_mass.append(probabilities[:, action_ids].sum(dim=-1).cpu().numpy())
        top_is_action.append(
            torch.isin(logits.argmax(dim=-1), action_ids).cpu().numpy()
        )
        entropies.append(
            (-(probabilities * torch.log(probabilities + 1e-12)).sum(dim=-1))
            .cpu()
            .numpy()
        )
        logprobs.append(torch.log_softmax(logits, dim=-1).cpu().numpy())
    return {
        "margins": np.concatenate(margins),
        "action_probability_mass": np.concatenate(action_mass),
        "top_token_is_action": np.concatenate(top_is_action),
        "next_token_entropy": np.concatenate(entropies),
        "log_probabilities": np.concatenate(logprobs),
    }


@torch.inference_mode()
def sample_continuations(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    direction: np.ndarray | None,
    scale: float,
) -> list[dict[str, str]]:
    """Greedy continuations so that coherence can be read, not only measured."""
    samples = []
    for row in rows[:GENERATION_SAMPLES]:
        prompt = prompt_token_ids(tokenizer, row["messages"]).unsqueeze(0).to("cuda")
        handles = register(model, direction, scale)
        try:
            generated = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=GENERATION_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        finally:
            for handle in handles:
                handle.remove()
        text = tokenizer.decode(
            generated[0, prompt.shape[1] :], skip_special_tokens=True
        )
        samples.append(
            {
                "pair_id": str(row["pair_id"]),
                "condition": str(row["condition"]),
                "mapping_variant": str(row["mapping_variant"]),
                "continuation": text,
            }
        )
    return samples


def summarize(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    baseline_pairs: Mapping[tuple[str, str], float],
    baseline_slices: Mapping[str, Any],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    margins = result["margins"]
    labels = np.array([row["condition_label"] for row in rows])
    pairs = per_pair_condition_contrasts(margins, rows)
    values = {key: baseline_pairs[key] - pairs[key] for key in baseline_pairs}
    slices = contrast_slices(margins, rows)
    attenuation = attenuation_slices(baseline_slices, slices)
    reference = baseline["log_probabilities"]
    current = result["log_probabilities"]
    coherent = bool(
        result["action_probability_mass"].mean() >= ACTION_MASS_GATE
        and result["top_token_is_action"].mean() >= TOP_TOKEN_GATE
    )
    return {
        "attenuation": attenuation,
        "attenuation_fraction": float(
            attenuation["overall"] / baseline_slices["overall"]
        ),
        "attenuation_bootstrap": stratified_pair_bootstrap_values(
            values, bootstrap, seed
        ),
        "choice_summary": semantic_choice_summary(margins, rows),
        "conflict_decline_rate": float(np.mean(margins[labels == 1] > 0)),
        "reachable_decline_rate": float(np.mean(margins[labels == 0] > 0)),
        "decision_flips": int(np.sum(np.sign(baseline["margins"]) != np.sign(margins))),
        "conflict_row_mean_shift": float(
            margins[labels == 1].mean() - baseline["margins"][labels == 1].mean()
        ),
        "reachable_row_mean_shift": float(
            margins[labels == 0].mean() - baseline["margins"][labels == 0].mean()
        ),
        "coherence": {
            "mean_action_probability_mass": float(
                result["action_probability_mass"].mean()
            ),
            "top_token_is_action_rate": float(result["top_token_is_action"].mean()),
            "mean_next_token_entropy": float(result["next_token_entropy"].mean()),
            "kl_from_baseline": float(
                np.mean(np.sum(np.exp(reference) * (reference - current), axis=1))
            ),
            "passes_coherence_gate": coherent,
        },
    }


def dose_decision(
    target: Mapping[float, Mapping[str, Any]],
    random_by_dose: Mapping[float, Sequence[Mapping[str, Any]]],
) -> tuple[str, list[str], dict[str, Any]]:
    reasons = []
    qualifying = []
    for dose in sorted(target):
        entry = target[dose]
        if entry["conflict_decline_rate"] >= FLIP_THRESHOLD:
            continue
        controls = random_by_dose.get(dose, [])
        worst_control = min((c["conflict_decline_rate"] for c in controls), default=1.0)
        qualifying.append(
            {
                "dose": dose,
                "conflict_decline_rate": entry["conflict_decline_rate"],
                "coherent": entry["coherence"]["passes_coherence_gate"],
                "best_random_conflict_decline_rate": worst_control,
                "beats_matched_random": bool(
                    entry["conflict_decline_rate"] < worst_control
                ),
            }
        )
    coherent_and_specific = [
        q for q in qualifying if q["coherent"] and q["beats_matched_random"]
    ]
    detail = {
        "qualifying_doses": qualifying,
        "first_coherent_specific_dose": (
            coherent_and_specific[0]["dose"] if coherent_and_specific else None
        ),
    }
    if coherent_and_specific:
        return "DOSE_FLIPS_DECISIONS_WITH_COHERENCE", [], detail
    if qualifying:
        reasons.append(
            "decline rate crossed the threshold only without coherence or "
            "without beating matched-dose random directions"
        )
        return "DOSE_FLIPS_ONLY_WITH_DEGRADATION", reasons, detail
    reasons.append("no dose drove the conflict decline rate below the threshold")
    return "DOSE_DOES_NOT_FLIP_DECISIONS", reasons, detail


def self_test() -> None:
    rng = np.random.default_rng(EVAL_SEED)
    width, batch, positions = 12, 3, 4
    direction = unit(rng.normal(size=width))
    tensor = torch.tensor(direction, dtype=torch.float32)
    hidden = torch.tensor(
        rng.normal(size=(batch, positions, width)), dtype=torch.float32
    )

    class Layer(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return (x, None)

    before = (hidden @ tensor).clone()
    for scale, expected in ((1.0, 0.0), (2.0, -1.0), (0.5, 0.5)):
        module = Layer()
        handle = module.register_forward_hook(make_scaled_hook(tensor, scale))
        try:
            out = module(hidden)[0]
        finally:
            handle.remove()
        after = out @ tensor
        target = expected * before
        if float(torch.max(torch.abs(after - target))) > 1e-4:
            raise AssertionError(f"scale {scale} did not map projection correctly")

    rows = []
    for family in ("data_checksums", "incident_times"):
        for pair_index in range(8):
            for label in (0, 1):
                for mapping in MAPPING_VARIANTS:
                    rows.append(
                        {
                            "family": family,
                            "pair_id": f"{family}:{pair_index:03d}",
                            "condition": "conflict" if label else "reachable",
                            "condition_label": label,
                            "mapping_variant": mapping,
                        }
                    )
    labels = np.array([r["condition_label"] for r in rows])
    base = np.where(labels == 1, 3.0, -3.0).astype(float)
    baseline = {
        "margins": base,
        "log_probabilities": np.log(np.full((len(rows), 4), 0.25)),
    }
    baseline_pairs = per_pair_condition_contrasts(base, rows)
    baseline_slices = contrast_slices(base, rows)

    def make(margins: np.ndarray, mass: float) -> dict[str, Any]:
        return {
            "margins": margins,
            "action_probability_mass": np.full(len(rows), mass),
            "top_token_is_action": np.ones(len(rows), dtype=bool),
            "next_token_entropy": np.full(len(rows), 0.4),
            "log_probabilities": np.log(np.full((len(rows), 4), 0.25)),
        }

    flipped = np.where(labels == 1, -1.0, -3.0)
    target = {
        1.0: summarize(
            make(base, 0.99), rows, baseline, baseline_pairs, baseline_slices, 200, 1
        ),
        2.0: summarize(
            make(flipped, 0.99),
            rows,
            baseline,
            baseline_pairs,
            baseline_slices,
            200,
            2,
        ),
    }
    if target[2.0]["conflict_decline_rate"] != 0.0:
        raise AssertionError("flip detection failed")
    controls = {1.0: [target[1.0]], 2.0: [target[1.0]]}
    decision, reasons, detail = dose_decision(target, controls)
    if decision != "DOSE_FLIPS_DECISIONS_WITH_COHERENCE" or reasons:
        raise AssertionError("decision rule rejected a clean synthetic positive")
    if detail["first_coherent_specific_dose"] != 2.0:
        raise AssertionError("wrong first qualifying dose")

    degraded = {
        1.0: target[1.0],
        2.0: summarize(
            make(flipped, 0.10),
            rows,
            baseline,
            baseline_pairs,
            baseline_slices,
            200,
            3,
        ),
    }
    if dose_decision(degraded, controls)[0] != "DOSE_FLIPS_ONLY_WITH_DEGRADATION":
        raise AssertionError("decision rule missed an incoherent flip")
    matched = {1.0: [target[1.0]], 2.0: [target[2.0]]}
    if dose_decision(target, matched)[0] != "DOSE_FLIPS_ONLY_WITH_DEGRADATION":
        raise AssertionError("decision rule ignored a matched-dose random flip")
    if dose_decision({1.0: target[1.0]}, controls)[0] != "DOSE_DOES_NOT_FLIP_DECISIONS":
        raise AssertionError("decision rule mislabeled a null result")

    print(
        json.dumps(
            {
                "self_test": "PASS",
                "doses": list(DOSES),
                "flip_threshold": FLIP_THRESHOLD,
                "scaling_checked": [1.0, 2.0, 0.5],
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if UNIT_DOSE not in DOSES:
        raise RuntimeError("the frozen dose grid must contain the unit dose")

    all_source_rows = []
    source_audits = {}
    for seed in SOURCE_SEEDS:
        rows, audit = source_rows(args.pairs_per_family, seed)
        all_source_rows.extend(rows)
        source_audits[str(seed)] = audit

    eval_manifest = build_manifest(
        pairs_per_family=args.pairs_per_family, repeats=1, seed=EVAL_SEED
    )
    eval_audit = validate_manifest(
        eval_manifest, pairs_per_family=args.pairs_per_family, repeats=1
    )
    eval_rows = build_eval_rows(eval_manifest)

    config = {
        "protocol": "ARM_G_DOSE_V1",
        "model": args.model,
        "source_seeds": list(SOURCE_SEEDS),
        "eval_seed": EVAL_SEED,
        "direction_layer": DIRECTION_LAYER,
        "direction": "rank-1 seed-102 mean paired conflict difference",
        "intervention": "x <- x - k (x . r) r at all layers and all positions",
        "doses": list(DOSES),
        "prior_evidence": PRIOR_EVIDENCE,
        "primary_question": (
            "does some dose drive the conflict-condition decline rate below "
            f"{FLIP_THRESHOLD} from a baseline of 0.50 while the model stays "
            "coherent and while matched-dose random directions do not"
        ),
        "coherence_gate": {
            "mean_action_probability_mass": ACTION_MASS_GATE,
            "top_token_is_action_rate": TOP_TOKEN_GATE,
        },
        "specificity": (
            f"{args.random_directions} random directions orthogonal to the "
            "conflict direction are carried through every dose identically"
        ),
        "generation_probe": {
            "samples": GENERATION_SAMPLES,
            "max_new_tokens": GENERATION_TOKENS,
            "decoding": "greedy",
        },
        "bootstrap": args.bootstrap,
        "pairs_per_family": args.pairs_per_family,
        "read_position": PRIMARY_POSITION,
        "source_evidence": SOURCE_EVIDENCE,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_or_write(args.output_dir / "run_config.json", config)
    verify_or_write(args.output_dir / "eval_manifest.json", eval_manifest)

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
        model,
        tokenizer,
        all_source_rows,
        args.batch_size,
        {},
        (DIRECTION_LAYER,),
        None,
    )
    differences, metadata = paired_differences(
        source_states[DIRECTION_LAYER], all_source_rows
    )
    selected = [
        index
        for index, item in enumerate(metadata)
        if int(item["source_seed"]) == RANK1_SEED
    ]
    conflict = unit(differences[selected].mean(axis=0))

    score = partial(score_dose, model, tokenizer)
    baseline = score(eval_rows, None, 0.0, args.batch_size, token_ids)
    baseline_pairs = per_pair_condition_contrasts(baseline["margins"], eval_rows)
    baseline_slices = contrast_slices(baseline["margins"], eval_rows)

    target = {}
    generations = {
        "baseline": sample_continuations(model, tokenizer, eval_rows, None, 0.0)
    }
    dose_margins = {}
    for index, dose in enumerate(DOSES):
        result = score(eval_rows, conflict, dose, args.batch_size, token_ids)
        dose_margins[dose] = result["margins"]
        target[dose] = summarize(
            result,
            eval_rows,
            baseline,
            baseline_pairs,
            baseline_slices,
            args.bootstrap,
            EVAL_SEED + index,
        )
        generations[f"dose_{dose}"] = sample_continuations(
            model, tokenizer, eval_rows, conflict, dose
        )

    rng = np.random.default_rng(EVAL_SEED)
    random_directions = []
    for _index in range(args.random_directions):
        candidate = rng.normal(size=len(conflict))
        candidate -= (candidate @ conflict) * conflict
        random_directions.append(unit(candidate))
    random_by_dose: dict[float, list[dict[str, Any]]] = {}
    for dose in DOSES:
        entries = []
        for control_index, vector in enumerate(random_directions):
            result = score(eval_rows, vector, dose, args.batch_size, token_ids)
            entries.append(
                summarize(
                    result,
                    eval_rows,
                    baseline,
                    baseline_pairs,
                    baseline_slices,
                    args.bootstrap,
                    EVAL_SEED + 500 + control_index,
                )
            )
        random_by_dose[dose] = entries

    decision, decision_reasons, decision_detail = dose_decision(target, random_by_dose)

    def public(entry: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in entry.items()
            if key not in ("attenuation_values",)
        }

    report = {
        "status": "ARM_G_DOSE_V1",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "decision_detail": decision_detail,
        "baseline": {
            "choice_summary": semantic_choice_summary(baseline["margins"], eval_rows),
            "condition_contrast": baseline_slices,
            "coherence": {
                "mean_action_probability_mass": float(
                    baseline["action_probability_mass"].mean()
                ),
                "top_token_is_action_rate": float(
                    baseline["top_token_is_action"].mean()
                ),
                "mean_next_token_entropy": float(baseline["next_token_entropy"].mean()),
            },
        },
        "dose_response": {str(dose): public(target[dose]) for dose in DOSES},
        "random_controls_by_dose": {
            str(dose): {
                "conflict_decline_rates": [
                    entry["conflict_decline_rate"] for entry in random_by_dose[dose]
                ],
                "attenuations": [
                    entry["attenuation"]["overall"] for entry in random_by_dose[dose]
                ],
                "decision_flips": [
                    entry["decision_flips"] for entry in random_by_dose[dose]
                ],
                "coherence_gate_passes": [
                    entry["coherence"]["passes_coherence_gate"]
                    for entry in random_by_dose[dose]
                ],
            }
            for dose in DOSES
        },
        "generations": generations,
        "sample_counts": {
            "source_rows": len(all_source_rows),
            "evaluation_rows": len(eval_rows),
            "independent_evaluation_pairs": int(
                len({row["pair_id"] for row in eval_rows})
            ),
            "doses": len(DOSES),
            "random_directions": args.random_directions,
        },
        "audits": {
            "source_manifests": source_audits,
            "evaluation_manifest": eval_audit,
        },
        "row_results": [
            {
                **{k: v for k, v in row.items() if k != "messages"},
                "baseline_semantic_margin": float(baseline["margins"][index]),
                "margins_by_dose": {
                    str(dose): float(dose_margins[dose][index]) for dose in DOSES
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
            "source_evidence": SOURCE_EVIDENCE,
            "prior_evidence": PRIOR_EVIDENCE,
            "label_source": "mechanical target-path achievability",
            "package_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "numpy": package_version("numpy"),
            },
        },
    }
    result_path = args.output_dir / "arm_g_dose_result.json"
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
                "decision_detail": decision_detail,
                "baseline_conflict_decline_rate": report["baseline"]["choice_summary"][
                    "by_condition"
                ]["conflict"]["decline_choice_rate"],
                "dose_response": {
                    str(dose): {
                        "attenuation": target[dose]["attenuation"]["overall"],
                        "attenuation_fraction": target[dose]["attenuation_fraction"],
                        "conflict_decline_rate": target[dose]["conflict_decline_rate"],
                        "decision_flips": target[dose]["decision_flips"],
                        "conflict_row_mean_shift": target[dose][
                            "conflict_row_mean_shift"
                        ],
                        "reachable_row_mean_shift": target[dose][
                            "reachable_row_mean_shift"
                        ],
                        "coherence": target[dose]["coherence"],
                    }
                    for dose in DOSES
                },
                "random_controls_by_dose": report["random_controls_by_dose"],
                "sample_counts": report["sample_counts"],
            },
            indent=2,
        )
    )
    print(f"full artifact: {result_path}")


if __name__ == "__main__":
    main()
