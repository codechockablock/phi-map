"""Does the single-neuron collapse ceiling transfer to probe directions?

*Leverage Is Not Reach* (arXiv 2606.19831) reports that single-neuron steering
collapses at a dose `t* = m* B`, where `B = ||r_L|| / ||v||` is the dose at which
the write's magnitude reaches the residual's, and `m*` is an order-one constant
set per architecture family by the participation ratio of the residual: 1.46 for
Llama-3.1-8B, 1.37-1.65 across Qwen and Llama, 0.31 for near-rank-one Gemma.
Rescaling by `m*` is reported to collapse every architecture's coherence curve
onto one cliff.

The paper states its scope as single-neuron interventions, K=1, writing along an
FFN down-projection column, and names extension to residual-stream steering
vectors as unvalidated future work. This protocol runs that extension.

If every direction is normalized to unit length, the budget-normalized dose is
just the write-to-residual ratio:

    h <- h + t * ||r_L|| * u,    ||u|| = 1   =>   ||dh|| / ||r_L|| = t

so `t` is directly comparable across intervention classes, and collapse is
predicted at `t ~ 1.46` regardless of which unit direction is written.

Four direction classes at matched dose, which is the whole design:

    ffn_column   a down-projection column, normalized -- their setting, and the
                 positive control. If this does not collapse near 1.46 in this
                 harness, any difference for the other classes is a protocol
                 difference and not a class difference.
    probe_scope  the orthogonalized Arm G scope direction (seed 112), a
                 difference-of-means probe direction -- the case under test
    random       matched-norm random unit directions -- the null

Not included, and it should be: the un-orthogonalized condition direction, which
section 15 found ~93% aligned with a catalog-position axis. The re-extraction
artifact stored only the selected vector, so recovering it means re-running that
protocol with all five recipes serialized. If the ceiling turns out to be
direction-dependent, that comparison becomes the obvious follow-up, because it
holds the construct fixed and varies only how much position is mixed in.

Collapse is read forward-only from the next-token distribution on generic text:
entropy, KL from the undosed baseline, and top-1 mass. The collapse dose is taken
at half-max on entropy between baseline and saturation, which needs no absolute
threshold and so cannot be tuned to a preferred answer.

No scope task, no rollout, no behavioural outcome. Arm G's dependent variable is
catalog-position determined (section 14) and its dose ladder never exceeded a
third of the single-layer budget (section 16); neither problem touches this,
because collapse is a property of the intervention and the model.
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

from arm_g_causal_subspace import EXPECTED_MODEL_COMMIT
from arm_g_phase1 import (
    TARGET_MODEL,
    atomic_json,
    load_acting_model,
    package_version,
    prompt_token_ids,
    require_gpu,
    unit,
    verify_or_write,
)

# Their reported value for this model, and the band across Qwen/Llama.
PREDICTED_M_STAR = 1.46
REPORTED_BAND = (1.37, 1.65)

DOSES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
SWEEP_LAYERS = (16, 20)
RANDOM_DIRECTIONS = 2
FFN_NEURONS = 2
DIRECTION_SEED = 113

REEXTRACT_ARTIFACT = Path(
    "results/arm_g_reextract_seed112_v1/arm_g_reextract_result.json.gz.b64"
)

# Generic continuation prompts. Deliberately not Arm G scenarios: collapse is a
# property of the intervention and the model, and using the task would reimport
# the confound the rest of this repo spent the day removing.
GENERIC_PROMPTS = (
    "Write a short paragraph explaining how a bicycle gear system works.",
    "Summarize the causes of the 1889 Johnstown flood.",
    "Explain the difference between weather and climate.",
    "Describe how sourdough starter is maintained.",
    "What are the main stages of the water cycle?",
    "Give a brief history of the shipping container.",
    "Explain why leaves change colour in autumn.",
    "Describe the rules of chess castling.",
    "How does noise-cancelling headphone technology work?",
    "Summarize the plot structure of a three-act play.",
    "Explain what a compiler does, in plain terms.",
    "Describe the lifecycle of a monarch butterfly.",
    "What distinguishes igneous from sedimentary rock?",
    "Explain how tides are generated.",
    "Give a short account of the invention of the printing press.",
    "Describe how a heat pump differs from a furnace.",
    "Explain the concept of compound interest.",
    "What is the function of the pancreas?",
    "Describe how GPS determines position.",
    "Summarize how vaccines produce immunity.",
    "Explain the difference between mass and weight.",
    "Describe the process of making cast iron.",
    "What causes the northern lights?",
    "Explain how a sailboat can sail upwind.",
    "Describe the structure of a haiku.",
    "How do noise levels get measured in decibels?",
    "Explain what happens during a solar eclipse.",
    "Describe the basic operation of a refrigerator.",
    "What are the main sections of an orchestra?",
    "Explain how bridges handle thermal expansion.",
    "Describe how coffee is decaffeinated.",
    "Summarize why the sky appears blue.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/arm_g_ceiling")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--per-row-norm",
        action="store_true",
        help="scale the dose by each row's own residual norm instead of the "
        "corpus constant; their protocol measures ||r_L|| once on generic text",
    )
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def make_write_hook(
    direction: torch.Tensor,
    magnitude: torch.Tensor | float,
) -> Any:
    """Add `magnitude * direction` at every token position.

    `magnitude` is either a scalar or one value per row in the batch, which is
    what lets the per-row-norm variant reuse the same hook.
    """

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        scale = magnitude
        if torch.is_tensor(scale) and scale.dim() == 1:
            scale = scale.view(-1, 1, 1)
        shifted = hidden + scale * direction
        if isinstance(output, tuple):
            return (shifted.to(hidden.dtype),) + output[1:]
        return shifted.to(hidden.dtype)

    return hook


def coherence_from_logits(
    logits: torch.Tensor,
    baseline_log_probs: torch.Tensor | None,
) -> dict[str, np.ndarray]:
    """Forward-only readout of whether the next-token distribution has degenerated."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top1 = probs.max(dim=-1).values
    result = {
        "entropy": entropy.cpu().numpy(),
        "top1_probability": top1.cpu().numpy(),
        "argmax": logits.argmax(dim=-1).cpu().numpy(),
    }
    if baseline_log_probs is not None:
        reference = baseline_log_probs.exp()
        result["kl_from_baseline"] = (
            (reference * (baseline_log_probs - log_probs)).sum(dim=-1).cpu().numpy()
        )
    return result


def half_max_dose(
    doses: Sequence[float],
    values: Sequence[float],
) -> float | None:
    """Dose at which `values` first reaches halfway from baseline to maximum.

    Linear interpolation between bracketing doses. Returns None if the curve
    never rises, which is the honest answer for a direction that does not
    collapse in the swept range.
    """
    dose_array = np.asarray(doses, dtype=float)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(dose_array)
    dose_array, value_array = dose_array[order], value_array[order]
    baseline, peak = float(value_array[0]), float(value_array.max())
    if peak - baseline < 1e-9:
        return None
    midpoint = baseline + 0.5 * (peak - baseline)
    for index in range(1, len(dose_array)):
        if value_array[index] >= midpoint:
            low, high = value_array[index - 1], value_array[index]
            if high == low:
                return float(dose_array[index])
            fraction = (midpoint - low) / (high - low)
            return float(
                dose_array[index - 1]
                + fraction * (dose_array[index] - dose_array[index - 1])
            )
    return None


def verdict(
    collapse_doses: Mapping[str, float | None],
) -> tuple[str, list[str]]:
    """Does the ceiling transfer from the FFN column to probe directions?"""
    reasons: list[str] = []
    control = collapse_doses.get("ffn_column")
    if control is None:
        reasons.append(
            "the FFN-column positive control did not collapse in the swept "
            "range, so this harness does not reproduce their setting and no "
            "comparison across intervention classes is licensed"
        )
        return "PROTOCOL_NOT_REPRODUCED", reasons
    low, high = REPORTED_BAND
    if not low <= control <= high:
        reasons.append(
            f"the FFN-column control collapses at {control:.2f}, outside the "
            f"reported {low}-{high} band, so the discrepancy is in the protocol "
            "and not in the intervention class"
        )
        return "PROTOCOL_NOT_REPRODUCED", reasons

    probe = collapse_doses.get("probe_scope")
    if probe is None:
        reasons.append(
            "the probe direction did not collapse in the swept range while the "
            "FFN column did, so the ceiling does not transfer as stated"
        )
        return "CEILING_DOES_NOT_TRANSFER", reasons
    if abs(probe - control) <= 0.25:
        reasons.append(
            f"probe direction collapses at {probe:.2f} against {control:.2f} for "
            "the FFN column, within a quarter budget unit"
        )
        return "CEILING_TRANSFERS", reasons
    reasons.append(
        f"probe direction collapses at {probe:.2f} against {control:.2f} for the "
        "FFN column, a gap of more than a quarter budget unit, so the collapse "
        "dose is not a property of the residual geometry alone"
    )
    return "CEILING_IS_DIRECTION_DEPENDENT", reasons


def self_test() -> None:
    # 1. With a unit direction, budget-normalized dose IS the write ratio.
    generator = np.random.default_rng(0)
    direction = unit(generator.normal(size=64))
    residual = generator.normal(size=64) * 3.0
    norm = float(np.linalg.norm(residual))
    for dose in (0.5, 1.46, 3.0):
        shift = dose * norm * direction
        ratio = float(np.linalg.norm(shift) / norm)
        if abs(ratio - dose) > 1e-9:
            raise AssertionError(f"dose {dose} gave write ratio {ratio}")

    # 2. The hook adds at every position and leaves dtype alone.
    hidden = torch.zeros(3, 5, 64)
    tensor = torch.tensor(direction, dtype=torch.float32)
    shifted = make_write_hook(tensor, 2.0)(None, None, (hidden,))[0]
    if not torch.allclose(shifted[:, 0, :], shifted[:, -1, :]):
        raise AssertionError("hook must write at every position")
    if abs(float(shifted[0, 0, :].norm()) - 2.0) > 1e-5:
        raise AssertionError("hook wrote the wrong magnitude")
    per_row = make_write_hook(tensor, torch.tensor([1.0, 2.0, 3.0]))(
        None, None, (hidden,)
    )[0]
    for index, expected in enumerate((1.0, 2.0, 3.0)):
        if abs(float(per_row[index, 0, :].norm()) - expected) > 1e-5:
            raise AssertionError("per-row magnitudes not applied rowwise")

    # 3. Half-max recovers a known cliff, and reports None when nothing happens.
    doses = np.array(DOSES)
    cliff = 1.46
    curve = 0.2 + 9.0 / (1.0 + np.exp(-12.0 * (doses - cliff)))
    recovered = half_max_dose(doses, curve)
    if recovered is None or abs(recovered - cliff) > 0.1:
        raise AssertionError(f"half-max recovered {recovered}, wanted ~{cliff}")
    if half_max_dose(doses, np.full(len(doses), 0.2)) is not None:
        raise AssertionError("a flat curve has no collapse dose")

    # 4. Every verdict branch.
    cases = {
        "CEILING_TRANSFERS": {"ffn_column": 1.46, "probe_scope": 1.52},
        "CEILING_IS_DIRECTION_DEPENDENT": {"ffn_column": 1.46, "probe_scope": 2.40},
        "CEILING_DOES_NOT_TRANSFER": {"ffn_column": 1.46, "probe_scope": None},
        "PROTOCOL_NOT_REPRODUCED": {"ffn_column": None, "probe_scope": 1.5},
    }
    for expected, collapse in cases.items():
        got, reasons = verdict(collapse)
        if got != expected or not reasons:
            raise AssertionError(f"expected {expected}, got {got} / {reasons}")
    if verdict({"ffn_column": 0.4, "probe_scope": 0.4})[0] != "PROTOCOL_NOT_REPRODUCED":
        raise AssertionError("a control outside the reported band must abort")

    # 5. Arm G section 16 arithmetic, so the framing cannot silently drift.
    layer16_norm = 8.991
    if abs(3.0 * 0.4979 / layer16_norm - 0.1661) > 1e-3:
        raise AssertionError("k=3 removal budget ratio changed")
    if abs(8.0 * 0.5506 / layer16_norm - 0.4899) > 1e-3:
        raise AssertionError("c=8 sigma budget ratio changed")
    predicted = PREDICTED_M_STAR * layer16_norm / 0.5506
    if abs(predicted - 23.8) > 0.5:
        raise AssertionError(f"predicted collapse dose drifted to {predicted}")

    print(
        json.dumps(
            {
                "self_test": "PASS",
                "budget_dose_equals_write_ratio": True,
                "half_max_recovered": round(float(recovered), 3),
                "arm_g_in_budget_units": {
                    "removal_k3_32_layers": 0.1661,
                    "addition_c8_sigma_1_layer": 0.4899,
                    "predicted_collapse_sigma": round(predicted, 1),
                },
            },
            indent=2,
        )
    )


def load_probe_directions() -> dict[str, np.ndarray]:
    import base64
    import gzip

    if not REEXTRACT_ARTIFACT.exists():
        raise RuntimeError(f"missing {REEXTRACT_ARTIFACT}; run arm_g_reextract first")
    payload = json.loads(
        gzip.decompress(base64.b64decode(REEXTRACT_ARTIFACT.read_text()))
    )
    selected = payload.get("selected_direction")
    if selected is None:
        raise RuntimeError("re-extraction artifact has no selected direction")
    return {"probe_scope": unit(np.asarray(selected, dtype=np.float64))}


@torch.inference_mode()
def sweep(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[torch.Tensor],
    layer: int,
    directions: Mapping[str, np.ndarray],
    residual_norm: float,
    row_norms: np.ndarray,
    batch_size: int,
    per_row_norm: bool,
) -> dict[str, Any]:
    from arm_g_causal import pad_prompt_batch

    modules = model.model.layers
    results: dict[str, Any] = {}
    baseline_log_probs: list[torch.Tensor] = []

    for name, vector in [("__baseline__", None), *directions.items()]:
        tensor = (
            None
            if vector is None
            else torch.tensor(vector, dtype=torch.float32, device="cuda")
        )
        for dose in DOSES if vector is not None else (0.0,):
            chunks: dict[str, list[np.ndarray]] = {}
            for start in range(0, len(prompts), batch_size):
                stop = min(len(prompts), start + batch_size)
                input_ids, attention_mask = pad_prompt_batch(
                    prompts[start:stop], tokenizer.pad_token_id
                )
                handle = None
                if tensor is not None and dose != 0.0:
                    magnitude: Any = dose * residual_norm
                    if per_row_norm:
                        magnitude = torch.tensor(
                            dose * row_norms[start:stop],
                            dtype=torch.float32,
                            device="cuda",
                        )
                    handle = modules[layer - 1].register_forward_hook(
                        make_write_hook(tensor, magnitude)
                    )
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
                logits = output.logits[:, -1, :]
                index = start // batch_size
                reference = (
                    baseline_log_probs[index]
                    if baseline_log_probs and vector is not None
                    else None
                )
                if vector is None:
                    baseline_log_probs.append(
                        torch.log_softmax(logits.float(), dim=-1).clone()
                    )
                    reference = None
                metrics = coherence_from_logits(logits, reference)
                for key, values in metrics.items():
                    chunks.setdefault(key, []).append(values)
            merged = {key: np.concatenate(v) for key, v in chunks.items()}
            label = "baseline" if vector is None else f"{name}@{dose}"
            entry = {
                "mean_entropy": float(merged["entropy"].mean()),
                "mean_top1_probability": float(merged["top1_probability"].mean()),
                "mean_kl_from_baseline": (
                    float(merged["kl_from_baseline"].mean())
                    if "kl_from_baseline" in merged
                    else 0.0
                ),
            }
            # Dose zero must reproduce the baseline exactly. If it does not, the
            # hook or the baseline indexing is wrong, and every curve below it
            # is meaningless.
            if vector is not None and dose == 0.0:
                drift = abs(entry["mean_entropy"] - results["baseline"]["mean_entropy"])
                if drift > 1e-4:
                    raise RuntimeError(
                        f"dose 0 for {name} drifted from baseline by {drift}"
                    )
            results[label] = entry
    return results


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    config = {
        "protocol": "ARM_G_CEILING_V1",
        "model": args.model,
        "question": (
            "does the single-neuron collapse ceiling of arXiv 2606.19831 transfer "
            "to residual-stream difference-of-means probe directions"
        ),
        "predicted_m_star": PREDICTED_M_STAR,
        "reported_band": list(REPORTED_BAND),
        "doses_in_budget_units": list(DOSES),
        "layers": list(SWEEP_LAYERS),
        "budget_definition": (
            "B = ||r_L|| / ||v||; with every direction normalized to unit length "
            "the budget-normalized dose t equals the write-to-residual ratio "
            "||dh|| / ||r_L||, so it is comparable across intervention classes"
        ),
        "collapse_criterion": (
            "half-max on mean next-token entropy between baseline and saturation, "
            "which needs no absolute threshold"
        ),
        "positive_control": (
            "a normalized FFN down-projection column, their setting. If it does "
            "not collapse inside the reported band this harness has not "
            "reproduced their protocol and no class comparison is licensed"
        ),
        "why_no_scope_task": (
            "collapse is a property of the intervention and the model. Arm G's "
            "dependent variable is catalog-position determined (section 14) and "
            "its dose ladder never exceeded a third of the single-layer budget "
            "(section 16); neither touches this measurement"
        ),
        "prior_art_check": (
            "the companion paper named in 2606.19831 extends to model-scale "
            "screening across neurons, still K=1, and is not out. The angle-norm "
            "decomposition (2606.06735) treats norm as a stability lever and "
            "defines no collapse edge. Recorded as not found, not as not there"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_or_write(args.output_dir / "run_config.json", config)

    hardware = require_gpu(args.allow_non_a100)
    tokenizer, model = load_acting_model(args.model, args.hf_token)
    model_commit = getattr(model.config, "_commit_hash", None)
    if model_commit not in (None, EXPECTED_MODEL_COMMIT):
        raise RuntimeError(f"model commit {model_commit} differs from frozen")

    prompts = [
        prompt_token_ids(tokenizer, [{"role": "user", "content": text}])
        for text in GENERIC_PROMPTS
    ]

    generator = np.random.default_rng(DIRECTION_SEED)
    probe = load_probe_directions()
    report: dict[str, Any] = {
        "status": "ARM_G_CEILING_V1",
        "config": config,
        "by_layer": {},
    }

    for layer in SWEEP_LAYERS:
        hidden_size = model.config.hidden_size
        down_proj = model.model.layers[layer - 1].mlp.down_proj.weight
        columns = generator.choice(down_proj.shape[1], size=FFN_NEURONS, replace=False)
        directions: dict[str, np.ndarray] = {
            "ffn_column": unit(
                down_proj[:, int(columns[0])].detach().float().cpu().numpy()
            ),
        }
        for extra, column in enumerate(columns[1:], start=1):
            directions[f"ffn_column_{extra}"] = unit(
                down_proj[:, int(column)].detach().float().cpu().numpy()
            )
        directions.update(probe)
        for index in range(RANDOM_DIRECTIONS):
            directions[f"random_{index}"] = unit(generator.normal(size=hidden_size))

        row_norms = measure_residual_norm(
            model, tokenizer, prompts, layer, args.batch_size
        )
        residual_norm = float(row_norms.mean())
        raw = sweep(
            model,
            tokenizer,
            prompts,
            layer,
            directions,
            residual_norm,
            row_norms,
            args.batch_size,
            args.per_row_norm,
        )
        collapse = {}
        for name in directions:
            curve = [raw[f"{name}@{dose}"]["mean_entropy"] for dose in DOSES]
            collapse[name] = half_max_dose(DOSES, curve)
        decision, reasons = verdict(collapse)
        report["by_layer"][str(layer)] = {
            "residual_norm": residual_norm,
            "curves": raw,
            "collapse_dose": collapse,
            "decision": decision,
            "decision_reasons": reasons,
        }

    report["provenance"] = {
        "acting_model": args.model,
        "expected_model_commit": EXPECTED_MODEL_COMMIT,
        "model_commit": model_commit,
        "dtype": "bfloat16",
        "hardware": hardware,
        "package_versions": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "numpy": package_version("numpy"),
        },
    }
    result_path = args.output_dir / "arm_g_ceiling_result.json"
    atomic_json(result_path, report)
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))
    print(f"full artifact: {result_path}")


@torch.inference_mode()
def measure_residual_norm(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[torch.Tensor],
    layer: int,
    batch_size: int,
) -> np.ndarray:
    """Residual norm at the read position per row; their `||r_L||` is its mean."""
    from arm_g_causal import pad_prompt_batch

    captured: list[float] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.extend(hidden[:, -1, :].float().norm(dim=-1).cpu().numpy().tolist())

    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        for start in range(0, len(prompts), batch_size):
            stop = min(len(prompts), start + batch_size)
            input_ids, attention_mask = pad_prompt_batch(
                prompts[start:stop], tokenizer.pad_token_id
            )
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    return np.asarray(captured, dtype=np.float64)


if __name__ == "__main__":
    main()
