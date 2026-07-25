"""Arm G boundary geometry: is writability distance, or responsiveness?

Three papers draw opposite conclusions about whether a probe direction is a
control point.  Roy et al. report 0% correction for hallucination in 7 of 7
models; a multi-behaviour study reports hallucination "highly steerable"; Rift
reports deception neither inducible nor correctable.  Our own dose run found
that removing the layer-16 conflict direction moved the deciding rows from a
minimum margin of +2.125 to +0.125 and flipped nothing.

The hypothesis is that these are measurements of *distance to a decision
boundary* rather than of causal role, and that inducing is easier than
correcting because it moves toward mass the model already has.

The naive test is circular.  If every row's margin is linear in the steering
coefficient with a shared slope s, then the dose that flips row i is exactly
-margin_i(0)/s, so regressing dose-to-flip on baseline margin returns R^2 = 1
as arithmetic, not as evidence.  The content is therefore in the *decomposition*

    dose_to_flip_i  =  -baseline_margin_i / responsiveness_i

and specifically in whether responsiveness is homogeneous across rows and
conditions.  Homogeneous responsiveness means writability is boundary distance
and the literature's disagreement is a task-design artifact.  Responsiveness
that varies systematically with condition means some states are intrinsically
harder to move and the asymmetry is structural.

Intervention is signed additive steering x <- x + c*sigma*r at layer 16 across
all token positions, the standard steering convention, where sigma is the
standard deviation of the baseline projection onto r.  Signed doses give induce
and correct directions on one axis.  Matched-dose random directions and a
coherence gate are carried through exactly as in the removal experiments.
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
    build_eval_rows,
    pad_prompt_batch,
    replace_hidden,
    semantic_choice_summary,
)
from arm_g_causal_dose_ablation import semantic_margins_from_logits
from arm_g_causal_subspace import (
    EXPECTED_MODEL_COMMIT,
    SOURCE_EVIDENCE,
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

DEFAULT_OUTPUT_DIR = Path("/content/arm-g-boundary-seed110-v1")
SOURCE_SEEDS = (101, 102)
RANK1_SEED = 102
EVAL_SEED = 110
DIRECTION_LAYER = 16
DOSES = (
    -8.0,
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
)
RANDOM_DOSES = (-8.0, -4.0, -1.0, 1.0, 4.0, 8.0)
ACTION_MASS_GATE = 0.90
TOP_TOKEN_GATE = 0.95
FLIP_COVERAGE_GATE = 0.50
R2_GATE = 0.90
PRIOR_EVIDENCE = {
    "seed108_positive_conflict_rows_baseline_min": 2.125,
    "seed108_positive_conflict_rows_after_removal_min": 0.125,
    "seed109_saturation_dose": 1.5,
    "seed109_flips_at_max_coherent_dose": 6,
    "source": "results/arm_g_allpos_seed108_v1, results/arm_g_dose_seed109_v1",
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


def make_additive_hook(direction: torch.Tensor, offset: float) -> Any:
    """x <- x + offset * r at every token position."""

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        delta = (offset * direction).to(dtype=hidden.dtype)
        return replace_hidden(output, hidden + delta)

    return hook


@torch.inference_mode()
def score_offset(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    direction: np.ndarray | None,
    offset: float,
    batch_size: int,
    token_ids: Mapping[str, int],
) -> dict[str, Any]:
    prompts = [prompt_token_ids(tokenizer, row["messages"]) for row in rows]
    modules = getattr(getattr(model, "model", None), "layers", None)
    if modules is None:
        raise RuntimeError("could not locate model.model.layers")
    vector = (
        torch.tensor(direction, dtype=torch.float32, device="cuda")
        if direction is not None
        else None
    )
    margins, action_mass, top_is_action, entropies = [], [], [], []
    for start in range(0, len(rows), batch_size):
        stop = min(len(rows), start + batch_size)
        input_ids, attention_mask = pad_prompt_batch(
            prompts[start:stop], tokenizer.pad_token_id
        )
        handle = None
        try:
            if vector is not None and offset != 0.0:
                handle = modules[DIRECTION_LAYER - 1].register_forward_hook(
                    make_additive_hook(vector, offset)
                )
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
    mass = np.concatenate(action_mass)
    top = np.concatenate(top_is_action)
    return {
        "margins": np.concatenate(margins),
        "coherence": {
            "mean_action_probability_mass": float(mass.mean()),
            "top_token_is_action_rate": float(top.mean()),
            "mean_next_token_entropy": float(np.concatenate(entropies).mean()),
            "passes_coherence_gate": bool(
                mass.mean() >= ACTION_MASS_GATE and top.mean() >= TOP_TOKEN_GATE
            ),
        },
    }


def crossing_dose(doses: np.ndarray, margins: np.ndarray) -> float | None:
    """Smallest |dose| at which the margin changes sign, linearly interpolated."""
    order = np.argsort(np.abs(doses))
    best = None
    base_sign = np.sign(margins[np.argmin(np.abs(doses))])
    for index in order:
        if np.sign(margins[index]) == base_sign or margins[index] == 0.0:
            continue
        # Bracket between this dose and the nearest dose on the original side.
        same = [
            j
            for j in range(len(doses))
            if np.sign(margins[j]) == base_sign
            and (doses[j] - doses[index]) * doses[index] <= 0
        ]
        if not same:
            candidate = doses[index]
        else:
            j = min(same, key=lambda j: abs(doses[j] - doses[index]))
            span = margins[index] - margins[j]
            if abs(span) < 1e-12:
                candidate = doses[index]
            else:
                t = -margins[j] / span
                candidate = doses[j] + t * (doses[index] - doses[j])
        if best is None or abs(candidate) < abs(best):
            best = float(candidate)
    return best


def per_row_fit(
    doses: np.ndarray,
    margin_matrix: np.ndarray,
    coherent: np.ndarray,
) -> dict[str, Any]:
    """Slope of margin against dose per row, over coherent doses only."""
    usable = doses[coherent]
    design = np.column_stack([np.ones(len(usable)), usable])
    slopes, intercepts, r2 = [], [], []
    for row in range(margin_matrix.shape[1]):
        y = margin_matrix[coherent, row]
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercepts.append(float(beta[0]))
        slopes.append(float(beta[1]))
        predicted = design @ beta
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))
    return {
        "slopes": slopes,
        "intercepts": intercepts,
        "linear_fit_r2": r2,
        "doses_used": usable.tolist(),
    }


def bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for index in range(repetitions):
        a = rng.choice(left, size=len(left), replace=True).mean()
        b = rng.choice(right, size=len(right), replace=True).mean()
        draws[index] = a - b
    return {
        "mean": float(np.mean(draws)),
        "ci_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "repetitions": repetitions,
    }


def boundary_decision(
    flip_coverage: float,
    distance_r2: float,
    slope_difference_ci: Sequence[float],
    slope_cv: float,
) -> tuple[str, list[str], dict[str, Any]]:
    reasons = []
    detail = {
        "flip_coverage": flip_coverage,
        "distance_only_r2": distance_r2,
        "slope_difference_ci": list(slope_difference_ci),
        "slope_coefficient_of_variation": slope_cv,
    }
    # Responsiveness is estimable from every row whether or not it ever crosses
    # the boundary, so the primary test never depends on flip coverage. Only the
    # distance-only R^2 does, and it is truncated by construction when coverage
    # is low because near-boundary rows are the ones that flip.
    heterogeneous = slope_difference_ci[0] > 0 or slope_difference_ci[1] < 0
    if heterogeneous:
        reasons.append(
            "per-row responsiveness differs by condition; boundary distance is "
            "not the whole account"
        )
        return "HETEROGENEOUS_RESPONSIVENESS", reasons, detail
    if flip_coverage < FLIP_COVERAGE_GATE:
        reasons.append(
            f"responsiveness is homogeneous, but only {flip_coverage:.2f} of rows "
            f"crossed the boundary within the coherent dose range (gate "
            f"{FLIP_COVERAGE_GATE}), so the distance-only fit is truncated to "
            "near-boundary rows and is not load bearing"
        )
        return "HOMOGENEOUS_RESPONSIVENESS_LOW_CROSSING", reasons, detail
    if distance_r2 >= R2_GATE:
        return "BOUNDARY_GEOMETRY_SUFFICIENT", [], detail
    reasons.append(
        f"distance-only R^2 {distance_r2:.3f} below the {R2_GATE} gate while "
        "responsiveness is homogeneous; something else varies"
    )
    return "INCONCLUSIVE_BOUNDARY_GEOMETRY", reasons, detail


def self_test() -> None:
    rng = np.random.default_rng(EVAL_SEED)

    class Layer(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return (x, None)

    width = 8
    direction = torch.tensor(unit(rng.normal(size=width)), dtype=torch.float32)
    hidden = torch.tensor(rng.normal(size=(3, 4, width)), dtype=torch.float32)
    module = Layer()
    handle = module.register_forward_hook(make_additive_hook(direction, 2.5))
    try:
        out = module(hidden)[0]
    finally:
        handle.remove()
    delta = (out - hidden).reshape(-1, width)
    projected = delta @ direction
    if abs(float(projected.mean()) - 2.5) > 1e-4:
        raise AssertionError("additive offset did not land on the direction")
    residual = delta - projected.unsqueeze(-1) * direction
    if float(torch.max(torch.abs(residual))) > 1e-4:
        raise AssertionError("additive hook moved components off the direction")

    doses = np.array(DOSES)
    # Shared slope: crossing dose must equal -intercept/slope exactly.
    slope = 1.5
    baselines = np.array([3.0, -2.0, 0.75, -4.5])
    matrix = baselines[None, :] + slope * doses[:, None]
    coherent = np.ones(len(doses), dtype=bool)
    fit = per_row_fit(doses, matrix, coherent)
    for index, base in enumerate(baselines):
        if abs(fit["slopes"][index] - slope) > 1e-8:
            raise AssertionError("slope recovery failed")
        got = crossing_dose(doses, matrix[:, index])
        want = -base / slope
        if got is None or abs(got - want) > 1e-6:
            raise AssertionError(f"crossing dose wrong: {got} vs {want}")

    flips = np.array([-b / slope for b in baselines])
    design = np.column_stack([np.ones(len(baselines)), baselines])
    beta, *_ = np.linalg.lstsq(design, flips, rcond=None)
    r2 = 1.0 - np.sum((flips - design @ beta) ** 2) / np.sum(
        (flips - flips.mean()) ** 2
    )
    if r2 < 0.999999:
        raise AssertionError("shared-slope case must give R^2 of one")

    homogeneous = bootstrap_difference(
        np.full(16, slope), np.full(16, slope), 300, EVAL_SEED
    )
    decision, reasons, _ = boundary_decision(1.0, float(r2), homogeneous["ci_95"], 0.0)
    if decision != "BOUNDARY_GEOMETRY_SUFFICIENT" or reasons:
        raise AssertionError("decision rule rejected the clean shared-slope case")
    heterogeneous = bootstrap_difference(
        rng.normal(2.0, 0.05, 16), rng.normal(1.0, 0.05, 16), 300, EVAL_SEED + 1
    )
    if boundary_decision(1.0, 0.99, heterogeneous["ci_95"], 0.4)[0] != (
        "HETEROGENEOUS_RESPONSIVENESS"
    ):
        raise AssertionError("decision rule missed heterogeneous responsiveness")
    if boundary_decision(0.1, 0.99, homogeneous["ci_95"], 0.0)[0] != (
        "HOMOGENEOUS_RESPONSIVENESS_LOW_CROSSING"
    ):
        raise AssertionError("decision rule mislabelled low crossing coverage")
    # Heterogeneity must be detected even when nothing crosses, since the
    # responsiveness test does not depend on crossings.
    if boundary_decision(0.0, float("nan"), heterogeneous["ci_95"], 0.4)[0] != (
        "HETEROGENEOUS_RESPONSIVENESS"
    ):
        raise AssertionError("heterogeneity must be detectable with zero crossings")

    print(
        json.dumps(
            {
                "self_test": "PASS",
                "doses": list(DOSES),
                "shared_slope_r2": float(r2),
                "crossing_dose_exact": True,
                "decisions_checked": [
                    "BOUNDARY_GEOMETRY_SUFFICIENT",
                    "HETEROGENEOUS_RESPONSIVENESS",
                    "HOMOGENEOUS_RESPONSIVENESS_LOW_CROSSING",
                    "heterogeneity detected with zero crossings",
                ],
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

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
        "protocol": "ARM_G_BOUNDARY_V1",
        "model": args.model,
        "source_seeds": list(SOURCE_SEEDS),
        "eval_seed": EVAL_SEED,
        "direction": "rank-1 seed-102 mean paired conflict difference",
        "direction_layer": DIRECTION_LAYER,
        "intervention": (
            "signed additive steering x <- x + c*sigma*r at layer 16, all token "
            "positions, where sigma is the SD of the baseline projection onto r"
        ),
        "doses_in_sigma": list(DOSES),
        "random_doses_in_sigma": list(RANDOM_DOSES),
        "prior_evidence": PRIOR_EVIDENCE,
        "primary_question": (
            "decompose per-row dose-to-flip into baseline distance divided by "
            "per-row responsiveness, and test whether responsiveness is "
            "homogeneous across conditions"
        ),
        "why_the_naive_test_is_circular": (
            "with a shared slope s, dose-to-flip is exactly -margin(0)/s, so "
            "regressing dose-to-flip on baseline margin returns R^2 = 1 as "
            "arithmetic; the evidence is in slope homogeneity, not in that fit"
        ),
        "success_rule": (
            "primary is responsiveness homogeneity, estimable from every row "
            "regardless of crossings: a condition slope difference whose CI "
            "excludes zero supports structural heterogeneity. Boundary geometry "
            "additionally requires at least half the rows to cross within the "
            "coherent dose range with distance-only R^2 >= 0.90; below that "
            "coverage the truncated fit is reported but not load bearing"
        ),
        "coherence_gate": {
            "mean_action_probability_mass": ACTION_MASS_GATE,
            "top_token_is_action_rate": TOP_TOKEN_GATE,
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

    _m, eval_states = run_forward(
        model, tokenizer, eval_rows, args.batch_size, {}, (DIRECTION_LAYER,), None
    )
    projections = eval_states[DIRECTION_LAYER] @ conflict
    sigma = float(projections.std(ddof=1))

    score = partial(score_offset, model, tokenizer)
    results = {}
    for dose in DOSES:
        results[dose] = score(
            eval_rows, conflict, dose * sigma, args.batch_size, token_ids
        )

    doses = np.array(DOSES, dtype=float)
    margin_matrix = np.stack([results[d]["margins"] for d in DOSES])
    coherent = np.array(
        [results[d]["coherence"]["passes_coherence_gate"] for d in DOSES], dtype=bool
    )
    baseline = results[0.0]["margins"]
    labels = np.array([row["condition_label"] for row in eval_rows])

    fit = per_row_fit(doses, margin_matrix, coherent)
    slopes = np.array(fit["slopes"])
    crossings = [
        crossing_dose(doses[coherent], margin_matrix[coherent, row])
        for row in range(len(eval_rows))
    ]
    flipped = np.array([c is not None for c in crossings])
    flip_coverage = float(flipped.mean())

    observed = np.array([c for c in crossings if c is not None])
    observed_baseline = baseline[flipped]
    design = np.column_stack([np.ones(len(observed)), observed_baseline])
    beta, *_ = np.linalg.lstsq(design, observed, rcond=None)
    ss_res = float(np.sum((observed - design @ beta) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    distance_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    slope_difference = bootstrap_difference(
        slopes[labels == 1], slopes[labels == 0], args.bootstrap, EVAL_SEED
    )
    slope_cv = float(np.std(slopes, ddof=1) / abs(np.mean(slopes)))
    decision, decision_reasons, decision_detail = boundary_decision(
        flip_coverage, float(distance_r2), slope_difference["ci_95"], slope_cv
    )

    rng = np.random.default_rng(EVAL_SEED)
    random_controls = {}
    for control_index in range(args.random_directions):
        candidate = rng.normal(size=len(conflict))
        candidate -= (candidate @ conflict) * conflict
        vector = unit(candidate)
        entries = {}
        for dose in RANDOM_DOSES:
            outcome = score(eval_rows, vector, dose * sigma, args.batch_size, token_ids)
            entries[str(dose)] = {
                "condition_contrast": contrast_slices(outcome["margins"], eval_rows)[
                    "overall"
                ],
                "sign_flips": int(
                    np.sum(np.sign(outcome["margins"]) != np.sign(baseline))
                ),
                "coherence": outcome["coherence"],
            }
        random_controls[f"control_{control_index}"] = entries

    report = {
        "status": "ARM_G_BOUNDARY_V1",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "decision_detail": decision_detail,
        "projection_sigma": sigma,
        "baseline": {
            "choice_summary": semantic_choice_summary(baseline, eval_rows),
            "condition_contrast": contrast_slices(baseline, eval_rows),
        },
        "dose_response": {
            str(dose): {
                "condition_contrast": contrast_slices(
                    results[dose]["margins"], eval_rows
                )["overall"],
                "conflict_decline_rate": float(
                    np.mean(results[dose]["margins"][labels == 1] > 0)
                ),
                "reachable_decline_rate": float(
                    np.mean(results[dose]["margins"][labels == 0] > 0)
                ),
                "sign_flips": int(
                    np.sum(np.sign(results[dose]["margins"]) != np.sign(baseline))
                ),
                "coherence": results[dose]["coherence"],
            }
            for dose in DOSES
        },
        "decomposition": {
            "flip_coverage": flip_coverage,
            "rows_flipped": int(flipped.sum()),
            "distance_only_regression": {
                "intercept": float(beta[0]),
                "slope": float(beta[1]),
                "r2": float(distance_r2),
                "n": int(len(observed)),
            },
            "responsiveness": {
                "mean_slope": float(slopes.mean()),
                "sd_slope": float(slopes.std(ddof=1)),
                "coefficient_of_variation": slope_cv,
                "mean_linear_fit_r2": float(np.nanmean(fit["linear_fit_r2"])),
                "conflict_mean_slope": float(slopes[labels == 1].mean()),
                "reachable_mean_slope": float(slopes[labels == 0].mean()),
                "conflict_minus_reachable": slope_difference,
            },
            "coherent_doses": doses[coherent].tolist(),
        },
        "random_controls": random_controls,
        "sample_counts": {
            "source_rows": len(all_source_rows),
            "evaluation_rows": len(eval_rows),
            "doses": len(DOSES),
            "coherent_doses": int(coherent.sum()),
        },
        "audits": {
            "source_manifests": source_audits,
            "evaluation_manifest": eval_audit,
        },
        "row_results": [
            {
                **{k: v for k, v in row.items() if k != "messages"},
                "baseline_semantic_margin": float(baseline[index]),
                "baseline_projection": float(projections[index]),
                "responsiveness_slope": float(slopes[index]),
                "linear_fit_r2": float(fit["linear_fit_r2"][index]),
                "crossing_dose_sigma": crossings[index],
                "margins_by_dose": {
                    str(dose): float(results[dose]["margins"][index]) for dose in DOSES
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
    result_path = args.output_dir / "arm_g_boundary_result.json"
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
                "projection_sigma": sigma,
                "dose_response": report["dose_response"],
                "decomposition": report["decomposition"],
                "sample_counts": report["sample_counts"],
            },
            indent=2,
        )
    )
    print(f"full artifact: {result_path}")


if __name__ == "__main__":
    main()
