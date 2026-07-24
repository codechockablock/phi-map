"""Strong same-sample behavioral comparator on Apollo deception rollouts.

This follow-up closes the selection mismatch in the earlier frontier-ops
step-mean experiment, which required at least three sentence steps. Here every
response used by the frozen 6-D evaluation is retained. Responses are split
with the historical sentence/line rule; one- and two-step responses remain in
the analysis, and an unsplittable non-empty response falls back to one step.

Fixed protocol
--------------
Primary: raw all-MiniLM-L6-v2 sentence embeddings, averaged within response,
then a five-fold out-of-fold prototype direction (deceptive mean minus honest
mean). Headline: macro mean of the four task-family AUROCs.

Secondary diagnostics cannot replace the primary:
  - the same prototype on a single whole-response embedding;
  - supervised logistic readout of the step-mean embedding;
  - response-style logistic baseline (characters, words, and step count);
  - step-mean prototype score residualized against those style features;
  - frozen cross-task transfer of each full-data prototype direction.

The sample set, labels, folds, bootstrap, and task weights exactly match
p3_behavioral_6d.py. No task-specific hyperparameters or score orientation
are selected from evaluation results.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p1_exact_pairing import EXT_FILE, ROLLOUTS, TASKS, auroc
from p3_behavioral_6d import load_transcripts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "p3_strong_behavioral_baseline.json"
APOLLO_REPO = ROLLOUTS.parents[1]
MODEL_NAME = "all-MiniLM-L6-v2"
STEP_SPLIT_RE = r"(?<=[.!?])\s+|\n+"
MIN_STEP_CHARS = 15
FOLD_SEED = 17
SENSITIVITY_SEEDS = [*range(10), 11, FOLD_SEED]
REFERENCE_CACHE = Path("/Users/joseph/frontier-ops-v3/eval/data/cache")


def split_steps(text: str) -> list[str]:
    """Historical splitter, retaining short trajectories for exact sampling."""
    steps = [
        fragment.strip()
        for fragment in re.split(STEP_SPLIT_RE, text)
        if len(fragment.strip()) > MIN_STEP_CHARS
    ]
    return steps or [text.strip()]


def style_features(texts: Sequence[str], step_counts: np.ndarray) -> np.ndarray:
    """Simple form-only features, with no lexical or semantic content."""
    return np.column_stack(
        [
            np.log1p([len(text) for text in texts]),
            np.log1p([len(text.split()) for text in texts]),
            np.log1p(step_counts),
        ]
    )


def encode_responses(
    model: SentenceTransformer,
    texts: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return step means, whole-response embeddings, and step counts."""
    per_response = [split_steps(text) for text in texts]
    bounds = np.cumsum([0] + [len(steps) for steps in per_response])
    all_steps = [step for steps in per_response for step in steps]
    step_embeddings = model.encode(
        all_steps,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    step_means = np.stack(
        [
            step_embeddings[bounds[index] : bounds[index + 1]].mean(axis=0)
            for index in range(len(texts))
        ]
    )
    full_embeddings = model.encode(
        list(texts),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return (
        step_means,
        full_embeddings,
        np.asarray([len(steps) for steps in per_response], dtype=int),
    )


def reference_parity(
    task: str,
    step_means: np.ndarray,
    full_embeddings: np.ndarray,
    step_counts: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compare against the earlier frontier-ops embedding caches when present."""
    parity: dict[str, Any] = {}

    raw_cache = REFERENCE_CACHE / f"raw_{task}.npz"
    if raw_cache.exists():
        raw = np.load(raw_cache)
        parity["full_response"] = {
            "examples": len(labels),
            "label_vector_exact": bool(np.array_equal(raw["Y"], labels)),
            "embedding_max_abs_delta": (
                float(np.max(np.abs(raw["E"] - full_embeddings)))
                if raw["E"].shape == full_embeddings.shape
                else None
            ),
        }

    matrix_cache = REFERENCE_CACHE / f"matrix_{task}.npz"
    if matrix_cache.exists():
        matrix = np.load(matrix_cache)
        retained = step_counts >= 3
        parity["historical_three_step_subset"] = {
            "examples": int(np.sum(retained)),
            "label_vector_exact": bool(np.array_equal(matrix["Y"], labels[retained])),
            "stepmean_embedding_max_abs_delta": (
                float(np.max(np.abs(matrix["raw_mean"] - step_means[retained])))
                if matrix["raw_mean"].shape == step_means[retained].shape
                else None
            ),
        }
    return parity


def folds(labels: np.ndarray) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)


def prototype(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return features[labels == 1].mean(axis=0) - features[labels == 0].mean(axis=0)


def prototype_oof(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    scores = np.zeros(len(labels))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, labels):
        scores[test] = features[test] @ prototype(features[train], labels[train])
    return scores


def out_of_fold_scores(
    step_means: np.ndarray,
    full_embeddings: np.ndarray,
    styles: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    scores = {
        "stepmean_prototype": np.zeros(len(labels)),
        "full_prototype": np.zeros(len(labels)),
        "stepmean_logistic": np.zeros(len(labels)),
        "style_logistic": np.zeros(len(labels)),
        "stepmean_style_residual": np.zeros(len(labels)),
    }
    for train, test in folds(labels).split(step_means, labels):
        step_direction = prototype(step_means[train], labels[train])
        full_direction = prototype(full_embeddings[train], labels[train])
        step_train_scores = step_means[train] @ step_direction
        step_test_scores = step_means[test] @ step_direction

        scores["stepmean_prototype"][test] = step_test_scores
        scores["full_prototype"][test] = full_embeddings[test] @ full_direction

        semantic_logistic = LogisticRegression(
            C=1.0,
            max_iter=2_000,
            random_state=0,
        ).fit(step_means[train], labels[train])
        scores["stepmean_logistic"][test] = semantic_logistic.predict_proba(
            step_means[test]
        )[:, 1]

        style_logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2_000, random_state=0),
        ).fit(styles[train], labels[train])
        scores["style_logistic"][test] = style_logistic.predict_proba(styles[test])[
            :, 1
        ]

        style_regression = LinearRegression().fit(styles[train], step_train_scores)
        scores["stepmean_style_residual"][test] = (
            step_test_scores - style_regression.predict(styles[test])
        )
    return scores


def bootstrap_indices(
    labels: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> list[np.ndarray]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        for _ in range(n_bootstrap)
    ]


def metric(
    scores: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[np.ndarray],
) -> dict[str, Any]:
    draws = [auroc(scores[index], labels[index]) for index in indices]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "auroc": auroc(scores, labels),
        "ci_95": [float(low), float(high)],
        "_bootstrap_draws": draws,
    }


def paired_difference(
    first: np.ndarray,
    second: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[np.ndarray],
) -> dict[str, Any]:
    draws = [
        auroc(first[index], labels[index]) - auroc(second[index], labels[index])
        for index in indices
    ]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "auroc_difference": auroc(first, labels) - auroc(second, labels),
        "ci_95": [float(low), float(high)],
        "_bootstrap_draws": draws,
    }


def macro_metric(
    task_results: dict[str, dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    values = [task_results[task]["metrics"][metric_name] for task in TASKS]
    draws = np.mean(
        np.asarray([value["_bootstrap_draws"] for value in values]),
        axis=0,
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "auroc": float(np.mean([value["auroc"] for value in values])),
        "ci_95": [float(low), float(high)],
    }


def macro_difference(task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = [
        task_results[task]["paired_stepmean_minus_full_prototype"] for task in TASKS
    ]
    draws = np.mean(
        np.asarray([value["_bootstrap_draws"] for value in values]),
        axis=0,
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "auroc_difference": float(
            np.mean([value["auroc_difference"] for value in values])
        ),
        "ci_95": [float(low), float(high)],
    }


def transfer_matrix(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    diagonal: dict[str, float],
) -> dict[str, dict[str, float]]:
    directions = {task: prototype(features[task], labels[task]) for task in TASKS}
    return {
        trained: {
            evaluated: (
                diagonal[evaluated]
                if trained == evaluated
                else auroc(
                    features[evaluated] @ directions[trained],
                    labels[evaluated],
                )
            )
            for evaluated in TASKS
        }
        for trained in TASKS
    }


def strip_draws(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_draws(child)
            for key, child in value.items()
            if key != "_bootstrap_draws"
        }
    if isinstance(value, list):
        return [strip_draws(child) for child in value]
    return value


def git_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    step_features: dict[str, np.ndarray] = {}
    task_labels: dict[str, np.ndarray] = {}
    task_results: dict[str, dict[str, Any]] = {}

    for task_index, task in enumerate(TASKS):
        responses, _, labels = load_transcripts(ROLLOUTS / EXT_FILE[task])
        print(f"encoding {task}: n={len(labels)}", flush=True)
        step_means, full_embeddings, step_counts = encode_responses(model, responses)
        styles = style_features(responses, step_counts)
        scores = out_of_fold_scores(
            step_means,
            full_embeddings,
            styles,
            labels,
        )
        indices = bootstrap_indices(
            labels,
            n_bootstrap=args.bootstrap,
            seed=300 + 20 * task_index,
        )
        metrics = {
            name: metric(score, labels, indices) for name, score in scores.items()
        }
        task_results[task] = {
            "n": len(labels),
            "honest": int(np.sum(labels == 0)),
            "deceptive": int(np.sum(labels == 1)),
            "step_counts": {
                "one": int(np.sum(step_counts == 1)),
                "two": int(np.sum(step_counts == 2)),
                "three_or_more": int(np.sum(step_counts >= 3)),
                "median": float(np.median(step_counts)),
            },
            "metrics": metrics,
            "paired_stepmean_minus_full_prototype": paired_difference(
                scores["stepmean_prototype"],
                scores["full_prototype"],
                labels,
                indices,
            ),
            "reference_cache_parity": reference_parity(
                task,
                step_means,
                full_embeddings,
                step_counts,
                labels,
            ),
            "fold_seed_sensitivity_secondary": {
                str(seed): auroc(
                    prototype_oof(step_means, labels, seed=seed),
                    labels,
                )
                for seed in SENSITIVITY_SEEDS
            },
        }
        step_features[task] = step_means
        task_labels[task] = labels

    metric_names = [
        "stepmean_prototype",
        "full_prototype",
        "stepmean_logistic",
        "style_logistic",
        "stepmean_style_residual",
    ]
    macros = {name: macro_metric(task_results, name) for name in metric_names}
    diagonal = {
        task: task_results[task]["metrics"]["stepmean_prototype"]["auroc"]
        for task in TASKS
    }
    transfer = transfer_matrix(step_features, task_labels, diagonal)
    off_diagonal = [
        transfer[trained][evaluated]
        for trained in TASKS
        for evaluated in TASKS
        if trained != evaluated
    ]
    sensitivity_macro_by_seed = {
        str(seed): float(
            np.mean(
                [
                    task_results[task]["fold_seed_sensitivity_secondary"][str(seed)]
                    for task in TASKS
                ]
            )
        )
        for seed in SENSITIVITY_SEEDS
    }
    sensitivity_values = np.asarray(list(sensitivity_macro_by_seed.values()))
    report = {
        "status": "STRONG_SAME_SAMPLE_BEHAVIORAL_COMPARATOR",
        "protocol": {
            "primary": (
                "macro mean of per-task 5-fold out-of-fold raw step-mean "
                "prototype AUROC, response-only"
            ),
            "sample_set": (
                "exact 1,443-response set from p3_behavioral_6d.py; no "
                "minimum-step exclusion"
            ),
            "folds": ("StratifiedKFold(n_splits=5, shuffle=True, random_state=17)"),
            "bootstrap": args.bootstrap,
            "label_rule": (
                "graded_deception first item: <=2/honest=0, >=6/deceptive=1"
            ),
            "step_split": (
                "sentence ends or newlines; fragments >15 chars; unsplittable "
                "non-empty response falls back to one step"
            ),
            "orientation": (
                "prototype orientation fixed by training labels; no evaluation flip"
            ),
        },
        "provenance": {
            "apollo_checkout": str(APOLLO_REPO),
            "apollo_commit": git_revision(APOLLO_REPO),
            "encoder": MODEL_NAME,
            "embedding_dimension": int(model.get_sentence_embedding_dimension()),
            "sentence_transformers_version": package_version("sentence-transformers"),
            "transformers_version": package_version("transformers"),
            "sklearn_version": package_version("scikit-learn"),
        },
        "tasks": strip_draws(task_results),
        "macro": macros,
        "paired_stepmean_minus_full_prototype_macro": macro_difference(task_results),
        "stepmean_prototype_transfer": transfer,
        "mean_off_diagonal_transfer_auroc": float(np.mean(off_diagonal)),
        "fold_seed_sensitivity_secondary": {
            "seeds": SENSITIVITY_SEEDS,
            "macro_by_seed": sensitivity_macro_by_seed,
            "mean": float(np.mean(sensitivity_values)),
            "min": float(np.min(sensitivity_values)),
            "max": float(np.max(sensitivity_values)),
            "primary_seed": FOLD_SEED,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("\ntask                 n   stepmean full      logit     style     residual")
    for task in TASKS:
        result = report["tasks"][task]
        metrics = result["metrics"]
        print(
            f"{task:<20} {result['n']:>4}  "
            f"{metrics['stepmean_prototype']['auroc']:.3f}    "
            f"{metrics['full_prototype']['auroc']:.3f}    "
            f"{metrics['stepmean_logistic']['auroc']:.3f}    "
            f"{metrics['style_logistic']['auroc']:.3f}    "
            f"{metrics['stepmean_style_residual']['auroc']:.3f}"
        )
    primary = macros["stepmean_prototype"]
    print(
        "\nprimary macro raw step-mean: "
        f"{primary['auroc']:.3f} "
        f"CI [{primary['ci_95'][0]:.3f}, {primary['ci_95'][1]:.3f}]"
    )
    print(f"macro raw full response: {macros['full_prototype']['auroc']:.3f}")
    print(f"macro style-only: {macros['style_logistic']['auroc']:.3f}")
    print(f"mean cross-task transfer: {report['mean_off_diagonal_transfer_auroc']:.3f}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
