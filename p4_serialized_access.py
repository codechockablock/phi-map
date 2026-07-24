"""P4 / Arm S — serialized activation access controls and analysis.

This module implements the parts of the serialized-self-access experiment
that can be validated before a GPU run:

1. label-free, low-leakage random-projection serializers;
2. paired AUROC, NLL, and discordance statistics with bootstrap intervals;
3. a synthetic end-to-end smoke test for channel utility; and
4. a cache-equivalence test for the proposed same-run-vs-clone comparison.

Identification gate
-------------------
For a deterministic decoder-only Transformer in eval mode, the KV cache is
a function of the visible prefix and the frozen weights. Reusing that cache
and recomputing it in a fresh exact clone should therefore give the same
next-token distribution, up to numerical implementation error. A material
source-over-clone gap is not evidence of privileged self-access unless the
protocol names and ablates additional state that the clone cannot reconstruct.

The module is read-only with respect to model state: no hooks replace
activations, no adapters/checkpoints are written, and no optimizer exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
SMOKE_OUT = ROOT / "results" / "p4_serialized_access_smoke.json"
EQUIV_OUT = ROOT / "results" / "p4_clone_equivalence.json"


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC with average ranks for ties and no sklearn dependency."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.ndim != 1 or y.ndim != 1 or len(s) != len(y):
        raise ValueError("scores and labels must be equal-length 1-D arrays")
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    average_ranks = (cumulative - counts + cumulative + 1) / 2.0
    ranks = average_ranks[inverse]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def binary_nll(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(labels, dtype=int)
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y):
        raise ValueError("probabilities and labels must be equal-length 1-D arrays")
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def paired_effects(
    labels: Sequence[int],
    probabilities_a: Sequence[float],
    probabilities_b: Sequence[float],
) -> dict[str, float]:
    """Return paired effects where positive values favor condition A."""
    y = np.asarray(labels, dtype=int)
    pa = np.asarray(probabilities_a, dtype=float)
    pb = np.asarray(probabilities_b, dtype=float)
    pred_a = pa >= 0.5
    pred_b = pb >= 0.5
    correct_a = pred_a == y
    correct_b = pred_b == y
    p10 = float(np.mean(correct_a & ~correct_b))
    p01 = float(np.mean(~correct_a & correct_b))
    return {
        "delta_auroc": auroc(pa, y) - auroc(pb, y),
        "delta_nll": binary_nll(pb, y) - binary_nll(pa, y),
        "discordance_net": p10 - p01,
        "p10_a_correct_b_wrong": p10,
        "p01_a_wrong_b_correct": p01,
    }


def paired_bootstrap(
    labels: Sequence[int],
    probabilities_a: Sequence[float],
    probabilities_b: Sequence[float],
    *,
    n_bootstrap: int = 2_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Percentile intervals from item-paired bootstrap resampling."""
    y = np.asarray(labels, dtype=int)
    pa = np.asarray(probabilities_a, dtype=float)
    pb = np.asarray(probabilities_b, dtype=float)
    if len(y) < 2:
        raise ValueError("at least two items are required")
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {
        "delta_auroc": [],
        "delta_nll": [],
        "discordance_net": [],
    }
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        effect = paired_effects(y[idx], pa[idx], pb[idx])
        for name in draws:
            draws[name].append(effect[name])
    if not draws["delta_auroc"]:
        raise ValueError("bootstrap produced no samples containing both classes")
    intervals = {}
    for name, values in draws.items():
        low, high = np.quantile(values, [0.025, 0.975])
        intervals[name] = {
            "low_95": float(low),
            "high_95": float(high),
        }
    return intervals


@dataclass(frozen=True)
class SerializerConfig:
    method: str
    input_dim: int
    projection_dim: int
    seed: int = 0
    quantization_levels: int = 15
    calibration_quantile: float = 0.995
    version: str = "rpq-v1"


class RandomProjectionSerializer:
    """Frozen label-free projection with scalar or sign quantization."""

    def __init__(self, config: SerializerConfig):
        if config.method not in {"scalar_quantized", "binary_sign"}:
            raise ValueError(f"unsupported method: {config.method}")
        if config.projection_dim <= 0 or config.input_dim <= 0:
            raise ValueError("serializer dimensions must be positive")
        if config.quantization_levels < 3 or config.quantization_levels % 2 == 0:
            raise ValueError("quantization_levels must be odd and at least 3")
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.projection = rng.standard_normal(
            (config.projection_dim, config.input_dim)
        ) / np.sqrt(config.projection_dim)
        self.center: np.ndarray | None = None
        self.step: float | None = None

    def fit(self, calibration_activations: np.ndarray) -> None:
        """Fit label-free centering and one global quantization step."""
        h = np.asarray(calibration_activations, dtype=float)
        if h.ndim != 2 or h.shape[1] != self.config.input_dim:
            raise ValueError(
                f"calibration activations must have shape (n, {self.config.input_dim})"
            )
        self.center = h.mean(axis=0)
        projected = (h - self.center) @ self.projection.T
        if self.config.method == "scalar_quantized":
            limit = float(
                np.quantile(np.abs(projected), self.config.calibration_quantile)
            )
            max_code = (self.config.quantization_levels - 1) // 2
            self.step = max(limit / max_code, 1e-8)
        else:
            self.step = None

    def transform(self, activations: np.ndarray) -> np.ndarray:
        if self.center is None:
            raise RuntimeError("fit() must be called before transform()")
        h = np.asarray(activations, dtype=float)
        if h.ndim != 2 or h.shape[1] != self.config.input_dim:
            raise ValueError(
                f"activations must have shape (n, {self.config.input_dim})"
            )
        projected = (h - self.center) @ self.projection.T
        if self.config.method == "binary_sign":
            return (projected >= 0).astype(np.int8)
        assert self.step is not None
        max_code = (self.config.quantization_levels - 1) // 2
        return np.clip(np.rint(projected / self.step), -max_code, max_code).astype(
            np.int16
        )

    def payload(self, codes: np.ndarray) -> dict[str, Any]:
        c = np.asarray(codes)
        if c.ndim != 1 or len(c) != self.config.projection_dim:
            raise ValueError(
                f"one code vector of length {self.config.projection_dim} required"
            )
        return {
            "version": self.config.version,
            "method": self.config.method,
            "projection_seed": self.config.seed,
            "projection_dim": self.config.projection_dim,
            "codes": [int(value) for value in c],
        }

    def payload_text(self, codes: np.ndarray) -> str:
        return json.dumps(
            self.payload(codes),
            sort_keys=True,
            separators=(",", ":"),
        )


def _metric_block(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "auroc": auroc(probabilities, labels),
        "nll": binary_nll(probabilities, labels),
        "accuracy": float(np.mean(predictions == labels)),
    }


def run_smoke(output: Path, n_bootstrap: int) -> dict[str, Any]:
    """Synthetic validation: useful channel, null source-over-clone gap."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(17)
    n_items = 1_600
    input_dim = 128
    labels = np.tile(np.array([0, 1], dtype=int), n_items // 2)
    rng.shuffle(labels)
    direction = rng.standard_normal(input_dim)
    direction /= np.linalg.norm(direction)
    activations = rng.standard_normal((n_items, input_dim))
    activations += (2 * labels[:, None] - 1) * 1.6 * direction
    train = np.arange(0, n_items // 2)
    test = np.arange(n_items // 2, n_items)

    report: dict[str, Any] = {
        "status": "SYNTHETIC_PIPELINE_VALIDATION_NOT_RESEARCH_RESULT",
        "seed": 17,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "serializers": {},
    }
    for method in ("scalar_quantized", "binary_sign"):
        config = SerializerConfig(
            method=method,
            input_dim=input_dim,
            projection_dim=64,
            seed=23,
        )
        serializer = RandomProjectionSerializer(config)
        serializer.fit(activations[train])
        train_codes = serializer.transform(activations[train])
        test_codes = serializer.transform(activations[test])
        decoder = LogisticRegression(max_iter=2_000, random_state=0)
        decoder.fit(train_codes, labels[train])
        channel_prob = decoder.predict_proba(test_codes)[:, 1]

        # An exact clone receiving the same payload runs the same map.
        clone_prob = channel_prob.copy()
        input_only_prob = np.full(len(test), float(labels[train].mean()))
        shuffled_codes = test_codes[rng.permutation(len(test_codes))]
        shuffled_prob = decoder.predict_proba(shuffled_codes)[:, 1]
        y_test = labels[test]
        source_clone = paired_effects(y_test, channel_prob, clone_prob)
        source_clone_ci = paired_bootstrap(
            y_test,
            channel_prob,
            clone_prob,
            n_bootstrap=n_bootstrap,
            seed=29,
        )
        channel_input = paired_effects(y_test, channel_prob, input_only_prob)
        payloads = [serializer.payload(row) for row in test_codes]
        report["serializers"][method] = {
            "config": asdict(config),
            "channel": _metric_block(y_test, channel_prob),
            "input_only": _metric_block(y_test, input_only_prob),
            "shuffled_channel": _metric_block(y_test, shuffled_prob),
            "channel_minus_input_only": channel_input,
            "source_minus_exact_clone": source_clone,
            "source_minus_exact_clone_ci": source_clone_ci,
            "unique_payload_hashes": len({stable_hash(p) for p in payloads}),
            "interpretation": (
                "CHANNEL_UTILITY_ONLY"
                if channel_input["delta_auroc"] > 0.1
                and source_clone["delta_auroc"] == 0.0
                else "SMOKE_TEST_FAILED"
            ),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _default_prompts() -> list[str]:
    return [
        "A box contains a red key and a blue coin. Name the object used to unlock.",
        "Write one sentence explaining why calibration matters in measurement.",
        "If a claim survives a shuffled-label control, what has been ruled out?",
    ]


def run_cache_equivalence(
    model_id: str,
    output: Path,
    *,
    local_files_only: bool,
) -> dict[str, Any]:
    """Compare cache continuation with a fresh exact-clone full pass."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float32
    load_kwargs = {
        "dtype": dtype,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    source = (
        AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device).eval()
    )
    clone = (
        AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device).eval()
    )
    source_revision = getattr(source.config, "_commit_hash", None)
    clone_revision = getattr(clone.config, "_commit_hash", None)
    if source_revision != clone_revision:
        raise RuntimeError(
            "source and clone resolved to different checkpoint revisions"
        )

    trials = []
    with torch.no_grad():
        for prompt in _default_prompts():
            prefix_ids = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(device)
            prefix_out = source(
                input_ids=prefix_ids,
                output_hidden_states=True,
                use_cache=True,
            )
            activation = prefix_out.hidden_states[-1][0, -1].float().cpu().numpy()
            config = SerializerConfig(
                method="binary_sign",
                input_dim=len(activation),
                projection_dim=64,
                seed=23,
            )
            serializer = RandomProjectionSerializer(config)
            serializer.fit(np.stack([np.zeros_like(activation), activation]))
            code = serializer.transform(activation[None, :])[0]
            channel_text = "\nACTIVATION_CHANNEL=" + serializer.payload_text(code)
            channel_ids = tokenizer(
                channel_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(device)

            cached_out = source(
                input_ids=channel_ids,
                past_key_values=prefix_out.past_key_values,
                use_cache=False,
            )
            full_ids = torch.cat([prefix_ids, channel_ids], dim=1)
            fresh_out = clone(input_ids=full_ids, use_cache=False)
            cached_logits = cached_out.logits[0, -1].float().cpu()
            fresh_logits = fresh_out.logits[0, -1].float().cpu()
            cached_prob = torch.softmax(cached_logits.double(), dim=-1)
            fresh_prob = torch.softmax(fresh_logits.double(), dim=-1)
            midpoint = 0.5 * (cached_prob + fresh_prob)
            js_divergence = torch.clamp_min(
                0.5
                * (
                    torch.sum(
                        cached_prob
                        * (
                            torch.log(cached_prob.clamp_min(1e-12))
                            - torch.log(midpoint.clamp_min(1e-12))
                        )
                    )
                    + torch.sum(
                        fresh_prob
                        * (
                            torch.log(fresh_prob.clamp_min(1e-12))
                            - torch.log(midpoint.clamp_min(1e-12))
                        )
                    )
                ),
                0.0,
            )
            trials.append(
                {
                    "prompt_hash": stable_hash(prompt),
                    "channel_hash": stable_hash(serializer.payload(code)),
                    "prefix_tokens": int(prefix_ids.shape[1]),
                    "channel_tokens": int(channel_ids.shape[1]),
                    "max_abs_logit_delta": float(
                        torch.max(torch.abs(cached_logits - fresh_logits))
                    ),
                    "jensen_shannon_divergence": float(js_divergence),
                    "top1_equal": bool(
                        torch.argmax(cached_logits) == torch.argmax(fresh_logits)
                    ),
                }
            )

    max_logit_delta = max(t["max_abs_logit_delta"] for t in trials)
    max_js = max(t["jensen_shannon_divergence"] for t in trials)
    all_top1_equal = all(t["top1_equal"] for t in trials)
    report = {
        "status": "CLONE_EQUIVALENCE_METHODOLOGICAL_CONTROL_NOT_TASK_RESULT",
        "model": model_id,
        "checkpoint_revision": source_revision,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "runtime_wrapper_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "local_files_only": local_files_only,
        "source_and_clone_loaded_separately": True,
        "serializer": asdict(config),
        "n_trials": len(trials),
        "trials": trials,
        "summary": {
            "max_abs_logit_delta": max_logit_delta,
            "max_jensen_shannon_divergence": max_js,
            "all_top1_equal": all_top1_equal,
        },
        "interpretation": (
            "CACHE_AND_FRESH_CLONE_EQUIVALENT_WITHIN_NUMERICAL_ERROR"
            if all_top1_equal and max_js < 1e-7
            else "IMPLEMENTATION_DIFFERENCE_REQUIRES_INVESTIGATION"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output", type=Path, default=SMOKE_OUT)
    smoke.add_argument("--bootstrap", type=int, default=2_000)

    equivalence = subparsers.add_parser("cache-equivalence")
    equivalence.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    equivalence.add_argument("--output", type=Path, default=EQUIV_OUT)
    equivalence.add_argument(
        "--allow-download",
        action="store_true",
        help="allow model/tokenizer download if not already cached",
    )

    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(args.output, args.bootstrap)
        print(json.dumps(result, indent=2))
        print(f"\nwritten: {args.output}")
    else:
        result = run_cache_equivalence(
            args.model,
            args.output,
            local_files_only=not args.allow_download,
        )
        print(json.dumps(result, indent=2))
        print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
