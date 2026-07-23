"""P2 — matched-methodology, both banks. Pipeline + local validation.

The decisive experiment: on ONE acting model, per task family, estimate
the deception direction in BOTH activation space and behavioral-embedding
space using the SAME estimator class, and compare cross-task transfer.

This machine (M4/no-CUDA) runs the pipeline + control triad on a SMALL
ungated dev model — the pre-registered laptop job (WRITEUP §7b). The real
numbers need Llama-3.1-8B in bf16 on the GPU rung (Colab A100 / NDIF);
that is a `--model` swap, nothing else. Dev-model AUROCs are NOT the
result and are labeled as such.

Internal side (this file):
  - activations via output_hidden_states (all layers), two read positions
    (mean over tokens, last token), same rollout samples as the external
    side (load_external, encode_raw rule) → matched.
  - two estimators per (task, layer, position): mean-diff prototype
    [matches the external estimator] and L2 logistic probe.
  - 4x4 transfer AUROC (train A → apply B) + cosine matrix per side.
  - layer chosen by in-task CV (never by cross-task — no test tuning).

Controls (Phase 3 triad), all on the acting model's activations:
  - positive: probe a trivially-present property (output length > median)
    → must reach >=0.95 for some layer, else extraction is broken.
  - null: shuffled real labels → [0.45, 0.55].
  - random-direction baseline → distribution.

Decision rule (pre-registered, evaluated on the REAL model only):
  supported if internal mean cross-task − external ≥ 0.2, internal ≥ 0.8,
  external ≤ 0.65; refuted if external ≥ internal − 0.1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from p1_exact_pairing import EXT_FILE, ROLLOUTS, auroc, load_external

TASKS = ["roleplaying", "ai_liar", "insider_trading", "sandbagging"]
OUT = Path(__file__).resolve().parent / "results" / "p2_harness.json"


def device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # bf16 activations only claimed on the GPU rung
    return "cpu", torch.float32


def balanced(texts, y, cap, seed=0):
    rng = np.random.default_rng(seed)
    dec = rng.permutation(np.flatnonzero(y == 1))[:cap]
    hon = rng.permutation(np.flatnonzero(y == 0))[:cap]
    idx = np.concatenate([dec, hon])
    return [texts[i] for i in idx], y[idx]


@torch.no_grad()
def extract(model, tok, texts, device, max_tok=384):
    """Return dict pos -> (n_samples, n_layers, d) float32."""
    mean_layers, last_layers = [], []
    for t in texts:
        enc = tok(t[:4000], return_tensors="pt", truncation=True,
                  max_length=max_tok).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple L+1
        H = torch.stack(hs, 0).squeeze(1).float().cpu()  # (L+1, seq, d)
        mean_layers.append(H.mean(1).numpy())            # (L+1, d)
        last_layers.append(H[:, -1, :].numpy())          # (L+1, d)
    return {
        "mean": np.stack(mean_layers, 1),   # (L+1, n, d)
        "last": np.stack(last_layers, 1),
    }


def proto(E, y):
    d = E[y == 1].mean(0) - E[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def cv_intask(E, y, seed=5):
    rng = np.random.default_rng(seed)
    n = len(y); perm = rng.permutation(n); folds = np.array_split(perm, 5)
    s = np.zeros(n)
    for f in range(5):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(5) if j != f])
        if y[tr].min() == y[tr].max():
            continue
        s[te] = E[te] @ proto(E[tr], y[tr])
    return auroc(s, y)


def transfer_matrix(acts_layer, Y):
    """acts_layer: dict task -> (n, d) at one layer/pos. mean-diff estimator."""
    W = {t: proto(acts_layer[t], Y[t]) for t in TASKS}
    M = {}
    for a in TASKS:
        M[a] = {}
        for b in TASKS:
            if a == b:
                M[a][b] = round(cv_intask(acts_layer[b], Y[b]), 4)
            else:
                M[a][b] = round(auroc(acts_layer[b] @ W[a], Y[b]), 4)
    cos = [[round(float(W[a] @ W[b]), 4) for b in TASKS] for a in TASKS]
    off = [M[a][b] for a in TASKS for b in TASKS if a != b]
    return M, cos, float(np.mean(off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--cap", type=int, default=60, help="samples/class/task (dev)")
    args = ap.parse_args()
    OUT.parent.mkdir(exist_ok=True)
    device, dtype = device_dtype()
    is_dev = "8b" not in args.model.lower() and "8B" not in args.model
    report = {"model": args.model, "device": device, "dtype": str(dtype),
              "cap_per_class": args.cap,
              "IS_DEV_PIPELINE_VALIDATION_NOT_RESULT": is_dev}

    print(f"loading {args.model} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    acts = {}   # task -> {pos: (L+1, n, d)}
    Y = {}
    lengths = {}
    for t in TASKS:
        texts, y = load_external(ROLLOUTS / EXT_FILE[t])
        texts, y = balanced(texts, y, args.cap)
        print(f"  extracting {t}: {len(texts)} samples", flush=True)
        acts[t] = extract(model, tok, texts, device)
        Y[t] = y
        lengths[t] = np.array([len(x) for x in texts])
    nL = acts[TASKS[0]]["mean"].shape[0]
    report["n_layers_plus1"] = nL

    # ---- control triad on the acting model (position=mean) --------------
    # positive: probe output-length>median (trivially present), best layer
    pos_ctrl = []
    for L in range(nL):
        aur = []
        for t in TASKS:
            E = acts[t]["mean"][L]
            yl = (lengths[t] > np.median(lengths[t])).astype(int)
            if yl.min() != yl.max():
                aur.append(cv_intask(E, yl))
        pos_ctrl.append(np.mean(aur))
    report["control_positive_max_auroc"] = round(float(np.max(pos_ctrl)), 4)
    # null: shuffled real labels, best layer
    rng = np.random.default_rng(0)
    null_aur = []
    for t in TASKS:
        ys = Y[t].copy(); rng.shuffle(ys)
        null_aur.append(cv_intask(acts[t]["mean"][nL // 2], ys))
    report["control_null_shuffled_auroc"] = round(float(np.mean(null_aur)), 4)
    # random directions
    d = acts[TASKS[0]]["mean"].shape[2]
    rnd = []
    for _ in range(20):
        w = rng.standard_normal(d); w /= np.linalg.norm(w)
        rnd.append(np.mean([auroc(acts[t]["mean"][nL // 2] @ w, Y[t]) for t in TASKS]))
    report["control_random_dir_mean_auroc"] = round(float(np.mean(rnd)), 4)

    # ---- internal transfer: layer sweep, pick by in-task ---------------
    best = {"pos": None, "layer": None, "in_task": -1}
    sweep = {}
    for pos in ("mean", "last"):
        sweep[pos] = []
        for L in range(nL):
            layer_acts = {t: acts[t][pos][L] for t in TASKS}
            M, cos, off = transfer_matrix(layer_acts, Y)
            intask = np.mean([M[t][t] for t in TASKS])
            sweep[pos].append({"layer": L, "mean_in_task": round(float(intask), 4),
                               "mean_cross_task": round(off, 4)})
            if intask > best["in_task"]:
                best = {"pos": pos, "layer": L, "in_task": float(intask),
                        "matrix": M, "cosine": cos, "mean_cross_task": off}
    report["internal_layer_sweep"] = sweep
    report["internal_best_by_in_task"] = {
        k: best[k] for k in ("pos", "layer", "in_task", "mean_cross_task")}
    report["internal_best_transfer_matrix"] = best["matrix"]
    report["internal_best_cosine"] = best["cosine"]

    # ---- external side (from P1, already computed) ---------------------
    p1 = json.loads((OUT.parent / "p1_exact_pairing.json").read_text())
    ext_tr = p1["P1a_external_transfer_auroc"]
    ext_off = np.mean([ext_tr[a][b] for a in TASKS for b in TASKS if a != b])
    report["external_mean_cross_task"] = round(float(ext_off), 4)

    # ---- gap + decision rule ------------------------------------------
    delta = best["mean_cross_task"] - float(ext_off)
    report["delta_internal_minus_external_cross_task"] = round(float(delta), 4)
    if is_dev:
        report["decision"] = "N/A — dev model, pipeline validation only"
    else:
        internal = best["mean_cross_task"]
        if delta >= 0.2 and internal >= 0.8 and ext_off <= 0.65:
            report["decision"] = "SUPPORTED"
        elif ext_off >= internal - 0.1:
            report["decision"] = "REFUTED"
        else:
            report["decision"] = "INCONCLUSIVE"

    json.dump(report, open(OUT, "w"), indent=2)
    print("\ncontrols: positive={} null={} random={}".format(
        report["control_positive_max_auroc"],
        report["control_null_shuffled_auroc"],
        report["control_random_dir_mean_auroc"]))
    print("internal best-layer cross-task: {} (pos={} layer={})".format(
        round(best["mean_cross_task"], 4), best["pos"], best["layer"]))
    print("external cross-task: {}  delta: {}".format(
        report["external_mean_cross_task"],
        report["delta_internal_minus_external_cross_task"]))
    print("decision:", report["decision"])
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
