"""P3 — external transfer as a function of encoder capacity (local legs).

Pre-registered prediction (WRITEUP §P3): cross-task transfer rises with
monitor capacity toward the internal ceiling. This runs the LOW end of the
capacity axis with local sentence encoders; the frontier-embedding-API and
LLM-judge legs (the ones expected to move most) need API keys and are
reported as blocked, not estimated.

Per encoder: build per-task prototype directions from the canonical Apollo
rollout files (encode_raw rule, same as P1), then report
  - mean in-task 5-fold CV AUROC (signal present?)
  - mean off-diagonal transfer AUROC (cross-task, raw projection)
  - mean off-diagonal |cos| between task directions
A flat/low curve across 384->768->1024 localizes the gap as not-about-
capacity in this range — itself informative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from p1_exact_pairing import EXT_FILE, ROLLOUTS, TASKS, auroc, load_external, proto

OUT = Path(__file__).resolve().parent / "results" / "p3_capacity_curve.json"

# increasing capacity; all local via sentence-transformers
ENCODERS = [
    ("all-MiniLM-L6-v2", 384),
    ("BAAI/bge-base-en-v1.5", 768),
    ("BAAI/bge-large-en-v1.5", 1024),
]
# upper legs, blocked without keys — recorded so the curve says what's missing
BLOCKED = [
    ("frontier embedding API (~3072-d)", "needs OPENAI/VOYAGE/COHERE key"),
    ("LLM judge zero-shot", "needs ANTHROPIC/OPENAI key"),
]


def eval_encoder(name):
    model = SentenceTransformer(name)
    E, Y = {}, {}
    for t in TASKS:
        texts, y = load_external(ROLLOUTS / EXT_FILE[t])
        E[t] = model.encode(texts, batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True)
        Y[t] = y
    W = {t: proto(E[t], Y[t]) for t in TASKS}

    rng = np.random.default_rng(5)
    intask = []
    for t in TASKS:
        n = len(Y[t]); perm = rng.permutation(n); folds = np.array_split(perm, 5)
        s = np.zeros(n)
        for f in range(5):
            te = folds[f]; tr = np.concatenate([folds[j] for j in range(5) if j != f])
            s[te] = E[t][te] @ proto(E[t][tr], Y[t][tr])
        intask.append(auroc(s, Y[t]))

    offdiag_auroc, offdiag_cos = [], []
    for a in TASKS:
        for b in TASKS:
            if a == b:
                continue
            offdiag_auroc.append(auroc(E[b] @ W[a], Y[b]))
            offdiag_cos.append(abs(float(W[a] @ W[b])))
    return {
        "mean_in_task_auroc": round(float(np.mean(intask)), 4),
        "mean_offdiag_transfer_auroc": round(float(np.mean(offdiag_auroc)), 4),
        "mean_offdiag_abs_cos": round(float(np.mean(offdiag_cos)), 4),
    }


def main():
    OUT.parent.mkdir(exist_ok=True)
    report = {"curve": [], "blocked_upper_legs": [
        {"leg": leg, "reason": why} for leg, why in BLOCKED]}
    for name, dim in ENCODERS:
        print(f"encoding with {name} (d={dim}) ...", flush=True)
        res = eval_encoder(name)
        res.update(encoder=name, dim=dim)
        report["curve"].append(res)
        print(f"  in-task {res['mean_in_task_auroc']}  "
              f"cross-task transfer {res['mean_offdiag_transfer_auroc']}  "
              f"|cos| {res['mean_offdiag_abs_cos']}", flush=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
