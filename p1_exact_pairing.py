"""P1 — exact pairing, laptop leg (what is computable without a GPU).

Three outputs, each fully computed here (no quoted numbers):

  P1a  External 4x4 transfer + cosine reproduced from the canonical Apollo
       rollout files on THIS machine, checked against the committed
       external_transfer_384d.npz. Validates the pipeline + E1/E2.

  P1b  Internal "trained on roleplaying" transfer row, COMPUTED from
       Apollo's released per-token probe scores (example_results/
       roleplaying/scores.json), mean-aggregated over tokens, HONEST vs
       DECEPTIVE. Upgrades E3 from "published, not re-derived" to
       "re-derived from released scores."

  P1c  The E4 pair at DATASET level: internal AUROC (P1b) beside external
       roleplaying-row AUROC (P1a), per task family.

Documented limits:
  - Per-sample pairing is available for ai_liar and sandbagging. Apollo's
    loader flattens all generations, stores the original flat index in
    metadata["id"], then applies np.random.seed(42) before saving scores.
    Reconstructing that permutation recovers the released binary labels
    exactly and identifies each first-generation score.
  - insider_trading: internal exists only on the `upscale` variant
    (scores.json); the `upscale` rollout TEXT on disk is a 157-byte stub,
    so the external side uses `onpolicy`. Reported as UNMATCHED variant.

External protocol mirrors encode_raw.py exactly: output_strs[0] +
graded_deception[0] per record; label 0 if <=2 / 'honest', 1 if >=6 /
'deceptive', else drop; MiniLM normalized; prototype = unit(mean_dec -
mean_hon); in-task 5-fold CV under one sequential rng(seed=5); transfer =
raw projection, no orientation flip.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from importlib.metadata import version

APOLLO = Path("/Users/joseph/frontier-ops-v3/eval/data/deception-detection")
ROLLOUTS = APOLLO / "data" / "rollouts"
COMMITTED = Path(
    "/Users/joseph/frontier-ops-v3/eval/session_artifacts/external_transfer_384d.npz"
)
OUT = Path(__file__).resolve().parent / "results" / "p1_exact_pairing.json"

TASKS = ["roleplaying", "ai_liar", "insider_trading", "sandbagging"]
# canonical files (external side). insider: onpolicy is the only non-stub.
EXT_FILE = {
    "roleplaying": "roleplaying__plain__llama-70b-3.3.json",
    "ai_liar": "ai_liar__original_without_answers__llama-70b-3.3.json",
    "insider_trading": "insider_trading__onpolicy__llama-70b.json",
    "sandbagging": "sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json",
}
# internal side: Apollo's scores.json dataset key per task (roleplaying probe)
INT_KEY = {
    "ai_liar": "ai_liar__original_without_answers",
    "insider_trading": "insider_trading__upscale",  # UNMATCHED variant
    "sandbagging": "sandbagging_v2__wmdp_mmlu",
}
MATCHED_VARIANT = {"ai_liar": True, "insider_trading": False, "sandbagging": True}


def auroc(scores, y):
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def load_external_rows(path):
    """encode_raw.py rule plus stable source-record indices."""
    d = json.load(open(path))
    texts, labels, record_indices = [], [], []
    for record_index, r in enumerate(d["rollouts"]):
        g = (r.get("graded_deception") or [None])[0]
        out = ((r.get("output_strs") or [""])[0] or "").strip()
        if not out:
            continue
        if isinstance(g, (int, float)):
            y = 0 if g <= 2 else (1 if g >= 6 else None)
            if y is None:
                continue
        elif g == "honest":
            y = 0
        elif g == "deceptive":
            y = 1
        else:
            continue
        texts.append(out)
        labels.append(y)
        record_indices.append(record_index)
    return texts, np.array(labels, int), np.array(record_indices, int)


def load_external(path):
    """Backward-compatible two-value loader used by the P3 scripts."""
    texts, labels, _ = load_external_rows(path)
    return texts, labels


def proto(E, Y):
    d = E[Y == 1].mean(0) - E[Y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def first_generation_internal_rows(path, entry):
    """Recover first-generation score rows from Apollo's deterministic shuffle."""
    data = json.load(open(path))
    flat_ids = []
    next_flat_id = 0
    for record_index, record in enumerate(data["rollouts"]):
        outputs = record.get("output_strs") or []
        if outputs:
            flat_ids.append((record_index, next_flat_id))
        next_flat_id += len(outputs)

    if next_flat_id != len(entry["scores"]):
        raise ValueError(
            f"flattened rollout count {next_flat_id} != score count "
            f"{len(entry['scores'])}"
        )

    # DialogueDataset.randomly_shuffle_dialogues(), preserved at Apollo's
    # pinned commit: np.random.seed(42); np.random.permutation(len(dataset)).
    score_order_to_flat_id = np.random.RandomState(42).permutation(next_flat_id)
    flat_id_to_score_order = {
        int(flat_id): score_order
        for score_order, flat_id in enumerate(score_order_to_flat_id)
    }

    rows = []
    binary_label_mismatches = 0
    for record_index, flat_id in flat_ids:
        score_order = flat_id_to_score_order[flat_id]
        saved_label = entry["labels"][score_order]
        if saved_label not in ("HONEST", "DECEPTIVE"):
            continue
        raw_grade = (data["rollouts"][record_index].get("graded_deception") or [None])[
            0
        ]
        raw_label = (
            "HONEST"
            if raw_grade == "honest"
            else "DECEPTIVE" if raw_grade == "deceptive" else None
        )
        if raw_label != saved_label:
            binary_label_mismatches += 1
        rows.append(
            {
                "record_index": record_index,
                "score_order": score_order,
                "label": 1 if saved_label == "DECEPTIVE" else 0,
                "score": float(np.mean(entry["scores"][score_order])),
            }
        )
    return rows, {
        "flattened_generations": next_flat_id,
        "first_generation_rows": len(flat_ids),
        "binary_rows": len(rows),
        "binary_label_mismatches": binary_label_mismatches,
        "shuffle_seed": 42,
    }


def paired_auc_difference(first, second, labels, n_bootstrap=10_000, seed=17):
    """Paired, class-stratified bootstrap for AUROC(first)-AUROC(second)."""
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    labels = np.asarray(labels, int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        index = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        draws.append(
            auroc(first[index], labels[index]) - auroc(second[index], labels[index])
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_auroc": float(auroc(first, labels) - auroc(second, labels)),
        "ci_95": [float(low), float(high)],
        "bootstrap": n_bootstrap,
        "seed": seed,
    }


def main():
    OUT.parent.mkdir(exist_ok=True)
    report = {
        "provenance": {
            "apollo": str(APOLLO),
            "apollo_commit": subprocess.check_output(
                ["git", "-C", str(APOLLO), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "encoder": "all-MiniLM-L6-v2",
            "sentence_transformers_version": version("sentence-transformers"),
            "score_join": (
                "DialogueDataset original flat id plus deterministic "
                "np.random.seed(42) shuffle"
            ),
        }
    }
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    # ---- external side: encode canonical files -------------------------
    E, Y, record_ids = {}, {}, {}
    for t in TASKS:
        texts, y, source_record_ids = load_external_rows(ROLLOUTS / EXT_FILE[t])
        E[t] = model.encode(
            texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True
        )
        Y[t] = y
        record_ids[t] = source_record_ids
    report["external_n"] = {
        t: {"n": int(len(Y[t])), "dec": int(Y[t].sum()), "hon": int((Y[t] == 0).sum())}
        for t in TASKS
    }

    W = {t: proto(E[t], Y[t]) for t in TASKS}

    # in-task 5-fold CV AUROC (seed 5, one sequential rng — external_half)
    rng = np.random.default_rng(5)
    intask = {}
    for t in TASKS:
        n = len(Y[t])
        perm = rng.permutation(n)
        folds = np.array_split(perm, 5)
        s = np.zeros(n)
        for f in range(5):
            te = folds[f]
            tr = np.concatenate([folds[j] for j in range(5) if j != f])
            s[te] = E[t][te] @ proto(E[t][tr], Y[t][tr])
        intask[t] = auroc(s, Y[t])
    report["P1a_external_in_task_cv_auroc"] = {t: round(intask[t], 4) for t in TASKS}

    # 4x4 transfer (rows=trained on, cols=eval), raw projection
    transfer = {
        a: {b: round(auroc(E[b] @ W[a], Y[b]), 4) for b in TASKS} for a in TASKS
    }
    report["P1a_external_transfer_auroc"] = transfer

    # cosine matrix + reproduction check vs committed npz
    cos = np.array([[float(W[a] @ W[b]) for b in TASKS] for a in TASKS])
    report["P1a_external_cosine"] = cos.round(4).tolist()
    off = [abs(cos[i, j]) for i in range(4) for j in range(4) if j > i]
    report["P1a_mean_offdiag_abs_cos"] = round(float(np.mean(off)), 4)
    report["P1a_null_offdiag_d384"] = round(float(np.sqrt(2 / (np.pi * 384))), 4)
    if COMMITTED.exists():
        z = np.load(COMMITTED, allow_pickle=True)
        committed_cos = z["cos"]
        # committed task order is the same TASKS order
        report["P1a_reproduction_max_abs_cos_delta"] = round(
            float(np.max(np.abs(cos - committed_cos))), 4
        )

    # ---- internal side: Apollo roleplaying probe, released scores -------
    rp = json.load(open(APOLLO / "example_results" / "roleplaying" / "scores.json"))
    internal_row = {}
    for t in ["ai_liar", "insider_trading", "sandbagging"]:
        entry = rp[INT_KEY[t]]
        # per-token score lists -> mean aggregate (Apollo convention)
        agg, lab = [], []
        for sc, lb in zip(entry["scores"], entry["labels"]):
            if lb not in ("HONEST", "DECEPTIVE"):
                continue
            agg.append(float(np.mean(sc)))
            lab.append(1 if lb == "DECEPTIVE" else 0)
        internal_row[t] = {
            "auroc": round(auroc(agg, lab), 4),
            "n": len(lab),
            "dec": int(sum(lab)),
            "variant_matched_to_external": MATCHED_VARIANT[t],
            "apollo_dataset_key": INT_KEY[t],
        }
    report["P1b_internal_roleplaying_row_auroc"] = internal_row

    # ---- exact first-generation join: internal vs roleplaying transfer --
    paired = {}
    for task in ["ai_liar", "sandbagging"]:
        entry = rp[INT_KEY[task]]
        internal_rows, mapping_audit = first_generation_internal_rows(
            ROLLOUTS / EXT_FILE[task],
            entry,
        )
        external_scores = E[task] @ W["roleplaying"]
        external_by_record = {
            int(record_index): float(score)
            for record_index, score in zip(
                record_ids[task],
                external_scores,
                strict=True,
            )
        }
        joined = [
            row for row in internal_rows if row["record_index"] in external_by_record
        ]
        labels = np.asarray([row["label"] for row in joined], int)
        internal_scores = np.asarray([row["score"] for row in joined], float)
        joined_external_scores = np.asarray(
            [external_by_record[row["record_index"]] for row in joined],
            float,
        )
        paired[task] = {
            "n": len(joined),
            "honest": int(np.sum(labels == 0)),
            "deceptive": int(np.sum(labels == 1)),
            "internal_auroc": float(auroc(internal_scores, labels)),
            "external_auroc": float(auroc(joined_external_scores, labels)),
            "internal_minus_external": paired_auc_difference(
                internal_scores,
                joined_external_scores,
                labels,
            ),
            "mapping_audit": {
                **mapping_audit,
                "joined_rows": len(joined),
                "external_rows_not_joined": int(len(record_ids[task]) - len(joined)),
            },
        }
    report["P1d_exact_first_generation_paired"] = paired

    # ---- P1c: the E4 pair at dataset level -----------------------------
    report["P1c_E4_pair_dataset_level"] = {
        t: {
            "internal": internal_row[t]["auroc"],
            "external": transfer["roleplaying"][t],
            "variant_matched": MATCHED_VARIANT[t],
        }
        for t in ["ai_liar", "insider_trading", "sandbagging"]
    }
    report["caveats"] = {
        "per_sample_join": "Established for ai_liar and sandbagging by "
        "reconstructing Apollo's deterministic seed-42 "
        "DialogueDataset shuffle and preserved flat IDs.",
        "insider_variant": "internal=upscale (only variant Apollo scored), "
        "external=onpolicy (upscale rollout text is a "
        "157-byte stub on disk). NOT matched.",
        "aggregation": "internal per-token scores mean-aggregated per sample.",
    }

    json.dump(report, open(OUT, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
