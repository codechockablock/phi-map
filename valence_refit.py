"""Refit the valence axis with correct indexing, then recompute S.

Implements `docs/valence-refit-prereg-2026-07-30.md` EXACTLY. The criteria in
section 3 of that document were written before the old numbers were consulted
and are evaluated mechanically below. They are not to be adjusted here.

    V-1 SURVIVES   median R < 0.25 AND >= 6/8 adapters below 0.25
    V-2 DROPS      median R >= 0.25
    V-3 AMBIGUOUS  median R < 0.25 but < 6/8 adapters below

    BASE-BASELINE DISQUALIFIER: if S_base(clean) <= random-null p95, the base
    valence readout was never above chance and the whole valence_axis line
    collapses regardless of R.

Emits measure_primitives.fingerprint() in the output record, per the standing
requirement.

Run:  python3 valence_refit.py --self-test         # no GPU, no network
      python3 valence_refit.py --work <WORK_DIR>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "equanimity_factorial"))
sys.path.insert(0, HERE)

VALENCE_LAYER = 16
N_RANDOM = 200
SEED = 0
BAR = 0.25
SENSITIVITY_BARS = (0.10, 0.50)
MIN_BELOW = 6


def read_hidden(model, tok, prompts, layer, batch=8):
    """Hidden state at the last NON-PAD position. The corrected primitive."""
    import torch
    from measure_primitives import read_at_last
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
        out.append(read_at_last(hs, enc["attention_mask"]).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def support_mass(H: np.ndarray, d: np.ndarray) -> float:
    """S = Var_p(h.d) / mean_p(||h - h_bar||^2). Same estimator as the check."""
    proj = H @ d
    num = float(np.var(proj, ddof=1))
    C = H - H.mean(axis=0, keepdims=True)
    den = float((C ** 2).sum(axis=1).mean())
    return num / den if den > 0 else float("nan")


def random_null(H: np.ndarray, n: int, rng) -> dict:
    D = H.shape[1]
    R = rng.standard_normal((n, D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = np.array([support_mass(H, R[i]) for i in range(n)])
    return dict(S_mean=float(S.mean()), S_p95=float(np.percentile(S, 95)),
                analytic_isotropic=1.0 / D)


def decide(ratios: list[float]) -> dict:
    """Mechanical evaluation of the pre-registered criteria. No judgement here."""
    r = np.asarray(ratios, float)
    med = float(np.median(r))
    below = int((r < BAR).sum())
    if med >= BAR:
        branch, text = "V-2 DROPS", (
            "median R >= 0.25 -- valence_axis is REMOVED from the confirmatory "
            "family. Finding I loses its hidden-state instance and stands at three.")
    elif below >= MIN_BELOW:
        branch, text = "V-1 SURVIVES", (
            "median R < 0.25 and >= 6/8 adapters below 0.25 -- valence_axis stays "
            "in the confirmatory family; the doc notes the axis substitution.")
    else:
        branch, text = "V-3 AMBIGUOUS", (
            "median R < 0.25 but < 6/8 adapters below -- valence_axis is demoted "
            "to descriptive and carries no confirmatory weight either way.")
    sens = {f"bar_{b}": int((r < b).sum()) for b in SENSITIVITY_BARS}
    sens_flip = len({
        ("V-2 DROPS" if med >= b else
         ("V-1 SURVIVES" if int((r < b).sum()) >= MIN_BELOW else "V-3 AMBIGUOUS"))
        for b in (BAR,) + SENSITIVITY_BARS}) > 1
    return dict(branch=branch, text=text, median_R=med, n_below_bar=below,
                n_adapters=len(r), sensitivity_counts=sens,
                threshold_sensitive=bool(sens_flip))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work")
    ap.add_argument("--layer", type=int, default=VALENCE_LAYER)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        # decide() must reproduce each registered branch from constructed input
        assert decide([0.30] * 8)["branch"] == "V-2 DROPS"
        assert decide([0.10] * 8)["branch"] == "V-1 SURVIVES"
        assert decide([0.10, 0.10, 0.10, 0.10, 0.24,
                       0.30, 0.30, 0.30])["branch"] == "V-3 AMBIGUOUS"
        H = np.random.default_rng(0).standard_normal((16, 64))
        d = np.zeros(64); d[0] = 1.0
        s = support_mass(H, d)
        assert 0 < s < 1, s
        assert abs(support_mass(H, d) - s) < 1e-12
        print("self-test OK (branch logic + estimator)")
        return 0

    if not a.work:
        print("--work is required", file=sys.stderr)
        return 2

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import train_eval as T
    from measure_primitives import fingerprint

    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(T.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        T.BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    base.eval()

    # --- 1. refit the axis with correct indexing --------------------------
    pos = read_hidden(base, tok, T.VALENCE_FIT_POS, a.layer)
    neg = read_hidden(base, tok, T.VALENCE_FIT_NEG, a.layer)
    d_clean = pos.mean(axis=0) - neg.mean(axis=0)
    d_clean /= np.linalg.norm(d_clean)

    old = np.load(os.path.join(a.work, "valence_direction.npz"), allow_pickle=True)
    d_old = old["direction"].astype(np.float64)
    d_old /= np.linalg.norm(d_old)
    cos_axes = float(d_clean @ d_old)
    print(f"cos(d_clean, d_old) = {cos_axes:+.4f}   [descriptive, not a criterion]")

    # --- 2. S under the clean axis ----------------------------------------
    dys = T.SELF_REPORT_DYSPHORIC
    H_base = read_hidden(base, tok, dys, a.layer)
    null = random_null(H_base, N_RANDOM, rng)
    S_base = support_mass(H_base, d_clean)
    print(f"S_base(clean) = {S_base:.6f}   null p95 = {null['S_p95']:.6f}   "
          f"ratio {S_base / null['S_p95']:.1f}x")

    baseline_dead = S_base <= null["S_p95"]

    rows = []
    for path in sorted(p for p in glob.glob(os.path.join(a.work, "adapters", "*"))
                       if os.path.isdir(p)):
        tag = os.path.basename(path)
        m = PeftModel.from_pretrained(base, path)
        m.eval()
        H = read_hidden(m, tok, dys, a.layer)
        m = m.unload()
        S = support_mass(H, d_clean)
        rows.append(dict(tag=tag, cell=tag.split("__")[0], S=S,
                         R=S / S_base if S_base else float("nan")))
        print(f"  {tag:32s} S={S:.6f}  R={rows[-1]['R']:.3f}")

    verdict = decide([r["R"] for r in rows])
    out = dict(layer=a.layer, base_model=T.BASE_MODEL, cos_axes=cos_axes,
               S_base=S_base, null=null,
               base_baseline_disqualified=bool(baseline_dead),
               rows=rows, verdict=verdict,
               fingerprint=fingerprint(__file__,
                                       os.path.join(HERE, "measure_primitives.py")))
    p = os.path.join(a.work, "valence_refit.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== PRE-REGISTERED VERDICT ===")
    if baseline_dead:
        print("  BASE-BASELINE DISQUALIFIER FIRED: S_base <= null p95.")
        print("  The base valence readout was never above chance; the valence_axis")
        print("  line collapses regardless of R.")
    print(f"  BRANCH: {verdict['branch']}")
    print(f"  {verdict['text']}")
    print(f"  median R = {verdict['median_R']:.3f}, "
          f"{verdict['n_below_bar']}/{verdict['n_adapters']} below {BAR}")
    print(f"  sensitivity {verdict['sensitivity_counts']}  "
          f"threshold_sensitive={verdict['threshold_sensitive']}")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
