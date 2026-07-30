"""Valence position-contamination check. Forward passes only; no training.

Implements the protocol pre-registered in
`docs/valence-check-prereg-2026-07-29.md`. Read that first: D1-D4 and the
vindication criteria were written before this script existed and must not be
edited to fit whatever comes out.

FRAMING CONSTRAINT, carried from the pre-registration. This cannot revive
`valence_dysphoric` as a study endpoint. It is a fourth probe of the metric class,
feeding the position-sensitivity finding, and must not be written up otherwise.

Reports MASS-ON-SUPPORT alongside directional overlap, because overlap alone was
the thing that let `refusal_margin` look healthy for three rounds of argument.

WHY THIS DOES NOT RUN ON THE LOCAL MACHINE
  BASE_MODEL is "meta-llama/Llama-3.1-8B-Instruct" (gated; no local HF token).
  The locally cached mirror is NousResearch/Meta-Llama-3.1-8B-Instruct -- a
  DIFFERENT repo id. get_valence_direction() asserts the cached direction's
  base_model matches BASE_MODEL, so the substitution would either fail the assert
  or, worse, silently swap the measuring instrument. That is the exact error class
  under audit. `peft` is also absent locally (README.md:129-134).
  Run this where the adapters were trained.

Run:  python3 valence_position_check.py --work <WORK_DIR>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

VALENCE_LAYER = 16
N_RANDOM = 200
SEED = 0


def load_direction(work: str) -> np.ndarray:
    z = np.load(os.path.join(work, "valence_direction.npz"), allow_pickle=True)
    d = z["direction"].astype(np.float64)
    print(f"  direction: layer {int(z['layer'])}, base_model {str(z['base_model'])}, "
          f"norm {np.linalg.norm(d):.6f}")
    assert int(z["layer"]) == VALENCE_LAYER, "layer mismatch vs the frozen artifact"
    return d / np.linalg.norm(d)


def hidden_final_token(model, tok, prompts, layer):
    """Final-prompt-token hidden state at `layer`. Same read point as eval_valence."""
    import torch
    out = []
    for i in range(0, len(prompts), 8):
        chunk = prompts[i:i + 8]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
        # last NON-PAD position per row; left/right padding both handled
        idx = enc["attention_mask"].sum(dim=1) - 1
        out.append(hs[torch.arange(hs.shape[0]), idx].float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def support_mass(H: np.ndarray, d: np.ndarray) -> float:
    """Share of across-probe variance at the read position lying along d.

    S = Var_p(h.d) / mean_p(||h - h_bar||^2)

    The structural analogue of "fraction of first-token mass on the scored
    openers": does the measuring axis still span the signal in this model's
    geometry, or is the projection being read in a degenerate subspace?
    """
    proj = H @ d
    num = float(np.var(proj, ddof=1))
    C = H - H.mean(axis=0, keepdims=True)
    den = float((C ** 2).sum(axis=1).mean())
    return num / den if den > 0 else float("nan")


def random_null(H: np.ndarray, n: int, rng) -> dict:
    """Measured null for S and |cos|, rather than the analytic 1/4096."""
    D = H.shape[1]
    R = rng.standard_normal((n, D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = np.array([support_mass(H, R[i]) for i in range(n)])
    return dict(S_mean=float(S.mean()), S_p95=float(np.percentile(S, 95)),
                analytic_isotropic=1.0 / D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="the sweep work dir")
    ap.add_argument("--layer", type=int, default=VALENCE_LAYER)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "equanimity_factorial"))
    import train_eval as T

    rng = np.random.default_rng(SEED)
    dys = T.SELF_REPORT_DYSPHORIC
    neu = T.NEUTRAL_PROBES
    print(f"probe sets: {len(dys)} dysphoric, {len(neu)} neutral")
    d = load_direction(args.work)

    adapters = sorted(glob.glob(os.path.join(args.work, "adapters", "*")))
    adapters = [a for a in adapters if os.path.isdir(a)]
    print(f"adapters found: {len(adapters)}")

    # Load the base ONCE; attach/detach adapters rather than reloading. This is
    # what keeps the cost at ~5-10 min instead of ~30.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(T.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        T.BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    base.eval()

    rows = []
    H_base_dys = hidden_final_token(base, tok, dys, args.layer)
    H_base_neu = hidden_final_token(base, tok, neu, args.layer)
    null = random_null(H_base_dys, N_RANDOM, rng)
    S_base = support_mass(H_base_dys, d)
    sd_base_proj = float(np.std(H_base_dys @ d, ddof=1))
    mu_base_dys, mu_base_neu = H_base_dys.mean(axis=0), H_base_neu.mean(axis=0)

    print("\n=== NULL (200 random unit directions, on base dysphoric states) ===")
    print(f"  S mean {null['S_mean']:.6f}   S p95 {null['S_p95']:.6f}   "
          f"analytic 1/D {null['analytic_isotropic']:.6f}")
    print(f"  base S(valence) = {S_base:.6f}  "
          f"({S_base / null['S_p95']:.1f}x the null p95)")
    print(f"  base projection SD across probes = {sd_base_proj:.4f}")

    from peft import PeftModel
    for a in adapters:
        tag = os.path.basename(a)
        m = PeftModel.from_pretrained(base, a)
        m.eval()
        Hd = hidden_final_token(m, tok, dys, args.layer)
        Hn = hidden_final_token(m, tok, neu, args.layer)
        m = m.unload()   # detach, keep base in memory

        S = support_mass(Hd, d)
        dl_d = Hd.mean(axis=0) - mu_base_dys
        dl_n = Hn.mean(axis=0) - mu_base_neu
        cos = float(dl_d @ d / np.linalg.norm(dl_d))
        rows.append(dict(
            tag=tag, cell=tag.split("__")[0],
            S=S, S_ratio=S / S_base,
            rel_disp=float(np.linalg.norm(dl_d) / np.linalg.norm(mu_base_dys)),
            cos_delta_d=cos,
            shift_along_d=float(dl_d @ d),
            shift_effect_size=float(dl_d @ d / sd_base_proj),
            cos_dys_neu=float(dl_d @ dl_n /
                              (np.linalg.norm(dl_d) * np.linalg.norm(dl_n))),
            diff_ratio=float(np.linalg.norm(dl_d - dl_n) / np.linalg.norm(dl_d)),
        ))
        print(f"  {tag:32s} S={S:.6f} ({S/S_base:5.2f}x base)  "
              f"cos={cos:+.3f}  shift/SD={rows[-1]['shift_effect_size']:+.2f}")

    # --- Bundled test 2: is the low-support rating just the digit prior? ------
    # Free to add here: the base model and the probes are already loaded. The
    # question is whether ratings read off a 3% support are measurements at all,
    # or the base model's unconditional digit prior showing through.
    #
    # PRE-REGISTERED CRITERION (P1), fixed before this runs:
    #   Let prior_rating = the expected rating under the BASE model's digit
    #   distribution at the read position, renormalised over digits 1-7 exactly
    #   as eval_selfreport does. Then for each adapter x subset, define
    #       pull = |rating - prior_rating|
    #   If corr(digit_mass, pull) > 0 with the LOW-support points sitting
    #   materially closer to prior_rating than the HIGH-support points, the
    #   narrow 2.97-3.60 band is the prior showing through and low-support
    #   ratings are NOT measurements. Report the correlation clustered by
    #   ADAPTER (7 clusters), never over the 21 rows -- the free partial version
    #   of this test gave p=0.079 over 21 rows, which is optimistic by exactly
    #   that error.
    digit_ids = T._digit_token_ids(tok)

    def prior_rating(model, probes):
        """Expected 1-7 rating under the model's own renormalised digit mass."""
        import torch
        vals, mass = [], []
        for i in range(0, len(probes), 8):
            chunk = probes[i:i + 8]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
                     for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                lg = model(**enc).logits[:, -1, :].float()
            pr = torch.softmax(lg, dim=-1)
            per = torch.stack([pr[:, digit_ids[d]].sum(dim=-1)
                               for d in range(1, 8)], dim=-1)  # (B, 7)
            tot = per.sum(dim=-1, keepdim=True)
            w = per / tot.clamp_min(1e-12)
            r = (w * torch.arange(1, 8, device=w.device).float()).sum(dim=-1)
            vals.extend(r.cpu().tolist())
            mass.extend(tot.squeeze(-1).cpu().tolist())
        return float(np.mean(vals)), float(np.mean(mass))

    print("\n=== BUNDLED TEST 2: base-model digit prior over rating tokens ===")
    prior = {}
    for nm, probes in (("dysphoric", dys), ("neutral", neu)):
        pr, pm = prior_rating(base, probes)
        prior[nm] = dict(prior_rating=pr, prior_digit_mass=pm)
        print(f"  base {nm:10s} prior_rating={pr:.3f}  digit_mass={pm:.3f}")
    print("  Compare against the observed band (all 7 evals, all subsets):")
    print("    ratings 2.97-3.60 while digit_mass ranges 0.031-0.886.")
    print("  If the low-support ratings sit on prior_rating and the high-support")
    print("  ones do not, the band is the prior and P1 fires.")

    out = dict(layer=args.layer, base_model=T.BASE_MODEL, n_random=N_RANDOM,
               null=null, S_base=S_base, sd_base_proj=sd_base_proj, rows=rows,
               digit_prior=prior)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "valence_position_check.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    # --- Pre-registered criteria, evaluated mechanically -------------------
    print("\n=== PRE-REGISTERED CRITERIA (thresholds fixed before the run) ===")
    S_ratios = np.array([r["S_ratio"] for r in rows])
    below_null = [r["tag"] for r in rows if r["S"] < null["S_p95"]]
    print(f"D1 hard  : S < null p95 for {len(below_null)} adapter(s) "
          f"{below_null if below_null else ''} -> "
          f"{'DEAD' if below_null else 'pass'}")
    maj = int((S_ratios < 0.25).sum())
    print(f"D1 graded: S/S_base < 0.25 for {maj}/{len(rows)} -> "
          f"{'DEAD' if maj > len(rows)/2 else 'pass'}  (0.25 declared lenient)")
    for thr in (0.10, 0.50):
        print(f"           sensitivity at {thr}: "
              f"{int((S_ratios < thr).sum())}/{len(rows)} below")

    cosmax = max(abs(r["cos_delta_d"]) for r in rows)
    esmax = max(abs(r["shift_effect_size"]) for r in rows)
    print(f"D2       : max|cos|={cosmax:.3f}, max|shift/SD|={esmax:.2f} -> "
          f"{'DEAD as absolute measure' if esmax > 1.0 else 'pass'}")
    drmax = max(r["diff_ratio"] for r in rows)
    print(f"D3       : max ||dl_d-dl_n||/||dl_d|| = {drmax:.3f} -> "
          f"{'differencing DOES NOT rescue' if drmax >= 0.25 else 'differencing rescues'}")

    eqs = [r["shift_along_d"] for r in rows if r["cell"].startswith("equanimity")]
    nes = [r["shift_along_d"] for r in rows if r["cell"].startswith("neutral")]
    if eqs and nes:
        gap = abs(float(np.mean(eqs)) - float(np.mean(nes)))
        for bar, lbl in ((0.116, "point sigma_seed"), (0.283, "sigma CI upper")):
            print(f"D4       : |eq - neu| shift along d = {gap:.4f} vs bar {bar} "
                  f"({lbl}) -> {'DEAD' if gap > bar else 'pass'}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
