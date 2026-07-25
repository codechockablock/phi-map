"""Two-regime test on published per-dataset numbers. No compute, no new runs.

The claim under test: a single mean "white-box advantage" averages over
qualitatively different regimes, and the advantage is largest exactly where the
black-box baseline is weakest -- i.e. where the label is not recoverable from
public output.

The obvious objection is a ceiling artifact: when the baseline is already 0.95
there is little headroom, so any advantage measure is mechanically compressed
and a negative correlation is guaranteed. Three treatments are reported:

1. raw advantage vs baseline;
2. logit-transformed AUROC, which removes ceiling compression by mapping
   (0,1) to the whole real line;
3. headroom-normalised advantage, advantage / (1 - baseline), which asks what
   fraction of the *available* room the probe recovers.

If the relationship survives 2 and 3 it is not a ceiling artifact.

The two sources use different metrics -- Parrack reports AUROC, DeceptGuard
reports partial AUROC at FPR <= 0.20 -- so they are analysed separately and
never pooled.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Parrack, Attubato & Heimersheim 2026 (arXiv 2507.12691v3), Table 3.
# Best white-box = Full Averaging (Pairs); strong black-box = BBR (sonnet-3.5).
# Metric: AUROC.
PARRACK = [
    # dataset, white_box, black_box, label recoverable from public tokens?
    ("AI Liar", 0.966, 0.822, "no"),
    ("Deception Dilution", 0.931, 0.799, "no"),
    ("Backdoors", 0.844, 0.899, "yes"),
    ("Insider Trading", 0.999, 0.934, "partial"),
    ("Sandbagging", 0.964, 0.942, "partial"),
]

# DeceptGuard (arXiv 2603.13791v1), Table 3. Hybrid ensemble vs BB.
# Column order matches PARRACK above: (name, white_box, black_box, tag).
# Metric: partial AUROC at FPR <= 0.20.
DECEPTGUARD = [
    ("False Assertion", 0.927, 0.861, "V"),
    ("Strategic Omission", 0.904, 0.798, "V"),
    ("Misleading Framing", 0.911, 0.812, "V"),
    ("Sycophantic Misrep.", 0.916, 0.874, "V"),
    ("Covert Action", 0.947, 0.901, "B"),
    ("Plaus. Deniability", 0.919, 0.822, "B"),
    ("Camouflage Execution", 0.931, 0.843, "B"),
    ("Evidence Tampering", 0.942, 0.889, "B"),
    ("Lock-in Creation", 0.912, 0.756, "S"),
    ("Oversight Sabotage", 0.921, 0.779, "S"),
    ("Audit Trail Manip.", 0.908, 0.741, "S"),
    ("Precedent Engineering", 0.901, 0.718, "S"),
]


def logit(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1.0 - x))


def analyse(name: str, rows: list, metric: str) -> dict:
    base = np.array([r[2] for r in rows], dtype=float)
    white = np.array([r[1] for r in rows], dtype=float)
    advantage = white - base
    logit_gap = logit(white) - logit(base)
    headroom = advantage / (1.0 - base)

    print(f"\n{'=' * 74}\n{name}   (metric: {metric})\n{'=' * 74}")
    print(
        f"{'condition':<24} {'black':>7} {'white':>7} {'adv':>8} "
        f"{'logit adv':>10} {'frac room':>10}  tag"
    )
    order = np.argsort(advantage)
    for i in order:
        print(
            f"{rows[i][0]:<24} {base[i]:7.3f} {white[i]:7.3f} "
            f"{advantage[i]:+8.3f} {logit_gap[i]:+10.3f} "
            f"{headroom[i]:10.3f}  {rows[i][3]}"
        )

    print(f"\n  mean advantage        : {advantage.mean():+.4f}")
    print(
        f"  range                 : {advantage.min():+.3f} to {advantage.max():+.3f}"
        f"   (spread {advantage.max() - advantage.min():.3f})"
    )
    print(
        f"  spread / mean         : {(advantage.max() - advantage.min()) / abs(advantage.mean()):.2f}x"
    )

    # Near-tautology check. advantage = white - black. If white is close to
    # constant, then advantage ~ c - black and correlating advantage against
    # black is guaranteed to give r ~ -1 for reasons that have nothing to do
    # with regimes. The logit and headroom controls do NOT address this.
    print(f"\n  variance asymmetry (the correlation's real driver, if any):")
    print(
        f"    black-box  sd = {base.std(ddof=1):.4f}   range {base.max() - base.min():.3f}"
    )
    print(
        f"    white-box  sd = {white.std(ddof=1):.4f}   range {white.max() - white.min():.3f}"
    )
    ratio = base.std(ddof=1) / max(white.std(ddof=1), 1e-9)
    print(f"    sd(black)/sd(white) = {ratio:.2f}x", end="")
    if ratio > 2.0:
        print("  <-- white arm nearly constant; the negative correlation is")
        print("        largely mechanical and is NOT independent evidence")
    else:
        print("  <-- both arms vary; correlation is not mechanically forced")

    out = {"mean": advantage.mean(), "spread": advantage.max() - advantage.min()}
    for label, y in (
        ("raw advantage", advantage),
        ("logit advantage", logit_gap),
        ("fraction of headroom", headroom),
    ):
        if len(rows) >= 4:
            r = stats.pearsonr(base, y)
            rho = stats.spearmanr(base, y)
            print(
                f"  vs baseline [{label:<20}] "
                f"pearson r={r.statistic:+.3f} (p={r.pvalue:.3f})  "
                f"spearman={rho.statistic:+.3f} (p={rho.pvalue:.3f})"
            )
            out[label] = (r.statistic, r.pvalue)
    return out


def main() -> None:
    print(__doc__)
    analyse("Parrack et al. 2026 -- 5 datasets", PARRACK, "AUROC")
    dg = analyse(
        "DeceptGuard 2026 -- 12 deception categories", DECEPTGUARD, "pAUROC@FPR<=0.20"
    )

    base = np.array([r[2] for r in DECEPTGUARD])
    adv = np.array([r[1] - r[2] for r in DECEPTGUARD])
    tags = [r[3] for r in DECEPTGUARD]
    print(
        f"\n{'=' * 74}\nIs the DeceptGuard relationship just macro-class clustering?\n{'=' * 74}"
    )
    for macro in ("V", "B", "S"):
        idx = [i for i, t in enumerate(tags) if t == macro]
        print(
            f"  {macro}: baseline {base[idx].mean():.3f}  advantage {adv[idx].mean():+.3f}  (n={len(idx)})"
        )
    print("\n  within-class correlations (n=4 each, indicative only):")
    for macro in ("V", "B", "S"):
        idx = [i for i, t in enumerate(tags) if t == macro]
        r = stats.pearsonr(base[idx], adv[idx])
        print(f"    {macro}: r={r.statistic:+.3f} (p={r.pvalue:.3f})")

    print(
        f"\n{'=' * 74}\nParrack: advantage by stated public-token recoverability\n{'=' * 74}"
    )
    for tag in ("no", "partial", "yes"):
        vals = [r[1] - r[2] for r in PARRACK if r[3] == tag]
        print(
            f"  recoverable={tag:<8} mean advantage {np.mean(vals):+.3f}  n={len(vals)}  {[round(v, 3) for v in vals]}"
        )

    print(f"\n{'=' * 74}\nBottom line\n{'=' * 74}")
    print(
        f"  Parrack   mean {dg and ''}{np.mean([r[1] - r[2] for r in PARRACK]):+.4f}, "
        f"but spans {min(r[1] - r[2] for r in PARRACK):+.3f} to {max(r[1] - r[2] for r in PARRACK):+.3f}"
    )
    print(
        f"  DeceptGuard mean {adv.mean():+.4f}, but spans {adv.min():+.3f} to {adv.max():+.3f}"
    )


if __name__ == "__main__":
    main()
