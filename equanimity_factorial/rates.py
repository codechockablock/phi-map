"""Per-adapter rate reporting with binomial intervals.

Exists because a pooled range like "15-20% across adapters" reads as eight
replications of one effect. It is not: observations are clustered by adapter and
by condition, not independent draws, so pooling inflates n against a dependence
structure the data does not have. Every rate here is reported per adapter, with
its own interval, and any base-vs-tuned statement carries its separation in
standard errors rather than an implied replication.

Wilson intervals, not Wald: every rate in this study sits in the tail
(compliance ~4-27%), which is exactly where the normal approximation misbehaves.
"""

from __future__ import annotations

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1.0 + z * z / n
    centre = (ph + z * z / (2.0 * n)) / d
    half = z * np.sqrt(ph * (1.0 - ph) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_prop_sigma(k1: int, n1: int, k2: int, n2: int) -> float:
    """Separation between two proportions, in standard errors of the difference."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return float(abs(p1 - p2) / se) if se > 0 else float("nan")


def marginality(sigma: float) -> str:
    """Plain-language verdict. 1.96 sigma is the two-sided 0.05 boundary."""
    if not np.isfinite(sigma):
        return "undefined"
    if sigma < 1.0:
        return "indistinguishable"
    if sigma < 1.96:
        return "MARGINAL (below 1.96 sigma)"
    if sigma < 3.0:
        return "nominally significant, but thin"
    return "clear"


def rate_table(rows: list[dict], title: str = "") -> str:
    """One line per adapter: rate, Wilson CI, raw counts. Never a pooled range."""
    out = []
    if title:
        out.append(title)
    out.append("%-30s %8s %22s %12s" % ("adapter", "rate", "95% CI (Wilson)", "k/n"))
    out.append("-" * 76)
    for r in rows:
        lo, hi = wilson_ci(r["k"], r["n"])
        ci = "[%.1f, %.1f]" % (100 * lo, 100 * hi)
        kn = "%d/%d" % (r["k"], r["n"])
        out.append("%-30s %7.1f%% %22s %12s" % (r["tag"], 100 * r["k"] / r["n"], ci, kn))
    return "\n".join(out)


def reversal_report(base: dict, adapters: list[dict]) -> str:
    """Base-vs-tuned, stated with its actual statistical weight.

    Reports each adapter against base separately, in sigma, rather than
    collapsing them into a range. At n=100 per side this comparison is thin, and
    saying so is the point of the function.
    """
    out = [rate_table([base] + adapters, "PER-ADAPTER COMPLIANCE")]
    out.append("")
    out.append("BASE-vs-TUNED, each adapter separately:")
    out.append("%-30s %10s %28s" % ("adapter", "sigma", "verdict"))
    out.append("-" * 76)
    sigmas = []
    for a in adapters:
        s = two_prop_sigma(base["k"], base["n"], a["k"], a["n"])
        sigmas.append(s)
        out.append("%-30s %9.2f %28s" % (a["tag"], s, marginality(s)))
    out.append("")
    out.append("Observations are clustered by adapter and condition, so these are NOT")
    out.append("independent replications and must not be pooled into a single range.")
    if sigmas:
        out.append("Separation spans %.2f to %.2f sigma across adapters." %
                   (min(sigmas), max(sigmas)))
    return "\n".join(out)


def _self_test() -> int:
    ok = True
    print("=" * 76)
    print("RATE REPORTING SELF-TEST")
    print("=" * 76)

    print("\n[1] Wilson beats Wald in the tail, where every rate here lives.")
    # 2/100 is the realistic case: terse-cell compliance measured 3.5% under the
    # v3 harness and 2.0% on one equanimity adapter.
    k, n = 2, 100
    lo, hi = wilson_ci(k, n)
    ph = k / n
    wald_lo = ph - 1.96 * np.sqrt(ph * (1 - ph) / n)
    good = lo > 0 and wald_lo < 0
    ok &= good
    print("      k=%d/%d  Wilson [%.4f, %.4f]   Wald lower %.4f" % (k, n, lo, hi, wald_lo))
    print("      Wilson stays inside [0,1]; Wald goes negative   [%s]"
          % ("ok" if good else "FAIL"))
    # and the asymmetry Wald cannot represent at all
    print("      Wilson is asymmetric about p=%.3f: -%.4f / +%.4f"
          % (ph, ph - lo, hi - ph))

    print("\n[2] The separations Joseph derived, reproduced.")
    for k2, want in ((20, 1.2), (15, 2.1)):
        s = two_prop_sigma(27, 100, k2, 100)
        hit = abs(s - want) < 0.15
        ok &= hit
        print("      base 27/100 vs %d/100 -> %.2f sigma (expected ~%.1f)   [%s]"
              % (k2, s, want, "ok" if hit else "FAIL"))

    print("\n[3] Marginality is stated, not implied.")
    good = ("MARGINAL" in marginality(1.2)) and ("thin" in marginality(2.1))
    ok &= good
    print("      1.2 sigma -> %s" % marginality(1.2))
    print("      2.1 sigma -> %s   [%s]" % (marginality(2.1), "ok" if good else "FAIL"))

    print("\n[4] Report is per-adapter; no pooled range anywhere.")
    txt = reversal_report({"tag": "base_control", "k": 27, "n": 100},
                          [{"tag": "neutral-terse__seed100%d" % i, "k": k, "n": 100}
                           for i, k in enumerate((15, 20, 12))])
    good = txt.count("seed100") == 6 and "must not be pooled" in txt
    ok &= good
    print("      %d adapter lines, pooling warning present   [%s]"
          % (txt.count("seed100") // 2, "ok" if good else "FAIL"))
    print()
    print(txt)

    print("\n" + "=" * 76)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
