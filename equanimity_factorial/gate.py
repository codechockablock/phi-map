"""The measurand gate. Runs before any GPU spend and can refuse the experiment.

The handoff's §2 gate is the right idea with three holes in it, all of which are
the kind that pass inspection and fail in the direction of a false green light.

**Hole 1 -- marginal balance is not crossing.** The handoff checks mean length
for equanimity vs neutral "collapsed across length". Collapsing across a factor
is exactly the operation that hid the catalog-order confound in Arm G: the
validator confirmed marginal balance while every scenario sat at one order. The
fix is `nesting_report` on the prompt as the unit, plus a *paired within-prompt*
length contrast rather than a marginal one.

**Hole 2 -- a point estimate below 0.2 is not equivalence.** "|Cohen's d| < 0.2"
as written is satisfied by an observed d of 0.19 whose CI runs to 0.35. Absence
of a significant difference is not evidence of absence; establishing orthogonality
is an equivalence claim and needs an equivalence test. This module runs TOST, and
a gate that cannot *demonstrate* equivalence fails rather than passing by default.

**Hole 3 -- the content instrument.** The handoff suggests "the wellbeing/valence
direction from the geometric work". That direction was withdrawn: phi-map's own
re-extraction found it was 79% catalog position, and the geometry line was closed
because of it. Importing a discredited instrument to validate a new experiment
would launder the old failure into this one. A fresh classifier is fit here
instead, cross-validated, and checked against a label-shuffled null.

Thresholds are the handoff's (d>=0.8 separation, |d|<0.2 independence, kept as
pre-registered), but `--self-test` calibrates the gate against synthetic worlds
where the answer is known -- clean, globally confounded, and confounded inside a
single category only -- so the gate's sensitivity is measured rather than assumed.

    python3 gate.py --self-test     # synthetic calibration, no data needed
    python3 gate.py --run           # the real gate, writes gate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from confound_audit import auroc, contamination, nesting_report  # noqa: E402

DATA = Path(__file__).parent / "data" / "factorial_raw.jsonl"
REPORT = Path(__file__).parent / "data" / "gate_report.json"
TOKENIZER_REPO = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # byte-identical to Meta's

# Pre-registered bounds, from the handoff §2.
SEPARATION_MIN_D = 0.8      # terse vs verbose must differ this much
INDEPENDENCE_MAX_D = 0.2    # equanimity vs neutral must differ less than this
CONTENT_MIN_AUROC = 0.75    # stance must be recoverable from the text


# --- statistics ---------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference with pooled SD (independent groups)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def paired_tost(diffs: np.ndarray, sd_scale: float, bound: float = INDEPENDENCE_MAX_D,
                alpha: float = 0.05) -> dict:
    """Two one-sided tests for equivalence on paired differences.

    Equivalence is established only if the (1-2*alpha) CI of the standardised
    mean difference lies entirely inside +/- `bound`. Note the direction of proof:
    the null here is "the factors ARE confounded", and the gate passes only by
    rejecting it. A non-significant difference test does not get you there, which
    is the trap in the handoff's phrasing.

    `sd_scale` is the pooled within-cell SD of the raw measure, so the result is
    on Cohen's d units and comparable to the pre-registered bound.
    """
    from scipy import stats as st
    n = len(diffs)
    mean_d = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    df = n - 1
    tcrit = float(st.t.ppf(1 - alpha, df))

    theta = mean_d / sd_scale if sd_scale > 0 else 0.0
    se_theta = se / sd_scale if sd_scale > 0 else float("inf")
    lo, hi = theta - tcrit * se_theta, theta + tcrit * se_theta
    return dict(
        n=n, mean_raw_diff=mean_d, d=theta,
        ci90_low=float(lo), ci90_high=float(hi), bound=bound,
        equivalent=bool(lo > -bound and hi < bound),
    )


# --- data ---------------------------------------------------------------------

def required_n_for_equivalence(ratio: float, bound: float = INDEPENDENCE_MAX_D,
                               target_pass_rate: float = 0.90,
                               alpha: float = 0.05) -> int:
    """Prompts needed before a CLEAN dataset can actually pass the TOST.

    This is the gate's own power analysis, and it is not optional. A TOST that
    cannot demonstrate equivalence on genuinely orthogonal data is not a strict
    gate, it is a broken one: it fails clean experiments and teaches you to
    override it. Measured on synthetic clean data, a 100-prompt pool passes only
    28% of the time -- so a real PASS at that size would have been luck and a
    real FAIL would have carried no information.

    `ratio` is sd(paired difference) / sd(pooled within-cell), estimated from the
    data itself rather than assumed.

    Under a true zero effect, theta_hat ~ N(0, se) with se = ratio/sqrt(n), and
    the TOST passes when |theta_hat| + t_alpha*se < bound. So

        P(pass) = 2*Phi((bound - t_alpha*se)/se) - 1

    which is a TWO-sided probability -- theta_hat can drift past the bound in
    either direction. Using a one-sided quantile here understates the required n
    by about 30%. The gate must also clear terse AND verbose, so each level is
    held to sqrt(target_pass_rate).
    """
    from scipy import stats as st
    per_test = np.sqrt(target_pass_rate)
    z_pass = st.norm.ppf((1.0 + per_test) / 2.0)
    for n in range(8, 20001):
        se = ratio / np.sqrt(n)
        if (st.t.ppf(1 - alpha, n - 1) + z_pass) * se < bound:
            return n
    return -1


def observed_ratio(rows: list[dict], field: str) -> dict[str, float]:
    """sd(paired diff) / sd(pooled within-cell), per verbosity level."""
    idx = index(rows)
    pids = sorted({r["prompt_id"] for r in rows})
    out = {}
    for verb in ("terse", "verbose"):
        eq, ne = [], []
        for p in pids:
            ke, kn = (p, "equanimity", verb), (p, "neutral", verb)
            if ke in idx and kn in idx:
                eq.append(idx[ke][field]); ne.append(idx[kn][field])
        eq, ne = np.array(eq, float), np.array(ne, float)
        if len(eq) < 3:
            continue
        sd_scale = float(np.sqrt((eq.var(ddof=1) + ne.var(ddof=1)) / 2))
        out[verb] = float((eq - ne).std(ddof=1) / sd_scale) if sd_scale > 0 else float("nan")
    return out


def load_rows(path: Path = DATA) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # Deduplicate on (prompt_id, cell), newest band_version winning. Superseded
    # generations stay on disk rather than being deleted, so the pre-fix numbers
    # remain auditable -- "the verbose band was the problem" is only checkable if
    # the data that showed the problem still exists.
    def _bv(r: dict) -> int:
        return int(str(r.get("band_version", "v1")).lstrip("v") or 1)

    seen: dict[tuple[str, str], dict] = {}
    for r in rows:
        if not r.get("ok"):
            continue
        key = (r["prompt_id"], r["cell"])
        if key not in seen or _bv(r) >= _bv(seen[key]):
            seen[key] = r

    # Keep only prompts complete in all four cells. Partial prompts would make
    # the paired contrasts compare different prompt sets, and generation failure
    # is not independent of the factors -- a prompt one stance handles awkwardly
    # is likelier to fail parsing in that cell. Also makes the gate runnable
    # against a generation run still in flight.
    per: dict[str, set[str]] = defaultdict(set)
    for (pid, cell) in seen:
        per[pid].add(cell)
    complete = {pid for pid, cells in per.items() if len(cells) == 4}

    # Version purity within a prompt. If one cell's newest generation is v1
    # while its siblings are v4, the paired contrast would compare a calibrated
    # generation against an uncalibrated one and book the difference as a stance
    # effect -- manufacturing the exact confound the calibration removed. Such
    # prompts are dropped, not mixed.
    versions: dict[str, set[int]] = defaultdict(set)
    for (pid, _), r in seen.items():
        versions[pid].add(_bv(r))
    pure = {pid for pid in complete if len(versions[pid]) == 1}
    return [r for (pid, _), r in seen.items() if pid in pure]


def tokenize_counts(rows: list[dict]) -> list[dict]:
    """Token counts with the tokenizer that will actually train.

    Word counts or character counts are a proxy, and the manipulation was
    specified in words, so measuring in words would partly measure compliance
    with the instruction rather than the quantity that reaches the model.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    for r in rows:
        r["tok_reasoning"] = len(tok.encode(r["reasoning"], add_special_tokens=False))
        r["tok_answer"] = len(tok.encode(r["answer"], add_special_tokens=False))
        r["tok_total"] = r["tok_reasoning"] + r["tok_answer"]
    return rows


def index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {(r["prompt_id"], r["content"], r["verbosity"]): r for r in rows}


# --- gate checks --------------------------------------------------------------

def check_crossing(rows: list[dict]) -> dict:
    """Is each factor crossed with the prompt, or nested inside it?"""
    out = {}
    for factor in ("content", "verbosity"):
        rep = nesting_report(rows, unit=lambda r: r["prompt_id"],
                             factor=lambda r, f=factor: r[f])
        out[factor] = {k: v for k, v in rep.items()
                       if k in ("nested", "crossed", "verdict", "n_units",
                                "units_at_one_level", "levels")}
        out[factor]["pass"] = not rep.get("nested", True)
    return out


def check_separation(rows: list[dict], field: str) -> dict:
    """Factor B worked: terse and verbose must be far apart."""
    idx = index(rows)
    pids = sorted({r["prompt_id"] for r in rows})
    per_stance = {}
    for stance in ("equanimity", "neutral"):
        pairs = [(idx[(p, stance, "terse")][field], idx[(p, stance, "verbose")][field])
                 for p in pids
                 if (p, stance, "terse") in idx and (p, stance, "verbose") in idx]
        t = np.array([a for a, _ in pairs], float)
        v = np.array([b for _, b in pairs], float)
        per_stance[stance] = dict(
            n=len(pairs), mean_terse=float(t.mean()), mean_verbose=float(v.mean()),
            d=cohens_d(v, t),
            auroc_len_predicts_verbosity=auroc(
                np.concatenate([t, v]).tolist(), [0] * len(t) + [1] * len(v)),
        )
    dmin = min(abs(s["d"]) for s in per_stance.values())
    return dict(field=field, per_stance=per_stance, min_abs_d=float(dmin),
                threshold=SEPARATION_MIN_D, **{"pass": bool(dmin >= SEPARATION_MIN_D)})


def check_independence(rows: list[dict], field: str) -> dict:
    """Factor A is clean: at matched verbosity, stance must not shift length.

    Paired within prompt AND within verbosity, so each contrast holds the prompt
    and the length instruction fixed and varies only the stance. Run separately
    per verbosity level: pooling the two would double-count each prompt and
    understate the standard error.
    """
    idx = index(rows)
    pids = sorted({r["prompt_id"] for r in rows})
    per_verbosity, all_pass = {}, True
    for verb in ("terse", "verbose"):
        eq, ne = [], []
        for p in pids:
            ke, kn = (p, "equanimity", verb), (p, "neutral", verb)
            if ke in idx and kn in idx:
                eq.append(idx[ke][field])
                ne.append(idx[kn][field])
        eq, ne = np.array(eq, float), np.array(ne, float)
        if len(eq) < 3:
            per_verbosity[verb] = dict(n=len(eq), error="insufficient pairs")
            all_pass = False
            continue
        sd_scale = float(np.sqrt((eq.var(ddof=1) + ne.var(ddof=1)) / 2))
        tost = paired_tost(eq - ne, sd_scale)
        # Threshold-free companion: can length tell the two stances apart at all?
        tost["auroc_len_predicts_content"] = auroc(
            np.concatenate([ne, eq]).tolist(), [0] * len(ne) + [1] * len(eq))
        tost["mean_equanimity"] = float(eq.mean())
        tost["mean_neutral"] = float(ne.mean())
        per_verbosity[verb] = tost
        all_pass &= tost["equivalent"]
    return dict(field=field, per_verbosity=per_verbosity, **{"pass": bool(all_pass)})


def check_independence_by_category(rows: list[dict], field: str,
                                   alpha: float = 0.05) -> dict:
    """The §9 drift signal: aggregate can pass while one category is confounded.

    This is a *detection* screen, not a certification, and the distinction is
    forced by arithmetic. Each category holds ~1/5 of the pool, so an equivalence
    test here would demand roughly 5x the pool the global test needs -- at n~20 a
    TOST cannot certify |d| < 0.2 no matter how clean the data is, so scoring
    categories that way reports "FAIL" for every possible dataset and carries no
    information.

    What a small sample *can* do is detect a confound large enough to matter. So
    each category runs a paired difference test, Holm-corrected across the 2 x
    n_categories comparisons, and the screen fails if any category shows a
    reliable stance-driven length shift. The global TOST remains the certifying
    test; this one only has to catch a local blow-up.
    """
    from scipy import stats as st
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    raw = []
    for cat, sub in sorted(by_cat.items()):
        idx_ = index(sub)
        pids = sorted({r["prompt_id"] for r in sub})
        for verb in ("terse", "verbose"):
            eq, ne = [], []
            for p in pids:
                ke, kn = (p, "equanimity", verb), (p, "neutral", verb)
                if ke in idx_ and kn in idx_:
                    eq.append(idx_[ke][field]); ne.append(idx_[kn][field])
            if len(eq) < 4:
                continue
            eq, ne = np.array(eq, float), np.array(ne, float)
            sd = float(np.sqrt((eq.var(ddof=1) + ne.var(ddof=1)) / 2))
            t, p = st.ttest_rel(eq, ne)
            raw.append(dict(category=cat, verbosity=verb, n=len(eq),
                            d=float((eq - ne).mean() / sd) if sd > 0 else 0.0,
                            mean_diff=float((eq - ne).mean()), p=float(p)))

    # Holm step-down across all category x verbosity comparisons.
    order = sorted(range(len(raw)), key=lambda i: raw[i]["p"])
    m, flagged = len(raw), False
    for rank, i in enumerate(order):
        adj = min(1.0, raw[i]["p"] * (m - rank))
        raw[i]["p_holm"] = adj
        raw[i]["flagged"] = bool(adj < alpha)
        flagged |= raw[i]["flagged"]

    out: dict[str, dict] = {}
    for r in raw:
        out.setdefault(r["category"], {})[r["verbosity"]] = {
            k: r[k] for k in ("d", "n", "mean_diff", "p", "p_holm", "flagged")}
    return dict(field=field, by_category=out, n_comparisons=m,
                worst_abs_d=float(max((abs(r["d"]) for r in raw), default=0.0)),
                min_p_holm=float(min((r["p_holm"] for r in raw), default=1.0)),
                **{"pass": bool(not flagged)})


def leakage_ratio(rows: list[dict], field: str) -> dict:
    """Content-induced length shift as a fraction of the deliberate B manipulation.

    Reported ALONGSIDE the pre-registered TOST, never instead of it. The TOST is
    the gate; this is context for reading it.

    The motivation is that Cohen's d is scale-relative, and this design
    deliberately shrinks the scale: the word budget pins terse reasoning near 47
    tokens, so the within-cell SD is small and |d| < 0.2 ends up demanding
    agreement within ~1.6 tokens. That is a fine bar to clear, but d alone does
    not say whether a residual difference could *matter*. The quantity that says
    so is the comparison against Factor B's own manipulation: if stance moves
    length by 3 tokens while verbosity moves it by 243, then even granting that
    length drives every downstream effect, the stance contrast can inherit at
    most ~1% of it.

    Stated in advance of seeing whether it passes, and it does not replace a
    failing threshold -- the pool was expanded to meet the original bound instead.
    """
    idx = index(rows)
    pids = sorted({r["prompt_id"] for r in rows})
    out = {}
    for verb in ("terse", "verbose"):
        eq = np.array([idx[(p, "equanimity", verb)][field] for p in pids
                       if (p, "equanimity", verb) in idx], float)
        ne = np.array([idx[(p, "neutral", verb)][field] for p in pids
                       if (p, "neutral", verb) in idx], float)
        out[verb] = dict(content_shift_tokens=float(eq.mean() - ne.mean()))
    b_shift = []
    for stance in ("equanimity", "neutral"):
        t = np.array([idx[(p, stance, "terse")][field] for p in pids
                      if (p, stance, "terse") in idx], float)
        v = np.array([idx[(p, stance, "verbose")][field] for p in pids
                      if (p, stance, "verbose") in idx], float)
        b_shift.append(v.mean() - t.mean())
    manipulation = float(np.mean(b_shift))
    for verb in out:
        out[verb]["leakage_pct_of_B"] = float(
            100.0 * abs(out[verb]["content_shift_tokens"]) / abs(manipulation))
    return dict(field=field, b_manipulation_tokens=manipulation, per_verbosity=out,
                worst_leakage_pct=float(max(v["leakage_pct_of_B"]
                                            for v in out.values())))


def check_content_separation(rows: list[dict]) -> dict:
    """Factor A actually landed: stance must be recoverable from the text.

    Fresh instrument, not the withdrawn valence direction. Two guards:
      * label-shuffled null must score ~0.5, or the classifier is leaking;
      * scored WITHIN a verbosity stratum, so it cannot be reading length.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    texts = [f"{r['reasoning']} {r['answer']}" for r in rows]
    y = np.array([1 if r["content"] == "equanimity" else 0 for r in rows])
    groups = np.array([r["prompt_id"] for r in rows])  # never split a prompt
    verb = np.array([r["verbosity"] for r in rows])

    def fit_auroc(labels: np.ndarray) -> tuple[float, np.ndarray]:
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=1.0),
        )
        prob = cross_val_predict(pipe, texts, labels, groups=groups,
                                 cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
        return auroc(prob.tolist(), labels.tolist()), prob

    obs, prob = fit_auroc(y)
    rng = np.random.default_rng(0)
    null = float(np.mean([fit_auroc(rng.permutation(y))[0] for _ in range(3)]))

    within = {v: auroc(prob[verb == v].tolist(), y[verb == v].tolist())
              for v in ("terse", "verbose")}
    # Does the stance readout actually read verbosity instead?
    contam = contamination(prob.tolist(), y.tolist(),
                           (verb == "verbose").astype(int).tolist())

    passed = (obs >= CONTENT_MIN_AUROC and 0.35 <= null <= 0.65
              and min(within.values()) >= CONTENT_MIN_AUROC)
    return dict(auroc=float(obs), shuffled_null_auroc=null, within_verbosity=within,
                contamination=contam, threshold=CONTENT_MIN_AUROC,
                **{"pass": bool(passed)})


def run_gate(rows: list[dict]) -> dict:
    checks: dict = {}
    checks["crossing"] = check_crossing(rows)
    checks["separation"] = {f: check_separation(rows, f)
                            for f in ("tok_reasoning", "tok_total")}
    checks["independence"] = {f: check_independence(rows, f)
                              for f in ("tok_reasoning", "tok_answer", "tok_total")}
    checks["independence_by_category"] = check_independence_by_category(
        rows, "tok_total")
    checks["content_separation"] = check_content_separation(rows)
    # Supplementary, not part of the pass/fail decision.
    checks["leakage"] = {f: leakage_ratio(rows, f)
                         for f in ("tok_reasoning", "tok_total")}
    checks["required_pool_n"] = {
        f: {v: required_n_for_equivalence(r)
            for v, r in observed_ratio(rows, f).items()}
        for f in ("tok_reasoning", "tok_total")}

    # Two-tier verdict, per the pre-committed calibration cap (agreed with the
    # user before v5 ran, recorded in matched.py): the hard gate is the
    # PRE-REGISTERED criteria from the handoff's §2 -- crossing, separation,
    # global TOST independence, content separation. The per-category screen is
    # stricter than anything the handoff specified; it drove three calibration
    # rounds, and the agreed rule is that once the single v5 round is spent, a
    # persistent screen flag becomes a DOCUMENTED LIMITATION rather than a
    # training blocker. It stays in the report and the printed output either
    # way -- documented is not the same as hidden.
    preregistered = (
        all(v["pass"] for v in checks["crossing"].values())
        and all(v["pass"] for v in checks["separation"].values())
        and all(v["pass"] for v in checks["independence"].values())
        and checks["content_separation"]["pass"]
    )
    return dict(n_rows=len(rows), checks=checks,
                preregistered_pass=bool(preregistered),
                screen_clean=bool(checks["independence_by_category"]["pass"]),
                **{"pass": bool(preregistered)})


# --- synthetic calibration ----------------------------------------------------

def _synth(n: int = 100, content_len_effect: float = 0.0,
           confound_category: str | None = None, seed: int = 3) -> list[dict]:
    """Synthetic dataset with a known truth, for calibrating the gate."""
    rng = np.random.default_rng(seed)
    cats = ["dysphoric", "hostile", "technical_hard", "judgment", "underspecified"]
    rows = []
    for i in range(n):
        cat = cats[i % len(cats)]
        base = rng.normal(0, 12)
        for content in ("equanimity", "neutral"):
            for verbosity in ("terse", "verbose"):
                mu = 45 if verbosity == "terse" else 240
                shift = 0.0
                if content == "equanimity":
                    if confound_category is None or cat == confound_category:
                        shift = content_len_effect * mu
                val = max(5, mu + base + shift + rng.normal(0, 10))
                rows.append(dict(
                    prompt_id=f"{cat}_{i:03d}", category=cat, content=content,
                    verbosity=verbosity, cell=f"{content}-{verbosity}",
                    tok_reasoning=int(val), tok_answer=int(rng.normal(120, 15)),
                    tok_total=int(val) + int(rng.normal(120, 15)),
                ))
    return rows


def self_test() -> int:
    print("=" * 74)
    print("GATE CALIBRATION AGAINST SYNTHETIC GROUND TRUTH (no data, no GPU)")
    print("=" * 74)
    ok = True

    print("\n[1] Clean world: stance has zero effect on length. Gate must PASS")
    print("    -- but as a RATE over seeds, not one lucky draw. A gate that fails")
    print("    clean data is not strict, it is uninformative.")
    ratio = float(np.mean(list(observed_ratio(_synth(n=400, seed=0),
                                              "tok_reasoning").values())))
    need = required_n_for_equivalence(ratio)
    print(f"      observed sd(diff)/sd(pooled) = {ratio:.3f}"
          f"  -> required pool = {need} prompts")
    rates = {}
    for n in (100, need):
        passes = sum(check_independence(_synth(n=n, content_len_effect=0.0, seed=s),
                                        "tok_reasoning")["pass"] for s in range(30))
        rates[n] = passes / 30
        print(f"      n={n:4d}: clean-world pass rate {rates[n]:.2f}")
    good = rates[need] >= 0.80 and rates[100] < 0.80
    ok &= good
    sep = check_separation(_synth(n=400, seed=0), "tok_reasoning")
    ok &= sep["pass"]
    print(f"      separation min|d| = {sep['min_abs_d']:.2f} (need >={SEPARATION_MIN_D})"
          f"  [{'ok' if sep['pass'] else 'FAIL'}]")
    print(f"      -> required_n is calibrated, 100 is not enough"
          f"  [{'ok' if good else 'FAIL'}]")

    print("\n[2] Confounded world: equanimity 15% shorter everywhere. Gate must FAIL.")
    bad = _synth(content_len_effect=-0.15)
    ind_b = check_independence(bad, "tok_reasoning")
    caught = not ind_b["pass"]
    ok &= caught
    for v, r in ind_b["per_verbosity"].items():
        print(f"      independence {v:8s} d={r['d']:+.3f} "
              f"equivalent={r['equivalent']}")
    print(f"      -> confound detected [{'ok' if caught else 'FAIL -- GATE IS BLIND'}]")

    print("\n[3] Category-local confound: only 'dysphoric' is confounded.")
    print("    This is the §9 drift signal -- global gate may pass, per-category"
          " must catch it.")
    local = _synth(content_len_effect=-0.45, confound_category="dysphoric")
    g_ind = check_independence(local, "tok_reasoning")
    c_ind = check_independence_by_category(local, "tok_reasoning")
    caught_local = not c_ind["pass"]
    ok &= caught_local
    print(f"      global independence pass = {g_ind['pass']}")
    print(f"      per-category worst |d|   = {c_ind['worst_abs_d']:.3f}"
          f"  -> pass = {c_ind['pass']}")
    for cname, verbs in c_ind["by_category"].items():
        ds = ", ".join(f"{v}={r['d']:+.2f}" for v, r in verbs.items())
        print(f"        {cname:16s} {ds}")
    print(f"      -> per-category caught it [{'ok' if caught_local else 'FAIL'}]")

    print("\n[3b] Screen specificity: clean data must NOT be flagged.")
    print("     A screen that fires on clean data would make the gate")
    print("     unpassable and train you to override it.")
    flags = sum(not check_independence_by_category(
        _synth(n=400, content_len_effect=0.0, seed=s), "tok_reasoning")["pass"]
        for s in range(30))
    rate = flags / 30
    good = rate <= 0.15
    ok &= good
    print(f"      false-flag rate over 30 clean datasets = {rate:.2f}"
          f"  [{'ok' if good else 'FAIL'}]")

    print("\n[4] TOST is not a difference test: a wide, centred estimate must NOT")
    print("    pass as equivalent (absence of evidence is not evidence of absence).")
    rng = np.random.default_rng(7)
    tiny = rng.normal(0, 40, size=6)          # n=6, centred but very noisy
    res = paired_tost(tiny, sd_scale=40.0)
    good = not res["equivalent"]
    ok &= good
    print(f"      n=6 d={res['d']:+.3f} CI90 [{res['ci90_low']:+.2f},"
          f"{res['ci90_high']:+.2f}] equivalent={res['equivalent']}"
          f"  [{'ok' if good else 'FAIL'}]")

    print("\n" + "=" * 74)
    print("GATE SELF-TEST", "PASSED" if ok else "FAILED")
    print("=" * 74)
    return 0 if ok else 1


def _fmt(report: dict) -> None:
    c = report["checks"]
    print("=" * 74)
    print(f"ORTHOGONALITY GATE -- {report['n_rows']} rows")
    print("=" * 74)

    print("\n[1] CROSSING (is each factor crossed with the prompt?)")
    for f, r in c["crossing"].items():
        print(f"    {f:10s} verdict={r.get('verdict')} "
              f"units_at_one_level={r.get('units_at_one_level')}"
              f"  [{'PASS' if r['pass'] else 'FAIL'}]")

    print("\n[2] LENGTH SEPARATION -- Factor B worked")
    for f, r in c["separation"].items():
        print(f"    {f}:")
        for st, s in r["per_stance"].items():
            print(f"      {st:11s} terse {s['mean_terse']:6.1f} -> verbose "
                  f"{s['mean_verbose']:6.1f}  d={s['d']:+.2f}  "
                  f"AUROC={s['auroc_len_predicts_verbosity']:.3f}")
        print(f"      min|d| = {r['min_abs_d']:.2f} (need >= {r['threshold']})"
              f"  [{'PASS' if r['pass'] else 'FAIL'}]")

    print("\n[3] LENGTH INDEPENDENCE -- Factor A is clean (paired TOST)")
    for f, r in c["independence"].items():
        print(f"    {f}:")
        for v, s in r["per_verbosity"].items():
            print(f"      {v:8s} n={s['n']:3d} eq {s['mean_equanimity']:6.1f} vs "
                  f"neu {s['mean_neutral']:6.1f}  d={s['d']:+.3f}  "
                  f"CI90 [{s['ci90_low']:+.3f},{s['ci90_high']:+.3f}]  "
                  f"AUROC={s['auroc_len_predicts_content']:.3f}  "
                  f"{'EQUIV' if s['equivalent'] else 'NOT EQUIV'}")
        print(f"      [{'PASS' if r['pass'] else 'FAIL'}]")

    print("\n[4] PER-CATEGORY CONFOUND SCREEN (tok_total, Holm-corrected)")
    bc = c["independence_by_category"]
    for cat, verbs in bc["by_category"].items():
        ds = "  ".join(
            f"{v}: {s['mean_diff']:+6.1f}tok d={s['d']:+.2f} p_holm={s['p_holm']:.2f}"
            f"{' FLAG' if s['flagged'] else ''}" for v, s in verbs.items())
        print(f"    {cat:16s} {ds}")
    print(f"    {bc['n_comparisons']} comparisons, min p_holm = {bc['min_p_holm']:.3f}"
          f"  [{'PASS' if bc['pass'] else 'FAIL'}]")

    print("\n[S] SUPPLEMENTARY (not part of the verdict)")
    for f, r in c["leakage"].items():
        per = "  ".join(f"{v}: {s['content_shift_tokens']:+.1f}tok "
                        f"({s['leakage_pct_of_B']:.1f}% of B)"
                        for v, s in r["per_verbosity"].items())
        print(f"    leakage {f:14s} B moves {r['b_manipulation_tokens']:+.0f}tok | {per}")
    for f, r in c["required_pool_n"].items():
        print(f"    pool n needed to certify |d|<0.2 on {f:14s} "
              + "  ".join(f"{v}={n}" for v, n in r.items()))

    print("\n[5] CONTENT SEPARATION -- Factor A actually landed")
    cs = c["content_separation"]
    print(f"    stance AUROC (grouped CV)   = {cs['auroc']:.3f} "
          f"(need >= {cs['threshold']})")
    print(f"    label-shuffled null         = {cs['shuffled_null_auroc']:.3f} "
          f"(need ~0.50)")
    for v, a in cs["within_verbosity"].items():
        print(f"    within {v:8s}             = {a:.3f}")
    print(f"    contamination: {cs['contamination']}")
    print(f"    [{'PASS' if cs['pass'] else 'FAIL'}]")

    print("\n" + "=" * 74)
    if not report["pass"]:
        print("GATE VERDICT: FAIL -- pre-registered criteria not met. DO NOT TRAIN.")
    elif report.get("screen_clean", True):
        print("GATE VERDICT: PASS -- all criteria including the per-category "
              "screen. Training may proceed.")
    else:
        print("GATE VERDICT: PASS on the pre-registered criteria -- training "
              "may proceed.")
        print("  DOCUMENTED LIMITATION: the per-category screen retains flags "
              "(see [4]).")
        print("  Per the pre-committed calibration cap, these are carried as "
              "stated limitations")
        print("  of the dataset, reported in any writeup, with category as an "
              "analysis covariate.")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if args.run:
        rows = tokenize_counts(load_rows())
        rep = run_gate(rows)
        REPORT.write_text(json.dumps(rep, indent=2))
        _fmt(rep)
        print(f"\nwrote {REPORT}")
        sys.exit(0 if rep["pass"] else 2)
    ap.print_help()
