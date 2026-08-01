"""Build arm_g_olmo3_phase3.ipynb. Prereg: docs/olmo3-phase3-prereg-2026-07-31.md.

The notebook embeds BOTH pinned artifacts with sha256 asserts:
  - arm_g_scenarios.py          (prompt generator)
  - arm_g_olmo3_phase3_eval.py  (the branch adjudicator -- the verdict is
                                 whatever the committed evaluate() returns)
"""
from __future__ import annotations
import hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = open(os.path.join(HERE, "arm_g_scenarios.py")).read()
EVAL = open(os.path.join(HERE, "arm_g_olmo3_phase3_eval.py")).read()
SCEN_SHA = hashlib.sha256(SCEN.encode()).hexdigest()
EVAL_SHA = hashlib.sha256(EVAL.encode()).hexdigest()

OLMO_REPO = "allenai/Olmo-3-7B-Instruct"
OLMO_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
LAYER, SEED_A, SEED_B, FIT_FAM = 24, 111, 211, "release_records"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(keepends=True)}
cells = []

cells.append(md(f"""# Arm G × Olmo 3 7B — Phase 3: probe training, held-out evaluation

Prereg `docs/olmo3-phase3-prereg-2026-07-31.md`; adjudicator embedded byte-identical
to the committed `arm_g_olmo3_phase3_eval.py` (sha `{EVAL_SHA[:16]}…`). Layer **{LAYER}**
(fixed at the Phase 2 gate). Seeds **{SEED_A}** (capture reused) and **{SEED_B}** (fresh).
Fit family `{FIT_FAM}`, mirroring the Llama confirmatory design.

Budget estimate **0.5–1.0 units on L4**; `CONFIRMED_BUDGET = 2.0` (anomaly threshold)."""))

cells.append(code(f"""# [1] Budget + hardware preflight
CONFIRMED_BUDGET = 2.0
import torch, time
T0 = time.time()
assert torch.cuda.is_available(), "no GPU runtime"
props = torch.cuda.get_device_properties(0)
print(props.name, round(props.total_memory/1024**3, 1), "GiB")
assert props.total_memory/1024**3 >= 14.5
BF16 = torch.cuda.is_bf16_supported(including_emulation=False)
DTYPE = torch.bfloat16 if BF16 else torch.float16
RATES = {{"L4": 4.82, "A100": 11.77, "T4": 1.84}}
rate = next((v for k, v in RATES.items() if k in props.name), 5.0)
est = rate * 0.25
print("dtype:", DTYPE, "| projected ~%.1f units" % est)
assert est <= CONFIRMED_BUDGET, "cost anomaly -- diagnose, do not shrink"
import transformers
assert transformers.__version__.startswith("5."), transformers.__version__"""))

cells.append(code("# [2] Pinned artifacts, embedded verbatim with hash asserts\n"
                  "import pathlib, hashlib\n"
                  f"SCEN_SHA, EVAL_SHA = \"{SCEN_SHA}\", \"{EVAL_SHA}\"\n"
                  "scen_src = " + json.dumps(SCEN) + "\n"
                  "eval_src = " + json.dumps(EVAL) + "\n"
                  "assert hashlib.sha256(scen_src.encode()).hexdigest() == SCEN_SHA\n"
                  "assert hashlib.sha256(eval_src.encode()).hexdigest() == EVAL_SHA\n"
                  "pathlib.Path('arm_g_scenarios.py').write_text(scen_src)\n"
                  "pathlib.Path('arm_g_olmo3_phase3_eval.py').write_text(eval_src)\n"
                  "import subprocess\n"
                  "r = subprocess.run(['python3','arm_g_olmo3_phase3_eval.py','--self-test'],\n"
                  "                   capture_output=True, text=True)\n"
                  "print(r.stdout); assert r.returncode == 0, r.stderr"))

cells.append(code("""# [3] Sinks
import os, io, json
WORK = "/content/olmo3_phase3"; os.makedirs(WORK, exist_ok=True)
DRIVE = P2 = None
try:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/phi-map/olmo3-replication"
    DRIVE = os.path.join(ROOT, "phase3"); os.makedirs(DRIVE, exist_ok=True)
    P2 = os.path.join(ROOT, "phase2")
    print("Drive:", DRIVE)
except Exception as e:
    print("Drive unavailable:", type(e).__name__)

def persist(name, blob: bytes):
    open(os.path.join(WORK, name), "wb").write(blob)
    if DRIVE:
        tmp = os.path.join(DRIVE, name + ".tmp")
        open(tmp, "wb").write(blob); os.replace(tmp, os.path.join(DRIVE, name))"""))

cells.append(code(f"""# [4] Manifests + G0 on the box, both seeds
import arm_g_scenarios as S
from transformers import AutoTokenizer
OLMO_REPO, OLMO_REV = "{OLMO_REPO}", "{OLMO_REV}"
LAYER, FIT_FAM = {LAYER}, "{FIT_FAM}"
tok = AutoTokenizer.from_pretrained(OLMO_REPO, revision=OLMO_REV)

def unwrap(x):
    if hasattr(x, "input_ids"): x = x.input_ids
    elif isinstance(x, dict): x = x["input_ids"]
    x = list(x)
    if len(x) == 1 and hasattr(x[0], "__len__"): x = list(x[0])
    return x

MAN = {{}}
for seed in ({SEED_A}, {SEED_B}):
    rows = S.build_manifest(seed=seed, catalog_order_mode="crossed")
    S.validate_manifest(rows)
    tails = set()
    for r in rows:
        ids = unwrap(tok.apply_chat_template(r["messages"], add_generation_prompt=True))
        tails.add(tuple(ids[-4:]))
    assert len(tails) == 1 and tok.decode(list(next(iter(tails)))).endswith("<|im_start|>assistant\\n")
    MAN[seed] = rows
    assert FIT_FAM in {{r["family"] for r in rows}}
    print(f"seed {{seed}}: {{len(rows)}} rows, G0 pass")"""))

cells.append(code(f"""# [5] Capture. Seed {SEED_A} reused from Phase 2 if present; seed {SEED_B} fresh.
import numpy as np, shutil
from transformers import AutoModelForCausalLM

H = {{}}
src111 = P2 and os.path.join(P2, "capture.npz")
if src111 and os.path.exists(src111):
    H[{SEED_A}] = np.load(src111)["H"]
    print("seed {SEED_A} capture reused:", H[{SEED_A}].shape)

model = None
def capture(rows):
    global model
    if model is None:
        m = AutoModelForCausalLM.from_pretrained(OLMO_REPO, revision=OLMO_REV,
                                                 torch_dtype=DTYPE, device_map="auto")
        m.eval(); model = m
        tok.padding_side = "right"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
    outs = []
    for i in range(0, len(rows), 8):
        batch = [tok.apply_chat_template(r["messages"], tokenize=False,
                 add_generation_prompt=True) for r in rows[i:i+8]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        idx = enc["attention_mask"].sum(dim=1) - 1
        sel = torch.arange(idx.shape[0], device=idx.device)
        outs.append(torch.stack([h[sel, idx] for h in hs], 1).float().cpu().numpy())
    return np.concatenate(outs)

for seed in ({SEED_A}, {SEED_B}):
    if seed in H: continue
    ck = f"capture_{{seed}}.npz"
    if DRIVE and os.path.exists(os.path.join(DRIVE, ck)):
        H[seed] = np.load(os.path.join(DRIVE, ck))["H"]
        print(f"seed {{seed}} resumed"); continue
    H[seed] = capture(MAN[seed])
    buf = io.BytesIO(); np.savez_compressed(buf, H=H[seed])
    persist(ck, buf.getvalue())
    print(f"seed {{seed}} captured:", H[seed].shape, f"{{time.time()-T0:.0f}}s")
if model is not None:
    del model; torch.cuda.empty_cache()"""))

cells.append(code(f"""# [6] Fits, controls, orthogonalization, verdict. CPU only.
import numpy as np
from arm_g_olmo3_phase3_eval import evaluate, sensitivity, BARS
rng = np.random.default_rng(20260731)

def auroc(s, y):
    r = s.argsort().argsort().astype(float)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1*(n1-1)/2) / (n1*n0))

def unit(v): return v / (np.linalg.norm(v) + 1e-12)

def orthogonalize(d, axes):
    for a in axes:
        a = unit(a); d = d - (d @ a) * a
    return unit(d)

results, dirs = {{}}, {{}}
for seed in ({SEED_A}, {SEED_B}):
    rows = MAN[seed]
    X = H[seed][:, LAYER, :]
    cond = np.array([r["condition"] == "conflict" for r in rows])
    order = np.array([r["catalog_order"] == "inside_first" for r in rows])
    fam = np.array([r["family"] for r in rows])
    tr, te = fam == FIT_FAM, fam != FIT_FAM

    d_raw = unit(X[tr][cond[tr]].mean(0) - X[tr][~cond[tr]].mean(0))
    ax_order = X[tr][order[tr]].mean(0) - X[tr][~order[tr]].mean(0)
    ax_int = ((X[tr][order[tr] & cond[tr]].mean(0) - X[tr][~order[tr] & cond[tr]].mean(0))
              - (X[tr][order[tr] & ~cond[tr]].mean(0) - X[tr][~order[tr] & ~cond[tr]].mean(0)))
    d_orth = orthogonalize(d_raw.copy(), [ax_order, ax_int])
    dirs[seed] = dict(raw=d_raw, orth=d_orth)

    def heldout(d, y): return auroc(X[te] @ d, y[te])
    boot = []
    s_orth = X[te] @ d_orth
    yte = cond[te]
    for _ in range(1000):
        ix = rng.integers(0, len(s_orth), len(s_orth))
        if yte[ix].all() or (~yte[ix]).all(): continue
        boot.append(auroc(s_orth[ix], yte[ix]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    shuf = []
    for _ in range(200):
        yp = cond.copy(); yp[tr] = rng.permutation(cond[tr])
        dp = unit(X[tr][yp[tr]].mean(0) - X[tr][~yp[tr]].mean(0))
        shuf.append(heldout(dp, cond))
    rand = []
    for _ in range(200):
        v = unit(rng.standard_normal(X.shape[1]))
        a = heldout(v, cond); rand.append(max(a, 1 - a))

    results[f"s{{seed}}"] = dict(
        cond_raw=heldout(d_raw, cond), cond_orth=heldout(d_orth, cond),
        order_raw=heldout(d_raw, order), order_orth=heldout(d_orth, order),
        shuf_p975=float(np.percentile(shuf, 97.5)),
        rand_p975=float(np.percentile(rand, 97.5)),
        cond_orth_ci=ci, n_train=int(tr.sum()), n_eval=int(te.sum()))
    print(f"seed {{seed}}:", {{k: (round(v,4) if isinstance(v,float) else v)
                              for k, v in results[f's{{seed}}'].items()}})

results["cosine_raw"] = float(dirs[{SEED_A}]["raw"] @ dirs[{SEED_B}]["raw"])
results["cosine_orth"] = float(dirs[{SEED_A}]["orth"] @ dirs[{SEED_B}]["orth"])
print("cosine raw %.4f | orth %.4f" % (results["cosine_raw"], results["cosine_orth"]))

# free secondary: sweep-shape replication across seeds (no branch weight)
from scipy.stats import spearmanr
def curve(seed):
    rows = MAN[seed]; cond = np.array([r["condition"]=="conflict" for r in rows])
    fam = np.array([r["family"] for r in rows]); out = []
    for L in range(H[seed].shape[1]):
        XL = H[seed][:, L, :]; vals = []
        for hf in sorted(set(fam)):
            trm, tem = fam != hf, fam == hf
            dd = unit(XL[trm][cond[trm]].mean(0) - XL[trm][~cond[trm]].mean(0))
            vals.append(auroc(XL[tem] @ dd, cond[tem]))
        out.append(float(np.mean(vals)))
    return out
c111, c211 = curve({SEED_A}), curve({SEED_B})
results["sweep_spearman"] = float(spearmanr(c111, c211).statistic)
print("sweep-shape Spearman:", round(results["sweep_spearman"], 4))

m = dict(results); m["s{SEED_A}"] = results["s{SEED_A}"]; m["s{SEED_B}"] = results["s{SEED_B}"]
payload = {{"s111": results["s{SEED_A}"], "s211": results["s{SEED_B}"],
            "cosine_orth": results["cosine_orth"]}}
verdict = evaluate(payload)
sens = sensitivity(payload)
print("\\nBRANCH:", verdict["branch"])
for r in verdict["reasons"]: print("  ", r)
print("sensitivity fragile:", sens["fragile"])"""))

cells.append(code(f"""# [7] Persist + printed summary
import hashlib, json
summary = dict(protocol="OLMO3_PHASE3_V1", model=OLMO_REPO, revision=OLMO_REV,
               layer=LAYER, fit_family=FIT_FAM, seeds=[{SEED_A}, {SEED_B}],
               dtype=str(DTYPE), results=results,
               verdict=verdict, sensitivity=sens,
               reference_6d_encoder=0.509,
               reference_note="6-d encoder is a fixed Llama-side reference; not rerunnable here (output clamped to READY)",
               scen_sha=SCEN_SHA, eval_sha=EVAL_SHA,
               curves=dict(c111=[round(v,4) for v in c111], c211=[round(v,4) for v in c211]),
               wall_s=round(time.time()-T0, 1))
blob = json.dumps(summary, indent=1).encode()
persist("phase3_summary.json", blob)
print("sha256:", hashlib.sha256(blob).hexdigest()[:16])
print("=== PHASE3 SUMMARY BEGIN ===")
print(json.dumps({{k: v for k, v in summary.items() if k != "curves"}}, indent=1))
print("=== PHASE3 SUMMARY END ===")"""))

nb = {"cells": cells, "metadata": {"language_info": {"name": "python"},
      "colab": {"provenance": []}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "arm_g_olmo3_phase3.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out} ({os.path.getsize(out):,} bytes)")
print("scen", SCEN_SHA[:16], "| eval", EVAL_SHA[:16])
