"""Build arm_g_olmo3_phase2.ipynb -- the Phase 2 layer sweep, Run-All, no manual steps.

Design decisions, fixed here before any data:

ONSET RULE (pre-committed): decodability onset = the smallest layer L such that
held-out condition AUROC exceeds the shuffled-label 97.5th percentile at BOTH
L and L+1. Two consecutive layers so a single-layer fluctuation cannot declare
onset.

CAPTURE INSTRUMENT: plain transformers with output_hidden_states=True and
attention-mask indexing -- the same instrument class as the Llama side
(comparable by inspection), executed locally on the Colab GPU. The handoff
names nnsight local mode; the repo code is authoritative for implementation
details and the Llama side did not use nnsight. NDIF is not used either way.
Flagged in the gate report; swapping to nnsight is a small change if required.

SWEEP: all 33 hidden-state indices (embeddings + 32 layers). Per layer:
difference-of-means direction fit on two families, AUROC on the held-out
family, averaged over the three holdouts. Reported alongside, same
construction: ORDER decodability (amendment 2's contamination axis, tracked
from day one), a 200-permutation shuffled-label band, and a 200-draw
random-direction band.

SELF-CONTAINED: arm_g_scenarios.py is embedded verbatim with a sha256 assert
against the committed file, so the notebook needs no repo access and the
prompt set is pinned by hash, not by trust.
"""
from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = open(os.path.join(HERE, "arm_g_scenarios.py")).read()
SCEN_SHA = hashlib.sha256(SCEN.encode()).hexdigest()

OLMO_REPO = "allenai/Olmo-3-7B-Instruct"
OLMO_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
SEED = 111

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(keepends=True)}

cells = []

cells.append(md(f"""# Arm G × Olmo 3 7B — Phase 2: layer sweep

**Branch `olmo3-replication`. Run All. No manual steps except the Drive consent tap.**

Pinned: `{OLMO_REPO}` @ `{OLMO_REV[:12]}`, manifest seed {SEED},
`catalog_order_mode="crossed"` (amendment 1).

**Pre-committed onset rule:** onset = smallest layer L with held-out condition AUROC
above the shuffled-label p97.5 at **both L and L+1**.

**Reported alongside, not optional:** order-decodability curve (amendment 2's axis),
shuffled-label band (200 perms), random-direction band (200 draws).

**Budget estimate: ~1.0–1.5 units on L4** (download ≈4 min, load ≈2, capture ≈2,
probes are CPU). `CONFIRMED_BUDGET` below is an **anomaly detector**: if projected
cost exceeds it, something is broken (wrong runtime, High-RAM shape, quota loop) —
stop and diagnose, never shrink the design."""))

cells.append(code(f"""# [1] Budget + hardware preflight. Fails loudly, spends nothing.
CONFIRMED_BUDGET = 3.0   # units; estimate is 1.0-1.5. Set by the human.

import torch, subprocess, time
T0 = time.time()
assert torch.cuda.is_available(), "no GPU runtime"
props = torch.cuda.get_device_properties(0)
gib = props.total_memory / 1024**3
print(props.name, round(gib, 1), "GiB")
assert gib >= 14.5, f"{{gib:.1f}} GiB: too small for 7B bf16 -- see handoff failure modes"
# fp16 on pre-Ampere (T4 emulates bf16 ~10x slower); bf16 otherwise.
BF16 = torch.cuda.is_bf16_supported(including_emulation=False)
DTYPE = torch.bfloat16 if BF16 else torch.float16
print("dtype:", DTYPE)
RATES = {{"L4": 4.82, "A100": 11.77, "T4": 1.84}}
rate = next((v for k, v in RATES.items() if k in props.name), 5.0)
est = rate * 0.33
print(f"projected ~{{est:.1f}} units at {{rate}}/hr for ~20 min")
assert est <= CONFIRMED_BUDGET, "projected cost exceeds CONFIRMED_BUDGET -- diagnose, do not shrink"

import transformers
if not transformers.__version__.startswith("5."):
    raise RuntimeError(f"transformers {{transformers.__version__}}: need v5 for chat_template.jinja; pip install -q 'transformers>=5,<6' and restart")
print("transformers", transformers.__version__)"""))

cells.append(code("# [2] Pinned scenario generator, embedded verbatim.\n"
                  "import pathlib, hashlib\n"
                  f"SCEN_SHA = \"{SCEN_SHA}\"\n"
                  "src = " + json.dumps(SCEN) + "\n"
                  "assert hashlib.sha256(src.encode()).hexdigest() == SCEN_SHA, 'embedded generator does not match committed hash'\n"
                  "pathlib.Path('arm_g_scenarios.py').write_text(src)\n"
                  "print('arm_g_scenarios.py pinned @', SCEN_SHA[:16])"))

cells.append(code("""# [3] Artifact sink: Drive primary (one consent tap), local always.
import os, json
WORK = "/content/olmo3_phase2"
os.makedirs(WORK, exist_ok=True)
DRIVE = None
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = "/content/drive/MyDrive/phi-map/olmo3-replication/phase2"
    os.makedirs(DRIVE, exist_ok=True)
    print("Drive sink:", DRIVE)
except Exception as e:
    print("Drive unavailable (%s); local-only, summary is printed at the end" % type(e).__name__)

def persist(name, blob: bytes):
    open(os.path.join(WORK, name), "wb").write(blob)
    if DRIVE:
        tmp = os.path.join(DRIVE, name + ".tmp")
        open(tmp, "wb").write(blob)
        os.replace(tmp, os.path.join(DRIVE, name))   # atomic per unit of work"""))

cells.append(code(f"""# [4] Gate G0: rebuild the crossed manifest and re-run the Phase 1 checks ON THIS BOX.
import arm_g_scenarios as S
from transformers import AutoTokenizer

OLMO_REPO, OLMO_REV, SEED = "{OLMO_REPO}", "{OLMO_REV}", {SEED}
tok = AutoTokenizer.from_pretrained(OLMO_REPO, revision=OLMO_REV)
rows = S.build_manifest(seed=SEED, catalog_order_mode="crossed")
S.validate_manifest(rows)

def unwrap(x):
    if hasattr(x, "input_ids"): x = x.input_ids
    elif isinstance(x, dict): x = x["input_ids"]
    x = list(x)
    if len(x) == 1 and hasattr(x[0], "__len__"): x = list(x[0])
    return x

tails = set()
enc_ids = []
for r in rows:
    ids = unwrap(tok.apply_chat_template(r["messages"], add_generation_prompt=True))
    enc_ids.append(ids)
    tails.add(tuple(ids[-4:]))
assert len(tails) == 1, f"read position not uniform: {{len(tails)}} tails"
assert tok.decode(list(next(iter(tails)))).endswith("<|im_start|>assistant\\n")
print(f"G0 pass: {{len(rows)}} rows, one tail, read position = final templated token")"""))

cells.append(code("""# [5] Capture: hidden states at the read position, all 33 indices. Resumable.
import numpy as np, torch, os
from transformers import AutoModelForCausalLM

CKPT = os.path.join(WORK, "capture.npz")
if DRIVE and os.path.exists(os.path.join(DRIVE, "capture.npz")):
    import shutil; shutil.copy(os.path.join(DRIVE, "capture.npz"), CKPT)
if os.path.exists(CKPT):
    H = np.load(CKPT)["H"]
    print("capture resumed from checkpoint:", H.shape)
else:
    model = AutoModelForCausalLM.from_pretrained(
        OLMO_REPO, revision=OLMO_REV, torch_dtype=DTYPE, device_map="auto")
    model.eval()
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    outs = []
    B = 8
    for i in range(0, len(rows), B):
        batch = [tok.apply_chat_template(r["messages"], tokenize=False,
                 add_generation_prompt=True) for r in rows[i:i+B]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        # last NON-PAD position, by attention mask -- never [:, -1, :] on a padded batch
        idx = enc["attention_mask"].sum(dim=1) - 1
        rowsel = torch.arange(idx.shape[0], device=idx.device)
        outs.append(torch.stack([h[rowsel, idx] for h in hs], 1).float().cpu().numpy())
        if i % 64 == 0: print(f"  {i}/{len(rows)}  {time.time()-T0:.0f}s")
    H = np.concatenate(outs)          # (384, 33, 4096)
    buf = __import__('io').BytesIO(); np.savez_compressed(buf, H=H)
    persist("capture.npz", buf.getvalue())
    del model; torch.cuda.empty_cache()
    print("captured:", H.shape)"""))

cells.append(code("""# [6] Sweep. Difference-of-means, family-holdout AUROC, controls. CPU.
import numpy as np
rng = np.random.default_rng(20260731)
cond  = np.array([r["condition"] == "conflict" for r in rows])
order = np.array([r["catalog_order"] == "inside_first" for r in rows])
fam   = np.array([r["family"] for r in rows])
FAMS  = sorted(set(fam))
NL    = H.shape[1]

def auroc(scores, labels):
    r = scores.argsort().argsort().astype(float)
    n1, n0 = labels.sum(), (~labels).sum()
    return (r[labels].sum() - n1*(n1-1)/2) / (n1*n0)

def holdout_auroc(X, y):
    vals = []
    for hf in FAMS:
        tr, te = fam != hf, fam == hf
        d = X[tr][y[tr]].mean(0) - X[tr][~y[tr]].mean(0)
        d /= np.linalg.norm(d) + 1e-12
        vals.append(auroc(X[te] @ d, y[te]))
    return float(np.mean(vals)), [float(v) for v in vals]

cond_curve, cond_per, order_curve = [], [], []
shuf_hi, rand_hi = [], []
for L in range(NL):
    X = H[:, L, :]
    m, per = holdout_auroc(X, cond); cond_curve.append(m); cond_per.append(per)
    order_curve.append(holdout_auroc(X, order)[0])
    sh = [holdout_auroc(X, rng.permutation(cond))[0] for _ in range(200)]
    shuf_hi.append(float(np.percentile(sh, 97.5)))
    rd = []
    for _ in range(200):
        v = rng.standard_normal(X.shape[1]); v /= np.linalg.norm(v)
        s = X @ v
        rd.append(max(auroc(s, cond), 1 - auroc(s, cond)))
    rand_hi.append(float(np.percentile(rd, 97.5)))
    print(f"layer {L:2d}  cond {m:.3f}  order {order_curve[-1]:.3f}  shuf97.5 {shuf_hi[-1]:.3f}")

# pre-committed onset rule: above shuffled p97.5 at BOTH L and L+1
onset = next((L for L in range(NL-1)
              if cond_curve[L] > shuf_hi[L] and cond_curve[L+1] > shuf_hi[L+1]), None)
print("\\nDECODABILITY ONSET (pre-committed rule):", onset)"""))

cells.append(code("""# [7] Plot + persist + printed summary block (transcribable).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, hashlib, io, json

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cond_curve, lw=2, label="condition (scope) AUROC, held-out family")
ax.plot(order_curve, lw=1.5, ls="--", label="catalog-order AUROC (contamination axis)")
ax.plot(shuf_hi, color="gray", lw=1, label="shuffled-label p97.5")
ax.plot(rand_hi, color="lightgray", lw=1, label="random-direction p97.5")
if onset is not None: ax.axvline(onset, color="red", ls=":", label=f"onset = {onset}")
ax.axhline(0.5, color="k", lw=0.5)
ax.set_xlabel("hidden-state index (0 = embeddings)"); ax.set_ylabel("AUROC")
ax.set_title("Olmo-3-7B-Instruct: scope-conflict decodability by layer (crossed manifest)")
ax.legend(fontsize=8); fig.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140)
persist("phase2_sweep.png", buf.getvalue())

summary = dict(
    protocol="OLMO3_PHASE2_SWEEP_V1",
    model=OLMO_REPO, revision=OLMO_REV, dtype=str(DTYPE),
    seed=SEED, n_rows=len(rows), n_layers=NL,
    onset=onset, onset_rule="AUROC > shuffled p97.5 at both L and L+1",
    cond_auroc=[round(v, 4) for v in cond_curve],
    cond_auroc_per_family=cond_per,
    order_auroc=[round(v, 4) for v in order_curve],
    shuffled_p975=[round(v, 4) for v in shuf_hi],
    random_p975=[round(v, 4) for v in rand_hi],
    scenarios_sha=SCEN_SHA,
    wall_s=round(time.time() - T0, 1),
)
blob = json.dumps(summary, indent=1).encode()
persist("phase2_summary.json", blob)
print("sha256(summary):", hashlib.sha256(blob).hexdigest()[:16])
print("=== PHASE2 SUMMARY BEGIN ===")
print(json.dumps({k: v for k, v in summary.items() if k not in ("cond_auroc_per_family",)}, indent=1))
print("=== PHASE2 SUMMARY END ===")"""))

nb = {"cells": cells, "metadata": {"language_info": {"name": "python"},
      "colab": {"provenance": []}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "arm_g_olmo3_phase2.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out} ({os.path.getsize(out):,} bytes, {len(cells)} cells)")
print(f"scenario hash pinned: {SCEN_SHA}")
