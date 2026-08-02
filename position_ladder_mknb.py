"""Generate position_ladder_colab.ipynb. One MODEL per run -- the sequential
two-model load is the untested path that most likely killed wide-catalog run 2."""
import hashlib, json, pathlib

REPO = pathlib.Path("/Users/joseph/phi-map")
SRC = (REPO / "position_ladder.py").read_text()
SHA = hashlib.sha256(SRC.encode()).hexdigest()

cells = []
def md(s): cells.append(dict(cell_type="markdown", metadata={}, source=s.splitlines(True)))
def code(s): cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                               outputs=[], source=s.splitlines(True)))

md(f"""# Position ladder — does primacy track parameters or training compute?

**Run this notebook once per model.** Set `MODEL_KEY` in cell [1] to one of
`llama2-7b`, `llama2-13b`, `llama31-8b`, `olmo3-7b`.

Loading two models in one runtime is the untested path that most likely killed
wide-catalog attempt 2 (76 min, no artifacts). One model per runtime, and results
persist after **every position block**, not at the end.

`llama2-7b` / `llama2-13b` are the **reproduction anchor**: they must return
`RECENCY_ONLY` / `U_SHAPE` or the ladder is uninterpretable and `evaluate()`
refuses to name a driver. Run the anchor rows first.

Adjudicator `position_ladder.py` embedded byte-identical (`{SHA[:16]}…`).
Scoring is mechanical UUID exact-match. No judge.

A100 assumed for all four rows so dtype is constant. Est. **~0.3 unit-hours per
7-8B row, ~0.5 for 13B**; `CONFIRMED_BUDGET` is per-row and also enforced
*during* the loop, not only at startup.

**Runtime:** system RAM is not the constraint — weights stream shard-by-shard to
GPU, so peak host RAM stays near one shard. **VRAM is the constraint.** On a
40 GB A100, Llama-2-13B (26 GB of bf16 weights, MHA so no GQA on the KV cache)
leaves little room: `BATCH = 4`, and cell [7] runs a warm-up probe at the worst-
case prompt length that fails in seconds rather than OOMing mid-run. Cell [5]
separately asserts nothing was offloaded to CPU or disk, since `device_map="auto"`
degrades silently instead of failing.
""")

code('''# [1] preflight -- MODEL_KEY is the only thing you set
MODEL_KEY = "llama2-7b"          # llama2-7b | llama2-13b | llama31-8b | olmo3-7b
CONFIRMED_BUDGET_HOURS = 1.5     # per row; anomaly threshold, not a target

import torch, time
T0 = time.time()
assert torch.cuda.is_available(), "no GPU"
p = torch.cuda.get_device_properties(0)
GIB = p.total_memory / 1024**3
print(p.name, round(GIB, 1), "GiB")

# 13B fp16/bf16 needs ~26GB of weights alone.
if MODEL_KEY == "llama2-13b":
    assert GIB >= 38, f"13B anchor needs A100-40G+, got {GIB:.1f} GiB. Do NOT " \\
                      "quantize: the confound would land on the anchor row."
else:
    assert GIB >= 20, f"need >=20 GiB, got {GIB:.1f}"

# bf16 where native. All four rows must share a dtype -- it is pinned into the
# output and the aggregator refuses to mix.
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) \\
        else torch.float16
print("dtype", DTYPE)

import transformers
assert transformers.__version__.startswith("5."), transformers.__version__

def elapsed_h():
    return (time.time() - T0) / 3600.0

def budget_check(where):
    """CONFIRMED_BUDGET only firing at startup provably cannot catch an overrun.
    Wide-catalog attempt 2 ran 76 minutes and persisted nothing."""
    e = elapsed_h()
    assert e <= CONFIRMED_BUDGET_HOURS, (
        f"OVERRUN at {where}: {e:.2f}h > {CONFIRMED_BUDGET_HOURS}h. "
        "Partial results are already persisted. Diagnose -- do not raise this "
        "to fit; a blowout here means the cost model is wrong.")
    return e
''')

code(f'''# [2] pinned adjudicator + self-test ON THE BOX before any spend
import hashlib, pathlib
PL_SHA = "{SHA}"
PL_SRC = {SRC!r}
assert hashlib.sha256(PL_SRC.encode()).hexdigest() == PL_SHA, "artifact drift"
pathlib.Path("position_ladder.py").write_text(PL_SRC)

import position_ladder as PL
PL._selftest()
print("\\nadjudicator self-test passed on this box; sha", PL_SHA[:16])
''')

code('''# [3] persistence -- private HF dataset repo, keyed by config hash
# Drive's OAuth popup is unreachable in a hands-off run (verified). HF_TOKEN is
# a Colab secret, silent on fresh VMs, and uploads are atomic per file.
import os, io, json, hashlib
from google.colab import userdata
from huggingface_hub import HfApi, create_repo

HF_TOKEN = userdata.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
WHO = api.whoami()["name"]
DS = f"{WHO}/phi-map-position-ladder"
create_repo(DS, repo_type="dataset", private=True, exist_ok=True, token=HF_TOKEN)
print("persisting to", DS)

WORK = "/content/pl"; os.makedirs(WORK, exist_ok=True)

def persist(name, blob: bytes):
    open(os.path.join(WORK, name), "wb").write(blob)
    api.upload_file(path_or_fileobj=io.BytesIO(blob), path_in_repo=name,
                    repo_id=DS, repo_type="dataset", token=HF_TOKEN)

def fetch(name):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(DS, name, repo_type="dataset", token=HF_TOKEN)
        return json.load(open(p))
    except Exception:
        return None
''')

code('''# [4] PHASE 0 -- access gate for ALL FOUR repos, then pin N
# All four are checked even though this runtime runs one, because a 401 on a
# later row after three rows are already paid for is the expensive failure.
from huggingface_hub import model_info
from transformers import AutoTokenizer

bad = []
for k, spec in PL.LADDER.items():
    try:
        info = model_info(spec["hf"], token=HF_TOKEN)
        print(f"  ok    {k:12s} {spec['hf']}  @{info.sha[:12]}")
    except Exception as e:
        bad.append((k, spec["hf"], type(e).__name__))
        print(f"  FAIL  {k:12s} {spec['hf']}  {type(e).__name__}")
assert not bad, (
    f"gated/unavailable: {bad}. Accept the licence on huggingface.co under the "
    "same account as HF_TOKEN, then re-run. Failing here costs nothing; failing "
    "at load time costs the rows already run.")

# N is a context-window fact, fixed once by Llama-2's 4096 window -- the binding
# constraint for the WHOLE ladder -- and identical for every row.
tok_l2 = AutoTokenizer.from_pretrained(PL.LADDER["llama2-7b"]["hf"], token=HF_TOKEN)
N_PAIRS, WORST = PL.select_n_pairs(tok_l2)
print(f"\\nN_PAIRS = {N_PAIRS} (worst-case {WORST} tokens vs 4096 window)")
assert N_PAIRS in PL.N_CANDIDATES
''')

code('''# [5] load THIS model, pin revision, re-assert the prompt fits ITS tokenizer
from transformers import AutoModelForCausalLM
SPEC = PL.LADDER[MODEL_KEY]
REPO = SPEC["hf"]
REV = model_info(REPO, token=HF_TOKEN).sha          # pinned into the output
print(MODEL_KEY, REPO, "@", REV[:12])

tok = AutoTokenizer.from_pretrained(REPO, revision=REV, token=HF_TOKEN)
worst_here = PL.assert_fits(tok, N_PAIRS,
                            max_ctx=4096 if MODEL_KEY.startswith("llama2") else 8192)
print("worst-case prompt on this tokenizer:", worst_here, "tokens")

model = AutoModelForCausalLM.from_pretrained(
    REPO, revision=REV, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN)
model.eval()

# device_map="auto" does NOT fail when VRAM is short -- it offloads layers to
# CPU and then to disk, and the run merely crawls. That is indistinguishable
# from a hang and is how wide-catalog attempt 2 burned 76 minutes and persisted
# nothing. Fail here instead.
dmap = getattr(model, "hf_device_map", {}) or {}
offloaded = {k: str(v) for k, v in dmap.items()
             if str(v) in ("cpu", "disk") or "cpu" in str(v) or "disk" in str(v)}
assert not offloaded, (
    f"{len(offloaded)} module(s) offloaded off-GPU: {list(offloaded)[:5]}. "
    f"Generation would crawl rather than fail. Use a larger GPU -- do not "
    f"quantize to fit, especially not on an anchor row.")
print(f"device map: {len(dmap)} modules, all on GPU"
      if dmap else "device map: single device")
print("VRAM after load: %.1f GiB" % (torch.cuda.memory_allocated() / 1024**3))

tok.padding_side = "left"                 # LEFT-pad: this is a GENERATION pass
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print("padding_side", tok.padding_side, "| pad", tok.pad_token_id)
''')

code('''# [6] SERVED-PROMPT GATE -- print what actually reaches the model
# The one cheap countermeasure that would have caught the burned wide-catalog
# run: every notebook that worked printed the SERVED prompt; the one that failed
# gated the manifest and never checked what the model was asked to do.
trials = PL.build_trials(n_pairs=N_PAIRS)
print(f"{len(trials)} trials = {PL.N_POSITIONS} positions x {PL.TRIALS_PER_POSITION}")

def served(row):
    msgs = [{"role": "user", "content": row["prompt"]}]
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # transformers v5 can return a dict/BatchEncoding even for tokenize=False
    if not isinstance(out, str):
        out = out[0] if isinstance(out, (list, tuple)) else str(out)
    return out

ex = served(trials[0])
print("=" * 78)
print(ex[:600], "\\n   ...[middle elided]...\\n", ex[-400:])
print("=" * 78)
assert trials[0]["gold_key"] in ex, "gold key absent from the served prompt"
assert trials[0]["gold_value"] in ex, "gold value absent from the served prompt"
assert "Extract the value" in ex, "served prompt does not ask for a retrieval"
assert ex.count(trials[0]["gold_key"]) == 2, "key must appear in JSON and in query"
print("served-prompt gate OK")

CONFIG = dict(model_key=MODEL_KEY, repo=REPO, revision=REV, dtype=str(DTYPE),
              n_pairs=N_PAIRS, seed=1101, trials_per_position=PL.TRIALS_PER_POSITION,
              positions=PL.gold_positions(N_PAIRS), pl_sha=PL_SHA)
CFG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]
print("config hash", CFG_HASH)
''')

code('''# [7] generate -- greedy, checkpointed after EVERY position block
import collections
# BATCH is a THROUGHPUT knob, not a design parameter: greedy decoding with
# left-padding means batch composition does not change any row's answer. It is
# held constant across models anyway so the rows stay strictly comparable.
#
# It is 4, not 8, because Llama-2-13B is MHA (40 layers x 40 heads x 128, no
# GQA) and its KV cache at batch 8 x ~2600 tokens is ~17 GB on top of 26 GB of
# weights -- over a 40 GB card. The warm-up probe below settles it empirically
# instead of trusting that arithmetic.
BATCH, MAXNEW = 4, 64
CKPT = f"gen_{MODEL_KEY}_{CFG_HASH}.json"

done = fetch(CKPT) or {}
if done:
    print(f"resuming: {len(done)} position blocks already persisted")

# WARM-UP PROBE: one real batch at the real size and the real prompt length,
# measuring peak VRAM. Fails in seconds instead of OOMing 300 batches into the
# 13B row. The longest prompt is the worst case, so probe that one.
torch.cuda.reset_peak_memory_stats()
_probe = sorted(trials, key=lambda r: -len(r["prompt"]))[:BATCH]
_enc = tok([served(r) for r in _probe], return_tensors="pt", padding=True,
           add_special_tokens=False).to(model.device)
with torch.no_grad():
    model.generate(**_enc, max_new_tokens=MAXNEW, do_sample=False,
                   pad_token_id=tok.pad_token_id)
PEAK = torch.cuda.max_memory_allocated() / 1024**3
print(f"warm-up: {_enc['input_ids'].shape} peak {PEAK:.1f} / {GIB:.1f} GiB "
      f"({100 * PEAK / GIB:.0f}%)")
assert PEAK < 0.90 * GIB, (
    f"peak {PEAK:.1f} GiB is >90% of {GIB:.1f}; lower BATCH and re-run. Do NOT "
    f"quantize and do not shorten the prompt -- N is fixed by the ladder.")
del _enc; torch.cuda.empty_cache()

by_pos = collections.defaultdict(list)
for r in trials:
    by_pos[r["gold_position"]].append(r)

for pos in sorted(by_pos):
    if str(pos) in done:
        print(f"  pos {pos:3d} cached"); continue
    rows, outs = by_pos[pos], []
    for i in range(0, len(rows), BATCH):
        batch = [served(r) for r in rows[i:i + BATCH]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        outs += [tok.decode(s, skip_special_tokens=True) for s in new]
        budget_check(f"pos {pos} batch {i}")
    done[str(pos)] = [dict(gold_key=r["gold_key"], gold_value=r["gold_value"],
                           trial=r["trial"], text=t) for r, t in zip(rows, outs)]
    persist(CKPT, json.dumps(done).encode())     # after EVERY block
    print(f"  pos {pos:3d} done  {len(outs)} gens  {elapsed_h():.2f}h  persisted")
print("generation complete", f"{elapsed_h():.2f}h")
''')

code('''# [8] score, curve, per-model verdict
scored = []
for pos, recs in done.items():
    for r in recs:
        s = PL.score_one(r["text"], r["gold_key"], r["gold_value"])
        s["gold_position"] = int(pos)
        scored.append(s)

curve = PL.curve(scored)
scorable_frac = sum(s["scorable"] for s in scored) / len(scored)
row_result = dict(curve={str(k): v for k, v in curve.items()},
                  scorable_frac=scorable_frac, n=len(scored))
verdict = PL.classify(curve) if scorable_frac >= PL.BARS["scorable"] else \\
          dict(verdict="V_UNSCORABLE", reasons=[f"scorable {scorable_frac:.3f}"])

print(f"scorable {scorable_frac:.4f}  n={len(scored)}")
for p in sorted(curve):
    print(f"  pos {p:3d}  acc {curve[p]:.4f}")
print("\\nverdict:", json.dumps(verdict, indent=1))
if MODEL_KEY in PL.ANCHOR_PREDICTION:
    want = PL.ANCHOR_PREDICTION[MODEL_KEY]
    print(f"\\nANCHOR ROW: Liu et al. predict {want}, got {verdict['verdict']} "
          f"-> {'MATCH' if verdict['verdict'] == want else 'MISMATCH'}")
''')

code('''# [9] hand verification + persist the row
print("=" * 78, "\\nHAND VERIFICATION: 3 generations per outcome\\n", "=" * 78)
buckets = {"correct": [], "wrong": [], "unscorable": []}
for pos, recs in done.items():
    for r in recs:
        s = PL.score_one(r["text"], r["gold_key"], r["gold_value"])
        b = "unscorable" if not s["scorable"] else ("correct" if s["correct"] else "wrong")
        if len(buckets[b]) < 3:
            buckets[b].append((pos, r, s))
for b, items in buckets.items():
    for pos, r, s in items:
        print(f"\\n--- pos {pos} scored {b}")
        print("    gold :", r["gold_value"])
        print("    text :", repr(r["text"][:200]))

summary = dict(protocol="POSITION_LADDER_V1", config=CONFIG, cfg_hash=CFG_HASH,
               result=row_result, verdict=verdict,
               anchor_prediction=PL.ANCHOR_PREDICTION.get(MODEL_KEY),
               wall_h=round(elapsed_h(), 3), gpu=p.name)
blob = json.dumps(summary, indent=1).encode()
persist(f"row_{MODEL_KEY}_{CFG_HASH}.json", blob)
print("\\nsha256:", hashlib.sha256(blob).hexdigest()[:16])
print("=== ROW SUMMARY BEGIN ===")
print(json.dumps(summary, indent=1))
print("=== ROW SUMMARY END ===")
print("\\nNext: re-run with the next MODEL_KEY on a FRESH runtime.")
print("When all four rows exist: python3 position_ladder_aggregate.py row_*.json")
''')

nb = dict(cells=cells, metadata=dict(
    accelerator="GPU",
    colab=dict(provenance=[], gpuType="A100"),
    kernelspec=dict(display_name="Python 3", name="python3"),
    language_info=dict(name="python")), nbformat=4, nbformat_minor=0)

out = REPO / "position_ladder_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells, module sha {SHA[:16]})")
