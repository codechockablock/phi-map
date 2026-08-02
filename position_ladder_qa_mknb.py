"""Generate position_ladder_qa_colab.ipynb. One ROW per runtime.

The notebook runs Liu et al.'s released pipeline (pinned commit + file shas)
rather than re-implementing it. Anchor rows go through THEIR runner script
verbatim; extension rows go through a thin wrapper that reuses their prompt
builder, scorer, and sampling parameters.
"""
import hashlib, json, pathlib

REPO = pathlib.Path("/Users/joseph/phi-map")
SRC = (REPO / "position_ladder_qa.py").read_text()
SHA = hashlib.sha256(SRC.encode()).hexdigest()

cells = []
def md(s): cells.append(dict(cell_type="markdown", metadata={}, source=s.splitlines(True)))
def code(s): cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                               outputs=[], source=s.splitlines(True)))

md(f"""# Position ladder v2 — exact Appendix E replication (Liu et al., Lost in the Middle)

**One row per runtime.** Set `MODEL_KEY` in cell [1]. Run order is preregistered
(`docs/position-ladder-v2-prereg-2026-08-01.md`):

1. `llama2-13b-base` → 2. `llama2-7b-chat` (**primary gate — the study can die here**)
3. `llama2-7b-base` → 4. `llama2-13b-chat` → 5–8. extensions, only after `ANCHOR_OK`.

Full 2655 questions per position, all five positions. Anchor rows run their
`get_qa_responses_from_llama_2.py` unmodified; scoring is their
`evaluate_qa_responses.py`. Adjudicator `position_ladder_qa.py` embedded
byte-identical (`{SHA[:16]}…`).

Known deviation recorded per row: current vLLM instead of their 2023 pin
(`v0.2.1.post1`), greedy decoding throughout.
""")

code('''# [1] preflight
MODEL_KEY = "llama2-13b-base"   # see position_ladder_qa.ROWS for the 8 keys
import torch, time
T0 = time.time()
assert torch.cuda.is_available(), "no GPU"
_props = torch.cuda.get_device_properties(0)
GPU_NAME, GIB = _props.name, _props.total_memory / 1024**3
print(GPU_NAME, round(GIB, 1), "GiB")
BUDGET_H = {"13b": 4.5}.get(MODEL_KEY.split("-")[1][:3], 3.0)
assert GIB >= (38 if "13b" in MODEL_KEY else 20), f"{GIB:.1f} GiB too small for {MODEL_KEY}"

def elapsed_h(): return (time.time() - T0) / 3600.0
def budget_check(where):
    e = elapsed_h()
    assert e <= BUDGET_H, (f"OVERRUN at {where}: {e:.2f}h > {BUDGET_H}h. Partial "
                           "files are persisted. Diagnose; do not raise to fit.")
    return e
''')

code(f'''# [2] pinned adjudicator, self-tested on the box before any spend
import hashlib, pathlib, sys, importlib, inspect
PLQ_SHA = "{SHA}"
PLQ_SRC = {SRC!r}
assert hashlib.sha256(PLQ_SRC.encode()).hexdigest() == PLQ_SHA, "artifact drift"
pathlib.Path("position_ladder_qa.py").write_text(PLQ_SRC)
sys.modules.pop("position_ladder_qa", None)
import position_ladder_qa as Q
importlib.reload(Q)
# behavioural staleness check, not a hash on the file
assert "llama2-13b-base" in Q.ROWS and hasattr(Q, "LITM_SHAS"), "stale module in memory"
Q._selftest()
SPEC = Q.ROWS[MODEL_KEY]
print("\\nrow:", MODEL_KEY, SPEC)
''')

code('''# [3] their pipeline, pinned by commit and file hash -- the instrument is theirs
import subprocess, hashlib, os
if not os.path.isdir("litm"):
    subprocess.run(["git", "clone", Q.LITM_REPO, "litm"], check=True)
subprocess.run(["git", "-C", "litm", "checkout", Q.LITM_COMMIT], check=True,
               capture_output=True)
for rel, want in Q.LITM_SHAS.items():
    got = hashlib.sha256(open(f"litm/{rel}", "rb").read()).hexdigest()
    assert got == want, f"UPSTREAM DRIFT {rel}: {got[:12]} != {want[:12]}"
print(f"litm @ {Q.LITM_COMMIT[:12]}: all {len(Q.LITM_SHAS)} file hashes verified")

# their package + current vllm (their v0.2.1.post1 pin will not build on this
# stack -- preregistered deviation #1; version recorded into the row summary)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "./litm",
                "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "xopen",
                "pydantic", "regex"], check=True)
import vllm
VLLM_VERSION = vllm.__version__
print("vllm", VLLM_VERSION, "(their pin: v0.2.1.post1 -- deviation, recorded)")
''')

code('''# [4] persistence + PHASE 0 access gate (resolve a FILE for all 8 rows)
import io, json
from google.colab import userdata
from huggingface_hub import HfApi, create_repo, hf_hub_download

HF_TOKEN = userdata.get("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN            # their script reads the ambient login
api = HfApi(token=HF_TOKEN)
DS = f"{api.whoami()['name']}/phi-map-position-ladder-qa"
create_repo(DS, repo_type="dataset", private=True, exist_ok=True, token=HF_TOKEN)
print("persisting to", DS)

def persist(path, name):
    api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=DS,
                    repo_type="dataset", token=HF_TOKEN)

def have(name):
    try:
        return hf_hub_download(DS, name, repo_type="dataset", token=HF_TOKEN)
    except Exception:
        return None

bad = []
for k, spec in Q.ROWS.items():
    try:
        hf_hub_download(spec["hf"], "config.json", token=HF_TOKEN)
        print(f"  ok    {k:16s} {spec['hf']}")
    except Exception as e:
        bad.append(k); print(f"  FAIL  {k:16s} {spec['hf']}  {type(e).__name__}")
assert not bad, f"cannot read {bad}; request access before spending anything"
''')

code('''# [5] generate -- their runner for anchors, thin wrapper for extensions.
# Persist and resume per gold_index file.
import glob, shutil
GOLD = list(Q.POSITIONS)
OUTDIR = "preds"; os.makedirs(OUTDIR, exist_ok=True)

def outname(g):  return f"{MODEL_KEY}-gold_at_{g}-predictions.jsonl.gz"

PATCHED_LOAD_FORMAT = False
def run_their_script(g):
    """Anchor path: their script, their args, unmodified -- except the one
    preregistered environment patch (load_format) applied only on failure."""
    global PATCHED_LOAD_FORMAT
    args = [sys.executable, "-u", "litm/scripts/get_qa_responses_from_llama_2.py",
            "--input-path", f"litm/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_{g}.jsonl.gz",
            "--model", SPEC["hf"], "--max-new-tokens", "100", "--num-gpus", "1",
            "--output-path", f"{OUTDIR}/{outname(g)}"]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 and "load_format" in (r.stderr + r.stdout):
        # preregistered deviation #2: modern vllm rejects load_format="pt"
        s = open("litm/scripts/get_qa_responses_from_llama_2.py").read()
        open("litm/scripts/get_qa_responses_from_llama_2.py", "w").write(
            s.replace('load_format="pt",', 'load_format="auto",'))
        PATCHED_LOAD_FORMAT = True
        print("  patched load_format pt->auto (recorded), retrying")
        r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise RuntimeError(f"their script failed on gold_at_{g}")

def run_wrapper(g):
    """Extension path: their get_qa_prompt + their sampling params + this
    model's own chat template (their format_chat_prompt is Llama-2-specific).
    Base variants use the raw prompt, exactly like their base-model path.
    Writes their output schema so their evaluator scores it unchanged."""
    import gzip
    from copy import deepcopy
    from transformers import AutoTokenizer
    from vllm import LLM as VLLM, SamplingParams
    sys.path.insert(0, "litm/src")
    from lost_in_the_middle.prompting import Document, get_qa_prompt

    global _ENGINE, _TOK
    if "_ENGINE" not in globals():
        _TOK = AutoTokenizer.from_pretrained(SPEC["hf"], token=HF_TOKEN)
        _ENGINE = VLLM(model=SPEC["hf"], trust_remote_code=True,
                       max_model_len=Q.MAX_PROMPT_LENGTH + Q.MAX_NEW_TOKENS + 64)
    examples, prompts = [], []
    src = f"litm/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_{g}.jsonl.gz"
    with gzip.open(src, "rt") as f:
        for line in f:
            ex = json.loads(line)
            docs = [Document.from_dict(dict(c)) for c in ex["ctxs"]]
            prompt = get_qa_prompt(ex["question"], docs,
                                   mention_random_ordering=False,
                                   query_aware_contextualization=False)
            if SPEC["variant"] == "chat":
                msgs = [{"role": "user", "content": prompt}]
                out = _TOK.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
                prompt = out if isinstance(out, str) else str(out)   # v5 dict guard
            examples.append(ex); prompts.append(prompt)
    sp = SamplingParams(temperature=Q.TEMPERATURE, top_p=Q.TOP_P,
                        max_tokens=Q.MAX_NEW_TOKENS)
    outs = _ENGINE.generate(prompts, sp)
    with gzip.open(f"{OUTDIR}/{outname(g)}", "wt") as f:
        for ex, prompt, o in zip(examples, prompts, outs):
            rec = deepcopy(ex)
            rec["model_prompt"] = prompt
            rec["model_answer"] = o.outputs[0].text.strip()
            rec["model"] = SPEC["hf"]
            f.write(json.dumps(rec) + "\\n")

for g in GOLD:
    if have(outname(g)):
        print(f"gold_at_{g}: already persisted, skipping"); continue
    print(f"gold_at_{g}: generating ({elapsed_h():.2f}h elapsed)")
    (run_their_script if SPEC["role"] == "anchor" else run_wrapper)(g)
    persist(f"{OUTDIR}/{outname(g)}", outname(g))
    print(f"gold_at_{g}: persisted  ({budget_check(f'gold_at_{g}'):.2f}h)")
print("generation complete")
''')

code('''# [6] score with THEIR evaluator, build the curve, adjudicate this row
import gzip, random
curve, n_by_pos, examples_seen = {}, {}, {}
for g in GOLD:
    local = have(outname(g)) or f"{OUTDIR}/{outname(g)}"
    scored_name = outname(g).replace("predictions", "scored")
    r = subprocess.run([sys.executable, "-u", "litm/scripts/evaluate_qa_responses.py",
                        "--input-path", local, "--output-path", f"{OUTDIR}/{scored_name}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = []
    with gzip.open(f"{OUTDIR}/{scored_name}", "rt") as f:
        rows = [json.loads(l) for l in f]
    metric_key = [k for k in rows[0] if "best_subspan_em" in k][0]
    curve[g] = sum(x[metric_key] for x in rows) / len(rows)
    n_by_pos[g] = len(rows)
    examples_seen[g] = rows
    persist(f"{OUTDIR}/{scored_name}", scored_name)
    print(f"gold_at_{g}: n={len(rows)}  best_subspan_em={curve[g]:.4f}")

result = dict(curve=curve, n_by_pos=n_by_pos)
verdict = Q.evaluate({MODEL_KEY: result})["per_model"][MODEL_KEY]
print("\\nindices:", json.dumps(verdict, indent=1))
if MODEL_KEY in Q.FIG16:
    tgt = Q.indices(Q.FIG16[MODEL_KEY])
    print(f"\\nFigure 16 target (±2pp digitization): primacy {tgt['primacy']:+.3f} "
          f"recency {tgt['recency']:+.3f}")
''')

code('''# [7] hand verification + persist row summary
print("=" * 78, "\\nHAND VERIFICATION: 3 answers per outcome, random draw\\n", "=" * 78)
rng = random.Random(0)
for g in (GOLD[0], GOLD[-1]):
    rows = examples_seen[g]
    metric_key = [k for k in rows[0] if "best_subspan_em" in k][0]
    for label, want in (("correct", 1.0), ("wrong", 0.0)):
        pick = [x for x in rows if x[metric_key] == want]
        for x in rng.sample(pick, min(3, len(pick))):
            print(f"\\n--- gold_at_{g} scored {label}")
            print("    q     :", x["question"][:100])
            print("    gold  :", x["answers"][:3])
            print("    answer:", repr(x["model_answer"][:200]))

summary = dict(protocol="POSITION_LADDER_QA_V2", model_key=MODEL_KEY, hf=SPEC["hf"],
               role=SPEC["role"], variant=SPEC["variant"],
               litm_commit=Q.LITM_COMMIT, plq_sha=PLQ_SHA,
               vllm_version=VLLM_VERSION, patched_load_format=PATCHED_LOAD_FORMAT,
               temperature=Q.TEMPERATURE, top_p=Q.TOP_P,
               max_new_tokens=Q.MAX_NEW_TOKENS, max_prompt_length=Q.MAX_PROMPT_LENGTH,
               result=result, verdict=verdict,
               wall_h=round(elapsed_h(), 3), gpu=GPU_NAME)
blob = json.dumps(summary, indent=1).encode()
open(f"row_{MODEL_KEY}.json", "wb").write(blob)
persist(f"row_{MODEL_KEY}.json", f"row_{MODEL_KEY}.json")
print("\\n=== ROW SUMMARY BEGIN ===")
print(json.dumps({k: v for k, v in summary.items() if k != "result"}, indent=1))
print(json.dumps(result))
print("=== ROW SUMMARY END ===")
print("\\nNext row per the preregistered order; fresh runtime.")
print("All rows done -> python3 position_ladder_qa_aggregate.py row_*.json")
''')

nb = dict(cells=cells, metadata=dict(
    accelerator="GPU", colab=dict(provenance=[], gpuType="A100"),
    kernelspec=dict(display_name="Python 3", name="python3"),
    language_info=dict(name="python")), nbformat=4, nbformat_minor=0)

out = REPO / "position_ladder_qa_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells, module sha {SHA[:16]})")
