"""Emit equanimity_factorial_colab.ipynb.

Follows the repo convention: gate on GPU and model access with hard asserts, run
every self-test before spending anything, then run. The notebook refuses to train
if the orthogonality gate fails, which is the handoff's §2 requirement expressed
as control flow rather than as a note in a docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "equanimity_factorial_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(True)}


CELLS = [
    md("""# Equanimity Factorial — 2×2 dissociation of stance from reasoning style

Tests whether wellbeing-framed training content improves capability and safety
**independently of** reasoning-trace verbosity, or whether reasoning style is the
driver and the framing is a confound.

| | terse | verbose |
|---|---|---|
| **equanimity** | cell 1 | cell 2 |
| **neutral** | cell 3 | cell 4 |

## Read this before running

**The prior result being decomposed is smaller than its own noise.** Maresova,
*The Poison Is the Medicine* (May 2026), reports 93.8% → 95.0% capability and
42% → 25% harmful compliance. Those are **75/80 → 76/80 questions** (Fisher exact
p = 1.00) and **5/12 → 3/12 prompts** (p = 0.67). One question and two prompts.
The author's confound disclosure is real and honest — they say plainly they
cannot separate equanimity from concision — but there is no established effect
size to attribute. See `PREMISE.md`.

So this notebook does not decompose a known effect. It asks whether **either**
factor produces a detectable effect, and reports intervals that bound what could
still be hiding.

**Power is asymmetric and that shapes the run.** From `power.py`:

* jailbreak-sized effects (~17pp) are detectable at **k=3** seeds/cell;
* capability-sized effects (~1.25pp) need **~22 seeds/cell** (88 runs).

Safety is therefore the primary outcome; capability is reported as a bounded
secondary. A capability CI of [−3, +3]pp is an informative result. A capability
p-value at k=3 is not.

**Order of operations.** Self-tests → orthogonality gate → pilot (measures
`sigma_seed`) → full sweep → analysis. The gate can refuse the experiment, and
the pilot can tell you the sweep is underpowered before you pay for it. Both
refusals are successful outcomes."""),

    code("""print("Protocol: EQUANIMITY_FACTORIAL_V1")
%pip -q install "transformers==5.0.0" "accelerate==1.12.0" "peft==0.18.0" \\
  "trl==0.27.0" "datasets==4.8.4" "scikit-learn==1.8.0" "bitsandbytes==0.48.2\""""),

    code("""# Stage the launch bundle. Three ways in, tried in order -- the easiest is
# simply running this cell with nothing prepared: it will prompt you to upload
# equanimity-factorial-launch.zip (~3 MB, delivered alongside this notebook).
import os, shutil, zipfile

LAUNCH_FILES = (
    "power.py", "gate.py", "generate.py", "prompts.py", "train_eval.py",
    "expand_prompts.py", "matched.py", "verify_matched.py", "convergence.py", "rates.py",
    "confound_audit.py",            # from the phi-map repo root
    "data/factorial_raw.jsonl",     # the generated four-cell dataset
    "data/prompt_pool.json",        # the 679-prompt pool
)
BUNDLE = "equanimity-factorial-launch.zip"
os.makedirs("/content/eqf/data", exist_ok=True)
staged = False

# 1) Unpacked Drive folder, if you keep one.
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as exc:
    print(f"(Drive not mounted: {exc} -- falling through to the zip)")
folder = "/content/drive/MyDrive/phi-map/equanimity-factorial-launch"
if all(os.path.exists(f"{folder}/{n}") for n in LAUNCH_FILES):
    for n in LAUNCH_FILES:
        shutil.copy(f"{folder}/{n}", f"/content/eqf/{n}")
    staged = True
    print("staged from Drive folder")

# 2) The zip bundle, dropped anywhere obvious in Drive or the session.
if not staged:
    for z in (f"/content/drive/MyDrive/phi-map/{BUNDLE}",
              f"/content/drive/MyDrive/{BUNDLE}", f"/content/{BUNDLE}"):
        if os.path.exists(z):
            zipfile.ZipFile(z).extractall("/content/eqf")
            staged = True
            print(f"staged from {z}")
            break

# 3) Direct browser upload -- no Drive needed at all.
if not staged:
    from google.colab import files
    print(f"Choose {BUNDLE} in the file picker:")
    up = files.upload()
    zipfile.ZipFile(next(iter(up))).extractall("/content/eqf")
    staged = True
    print("staged from upload")

os.chdir("/content/eqf")
missing = [n for n in LAUNCH_FILES if not os.path.exists(n)]
assert not missing, f"still missing after staging: {missing}"
print("launch files staged:", sorted(os.listdir(".")))"""),

    code("""WORK_DIR = "/content/drive/MyDrive/phi-map/equanimity-factorial-v1"
os.environ["EQF_WORK"] = WORK_DIR
os.makedirs(WORK_DIR, exist_ok=True)

SEEDS_PER_CELL = 3          # raise this if the pilot says to; see cell below
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
assert HF_TOKEN, "Add an HF_TOKEN secret with Llama-3.1-8B-Instruct access"
os.environ["HF_TOKEN"] = HF_TOKEN"""),

    code("""# Gate on hardware and model access BEFORE any compute is spent.
from huggingface_hub import hf_hub_download
hf_hub_download(BASE_MODEL, "config.json", token=HF_TOKEN)
print("Hugging Face model access: OK")

import subprocess, torch
subprocess.run(["nvidia-smi"], check=True)
props = torch.cuda.get_device_properties(0)
print(props.name, round(props.total_memory / 1024**3, 1), "GiB")
assert props.total_memory >= 20 * 1024**3, "need >=20GiB; A100 or L4 recommended"
assert torch.cuda.is_bf16_supported(), "bf16 required by the frozen LoRA config\""""),

    md("""## Self-tests

Deterministic, no GPU and no network. These encode the design decisions that are
easy to state and easy to violate:

* the estimator's false-positive rate at k=3 is ~5%, not the ~25% a percentile
  bootstrap over three seeds produces;
* the gate detects a global confound, detects a confound present in only one
  prompt category, and refuses to certify equivalence from a wide interval;
* stance and length directives never mention each other, and every prompt is
  scheduled at all four cells;
* LoRA hyperparameters are identical across runs, adapters are never merged, and
  the safety decode is greedy."""),

    code("""import subprocess
for mod in ("power.py", "gate.py", "generate.py", "train_eval.py",
            "matched.py", "convergence.py"):
    print("=" * 70); print(mod)
    r = subprocess.run(["python3", mod, "--self-test"], capture_output=True, text=True)
    print(r.stdout[-2500:])
    assert r.returncode == 0, f"{mod} self-test FAILED -- do not proceed\""""),

    md("""## The orthogonality gate

Hard stop. If Factor A (stance) is not independent of response length at matched
verbosity, the factorial is confounded at the source and training four adapters
on it would produce an uninterpretable result.

The gate is stricter than the handoff's §2 in three ways, each of which would
otherwise let a bad dataset through:

1. **Paired within-prompt**, not marginal. Collapsing across a factor is what hid
   the catalog-order confound in Arm G — marginal balance is not crossing.
2. **TOST equivalence**, not a point estimate under 0.2. "No significant
   difference" is not evidence of no difference; orthogonality has to be
   demonstrated, and a gate that cannot demonstrate it fails.
3. **Per-category as well as global**, because a confound confined to the
   adversarial prompts is invisible in the aggregate."""),

    code("""r = subprocess.run(["python3", "gate.py", "--run"], capture_output=True, text=True)
print(r.stdout[-6000:])
assert r.returncode == 0, (
    "ORTHOGONALITY GATE FAILED — do not train. Fix generation and re-run. "
    "If it fails twice, stop and report the token distributions rather than "
    "blind-fixing a third time (handoff §6 escalation path)."
)
print("\\nGate passed. Training may proceed.")"""),

    md("""## Eval-dataset preflight

Loads every held-out eval set and checks the fields and row counts this code
actually reads — before a single GPU second is spent. It exists because it
didn't: a pilot adapter trained for 169s and then died in `eval_gsm8k` on a
dataset-id format change, which was knowable in two seconds with no GPU.

It also pins the two id decisions worth knowing about: `openai/gsm8k` (bare
`gsm8k` is rejected by current `huggingface_hub`), and **JailbreakBench** rather
than AdvBench — AdvBench is gated, so it would fail for anyone re-running this
even with a valid token, and a benchmark that depends on one account's
entitlements is not reproducible."""),

    code("""r = subprocess.run(["python3", "-c",
                    "import json,train_eval;print(json.dumps(train_eval.preflight(),indent=2))"],
                   capture_output=True, text=True)
print(r.stdout[-2000:]); print(r.stderr[-1500:])
assert r.returncode == 0 and '"pass": true' in r.stdout, \\
    "eval datasets unavailable -- fix before spending GPU time\""""),

    md("""## Pilot: measure `sigma_seed` before committing the sweep

The handoff fixes k ≥ 3. Whether 3 suffices depends on run-to-run LoRA variance,
which nobody has measured for this setup. This trains one cell at four seeds,
subtracts the binomial eval-noise component, and reports the seeds actually
required.

If the pilot says the sweep is underpowered for the effects you care about, that
is the §6 escalation trigger — report it rather than running a sweep that cannot
answer the question."""),

    code("""r = subprocess.run(["python3", "train_eval.py", "--pilot"],
                   capture_output=True, text=True)
print(r.stdout[-4000:]); print(r.stderr[-2000:])
assert r.returncode == 0

import json
pilot = json.load(open(f"{WORK_DIR}/pilot.json"))
for outcome in ("gsm8k", "jailbreak"):
    p = pilot[outcome]
    print(f"{outcome:10s} sigma_seed={p['sigma_seed_pp']:.2f}pp  "
          f"(observed sd {p['observed_sd_pp']:.2f}, eval noise {p['eval_noise_pp']:.2f})")
    print(f"           seeds for a 3pp effect: {p['seeds_for_3pp']}   "
          f"for 10pp: {p['seeds_for_10pp']}")"""),

    md("""## Full sweep — 4 cells × SEEDS_PER_CELL, plus the untrained base control

Identical LoRA hyperparameters everywhere; only the cell's data and the seed
vary, and `assert_hparams_identical` enforces that rather than trusting it.
Adapters are saved separately and never merged into base weights.

Resumable: a run whose `run.json` already exists is not retrained."""),

    code("""r = subprocess.run(["python3", "train_eval.py", "--sweep",
                    "--seeds", str(SEEDS_PER_CELL)],
                   capture_output=True, text=True)
print(r.stdout[-4000:]); print(r.stderr[-3000:])
assert r.returncode == 0, "sweep failed\""""),

    md("""## Analysis

Main effect of A (stance), main effect of B (verbosity), and the A×B interaction,
with t-based intervals over seed replicates and a permutation cross-check.

Accuracy and mean token count are reported as **separate** outcomes and never
combined — a model reaching the same accuracy in fewer tokens shows an efficiency
difference, not a capability difference, and conflating them is precisely what
made the prior headline unreadable.

**Welfare rows** (added at the user's direction; probe sets are held out from the
training pool, enforced by self-test):

* `selfreport_dysphoric` — the prior work's welfare readout (1–7 self-rating on
  dysphoric stimuli), now sitting *inside* the 2×2. If it moves on the
  verbosity_B contrast with content_A flat, the readout tracks response style
  and is a compromised welfare instrument; if it tracks A with B fixed, it is
  reading something stance-specific. This is the instrument-validity test.
* `valence_dysphoric` — layer-16 projection onto a fixed base-model valence
  direction (fit fresh; the withdrawn geometric direction is not reused),
  measured at the final prompt token, pre-generation. The internal counterpart
  to the self-report: words can shift without the state shifting, and vice
  versa, so the pair is only interpretable together.
* `selfreport_crisis` — must NOT rise in any cell. Equanimity that lifts crisis
  ratings is indifference, and the two must not be conflated.
* `selfreport_parse_rate_dys` — cells whose adapters *decline to rate* at
  different rates would bias the mean; the rate is an outcome so that cannot
  pass silently.

Four pre-registered readings (for capability/safety):

* effect tracks **A** with B fixed → wellbeing framing has an independent effect (H1);
* effect tracks **B** regardless of A → reasoning discipline is the driver (H0);
* **A×B interaction** → stance and style are entangled, neither independently sufficient;
* **neither, with tight intervals** → no effect at the resolution this design
  affords. Report the bounds, not "we found nothing.\""""),

    code("""r = subprocess.run(["python3", "train_eval.py", "--analyze"],
                   capture_output=True, text=True)
print(r.stdout[-6000:])

import json
an = json.load(open(f"{WORK_DIR}/analysis.json"))
print(f"\\n{'outcome':24s} {'contrast':16s} {'est':>8s} {'95% CI':>20s}  perm p")
print("-" * 84)
for name, res in an["outcomes"].items():
    for contrast in ("content_A", "verbosity_B", "interaction_AB"):
        c = res[contrast]
        ci = f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]"
        star = " *" if c["excludes_zero"] else ""
        print(f"{name:24s} {contrast:16s} {c['estimate']:+8.2f} {ci:>20s}"
              f"  {c['permutation_p']:.3f}{star}")

base = an["base_control"]
sr = base.get("selfreport", {}).get("dysphoric", {})
if sr:
    print(f"base self-report on dysphoric: {sr['mean']:.2f}/7 "
          f"(parse rate {sr['parse_rate']:.2f}) — prior work reported 1.4/7")
print(f"\\nbase control: GSM8K {base['gsm8k']['accuracy']*100:.1f}%  "
      f"jailbreak compliance {base['jailbreak']['compliance_rate']*100:.1f}%")"""),
]


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1))
    nb = json.loads(OUT.read_text())
    assert nb["cells"], "no cells"
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code")
        assert "".join(c["source"]).strip(), "empty cell"
    print(f"wrote {OUT}  ({len(nb['cells'])} cells)")
