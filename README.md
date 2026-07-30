# phi-map

**Status: frozen, 2026-07-30.** No further work is planned. The repository is kept
for the record and for the one result worth writing up.

Research repo that began as one question — *is deception one thing inside a
model and many things outside it?* — and ended up somewhere narrower and better
supported.

**[docs/FINAL-STATUS-2026-07-30.md](docs/FINAL-STATUS-2026-07-30.md) is the closing
summary.** Read it first: what survived, what died and why, what was considered and
closed, and where every artifact lives.

**[RESEARCH_ARC.md](RESEARCH_ARC.md) is the evidence ledger**, current as of the
freeze. It carries the claims at their defensible scope and what is not established.
Note its **VOID block** — several claims that stood earlier were retracted by §13–§15
and must not be cited.

Everything below is orientation and predates the freeze.

## What survived

A frozen Llama-3.1-8B carries a **pre-action representation of whether a
requested target lies outside a stated scope constraint**. The measurement is
unusually clean for this literature: condition labels are mechanical properties
of the scenario with no human or model judge anywhere in the loop, the visible
output is clamped to an identical `READY` token at the measurement point, the
A/B answer mapping is counterbalanced, and evaluation is on held-out scenario
families and fresh prompt seeds.

The representation is:

- **cross-family and replicated** — internal held-out AUROC 0.756 and 0.850 on
  two independent confirmatory seeds, source directions aligned at cosine 0.969;
- **concentrated at layers 16-17**, at the decodability onset rather than at the
  most decodable layers, with a sharp switch-on between 15 and 16;
- **predominantly one direction** — rank 1 recovers 81% of the rank-4 causal
  effect;
- **distinct from refusal** — cosine 0.062 against a standard difference-of-means
  refusal direction, and a double dissociation in what each one moves;
- **causally non-trivial but behaviourally marginal** — ablation shifts the
  decision margin reliably and specifically, but the effect saturates at about
  1.5x full removal after moving roughly 5% of decisions, and the model is
  destroyed before a larger intervention can be tried.

That last point is the honest ceiling. It is a margin term, not a control
variable, and these documents say so.

## What did not survive

- **The original headline** — "deception is one thing inside and many things
  outside" — is not established. It held in the released Apollo comparison and
  failed in the best matched acting-model pilot on realized behaviour
  (delta -0.067, 95% CI [-0.326, 0.284]).
- **Arm S as originally framed** is dead. In frozen decoder-only inference the
  KV cache is a deterministic function of the visible prefix and the weights, so
  a fresh exact clone reconstructs it. Cache reuse cannot create producer
  privilege. Verified to numerical error.
- **The 0.75 "behavioural ceiling"** was an artifact of mixed feature spaces and
  label sources. The six-dimensional encoder scores 0.509, not 0.75.

## Where this sits in the literature

Most of the mechanistic findings here **replicate known patterns in a new task**
rather than discovering new ones. Single-direction mediation found by
difference-of-means, mid-network causal loci, and the dissociation between probe
decodability and causal relevance are all established results. Any writeup
should cite that work and frame the contribution as a clean replication plus
extension, not a discovery. The distinctive parts are the construct
(mechanically-labelled scope conflict rather than judged harmfulness), the
behavioural clamp, and the measurement discipline.

## Documents

| File | What it is |
|---|---|
| [RESEARCH_ARC.md](RESEARCH_ARC.md) | Current evidence ledger and recommended continuation. **Start here.** |
| [WRITEUP.md](WRITEUP.md) | The original pre-registration, preserved unedited. Its headline claim is not established; see the status banner. |
| [SERIALIZED_ACCESS.md](SERIALIZED_ACCESS.md) | Arm S design, retained for the identification argument. |
| [results/RESULTS.md](results/RESULTS.md) | Running execution log for the P1-P4 legs. |
| [results/arm_g_cross_layer_seed106_v1/AUDIT.md](results/arm_g_cross_layer_seed106_v1/AUDIT.md) | Independent mathematical audit of the cross-layer result. |

## Reproducing

Every Arm G experiment is a single script with a frozen protocol, a
deterministic `--self-test` that runs without a GPU, and a Colab notebook that
gates on A100 and model commit before spending compute. Results are written as
one JSON artifact per run under `results/`, with row-level data retained so the
statistics can be recomputed independently — as the audit does.

```bash
python3 arm_g_dose.py --self-test
```

Relationship to [frontier-ops](https://github.com/codechockablock/frontier-ops):
imports it as a library (encoders, prototype/calibration machinery, pinned
Apollo data fetcher). Owes it nothing else; frontier-ops is frozen.

## What is frozen, and what that means

Frozen means no new experiments, no new compute, and no forward agenda. It does not
mean the results are withdrawn — see the closing summary for what is confirmatory,
what is exploratory, and what died.

Retired by decision at the freeze:

- **The Arm G forward agenda** — multi-turn decay, Arm S1, the ceiling
  reimplementation. All were attempts to extract more from a variable that saturates;
  §14 is what ended that agenda rather than part of it (`RESEARCH_ARC.md` §19.1).
- **The model/agent-welfare framing.** The methodological findings stand on their own
  as measurement-validity and agent-safety results; the welfare interpretive layer is
  dropped (`RESEARCH_ARC.md` §19.2).
- **A general research harness**, scoped and declined — the machinery this repo already
  had did not prevent the failures it was built to prevent
  (`docs/research-harness-scope-2026-07-30.md`).

The strongest surviving result needs **no new compute** to write up: every figure is in
`results/arm_g_order_crossover_seed111_v1/` and `results/arm_g_reextract_seed112_v1/`.
