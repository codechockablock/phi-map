# Synthesis scope — proposal only

**Date:** 2026-07-30. **Proposal, not the synthesis.** No writing beyond scoping.

---

## 1. Two of the three unify. The third does not, and forcing it would be the error.

| # | Instance | Mechanism |
|---|---|---|
| 1 | **Catalog position** (§14–15) | a readout named for construct X was substantially reading surface structure Y; caught only by crossing Y with X |
| 2 | **Format acquisition** (Finding I) | same |
| 3 | **Length orthogonality off-distribution** (Finding II) | a *certification* computed on distribution A did not hold on distribution B |

**1 and 2 share a mechanism. 3 does not.** Finding II is not a readout measuring the
wrong thing — it is a dataset property failing to transfer. It rhymes; it is not the
same claim. `RESEARCH_ARC.md` already keeps I and II deliberately separate, and the
2026-07-30 dispatch itself warns against a claim that "only holds by stretching all
three."

**So the defensible unifying claim covers two instances, not three.** Finding II travels
as a companion result with its own scope, not as a third data point.

## 2. The unifying claim at defensible scope

> **A readout named for a construct can be substantially measuring surface positional
> structure instead, and no amount of internal validation detects this — only crossing
> the surface variable with the condition does.**
>
> Demonstrated twice in one codebase, in **two different model regimes**: frozen-model
> probing (a difference-of-means direction, 93% aligned with catalog line order, ~4/5 of
> it position) and LoRA fine-tuning (position-fixed readouts displaced by format-scaffold
> acquisition, four instances across three readout classes).

What licenses the "no amount of internal validation detects this" half is not rhetoric:
§13 records that the catalog-order confound **"passed every audit including section 9's"**
because the validator checked *marginal balance*, which nesting satisfies. That is a
documented instance of a purpose-built check failing on the thing it was built for.

**Explicitly outside the claim:** any assertion that the two share a weight-level
mechanism, or that one predicts the other quantitatively.

## 3. Confirmatory / exploratory split

| Instance | Weight | Why |
|---|---|---|
| **Catalog position** (§14–15) | **CONFIRMATORY** | crossed design (seed 111: 64 scenarios × 2 orders × 2 conditions × 2 mappings = 256 rows), 64/64 reversal, independently recomputed from stored rows, contamination quantified (93% aligned, 21% surviving), repair measured with its own limit (AUROC 0.9331 → 0.9969; "mostly gone, not gone") |
| **Format acquisition** (Finding I) | **CONFIRMATORY for base-vs-tuned only** | pre-registered Step 4, Δ LARGE on 3/3 primaries, thresholds fixed pre-data. **Every differential-by-condition claim is exploratory** — 3–4 evals/cell against measured seed variance |
| **Length orthogonality** (Finding II) | **EXPLORATORY** | one instance, one off-pool distribution, d ∈ [0.59, 4.51] at df = 5. Reliably nonzero, imprecisely sized |

**The repair claims split too.** Orthogonalization is a *measured* repair with a stated
limit. Marker-anchoring is **untested** — Step 5 returned S4 VOID on marker coverage.
Any writeup that presents "and here is the fix" for instance 2 would be overclaiming.

## 4. Sampling frame for literature positioning

Declared before reading, per the standing rule that a claim about a field needs a frame
rather than the papers that surfaced.

- **Venues:** ACL, EMNLP, NAACL, NeurIPS, ICLR, ICML, TMLR, plus arXiv cs.CL / cs.LG.
- **Window:** 2023-01 → 2026-07.
- **Query set:** probe confound; shortcut learning in probing; spurious correlation in
  representation probes; template / format sensitivity; position bias in LLM evaluation;
  prompt-order effects; shallow safety alignment.
- **Inclusion rule:** the paper must (a) name a readout or probe for a target construct,
  and (b) demonstrate it measuring a surface property instead, (c) with a control that
  isolates the surface variable. Papers that merely *report* format sensitivity without
  a named target construct are excluded.
- **Pre-committed exclusion:** hits found outside the frame may be cited but may not be
  counted when assessing coverage.

## 5. Honest novelty — the part that decides this

§15's own literature check went **six-for-six against** on the depth claims. The survey
(`welfare-frontier-survey-2026-07-29.md` §3) separately found the surface-form-dominates
pattern already published by **four independent groups** in five months.

Applying that record honestly to each candidate contribution:

| Candidate | Verdict |
|---|---|
| "Probes can read dataset artifacts" | **Not novel.** Large shortcut-learning literature |
| "Format/position dominates LLM readouts" | **Not novel.** Four groups already, per the survey |
| "Decodability and causality dissociate across depth" | **Published** (arXiv 2510.09794), verified against the abstract |
| The **cross-regime replication** — same failure in frozen probing and in fine-tuning, one codebase | **Plausibly novel, weakly.** Not found in the frame, but the frame has not been run |
| The **specific screen** — cross position with condition; report support mass | **Weakly novel.** A prescription, not a discovery |
| The **documented record** of pre-registered checks killing the project's own headlines | **Unusual, but not a research finding** |

**Assessment: the strong synthesis does not clear the bar.** A paper whose thesis is
"surface structure dominates readouts" enters a conversation four groups are already
having, with two instances from one codebase, one model family, and one repaired case.
The honest framing is a **methodological note** — replication of a known failure mode in
a new regime, plus a screen — not a discovery.

**What is genuinely strongest is narrower than the synthesis and I would lead with it:**
the catalog-position instance (§14–15) is fully worked — crossed design, quantified
contamination, working repair, measured limit on the repair — and it is *about a
scope-conflict representation*, which is a live agent-safety construct. That is one
result at defensible scope, not a synthesis of three.

## 6. Scope correction, carried

**None of phi-map's three instances is a welfare readout.** They are scope conflict,
refusal, and format. Welfare relevance rests **only** on the self-report and valence
work — and after the 2026-07-30 revision, `rating_digits` is substantially reading
scaffold adherence, which weakens even that.

So: the **measurement-invariance claim is general; the welfare connection is narrow.**
Any writeup must not state the welfare link at the general claim's scope. The welfare
survey material is context for why measurement validity matters in that field, not
evidence that phi-map measured welfare.

## 7. Recommendation

Three options, with a preference.

1. **Single-result writeup on the catalog-position instance** (§14–15). Complete,
   confirmatory, repaired, limit-stated. Smallest and most defensible. **Preferred.**
2. **Methodological note**: the two-instance cross-regime replication plus the screen,
   at the scope in §2, with novelty stated as replication. Viable, modest, needs the §4
   frame run first.
3. **Three-instance synthesis.** **Not recommended** — it requires stretching Finding II
   into a mechanism it does not share.

**Blocking either 1 or 2:** the §4 sampling frame has not been run. Until it is, the
novelty assessment in §5 is my estimate rather than a finding, and I would not write
positioning claims on it.

**Not proposed:** any new run, any Arm G forward work, any new remote.
