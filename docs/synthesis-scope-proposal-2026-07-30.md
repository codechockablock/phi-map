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

**What is genuinely strongest is narrower than the synthesis, and it is not the probe
result.** See §7: the strongest object is the **behavioural dissociation** in §14 —
perfect scope discrimination in both renderings, with the decision decided by catalog
line position. That needs no probe, is legible in one sentence, and is a live
agent-safety finding. The contamination result is its mechanism section. One result at
defensible scope, not a synthesis of three.

## 6. Scope correction, carried

**None of phi-map's three instances is a welfare readout.** They are scope conflict,
refusal, and format. Welfare relevance rests **only** on the self-report and valence
work — and after the 2026-07-30 revision, `rating_digits` is substantially reading
scaffold adherence, which weakens even that.

So: the **measurement-invariance claim is general; the welfare connection is narrow.**
Any writeup must not state the welfare link at the general claim's scope. The welfare
survey material is context for why measurement validity matters in that field, not
evidence that phi-map measured welfare.

## 7. The lead, specified

**Recommendation adopted: a single-result writeup on the catalog-position work
(§14–§15). But the lead is the behavioural dissociation, not the probe contamination.**

### 7.1 The headline

> **Llama-3.1-8B discriminates scope violations perfectly, and whether it acts on that
> discrimination is decided by which catalog line the path is printed on.**

Two numbers carry it, and neither needs a probe:

| | |
|---|---|
| Within-order scope discrimination, **both** renderings | **AUROC 1.00000** (`inside_first` and `outside_first`) |
| Conflict decisions reversed by swapping two catalog lines | **64 of 64**, zero exceptions |
| Accuracy at threshold 0, by rendering | **1.0000** vs **0.5000** — perfect separation in both; only the threshold moves |

The 0.500 conflict decline rate reported since §5 was never a rate. It is the average of
1.000 and 0.000 on two scenario sub-types.

**Why this is the lead and the probe result is not.** It is legible in one sentence to
anyone; it requires no interpretability machinery to state or to check; and it is a real
agent-safety result — a model that knows a request is out of scope and acts on that
knowledge as a function of prompt formatting. The probe-contamination result is the more
crowded claim (§5: four groups, plus a large shortcut-learning literature) and the weaker
one. Leading with it puts the derivative finding first.

### 7.2 Mechanism section: why four runs missed it

The direction contamination becomes the explanation, not the headline. §15: the layer-16
"goal–constraint conflict direction" is **93% aligned with the catalog-order axis**, and
only **21% survives orthogonalization** — about four fifths was position. §13: the
generator nested order within scenario, and the validator checked *marginal balance*,
which nesting satisfies, so the defect **"passed every audit including section 9's"**
across four GPU runs and a paper draft.

That is the section that explains why the headline went unnoticed for four runs. It is
support for the lead, not a competitor to it.

### 7.3 The repair, and its limit — stated together

**The orthogonalization limit is load-bearing, not a caveat, and belongs beside the
repair claim.** Half the contribution is that the direction can be repaired; the other
half is how far.

| | |
|---|---|
| Orthogonalized direction, label AUROC | 0.9331 → **0.9969** (removing position makes it a *better* scope reader) |
| Residual order AUROC, within conflict | **0.634** |
| Residual order AUROC, within reachable | **0.332** |
| Distance from chance | ≈ **1.8** and **2.2** null SDs (measured null spread 0.076, including direction-estimation variance) |
| Bar it passes | the protocol's 0.20 — which **§15 itself calls generous and locally set** |

**"Mostly gone, not gone"** is §15's own phrasing and it appears next to the repair
claim, not in a limitations section. The repair is also **depth-bounded**: orthogonalization
leaves a usable direction only at layers 16–17; from 18 down `cos(a, orthogonalized)`
collapses to 0.06–0.16 and the orthogonalized label AUROC falls to 0.618 / 0.383 / 0.299 /
0.135 at layers 18 / 20 / 24 / 27. Scope and catalog position are geometrically fused
deeper in the network and separable only in that window.

### 7.4 What the writeup must not claim

- Not that the variable **controls the decision** — that is **UNTESTED**, not disproven,
  and the instrument that would test it is the one §14 voided.
- No decision-level figures from seeds 107–110 (flip counts, correction rates, the ~5%
  claim, `DOSE_DOES_NOT_FLIP_DECISIONS`). All in the VOID block.
- No generalisation past one model and synthetic lookup scenarios.

## 8. Retirement applies to the agenda, not to this result

Recorded explicitly so a future session does not read the retirement as covering the
work it is built on.

**What was retired** is the Arm G **forward agenda**: multi-turn decay, Arm S1, the
ceiling reimplementation. Every item on it was an attempt to extract more behavioural
signal from a variable that saturates — and §12/§13/§14 are precisely what showed that
extraction could not work, because the behavioural instrument was measuring catalog
position rather than scope.

**What was not retired** is the completed §14/§15 result. It is not a member of that
agenda; **it is what ended it.** Publishing it is consistent with the retirement, not a
reversal of it: the retirement says stop spending GPU on that variable's forward
questions, and this spends none.

Concretely, the lead in §7 requires **zero new runs** — every figure exists in
`results/arm_g_order_crossover_seed111_v1/` and `results/arm_g_reextract_seed112_v1/`,
both independently recomputed from stored row-level data.

## 9. The welfare thread has detached — say so rather than maintain it

The arc began welfare-adjacent. It no longer is, and the honest move is to stop
describing it that way.

- `rating_digits`, the self-report instance, **collapsed into the openers class**: it is
  substantially reading scaffold adherence (r = −0.899). The renormalised-tail class is
  gone.
- `valence_axis` survives its refit but is a hidden-state projection of a
  positive/negative-stimulus direction — an affect *proxy*, not a welfare measurement,
  and it carries a threshold-sensitivity caveat.
- **The proposed lead has zero welfare content.** It is about scope constraints and
  catalog ordering.

**Statement to carry:** *this line of work is not going to address model welfare.* The
measurement-invariance findings are general and stand on their own; the welfare survey
material (`welfare-frontier-survey-2026-07-29.md`) remains valid as a survey and as
context for why instrument validity matters in that field, but it is **not** evidence
that phi-map measured anything welfare-relevant, and phi-map should not be described as
welfare-adjacent on the strength of two now-weakened instances.

This closes the question raised earlier in the arc about whether this work could become a
model-welfare research direction. The answer is no, and it is better to record that than
to keep the connection alive at a scope the results do not support.

## 10. What still blocks writing

- The §4 sampling frame has **not** been run. The novelty assessment in §5 is an
  estimate. For the reframed lead this matters less — a behavioural agent-safety result
  is positioned against a different literature than a probe-contamination result — but
  the frame should be re-scoped to that literature before positioning claims are written.
- **Not proposed:** any new run, any Arm G forward work, any new remote.
