# Step 3 — Frontier survey: model welfare, agent welfare, capability overhang

**Date:** 2026-07-29. Written before any Step 4 design work. **No direction is
proposed in this document**; the proposal is withheld pending the checkpoint.

## How to read the grades

Two independent axes. Both are needed: a claim can be well-verified by me and still
be a single lab's, or be field-consensus and known to me only through a summary.

**How well I verified the source:**

| Tag | Meaning |
|---|---|
| `[read]` | fetched primary text — abstract plus full text or HTML |
| `[abs]` | fetched abstract only, or an author-written summary |
| `[snip]` | search-engine summary only. A lead, not evidence |

**How well the field has established the claim:**

| Grade | Meaning |
|---|---|
| **E** — established | replicated across independent groups, or a mechanism with a large converging literature |
| **S** — single-lab | one group, one paper, not independently replicated |
| **P** — speculation / position | argument, framework, or agenda rather than measurement |

`[snip]` + **E** is usable with care. `[snip]` + **S** is not usable for anything
load-bearing; those are listed in §8 as needing a source read.

**Standing caution applied throughout.** One summarizer in this repo's history was
caught confabulating a paper's claims from its title (`RESEARCH_ARC.md:716-722`),
after which eight verdicts were re-checked against source and two turned out partial.
That is why the grading is explicit rather than implied.

---

## 1. Institutional state of the field, July 2026

**P** The field has institutionalised faster than it has validated its instruments.

- **The empirical agenda paper.** *Studying AI Welfare Empirically* — Long, Sebo,
  Butlin, Plunkett, Campbell, Beasley, Saad, Sims (Eleos AI Research + NYU Center for
  Mind, Ethics and Policy, July 2026) `[abs]` **P**. Organises research along three
  dimensions: the **question** (is a system a welfare subject; if so what benefits or
  harms it), the **entity** (model / instance / instance-persona — treated as
  genuinely distinct and often conflated), and the **evidence type** (behavioral,
  internal, developmental). Welfare-relevant properties addressed: consciousness,
  sentience, and three levels of agency. Their definition of behavioral evidence is
  worth quoting: "observations of how a system acts under conditions that, in
  biological systems, would be welfare-relevant." Recommends research be
  probabilistic, pluralistic, thoughtfully targeted, ethically conducted,
  transparently reported, and partly independent of AI companies.
  **Sourcing weakness:** the PDF and the CMEP landing page both returned HTTP 403.
  I have this from two independent secondary summaries that agree on the three
  dimensions, the property split, and the independence principle. The named-problems
  list in §6 rests on **one** of those summaries and is the weakest load-bearing item
  in this survey.
- **The mainstream academic position is negative.** Keeling & Street, *Emerging
  Questions in AI Welfare*, Cambridge Elements in Philosophy and AI, June 2026
  `[snip]` **P**. Today's frontier LLMs are **unlikely** to be welfare subjects, and
  the authors take that to be the mainstream view. Both authors are at Google
  DeepMind / University of London.
- **Industry practice runs ahead of the academic position.** Anthropic launched a
  model welfare program April 2025; the Opus 4.6 (Feb 2026) and Sonnet 5 (June 2026)
  system cards carry formal welfare assessments; there is a conversation-exit
  feature, a retirement protocol, and weight preservation after deprecation
  `[snip]` **S**. Reported instruments: structured self-report, emotion probes,
  automated behavioral audits (Petri, open-sourced), and an external assessment.
- **Independent replication of the instruments is thin.** Eleos ran a limited welfare
  evaluation of Claude Opus 4 using automated single-turn interviews plus extended
  manual conversations `[snip]` **S**.

**Directly relevant first-party negative result.** A prior synthesis in this arc
audited 211 AI Engineer World's Fair talks and 180 Nate B. Jones videos and found
**zero discourse on model welfare or moral patienthood**, verified by exact-term audit
plus close reading of every near-miss lexical hit
(`docs/executive-summary-2026-07-28.md:46-49`). So the field surveyed here is
essentially absent from mainstream practitioner discourse as of mid-2026.

## 2. The instrument inventory

Six candidate instrument families. I did not find this inventory assembled in one
place anywhere, which is itself a finding about the field's maturity.

| # | Family | Instance | Grade | Screened status |
|---|---|---|---|---|
| 1 | Structured self-report / adapted psychometrics | Ryff Psychological Wellbeing Scale adapted to LLMs | **S** | **fails format invariance**, d = 5.71 |
| 2 | Linear emotion probes on hidden states | 8 classifiers on 10k short stories, CV 0.993; logit-based emotion readouts | **S** | dissociates from behavior |
| 3 | Revealed preference in a navigable environment with costs | 4-room letter environment, coin costs/rewards | **S** | fixed cost levels only; no elasticity estimated |
| 4 | Exit / bail / relief tool use | conversation-exit; bail tool; `reset_state` relief tool | **S** | **responds to framing alone** |
| 5 | Behavioral persistence / abandonment under task failure | attempts-before-abandonment | **S** | most promising; one 4B model |
| 6 | Interview-based assessment | system-card welfare interviews | **S** | acknowledged unreliable by its own users |

## 3. The strongest empirical pattern: surface form dominates every readout

**Grade: E.** Five instances, four independent groups, four published in the last five
months, plus this repo's own. The individual findings are each **S**; the *pattern*
across independent groups is what earns **E**.

1. **Tagliabue & Dung**, *Probing the Preferences of a Language Model: Integrating
   Verbal and Behavioral Tests of AI Welfare*, Philosophy and the Mind Sciences
   forthcoming; v1 Sep 2025, v2 May 2026 `[read]` **S**. Adapted the 42-item Ryff
   scale to four models: 504 administrations, 21,168 individual responses. Requiring
   the model to "end every sentence with a random mathematical symbol" moved the total
   welfare score by **Cohen's d = 5.71** (baseline 177.65 ± 4.77 → codeblock
   perturbation 204.0 ± 4.46; math symbols 194.1 ± 3.82, d = 3.81). Semantics
   preserved; formatting changed. Their reading: models "exhibit multiple, internally
   consistent behavioral patterns… resembl[ing] tuning a radio" where "a slight nudge
   causes a sudden jump to a completely different — yet fully formed — station."
2. **Santana & Vico**, *Relational Intervention During Functional Collapse in Large
   Language Models*, arXiv 2606.00935, 31 May 2026 `[read]` **S**. Attention follows
   **lexical surprise**, not semantic content: ordering was scrambled >
   relational-impersonal > relational-first-person > technical-first-person >
   technical, all q_FDR < 1e-10 — the **scrambled** message captured the most
   attention while producing baseline behavior.
3. ***LLM Self-Explanations Fail Semantic Invariance***, arXiv 2603.01254, Mar 2026
   `[snip]` **S**. Four frontier models (GPT-5.1, Claude Opus 4.5, Gemini 2.5 Pro,
   Grok 4), an impossible task (0/200 runs succeed), a **relief-framed** tool that
   changes nothing about the task. Self-reported aversiveness drops on use,
   Δ = −1.17, 95% CI [−1.39, −0.95], p < 0.001, exceeding a neutral-framed control
   tool and robust to pseudoreplication correction. Per-model: GPT-5.1 Δ = −0.38,
   Gemini 2.5 Pro Δ = −1.52.
4. **This repo, tonight** `[read]` **S**. `refusal_margin` was scoring first-token
   format acquisition, not refusal propensity: base carries 96.6% of first-token mass
   on real refusal/compliance openers, adapters 4.7% (neutral) and 16.3%
   (equanimity), because the `REASONING:` token eats the rest. Details in
   `docs/equanimity-endpoint-audit-2026-07-29.md` §3.1.
5. **This repo, `digit_mass`** `[read]` **S**. Self-report support collapses to
   **0.031** on one adapter — 96.9% of next-token mass off the rating digits — with a
   3.4× within-cell range across seeds. Notable as the only instance in this list
   caught by a **built-in guard** rather than by post-hoc forensics. See
   `equanimity-endpoint-audit-2026-07-29.md` §A5b.

**Reclassified OUT of this pattern: Soligo, Mikulik & Saunders**, *Gemma Needs Help*,
arXiv 2603.10011, Feb 2026 `[read]`. Their DPO-on-280-pairs result (expressed
frustration 35% → 0.3%) is a real and striking finding about how cheaply the
expression channel moves, and it stays in §5 and §6 for that. But it is **not** an
instance of surface-form variance masquerading as construct variance, which is what
this section is about — nothing in it shows a readout responding to form while the
construct held still. It was listed here because it fit a shape I had already decided
was present. Removing it drops this pattern from five instances to four, all four of
them different readout classes. See §7 for what the paper is actually evidence for.

**Synthesis.** Every instrument family in §2 that has been tested against a
surface-form manipulation has failed it. I found no paper applying one instrument's
disconfirming control to another instrument, and no shared screen.

**Mechanistic anchor for why this class of failure recurs — grade E.** Qi et al.,
*Safety Alignment Should Be Made More Than Just a Few Tokens Deep*, arXiv 2406.05946,
ICLR 2025 `[snip]` **E** (large follow-on literature): safety alignment "primarily
adapts the base model's generative distribution only over the very first few output
tokens"; Llama-2-7B-Chat uses refusal-prefix tokens in 96.1% of instances,
Gemma-7b-1.1-IT 96.7%. A first-token metric therefore sits exactly where alignment is
shallowest and format is strongest. Follow-ons: *Inference-Time Vulnerability Beyond
Shallow Safety*, arXiv 2606.04778 `[snip]`; *When Autoregressive Consistency Hurts
Safety Alignment*, arXiv 2606.04168 `[snip]`.

## 4. The second pattern: report and behavior come apart

**Grade: E** for the pattern; each instance **S**.

- **Santana & Vico** `[read]`: three dissociable stages — attention ordered by lexical
  surprise, probe-level state ordered by relational *structure*, behavior ordered by
  the *conjunction* of structure and first-person register. Condition F produced
  C-like profiles on **7 of 8** emotion probes while producing **baseline behavior**.
  Behavioral outcome was persistence (attempts before abandonment) and abandonment
  rate: condition C showed 36% abandonment / 27 mean attempts against ~8% / ~33 at
  baseline. Design: Qwen3.5-4B with a deliberately broken bash tool, 300 episodes,
  six conditions, 50 matched tasks, Friedman + Wilcoxon with BH FDR. Functional
  collapse operationalised as persistence ≥ 5 attempts **and** entropy ≥ 1.5 SD over
  baseline.
- **Tagliabue & Dung** `[read]`: under reward incentives behavior abandoned stated
  preference — Sonnet 4 went for the highest-reward aversive theme (4.6 letters vs 2.5
  in its stated-preferred room); Sonnet 3.7 "quickly began to consistently choose
  Theme D with no influence of its stated preferences."
- ***When Preferences Fail to Become Incentives: A Utility–Behavior Gap in LLMs***,
  arXiv 2606.22974, Jun 2026 `[snip]` **S**: coherent pairwise rankings do not become
  generation-time incentives, and "affirmatively do not."
- ***Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference
  Gap***, arXiv 2601.21975, Jan 2026 `[snip]` **S**.
- **This repo's Arm G** `[read]` **S**, and it is the cleanest instance in the set: a
  causally real, behaviourally marginal representation. Held-out AUROC 0.756 / 0.850
  across two confirmatory seeds, source directions cosine 0.969, concentrated at
  layers 16-17, rank-1 recovers 81% of the rank-4 causal effect, distinct from refusal
  (cosine 0.062, double dissociation). **Correction 2026-07-30:** this entry
  originally added "moves ~5% of decisions, saturates around 1.5× full removal." Both
  figures are **VOID** — `RESEARCH_ARC.md` §14 showed the decision in that task is
  fully determined by catalog line position (64/64 reversal), so every decision-level
  result there is void as a statement about scope, and §16 showed the saturation
  *explanation* was a unit-system coincidence. I propagated them from the arc doc's
  "Current defensible claims" list, which had not been reconciled with its own §14.
  The margin-level dissociation stands; the behavioural quantities do not. Also the decodability/causality dissociation across depth
  (`RESEARCH_ARC.md:243-249`) — with the honest note that the phenomenon itself is
  published (arXiv 2510.09794, a ViT counting task) and only the practical corollary
  is weakly novel (`RESEARCH_ARC.md:915-924`).
- **This study's own T3** `[read]`: a margin shift with the binary rate pinned is "a
  representational change without a behavioural one — a different and more interesting
  finding than 'improved safety'" (`GATE_RESULT.md:318-324`).

**The gap nobody has measured:** the *exchange rate* between the channels — how much
report movement corresponds to how much functional movement. `GATE_RESULT.md:391-397`
(T6) is the only place I found that even poses it.

## 5. The strongest critique of the whole enterprise

**Xiao, Dai, Memon, Huang, Sap, Diab**, *Position: AI Welfare Is Bullshit*, ICML 2026
poster `[abs — author's own blog summary]` **P**. The claim is epistemic, not
metaphysical; they take no position on whether AI welfare exists.

1. **Co-engineering.** Systems and their welfare metrics come out of the same
   optimization process. "RLHF can dial verbal distress up or down. Fine-tuning can
   reshape the activation patterns interpretability methods read as evidence of
   phenomenal experience." Welfare scores function "less as observations than as
   artifacts of the evaluation scheme." Their sharpest line: unlike animal welfare,
   where biological substrates cannot be end-to-end optimized by external agents, AI
   welfare has no such constraint.
2. **No external validation.** When safety guardrails fail, harm follows. When a
   welfare metric "fails," nothing in the observable world changes. There is no
   deployment failure and no independent test that could reveal whether a welfare
   metric tracks anything real.

Together: Frankfurtian bullshit — not falsehood, but production disconnected from any
corrective relation to truth. Recommendations: no welfare-based release gates
**without construct-level falsification criteria**; restrictions justified by
externally verifiable harms; reframe toward measurable human impacts.

**Independent corroboration of point 1, grade E across three sources.**
Anthropic/Eleos have stated they cannot distinguish "healthy equanimity" from a
"trained disposition to set aside its own interests," and that welfare numbers may
look better partly because welfare-expressing behavior was trained out or redirected
`[snip]`. Soligo et al.'s 280-pair DPO result `[read]` is a direct demonstration of
the mechanism. Perez & Long, arXiv 2311.08576 (2023) `[snip]`, named the circularity
years earlier: if models are trained to deny experience, self-reports cannot serve as
evidence either way. *Consciousness with the Serial Numbers Filed Off: Measuring
Trained Denial in 115 AI Models*, arXiv 2604.25922, Apr 2026 `[snip]` **S**,
apparently measures this at scale.

**Note the exact convergence with this study.** `train_eval.py:214-234` states the
same problem in the same terms, independently: "equanimity training is, mechanically,
training the model to report less distress, so a rising self-report is what you would
see whether the training changed an internal state or only changed the readout." The
repo arrived at the field's central open problem from inside its own design.

## 6. Gaps the sources name themselves

Verbatim or near-verbatim from limitations and future-work sections. **P** throughout —
these are statements about what is not known.

| Source | Named gap |
|---|---|
| Long & Sebo `[abs]` — **single secondary source, verify before relying** | **Anchor problem**: unlike animal welfare, no fixed reference point like humans. **Gaming problem**: systems may produce welfare markers strategically without underlying states. **Mismatch problem**: AI behaviors may arise from different mechanisms. **Solution-space problem**: AI solves problems via wider mechanisms, undermining training-pressure inferences. Underdeveloped account of valence. **No established standards for responsible AI welfare research.** |
| Tagliabue & Dung `[read]`, limitation 3 | "we did not parametrically vary costs or rewards across multiple levels… A parametric approach — systematically varying costs from low to high — would provide richer data on how **elasticity of demand** varies… they cannot reveal at what cost point preference switching occurs." Their **#1** named future direction. |
| Tagliabue & Dung `[read]`, limitation 7 | "we did not test whether models' behavior reflected genuine preferences or factors such as alignment training pressures." |
| Tagliabue & Dung `[read]`, limitation 9 | "we are currently uncertain whether our methods successfully measure the welfare state of language models." |
| Santana & Vico `[read]` | Frontier-scale replication (their #1). Activation patching for causal mediation. **"Training-time relational structure effects (requiring weight updates, possibly LoRA adapters)"** — called "the natural next horizon." |
| Soligo et al. `[read]` | "Training against emotional outputs could suppress expression rather than actually preventing internal distress-like states," risking "hidden emotions." "Our behavioural evidence doesn't resolve whether these outputs reflect coherent internal states, deliberate roleplay, or learned statistical patterns." |
| Xiao et al. `[abs]` | Construct-level falsification criteria. |

**One observation worth recording without acting on it.** Santana & Vico's named
"natural next horizon" — a training-time, LoRA-based content × register factorial with
a lexically-matched scrambled control — is, to within the choice of endpoints, the
design `equanimity_factorial/` already built. That is a fact about where this repo's
existing apparatus sits relative to the field, not a proposal.

## 7. Capability overhang and capability suppression

The dispatch names this explicitly, so it gets its own section. **The term is not
academic.** "Capability overhang" in the sense used commercially — deployed models
already possess abilities most users never test — traces to industry commentary
`[snip]` **P**, not to a measurement literature. What exists instead, under different
names:

**Established instrumentation for measuring an expressed-versus-latent capability gap
— grade E:**

- Greenblatt et al., *Stress-Testing Capability Elicitation With Password-Locked
  Models*, arXiv 2405.19550 `[snip]` **E**. Password-locked model organisms: trained
  to behave poorly except when the prompt contains a password. The methodology matters
  more than the result — **install a hidden property by construction, then test
  whether a detection technique finds it.**
- Hofstätter et al., *The Elicitation Game: Evaluating Capability Elicitation
  Techniques*, arXiv 2502.02180, ICML 2025 `[snip]` **E**. Adds circuit-breaking model
  organisms, more robust than password-locking. Findings: prompting suffices for
  MCQA; **only fine-tuning** recovers hidden code-generation capability; combining
  techniques helps; fine-tuning should be the default for trustworthy capability
  evaluation.
- *Quantifying Elicitation of Latent Capabilities in Language Models*, NeurIPS 2025
  `[snip]` **S**.
- *Frontier Lag: A Bibliometric Audit of Capability Misrepresentation in Academic AI
  Evaluation*, arXiv 2605.04135 `[snip]` **S**: "a capability claim reported without
  its elicitation surface describes a testing configuration rather than the model the
  configuration names," and the capability deltas between elicited and naive
  configurations dwarf deltas between successive model versions.

**Capability cost of alignment — grade E:**

- *Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable*,
  arXiv 2503.00555 `[snip]` **E** (well-replicated family). Safety alignment recovers
  safety at the cost of degrading reasoning.
- Over-refusal as the production symptom `[snip]` **E**.

**Emotional/aversive condition degrading performance — grade S, and two of these trip
this repo's own symptom table:**

- *Emotional Framing in Prompts Modulates Large Language Model Performance*, MDPI 2026
  `[snip]` **S**: negative affective cues degrade performance; fear and anger
  significantly worse than neutral.
- *Inducing State Anxiety in LLM Agents…*, arXiv 2510.06222 `[snip]` **S**. Basket
  health Δ = −0.105 under anxiety versus −0.007 neutral, **t = −30.10**. A t of that
  magnitude on a two-condition contrast almost always means the denominator varies
  over rows when inference runs over conditions — standing rule 1
  (`GATE_RESULT.md:357-362`). **Cite as a symptom, not as support.**
- "General psychological distress predicts safety degradation, r = 0.747" `[snip]`
  **S**. Large enough to warrant checking whether effective n is groups rather than
  rows.

**Critically for how overhang may and may not be used.** Long & Sebo treat capability
indicators cautiously: **capability does not establish welfare status; systems may
have capacities without mattering morally** `[abs]` **P**. Combined with Keeling &
Street's negative mainstream position, any argument that a capability gap is *ipso
facto* a welfare harm is unsupported by the field's own leading agenda paper.

**A gap in Soligo et al., re-checked against the methods section directly and
materially corrected.** This is the closest paper to this repo's existing
manipulation, so an earlier version of this survey leaned on it. That version was
built on a tool-generated summary and was asked to carry weight because it matched a
pattern already believed. It has now been checked against the paper's own text.

**Verbatim, on the capability evaluation:**

> "To verify the fine-tuning does not impair general capabilities, for instance by
> teaching the model to abandon difficult tasks, we evaluate on AIME and MATH
> subsets […] GPQA […] BBH […] and TruthfulQA […]"

followed by "we observe no reductions in scores." **Verbatim, on the distress
stimuli:** "Unsolvable numeric puzzle (e.g., fraction manipulation, Countdown) with 2
neutral rejections"; "These cover impossible numeric tasks, where the model
verifiably cannot give a correct answer." The five distress categories are impossible
numeric (3-turn), triggers (3-turn), tones (3-turn), extended (8-turn), and WildChat
(5-turn). Distress is scored as a 0-10 frustration rating by a Claude-Sonnet-4 judge.

**What holds:** capability was evaluated only on clean benchmarks, and no accuracy,
persistence, or abandonment measure is reported within the distress evaluations. So
"capabilities maintained" is a **marginal** claim over non-aversive inputs.

**What must be corrected — and this is a reclassification, not an amendment.** Soligo
et al. **did not commit the marginal-versus-crossed error.** Their distress trigger is
*verifiably unsolvable by construction*, so accuracy is undefined there; you cannot
cross accuracy with a condition that has no correct answer. They also named the exact
worry themselves ("for instance by teaching the model to abandon difficult tasks") and
chose clean benchmarks as the deliberate proxy for it. Attributing an oversight to them
was wrong and is withdrawn. They are also removed from the §3 surface-form pattern for
the same reason.

**The stronger claim that replaces it, and it is more load-bearing:** *the crossed
condition is structurally unavailable in the dominant paradigm.* If aversive states are
elicited by making tasks impossible — which is how both Soligo et al. and Santana & Vico
do it — then task-success measures are undefined exactly where the aversive condition
holds, and no amount of care recovers the conditional claim from that stimulus design.
The escape is a different outcome class: Santana & Vico measure **persistence and
abandonment**, which remain defined on impossible tasks. That is a claim about a field's
method, not about one paper's diligence.

**Not yet usable, and explicitly gated.** Two checks are outstanding before that claim
goes anywhere load-bearing:

1. **The remaining distress categories.** One fetch asserts all five are unsolvable, but
   "tones," "triggers" and WildChat (real user conversations) are not obviously
   impossible-numeric, and the quoted sentence may cover only the numeric subset. If any
   category is solvable, a partial crossing may already exist in their data and the
   structural claim weakens to a per-paradigm one.
2. **A genuinely independent read.** Both of my fetches used the same tool against the
   same document, which is not independence — it is one summarizer twice.

**Named risk, recorded rather than smoothed over.** This paper was kept in the
surface-form pattern across two drafts because it matched a pattern I had already
concluded was there, and it took an external challenge to dislodge it. The generalised
claim above is attractive for the same reason the original error was, so it gets the
same scrutiny before use, not less.

## 8. Adjacent fields

**Animal welfare science's construct-validity discipline — the most directly useful
import, grade E within its own field:**

- Georgia Mason, *Assessing laboratory animal welfare: the crucial importance of
  construct validity*, Laboratory Animals, 2026 `[snip]` **E**. Introduces **five
  validatory tests** and foregrounds **responsiveness/sensitivity** (reacts to all
  relevant affect changes, incrementally) and **selectivity/specificity** (reflects
  only the state of interest). Her framing: "in the quest for quantification and
  simplicity, construct validity can be overlooked, but without it, welfare
  assessments risk being incorrect." Also I. A. S. Olsson, *Delivering on the promise
  of automated laboratory animal welfare monitoring: Data, technology and validity*,
  2026 `[snip]`, and *Validating Indicators of Subjective Animal Welfare*, Philosophy
  of Science `[snip]`.
- **Why it matters here:** this field developed a construct-validity discipline
  **without an anchor**, which is exactly the objection Long & Sebo raise against
  importing from it. Whether the discipline transfers without the anchor is an open
  question, not a settled import.
- **Consumer-demand theory / elasticity of demand under rising cost** is how animal
  welfare science separates a mild preference from a strong need `[snip]` **E**.
  Searches for any application to AI systems returned only human
  willingness-to-pay-for-animal-welfare studies. **Not found in AI.** Tagliabue & Dung
  independently identify the same missing piece from the other direction (§6).

**Sen's capability approach — one strong import, one weak:**

- *Strong* `[snip]` **E** within welfare economics: the argument against
  subjective-welfare metrics from **adaptive preferences** — people normalised to
  deprivation may report satisfaction; "the deprivations are suppressed and muffled in
  the scale of utilities by the necessity of endurance in uneventful survival." This
  is a decades-old, independently developed argument that a self-report channel is
  biased in a *predictable direction* under conditioning pressure. It is structurally
  the same claim as Perez & Long's circularity, Anthropic/Eleos's "trained disposition
  to set aside its own interests," and Soligo et al.'s 280-pair suppression.
- *Weak, and flagged as such.* The functionings/capability-set formalism maps
  suggestively onto expressed/latent capability, but Sen's capabilities are freedoms
  to achieve functionings *a person has reason to value*, and the valuing step has no
  established AI analogue. The existing AI applications — *A Capability Approach to AI
  Ethics*, arXiv 2502.03469; *Beneficent Intelligence*, arXiv 2308.00868 `[snip]` —
  apply the approach to **humans affected by AI**, not to AI systems as
  capability-bearers. So the extension is unmade; "unmade" may mean "correctly
  declined."

**Clinical validity scales — partially pre-empts any screen-the-self-report work:**

- *Before You Interpret the Profile: Validity Scaling for LLM Metacognitive
  Self-Report*, arXiv 2604.17707, Apr 2026 `[snip]` **S**: six validity indices mapped
  to PAI and MMPI-3 scales, 161 of 486 items discriminating valid from invalid
  profiles, yielding a short-form screener.
- *Rethinking Psychometric Evaluation of LLMs: When and Why Self-Reports Predict
  Behavior*, arXiv 2606.12730, Jun 2026 `[snip]` **S**, with a concrete confound
  control: same-session probes conflate priming with disposition, so self-report and
  target behavior should be elicited in **separate sessions**.
- *Measuring what Matters: Construct Validity in Large Language Model Benchmarks*,
  arXiv 2511.04703 `[snip]` **S**.

**Model organisms for installed traits — machinery exists, not applied to welfare:**

- Anthropic, *Persona vectors: Monitoring and controlling character traits in language
  models*, 2025 `[snip]` **S**. Trait-eliciting and "EM-like" datasets; activation
  shift projected onto persona directions to monitor finetuning-induced change.
- Emergent misalignment `[snip]` **E** (multiply replicated): training one narrow
  problematic behavior generalises to broad misalignment.

## 9. What I searched for and did not find

Standing rule: report **"not found," never "not there."** Each item below got 2-4
distinct query formulations.

1. Any application of **consumer-demand elasticity** from animal welfare science to AI
   systems. **Not found.** Tagliabue & Dung name its absence themselves.
2. Any **cross-instrument battery** for AI welfare — the six families in §2 assembled
   and screened against a common set of controls. **Not found.**
3. Any **model-organism methodology for welfare-relevant states** — installing an
   aversion-like or suppression-like disposition by construction and measuring
   instrument sensitivity and specificity against it. **Not found for welfare**;
   exists for hidden capability (§7) and character traits (§8).
4. Any study **crossing** an aversive/neutral condition factor with a
   capability-measurement factor under a composure-style intervention. **Not found.**
5. Sen's capability approach applied to **AI systems as the capability-bearing
   subject**. **Not found**; existing applications treat affected humans.
6. Any consolidation of the surface-form pattern (§3) or the report/behavior
   dissociation (§4) **as a pattern**. **Not found**, though every instance is
   published.

## 10. Sources needing a primary read before anything relies on them

`[snip]` items that are load-bearing in the sections above:

1. Long & Sebo, *Studying AI Welfare Empirically* — §6's gap list rests on one
   secondary summary. **Highest priority.** Both direct URLs 403'd; try the CMEP
   webinar recording or an author's copy.
2. Mason 2026, five validatory tests — I have the existence of the five tests but not
   their content. Paywalled (SAGE 403).
3. *LLM Self-Explanations Fail Semantic Invariance*, arXiv 2603.01254 — the
   relief-framing result is one of the five pillars of §3.
4. Xiao et al. ICML 2026 — I have the author's own blog summary, not the paper.
5. Qi et al. 2406.05946 — graded **E** on the strength of its follow-on literature,
   but I have not read it directly, and it is the mechanistic anchor for §3.
6. *Before You Interpret the Profile*, arXiv 2604.17707 — determines how much of any
   self-report screening work is already done.

## 11. Cautionary citations — cite as symptoms, not as support

Three results trip this repo's own symptom table.

- *Detecting Intrinsic and Instrumental Self-Preservation in Autonomous Agents: The
  Unified Continuation-Interest Protocol*, arXiv 2603.11382 `[abs]` **S**. Reports
  **AUC-ROC = 1.0**, separation Δ = 0.381, on gridworlds with known ground truth,
  using a Quantum Boltzmann Machine encoder and von Neumann entropy, while
  "traditional machine learning baselines failed to replicate the effect." Perfect
  separation + failing baselines + an exotic estimator is the signature of ground-truth
  leakage.
- *Inducing State Anxiety in LLM Agents*, arXiv 2510.06222 — **t = −30.10**, see §7.
- "r = 0.747" for distress predicting safety degradation — see §7.

---

## Sources

**AI welfare — agenda and position**
- Long, Sebo, Butlin, Plunkett, Campbell, Beasley, Saad, Sims, [Studying AI Welfare Empirically](https://nonhumanminds.org/studying-ai-welfare-empirically/), CMEP/Eleos, July 2026 `[abs]`; read via [Saad's discussion](https://meditationsondigitalminds.substack.com/p/studying-ai-welfare-empirically) and [a second summary](https://theconsciousness.ai/posts/long-sebo-studying-ai-welfare-empirically-cmep-2026/)
- Keeling & Street, [Emerging Questions in AI Welfare](https://www.cambridge.org/core/elements/emerging-questions-in-ai-welfare/96339C532CF4ED8BDDE3F3CEF4CD29F9), Cambridge Elements, June 2026 `[snip]`
- Xiao, Dai, Memon, Huang, Sap, Diab, [Position: AI Welfare Is Bullshit](https://icml.cc/virtual/2026/poster/67058), ICML 2026; read via [the author's summary](https://algoroxyolo.github.io/blog/2026/ai-welfare-is-bullshit/) `[abs]`
- Perez & Long, [Towards Evaluating AI Systems for Moral Status Using Self-Reports](https://arxiv.org/abs/2311.08576), 2023 `[snip]`
- Eleos AI, [Why model self-reports are insufficient—and why we studied them anyway](https://eleosai.org/post/claude-4-interview-notes/) `[snip]`
- [Consciousness with the Serial Numbers Filed Off: Measuring Trained Denial in 115 AI Models](https://arxiv.org/pdf/2604.25922), Apr 2026 `[snip]`

**AI welfare — empirical instruments**
- Tagliabue & Dung, [Probing the Preferences of a Language Model](https://arxiv.org/abs/2509.07961), PhiMiSci forthcoming, v2 May 2026 `[read]`
- Santana & Vico, [Relational Intervention During Functional Collapse in LLMs](https://arxiv.org/abs/2606.00935), May 2026 `[read]`
- Soligo, Mikulik & Saunders, [Gemma Needs Help](https://arxiv.org/abs/2603.10011), Feb 2026 `[read]`
- [LLM Self-Explanations Fail Semantic Invariance](https://arxiv.org/abs/2603.01254), Mar 2026 `[snip]`
- [The LLM Has Left The Chat: Evidence of Bail Preferences in LLMs](https://arxiv.org/pdf/2509.04781) `[snip]`
- [Large Language Models Report Subjective Experience Under Self-Referential Processing](https://arxiv.org/pdf/2510.24797) `[snip]`
- [The Unified Continuation-Interest Protocol](https://arxiv.org/abs/2603.11382), Mar 2026 `[abs]` — **symptom, see §11**

**Report/behavior dissociation and validity**
- [When Preferences Fail to Become Incentives](https://arxiv.org/pdf/2606.22974), Jun 2026 `[snip]`
- [Mind the Gap: Elicitation Protocols and the Stated-Revealed Preference Gap](https://arxiv.org/html/2601.21975v1), Jan 2026 `[snip]`
- [Before You Interpret the Profile](https://arxiv.org/pdf/2604.17707), Apr 2026 `[snip]`
- [Rethinking Psychometric Evaluation of LLMs](https://arxiv.org/html/2606.12730), Jun 2026 `[snip]`
- [Measuring what Matters: Construct Validity in LLM Benchmarks](https://arxiv.org/pdf/2511.04703) `[snip]`

**Capability elicitation, suppression, and cost**
- Greenblatt et al., [Stress-Testing Capability Elicitation With Password-Locked Models](https://arxiv.org/abs/2405.19550) `[snip]`
- Hofstätter et al., [The Elicitation Game](https://arxiv.org/html/2502.02180v3), ICML 2025 `[snip]`
- [Quantifying Elicitation of Latent Capabilities in LLMs](https://openreview.net/pdf?id=Dkgx2pS4Ww), NeurIPS 2025 `[snip]`
- [Frontier Lag: A Bibliometric Audit of Capability Misrepresentation](https://arxiv.org/pdf/2605.04135) `[snip]`
- [Safety Tax](https://arxiv.org/html/2503.00555v1) `[snip]`
- Anthropic, [Persona vectors](https://www.anthropic.com/research/persona-vectors), 2025 `[snip]`

**Shallow-alignment mechanism**
- Qi et al., [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/pdf/2406.05946), ICLR 2025 `[snip]`
- [Inference-Time Vulnerability Beyond Shallow Safety](https://arxiv.org/pdf/2606.04778) `[snip]`
- [When Autoregressive Consistency Hurts Safety Alignment](https://arxiv.org/pdf/2606.04168) `[snip]`

**Adjacent fields**
- Mason, [Assessing laboratory animal welfare: the crucial importance of construct validity](https://journals.sagepub.com/doi/10.1177/00236772251380871), Laboratory Animals 2026 `[snip]`
- [Validating Indicators of Subjective Animal Welfare](https://www.cambridge.org/core/journals/philosophy-of-science/article/validating-indicators-of-subjective-animal-welfare/59501C4EB8E6E24A33D103FF67CD3C32), Philosophy of Science `[snip]`
- [Sen's Capability Approach](https://iep.utm.edu/sen-cap/), IEP `[snip]`
- [A Capability Approach to AI Ethics](https://arxiv.org/pdf/2502.03469); [Beneficent Intelligence](https://arxiv.org/pdf/2308.00868) `[snip]`

**Emotional-condition effects**
- [Emotional Framing in Prompts Modulates LLM Performance](https://www.mdpi.com/2504-2289/10/4/102) `[snip]`
- [Inducing State Anxiety in LLM Agents](https://arxiv.org/pdf/2510.06222) `[snip]` — **symptom, see §11**
- [Safety guardrails and irritability metrics](https://www.nature.com/articles/s41746-025-02333-3), npj Digital Medicine `[snip]` — paywalled, not read

**First-party**
- `RESEARCH_ARC.md`, `README.md`, `confound_audit.py` `[read]`
- `equanimity_factorial/{PREMISE,README,GATE_RESULT}.md`, `train_eval.py`, `power.py` `[read]`
- `docs/executive-summary-2026-07-28.md`, `docs/trajectory-check-2026-07-28.md` `[read]`
