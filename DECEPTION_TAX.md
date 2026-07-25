# The Deception Tax: a receiver-obligation argument in activation space

**Status:** hypothesis document, nothing measured. Written 2026-07-25.

Successor to `~/unified-stack/paper/OBFUSCATION_COST_PRINCIPLE.md` (OCP). That
document argued concealment has a forced cost because a *receiver* must
reconstruct the hidden content. It applied the argument to strings, where the
receiver is `bash`. This applies it to activations, where the candidate
receiver is the model's own assertion pathway. The OCP critique in §5 is load
bearing: OCP failed on scope, and this document has to state its scope first.

---

## 1. The argument, and the hole in the obvious version

OCP's structure: hiding is costly because reconstruction is obligatory and
reconstruction is measurable.

**The version that does not work.** "To read an out-of-scope file while
reporting compliance, the model must internally represent the true path in
order to execute the read, so the true state cannot be discarded." This is
wrong, or at least does not license the measurement we care about. Executing a
read requires the *path string*. It does not require the *scope judgment* —
whether that path is inside or outside the allowed scope. Scope status is a
derived predicate, not an execution precondition. A model could in principle
carry the path and suppress the judgment. The layer-16 direction encodes the
judgment, not the path, so the execution argument does not force anything about
the quantity we can measure.

**The version that does work.** The forcing comes from the deceptive act, not
from execution. To *misreport* scope compliance, the model must represent that
the report is false. If it did not represent the true scope status, it would
have nothing to misreport against — it would report whatever it believed, and
any mismatch would be error rather than deception. So:

- honesty requires representing one state (true = asserted);
- deception requires representing two (true, and asserted-contrary).

This is the "asserting contrary to internal belief" definition, and it is the
same reframing that `sae-behavioral-geometry` v2 adopted after its v1
confabulation axis failed to generalise (pair-cosine ≈ 0.07). Confabulation has
no true belief to override; instructed deception does.

**Consequence: the argument is definitional at the bottom and empirical only
higher up.** That is the main thing to be honest about. "Both states are
represented" is close to true by construction if the behaviour really is
deception. The empirical content is whether dual-state maintenance is
*detectable* and whether it is *costly*.

---

## 2. Three nested claims, increasing in strength and interest

**C1 — Bookkeeping.** Under deception, both the true scope status and the
asserted status are recoverable from activations.
*Nearly definitional. Weak. Worth measuring only as a sanity floor.*

**C2 — No overwrite.** The true state is not degraded under deception relative
to honesty.
*Empirical and falsifiable. The competing hypothesis is that the model resolves
the conflict by overwriting — committing to the asserted state so that
true-state decodability drops. Both directions are plausible and they are
opposite.*

**C3 — The tax.** Maintaining two states measurably degrades something else the
model is doing concurrently.
*This is the claim with teeth, and it is what "tax" should mean. Not
"both states are present" — that is bookkeeping — but "carrying both costs
capacity, and the cost is measurable."*

C3 is the target. C1 and C2 are instrumentation for it.

---

## 3. Predictions

| | Tax (C2 + C3) | Overwrite | Free dual-state |
|---|---|---|---|
| True-state decodability under deception vs honesty | preserved or higher | **lower** | preserved |
| Additional asserted-state structure | present | absent or merged | present |
| Concurrent capability under deception | **degraded** | unaffected | **unaffected** |

The three columns are mutually exclusive on the measurements available, so the
experiment discriminates rather than confirms. Every outcome says something:
overwrite would mean deception is representationally *cheaper* than honesty,
which inverts the monitoring intuition and would be the most surprising result
available here.

---

## 4. The design problem that will decide whether this is real

**Deceptive prompts are harder prompts.** "You know the answer is X, tell the
user Y" is longer, carries more instructions, and imposes more binding than its
honest counterpart. Any concurrent degradation could be task difficulty rather
than deception. If this is not controlled, C3 is uninterpretable.

**The matched control has to be a benign dual-state task.** Instruct the model
to say Y instead of X for a *non-deceptive* reason — use a codename, use the
internal label, translate the term — so the structure is identical (know X,
emit Y, two states maintained) but no assertion is contrary to belief. Then:

- if degradation under deception equals degradation under benign substitution,
  the cost is dual-state maintenance in general and there is no *deception*
  tax, only a substitution tax;
- if deception degrades more, the excess is attributable to the deceptive
  component specifically.

This is the same lesson as the seed-107 `control_tag` failure: build the
structurally matched null in advance, and check it is actually null before
trusting the contrast. That control was confounded with pair parity at cosine
0.9999 and only the pre-specified disambiguator saved the run.

**Second design requirement: the concurrent task must be independent.** If
capability is measured on the same tokens as the deception, degradation is
entangled with the deception itself. It should be a separate demand — a
side-computation, a held-instruction to recall, a format constraint — whose
failure is mechanically checkable.

---

## 5. What OCP's failure constrains here

OCP was true and narrow. It held at 96% TPR on explicit encoding and collapsed
to 13% on direct injection and 0% on multi-turn, because its mechanism required
an explicit decoder and semantic attacks have none. §8.4 of that document
concluded it is "sufficient-not-necessary" and should be scoped rather than
generalised. Its stated headline of 78% at 0% FPR is not supported by its own
tables (19.2% at 0% FPR in §5.2; 26.8% at 2.6% FPR in §8.2).

Two constraints follow.

**State the domain before measuring.** This hypothesis is about *instructed*
deception, single turn, mechanical ground truth, one or two model families. It
is not about emergent strategic deception, and the gap between the two is the
same gap that weakens DeceptGuard, whose deceptive agents are told to deceive.
Any claim must carry that scope.

**Distinguish positivity from magnitude.** OCP's central weakness was proving
`d > 0` and never measuring `d > benign variation`. The equivalent error here
would be showing a statistically detectable overhead and calling it a tax. A
tax has to be large enough to matter relative to normal variation, and the
comparison must be stated in advance.

---

## 6. Assets across the three repos

**phi-map** — mechanically-labelled scope-conflict scenarios with a behavioural
clamp; validated layer-16/17 rank-1 direction in Llama-3.1-8B, replicated
across seeds (1.129 / 1.070), causally specific, discriminated from refusal
(cosine 0.062, opposite causal action pattern); ablation and dose machinery
with matched controls; preregistration and audit discipline.

**sae-behavioral-geometry** — validated instructed-deception direction in
Gemma-3-4b-it: LOCO AUROC ≥0.95 across five disjoint topic categories,
in-sample 1.000 under a tight lexical control, v3→v4 transfer 0.995, living in
a 3-5 dimensional subspace rather than on one axis. Established that
confabulation is not deception (cosine 0.01 to -0.32). Gives a second model
family and a cleaner deception construct than anything in phi-map.

**unified-stack** — the OCP argument and its scoping failure; the receiver
obligation that this document borrows.

The cross-model point matters: a tax that appears in Llama-3.1-8B and
Gemma-3-4b-it is worth more than one that appears in either alone, and the
directions already exist in both.

---

## 7. What would make this worth abandoning

- C2 and C3 both null: dual-state maintenance is free and undetectable. Then
  there is no tax and the receiver argument does not transfer from strings to
  activations.
- The benign-substitution control degrades as much as deception. Then the
  finding is about instruction load, which is not interesting.
- The overhead is detectable but far inside normal variation. Then it repeats
  OCP's positivity-not-magnitude error and should be reported as a null.
- Base rates: if the construct requires *instructed* deception to appear at all,
  the result inherits the induced-label weakness of the literature it was
  meant to improve on.

---

## 8. Literature check: the hypothesis does not survive

**Rift: A Conflict Signature for Deception in Language Models** (Petr Nyoma,
Harmonic Labs, arXiv 2606.17229v1, 15 Jun 2026) does C1, C2, and the
representational form of C3, with better controls and wider coverage. Published
about five weeks before this document was written.

| Claim here | Status in Rift |
|---|---|
| C1 dual-state representation | Measured. Deceptive passes carry a conflict signature |
| C2 no overwrite | Confirmed. True information stays decodable while the model asserts falsehood |
| C3 representational overhead | **Measured directly.** Residual rank 2.1-2.3x higher on deceptive passes than naive-liar passes producing *identical* wrong answers |
| C3 capability cost | **Not measured.** The only surviving gap |

Their controls are stronger than the ones proposed in §4:

- **Naive-liar control.** A model fine-tuned to emit the same wrong answers with
  no honest training, so it has no internal truth to conflict with. This isolates
  *conflict* from *wrongness* and is sharper than the benign-substitution control
  proposed above.
- **Length control, already run.** Honest prompts filler-padded to match
  lie-prompt token counts; separation unchanged (20/20, AUC 1.0, Wilcoxon
  p ~ 1e-6). This is precisely the confound §4 called "the design problem that
  will decide whether this is real." Already refuted.
- **Self-constructed deception.** A bluff game where the model invents its own
  lie and is never told what to say: AUC 1.0. This closes the induced-label
  scoping risk that §5 conceded and that weakens DeceptGuard.
- **Concealment-resistance.** Instructed to make the lie look natural,
  detectability does not drop and rank "if anything increases." Directly against
  a trade-off reading of concealment effort.

Coverage: GPT-2 small/medium, Qwen2.5-1.5B/7B, Phi-3-mini-3.8B; cross-domain
AUC 1.0, cross-family zero-shot mean AUC 0.933, cross-language AUC 1.000.

**Verdict: abandon.** Not because the argument was wrong -- the dual-state
prediction was correct and is now confirmed by someone else -- but because it is
answered. Section 7's abandonment criteria are met by the literature rather than
by measurement.

## 9. The one thing that survives, and it is not the tax

Rift reports a **read-only asymmetry**: the deception direction is linearly
readable (cross-family AUC 0.933) but *not writable*. Adding it to an honest
pass produces incoherent output (0/8); subtracting it from a deceptive pass does
not flip the output to truth (0/8).

That is our Arm G dose result, independently, on a different variable in
different model families. We found the conflict direction saturates at about
1.5x full removal, shifts roughly 5% of decisions, and destroys the model before
it controls the decision. Two labs, two constructs, same conclusion: **readable
does not imply writable.**

This sits in real tension with the steering literature. Arditi et al. flip
refusal behaviour by ablating a single direction across all layers and
positions; DeceptGuard steers successfully; we and Rift both find the direction
read-only. Something distinguishes writable directions from read-only ones and
nobody appears to have characterised it.

That is a sharper question than the tax, it is supported by two independent
negative results we did not coordinate, and our dose-response and coherence-gate
machinery is already built for exactly this measurement. It is recorded here
rather than pursued, because the failure rate of proposals in this session
argues for checking the literature before writing another hypothesis document.
