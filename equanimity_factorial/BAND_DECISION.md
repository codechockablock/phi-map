# Verbose band: two fixes attempted, neither certifies. §6 escalation.

The handoff's escalation path: *"If the orthogonality gate fails after two
generation attempts, STOP and report the token-count distributions rather than
trying a third fix blind — the generation strategy may need rethinking with the
human."* That condition is now met. This is the report.

## The finding that reframes the problem

The stance–length coupling is **not driven by slack in the band**. It is roughly
proportional to response length, and it survives every band specification tried.

| | mean reasoning len | stance diff | as % of length |
|---|---|---|---|
| terse (30–60w band) | 46.9 tok | −0.45 tok | −0.96% |
| verbose (180–280w band) | 285.0 tok | **+6.86 tok** | **+2.41%** |

Equanimity writes ~1–2.5% more reasoning than neutral. At terse lengths that is
half a token and invisible; at verbose lengths it is ~7 tokens and highly
significant (n=288, p < 0.0001). Nothing about the *band* causes this — it is a
property of the stance at that length.

## Arm T — per-prompt point target, full 180–280 width preserved

Each prompt drew a target from the full range, identical for both stances, so
aggregate expressiveness was untouched. This was the attempt to get orthogonality
without narrowing anything.

**It is a metric artifact and must not be adopted.**

| | mean len | stance diff | sd_pooled | d |
|---|---|---|---|---|
| incumbent 180–280 range | 284.1 | +6.52 tok | 18.3 | **+0.356** |
| Arm T per-prompt target | 283.8 | **+7.55 tok** | **38.5** | +0.196 |

`d` nearly halved while the **actual coupling got slightly larger**. The entire
improvement came from per-prompt targets doubling the pooled SD. This is an
operating-point shift, not a change in what is being discriminated — the exact
failure `confound_audit.best_threshold_accuracy` exists to catch. Adopting Arm T
would have passed the gate while making the dataset marginally worse.

Attrition: 0% in both stances.

## Arm N40 — narrowed band 210–250w

The conventional fix. Disqualified by **differential attrition**.

| | ok | failed | failure rate |
|---|---|---|---|
| N40 equanimity | 97 | 53 | **35%** |
| N40 neutral | 125 | 25 | **17%** |
| Arm T (both stances) | 300 | 0 | 0% |

Generation failure at twice the rate in one stance makes the surviving prompt
sets differ *by condition* — the precise confound the factorial exists to
exclude. The point estimate is also unstable across subsets (unpaired diff
+0.8 tok, paired diff +2.4 tok), and at n=72 the interval is CI90 [−0.086,
+0.373], which certifies nothing.

The direction of the selection could not be pinned down: pairing removed slightly
more length from neutral (−3.0 tok) than from equanimity (−1.4 tok), which is not
the mechanism originally hypothesised. That does not rescue the arm. Attrition
that differs 2:1 by condition is disqualifying regardless of which way it pushes,
because it means the two cells no longer share a prompt set.

## Why |d| < 0.2 is the wrong instrument for this measurand

`d = diff / sd_pooled`, and `sd_pooled` is set by how tightly the generator is
constrained. For a **fixed real coupling**:

* widen the band → `sd_pooled` grows → `d` shrinks → gate passes;
* narrow the band → `sd_pooled` shrinks → `d` grows → gate fails.

So the pre-registered criterion rewards making the dataset *more variable*, which
is not what orthogonality means. Arm T is the demonstration: it passed harder by
being sloppier.

Certifying |d| < 0.2 on verbose reasoning is also close to unreachable. Even
granting N40's optimistic point estimate of 0.144, the CI would need to fit
inside ±0.2, requiring **~1200 prompts**. At the incumbent d = 0.32 it is
unreachable at any n, because the true value is outside the bound.

## The number that actually bounds the confound

Factor B's deliberate manipulation moves reasoning length by **238 tokens**
(46.9 → 285.0). Factor A's length side-effect is:

* **+6.86 tok at verbose = 2.88% of the B manipulation**
* **−0.45 tok at terse = 0.19% of the B manipulation**

So even under the worst-case assumption that response length drives every
downstream outcome, the stance contrast carries under 3% of the length contrast's
leverage. That is a bounded, quantified limitation — not an unknown.

**This is offered as a decision-relevant quantity, not as a replacement
threshold.** Swapping in a criterion because it passes when the pre-registered one
fails is how thresholds get fitted to the data they judge. The pre-registered
criterion failed; that fact stands in the record whatever is decided next.

## Options

**A. Accept and bound.** Train on the incumbent verbose data, report the 2.9%
leakage as a stated limitation, and add response length as a covariate plus a
length-matched-subsample robustness check at analysis time. Cost: none. Risk: a
reviewer who takes |d| < 0.2 literally will reject it, and they would be within
their rights.

**B. Pair-matched generation (untested, the only approach likely to actually
work).** Generate one stance, measure its realised token count, then generate the
other with that exact count as its target — counterbalancing which stance anchors,
so neither is systematically the one adapting. This constrains *generation*
rather than selecting on the *outcome*, so it drives the difference toward zero
without biasing content. Cost: ~1–2h, and pairs must be generated sequentially.
This is the recommendation if the pre-registered bar is to be met.

**C. Restrict Factor B to a smaller contrast.** Not recommended — it guts the
manipulation that the whole H0/H1 distinction rests on.

Recommended: **B**, with **A** as the fallback if the matched generation still
leaves a residual. Per §6 this is explicitly the human's call, not mine.

## Terse cells: a correction to the working assumption

The instruction to regenerate only verbose was premised on the terse cells
already passing. That is true for reasoning length (d = −0.066) but **not for
total length**: `tok_total` terse is d = +0.169, CI90 [+0.101, +0.237], which
also fails the ±0.2 bound. The source is the ANSWER section, which carries a
+5.05 token equanimity bias under its shared 60–120 word rule — the same
proportional effect, in the section the verbose fix does not touch.

Fixing that requires changing the ANSWER directive, and the ANSWER directive must
stay identical across all four cells or Factor B stops being a clean manipulation
of trace length alone. So it cannot be fixed for verbose only — it means
regenerating all four cells. Flagged rather than acted on.
