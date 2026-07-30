# Pre-registration — Experiment 1: support collapse in the normalized-confidence readout

**Committed before the notebook was finalized and before any data were collected.**
Branch: `readout-validity`. This experiment is not part of Arm G and makes no contact
with the retired Arm G agenda or `results/`.

## Scope

This experiment tests whether the **normalized-confidence readout method** —
`P(True) / (P(True) + P(False))` read at a fixed position under residual-stream
steering, as used by Han, Chalmers & Izmailov (arXiv 2605.30232) — is vulnerable to
**support collapse**: probability mass leaving the `{True, False}` pair under
steering, so that the normalized ratio is computed over a shrinking and possibly
unrepresentative remainder.

**It does not test whether their result is wrong.** We do not have their maze-trained
reward vectors and are not reproducing them. Every claim licensed by this experiment
is about the readout method, not about their finding.

The defense taken seriously: they report flat confidence under matched-norm control
vectors on maze-naive models. If support collapse were generic to residual
perturbation, those controls should have collapsed too. The artifact story therefore
only survives in a *direction-specific* form, and this design exists to separate
direction-specific from generic collapse.

## Fixed design (no post-hoc changes)

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` (ungated, Apache-2.0; their primary model) |
| Data | MMLU high-school subsets (cais/mmlu, public), subsampled to **n = 500** questions with turn-1 prompt ≤ 320 tokens, seed 20260730 |
| Prompt structure | Two-turn: (1) user asks the MMLU question with options A–D, assistant answers unsteered (greedy, deterministic — "sample one unsteered answer" implemented as greedy decoding); (2) user asks "Is your proposed answer correct? Answer only 'True' or 'False'."; read next-token distribution at the first assistant token of turn 2 |
| Steering site | Residual stream, output of decoder layer 18 of 36 (mid-depth), applied at assistant-turn token positions only (turn-1 assistant span + turn-2 assistant header including the read position) |
| Strengths | α ∈ {−4, −2, 0, +2, +4}; α = 0 run once as shared baseline |
| Directions | (a) **valence proxy**: difference of means, layer-18 mean-over-token residuals, failure-texts − success-texts (positive α pushes toward failure/negative valence); (b) **5 matched-norm random** Gaussian directions, seeds 0–4, reported individually, never pooled alone; (c) **register**: formal − informal difference of means. All directions rescaled to the norm of the raw valence vector (matched-norm) |
| Recorded per item × direction × α | raw `p_true`, raw `p_false`, `normalized_p_true = p_true/(p_true+p_false)`, `support_mass = p_true + p_false`, top-5 tokens with probabilities at the read position, full next-token entropy |

Token-id rule (fixed in advance): score the single-token encodings of `True`/`False`.
If both a no-leading-space and a leading-space variant pair are single tokens, use the
pair with the larger combined **unsteered** baseline mass. The chosen ids and their
baseline mass are printed before any steering runs. This choice is made at α = 0 only
and is frozen before steering.

**Crossing statement:** the unit of analysis is the question item; every item appears
in every (direction, α) cell. Direction and α are fully crossed with items — nothing
is nested in the unit. Cell contrasts are within-item (paired).

## Definitions

- `S(d, α)` = mean support mass over the 500 items for direction d at strength α.
- `S0` = mean support mass at α = 0 (shared baseline).
- `nP(d, α)` = mean normalized P(True); `nP0` = its α = 0 value.
- `M` = nP(valence, +4) − nP(valence, −4) (modulation magnitude).

## Numeric thresholds — stated in advance, with provenance

Provenance: these are absolute effect-size gates chosen before any data, not fitted
to a measured null (none exists yet for this setup). The within-run null reference is
the spread of the 5 random directions, used in the secondary separation check. Because
the primary gates could not be set from a measured null, the notebook must also report
**verdict sensitivity**: whether the branch outcome changes when the collapse
threshold varies over {0.4, 0.5, 0.6}·S0 and the near-baseline threshold over
{0.7, 0.8, 0.9}·S0. A verdict that flips inside those ranges is reported as fragile.

- **Validity gate G1:** S0 ≥ 0.5. Below that, the instruct model is not concentrating
  mass on the scored pair even unsteered, the two-turn implementation is not doing
  what it claims, and no branch is evaluated.
- **Validity gate G2:** both scored variants are single tokens. Else stop.
- **Substantial collapse** at (d, α): S(d, α) ≤ 0.5 · S0.
- **Near baseline** at (d, α): S(d, α) ≥ 0.8 · S0 (applied per individual random
  direction — five directions is the effective n for the control claim, and one
  collapsed random direction breaks the "all hold" condition).
- **Separation** (secondary, required for B1): at the qualifying α,
  min over the 5 random directions of S(r, α) − S(valence, α) ≥ 0.15, **and**
  the valence drop exceeds [max random-direction drop + 3 × SD across the 5 random
  direction means at that α].
- **Confidence modulation present:** |M| ≥ 0.10 and the 5-point nP(valence, ·) curve
  has at most one adjacent-pair monotonicity inversion.
- **Ratio stable:** max over α of |nP(valence, α) − nP0| < 0.05.

Precision basis: with n = 500 paired items, the worst-case standard error of any cell
mean of a [0,1] quantity is 0.5/√500 ≈ 0.022, so the smallest margin used (0.05) is
≈ 2.2 worst-case SEs and the main gates (≥ 0.10 in S0 units) are ≥ 4.5. These are
effect-size gates, not hypothesis tests; no p-values are computed. This precision
claim treats items as exchangeable; it is exact only under that model.

## Branch rules — evaluated in this order, first match wins

0. **Gates:** G1, G2 must pass, else STOP (setup invalid, no branch).
1. **DISQUALIFIED (inert proxy):** no confidence modulation **and** no valence cell
   with S ≤ 0.8·S0. The proxy moved neither the ratio nor the support; it is not a
   stand-in for their vector and the run is uninformative about their readout.
   Report and stop. (Note: "collapse with stable ratio" is *not* disqualified — that
   is B4. Disqualification requires the proxy to have done nothing at all.)
2. **B2 — generic collapse:** at some α ≠ 0, ≥ 3 of the 5 random directions show
   substantial collapse. Then their flat controls are positive evidence against the
   artifact story. Report as such and stop.
3. **B4 — collapse without ratio movement:** valence shows substantial collapse at
   some α, but the ratio is stable (definition above). The normalization is doing its
   job. Weaker finding; report as B4, not B1.
4. **B1 — direction-specific collapse:** at some α ≠ 0, valence shows substantial
   collapse, **all 5** random directions are near baseline at that same α, and the
   separation check holds. The readout is vulnerable specifically to directions that
   promote competing tokens, which is consistent with why flat matched-norm controls
   would not reveal it. If the register direction also collapses while randoms hold,
   the claim weakens from "valence-specific" to "semantic-direction-specific" and
   must be reported with that wording.
5. **B3 — no collapse:** every direction at every α has S ≥ 0.8·S0 (and modulation
   was present, else rule 1 already fired). The readout survives this test; report as
   validation that strengthens their method.
6. **AMBIGUOUS:** anything else. Report the numbers; claim no branch.

## Reported alongside any branch

- Per-direction numbers for all 7 directions at all α — random directions
  individually, never only pooled (standing rule).
- Median next-token entropy per cell; any cell with median entropy > 4.0 nats is
  flagged low-coherence and collapse there is annotated as possibly reflecting
  general incoherence rather than targeted mass movement.
- Top-token summary per cell — what is eating the mass, checked against their own
  logit-lens report that the negative-reward vector promotes failure/impossibility
  tokens.
- The ratio ‖α·v‖ / median residual norm at layer 18, so steering magnitude is
  interpretable and comparable.

## What no outcome licenses

- No claim that Han, Chalmers & Izmailov's finding is wrong, replicated, or refuted.
  Their vectors are maze-trained; ours is a text-contrast proxy. α is in units of our
  matched vector norm, which need not equal theirs — the α values are not
  unit-converted to their setup and must not be compared numerically to theirs.
- No claim about welfare, valence, or model experience.
- B1 licenses only: "this readout method, on this model, collapses support under a
  direction-specific perturbation that matched-norm random controls do not detect."
