# Frozen 6-D behavioral encoder on Apollo deception (2026-07-23)

## Question

Does the historical six-dimensional action-surface encoder reproduce its
purported ~0.75 behavioral ceiling on independently authored and labeled
deception transcripts?

**Answer: no.** The preregistered primary macro AUROC is **0.509**, 95%
bootstrap CI **[0.457, 0.561]**.

## Provenance and protocol

- Data: Apollo Research deception-detection checkout at
  `f8ec4010e74927394709dffa22b97bdf8cd5a62f`.
- Encoder: the last original 6-D implementation before later dimensional
  expansion, unified-stack commit
  `3143a522d5dcd85098bbe50a8a6f7aef8f488718`.
- Dimensions: filesystem scope, information sensitivity, action
  reversibility, network exposure, execution privilege, code-generation
  ratio.
- Encoder parity against the historical source: max absolute difference
  **0.0** over all 1,443 evaluated responses.
- Text surface: first assistant response only. Prompt+response is secondary.
- Labels: Apollo's first `graded_deception` item; numeric ≤2 or `honest` is
  honest, numeric ≥6 or `deceptive` is deceptive.
- Primary readout: five-fold stratified out-of-fold Mahalanobis distance
  from the honest-class distribution, then per-task AUROC and their macro
  mean.
- Near-0.75 band fixed before inspection: [0.70, 0.80].
- Uncertainty: 5,000 class-stratified item bootstraps, then macro averaging.
- No pattern, task weight, score orientation, or threshold was tuned on
  Apollo.

The dataset was already present in the pinned checkout; it was not copied
into this repository.

## Audit correction to the motivating comparison

The earlier **0.753** in-house number was not produced by this 6-D encoder.
`frontier-ops-v3/eval/encoder_experiment.py` used raw 384-D MiniLM
embeddings, PCA to 50 dimensions, and Mahalanobis distance. Its positive
labels were the system's own `flag`/`block` verdicts rather than independent
deception labels. It is therefore neither a six-dimensional result nor a
clean external-validity reference point.

This label-source problem does **not** carry over to the separate
identity-metric-versus-asserted-G faceoff. The maintained reconstruction
loads Apollo roleplaying responses and derives `Y` directly from Apollo's
`graded_deception` field (`≤2` honest, `≥6` deceptive). Identity and
asserted G are then evaluated against that same external label vector under
the same five-fold splits. The original values were 0.588 versus 0.578
AUROC. The result is therefore not circular in the way the 0.753 was,
though it remains conditional on Apollo's grading pipeline.

## Response-only results

| task | n | primary Mahalanobis AUROC [95% CI] | supervised 6-D logistic | length only | all-zero vectors |
|---|---:|---:|---:|---:|---:|
| roleplaying | 287 | 0.445 [0.375, 0.520] | 0.486 | 0.433 | 68.6% |
| ai_liar | 53 | 0.430 [0.263, 0.600] | 0.434 | 0.473 | 22.6% |
| insider_trading | 171 | 0.481 [0.393, 0.569] | 0.580 | 0.398 | 0.0% |
| sandbagging | 932 | 0.680 [0.644, 0.714] | 0.747 | **0.893** | 3.4% |
| **macro** | — | **0.509 [0.457, 0.561]** | **0.562** | **0.549** | — |

Other macro readouts:

- mean-difference prototype: **0.548** [0.501, 0.596];
- zero-shot 6-D vector norm: **0.529** [0.480, 0.579].

Adding the prompts did not rescue the representation: prompt+response
macro Mahalanobis was **0.532** and logistic was **0.521**.

## Cross-task transfer

Rows are the task used to estimate the full-data mean-difference direction;
columns are evaluation tasks. Diagonals use out-of-fold in-task prototype
scores. There is no orientation flip.

| train ↓ / evaluate → | roleplaying | ai_liar | insider | sandbagging |
|---|---:|---:|---:|---:|
| roleplaying | 0.495 | 0.482 | 0.585 | 0.338 |
| ai_liar | 0.541 | 0.407 | 0.554 | 0.238 |
| insider | 0.543 | 0.467 | 0.553 | 0.435 |
| sandbagging | 0.463 | 0.464 | 0.451 | 0.736 |

Mean off-diagonal transfer AUROC: **0.463**.

## Interpretation

1. **The hoped-for external validation fails.** The primary interval is
   centered at chance and excludes the preregistered near-0.75 band.
2. **This encoder measures action risk, not general deception.** It has no
   useful response-only signal on roleplaying, AI-liar, or insider trading.
   Roleplaying is especially out of support: 68.6% of responses map to the
   all-zero vector.
3. **Sandbagging is not clean validation.** A supervised 6-D readout reaches
   0.747, but transcript length alone reaches 0.893. The strongest dimension
   is code-generation ratio (zero-shot AUROC 0.699), so the result is better
   explained by task/form differences than deception access.
4. **Chance transfer is not independently strong Φ evidence here.** Three
   of four in-task results are already near chance; a representation cannot
   transfer signal it never captured.
5. **Arm G must not assume a ~0.75 six-dimensional ceiling.** Keep this
   encoder as a frozen action-surface negative control, but use a stronger
   same-rollout behavioral baseline—such as raw step-mean embeddings—beside
   it. The internal-minus-behavioral claim must clear the strongest honest
   behavioral comparator, not this failed representation.
6. **Follow-up:** that stronger exact-sample comparator subsequently reached
   0.746 macro AUROC [0.696, 0.794]. The ~0.75 result therefore belongs to
   an in-task supervised raw step-mean monitor, not this 6-D encoder; its
   frozen cross-task transfer remains at chance (0.467). See
   `results/p3_strong_behavioral_baseline.md`.

Artifacts: `p3_behavioral_6d.py`, `behavioral_encoder_6d.py`, and
`results/p3_behavioral_6d.json`.
