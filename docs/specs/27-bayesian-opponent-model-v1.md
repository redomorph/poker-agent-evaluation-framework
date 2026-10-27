# Bayesian Opponent Model v1

## Status

This specification freezes the first Phase-2 Bayesian online opponent-observation model.

```text
model_version                 = bayesian-opponent-model-v1
prior_version                 = bayesian-opponent-prior-v1
opportunity_definition        = calibration-opportunities-v2
```

The model is a **measurement and uncertainty component**. It does not itself choose poker actions, define an exploitative policy, change the static-profile policy, or consume hidden game state.

## 1. Scope

The Phase-2 roadmap requires a Bayesian online learner that estimates observable opponent frequencies together with uncertainty. The v1 learner tracks only quantities supported by the frozen public-action opportunity contract:

- VPIP;
- PFR;
- 3Bet frequency;
- CBet frequency;
- FoldToCBet frequency;
- postflop aggressive-action share.

The learner receives only already-classified observable evidence counts. It does not receive opponent hole cards, future board cards, hidden deck state, hidden RNG state, or the opponent's true parameter vector $\theta$.

## 2. No project-wide default prior

This specification deliberately does **not** choose an experiment prior.

Every use of the learner must supply an explicit `bayesian-opponent-prior-v1` vector before the first observation. Each metric uses an integer-shape Beta prior:

$$
q_m \sim Beta(\alpha_{0,m},\beta_{0,m}),
$$

where

$$
\alpha_{0,m}\in\mathbb{Z}_{>0},\qquad
\beta_{0,m}\in\mathbb{Z}_{>0}.
$$

The six prior pairs are part of the later adaptive-agent experiment configuration and must be versioned and persisted there. A neutral `Beta(1,1)` prior may be used in tests or a future experiment only if that experiment explicitly freezes it; it is not an implicit framework default.

## 3. Bernoulli evidence model

For a frequency metric with $e$ observed events in $o$ eligible opportunities,

$$
q_m \mid data \sim
Beta(\alpha_{0,m}+e,\;\beta_{0,m}+o-e).
$$

The posterior mean is

$$
E[q_m\mid data]
=\frac{\alpha_m}{\alpha_m+\beta_m},
$$

and the posterior variance is

$$
Var[q_m\mid data]
=\frac{\alpha_m\beta_m}
{(\alpha_m+\beta_m)^2(\alpha_m+\beta_m+1)}.
$$

The implementation stores the exact integer parameters and serialises exact rational mean/variance values as numerator/denominator pairs. It does not require floating-point arithmetic to define the posterior state.

## 4. Aggression treatment

The project aggression statistic is

$$
Aggression=\frac{A}{P},
$$

where $A$ is the number of postflop aggressive actions and $P$ is the number of postflop passive actions; folds are excluded.

A Beta-Bernoulli model cannot directly treat this ratio as a Bernoulli probability. Therefore v1 models the associated aggressive-action share

$$
q_A=\frac{A}{A+P}.
$$

Aggressive postflop actions are Bernoulli successes and passive postflop actions are failures. Folds contribute to neither side.

For a posterior $Beta(\alpha_A,\beta_A)$, the project exposes the aggression ratio implied by the posterior mean share as

$$
\frac{E[q_A]}{1-E[q_A]}
=\frac{\alpha_A}{\beta_A}.
$$

This quantity is explicitly a transformation of the posterior-mean share. It is not claimed to equal $E[q_A/(1-q_A)]$.

## 5. Opportunity definitions

All evidence must use exactly:

```text
calibration-opportunities-v2
```

The same opportunity semantics therefore apply as in policy calibration:

- one VPIP opportunity per dealt hand;
- PFR only when a legal wager increase exists;
- 3Bet only for the frozen qualifying single-prior-raise opportunity;
- CBet only for the frozen qualifying continuation-bet opportunity;
- FoldToCBet only when directly facing a qualifying CBet;
- postflop aggression over bet/raise versus call/check actions, with folds excluded and all-ins classified by betting effect.

Evidence from another opportunity-definition version is rejected.

## 6. Immutable sequential state

The learner state is immutable. Updating returns a new state.

Every update supplies `first_hand_index`. It must equal the state's current `hands_observed` value. This makes duplicate or reordered evidence an error at the model boundary.

Because `calibration-opportunities-v2` assigns exactly one VPIP opportunity per supplied dealt hand, cumulative

```text
hands_observed == cumulative VPIP opportunities
```

is an invariant.

A batch with zero opportunities for a particular metric leaves that metric's posterior unchanged.

## 7. RNG and hidden-information contract

Posterior updating is deterministic and consumes **no RNG**.

The learner accepts only observable behaviour counts. It must not inspect `TRUE_GAME_STATE` or any hidden fields. Any later adaptive policy that uses this state remains subject to the existing hidden-information gate and event-indexed agent-RNG contract.

## 8. Persistence and replay identity

The full learner state serialises:

- model version;
- opportunity-definition version;
- hands observed;
- explicit prior vector;
- cumulative event/opportunity counts;
- derived posterior $\alpha$ and $\beta$;
- exact rational posterior mean and variance;
- derived aggression ratio from posterior mean.

The canonical JSON representation is SHA-256 hashed as `state_hash`.

Deserialisation recomputes every derived posterior quantity from the explicit prior and cumulative evidence. A payload whose derived values do not match is rejected.

## 9. Deliberate non-goals

`bayesian-opponent-model-v1` does not freeze:

- a default prior;
- an opponent-dependent action policy;
- dynamic preflop ranges;
- postflop response models;
- MCTS determinization or tree semantics;
- a posterior credible-interval quantile algorithm;
- non-stationary-opponent dynamics;
- a confirmatory H4 experiment.

Those are separate Phase-2 specifications and must not be inferred from this learner.

## 10. Validation requirements

Before this learner can be used by an adaptive benchmark agent, tests must verify at minimum:

1. positive explicit prior parameters;
2. exact Beta-Bernoulli updates for every tracked metric;
3. correct aggressive-share treatment;
4. zero-opportunity stability;
5. sequential/batch update equivalence;
6. duplicate/reordered evidence rejection;
7. opportunity-version rejection;
8. canonical deterministic state hashing;
9. tampered derived-posterior rejection;
10. no action RNG consumption by the learner.
