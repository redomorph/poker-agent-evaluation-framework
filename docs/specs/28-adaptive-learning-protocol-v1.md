# Adaptive Learning Protocol v1

## Status

This specification freezes the **observation and learning mechanics** for the first Phase-2 adaptive-agent branch.

```text
protocol_id                       = phase2-adaptive-learning-protocol-v1
protocol_version                  = adaptive-learning-protocol-v1
opponent_model_version            = bayesian-opponent-model-v1
opponent_prior_version            = bayesian-opponent-prior-v1
opportunity_definition_version    = calibration-opportunities-v2
public_action_trace_version       = public-action-trace-v1
agent_observation_version         = agent-observation-v2
position_pairing_version          = adaptive-position-pairing-v1
replay_version                    = adaptive-learning-replay-v1
statistical_unit                  = trajectory_cluster
```

The machine-readable freeze is:

```text
configs/experiments/phase2-adaptive-learning-protocol-v1.json
```

Its canonical payload SHA-256 is:

```text
03f8ab5b2b35c86910d46df1d5187c306814e427b3f514eb85748a9a424adf5c
```

This specification is intentionally **not an adaptive poker policy**. It freezes how public evidence becomes Bayesian state. Posterior-to-range mapping, posterior-to-action mapping, dynamic preflop ranges, dynamic postflop strategy, MCTS semantics, and the final H4 experiment remain separate specifications.

## 1. Scientific purpose

The Phase-2 roadmap requires an online opponent learner with uncertainty. `bayesian-opponent-model-v1` already freezes the posterior mathematics but deliberately leaves experiment priors and trajectory mechanics open.

This protocol closes those mechanics before adaptive policy implementation begins. The objectives are:

1. prevent hidden-information leakage;
2. make the learning history replayable;
3. prevent position-pair order from conferring an information advantage;
4. define the temporal dependency structure before statistical analysis;
5. prevent a later adaptive policy from silently changing how evidence was learned.

## 2. Frozen prior vector

The first adaptive branch uses independent symmetric uniform Beta priors for all six observable Bernoulli quantities:

$$
q_m \sim Beta(1,1).
$$

The frozen vector is:

```text
VPIP                       Beta(1,1)
PFR                        Beta(1,1)
3Bet frequency             Beta(1,1)
CBet frequency             Beta(1,1)
postflop aggressive share  Beta(1,1)
FoldToCBet                 Beta(1,1)
```

Version:

```text
prior_selection_basis = independent-uniform-beta-1-1-v1
```

### 2.1 Rationale and limitation

`Beta(1,1)` is chosen as a transparent symmetric prior for the first adaptive baseline. It does not encode the Baseline Synthetic population, the five static-profile weights, or a guessed real-player population.

This is a researcher-controlled Phase-2 design choice, not a universal framework default. `bayesian-opponent-model-v1` continues to require explicit priors. A later informative-prior experiment must use a new versioned protocol.

The six posteriors are independent in v1. Cross-metric covariance is not modeled.

## 3. Evidence source and hidden-information boundary

The learner may consume only public behavioral evidence derived from:

```text
public-action-trace-v1
agent-observation-v2
```

Behavioral opportunities and events are classified using exactly:

```text
calibration-opportunities-v2
```

The learner must not use:

- opponent hole cards, including cards revealed at showdown;
- undealt or future board cards;
- hidden deck state;
- opponent RNG state;
- the opponent's true parameter vector $\theta$;
- any field available only in `TRUE_GAME_STATE` and not in the public observation/history contract.

Framework-generated timeout resolutions are not treated as opponent strategic actions. The later adaptive execution contract must preserve the existing timeout/failure evidence separately from the opponent-behavior trace.

## 4. Update schedule

### 4.1 One update per completed hand

Evidence is accumulated for one complete terminal hand and applied exactly once after the hand terminates:

```text
update_timing = after-terminal-hand-v1
batches_per_hand = 1
```

No partial-hand posterior update occurs in v1.

Consequently, observations from hand $h$ may influence actions only from hand $h+1$ onward:

```text
same_hand_adaptation = false
first_effective_hand_offset = 1
```

This boundary prevents a completed-hand classifier from retroactively affecting an earlier decision from the same hand.

### 4.2 Chronology

Updates are strictly chronological. For a Bayesian model with `hands_observed = h`, the next update uses:

```text
first_hand_index = h
```

Duplicate, skipped-replay, or reordered update batches are rejected by the `bayesian-opponent-model-v1` state boundary.

## 5. State persistence and reset semantics

Within one adaptive run against one fixed opponent identity:

- the posterior starts from the frozen prior vector;
- state persists from one completed hand to the next;
- the opponent identity remains fixed for the run.

The state resets to the original prior vector:

- at the start of every independent run/trajectory;
- whenever the opponent identity changes;
- at the start of the second symmetric positional run described in Section 6.

No Bayesian state is shared across independent trajectory clusters, experiment seeds, opponent identities, or the two symmetric position conditions.

## 6. Position compensation for adaptive agents

This section follows the active RNG specification for opponent modeling.

The two positional conditions are separate symmetric runs:

```text
Run 1: A-BTN against B-BB
Run 2: B-BTN against A-BB
```

Both use the same defined card realizations under the CRN plan:

```text
chance_relation = identical-defined-card-realisations
```

but their adaptation histories are initialized separately:

```text
adaptation_histories = separately-initialized
share_state_between_position_runs = false
```

The second run must never inherit information learned in the first run. This prevents an order-dependent information advantage in the paired position comparison.

If a future adaptive architecture cannot preserve identical chance realizations, that experiment must explicitly use a new pairing contract and classify the relation as seed-synchronized rather than identical-realization CRN.

## 7. Statistical dependency structure

Hand observations within one adaptive run are temporally dependent because later decisions depend on earlier evidence.

Therefore:

```text
statistical_unit       = trajectory_cluster
bootstrap_unit         = trajectory_cluster
hand_level_independence = false
```

Individual hands or decisions from one adaptive trajectory are never promoted to independent inferential observations.

The paired trajectory cluster contains the two symmetric positional runs from Section 6, each with its own independently initialized learning history. Exact H4 contrast construction is deferred to the H4 experiment specification.

## 8. Frozen adaptation horizons and segments

The descriptive learning-curve horizons from the active opponent-modeling specification are preserved:

```text
10
20
50
100
500 hands
```

The reporting segments are:

```text
Early Adaptation   hands 1-20
Main Adaptation    hands 21-100
Steady State       hands 101-500
```

The 100-hand value remains a cumulative learning-curve endpoint over hands 1-100. It is not the 100th hand and is not automatically the primary confirmatory H4 endpoint.

This protocol does **not** freeze the eventual primary H4 horizon:

```text
primary_h4_horizon = null
```

That choice must be made only after an adaptive policy mechanism exists and before the final H4 experiment.

## 9. RNG contract

Evidence classification and Bayesian updating consume no RNG:

```text
evidence_classifier_rng_draws = 0
posterior_update_rng_draws     = 0
rng_policy                     = no-learning-rng-v1
```

A later adaptive action policy remains subject to the existing event-indexed action RNG and domain-separated equity RNG contracts. The learning layer must not advance either stream.

## 10. Replay and persistence contract

Every completed-hand learning transition must be replayable from public evidence.

For each learning update, the adaptive execution record must preserve at least:

```text
model_state_before_hash
public_evidence_counts
model_state_after_hash
```

The replay implementation must reconstruct the Bayesian transition from the same public evidence and verify both state hashes.

The persisted evidence counts must identify `calibration-opportunities-v2`. Hidden showdown cards are not part of the learning evidence even when the replay system stores them elsewhere as true-state evidence.

## 11. Deliberate non-goals

This protocol does not freeze or implement:

- posterior-to-range mapping;
- posterior-to-action mapping;
- dynamic preflop ranges;
- dynamic postflop strategy;
- MCTS determinization/tree/rollout semantics;
- the primary H4 horizon;
- the H4 power or precision plan.

Those choices materially affect the adaptive strategy or confirmatory experiment and require separate versioned specifications.

## 12. Validation gate before adaptive policy implementation

Before posterior-to-policy work begins, tests must verify at minimum:

1. the canonical machine-readable protocol and SHA-256 identity;
2. all six explicit `Beta(1,1)` priors;
3. exactly one post-terminal update per hand;
4. no same-hand use of completed-hand evidence;
5. public-only evidence and `calibration-opportunities-v2` reuse;
6. persistence within a run and reset between independent runs/opponents;
7. independently initialized symmetric positional histories;
8. `trajectory_cluster` statistical/bootstrap unit;
9. the 10/20/50/100/500 horizons and three frozen segments;
10. zero RNG consumption by evidence classification/updating;
11. the replay before/evidence/after state-hash chain;
12. rejection of semantically altered protocol fields even when an outer hash is recomputed.

Only after this gate is green may the project freeze the separate posterior-to-range/policy mechanism.
