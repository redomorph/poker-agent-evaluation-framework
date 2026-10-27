# Adaptive Lifecycle v1

## Status

This specification freezes the **runner-managed lifecycle integration** for the first adaptive hand-strength agent.

```text
protocol_id       = phase2-adaptive-lifecycle-v1
protocol_version  = adaptive-lifecycle-v1
agent_version     = adaptive-hand-strength-agent-v1
transition_version = adaptive-learning-transition-v1
```

The canonical machine-readable freeze is:

```text
configs/experiments/phase2-adaptive-lifecycle-v1.json
```

This specification does not change the Bayesian mathematics, posterior-to-range mapping, or action policy frozen in Specs 27-30. It only defines how those components are connected to complete trajectories.

## 1. Controlled lifecycle

At the start of each independent adaptive position run, the adaptive agent starts from the six frozen independent `Beta(1,1)` priors from Spec 28.

For one normal timeout-free hand:

```text
current Bayesian model
        ↓
all decisions in hand h use that model
        ↓
hand h terminates normally
        ↓
runner reconstructs calibration-opportunities-v2 public evidence
        ↓
one Bayesian transition
        ↓
new model may affect hand h+1
```

No evidence from hand $h$ may affect a decision within hand $h$.

## 2. Position pairing and resets

The runner resets lifecycle-managed adaptive agents:

1. immediately before the first positional run;
2. immediately before the second mirrored positional run.

The two runs therefore use separately initialized learning histories exactly as required by Spec 28. State is never shared from `A_BTN` into `B_BTN`.

The adaptive statistical unit remains:

```text
trajectory_cluster
```

## 3. Public evidence boundary

The runner, not the agent, converts the persisted hand trace into `BehaviorCounts` using exactly:

```text
calibration-opportunities-v2
```

The adaptive agent receives only those public behavioral counts. It is never given opponent hole cards, deck state, or `TRUE_GAME_STATE` solely for learning.

Stable bot identity, not seat identity, determines which player's public behavior is learned. This remains correct when bots swap seats in the mirrored run.

## 4. Timeout and exceptional hands

Framework timeout resolutions are not opponent strategy. The existing calibration classifier rejects timeout-altered hands rather than silently including framework-generated actions.

For v1, a timeout-altered hand therefore has:

```text
Bayesian update = none
adaptive trajectory status = invalid
replacement seed = forbidden
```

The timeout/forfeit hand remains durably recorded under the existing timeout contract. The adaptive trajectory is excluded rather than continuing with a posterior history containing a silent evidence gap.

Illegal-action, agent-error, and engine-error hands likewise produce no learning transition and already invalidate the run under the runner failure contract.

## 5. Adaptive agent

`adaptive-hand-strength-agent-v1` is a stateful wrapper around the pure Spec-30 decision function.

It:

- reports `is_adaptive = True`;
- owns exactly one immutable `BayesianOpponentModel` state at a time;
- initializes and resets to the frozen Spec-28 prior;
- delegates every action to `adaptive-hand-strength-policy-v1`;
- consumes zero action-selection RNG draws beyond the frozen policy behavior;
- accepts exactly one `BehaviorCounts` batch per normal terminal hand;
- rejects duplicate or reordered trajectory-hand updates.

The frozen pure policy remains unchanged.

## 6. Learning-transition persistence

Every successful hand-level Bayesian transition is persisted on the corresponding `RecordedHand` as `adaptive_learning_events`.

Each event contains at least:

```text
learner_bot_id
opponent_bot_id
trajectory_hand_index
model_state_before_hash
public_evidence_counts
model_state_after_hash
transition_hash
```

The transition hash covers all preceding fields and the transition version.

Static hands preserve the existing serialized shape when no learning events exist; the optional field is emitted only when non-empty.

## 7. Replay verification

Adaptive learning replay must not trust persisted behavioral counts by themselves.

For every recorded transition it must:

1. reconstruct the same persisted hand;
2. re-run `calibration-opportunities-v2` for the recorded opponent bot identity;
3. compare the recomputed counts with the persisted counts;
4. verify the before-state hash;
5. apply the Bayesian update chronologically;
6. verify the after-state hash and transition hash.

A new positional `config_id` starts from the frozen prior again.

## 8. Deliberate non-goals

This lifecycle does not freeze or implement:

- mid-hand Bayesian updates;
- learning from timeout-generated framework actions;
- CBet/FoldToCBet/aggression-based action adjustment;
- H4 horizon/sample-size choices;
- MCTS semantics.

## 9. Validation gate

Before adaptive pilot design begins, tests must verify at minimum:

1. canonical protocol identity and semantic tamper rejection;
2. frozen-prior initialization and reset;
3. one normal post-terminal update per hand;
4. duplicate/reordered update rejection;
5. reset before both mirrored runs;
6. stable bot identity under seat swaps;
7. per-hand transition persistence and hash integrity;
8. replay reclassification of public evidence and before/after hash verification;
9. static-run serialization compatibility;
10. timeout-altered hands cannot carry adaptive learning events;
11. lifecycle-managed timeout trajectories are invalid rather than silently continued.
