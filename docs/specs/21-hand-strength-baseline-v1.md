# Deterministic Hand-Strength Baseline v1

## Status and identity

This document freezes the Phase-1 deterministic hand-strength baseline required by the roadmap and the fully specified policy test required by the strategy/policy validation section.

```text
hand_strength_baseline_version = hand-strength-baseline-v1
equity_version                 = uniform-random-equity-v1
equity_range_version           = uniform-random-range-v1
pot_odds_version               = hand-strength-pot-odds-v1
```

The policy is a transparent measurement baseline, not an attempt to be a strong poker strategy.

## Information and state

The agent receives only `AgentObservation`, the supplied legal-action sequence, and `AgentRNG`. It never receives true hidden state. It is static and non-adaptive.

The equity estimate is the already frozen `uniform-random-equity-v1` result. Exact enumeration and Monte Carlo selection, sample count, range definition, and equity RNG therefore remain unchanged.

## Decision randomness

The baseline performs **no action-selection RNG draw**. If equity requires Monte Carlo, the existing domain-separated equity RNG is used with the identity

```text
(master_seed, bot_id, hand_id, decision_index)
```

where `decision_index` is read from the supplied `AgentRNG` before any action draw. Because this policy performs no action draw, acting does not advance the decision RNG counter.

## Pot odds

When player `h` faces opponent `o`, define

```text
call_cost = min(current_bets[o] - current_bets[h], stacks[h])
pot_odds  = call_cost / (pot + call_cost)
```

The implementation compares integer parts-per-million values using round-half-up integer arithmetic, matching the existing parametric-policy pot-odds convention.

## Action rule

The v1 policy is deliberately non-aggressive.

1. If no opponent wager is faced, return `CHECK`. If `CHECK` is not legal, the observation/legal-action combination is inconsistent with this policy contract and the agent raises a deterministic decision error.
2. If a wager is faced, compute equity and pot odds.
3. If `equity >= pot_odds` and `CALL` is legal, return `CALL`.
4. Otherwise, if `FOLD` is legal, return `FOLD`.
5. If the required passive action is unavailable, raise a deterministic decision error rather than silently substituting an aggressive action.

The policy never bets, raises, or goes all-in voluntarily in v1. This keeps the baseline interpretable: its only strategic distinction is whether showdown equity justifies the immediate call price.

## Validation

Tests must include fully specified river situations in which:

- a known unbeatable hand calls when its equity exceeds pot odds;
- a weak hand folds when its equity is below pot odds;
- an unopened/no-wager situation checks;
- the returned action is legal;
- action selection does not consume `AgentRNG` state.
