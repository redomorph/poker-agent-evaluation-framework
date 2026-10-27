# 30 — Adaptive Hand-Strength Policy v1

Status: **normative / frozen**

This specification freezes the first range-conditioned adaptive poker decision mechanism. It depends on `bayesian-opponent-model-v1`, `adaptive-learning-protocol-v1`, and `posterior-to-range-v1` and intentionally changes exactly one strategic input relative to `hand-strength-baseline-v1`: the opponent range used by the equity calculation.

## 30.1 Scientific role

The v1 adaptive policy is a controlled treatment for later H4 evaluation.

Control:

```text
hand-strength-baseline-v1
uniform-random opponent range
```

Treatment:

```text
adaptive-hand-strength-policy-v1
Bayesian posterior-derived opponent range
```

Everything else remains passive and equivalent in intent: check when free; when facing a wager, call if equity meets integer pot odds, otherwise fold. The policy never voluntarily bets, raises, or jams.

CBet, FoldToCBet, and postflop aggressive-share posteriors continue to be learned and persisted, but they are **action-inactive in v1**. This prevents a first adaptive experiment from mixing range adaptation with additional behavioral heuristics.

## 30.2 Frozen dependencies

```text
adaptive-learning-protocol-v1
  payload hash 03f8ab5b2b35c86910d46df1d5187c306814e427b3f514eb85748a9a424adf5c

bayesian-opponent-model-v1

posterior-to-range-v1
  payload hash af19b040f9049ecd3a3a12615dcbe049e1d1f859546681487176238e9d2ed668

preflop-range-strength-reference-v1
  payload hash cb110768ad74a9747064ba25e88af635055b1491a58deae8f09112aae801fc1e

hand-strength-baseline-v1
hand-strength-pot-odds-v1
```

The adaptive policy requires the `Beta(1,1)` prior vector frozen by `adaptive-learning-protocol-v1`. A different prior vector requires a new policy/experiment version.

## 30.3 Range-conditioned equity

Version:

```text
posterior-range-equity-v1
```

Given the current `OpponentRangeSnapshot`, known hero cards, and visible board cards:

1. apply blockers to the selected range exactly as frozen by `posterior-to-range-v1`;
2. assign each surviving opponent combination equal probability;
3. conditional on an opponent combination, assign every legal future board runout equal probability.

If the live opponent range contains $K$ combinations and $b$ future board cards remain, with $R$ cards available after removing hero-visible cards, the joint outcome count is

$$
J = K\binom{R-2}{b}.
$$

The exact/Monte-Carlo boundary remains:

```text
J <= 500000  -> exact
J >  500000  -> Monte Carlo
```

Monte Carlo uses exactly 10,000 samples.

For each sample:

1. sample one live opponent combination uniformly;
2. remove that combination from the remaining deck;
3. sample the future board uniformly without replacement;
4. evaluate showdown with the production hand evaluator.

Ties contribute one half. Equity is persisted in integer ppm using nearest-half-up rounding.

## 30.4 Equity RNG

Version:

```text
posterior-range-equity-rng-v1
```

The domain-separated seed identity is

```text
(master_seed, bot_id, hand_id, decision_index)
```

under the `posterior-range-equity-rng-v1` namespace.

The range hash is deliberately **not** part of the RNG identity. This preserves the same decision-indexed random source when the inferred range changes and avoids coupling RNG identity to model-state serialization.

Range-conditioned equity consumes no `AgentRNG` action draws. Bayesian posterior sampling is not used.

## 30.5 Pot odds

Pot-odds semantics are the same as `hand-strength-pot-odds-v1`.

When facing a wager:

$$
C = \min(B_{opp}-B_{hero}, S_{hero}),
$$

$$
PotOdds = \frac{C}{Pot+C}.
$$

Pot odds are converted to integer ppm with nearest-half-up rounding.

## 30.6 Action policy

The exact action precedence is:

```text
No wager faced:
    CHECK

Wager faced:
    construct posterior-to-range-v1 snapshot
    compute posterior-range-equity-v1

    if equity_ppm >= pot_odds_ppm and CALL is legal:
        CALL
    else if FOLD is legal:
        FOLD
    else:
        deterministic policy error
```

The policy never substitutes BET/RAISE/ALL_IN when its frozen passive action is unavailable.

The threshold is inclusive: equality calls.

## 30.7 Public-history range signal

The current preflop range signal is derived only from `public-action-trace-v1` using `posterior-to-range-v1`:

```text
uninformative    -> full range
passive entry    -> VPIP-derived range
first raise      -> PFR-derived range
qualifying 3-bet -> 3Bet-derived range
```

Postflop actions do not further narrow the v1 card range.

## 30.8 Action-inactive learned metrics

The following Bayesian posteriors do not modify card-range membership or action thresholds in v1:

```text
CBet
FoldToCBet
postflop aggressive share
```

They remain observable model state for later separately versioned policies.

## 30.9 Replay and provenance

Each adaptive decision must be able to persist or reconstruct at least:

```text
model_state_hash
preflop range signal
range_hash
range_width_ppm
selected range-combination count
live range-combination count after blockers
range-conditioned equity method/result/sample count
pot_odds_ppm when applicable
chosen action
decision_hash
```

The decision hash is canonical SHA-256 over the scientific decision payload and version identities.

## 30.10 Scope boundary

This specification freezes a pure decision mechanism given a current Bayesian model state. It does **not** yet freeze or implement the runner-facing terminal-hand update hook required to advance that state between hands.

That integration is the next dependency and must respect Spec 28's update timing, mirrored-run isolation, trajectory reset, and replay hash-chain requirements.

## 30.11 Non-goals

The following are explicitly outside v1:

- mid-hand Bayesian updates;
- postflop action-conditioned hidden-card likelihoods;
- CBet/FoldToCBet/aggression action-threshold adjustments;
- adaptive betting or raising;
- MCTS;
- H4 sample-size/power/precision design;
- opponent-model state sharing between mirrored positional runs.

## 30.12 Machine-readable freeze

The normative machine-readable companion is:

```text
configs/experiments/phase2-adaptive-hand-strength-policy-v1.json
```

Its canonical payload SHA-256 is:

```text
76b6c561c2cb3715cc6352520dfbd3b0b01ce219a1b37d1a0acb62bb6ce10e4a
```
