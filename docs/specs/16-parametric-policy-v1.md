# Frozen Parametric Policy Mechanism v1

## Status and Authority

This file freezes the baseline policy mechanism $\phi$ that maps a behavioural target vector $\theta$ to an executable parametric opponent policy.

The authoritative version identifiers are:

```text
parametric_policy_version        = parametric-policy-v1
agent_observation_version        = agent-observation-v2
public_action_trace_version      = public-action-trace-v1
equity_version                   = uniform-random-equity-v1
equity_range_version             = uniform-random-range-v1
strength_reference_version       = policy-strength-reference-v1
gate_map_version                 = policy-strength-gates-v1
action_sizing_version            = strength-sizing-v1
calibration_environment_version  = policy-calibration-environment-v1
```

For `parametric-policy-v1`, this file is the normative definition of $\phi$. It supersedes any earlier generic description of the policy mechanism that does not specify an exact mapping.

The policy applies only to the baseline heads-up NLHE environment defined in `02-game-environment.md`.

---

## 1. Scientific Role

A static parametric opponent is defined by:

$$
(\theta,\phi)
$$

where:

- $\theta$ contains behavioural target parameters;
- $\phi$ is the fixed mechanism defined by this file.

For the initial profile schema:

$$
\theta =
[
VPIP,
PFR,
3BetFreq,
CBetFreq,
Aggression,
FoldToCBet
].
$$

The mechanism is deliberately simple and transparent. Its purpose is to create reproducible benchmark opponents with measurable behavioural targets. It is not intended to approximate optimal poker strategy.

`parametric-policy-v1` is a **static policy**:

- it maintains no opponent model;
- it maintains no counters across hands;
- it does not adapt its target parameters;
- `reset()` does not change strategic parameters;
- it never receives the opponent's true $\theta$;
- its only game input is the agent-visible public/private observation defined below.

A deterministic or degenerate distribution remains a valid policy $\pi(a|s,\theta)$. The only action randomness in v1 is the explicitly defined boundary tie-breaking draw.

---

## 2. Required Public Observation Contract

`parametric-policy-v1` requires:

```text
agent-observation-v2
```

The existing hidden-information boundary remains mandatory. The observation must contain only information available to the acting player.

In addition to the existing observation fields, `agent-observation-v2` adds:

```text
dealer: int
public_action_history: tuple[PublicActionEvent, ...]
```

`dealer` is public table-position information.

The public action trace uses:

```text
public-action-trace-v1
```

with the immutable logical record:

```text
PublicActionEvent
    actor: int
    street: Street
    action: Action
    wager_increase: bool
```

`wager_increase` records the effect of the executed engine action:

- `True` if the action increased the current wager faced by the opponent;
- `False` otherwise.

Therefore an `ALL_IN` is represented according to its actual betting effect rather than its action name.

The public action history contains only already executed actions. It must not contain:

- opponent hole cards;
- unrevealed board cards;
- deck order;
- future chance outcomes;
- hidden opponent RNG state;
- hidden policy parameters.

The existing simple `action_history` may remain for compatibility, but `parametric-policy-v1` uses `public_action_history` for opportunity classification.

---

## 3. Decision Event and RNG Consumption

Every call to `act()` is one policy decision event.

Let the incoming action RNG expose the next event index:

```text
decision_index = rng.decision_index
```

The policy consumes **exactly one action-RNG event per `act()` call**, regardless of street, branch, legal-action count, or whether the equity calculation is exact or Monte Carlo:

```text
decision_draw_ppm = rng.randint(0, 999_999)
```

This draw is used only for calibrated threshold-tie handling.

No other policy action draw is permitted in `parametric-policy-v1`.

The equity subsystem uses a separate domain-separated RNG namespace and must not advance the action-RNG stream.

This rule ensures that:

- action randomness is event-indexed;
- variable equity-computation cost cannot shift later action randomness;
- paired position-compensation runs preserve the intended RNG isolation.

---

## 4. Uniform Random Equity v1

### 4.1 Range

The range version is:

```text
uniform-random-range-v1
```

The opponent range is the uniform distribution over every legal two-card opponent holding consistent with the acting player's observation.

The range is **not conditioned on opponent actions** in v1.

This is an intentional baseline simplification. Action-conditioned and learned ranges require a future policy-version change.

### 4.2 Candidate Deck

The candidate deck is:

```text
full 52-card deck
minus hero hole cards
minus currently visible board cards
```

The actual opponent hole cards and unrevealed board cards from the true engine state must never be used.

### 4.3 Equity Definition

For a legal opponent holding and board completion, the final showdown result is evaluated with the benchmark hand evaluator.

Equity is:

$$
E = P(\text{win}) + \frac{1}{2}P(\text{tie}).
$$

The integer comparison representation is:

```text
equity_ppm in [0, 1_000_000]
```

If the evaluation contains $W$ wins, $T$ ties, and $N$ total outcomes, define:

$$
n = 2W + T
$$

and:

$$
d = 2N.
$$

Then:

$$
equity_{ppm}
=
\left\lfloor
\frac{n\cdot 1{,}000{,}000 + \lfloor d/2 \rfloor}{d}
\right\rfloor.
$$

Policy comparisons use `equity_ppm`, not binary floating-point equity.

### 4.4 Exact vs. Monte Carlo

Let:

- $R$ be the number of candidate cards;
- $b$ be the number of unrevealed board cards.

The number of compatible joint hidden-state/runout outcomes is:

$$
K =
\binom{R}{2}
\binom{R-2}{b}.
$$

For `parametric-policy-v1`, this joint-outcome definition is the precise interpretation of the earlier generic exact-enumeration threshold.

If:

$$
K \le 500{,}000,
$$

all compatible opponent holdings and board completions are enumerated exactly.

If:

$$
K > 500{,}000,
$$

Monte Carlo is used with exactly:

```text
equity_samples = 10_000
```

trials.

Each Monte Carlo trial:

1. samples two opponent cards uniformly without replacement from the candidate deck;
2. samples the required future board cards uniformly without replacement from the remaining candidate cards;
3. evaluates the resulting showdown.

Trials are independent at the joint-outcome level.

### 4.5 Equity RNG

Monte Carlo uses a local deterministic RNG seeded from:

```text
master_seed
bot_id
hand_id
decision_index
purpose = "equity"
equity_version
```

through SHA-256 domain separation.

The benchmark action RNG object is not consumed by the equity estimator.

Python version, engine version, RNG version, equity version, and `equity_samples` remain persisted experiment metadata.

---

## 5. Strength Reference and Quantile Gates

The mechanism converts target frequencies into strength-ordered decision gates.

The reference-strength dataset is:

```text
policy-strength-reference-v1
```

and is generated only from the fixed fitting scenarios in `06-policy-calibration.md`.

For every reference opportunity, the policy computes `equity_ppm`.

### 5.1 Gate Representation

A calibrated gate is:

```text
direction: high | low
threshold_ppm: int
tie_probability_ppm: int
```

where:

```text
0 <= threshold_ppm <= 1_000_000
0 <= tie_probability_ppm <= 1_000_000
```

A high-tail gate passes if:

```text
equity_ppm > threshold_ppm
```

or, at equality:

```text
equity_ppm == threshold_ppm
and
decision_draw_ppm < tie_probability_ppm
```

A low-tail gate passes if:

```text
equity_ppm < threshold_ppm
```

or the same equality/tie rule.

The following are high-tail gates:

```text
VPIP
PFR
3BetFreq
CBetFreq
Aggression
```

`FoldToCBet` is a low-tail gate.

Thus stronger states are preferentially selected for participation and aggression, while weaker states are preferentially selected for folding to continuation bets.

### 5.2 Gate Construction

For a high-tail target probability $t$ and a reference multiset of $N$ strength values, choose a threshold value $x$ satisfying:

$$
A_x \le tN \le A_x + Q_x
$$

where:

- $A_x$ is the number of reference strengths strictly greater than $x$;
- $Q_x$ is the number equal to $x$.

Then:

$$
\rho =
\frac{tN-A_x}{Q_x}.
$$

The stored tie probability is $\rho$ rounded to integer parts per million using round-half-up.

For a low-tail target, the same construction is used with counts strictly below the threshold.

If more than one threshold value satisfies the inequality because $tN$ lies exactly on a boundary between two distinct strength groups, the serialized representation is made unique by scanning from the selected tail and choosing the first satisfying threshold: distinct strengths are inspected in strictly descending order for a high-tail gate and in strictly ascending order for a low-tail gate. At an exact interior boundary this yields `tie_probability_ppm = 1_000_000` for the selected boundary group.

Boundary targets are represented as:

```text
high target 0: threshold=1_000_000, tie=0
high target 1: threshold=0,         tie=1_000_000

low target 0:  threshold=0,         tie=0
low target 1:  threshold=1_000_000, tie=1_000_000
```

### 5.3 Nested VPIP/PFR Gate

VPIP and PFR use the same preflop opening reference distribution.

Because the parameter schema requires:

$$
0 \le PFR \le VPIP \le 1,
$$

the PFR high-tail gate must be at least as selective as the VPIP high-tail gate.

If both targets fall on the same strength threshold, the PFR tie probability must not exceed the VPIP tie probability.

The same `decision_draw_ppm` is used for both gates at a decision. Therefore every PFR event at the opening decision is also a VPIP event.

### 5.4 Aggression Ratio Conversion

For a specified aggression target:

$$
Aggression=\frac{A}{P},
$$

the high-tail aggressive-action share is:

$$
q_A = \frac{Aggression}{1+Aggression}.
$$

The aggression gate is calibrated to $q_A$, not directly to the raw ratio.

If Aggression is explicitly `don't-care`, as for the Calling Station profile, generic postflop aggression is disabled:

```text
generic_aggression_share = 0
```

This is a policy-mechanism fallback, not an acceptance target. Qualifying CBet behaviour remains controlled by `CBetFreq`.

---

## 6. Public Opportunity Classification

For the acting player `hero`:

```text
opponent = 1 - hero
faces_wager =
    current_bets[opponent] > current_bets[hero]
```

A legal wager-increasing action exists if the legal action set contains:

- a legal `BET_*` action; or
- `ALL_IN` and `current_bets[hero] + stacks[hero] > current_bets[opponent]`.

A non-increasing all-in is classified as a passive all-in call.

### 6.1 Preflop Raise Count

Define:

```text
preflop_increases =
    all public_action_history events
    with street == PREFLOP
    and wager_increase == True
```

### 6.2 Qualifying 3-bet Opportunity

The current decision is a qualifying 3-bet opportunity if and only if:

1. the street is preflop;
2. `faces_wager` is true;
3. `len(preflop_increases) == 1`;
4. the sole prior wager-increase actor is the opponent; and
5. a legal wager-increasing action exists.

This classification is identical in meaning to `calibration-opportunities-v2`.

### 6.3 Final Preflop Aggressor

The final preflop aggressor is the actor of the last preflop public action event with:

```text
wager_increase == True
```

If there is no such event, there is no final preflop aggressor.

### 6.4 Qualifying CBet Opportunity

The current decision is a qualifying CBet opportunity if and only if:

1. the street is the flop;
2. hero is the final preflop aggressor;
3. hero has not yet acted on the flop;
4. no opponent flop action before the current decision has `wager_increase == True`; and
5. a legal wager-increasing action exists.

An opponent check before hero acts does not remove the opportunity.

### 6.5 Directly Facing a Qualifying CBet

Hero directly faces a qualifying CBet if:

1. the opponent was the final preflop aggressor;
2. the opponent's first flop action is a wager-increasing action;
3. no hero wager-increasing flop action occurred before it;
4. the qualifying CBet is the most recent executed public action;
5. hero currently faces a wager.

---

## 7. Pot Odds and Call-Continuation Rule

When hero faces a wager:

$$
C =
\min(
current\_bets_{opp} - current\_bets_{hero},
stack_{hero}
).
$$

Let the current pot be $P$.

For $C>0$:

$$
PotOdds =
\frac{C}{P+C}.
$$

The integer comparison value is rounded to parts per million with the same round-half-up rule used for equity.

A generic call is permitted by the policy only if:

```text
equity_ppm >= pot_odds_ppm
```

unless a higher-precedence target rule such as FoldToCBet explicitly controls the fold/continue decision.

---

## 8. Deterministic Decision Precedence

The following precedence is normative.

No later branch may override an earlier matching branch.

### 8.1 Preflop — Qualifying 3-bet Opportunity

If the current decision is a qualifying 3-bet opportunity:

1. if the `3BetFreq` high-tail gate passes and an aggressive action is legal, choose an aggressive action;
2. otherwise, if the VPIP high-tail gate passes, a call-equivalent passive action is legal, and `equity_ppm >= pot_odds_ppm`, choose the passive call-equivalent action;
3. otherwise fold.

### 8.1.1 Preflop — Single Raise Without a Legal Re-raise

If exactly one prior preflop wager-increase event exists, that event was made by the opponent, hero currently faces the wager, but no legal wager-increasing action exists, the decision is not a qualifying 3-bet opportunity.

The policy then:

1. continues only if the VPIP high-tail gate passes, a passive call-equivalent action exists, and `equity_ppm >= pot_odds_ppm`;
2. otherwise folds.

A legal non-wager-increasing `ALL_IN` is the passive call-equivalent action when the remaining stack is smaller than the full call amount.

This branch makes the preflop policy total for short-stack responses without incorrectly counting them as 3-bet opportunities.

### 8.2 Preflop — No Prior Voluntary Raise

If there has been no preflop wager-increase event:

1. if the PFR high-tail gate passes and an aggressive action is legal, choose an aggressive action;
2. otherwise, if the VPIP high-tail gate passes and `CALL` is legal, call;
3. otherwise, if `CHECK` is legal, check;
4. otherwise fold.

For the baseline Button/SB opening decision, this produces nested PFR/VPIP behaviour.

For the Big Blind after a limp, no passive voluntary contribution is available; the policy raises only through the PFR branch and otherwise checks.

### 8.3 Preflop — Higher-Order Re-raise Situations

If two or more preflop wager-increase events have already occurred, `parametric-policy-v1` does not initiate a 4-bet or higher re-raise.

It:

1. calls only if the VPIP high-tail gate passes and `equity_ppm >= pot_odds_ppm`;
2. otherwise folds.

This limitation is deliberate and versioned. A policy that parameterises 4-bet or higher-order aggression requires a future policy version.

### 8.4 Postflop — Direct FoldToCBet Response

If hero directly faces a qualifying CBet:

1. if the `FoldToCBet` low-tail gate passes, fold;
2. otherwise, if a specified Aggression gate passes and an aggressive action is legal, choose an aggressive action;
3. otherwise choose the call-equivalent passive action.

For a profile with Aggression=`don't-care`, step 2 is disabled.

### 8.5 Postflop — Qualifying CBet Opportunity

If hero has a qualifying CBet opportunity:

1. if the `CBetFreq` high-tail gate passes, choose an aggressive action;
2. otherwise check.

### 8.6 Postflop — Other Decision Facing a Wager

For every other postflop decision facing a wager:

1. if `equity_ppm < pot_odds_ppm`, fold;
2. otherwise, if the Aggression high-tail gate passes and an aggressive action is legal, choose an aggressive action;
3. otherwise choose the call-equivalent passive action.

For Aggression=`don't-care`, step 2 is disabled.

### 8.7 Postflop — Other Decision Not Facing a Wager

For every other postflop decision not facing a wager:

1. if the Aggression high-tail gate passes and an aggressive action is legal, choose an aggressive action;
2. otherwise check.

For Aggression=`don't-care`, step 1 is disabled.

---

## 9. Concrete Legal Action Selection

The policy must always return an action from the supplied legal action set.

This deterministic mapping is part of $\phi$ and is not an illegal-action substitution policy.

### 9.1 Aggressive Action Set

The ordered aggressive action ranks are:

```text
0  BET_33
1  BET_66
2  BET_100
3  BET_200
4  ALL_IN
```

`ALL_IN` belongs to the aggressive set only if its execution would increase the current wager.

### 9.2 Strength-Based Preferred Size

The preferred aggressive action is:

```text
equity_ppm < 600_000                 -> BET_33
600_000 <= equity_ppm < 750_000      -> BET_66
750_000 <= equity_ppm < 900_000      -> BET_100
900_000 <= equity_ppm < 980_000      -> BET_200
980_000 <= equity_ppm                -> ALL_IN
```

If the preferred action is legal and aggressive, it is chosen.

Otherwise choose the legal aggressive action with the smallest absolute rank distance from the preferred action.

If two legal actions have the same rank distance, choose the smaller rank.

If no aggressive action exists, the policy falls through to the passive/decline rule for that decision branch.

### 9.3 Passive Call-Equivalent Action

When facing a wager, the passive call-equivalent action is:

1. `CALL` if legal;
2. otherwise a legal non-wager-increasing `ALL_IN`;
3. otherwise no passive call-equivalent action exists.

### 9.4 Passive No-Wager Action

When not facing a wager, the passive action is `CHECK`.

### 9.5 Decline Action

When a branch requires declining voluntary action:

1. choose `CHECK` if legal;
2. otherwise choose `FOLD` if legal;
3. otherwise choose the only legal passive call-equivalent action;
4. if none exists, raise a policy-decision error.

No silent engine-level action substitution is permitted.

---

## 10. Calibration Map Generation

A `PolicyCalibrationMap` is generated from:

```text
theta
parametric_policy_version
strength_reference_version
gate_map_version
equity_version
equity_range_version
```

It contains at least:

```text
vpip_gate
pfr_gate
three_bet_gate
cbet_gate
aggression_gate | disabled
fold_to_cbet_gate
reference_strength_hash
```

The map is immutable.

For static profiles, the profile name and `static-profiles-v2` version are also recorded.

The canonical JSON representation of the map receives a SHA-256 hash.

There is no profile-specific manual threshold tuning.

The same deterministic gate-construction algorithm is used for:

- static profiles;
- future continuous $\theta$ samples.

Changing a threshold formula, reference distribution, strength definition, action sizing rule, or branch precedence requires a new policy or gate-map version.

---

## 11. Persistence and Reproducibility

Every run using `parametric-policy-v1` must persist, directly or through the immutable experiment configuration:

```text
theta
profile_name, if applicable
profile_version, if applicable
parametric_policy_version
agent_observation_version
public_action_trace_version
equity_version
equity_range_version
equity_samples
strength_reference_version
reference_strength_hash
gate_map_version
gate_map_hash
action_sizing_version
calibration_environment_version
calibration_result_hash
```

A static benchmark profile may not be used if its required calibration result is absent or failing.

---

## 12. Mandatory Policy Tests Before Gate 6 Can Pass

Implementation tests must cover at least:

1. observation-v2 contains only public/hero-private information;
2. public action events classify wager-increasing `ALL_IN` correctly;
3. exactly one action RNG event is consumed per `act()` call;
4. equity Monte Carlo does not consume the action RNG;
5. same seed/state/version produces identical equity and action;
6. PFR implies VPIP at the opening decision;
7. a qualifying one-raise response uses the 3Bet gate;
8. 4-bet and higher opportunities are excluded from the 3Bet gate;
9. a prior flop check does not remove a CBet opportunity;
10. a prior flop wager increase removes the CBet opportunity;
11. direct FoldToCBet uses the low-strength gate;
12. generic postflop fold uses the pot-odds rule;
13. Calling Station generic aggression is disabled while CBet remains active;
14. strength-based bet sizing and legal-action fallback are deterministic;
15. no returned action is illegal;
16. all five `static-profiles-v2` profiles pass the frozen acceptance calibration environment;
17. changing the validation seed does not change the fitted gate map;
18. the acceptance/validation sample is never used to fit thresholds.

Until all required tests and calibration checks pass, Phase-0 policy-calibration Gate 6 remains open.

---

## 13. Explicit v1 Limitations

The following are deliberate limitations, not implementation gaps:

- opponent range is uniform and action-independent;
- no 4-bet or higher-order aggressive policy is parameterised;
- no turn/river continuation-bet statistic is modelled;
- no check-raise statistic is independently targeted;
- no donk-bet statistic is independently targeted;
- no opponent modeling is used;
- target acceptance is defined in the frozen calibration environment rather than claimed to be invariant under every possible opponent action distribution;
- Aggression=`don't-care` disables generic aggression rather than creating an unreported hidden target.

Any change to these scientific semantics requires an explicitly versioned successor to `parametric-policy-v1`.
