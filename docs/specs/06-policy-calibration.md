# Policy Calibration and Static Benchmark Profiles

## 7.2.1 Versioned Observable-Statistic Opportunity Definitions

The benchmark uses the following opportunity-definition version:

```text
calibration-opportunities-v2
```

All behavioural statistics are calculated from executed game traces. Policy target parameters and observed behavioural statistics remain separate quantities.

### VPIP

A VPIP opportunity is one dealt hand for the observed player.

A VPIP event occurs if the player voluntarily commits additional chips preflop beyond the mandatory blind contribution. A preflop call, bet, raise, or all-in contribution counts as voluntary participation. A check or fold does not.

Each hand contributes at most one VPIP event.

$$
VPIP = \frac{\text{hands with voluntary preflop contribution}}{\text{dealt-hand opportunities}}
$$

### PFR

A PFR opportunity is one dealt hand in which the player receives at least one preflop decision at which a legal action exists that would increase the current wager.

A PFR event occurs if the player makes at least one such aggressive preflop action during the hand.

PFR is calculated once per eligible hand rather than once per individual decision.

### 3BetFreq

A qualifying 3-bet opportunity occurs when:

1. exactly one voluntary preflop raise has previously occurred;
2. the player is facing that raise;
3. the prior raiser is the opponent; and
4. the player has a legal action that would increase the current wager.

A 3-bet event occurs when the player responds to that opportunity with a legal action that increases the current wager.

Later 4-bet and higher-order re-raise opportunities are not included in the `3BetFreq` denominator.

$$
3BetFreq = \frac{\text{qualifying 3-bet events}}{\text{qualifying 3-bet opportunities}}
$$

### CBetFreq

A player has a qualifying flop continuation-bet opportunity if:

1. the player was the final preflop aggressor;
2. the hand reaches the flop;
3. this is the player's first flop action;
4. no opponent bet or raise has occurred on the flop before that action; and
5. a legal action exists that would increase the current wager.

A CBet event occurs when the player makes such an aggressive action.

A prior opponent check does not remove the continuation-bet opportunity. A prior opponent bet or raise does.

$$
CBetFreq = \frac{\text{qualifying continuation bets}}{\text{qualifying continuation-bet opportunities}}
$$

### FoldToCBet

A FoldToCBet opportunity occurs when a player directly faces a qualifying continuation bet as defined above.

A FoldToCBet event occurs if the player folds in response to that qualifying continuation bet.

$$
FoldToCBet = \frac{\text{folds to qualifying continuation bets}}{\text{qualifying continuation bets faced}}
$$

### Aggression

Postflop aggression is defined as:

$$
Aggression = \frac{A}{P}
$$

where:

```text
A = postflop bets and raises
P = postflop calls and checks
```

Folds are excluded from the denominator.

An `ALL_IN` action is classified according to its betting effect:

- if it increases the current wager, it is aggressive;
- if it does not increase the current wager and functions as an all-in call, it is passive.

All opportunity definitions are stored with the calibration result and policy version.

---

### 7.3 Policy Calibration

Each parametric opponent policy must be calibrated before being used as a benchmark opponent.

For a parameter vector $\theta$, a longer calibration run is performed.

After that, at least the following quantities are compared:

$$
|\widehat{VPIP} - VPIP_{target}|
$$
$$
|\widehat{PFR} - PFR_{target}|
$$
$$
|\widehat{3Bet} - 3Bet_{target}|
$$
$$
|\widehat{CBetFreq} - CBetFreq_{target}|
$$
$$
|\widehat{Aggression} - Aggression_{target}|
$$
$$
|\widehat{FoldToCBet} - FoldToCBet_{target}|
$$

For each parameter, a pre‑defined tolerance $\epsilon$ is set.

**Tolerances:**
- VPIP: ±3 percentage points
- PFR: ±3 percentage points
- 3Bet: ±2 percentage points
- CBetFreq: ±5 percentage points
- Aggression: ±0.2
- FoldToCBet: ±5 percentage points

These values are practical benchmarks and not statistical significance thresholds. They may be varied in later ablations.

A policy may be used as a benchmark profile only if the relevant observed frequencies lie within these tolerances.

The calibration is stored and versioned as a separate test step.

---

## 7.3.1 Frozen Calibration Environment for `parametric-policy-v1`

The calibration environment version is:

```text
policy-calibration-environment-v1
```

This environment separates **gate fitting** from **acceptance validation**.

The validation sample must never be used to tune or fit the policy gates.

### Reference and Validation Sizes

For every designated metric scenario:

```text
fit opportunities        = 5_000
validation opportunities = 5_000
```

The fixed base seeds are:

```text
fit_base_seed        = 5_100_001
validation_base_seed = 5_100_002
```

Each scenario receives a domain-separated deal seed:

```text
SHA256(
    calibration_environment_version
    + ":"
    + base_seed
    + ":"
    + scenario_id
)
```

interpreted as an unsigned integer seed.

Cards are generated with the standard deterministic `DealSchedule`.

### Probe Equity RNG Identity

For every fit or validation probe, the designated policy decision uses the following frozen equity identity:

```text
equity master_seed = scenario deal seed
bot_id             = hero seat
hand_id            = opportunity index, 0..4,999
decision_index     = 0
```

The scripted prefix actions used only to construct a probe state are not policy decisions and do not advance the policy decision index. Each probe invokes the observed policy only at its single designated opportunity, so `decision_index = 0` is normative for the calibration environment.

The same identity rule is used for fit and validation. Because the fit and validation base seeds differ, their domain-separated scenario deal seeds—and therefore their equity Monte Carlo streams—also differ.

### Scenario A — `OPEN_BTN`

Observed player:

```text
hero seat = 0
dealer    = 0
```

The policy is invoked at the initial preflop Button/SB decision.

This scenario supplies the reference and validation opportunities for:

```text
VPIP
PFR
```

Every hand is one VPIP opportunity and one PFR opportunity.

### Scenario B — `THREE_BET_BB`

Observed player:

```text
hero seat = 1
dealer    = 0
```

Before hero is invoked, the probe script executes:

```text
seat 0: BET_200
```

Hero therefore directly faces exactly one voluntary preflop raise.

This scenario supplies only:

```text
3BetFreq
```

for policy gate fitting and acceptance.

### Scenario C — `CBET_BTN`

Observed player:

```text
hero seat = 0
dealer    = 0
```

The engine executes the legal probe prefix:

```text
seat 0: BET_200
seat 1: CALL
flop:
seat 1: CHECK
```

Hero is then invoked for its first flop action as final preflop aggressor.

This scenario supplies only:

```text
CBetFreq
```

for policy gate fitting and acceptance.

The scripted hero preflop raise exists solely to construct the public qualifying state and is not counted as an observed policy action.

### Scenario D — `FOLD_TO_CBET_BB`

Observed player:

```text
hero seat = 1
dealer    = 0
```

The engine executes the legal probe prefix:

```text
seat 0: BET_200
seat 1: CALL
flop:
seat 1: CHECK
seat 0: BET_66
```

Hero is then invoked directly facing the qualifying flop continuation bet.

This scenario supplies only:

```text
FoldToCBet
```

for policy gate fitting and acceptance.

The scripted hero call/check actions exist only to construct the qualifying state and are not counted as policy actions.

### Scenario E — `POSTFLOP_AGGRESSION_BTN`

Observed player:

```text
hero seat = 0
dealer    = 0
```

The engine executes:

```text
seat 0: CALL
seat 1: CHECK
flop:
seat 1: CHECK
```

Hero is then invoked in an unopened, non-CBet flop decision.

This scenario supplies only:

```text
Aggression
```

for gate fitting and acceptance.

Because the decision is not facing a wager, the passive action is `CHECK` and the aggressive action is a wager increase. Therefore the resulting aggressive share maps directly to $A/(A+P)$.

For Aggression=`don't-care`, the scenario is still run as a diagnostic, but it is excluded from profile acceptance.

### Metric Isolation

The calibration suite does **not** pool unrelated metric counts across the five probe scenarios.

The designated metric(s) from each scenario are used for the corresponding gate and acceptance test:

```text
OPEN_BTN                    -> VPIP, PFR
THREE_BET_BB                -> 3BetFreq
CBET_BTN                    -> CBetFreq
FOLD_TO_CBET_BB             -> FoldToCBet
POSTFLOP_AGGRESSION_BTN     -> Aggression
```

All event and opportunity classifications inside each scenario still use `calibration-opportunities-v2`.

### Fitting and Acceptance Separation

`policy-strength-reference-v1` is generated only from the fit scenarios and `fit_base_seed`.

For a target $\theta$, `policy-strength-gates-v1` is generated from that fixed reference distribution.

The fitted map is then frozen.

Acceptance is evaluated only on the independent validation scenarios and `validation_base_seed`.

No threshold, tie probability, sizing rule, or decision precedence may be changed in response to validation results without changing the relevant versioned mechanism and rerunning the full validation process.

### Acceptance

For every required metric:

1. the validation denominator must equal 5,000 designated opportunities;
2. the observed statistic must be defined;
3. the absolute deviation from target must be within the Section 7.3 tolerance.

The Calling Station Aggression result is reported but excluded because the profile explicitly declares Aggression=`don't-care`.

A calibration result records:

```text
calibration_environment_version
fit_base_seed
validation_base_seed
opportunities_per_metric
strength_reference_hash
gate_map_hash
observed metric values
numerators
denominators
tolerances
pass/fail per metric
overall pass/fail
```

### Interpretation Outside the Calibration Environment

The frozen calibration environment defines whether $(\theta,\phi)$ is accepted as a benchmark profile.

Ordinary benchmark matches continue to report `calibration-opportunities-v2` behavioural statistics over their complete executed traces as diagnostics.

Those ordinary-match statistics may differ from the calibration targets because the distribution of opportunities depends on the opponent and game trajectory. They do not retroactively invalidate a profile that passed the frozen calibration environment.

A claim that a target frequency is invariant across arbitrary opponent distributions is explicitly prohibited.

---

## 7.3.1 Frozen Calibration Environment for `parametric-policy-v1`

The calibration environment version is:

```text
policy-calibration-environment-v1
```

This environment separates **gate fitting** from **acceptance validation**.

The validation sample must never be used to tune or fit the policy gates.

### Reference and Validation Sizes

For every designated metric scenario:

```text
fit opportunities        = 5_000
validation opportunities = 5_000
```

The fixed base seeds are:

```text
fit_base_seed        = 5_100_001
validation_base_seed = 5_100_002
```

Each scenario receives a domain-separated deal seed:

```text
SHA256(
    calibration_environment_version
    + ":"
    + base_seed
    + ":"
    + scenario_id
)
```

interpreted as an unsigned integer seed.

Cards are generated with the standard deterministic `DealSchedule`.

### Scenario A — `OPEN_BTN`

Observed player:

```text
hero seat = 0
dealer    = 0
```

The policy is invoked at the initial preflop Button/SB decision.

This scenario supplies the reference and validation opportunities for:

```text
VPIP
PFR
```

Every hand is one VPIP opportunity and one PFR opportunity.

### Scenario B — `THREE_BET_BB`

Observed player:

```text
hero seat = 1
dealer    = 0
```

Before hero is invoked, the probe script executes:

```text
seat 0: BET_200
```

Hero therefore directly faces exactly one voluntary preflop raise.

This scenario supplies only:

```text
3BetFreq
```

for policy gate fitting and acceptance.

### Scenario C — `CBET_BTN`

Observed player:

```text
hero seat = 0
dealer    = 0
```

The engine executes the legal probe prefix:

```text
seat 0: BET_200
seat 1: CALL
flop:
seat 1: CHECK
```

Hero is then invoked for its first flop action as final preflop aggressor.

This scenario supplies only:

```text
CBetFreq
```

for policy gate fitting and acceptance.

The scripted hero preflop raise exists solely to construct the public qualifying state and is not counted as an observed policy action.

### Scenario D — `FOLD_TO_CBET_BB`

Observed player:

```text
hero seat = 1
dealer    = 0
```

The engine executes the legal probe prefix:

```text
seat 0: BET_200
seat 1: CALL
flop:
seat 1: CHECK
seat 0: BET_66
```

Hero is then invoked directly facing the qualifying flop continuation bet.

This scenario supplies only:

```text
FoldToCBet
```

for policy gate fitting and acceptance.

The scripted hero call/check actions exist only to construct the qualifying state and are not counted as policy actions.

### Scenario E — `POSTFLOP_AGGRESSION_BTN`

Observed player:

```text
hero seat = 0
dealer    = 0
```

The engine executes:

```text
seat 0: CALL
seat 1: CHECK
flop:
seat 1: CHECK
```

Hero is then invoked in an unopened, non-CBet flop decision.

This scenario supplies only:

```text
Aggression
```

for gate fitting and acceptance.

Because the decision is not facing a wager, the passive action is `CHECK` and the aggressive action is a wager increase. Therefore the resulting aggressive share maps directly to $A/(A+P)$.

For Aggression=`don't-care`, the scenario is still run as a diagnostic, but it is excluded from profile acceptance.

### Metric Isolation

The calibration suite does **not** pool unrelated metric counts across the five probe scenarios.

The designated metric(s) from each scenario are used for the corresponding gate and acceptance test:

```text
OPEN_BTN                    -> VPIP, PFR
THREE_BET_BB                -> 3BetFreq
CBET_BTN                    -> CBetFreq
FOLD_TO_CBET_BB             -> FoldToCBet
POSTFLOP_AGGRESSION_BTN     -> Aggression
```

All event and opportunity classifications inside each scenario still use `calibration-opportunities-v2`.

### Fitting and Acceptance Separation

`policy-strength-reference-v1` is generated only from the fit scenarios and `fit_base_seed`.

For a target $\theta$, `policy-strength-gates-v1` is generated from that fixed reference distribution.

The fitted map is then frozen.

Acceptance is evaluated only on the independent validation scenarios and `validation_base_seed`.

No threshold, tie probability, sizing rule, or decision precedence may be changed in response to validation results without changing the relevant versioned mechanism and rerunning the full validation process.

### Acceptance

For every required metric:

1. the validation denominator must equal 5,000 designated opportunities;
2. the observed statistic must be defined;
3. the absolute deviation from target must be within the Section 7.3 tolerance.

The Calling Station Aggression result is reported but excluded because the profile explicitly declares Aggression=`don't-care`.

A calibration result records:

```text
calibration_environment_version
fit_base_seed
validation_base_seed
opportunities_per_metric
strength_reference_hash
gate_map_hash
observed metric values
numerators
denominators
tolerances
pass/fail per metric
overall pass/fail
```

### Interpretation Outside the Calibration Environment

The frozen calibration environment defines whether $(\theta,\phi)$ is accepted as a benchmark profile.

Ordinary benchmark matches continue to report `calibration-opportunities-v2` behavioural statistics over their complete executed traces as diagnostics.

Those ordinary-match statistics may differ from the calibration targets because the distribution of opportunities depends on the opponent and game trajectory. They do not retroactively invalidate a profile that passed the frozen calibration environment.

A claim that a target frequency is invariant across arbitrary opponent distributions is explicitly prohibited.

---

## 7.5 Static Profiles

Static profile schema version:

```text
static-profiles-v2
```

The initial benchmark profiles are:

| Profile | VPIP | PFR | 3BetFreq | CBetFreq | Aggression | FoldToCBet |
|---|---:|---:|---:|---:|---:|---:|
| Nit / Tight-Passive | 10% | 4% | 2% | 45% | 0.3 | 55% |
| TAG / Tight-Aggressive | 22% | 18% | 7% | 65% | 0.8 | 45% |
| LAG / Loose-Aggressive | 32% | 26% | 10% | 70% | 0.9 | 38% |
| Calling Station | 50% | 5% | 2% | 35% | don't-care | 15% |
| Maniac | 55% | 45% | 18% | 80% | 1.5 | 25% |

These values are synthetic benchmark targets. They are **not empirically validated real-player population estimates**.

A later realistic population must be based on a separate empirical data foundation.

The aggression values refer to the ratio of aggressive actions to passive actions. A value of 1.0 means the agent acts aggressively in 50% of situations when $A+P$ is the complete denominator. A value of 1.5 corresponds to a 1.5:1 ratio (60% aggressive), and a value of 0.3 corresponds to a 0.3:1 ratio (approximately 23% aggressive).

Every specified metric is subject to the Section 7.3 calibration tolerance:

```text
VPIP          ±0.03
PFR           ±0.03
3BetFreq      ±0.02
CBetFreq      ±0.05
Aggression    ±0.20
FoldToCBet    ±0.05
```

A `don't-care` value is excluded from the calibration acceptance criterion and must be explicitly declared. Missing values are not implicitly interpreted as don't-care.

The Calling Station profile deliberately leaves Aggression unspecified because its defining postflop acceptance target is `FoldToCBet` rather than a fixed aggression ratio.

The stated values are target parameters and must be validated by an actual calibration run before the profile is used as a benchmark opponent.
