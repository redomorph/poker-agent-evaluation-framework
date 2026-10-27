# Opponent Modelling and Parametric Strategy Space

## 7. Opponent Modelling: The Parametric Strategy Space

Instead of exclusively using hand‑crafted opponents, a **parameterisable policy agent** is implemented.

The opponent is described by a parameter vector

$$
\theta = [VPIP, PFR, 3BetFreq, CBetFreq, Aggression, FoldToCBet, \ldots]
$$

The parameter schema enforces structural constraints. At minimum:

$$0 \le PFR \le VPIP \le 1$$

and all frequency parameters are bounded to $[0,1]$. `3BetFreq` is defined over explicitly eligible 3-bet opportunities, and `FoldToCBet` only over situations in which a qualifying continuation bet is actually faced. `CBetFreq` is the frequency with which the agent makes a qualifying continuation bet when such an opportunity exists. Impossible or undefined parameter combinations are rejected before calibration.

The parameters alone, however, do not yet constitute a strategy.

All observable statistics use explicitly versioned opportunity definitions. VPIP uses dealt-hand opportunities as defined by the benchmark rules; PFR uses eligible preflop opportunities; 3Bet uses eligible responses to a qualifying prior raise; FoldToCBet uses hands in which a continuation bet is actually faced; Aggression uses the defined post-flop decision denominator. These denominators are stored with the policy version.

Therefore, a policy function is defined:

$$
\pi(a|s,\theta)
$$

It describes how, from a game state $s$ and the parameter vector $\theta$, a concrete action $a$ is generated.

---

### 7.1 Policy Parameters vs. Observable Frequencies

An important distinction is made explicitly:

**Policy parameters** are target quantities of the parametric policy.

**Observable statistics** are the frequencies actually measured during a match.

For example:

$$
VPIP_{target} \text{ and } \widehat{VPIP}_{observed}
$$

are not identical, but must be mapped as closely as possible by the policy.

The same applies to:

- PFR
- 3‑bet frequency
- CBet frequency
- Aggression
- Fold‑to‑CBet

---

### 7.2 Concrete Implementation of the Policy Function

The policy function is implemented as a heuristic model based on a hand‑strength evaluation.

For each hand, an equity against a defined range is computed. The equity subsystem is versioned and explicitly records:

```text
equity_method       = exact | monte_carlo
equity_samples      = N (required for Monte Carlo)
equity_range        = versioned range definition
range_representation = versioned preflop/postflop representation
rng_scheme           = deterministic and event-indexed where sampling is used
```

Phase 1 uses the simplest exact method available for the defined situation; Monte Carlo is used only where exact enumeration is not practical and its sample count and RNG configuration are part of the agent version. Exact enumeration is used when the number of remaining board runouts does not exceed 500,000; otherwise Monte Carlo with `equity_samples = 10,000` is used. The range representation is a versioned policy interface rather than an implicit implementation detail. Detailed dynamic preflop range models are deferred to the later opponent-modeling phase.

The parameters $\theta$ then do not directly control an alleged "meaning" of the metric, but rather the **target frequencies or action probabilities**.

For example:

- `VPIP_target` = desired long‑term preflop involvement frequency
- `PFR_target` = desired long‑term preflop raise frequency
- `3Bet_target` = desired long‑term 3‑bet frequency
- `CBetFreq_target` = desired long‑term qualifying continuation-bet frequency
- `Aggression_target` = defined target for aggressive actions
- `FoldToCBet_target` = probability of folding to a continuation bet

The hand‑strength evaluation determines the relative strength of a situation. A calibration mechanism translates this strength together with the target parameter into an action.

This avoids, for example, incorrectly defining VPIP as a mere equity threshold.

### 7.2.2 Frozen Baseline Policy Mechanism

The previously generic mechanism $\phi$ is frozen for the baseline as:

```text
parametric-policy-v1
```

Its complete normative definition is in `16-parametric-policy-v1.md`.

For this version, that file defines the exact observation contract, equity estimator, RNG consumption, strength-gate construction, decision precedence, legal action selection, bet sizing, persistence metadata, and explicit limitations.

The precise `parametric-policy-v1` joint hidden-state/runout threshold in `16-parametric-policy-v1.md` supersedes any less-specific interpretation of the generic exact-enumeration sentence above.

No implementation of `ParametricPolicyAgent` may precede the tests required by that frozen mechanism.

**Aggression** is defined as the ratio of aggressive actions (bet, raise) to passive actions (call, check):

$$Aggression=\frac{A}{P}.$$

This is a ratio, not an aggressive-action percentage. The corresponding aggressive-action share is $A/(A+P)$. Thus 0.8 corresponds to 44.4%, 1.0 to 50%, and 1.5 to 60% when $A+P$ is the complete denominator. The denominator and treatment of folds are fixed in the policy specification. 3Bet is the proportion of explicitly defined eligible preflop 3-bet opportunities in which the player re-raises.

---

> Policy calibration, versioned opportunity definitions, and static benchmark profiles are specified in `06-policy-calibration.md`.

### 7.4 Considered State Information

The policy takes into account, among other things:

- Position
- Street
- Hand strength
- Range
- Board texture
- Pot odds
- Stack‑to‑pot ratio
- Previous action history
- Parameter vector $\theta$

Thus, the mapping from the parameter vector to a reproducible playing agent is defined by $(\theta,\phi)$, where $\theta$ contains behavioural target parameters and $\phi$ denotes the versioned policy mechanism. Behavioural statistics alone do not uniquely identify a strategy.

---

### 7.6 Population Testing

For robustness, the opponent parameter vector is drawn randomly from a defined distribution.

For example:

```text
VPIP        ~ Beta(2,5)
Aggression  ~ Gamma(2,1)
```

The full sampling distribution is documented for each parameter.

A distinction is made between two concepts:

$$
\Theta = \text{possible parameter space}
$$

and $\theta \sim D(\Theta)$ where $D$ is the actual sampling distribution used.

The robustness analysis thus measures performance over a **distribution of plausible opponents**, not the mathematical worst case over the entire parameter space.

---

### 7.7 Population Scenarios

To avoid results depending on a single arbitrary opponent distribution, at least four populations are distinguished:

#### 1. Balanced Population

Equally weighted mixture of the five defined static profiles: $P(profile_i) = 0.2$

This population does **not** represent a uniform distribution over the continuous parameter space.

#### 2. Baseline Synthetic Population

A deliberately defined synthetic benchmark distribution:

- Nit: 10%
- TAG: 35%
- LAG: 30%
- Calling Station: 15%
- Maniac: 10%

This weighting does **not** represent an empirically validated real player population.

It serves as a reproducible reference population.

#### 3. Stress Population

The former "Adversarial Population" is now called **Stress Population**.

It is not adapted to the agent being tested, but contains a priori defined, challenging parameter combinations, for example:

- high VPIP (40‑60%) + high aggression (1.5‑2.0)
- very low fold‑to‑CBet values (0‑10%) + high aggression
- further explicitly documented extreme profiles

Since the population is not agent‑specifically optimised, it is not interpreted as a true adversarial best response.

#### 4. Randomized Population

The Randomized Population draws $\theta$ from a pre‑defined continuous distribution.

It serves to test robustness not only against five discrete archetypes.

The population used is always stored together with the experiment.

---

### 7.8 Development, Validation, and Holdout Populations

To avoid overfitting, populations are separated by their use:

**Development Population**
- may be inspected during development
- may be used for debugging and initial parameter tuning

**Validation Population**
- may be used for hyperparameter selection and model comparison
- may not be used for final result interpretation

**Holdout / Test Population**
- fixed before the final benchmark
- may not be used for optimisation during agent development
- serves as the final generalisation assessment

For agents without training, e.g., a pure random or hand‑strength agent, the term "training population" is not relevant. In this case, the Validation Population is used as the primary reference population, and the Holdout Population serves solely for final generalisation measurement.

---

### 7.9 Opponent Modeling

The bot receives no direct information about the drawn opponent parameter.

It observes only the opponent's actions and updates its beliefs from them.

Instead of using a single point estimate after a few hands, a Bayesian model is provided.

For example: $VPIP \sim Beta(\alpha,\beta)$

After observing $x$ VPIP events in $n$ relevant situations, we have: $VPIP|Data \sim Beta(\alpha+x, \beta+n-x)$

Thus, the agent possesses not only an estimate but also an uncertainty distribution.

The first 20 hands are examined as an **Early‑Adaptation Phase**, but are not assumed to provide an accurate estimate of the long‑term opponent parameters.

---

### 7.10 Segmented EV Computation

The EV computation is performed in three segments:

- (a) Early‑Adaptation Phase: hands 1‑20
- (b) Main‑Adaptation Phase: hands 21‑100
- (c) Steady‑State Phase: hands 101‑500

Results are reported separately for each segment to visualise the convergence of opponent modeling.

The primary EV metric is computed over all hands, but the segmentation enables analysis of learning progress.

Performance is investigated over several adaptation horizons:

- 10 hands
- 20 hands
- 50 hands
- 100 hands
- 500 hands

This allows measuring how quickly an agent actually benefits from opponent modeling.

---

### 7.11 Stationarity Assumption

In the baseline experiment, it is assumed that the opponent has a constant parameter $\theta$.

Thus, a stationary environment is assumed.

In a later extension, the robustness of opponent modeling against non‑stationary opponents will be investigated by gradually changing $\theta$ during the match.

For example:

```text
VPIP: hands 1‑50    = 20%
      hands 51‑100  = 40%
```

This extension is not performed in the baseline experiment.

---
