# Project Document: "Poker Agent Evaluation Framework – A Scientific Approach to Strategy Assessment in Imperfect Information Games"

---

## Abstract

This document describes the conception of a comprehensive poker agent evaluation framework. In contrast to classic bot development, which primarily aims at achieving the highest possible win rates, the project defines a **multidimensional notion of quality** consisting of population performance, robustness, and game-theoretic quality. The goal is to develop a reproducible, variance-reduced benchmarking methodology for agents in imperfect, stochastic games.

The project transfers experience from solving perfect-information games, particularly the development of a Mancala minimax engine, to the domain of imperfect information games. The central challenges – hidden information, stochasticity, continuous bet-sizing spaces, and the practical computability of game-theoretic equilibrium properties – are addressed through an explicitly defined game abstraction, controlled opponent populations, and reproducible experimental designs.

The framework explicitly distinguishes between three evaluation dimensions:

1. **Population Performance:** expected value against a defined population of opponents.
2. **Robustness:** stability of performance across different, plausible opponent parameters.
3. **Game-Theoretic Quality:** exploitability or distance to game value in games where such computation is actually feasible.

For abstracted NLHE, exploitability is not directly computed. Instead, we investigate whether an agent's Leduc exploitability correlates with its performance in abstracted NLHE – an empirical test of whether game-theoretic properties from small games transfer to larger domains.

The CLI application to be developed is primarily understood as a scientific testbed for strategies and, secondarily, as a playing bot. The first milestone is therefore not a powerful poker agent, but a **validated, reproducible, and statistically defensible evaluation infrastructure**.

---

## 1. Introduction & Motivation

The transition from deterministic or perfect-information games to poker motivates a different evaluation methodology. In idealised games such as Mancala, mathematical claims about outcomes under perfect play can sometimes be established.

Such certainty does not exist in Hold'em poker.

Poker is an **imperfect, stochastic zero-sum game** with hidden information, random cards, and high variance. A simple win rate over a limited number of hands can therefore paint a misleading picture of an agent's strategic quality.

The project is therefore understood not primarily as *bot development*, but as **methodology development for reproducible agent evaluation**.

The central research question is:

> **How can the quality of a poker agent be evaluated reproducibly and multidimensionally in an imperfect, stochastic environment, without reducing the evaluation to a single, highly variable win rate?**

Further research questions arise:

- How strongly does the choice of opponent population influence the measured performance?
- How robust is an agent to changes in opponent parameters?
- How well does a game-theoretic metric from a small, solvable game correlate with practical performance in abstracted NLHE?
- What effects does the bet-sizing abstraction have on the validity of the evaluation?
- How large must the sample be for a measured performance difference to be statistically reliable?
- How much can online opponent modeling improve performance against unknown opponents?
- How well do the actually observed behavioural frequencies of parametric opponents match their specified target parameters?

---

## 2. Problem Definition & Game Selection

### 2.1 Focus

**Heads-Up No-Limit Texas Hold'em (HU NLHE) in a controlled cash-game benchmark environment.**

To make the game fully reproducible, the benchmark environment is explicitly defined:

- 2 players
- Heads-Up
- No-Limit Texas Hold'em
- 100 BB starting stack
- Small Blind = 0.5 BB
- Big Blind = 1 BB
- no antes
- no rake in the scientific baseline benchmark
- stack reset to 100 BB after each hand
- no rebuys within a hand
- deterministic card order per experiment seed
- discrete, explicitly defined bet-sizing space
- identical game rules for all tested agents

This configuration initially does **not attempt to simulate a real poker room**, but defines a controlled research environment.

A later extension with rake, different stack sizes, antes, or 6-max will be treated as a separate experiment and not mixed with the baseline results.

### 2.2 Justification

- Two players → zero-sum structure and clear definition of net chip EV.
- No multiplayer dynamics such as team formation or collusion.
- Good suitability for later extensions.
- High strategic complexity due to hidden information.
- Bet-sizing and information sets present a relevant algorithmic challenge.
- The game definition can be fully controlled and reproduced.

---

## 3. Evaluation Model

The previous one-dimensional metric "win rate" is discarded.

Instead, three **complementary evaluation dimensions** are used:

| Dimension | Definition | Primary Metric |
| :--- | :--- | :--- |
| **Population Performance** | Expected value against a defined opponent population | empirical EV estimator / 100 in BB |
| **Robustness** | Stability of EV across a distribution of plausible opponents | Mean, Median, 10%-Quantile |
| **Game-Theoretic Quality** | Resistance against Best Response in an explicitly defined game environment | Exploitability / Distance to game value |

The dimensions are deliberately **not called mathematically orthogonal**. Correlations may exist between them. Precisely these interactions are the subject of investigation.

---

### 3.1 Population Performance

For an agent $\pi$ and an opponent parameter vector $\theta$, let $EV(\pi,\theta)$ be the expected net chip win of the agent per hand.

The performance over an opponent population $D$ is defined as:

$$
P(\pi\mid D)=\mathbb{E}_{\theta\sim D}[\mu(\pi,\theta)]
$$

For simulated data, a distinction is made between the theoretical expectation and its empirical estimator.

The empirical EV estimator is:

$$
\widehat{EV}_{100} = 100 \cdot \frac{1}{N} \sum_{i=1}^{N} Profit_i
$$

where $Profit_i$ is the change in the agent's net chip stack in hand $i$ in big blinds.

The symbol $\widehat{EV}_{100}$ therefore denotes an **empirical estimator of the expected net chip win**, not an immediately computed game-theoretic decision EV.

All performance figures are reported in BB/100 unless another unit is explicitly stated. The realised profit of player A in hand $i$ is $\mathrm{Profit}_i=S_{i,after}-S_{i,before},$ where $S_{i,before}$ is the stack immediately **before posting blinds** and $S_{i,after}$ is the stack after the hand. The empirical mean-payoff estimator is

$$
\widehat{\mu}_{100}=100\cdot\frac{1}{N}\sum_{i=1}^{N}\mathrm{Profit}_i.
$$

This empirical mean payoff may be reported as empirical EV/100 for readability, but it is not automatically a decision EV or game-theoretic EV.

---

### 3.2 Robustness

Robustness should not be confused with a theoretical "worst case".

The true worst case would be:

$$
R_{min}(\pi) = \min_{\theta \in \Theta} EV(\pi,\theta)
$$

For a sample from an opponent population, the 10%-quantile is used instead:

$$
R_{10}(\pi\mid D)=Q_{0.10}\left(\widehat{\mu}_{100}(\pi,\theta):\theta\sim D\right)
$$

The 10%-quantile is computed over the opponent parameters ( $\theta \sim D$ ), not over individual hands. For each $\theta$, a stable EV estimator is computed, and these EV values are quantiled over the $\theta$ -distribution.

Additionally, the following values are reported:

- Mean EV
- Median EV
- 10%-quantile
- 25%-quantile
- Standard deviation
- Empirical minimum value (always reported descriptively; it is not interpreted as a guaranteed worst case and is not conditionally suppressed as an outlier)

The 10%-quantile is explicitly referred to as a **robustness metric** and not as a mathematical worst case. A confidence interval for $R_{10}$ is reported using a profile-level or nested bootstrap that respects the hierarchy of opponent profiles and simulation runs. For important results, the preferred procedure is: (1) resample opponent profiles, (2) if repeated runs exist per profile, resample runs within profiles, (3) recompute the profile-level EV, and (4) recompute the 10%-quantile. Small profile counts are flagged as insufficient for precise tail estimation. The empirical minimum is always reported descriptively and is not conditionally suppressed as an outlier.

---

### 3.3 Exploitability

Exploitability is only referred to as such where a defined best-response computation and a reference value of the game are available.

For a game with game value $V^*$, we define: $X(\pi) = BR(\pi) - V^*$ where $BR(\pi)$ denotes the maximum EV achievable by an optimal best-response opponent against $\pi$.

**For Leduc Hold'em:**
- $V^*_{Leduc} = 0$ is used for the defined symmetric zero-sum benchmark environment.
- $X_{Leduc}(\pi)$ is reported in millibigblinds per hand (mBB/h).
- $BR(\pi)$ is computed via the defined OpenSpiel reference environment.
- The exact computation and configuration of the best response is versioned and stored.
- Native OpenSpiel utility units and any conversion to mBB/hand are documented separately. mBB/hand is used only when an explicit Leduc blind/stack convention and conversion factor exist; otherwise native utility units are reported.

**For NLHE:**
- $BR(\pi)$ is not computed in the baseline experiment.
- Exploitability in NLHE is therefore not referred to as such.

---

### 3.4 H5: Statistical Analysis of the Leduc–NLHE Relationship

H5 is formulated as a **predictive/associational analysis**, not as a causal claim. A correlation between Leduc exploitability and NLHE performance does not establish that Leduc exploitability causes NLHE performance or that game-theoretic properties transfer causally between the environments.

The original idea of correlating each individual $EV(\pi,\theta)$ measurement with the same $X_{Leduc}(\pi)$ would not correctly account for the nested data structure.

Therefore, H5 is primarily evaluated at the **agent level**.

For each agent, we first aggregate over the defined test population:

$$
\overline{EV}_{NLHE}(\pi) = \mathbb{E}_{\theta\sim D_{test}}[ EV(\pi,\theta) ]
$$

and: $R_{10}(\pi) = Q_{0.10}[ EV(\pi,\theta) ]$ Then the following relationships are examined:

$$
corr\left( X_{Leduc}(\pi), \overline{EV}_{NLHE}(\pi) \right)
$$

and separately:

$$
corr\left( X_{Leduc}(\pi), R_{10}(\pi) \right)
$$

Thus, **one observation corresponds to one agent**, and no artificial duplication of the same Leduc exploitability over many opponent samples occurs.

H5 requires at least $N_{min}=12$ independent agent variants for a meaningful rank-correlation analysis. Below this threshold, H5 is reported as exploratory with an explicit sample-size limitation. This is a pragmatic project criterion, not a universal statistical requirement.

The association is reported for each previously defined population separately and over the entire set of independent agent variants. Agent family and variant identifiers are retained so that multiple seeds or variants of the same trained policy are not silently treated as independent scientific observations.

If a later extension models individual $\theta$ -samples directly, a hierarchical or mixed-effects model that explicitly accounts for the nesting "opponent sample within agent" will be used.

For aggregation, at least 10 independent opponent samples ( $\theta \sim D_{test}$ ) are used per agent. Each sample itself comprises a sufficient number of hands to obtain a stable $\widehat{EV}_{100}$ estimate (see power analysis, 3.5). The number of samples is documented and varied in sensitivity analyses.

---

### 3.5 Statistical Power Analysis

Before any inferential statistical comparison, a power analysis is performed, provided that a concrete hypothesis test is planned.

**Parameters:**
- $\alpha = 0.05$ (significance level)
- $\beta = 0.20$ (power = 80 %)
- expected minimum relevant EV difference $\delta$
- estimated standard deviation of the paired differences $\sigma_d$

For a paired comparison, approximately:

$$
n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \, \sigma_d^2}{\delta^2}
$$

where $n$ denotes the required number of independent pairs.

The computation is thus directly based on the expected difference and variance, not solely on a generic Cohen's $d$ value.

**Pilot study:** Before a major experiment, a pilot study is conducted to estimate the variance of the paired differences.

The pilot size of 1,000–5,000 hands serves solely as an initial estimate and is not a universal statistical standard.

The estimated variance is used to determine the final sample size. If no pilot data are available for a first cold-start power analysis, a conservative planning value of $\sigma_d \approx 100$ BB/100 is used provisionally; the assumption is replaced by pilot-based variance estimates before the final confirmatory experiment.

---

### 3.6 Precision Target and Sample Size

Power analysis and precision target are treated separately.

**Power analysis** answers:

> How many observations are needed to detect a pre‑defined minimum relevant effect with a certain probability?

**Precision analysis** answers:

> How many observations are needed so that the estimate has a desired maximum uncertainty?

For important benchmark comparisons, an additional estimator-specific precision target is defined, for example:

$$
95\% \text{ CI} \approx \widehat{EV} \pm 1 \,\text{BB/100}
$$

The sample size is determined **before the final experiment** based on a pilot variance or a conservative variance estimator. The precision target explicitly records the estimand, target CI half-width, confidence level, statistical unit, and variance estimator. A mean-EV precision target does not automatically guarantee precision of a population 10%-quantile.

A final experiment is **not** stopped iteratively as soon as an ordinary 95% confidence interval randomly falls below a certain width. This avoids unadjusted sequential stopping procedures altering the nominal error probability.

If adaptive stopping is used in a later study, it must be specified as a separate sequential design and evaluated statistically accordingly.

---

## 4. Variance Reduction & Experimental Design

To reduce the random noise of card realisations, a **paired experimental methodology with Common Random Numbers (CRNs)** is used.

The methodology controls in particular:

- Card realisations
- Position
- Experiment seed
- Bot version
- Opponent configuration

It does not eliminate all sources of variance, but reduces systematic differences and improves the statistical precision of comparisons.

---

### 4.1 Common Random Numbers (CRNs)

Each experimental run receives a deterministic master seed. Chip arithmetic uses integer chip units wherever possible. The chance process is separated from agent-action randomness. A deterministic, pre-defined deal schedule is generated independently of action-dependent RNG consumption. Agent randomness MUST be event-indexed by `(bot_id, hand_id, decision_index, purpose)`. A stream-based RNG fallback is permitted only for external library integrations that cannot expose event-indexed randomness; every such integration must be documented as a reproducibility limitation. “Bit-exact” reproducibility is scoped to the fully pinned software, dependency, configuration, and execution environment.

From this master seed, separate namespaces are generated:

- `DEAL_SCHEDULE` – deterministic chance-event schedule for cards
- `RNG_BOT_A` – stochastic decisions of A
- `RNG_BOT_B` – stochastic decisions of B

The deal schedule is generated independently of agent action counts. For each hand, the complete deck permutation or equivalent deterministic schedule is fixed before the first decision. The same schedule is reused for corresponding paired conditions whenever the CRN design requires identical chance realisations.

It is important that the random number consumption of one bot does not shift the chance sequence of another condition. Agent RNG therefore MUST be event-indexed rather than based on total previous random draws, subject only to the documented external-library fallback above.

This ensures that:

```text
poker run --seed 12345
```

can produce bit‑exactly reproducible results given identical software and configuration versions.

Additionally, the following metadata are stored:

- Python version
- Engine version
- Bot version
- Abstraction version
- Experiment configuration
- Master seed
- RNG configuration
- Git commit / release version

---

### 4.2 Seed Dependence

A single seed must not be regarded as representative for a final benchmark.

For development and sanity experiments, at least three different master seeds are used.

For final results, the number of seeds is determined together with the desired statistical precision.

Seeds are not a substitute for sample size. The statistical unit is defined by the dependency structure. For non-adaptive fixed policies, paired deals may be the primary unit. For adaptive/stateful agents, hands within a run are temporally dependent; the primary inferential unit is the independent run/trajectory cluster. CIs and bootstraps must not treat all hands from one adaptive trajectory as independent observations.

Additionally, we report:

- Mean over seeds
- Spread between seeds
- Confidence interval over seed‑based aggregations, if applicable
- Total number of paired observations

Thus, a distinction is made between **sampling variance within an experiment** and **seed variance between experiments**.

For final benchmark results, multiple independent master seeds are used. Seed-level aggregation is treated as a cluster structure rather than as a substitute for the underlying sample size. If seed-level uncertainty is reported, a seed-level $t$ interval or explicitly defined cluster/bootstrap method is used. The number of seeds is increased according to a pre-defined precision criterion, not an arbitrary CV threshold.

---

### 4.3 Position Compensation

For each deal, a position pair is generated:

1. Bot A as button / small blind, bot B as big blind.
2. Bot B as button / small blind, bot A as big blind.

The intended common card realisations are reused in a controlled manner.

The paired difference is defined as

$$
d_i = Profit(A_{BTN},B_{BB}) - Profit(A_{BB},B_{BTN})
$$

The pairing controls for position and common card realisations and thereby reduces the variance of the comparison.

It is not interpreted as a complete elimination of all positional or strategic effects.

---

### 4.4 Position Compensation with Opponent Modeling

For agents with opponent modeling, the order of observations can influence the agent's internal state.

Therefore, the two positional conditions are performed in separate, symmetrically constructed runs:

- Run 1: A‑BTN against B‑BB
- Run 2: B‑BTN against A‑BB

Both runs use the same defined card realisations, i.e., the same CRN plan. If an agent architecture cannot preserve identical chance realisations, the experiment must explicitly document the incompatibility and classify the comparison as seed-synchronised rather than identical-realisation CRN.

The adaptation history is initialised separately for each run.

This prevents an agent from carrying an information advantage from the first half of the paired comparison into the second half.

For adaptive agents, it is additionally documented whether the pairing is based on **identical random realisations** or only **seed‑synchronised random streams**.

---

### 4.5 Statistical Evaluation

The statistical unit is fixed **before simulation** and stored as an experiment-level field. Allowed values are `paired_deal`, `independent_run`, or `trajectory_cluster`, depending on the dependency structure.

For static policies, paired deals may be the primary unit when independence is justified. For adaptive/stateful policies, the independent unit is the complete run/trajectory because later decisions depend on earlier observations. Segment estimates (10/20/50/100/500 hands) are repeated measurements within a trajectory, not independent samples. Bootstrap and confidence intervals therefore resample trajectories or other pre-defined independent clusters for adaptive agents.

The number of simulated hands and the number of independent statistical units are always reported separately.

For the H1 independent baseline, the two agents are evaluated in separate runs with the same total number of simulated hands but **without the paired CRN structure**. Distinct random seeds and independent chance realisations are used across the two conditions. The paired condition uses the predefined CRN deal schedule from Section 4.1. Repeated benchmark replications use independent master seeds.

The primary estimate is based on the mean paired difference:

$$
\bar d = \frac{1}{K} \sum_{k=1}^{K} d_k
$$

For static policies, $K$ may correspond to independent paired deals. For adaptive/stateful policies, $K$ denotes independent trajectory clusters.

Reported are:

- Mean
- Standard deviation
- 95% confidence interval
- Number of independent pairs
- Effect size

As a standard procedure, a paired t‑test may be used, provided the assumptions are reasonably met.

Additionally, a **paired bootstrap confidence interval** is provided as a robust sensitivity analysis for important results.

The bootstrap is performed at the level of the independent statistical clusters. For static policies this may be the paired-deal level when independence is justified. For adaptive/stateful policies it is the complete paired run or another pre-defined independent trajectory cluster; individual decisions and temporally dependent hands are never treated as independent bootstrap observations.

The standard is 10,000 resamples.

The 95% confidence interval is defined as the 2.5% and 97.5% percentiles of the bootstrap distribution of the mean paired difference.

---

### 4.6 Effect Size

For paired comparisons, a standardised effect size is reported:

$$d_z=\frac{\bar d}{s_d},$$

where $s_d$ is the sample standard deviation of the paired differences.

The effect size is reported in addition to the absolute EV difference.

The classification into "small", "medium", or "large" is used only as a conventional orientation and not as a substitute for the practical relevance of the EV difference.

In particular, a statistically significant but practically very small EV difference is not automatically interpreted as strategically relevant.

---

## 5. The Central Technical Hurdle: Bet‑Sizing Abstraction

The full no‑limit bet space is continuous. An exact enumeration of all possible bet sizes leads to an extremely large action set.

Therefore, a discrete bet‑sizing abstraction is introduced for the benchmark.

### 5.1 Baseline Abstraction

```python
BET_ABSTRACTION = [0.33, 0.66, 1.0, 2.0]

ALLOWED_ACTIONS = [ FOLD, CHECK, CALL, BET_33, BET_66, BET_100, BET_200, ALL_IN, ]
```

The numbers represent standardised pot multipliers for bets, and – after a separate raise definition – for raises.

### 5.2 Exact Semantics

The abstraction must unambiguously define for every situation:

- Reference pot size
- Computation of a bet
- Computation of a raise
- Minimum bet
- Minimum raise
- Handling of all‑in
- Rounding
- Stack limiting
- Behaviour with short stacks

In particular, `BET_100` must not be interpreted differently depending on whether a bet has already occurred previously.

The concrete formula for raises is documented as part of the game definition. Before final benchmarking, the exact function must be immutable and versioned. The abstract action label is context-dependent: in an unopened betting situation, `BET_k` maps to a bet of size `kP`; in a raise situation, the same abstract label maps to a deterministic raise-to amount defined by the versioned function below. It is therefore one context-dependent abstract command, not two different action identifiers. For example, with reference pot

$$
Amount_{bet}=\min(kP,S)\,,\qquad Amount_{raise}=f(P,B,k,S)\,.
$$

Here, $P$ is the reference pot size, $B$ is the opponent wager, $k$ is the abstract multiplier, and $S$ is the maximum legal amount. The exact function $f$ must be specified. Minimum bet, minimum raise, incomplete all-in raises, action reopening, rounding, stack caps, and short-stack behaviour are part of the game configuration.

For example, a raise can be defined as a function of current pot, opponent's bet, desired multiplier, and stack limit:

$$
RaiseAmount = f(Pot, OpponentBet, k, S)
$$

The implementation must ensure that each abstract action identifier deterministically generates exactly one legal bet amount.

`ALL_IN` maps to the player's entire remaining stack and is treated as a bet if no wager exists on the current street, or as a raise-to the all-in amount otherwise. It is always accepted if the amount is at least the minimum legal bet/raise; otherwise it is treated as an illegal action.

### 5.3 Consequences

Thus, the agent is strictly playing an **abstracted version of NLHE**.

The evaluation therefore initially applies only to this defined abstract game environment.

A later transfer to more realistic bet sizes or human players will be treated as a separate experiment.

Additionally, the sensitivity to the abstraction will be investigated:

- smaller action sets
- baseline action set
- expanded action set

This allows empirical testing of how strongly the results depend on the chosen abstraction.

---

## 6. Test Pyramid: Game‑Theoretic Validation vs. Population Performance

The evaluation is divided into three levels:

| Level | Environment | Purpose | Reference Implementation |
| :--- | :--- | :--- | :--- |
| **Unit / Proof** | **Leduc Hold'em** | Testing game‑theoretic computations | OpenSpiel / Best Response |
| **Integration** | **Limit Hold'em** | Validation of hand evaluation, game rules, and simple strategies | OpenSpiel / own engine |
| **Production** | **Abstracted NLHE** | Main agent benchmarking | Own Python engine |

These levels have **different scientific purposes**.

Leduc is not a direct simulation of NLHE, but a controllable environment in which best‑response and exploitability computations can be verified.

Limit Hold'em serves as an intermediate step for validating core game logic.

Abstracted NLHE is the actual production environment for population performance and robustness.

The later analysis explicitly examines whether properties from the smaller games correlate with NLHE performance.

---

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
- `Aggression_target` = defined target for aggressive actions
- `FoldToCBet_target` = probability of folding to a continuation bet

The hand‑strength evaluation determines the relative strength of a situation. A calibration mechanism translates this strength together with the target parameter into an action.

This avoids, for example, incorrectly defining VPIP as a mere equity threshold.

**Aggression** is defined as the ratio of aggressive actions (bet, raise) to passive actions (call, check):

$$Aggression=\frac{A}{P}.$$

This is a ratio, not an aggressive-action percentage. The corresponding aggressive-action share is $A/(A+P)$. Thus 0.8 corresponds to 44.4%, 1.0 to 50%, and 1.5 to 60% when $A+P$ is the complete denominator. The denominator and treatment of folds are fixed in the policy specification. 3Bet is the proportion of explicitly defined eligible preflop 3-bet opportunities in which the player re-raises.

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

### 7.5 Static Profiles

The following profiles are initially defined as benchmark opponents:

- **Nit / Tight‑Passive:** VPIP = 10%, PFR = 4%, Aggression = 0.3
- **TAG / Tight‑Aggressive:** VPIP = 22%, PFR = 18%, Aggression = 0.8
- **LAG / Loose‑Aggressive:** VPIP = 32%, PFR = 26%, Aggression = 0.9
- **Calling Station:** VPIP = 50%, PFR = 5%, FoldToCBet = 15%. The Calling Station profile does not specify an Aggression target; the parameter is treated as don't-care for this profile.
- **Maniac:** VPIP = 55%, PFR = 45%, Aggression = 1.5

These values are **benchmark profiles, not empirically validated real player populations**.

A later realistic population must be based on a separate empirical data foundation.

The aggression values refer to the ratio of aggressive actions to passive actions. A value of 1.0 means the agent acts aggressively in 50% of situations (1:1 ratio). A value of 1.5 corresponds to a 1.5:1 ratio (60% aggressive), a value of 0.3 to 0.3:1 (23% aggressive).

The stated values are understood as **target parameters** and validated via the calibration test.

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

## 8. Architecture & Data Management

### 8.1 Data Model

Each match run is persisted in SQLite or Parquet.

Minimal model:

```sql
MATCHES ( id, seed, bot_version, opponent_hash, hands_played, ev_bb_100, ci_low, ci_high )

HANDS ( id, match_id, street, hero_cards, villain_cards, board, actions_json, pot, ev_change )
```

For scientific reproducibility, the following are additionally stored:

```text
experiment_id
game_config_hash
abstraction_version
engine_version
python_version
rng_version
git_commit
starting_stack
small_blind
big_blind
rake
population_id
population_seed
opponent_parameters
bot_configuration
statistical_method
sample_size
precision_target
```

The complete experiment configuration is stored as an immutable configuration and hashed.

The configuration hash is computed as **SHA-256 over a canonical JSON representation** of the frozen configuration. Keys are sorted and values are normalised deterministically so that equivalent configurations produce the same hash across platforms.

The project uses **Poetry** for Python dependency management. `pyproject.toml` and `poetry.lock` are stored with the project and the relevant lockfile state is included in experiment metadata.

For SQLite persistence, the schema includes a `schema_version` table. Schema changes are versioned through explicit migrations; **Alembic** is used for schema migration once the SQLite schema is established. A database is never modified by an implicit or undocumented schema change.

The reference benchmark uses the developer's **physical local machine**, not rented cloud instances. Hardware metadata records a platform-independent fingerprint derived from CPU model, core count, and architecture. The OS identifier is the output of `platform.platform()`. Exact machine identifiers that are unnecessary for reproducibility are not required.

This allows a result to be reconstructed not only by a seed, but by the entire relevant experiment environment.

---

### 8.2 Action Validation, Illegal Actions, and Agent Budgets

Every action emitted by an agent is validated against the current legal-action set before it reaches the game engine.

**Illegal-action policy:** an illegal action is a deterministic agent error. The engine rejects it and records the complete state, requested action, legal action set, and agent version. The benchmark does **not** silently substitute the nearest legal action, because doing so would change the agent's requested behaviour and make results difficult to interpret. Development runs may raise an exception; benchmark runs terminate the affected run with a defined failure status. The frozen benchmark policy is `illegal_action_handling = exclude_run`: the failed run is excluded from the inferential dataset rather than counted as a lost pot or silently repaired. It is never automatically replaced with a new seed. The failure is retained in the experiment log and reported separately. The planned power/precision analysis is based on the number of valid independent statistical units; if failures reduce the valid sample below the pre-specified target, the benchmark is incomplete and additional valid runs are scheduled before final analysis.

Every non-trivial agent also declares an execution budget. At minimum:

```text
agent_simulation_budget
agent_time_budget_ms   (optional safety limit)
timeout_policy
```

If an agent exceeds `agent_time_budget_ms`, the engine executes the `timeout_policy`. The default benchmark policy is `fold`: the hand is terminated and the agent forfeits the pot. Other policies (e.g., `check_if_legal_else_fold`) must be explicitly declared and versioned.

For search agents, the primary scientific budget is the number of simulations per decision. A time limit may be used as a technical safety bound but is not silently substituted for the simulation budget. Budget, timeout, and failure behaviour are stored with the agent configuration.

### 8.3 Performance Target

The Python engine is optimised to a speed of at least 10,000 simulated simple engine hands per second in a single‑thread benchmark **as an engineering target**.

The actual performance is measured on defined hardware.

**Reference hardware:** the physical local development/benchmark machine used for the project. The exact CPU model, core count, architecture, RAM, Python version, and OS identifier are recorded with each benchmark. No cloud instance is required for the scientific baseline; parallelisation may use the local machine's cores when needed.

For more complex agents such as MCTS or NFSP, a significantly lower simulation rate is to be expected.

The benchmark speed is not a scientific target metric for agent quality.

Large benchmark matrices can be parallelised over multiple cores or a cluster if needed.

Parallelisation is performed at the match level: each match (defined by agent, opponent, seed) is executed as an independent job in a job queue. Results are merged into a central database. Parallelisation is purely for runtime optimisation and does not affect the statistical evaluation.

---

### 8.4 Debugging & Replay Tool

A mandatory CLI module serves to analyse large losing pots:

```bash
poker inspect \
  --match-id 42 \
  --filter "hero_loss > 50 AND street == 'river'"
```

This allows critical hands to be examined in detail.

Additional filters are provided:

```bash
poker inspect --match-id 42 --street river
poker inspect --match-id 42 --action call
poker inspect --match-id 42 --min-pot 50
```

This prevents the developer from relying solely on aggregated JSON or database data.

The replay mechanism shall display the complete decision chain. Internally, **true game state** and **agent-visible observation** are stored separately so replay data cannot accidentally leak hidden information into a policy:

- Hole cards
- Board
- Pot
- Stack
- Position
- Action history
- Available actions
- Bot evaluation
- Chosen action
- Resulting profit
- Estimated decision EV, if applicable

---

## 9. Validation & Sanity Gates

Before complex AI methods are implemented, the engine must pass a series of tests.

---

### 9.1 Unit Tests: Engine

At least:

- Card deck
- Shuffle
- Hand ranking
- Board development
- Pot calculation
- Fold
- Check
- Call
- Bet
- Raise
- All‑in
- Minimum raise
- Position rotation
- Showdown
- Split pot
- Stack reset
- Illegal-action rejection
- Deterministic abstract-action-to-amount mapping

These tests verify only the correct implementation of the game mechanics.

---

### 9.2 Symmetry Test

Two completely identical bots must on average have no systematic advantage: $EV(A,A) \approx 0$

The experiment must in particular verify the position compensation and the RNG mechanics.

---

### 9.3 Random‑vs‑Random Test

Two identical random bots are played against each other over a large number of paired deals.

The result should be statistically compatible with zero and practically close to zero under the pre-defined symmetry tolerance $\epsilon_{sym}=0.5$ BB/100. The 95% CI of the paired difference must be fully contained in $[-0.5,+0.5]$ BB/100. A CI containing zero is a sanity check, not proof of engine correctness. Repeated seeds must also show no systematic positional or RNG bias.

The test is interpreted as a sanity test, not as proof of the general correctness of the engine.

---

### 9.4 Determinism Test

The same run:

```bash
poker run --seed 12345
```

must produce exactly the same sequence and the same result given identical software and configuration versions.

---

### 9.5 Regression Tests: Engine and Rule Tests

All known error cases are stored as fixed regression cases.

The regression tests include at least:

- defined preflop all‑in situations with **pre‑computed expected equity** and fixed card/deck definition
- defined river situations with known pot and stack outcomes
- split pot with identical hands
- correct handling of side pots as a separate future multiplayer test (this is prepared for the later 6‑max extension but is **not** part of the HU Phase‑0 gate; see 9.8)
- correct minimum‑raise rule
- correct stack limiting
- correct pot distribution
- correct position rotation

For fixed equity test cases, no general statements such as "AA vs 72o must deliver 80+% EV" are used.

Instead, **fully specified test cases** with a known reference value are used.

A distinction is explicitly made between **equity**, **realised profit**, and **EV**.

---

### 9.6 Strategy and Policy Tests

Strategic expectations are separated from engine regression tests.

Examples:

- A calling station shall achieve defined FoldToCBet target values.
- A TAG profile shall lie within a tolerance of its VPIP/PFR target values.
- A policy with a defined action shall correctly pass that action to the engine.
- A hand‑strength policy shall select the expected action in a fully specified situation.

These tests verify the policy, not the underlying engine mechanics.

---

### 9.7 Policy Calibration Test

For each parametric profile, a calibration run is performed.

The observed frequencies are compared with the target parameters:

```text
target VPIP       vs. observed VPIP
target PFR        vs. observed PFR
target 3Bet       vs. observed 3Bet
target CBetFreq   vs. observed CBetFreq
target Aggression vs. observed Aggression
target FoldToCBet vs. observed FoldToCBet
```

A profile is considered calibrated only if all relevant deviations lie within the defined tolerances.

---

### 9.8 Hidden-Information Safety Gate

The engine maintains two explicitly separate representations:

```text
TRUE_GAME_STATE
AGENT_OBSERVATION
```

For player $i$, only

$$o_i = O_i(s)$$

is supplied to the policy. The observation must not contain opponent hole cards, undealt cards, future board cards, hidden deck state, hidden opponent RNG state, or any other information unavailable to that player.

A dedicated Phase-0 test constructs states with identical legal observations but different hidden information and verifies that a deterministic policy produces the same action. This test is mandatory before MCTS or RL agents are benchmarked.

### 9.9 Phase‑0 Gate

Phase 0 is considered complete only when the following conditions are met:

- Engine unit tests passed
- Symmetry test passed
- Random‑vs‑random test passed
- Determinism test passed
- Regression tests passed
- Policy calibration tests passed
- Statistical evaluation validated
- Statistical unit explicitly represented in configuration
- Cluster-aware bootstrap/CI validated for adaptive trajectories
- Hidden-information safety gate passed
- Illegal-action behaviour validated
- Agent budget and timeout behaviour defined
- Experiment configuration fully persisted
- Replay of any match is possible

Side‑pot and 6‑max tests are **not part of the HU Phase‑0 gate**; they will be activated with the later multiplayer extension.

---

## 10. Initial Technical Project Setup & Implementation Roadmap

The first coding step is deliberately **technical project preparation**, not AI development. The goal is to create a small, reproducible local software project in which the Phase-0 gates can later be implemented without changing the experimental assumptions ad hoc.

### 10.1 Local Project Foundation

The baseline is developed and benchmarked on a **local reference machine**. Cloud instances are not required for the baseline. The exact machine and software environment are recorded for reproducibility.

Recommended initial repository structure:

```text
poker-agent-evaluation/
├── pyproject.toml
├── poetry.lock
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── pokerbench/
│       ├── engine/
│       ├── agents/
│       ├── evaluation/
│       ├── experiments/
│       ├── replay/
│       └── cli/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── validation/
├── configs/
│   ├── game/
│   ├── agents/
│   ├── populations/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
└── scripts/
```

Poetry is the dependency-management and lockfile tool. The initial environment is created from `pyproject.toml`, committed with `poetry.lock`, and never changed silently between benchmark runs.

### 10.2 First Implementation Sequence

The first implementation sequence is:

1. initialise the Git repository and Poetry project;
2. create the package and test structure;
3. freeze the game configuration as a versioned data object;
4. implement canonical configuration serialisation and SHA-256 hashing;
5. implement the card/deal schedule independently of agent RNG;
6. implement the immutable game state and legal-action generator;
7. implement and freeze the exact bet/raise semantics defined in Section 5.2, including versioning and unit-test coverage, before Phase 0 can be considered complete;
8. implement hand evaluation, pot accounting, stack reset, and position rotation;
9. implement the `TRUE_GAME_STATE` / `AGENT_OBSERVATION` boundary;
10. implement deterministic replay of a single hand;
11. add unit and regression tests before building any sophisticated agent;
12. implement the paired runner and statistical-unit metadata;
13. implement the symmetry and random-vs-random gates;
14. implement policy calibration in the frozen calibration environment;
15. only then implement baseline agents and larger benchmark runs.

### 10.3 Configuration and Reproducibility Contract

Phase 0 uses a **binding technical specification**: rules that affect legal actions, payoffs, information availability, RNG, or statistical interpretation are frozen and versioned before scientific benchmarking.

Every frozen configuration is serialised canonically and hashed with SHA-256. The hash covers the complete relevant configuration rather than merely the game name or seed.

### 10.4 Hardware and Execution Environment

The primary benchmark environment is a local reference machine. For each benchmark, record:

```text
cpu_model
cpu_core_count
cpu_architecture
ram
python_version
os_identifier = platform.platform()
poetry_lock_hash
git_commit
```

A platform-independent hardware fingerprint may be derived from CPU model, core count, and architecture. Exact hardware serial numbers are not required.

### Implementation Roadmap

The development strictly follows the principle:

> **Evaluation Framework first, Bot second.**

---

### Phase 0: Core Infrastructure & Validation (Weeks 1–2)

- [ ] Implement deterministic NLHE engine
- [ ] Freeze canonical configuration and SHA-256 hash
- [ ] Card deck
- [ ] Board development
- [ ] Pot calculation
- [ ] Stack management
- [ ] Hand ranking
- [ ] Defined bet‑sizing abstraction
- [ ] Deterministic pre-defined deal schedule
- [ ] Isolated/event-indexed agent RNG
- [ ] Hidden-information safety boundary
- [ ] `PairedRunner`
- [ ] Statistical evaluation
- [ ] Bootstrap CI
- [ ] Parametric bot factory
- [ ] Explicit separation of policy parameters and observable frequencies
- [ ] Policy calibration test
- [ ] SQLite/Parquet data model
- [ ] Replay data structure
- [ ] Unit tests
- [ ] Symmetry test
- [ ] Random‑vs‑random test
- [ ] Determinism test
- [ ] Regression tests
- [ ] Illegal-action validation
- [ ] Agent budget/timeout policy
- [ ] Calibration environment and CBetFreq validation
- [ ] Statistical-unit and cluster-aware bootstrap validation
- [ ] Poetry lockfile and execution metadata persisted

**Deliverable:**

```bash
poker battle \
  --bot-a random \
  --bot-b tag \
  --seeds 10000
```

with a reproducible statistical report.

The statement "statistically valid" is only used after the sanity and precision tests have been passed.

---

### Phase 1: Baselines & Measurement (Weeks 3–4)

Phase 1 intentionally contains **no MCTS**. The purpose is to establish transparent baseline agents and verify that the benchmark infrastructure produces interpretable measurements before search complexity is introduced.

- [ ] Develop a random baseline agent
- [ ] Develop a deterministic hand-strength baseline
- [ ] Implement exact equity computation where feasible
- [ ] Implement pot-odds decisions
- [ ] Define and version the equity/range representation
- [ ] Full benchmark matrix against calibrated static profiles
- [ ] Measure performance against the baseline bet abstraction
- [ ] Run the predefined action-abstraction sensitivity checks where required
- [ ] Calibrate parametric opponents in the frozen calibration environment
- [ ] Verify reporting of hands, runs, profiles, statistical unit, seeds, and configuration hashes

The Phase-1 baseline is deliberately simple. Its purpose is measurement validation, not strategic strength.

---

### Phase 2: Search, Opponent Modeling & Game Theory (Weeks 5–8)

- [ ] Develop an imperfect-information MCTS agent (IS-MCTS or another explicitly specified imperfect-information method)
- 100 simulations per decision as the initial engineering baseline
- extension to 1,000 in later experiments if justified
- rollout policy: hand strength + pot odds
- UCB formula:

$$
UCB = Q(s,a) + c \sqrt{ \frac{\ln(N(s))}{N(s,a)} }
$$

with $c=1.0$.

- [ ] Define search depth as number of actions
- [ ] Ablation: Depth 1 / 2 / 4 / 8
- [ ] Enforce the hidden-information safety boundary during search
- [ ] Record search budget and timeout behaviour
- [ ] Bayesian online learner
- [ ] Estimation of VPIP/PFR
- [ ] Estimation of further observable frequencies
- [ ] Uncertainty model
- [ ] Adaptation horizons
- [ ] Exploitative agent
- [ ] Define dynamic preflop ranges through a versioned range representation
- [ ] Dynamic postflop strategy
- [ ] Integration of OpenSpiel
- [ ] Best‑response analysis in Leduc
- [ ] Validation of exploitability computation
- [ ] Analysis of correlation between Leduc exploitability and NLHE performance

Important:

Leduc exploitability is stored as a **separate metric** and is **not** misinterpreted as a direct NLHE exploitability score.

---

#### Phase-2 MCTS Rollout Semantics

MCTS is an **imperfect-information** method. A perfect-information search over the hidden true state is invalid because it leaks information unavailable to the agent. The exact search method (preferably Information Set MCTS) is versioned.

The rollout policy is deterministic and based on:

1. equity against a defined versioned range;
2. pot odds;
3. a heuristic payoff estimate for each legal action.

For a simplified call model, with equity $E$, pot before calling $P$, and additional call cost $C$: $\widehat{Payoff}(call)=E(P+C)-C$. Here $\widehat{Payoff}$ is a **heuristic rollout payoff estimate**, not a game-theoretic EV. Bet and raise actions require their own response model. If such a model is incomplete, the rollout remains explicitly labelled heuristic.

---

### Phase 3: Meta‑Optimisation & Pareto Analysis (Weeks 9–12)

- [ ] Train a simple reinforcement learning agent, e.g., NFSP
- [ ] Train against parametric development population
- [ ] Hyperparameter selection on validation population
- [ ] Final evaluation on unseen holdout population
- [ ] Ablation of opponent distribution
- [ ] Ablation of bet abstraction
- [ ] Derive three agent archetypes:

#### Titan

Minimises game‑theoretic exploitability in the Leduc environment.

Its performance in NLHE is measured as a transfer result.

A possible later variant can be trained in a reduced NLHE abstraction with CFR. In that case, an approximate exploitability estimate is explicitly labelled as **estimate** and not as exact exploitability.

#### Predator

Maximises population performance against clearly defined opponent populations or difficult population sub‑regions.

For example, Predator can be optimised for a pre‑defined high VPIP range.

The 80% percentile of the VPIP distribution can be used as a **concrete experiment**, but is not assumed as a general definition of "difficult".

#### Adaptive

Combines a stable baseline strategy with opponent modeling and switches between different strategic modes depending on the posterior certainty.

---

### 10.5 Pareto Analysis

The Pareto analysis uses **commensurable NLHE objectives** as its primary optimisation view:

$$
Y(\pi) = \overline{EV}_{NLHE}(\pi) \quad \text{maximise}
$$
$$
Z(\pi) = R_{10}(\pi) \quad \text{maximise}
$$

The standard Pareto visualisation therefore uses:

- X-axis: $\overline{EV}_{NLHE}$
- Y-axis: $R_{10}$

Leduc exploitability $X_{Leduc}$ remains a separate game-theoretic metric and is analysed in H5. It is not converted into BB/100 merely to share a plot with NLHE metrics.

Each agent is represented as **one aggregated point**. The population over which $\overline{EV}_{NLHE}$ and $R_{10}$ are computed is fixed before the experiment. A visualisation against individual $\theta$ -samples is treated as a separate sensitivity analysis and not mixed with the primary Pareto front.

The Pareto front is not assumed a priori, but determined empirically from the results.

---

## 11. Scientific Experimental Design

Each central benchmark is considered a complete experiment object.

An experiment defines:

```text
Game Configuration + Bet Abstraction + Bot Version +
Opponent Population + Population Distribution + Population Version +
Random Seed(s) + RNG Configuration +
Sample Size + Power Target, if hypothesis test + Precision Target +
Statistical Method
```

---

### 11.1 Statistical Reporting Hierarchy

Every benchmark distinguishes the following levels:

```text
Hands ↓ Runs / trajectories ↓ Opponent profiles ↓ Population estimator ↓ Agent-level aggregate ↓ Agent family / variant structure
```

No lower-level observation is automatically promoted to an independent observation at a higher level. For adaptive agents, the trajectory/run is the primary independent unit. For H5, independent agent variants are the relevant scientific observations; multiple seeds of the same trained policy are not automatically treated as fully independent agents.

### 11.2 Benchmark Matrix

At a minimum:

| Agent | Nit | TAG | LAG | Calling Station | Maniac | Randomized Population | Stress Population | Holdout |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hand Strength | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MCTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Exploitative | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Adaptive | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The actual sample sizes are determined according to power and precision targets.

The holdout results are interpreted as final test results only after the agent configuration has been finalised.

---

### 11.3 Ablation Studies

To assess the contribution of individual components, targeted ablations are performed:

1. Without opponent modeling
2. With opponent modeling
3. Without position compensation
4. With position compensation
5. Small bet abstraction
6. Baseline bet abstraction
7. Expanded bet abstraction
8. Different RNG configurations
9. Different populations
10. Different adaptation horizons
11. Different policy‑calibration tolerances
12. Different holdout populations

This prevents an observed improvement from being merely due to an uncontrolled change in experimental conditions.

---

## 12. Challenges & Risk Mitigation

| Challenge | Solution / Workaround |
| :--- | :--- |
| **Large game tree** | Bet‑sizing and possibly card abstraction |
| **Continuous bet space** | Explicitly defined discrete action abstraction |
| **Ambiguous raise semantics** | Formal definition of all bet/raise rules |
| **High poker variance** | CRNs + position pairing + confidence intervals |
| **Statistical spurious precision** | Separate power/precision planning + bootstrap sensitivity analysis |
| **Overfitting to static profiles** | Randomised populations + stress population + holdout |
| **Arbitrary opponent distributions** | Explicit documentation and ablation of different populations |
| **Weak opponent definition** | Formal policy $\pi(a|s,\theta)$ |
| **Policy calibration error** | Calibration tests between target and observed frequencies |
| **Uncertainty in opponent modeling** | Bayesian posterior distributions |
| **Best‑response runtime** | Restriction to Leduc or other exactly controllable games |
| **Misinterpretation of Leduc results** | Separate metrics for Leduc and NLHE |
| **Engine bugs** | Unit, symmetry, regression, and random‑vs‑random tests |
| **Reproducibility** | Seed hierarchy, version pinning, experiment hashes |
| **Overfitting to one bet abstraction** | Ablation of different action sets |
| **Pseudoreplication in H5** | Aggregation at agent level or hierarchical model |
| **Uncontrolled sequential stopping** | Pre‑determined sample size or explicit sequential design |

---

### 12.1 Failure Mode Analysis

The framework may fail in the following scenarios:

| Failure Mode | Detection | Countermeasure |
| :--- | :--- | :--- |
| **Engine bug** | Symmetry/regression test fails | Unit tests, regression tests |
| **Overfitting to population** | Holdout performance collapses | Holdout evaluation, stress population |
| **Seed dependence** | Results vary strongly across seeds | Multiple seeds, report seed variance |
| **Statistical noise** | CI width > relevant effect size | Increase sample size |
| **Model misspecification** | Leduc‑NLHE correlation absent | Reject H5, revise model assumptions |
| **Unrealistic opponents** | Agent performs poorly against real players | Empirical calibration as extension |
| **Policy calibration error** | Observed frequencies deviate systematically from targets | Re‑calibrate policy mechanism |
| **Pseudoreplication** | Results depend heavily on number of opponent samples per agent | Agent level as primary unit; mixed‑effects model for nested analyses |
| **Population leakage** | Holdout results improve during development | Use new sealed test population |
| **Abstraction sensitivity** | Agent ranking changes strongly with action set | Evaluate multiple abstractions |

---

## 13. Expected Deliverables

### 13.1 Software

1. **Python CLI tool** (`pokerbench`) with:

```text
run
battle
inspect
export
plot
replay
```

---

### 13.2 Benchmark Data

A reproducible benchmark database with several million up to – depending on the statistically required sample size – over 100 million simulated hands.

The final number is not considered a scientific goal in itself.

Crucial are:

- defined precision
- confidence intervals
- independent experiment pairs
- reproducibility
- documented populations
- traceable seed and RNG configuration

---

### 13.3 Scientific Report

Every scientific benchmark table additionally reports, where applicable:

```text
Agent
Population
N profiles
N runs
Hands/profile
Total hands
Statistical unit
Mean EV/100
95% CI
Median EV/100
R10
R10 CI
Seed count
Configuration hash
Bot version
Opponent version
Failed runs
Failure policy
```

Pairwise comparisons additionally report mean paired difference, 95% CI, number of independent pairs/clusters, effect size, practical threshold, p-value if confirmatory, and multiplicity-adjusted p-value if applicable.

The report examines in particular:

- Population performance
- Robustness
- Leduc exploitability
- Relationship between Leduc exploitability and NLHE performance
- Effects of bet‑sizing abstraction
- Effects of opponent modeling
- Trade‑off between exploitative and robust play
- Pareto front of agents
- Policy calibration
- Generalisation to holdout populations
- Failure modes and limitations of the methodology

---

### 13.4 Replay Viewer

A minimalistic HTML output visualises critical hands.

Display:

- Hole cards
- Board
- Position
- Stacks
- Pot
- Action history
- Available actions
- Chosen action
- Estimated hand strength
- Resulting profit
- Estimated decision EV, if applicable

---

## 14. Scientific Hypotheses

The framework shall not only produce software, but enable testable hypotheses.

### H1 – Variance Reduction

The paired CRN method produces narrower confidence intervals for the same number of played hands compared to independent random experiments.

Specifically, define

$$
VR = 1 - \frac{Var(\widehat{\Delta}_{paired})}{Var(\widehat{\Delta}_{independent})}.
$$

H1 predicts $VR\ge0.30$. The independent baseline uses the same total number of simulated hands but without paired CRNs: the two agents are evaluated in separate runs with distinct seeds and independent chance realisations. The paired design uses the common deal schedule defined in Section 4.1. The 30% threshold is a project-specific hypothesis, not a universal CRN property.

### H2 – Population Dependence

The ranking of agents by population performance differs systematically across opponent populations. The primary analysis uses a pre-specified rank statistic such as Kendall’s $\tau$ or Spearman’s $\rho$, and/or an agent-by-population interaction model.

### H3 – Robustness Trade‑off

Average population performance and robustness are not assumed to be equivalent. The relationship between $\overline{EV}_{NLHE}$ and $R_{10}$ is tested with a pre-specified correlation or regression model; a trade-off is not assumed a priori.

### H4 – Opponent Modeling

An agent with online opponent modeling achieves higher performance against unknown parametric opponents after sufficient observation time than an otherwise identical non‑adaptive agent.

The **100-hand result is a cumulative adaptation analysis over hands 1–100**, not performance in the 100th hand, not performance beginning after 100 hands, and not a post-100-hand steady-state estimate. It is reported as a learning-curve endpoint and should not by itself be treated as the primary confirmatory evidence for a small EV effect.

For the 100-hand endpoint:

$$
\Delta_{100} = \widehat{EV}_{100}^{adaptive} - \widehat{EV}_{100}^{nonadaptive}.
$$

The cumulative estimator is based on hands 1–100. A longer evaluation horizon is used for the primary confirmatory performance comparison once its sample size is determined by the applicable power/precision analysis. Other horizons (10, 20, 50, 100, 500 hands) remain secondary adaptation analyses.

### H5 – Leduc–NLHE Predictive Association

Lower Leduc best-response exploitability is associated with higher robustness and/or performance in the abstracted NLHE environment. H5 is a **predictive/associational** hypothesis: it asks whether Leduc exploitability explains or predicts variation in NLHE outcomes across independent agent variants. It does not assert causal transfer between games.

The primary analysis is performed at the agent-variant level using Spearman rank correlation:

$$
\rho_S = corr_{Spearman}(X_{Leduc}, \overline{EV}_{NLHE})
$$

and separately: $\rho_S = corr_{Spearman}(X_{Leduc}, R_{10})$

The exact primary statistic is frozen before final analysis.

Correlation of individual opponent samples with repeated $X_{Leduc}(\pi)$ is not used as the primary analysis.

### H6 – Abstraction Sensitivity

The measured performance and possibly the ranking of agents may depend on the chosen bet‑sizing abstraction.

### H7 – Policy Calibration Acceptance Criterion

H7 is treated as a validation/acceptance criterion rather than a conventional inferential hypothesis. A parametric opponent profile is accepted only if its required observed behavioural frequencies are compatible with the predefined target tolerances under the fixed calibration environment. This is a prerequisite for interpreting the synthetic opponent profiles as controlled benchmark opponents.

These hypotheses and validation criteria are not assumed as results, but evaluated according to their pre-specified roles.

---

## 15. Statistical Multiplicity and Confirmatory Status

The framework distinguishes **confirmatory** hypotheses from exploratory analyses. The confirmatory family is declared before the final experiment and includes only hypotheses with a pre-specified primary endpoint and inferential procedure. The default project family is H1–H4 if all four are formally tested confirmatorily; H5–H6 remain secondary/exploratory unless explicitly promoted before final data collection. H7 is a validation acceptance criterion and is not included in the inferential multiplicity family.

For the confirmatory family, the project uses Holm correction unless a different multiplicity procedure is explicitly pre-specified. The report stores:

```text
confirmatory_hypotheses
primary_endpoint_per_hypothesis
alpha
family_of_tests
multiplicity_correction
exploratory_analyses
```

Exploratory ablations, alternative populations, and sensitivity analyses are not converted into confirmatory claims solely because a nominal $p<0.05$ is observed.

Every statistical result reports the effect estimate, confidence interval, sample/cluster size, and practical relevance, not significance alone.

---

## 16. Reproducibility

Every published result must be reproducible via a complete experiment definition.

At a minimum, the following are required:

```text
Experiment ID
Game Configuration
Bet Abstraction
Bot Version
Opponent Version
Opponent Parameters
Population Distribution
Population Version
Master Seed
RNG Configuration
Engine Version
Python Version
Dependency Lockfile (Poetry)
Poetry Lock Hash
Git Commit
CPU Model
CPU Core Count
CPU Architecture
OS Identifier
Sample Size
Power Target
Precision Target
Statistical Method
Calibration Results
```

An experiment receives a unique configuration hash. The hash is a SHA-256 digest of a canonical, sorted, normalised JSON representation of the frozen configuration.

Python dependencies are locked with **Poetry** using `pyproject.toml` and `poetry.lock`. The lockfile state is stored with every experiment configuration.

A result is considered reproducible only if a second run with identical configuration produces the same raw data or deterministically identical results.

For non‑deterministic extensions, the reproducibility of the defined statistical distribution and the experimental design is documented instead.

---

## 17. Limitations of the Methodology

The framework deliberately does not claim to fully represent the strategic quality of a real‑world poker agent.

In particular, the following limitations apply:

1. The NLHE environment uses a discrete bet‑sizing abstraction.
2. The benchmark initially does not use real poker‑room rules.
3. The baseline benchmark uses no rake.
4. Parametric opponents are models and not a complete representation of real players.
5. Leduc exploitability is not a direct NLHE exploitability.
6. Results against a synthetic population cannot be automatically transferred to human players.
7. A large number of simulated hands does not compensate for systematic modelling errors.
8. The synthetic opponent populations cannot capture the strategic diversity of human players.
9. The results initially apply only to the specific bet‑sizing abstraction and cannot be directly transferred to other abstractions.
10. Leduc exploitability is not a substitute for NLHE exploitability; it serves exclusively as a proxy for empirical correlation analyses.
11. The quality of the population performance depends on the quality and calibration of the parametric opponent policy.
12. The holdout evaluation can measure true generalisation only if the holdout population was not used for optimisation during development.
13. A statistically significant effect is not automatically practically or strategically relevant.
14. The correlation between Leduc and NLHE may be limited by the small number of agents examined.
15. CRNs reduce variance, but neither eliminate card variance nor policy variance completely.
16. Benchmark runtime and throughput depend on the physical reference hardware and are not scientific measures of agent quality.

These limitations are explicitly discussed in the scientific report.

---

## 18. Possible Extensions

After completion of the MVP:

- Extension of bet abstraction
- Sensitivity analysis of different action sets
- Extension to 6‑max
- More realistic stack sizes
- Rake modelling
- Empirically calibrated opponent populations
- Stronger opponent models
- CFR variants
- NFSP and other RL methods
- Cython/Rust port (if the Python engine cannot achieve the required benchmark size, e.g., >100M hands, within acceptable time, e.g., <24 hours; the port would be limited to the core engine – hand ranking, pot calculation – while agent logic remains in Python)
- Acceleration to >1M simulated hands/second
- Comparison with historical poker agents
- Connection to external game environments
- Investigation of non‑stationary opponents
- True agent‑specific adversarial populations
- Hierarchical statistical models for nested opponent/agent data
- Sequential design methods for adaptive experiment runtimes

A connection to real poker rooms, if implemented at all, will be done exclusively in compliance with the respective terms of use and not as part of the scientific baseline benchmark.

---

## 19. Conclusion

The project does not develop a primary winning bot, but a reproducible evaluation framework for poker agents.

The central research question is:

> **What properties make a strategy in an imperfect game not only strong against known opponents, but simultaneously robust, reproducible, and difficult to exploit?**

For this purpose, three dimensions are separated:

1. **Population Performance** – average empirical EV estimator.
2. **Robustness** – lower quantiles over a defined opponent population.
3. **Game‑Theoretic Quality** – exploitability in Leduc.

Particularly relevant is the empirical investigation of whether Leduc exploitability correlates with NLHE performance or robustness.

The correlation is analysed at the agent level to avoid pseudoreplication through repeated opponent samples.

Another central component is the **calibration of the parametric opponents**. VPIP, PFR, 3‑bet, and other parameters are defined as target quantities and subsequently validated against the actually observed frequencies.

This prevents seemingly controlled opponent profiles from actually implementing a different strategy than their parameters suggest.

The first milestone is not a strong agent, but a deterministic, validated, reproducible, and statistically defensible evaluation infrastructure.

The project therefore consistently follows the principle:

> **Methodology before AI. Evaluation Framework first, Bot second.**

---

## 20. First Technical Milestone

The first milestone is the development of the deterministic paired-match engine and its surrounding validation infrastructure. It is a concrete technical milestone for the project.

**No AI optimisation begins until the following conditions are met:**

- [ ] Repository and Poetry environment initialised
- [ ] Canonical configuration and SHA-256 hashing implemented
- [ ] Engine rules fully specified
- [ ] Bet/raise semantics formally defined, versioned, and covered by unit tests
- [ ] Hand evaluation validated
- [ ] Pot and stack computation validated
- [ ] Pre-defined deal schedule and RNG hierarchy reproducible
- [ ] Position compensation implemented
- [ ] Random‑vs‑random test passed
- [ ] Symmetry test passed
- [ ] Determinism test passed
- [ ] Engine regression tests passed
- [ ] Hidden-information safety test passed
- [ ] Illegal-action handling validated
- [ ] Agent budget/timeout policy validated
- [ ] Policy calibration tests passed
- [ ] Statistical evaluation validated
- [ ] Power and precision planning implemented
- [ ] Experiment configuration fully persisted
- [ ] Replay of any match is possible

Side‑pot and multiplayer tests will only become mandatory engine gates with the later 6‑max extension.

Only when these conditions are met and reproducible results for identical and random bots are available does the implementation of more complex AI algorithms begin.

Thus, the project consistently adheres to the principle:

> **Evaluation Framework first, Bot second.**

---

*End of Document*
