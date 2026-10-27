# RNG, Variance Reduction, Position Pairing, and Statistical Units

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
