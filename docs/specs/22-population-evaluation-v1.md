# Population Evaluation v1

## Status and identities

This document freezes the first Phase-1 population-performance and robustness evaluator required by the evaluation model.

```text
population_evaluation_version = population-evaluation-v1
population_weight_scale       = 1000000
population_quantile_version   = weighted-inverse-cdf-v1
population_bootstrap_version  = nested-profile-run-bootstrap-v1
population_percentile_method  = linear-n-minus-1-v1
population_confidence_level   = 0.95
population_bootstrap_default  = 10000
```

The evaluator consumes already completed matchup evidence. It does not execute poker hands and does not tune policies.

## Executable Phase-1 static populations

Only population definitions that are numerically complete in the existing specification are executable in v1.

### Balanced static population

```text
population_id      = balanced-static-profiles
population_version = balanced-static-profiles-v1
Nit                 = 0.20
TAG                 = 0.20
LAG                 = 0.20
Calling Station     = 0.20
Maniac              = 0.20
```

### Baseline synthetic static population

```text
population_id      = baseline-synthetic-static-profiles
population_version = baseline-synthetic-static-profiles-v1
Nit                 = 0.10
TAG                 = 0.35
LAG                 = 0.30
Calling Station     = 0.15
Maniac              = 0.10
```

Weights are stored as integer parts per million and must sum exactly to 1,000,000.

The Stress Population and Randomized Population are **not executable in v1** because the active specification gives examples but does not freeze a complete finite stress set or complete joint continuous parameter distribution. Development/validation/holdout population instances likewise require a later experiment-specific freeze. The evaluator itself is generic enough to consume a future fully specified population definition.

## Input evidence

For each opponent profile, the evaluator receives one or more independent completed run-level EV estimates in BB/100 and the simulated-hand count represented by each run.

For a complete non-adaptive `PairedResult`, the existing position-controlled paired difference is

```text
d = ((profit_a_btn - profit_b_btn) + (profit_a_bb - profit_b_bb)) / 2
```

The baseline HU game is zero-sum with no rake, so the position-compensated player-A profit is exactly `d / 2`. Therefore the run-level player-A empirical EV estimator is

```text
hero_ev_bb100 = 50 * mean(paired_differences)
```

This conversion is valid only for complete `paired_deal` evidence from the frozen heads-up zero-sum baseline. Failed or incomplete paired results are rejected by the convenience conversion rather than silently included.

## Point estimators

Let each profile-level EV be the arithmetic mean of its independent run-level EV estimates. Population point estimates are computed over these profile-level EVs with the frozen population weights.

The report contains:

- weighted mean EV/100;
- weighted median EV/100;
- weighted 10%-quantile `R10`;
- weighted 25%-quantile;
- weighted population standard deviation;
- empirical minimum profile EV.

Weighted quantiles use the inverse empirical CDF: sort profiles by EV and return the smallest EV whose cumulative integer population weight is at least the requested quantile threshold.

## Nested bootstrap

The 95% confidence intervals for population mean and `R10` use the following deterministic nested bootstrap:

1. draw exactly `N_profiles` opponent-profile observations with replacement from the frozen population probabilities;
2. for every selected profile, draw one of that profile's run-level EV estimates with replacement;
3. compute the arithmetic mean of the sampled profile observations;
4. compute the sample `R10` by the inverse empirical CDF on those sampled observations;
5. repeat, by default, 10,000 times;
6. report the 2.5% and 97.5% percentiles using `linear-n-minus-1-v1` interpolation.

A local RNG seeded only by the explicitly supplied bootstrap seed is used. Bootstrap randomness cannot alter engine, deal, or agent RNG state.

## Tail-precision flag

The active evaluation specification requires small profile counts to be flagged as insufficient for precise tail estimation and elsewhere requires at least 10 independent opponent samples per agent for important aggregated analyses. v1 therefore records

```text
tail_precision_warning = true  if N_profiles < 10
```

This is a reporting flag, not an acceptance/failure criterion.

## Reproducibility

The population definition has a canonical SHA-256 hash. The complete input evidence has a canonical SHA-256 hash. The population evaluation has a canonical report hash and may be saved atomically as canonical JSON. Paths and wall-clock timestamps are excluded from scientific identity.
