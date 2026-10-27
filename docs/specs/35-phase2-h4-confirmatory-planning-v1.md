# Phase-2 H4 Confirmatory Planning v1

## 35. Status and Scope

This file is normative for the planning boundary between the completed `phase2-adaptive-pilot-v1` study and a future final confirmatory test of H4.

Planning identifier:

```text
phase2-h4-confirmatory-planning-v1
```

Planning version:

```text
phase2-h4-confirmatory-planning-protocol-v1
```

Status:

```text
planning_frozen_prerequisites_not_yet_satisfied
```

This specification does **not** authorize final H4 data collection. It freezes the primary endpoint class, practical-effect scale, power/precision targets, inferential method, data-separation rules, and the prerequisites that must be resolved before a final sample size and seed schedule can be frozen.

Specs 27–34 remain unchanged.

---

## 35.1 Scientific Question

H4 remains:

> An agent with online opponent modeling achieves higher performance against unknown parametric opponents after sufficient observation time than an otherwise identical non-adaptive agent.

The treatment/control pair remains exactly:

```text
treatment = adaptive-hand-strength-agent-v1
control   = hand-strength-baseline-v1
```

No policy rule, posterior update, posterior-to-range mapping, lifecycle rule, or action rule may change as part of H4 planning.

The primary estimand is the population-level adaptive-minus-control matched trajectory contrast in BB/100.

---

## 35.2 Binding Completed Pilot Evidence

Planning is based on the completed and integrity-validated adaptive pilot:

```text
pilot_protocol_id   = phase2-adaptive-pilot-v1
pilot_protocol_hash = ef86d3da94369a86ac1c496ddf981b39d30919e086b35e026039bac3afc8ccb4
pilot_report_hash   = 944b103894ec04e980efa3dc1ae2e50d578fe772996ab1bdb51ec2efb4b5415c
pilot_binding_hash  = e32d381f77320ac2d17f63d58f8c98ef443287eda62ae5bd6e7347e00f0d3d8e
```

The completed pilot contains:

```text
50 / 50 completed jobs
25 / 25 valid matched treatment/control pairs
0 failed jobs
0 invalid pairs
18,000 recorded engine hands
9,000 replay-validated treatment learning transitions
```

The only pilot quantities permitted to influence final H4 design are:

```text
trajectory-level variance evidence
runtime evidence
completeness / failure evidence
non-outcome learning-model stabilization diagnostics
```

The observed pilot treatment-effect mean, its sign, its apparent peak, and profile-specific outcome means are prohibited as inputs to the primary horizon, practical-effect threshold, population definition, alpha/beta, or final sample-size rule.

For traceability, the completed pilot provides replicated population-level variance only through 100 hands. At the Baseline Synthetic 100-hand endpoint:

```text
variance_estimator       = sample-sd-matched-trajectory-contrast-bb100-ddof1-v1
num_independent_clusters = 5
pilot_sd_bb100           = 58.31335019139056
```

At 500 hands the pilot has only one predeclared seed and therefore provides **no** valid variance, standard error, confidence interval, or power-planning estimate at that horizon.

---

## 35.3 Primary Confirmatory Horizon

The final H4 primary endpoint is frozen as:

```text
primary_h4_horizon = 500 cumulative hands per positional trajectory
```

The primary arm estimator for arm `a`, opponent unit `p`, seed `s`, and horizon 500 remains:

$$
Y_{a,p,s}(500)
=
100\cdot\frac{G^{BTN}_{a,p,s}(500)+G^{BB}_{a,p,s}(500)}{1000}.
$$

The matched adaptive-control contrast remains:

$$
D_{p,s}(500)
=
Y_{adaptive,p,s}(500)-Y_{control,p,s}(500).
$$

The 500-hand choice is made because:

1. active H4 explicitly requires the primary confirmatory comparison to use a horizon longer than the cumulative 100-hand learning-curve endpoint;
2. 500 is the already-frozen long adaptation horizon and contains the full frozen 101–500 steady-state segment;
3. the pilot learning diagnostics show that the opponent model is already materially informative by 100 hands, making a longer post-observation evaluation scientifically meaningful;
4. the choice is **not** based on the observed adaptive-minus-control pilot mean at 500 or any other horizon.

The following remain secondary and non-primary in the final H4 study:

```text
10-hand cumulative
20-hand cumulative
50-hand cumulative
100-hand cumulative
hands 1-20 Early Adaptation
hands 21-100 Main Adaptation
hands 101-500 Steady State
```

Secondary endpoints cannot replace the primary endpoint after final data are observed.

---

## 35.4 Final Population Role: Holdout/Test Required

The final H4 primary population role is frozen as:

```text
final_h4_population_role = holdout_test
```

The concrete holdout population identity, parameter distribution, support, weights, and hash are **not yet defined** by the active specification and therefore cannot be invented in implementation.

Before any long-horizon H4 variance pilot or final H4 hand is generated, a separate frozen holdout-population specification must define at least:

```text
population_id
population_version
population_hash
parameter schema and support
sampling or finite-support rule
population weights, if finite weighted support is used
parametric-policy version
calibration / acceptance rule
sealed generation seed(s), if sampling is used
```

The holdout population must satisfy all of the following:

1. it is sealed before any H4 holdout outcome is inspected;
2. it was not used to tune the adaptive policy, posterior mapping, action thresholds, or H4 decision rule;
3. the agent never receives the holdout identity or true parameter vector;
4. any controller-side true parameters are provenance only;
5. the five development-profile cells used in `phase2-adaptive-pilot-v1` are not silently relabeled as holdout evidence;
6. no holdout result may be used for additional agent optimization before the confirmatory H4 decision is frozen and executed.

The Baseline Synthetic and Balanced populations from the adaptive pilot remain development/validation planning evidence. They are not promoted to H4 holdout evidence by this specification.

---

## 35.5 Minimum Relevant Effect and Precision Target

The minimum practically relevant population-level adaptive advantage is frozen **before holdout data collection** as:

```text
minimum_relevant_difference_bb100 = 50.0
```

This value is not derived from the observed adaptive-pilot treatment effect. It uses the already-established initial project benchmark resolution scale of 50 BB/100 as the smallest effect that this first confirmatory H4 study will label practically relevant.

The final H4 study also freezes:

```text
confidence_level                     = 0.95
precision_target_half_width_bb100    = 50.0
confirmatory_alpha                   = 0.05
confirmatory_beta                    = 0.20
confirmatory_power                   = 0.80
```

The precision target applies to the primary population-level adaptive-minus-control mean contrast only. It does not imply equivalent precision for profile-specific estimates, robustness quantiles, or secondary horizons.

---

## 35.6 Primary Statistical Unit and Population Aggregation

Because adaptation makes hands within a trajectory dependent, the final primary independent unit remains a complete matched trajectory cluster.

For a frozen finite weighted holdout population with weights $w_p$, population aggregation must occur within master seed first:

$$
D_{population,s}(500)
=
\sum_p w_pD_{p,s}(500).
$$

Variance and inference are then taken only across independent master-seed population contrasts.

If the future holdout specification instead uses sampled opponent units rather than fixed finite weighted support, that specification must define an equivalent independent-cluster estimand before the long-horizon variance pilot begins. Individual hands, decisions, positional halves, profiles, or posterior updates must never be promoted to independent replicates.

---

## 35.7 Long-Horizon Variance Requirement

A final H4 sample size cannot yet be frozen because the active pilot has no replicated 500-hand variance and the final holdout population is not yet specified.

Therefore, before final H4 execution, a separately versioned **500-hand holdout variance pilot** is mandatory.

That variance pilot must:

1. use the exact frozen H4 treatment/control pair;
2. use the exact subsequently frozen H4 holdout population;
3. use 500 hands per positional trajectory;
4. reset adaptive state independently before both positional trajectories;
5. preserve the established matched CRN relationship;
6. use master seeds disjoint from both `phase2-adaptive-pilot-v1` and the later final H4 dataset;
7. use no replacement seeds;
8. estimate variance only from complete scientifically valid matched trajectory clusters;
9. be used only for variance, runtime, completeness, and sample-size planning;
10. prohibit its observed mean treatment effect from changing the horizon, population, $\delta$, alpha, beta, precision target, or decision rule frozen here.

The exact variance-pilot cluster count and seed schedule must be frozen in that separate protocol before its first hand is executed.

No variance estimate from the single 500-hand seed in `phase2-adaptive-pilot-v1` may substitute for this requirement.

---

## 35.8 Deterministic Final Sample-Size Rule

Let

```text
sigma_500
```

denote the sample standard deviation of the independent primary-population matched trajectory contrasts at 500 hands from the future frozen long-horizon variance pilot.

The variance estimator is:

```text
sample-sd-matched-population-trajectory-contrast-bb100-ddof1-v1
```

The normal planning constants are frozen as:

```text
z_0.975 = 1.9599639845400536
z_0.80  = 0.8416212335729143
```

Power planning uses the project paired-comparison approximation:

$$
n_{power,raw}
=
\frac{(z_{0.975}+z_{0.80})^2\sigma_{500}^2}{50^2}.
$$

Precision planning uses:

$$
n_{precision,raw}
=
\left(\frac{z_{0.975}\sigma_{500}}{50}\right)^2.
$$

The final independent-cluster count will be frozen as:

$$
n_{final}
=
\max\left(
10,
\lceil n_{power,raw}\rceil,
\lceil n_{precision,raw}\rceil
\right).
$$

The floor of 10 independent primary clusters is a pre-data design safeguard for this first confirmatory adaptive study and is not derived from pilot outcome means.

The future final H4 protocol must record:

```text
sigma_500
variance-pilot report hash
n_power_raw
n_precision_raw
n_final
planning half-width after rounding
exact final master-seed schedule
exact planned engine-hand count
```

These values are fixed before the first final H4 hand and are never recomputed to extend or stop the final study based on observed final outcomes.

---

## 35.9 Confirmatory Test and Decision Rule

The primary final H4 test is frozen as a two-sided paired/one-sample trajectory-cluster t-test of the independent population-level matched contrasts:

$$
H_0:\mu_D=0
\qquad\text{versus}\qquad
H_1:\mu_D\ne0.
$$

Because H4 predicts an adaptive advantage, statistical support for H4 additionally requires the estimated mean contrast to be positive.

The primary report must include:

```text
mean adaptive-minus-control difference in BB/100
sample SD across independent trajectory clusters
95% Student-t confidence interval
raw two-sided p-value
independent cluster count
standardized paired effect size dz
```

A 10,000-resample bootstrap over complete independent trajectory clusters is reported as a sensitivity interval and is not the primary decision rule.

The confirmatory statuses are separated:

```text
h4_statistical_support =
    (mean_difference_bb100 > 0)
    AND (two_sided_p_value < 0.05)

h4_practical_relevance =
    (mean_difference_bb100 >= 50.0)

h4_confirmatory_success =
    h4_statistical_support
    AND h4_practical_relevance
```

A statistically positive effect below 50 BB/100 is reported as statistically supported but below the frozen practical-relevance threshold. A point estimate above 50 BB/100 without statistical support is not a confirmatory success.

---

## 35.10 Multiplicity

For this first formally specified H4 experiment, the confirmatory family is frozen as:

```text
confirmatory_hypotheses = [H4]
family_of_tests         = phase2-h4-singleton-family-v1
multiplicity_correction = holm-v1
family_alpha            = 0.05
```

Holm correction on a one-hypothesis family leaves the H4 threshold at 0.05 while retaining the project's standard multiplicity mechanism.

H1–H3 have not been promoted to confirmatory tests by the completed Phase-1 baseline experiment. If a later project protocol promotes additional hypotheses into a joint project-level confirmatory family, it must be frozen before those additional final data are collected and must preserve the raw H4 result and provenance rather than altering this experiment after the fact.

All profile-level, Balanced-population, learning-curve, segment, and alternative-horizon analyses are secondary unless a later specification explicitly defines a separate confirmatory family before data collection.

---

## 35.11 Data Separation and Seed Domains

The following data domains must remain disjoint:

```text
completed adaptive pilot:
    6,300,000 ... 6,300,004

future 500-hand holdout variance pilot:
    separately frozen disjoint range

final H4 confirmatory dataset:
    separately frozen disjoint range
```

No pilot or variance-pilot trajectory may be pooled into the final H4 estimator, confidence interval, bootstrap, or p-value.

The final seed schedule must be frozen only after `n_final` is computed, but before any final H4 hand is executed.

Replacement seeds remain forbidden.

---

## 35.12 Optional Stopping and Failure Handling

Ordinary optional stopping is prohibited.

The final H4 experiment must not:

- stop because the ordinary confidence interval becomes significant;
- stop because the observed mean exceeds 50 BB/100;
- extend because the observed mean is smaller than expected;
- extend because the ordinary confidence interval is wider than 50 BB/100;
- replace failed seeds;
- change the holdout population after outcome inspection.

A failed frozen trajectory remains failed evidence under its original identity. The future final protocol must state whether any failure invalidates the confirmatory campaign or whether a fixed complete-case rule is scientifically sufficient; that rule must be frozen before final execution.

---

## 35.13 Execution and Provenance Boundary

This planning specification does not authorize:

```text
holdout generation
long-horizon variance-pilot execution
final H4 execution
final H4 inference
```

Before any of those occur, the corresponding versioned protocols must bind the reproducibility metadata required by Spec 14 and the existing adaptive execution stack, including protocol/population hashes, Git commit, lockfile hash, Python/runtime identity, hardware/OS provenance, RNG configuration, master seeds, trajectory horizon, and all adaptive dependency hashes.

The existing serial/parallel adaptive pilot execution authorities in Specs 33–34 do not automatically authorize H4 variance-pilot or final-H4 data generation.

---

## 35.14 Required Next Specifications

The next normative steps are, in order:

1. **H4 holdout population specification** — freeze the unseen parametric test population and its calibration/acceptance boundary.
2. **H4 500-hand variance-pilot protocol** — freeze replicated holdout trajectory clusters and seed schedule for `sigma_500` estimation only.
3. **Final H4 confirmatory experiment protocol** — consume the frozen variance-pilot report, compute `n_final` by Section 35.8, freeze the final disjoint seed schedule and exact execution matrix.
4. **Final H4 execution authority** — bind provenance, resume/checkpoint rules, and launch readiness before any confirmatory hand is generated.

Implementation must not skip directly from the completed adaptive pilot to final H4 execution.

---

## 35.15 Validation Gate

Before this planning specification is treated as closed, validation tests must verify at minimum:

1. Spec-35 identity/status and canonical documentation index entry;
2. exact binding to the completed adaptive-pilot protocol/report hashes;
3. the primary horizon is exactly 500 and cannot be 100 or less;
4. the final population role is holdout/test and no development population is silently promoted;
5. `minimum_relevant_difference_bb100 = 50.0` is frozen independently of pilot means;
6. alpha/beta/power and 50-BB/100 precision target are exact;
7. the primary statistical unit remains the independent matched trajectory cluster;
8. the single 500-hand pilot seed is rejected as a variance source;
9. a separately versioned replicated 500-hand holdout variance pilot is mandatory;
10. the deterministic power/precision sample-size formula and 10-cluster floor are exact;
11. the final H4 data are disjoint from both prior pilot domains;
12. ordinary optional stopping and replacement seeds are prohibited;
13. the primary test/CI/bootstrap definitions are exact;
14. Holm multiplicity metadata for the singleton H4 family are exact;
15. profile/segment/alternative-horizon analyses remain secondary;
16. no H4 execution is authorized by Spec 35 itself.

---

## 35.16 Frozen Planning Boundary

After this file is active, the following choices may not be changed because of any subsequently observed holdout variance-pilot or final-H4 treatment effect:

```text
primary_h4_horizon                    = 500
final_h4_population_role              = holdout_test
minimum_relevant_difference_bb100     = 50.0
precision_target_half_width_bb100     = 50.0
confirmatory_alpha                    = 0.05
confirmatory_beta                     = 0.20
primary_test                          = two-sided trajectory-cluster t-test
bootstrap_resamples                   = 10,000
multiplicity_correction               = holm-v1
optional_stopping                     = prohibited
replacement_seed_policy               = forbidden
```

Changing any of these requires a new versioned planning protocol and may not use already-inspected holdout outcomes as justification.
