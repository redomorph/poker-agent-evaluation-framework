# Phase-2 H4 500-Hand Holdout Variance Pilot v1

## 37. Status and Scope

This file is normative for the replicated long-horizon holdout variance study required by Specs 35-36 before the final H4 confirmatory sample size may be frozen.

```text
protocol_id       = phase2-h4-holdout-variance-pilot-v1
protocol_version  = phase2-h4-holdout-variance-pilot-protocol-v1
status            = design_frozen_execution_not_yet_authorized
purpose           = holdout_long_horizon_variance_runtime_and_final_h4_sample_size_planning_only
```

The variance pilot is **not** the final H4 confirmatory experiment and is not eligible for confirmatory H4 inference. Its observations MUST NOT be pooled into the later final H4 dataset.

This protocol freezes the exact replicated cluster count, disjoint seed schedule, 500-hand trajectory matrix, holdout population, treatment/control relationship, CRN relationship, all-or-nothing validity rule, variance estimator, permitted planning outputs, and data-separation boundary.

It does **not** authorize execution. A canonical machine-readable companion and a separately versioned execution authority/readiness gate remain mandatory before the first holdout-variance hand is generated.

The future machine-readable companion is:

```text
configs/experiments/phase2-h4-holdout-variance-pilot-v1.json
```

Specs 27-36 remain unchanged.

---

## 37.1 Scientific Purpose and Prohibited Uses

The only scientific uses of this study are:

```text
500-hand primary-population variance estimation
runtime estimation
storage / replay / completeness evidence
mechanical final-H4 sample-size planning under Spec 35
```

The variance pilot MUST NOT be used for:

```text
confirmatory H4 inference
final H4 effect estimation
policy or model tuning
holdout-population tuning
primary-horizon selection
MRD / alpha / beta / precision-target changes
optional stopping
pilot-plus-final pooling
variance-pilot-plus-final pooling
seed replacement
```

The observed adaptive-minus-control mean is not a permitted scientific design input. It may exist mechanically as the centering quantity needed to compute a sample standard deviation or as reconstructible provenance from seed-level contrasts, but it MUST NOT alter any design choice frozen in Specs 35-37.

---

## 37.2 Entry Gate and Frozen Bindings

No variance-pilot implementation or execution may proceed unless the Spec-36 holdout population and its official all-six-metric acceptance gate are green.

The exact holdout bindings are:

```text
population_id           = phase2-h4-holdout-population-v1
population_version      = phase2-h4-holdout-parametric-population-v1
population_role         = holdout_test
population_hash         = b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3
holdout_acceptance_hash = 9ba347fb1d615cec1d6458944df3c293dcac630564e8847a407e987bbb98af84
```

The population contains exactly:

```text
h4_holdout_000  weight 0.2
h4_holdout_001  weight 0.2
h4_holdout_002  weight 0.2
h4_holdout_003  weight 0.2
h4_holdout_004  weight 0.2
```

No unit may be removed, replaced, reweighted, regenerated, or substituted.

The treatment/control pair is exactly:

```text
control   = hand-strength-baseline-v1
treatment = adaptive-hand-strength-agent-v1
```

The treatment remains bound to the frozen scientific dependencies:

```text
adaptive-learning-protocol-v1
    payload_hash = 03f8ab5b2b35c86910d46df1d5187c306814e427b3f514eb85748a9a424adf5c
posterior-to-range-v1
    payload_hash = af19b040f9049ecd3a3a12615dcbe049e1d1f859546681487176238e9d2ed668
adaptive-hand-strength-policy-v1
    payload_hash = 76b6c561c2cb3715cc6352520dfbd3b0b01ce219a1b37d1a0acb62bb6ce10e4a
adaptive-lifecycle-v1
    payload_hash = bc7f26c66861a37dec8c544863a9e97fbcf33e07176fa2733e99f5426bf10edc
bayesian-opponent-model-v1
calibration-opportunities-v2
public-action-trace-v1
```

The opponent mechanism remains:

```text
parameter_schema_version  = opponent-parameters-v1
parametric_policy_version = parametric-policy-v1
```

The game remains:

```text
game abstraction = baseline-v1
game_config_hash  = b1916b8c91e234bb0b4d4863aab601db8ccfd8b73e336a924310e13644bae625
```

Any scientific change to these dependencies after holdout variance outcomes are observed invalidates use of this variance study for final-H4 planning under this version.

---

## 37.3 Primary Statistical Unit

The independent statistical unit is the complete master-seed holdout-population trajectory cluster:

```text
statistical_unit = matched_population_trajectory_cluster
```

For each arm `a`, holdout unit `p`, and master seed `s`, one arm trajectory cluster contains two independent positional trajectories:

```text
Position run 1: hero BTN, opponent BB
Position run 2: opponent BTN, hero BB
```

Each positional trajectory contains exactly 500 hands.

For the adaptive treatment, the frozen `Beta(1,1)` state vector is reset independently before each positional run. No adaptive state is shared:

```text
between BTN and BB positional runs
between holdout units
between master seeds
between treatment and control
between this variance pilot and any prior/future experiment
```

The control remains non-adaptive but is executed through the same full-match two-position trajectory structure.

Individual hands, decisions, positional halves, unit-level contrasts, posterior updates, and learning transitions are not independent variance-pilot observations.

---

## 37.4 Fixed Replicated Matrix

The variance pilot freezes exactly:

```text
independent master-seed population clusters = 10
holdout units per seed                       = 5
hero arms                                    = 2
position runs per arm                        = 2
hands per positional run                     = 500
```

For each master seed, the complete population cluster therefore contains:

```text
5 holdout units
x 2 arms
x 2 positions
x 500 hands
= 10,000 engine hands
```

The complete fixed study contains:

```text
10 seeds
x 5 holdout units
= 50 matched treatment/control unit pairs

50 unit pairs
x 2 arms
= 100 arm trajectory jobs

100 arm jobs
x 2 positions
x 500 hands
= 100,000 engine hands
```

The fixed cluster count of 10 is a pre-outcome planning choice. It provides replicated 500-hand variance evidence with nine variance degrees of freedom while keeping this study a bounded planning campaign rather than turning it into the final H4 experiment. It is also consistent with Spec 35's independently frozen minimum final-cluster floor of 10.

The count MUST NOT be increased or decreased after inspecting treatment effects, variance, confidence intervals, runtime, failures, or posterior behavior. Additional long-horizon variance evidence would require a new versioned protocol and may not be silently pooled into `phase2-h4-holdout-variance-pilot-v1`.

---

## 37.5 Frozen Seed Schedule

The seed schedule is:

```text
seed_schedule_version = phase2-h4-holdout-variance-pilot-seeds-v1
master seeds           = 6,400,000 ... 6,400,009 inclusive
```

The exact ordered seed vector is:

```text
[6400000, 6400001, 6400002, 6400003, 6400004,
 6400005, 6400006, 6400007, 6400008, 6400009]
```

This domain is disjoint from:

```text
Phase-1 pilot:              6,100,000 ... 6,100,099
Phase-1 final:              6,200,000 ... 6,202,768
Adaptive development pilot: 6,300,000 ... 6,300,004
```

The later final H4 master-seed schedule must be separately frozen after `n_final` is calculated and must be disjoint from every seed above.

No variance-pilot seed may be reused in the final H4 dataset. Replacement seeds are forbidden.

---

## 37.6 CRN and Deal Relationship

The identical ordered seed schedule is reused across:

```text
all five holdout units
control and treatment
both positional conditions
```

For a fixed master seed, holdout unit, position, and hand index, treatment and control use the same deterministic defined card realization.

The RNG contract remains:

```text
deal_relation             = identical_defined_card_realisations
agent_rng_relation        = event_indexed_common_identity_where_events_correspond
full_action_path_identity = not_claimed
```

Reuse of the same master seed across the five holdout units intentionally induces within-seed dependence. This is scientifically admissible because the complete five-unit population contrast is aggregated **inside seed** and only the ten master-seed population contrasts are treated as independent observations.

Opponent behavior, treatment behavior, and control behavior may diverge after the common deal because their public action histories may differ. No stronger CRN identity is claimed.

---

## 37.7 Information Boundary

Spec 36's holdout information boundary remains exact.

The acting hero MUST NOT receive:

```text
holdout unit id
population id or role
population weight
true opponent parameter vector
generation seed material
latent pfr_ratio
calibration target
calibration acceptance result
controller-side classification
```

The opponent controller may receive the exact frozen unit parameters because those parameters define the opponent policy. Such information is controller-side provenance only.

The adaptive hero may learn only from the frozen public evidence contract. Replay validation MUST prove that no forbidden holdout metadata entered hero observation, learning evidence, or posterior state.

---

## 37.8 Arm, Unit, and Population Estimators

For arm `a`, holdout unit `p`, and master seed `s`, define the 500-hand position-compensated arm estimator:

$$
Y_{a,p,s}(500)
=
100\cdot\frac{G^{BTN}_{a,p,s}(500)+G^{BB}_{a,p,s}(500)}{1000}.
$$

The matched unit-level adaptive-minus-control contrast is:

$$
D_{p,s}(500)
=
Y_{adaptive,p,s}(500)-Y_{control,p,s}(500).
$$

The primary population contrast is computed within master seed first:

$$
D_{population,s}(500)
=
\sum_{p=0}^{4}0.2D_{p,s}(500).
$$

Exactly ten values are planned:

$$
D_{population,6400000}(500),\ldots,D_{population,6400009}(500).
$$

These ten complete master-seed population contrasts, and only these values, form the variance sample.

---

## 37.9 Variance Estimator

Let the ten complete population contrasts be $D_1,\ldots,D_{10}$ and let

$$
\bar D=\frac{1}{10}\sum_{i=1}^{10}D_i.
$$

The required long-horizon variance quantity is:

$$
\sigma_{500}
=
s(D_1,\ldots,D_{10})
=
\sqrt{\frac{1}{9}\sum_{i=1}^{10}(D_i-\bar D)^2}.
$$

The estimator identifier is exactly:

```text
sample-sd-matched-population-trajectory-contrast-bb100-ddof1-v1
```

The sample mean is only an internal centering quantity for the variance calculation and an auditable consequence of the persisted seed-level contrasts. It is **not** a variance-pilot H4 estimate and MUST NOT be used to change the frozen final design.

The variance-pilot scientific report MUST NOT produce:

```text
H4 p-value
H4 Student-t confidence interval
H4 bootstrap interval
H4 effect-size decision
H4 statistical-support status
H4 practical-relevance status
H4 confirmatory-success status
```

---

## 37.10 Mechanical Final-Sample-Size Planning

If and only if all ten complete population clusters are scientifically valid, the variance report may mechanically apply the already-frozen Spec-35 planning rule.

The constants remain:

```text
minimum_relevant_difference_bb100 = 50.0
precision_target_half_width_bb100 = 50.0
z_0.975                            = 1.9599639845400536
z_0.80                             = 0.8416212335729143
minimum_final_clusters             = 10
```

Compute:

$$
n_{power,raw}
=
\frac{(z_{0.975}+z_{0.80})^2\sigma_{500}^2}{50^2},
$$

$$
n_{precision,raw}
=
\left(\frac{z_{0.975}\sigma_{500}}{50}\right)^2,
$$

and

$$
n_{final}
=
\max\left(10,\lceil n_{power,raw}\rceil,\lceil n_{precision,raw}\rceil\right).
$$

For traceability, the normal-approximation planning half-width after integer rounding is:

$$
h_{planning}
=
\frac{z_{0.975}\sigma_{500}}{\sqrt{n_{final}}}.
$$

The variance-pilot report may persist these mechanical outputs, but they do **not** themselves authorize final H4 execution. A separate final H4 protocol must consume the immutable variance-pilot report hash, freeze the exact final seed schedule, exact engine-hand matrix, confirmatory analysis plan, and execution boundary before any final H4 hand is generated.

No observed variance-pilot mean enters any formula in this section.

---

## 37.11 Failure, Completeness, and No-Replacement Rule

Validity is hierarchical and all-or-nothing.

An **arm trajectory job** is valid only if:

```text
both 500-hand positional runs complete
no invalid action / disqualifying timeout contaminates the cluster
required persistence is complete
required replay passes
adaptive learning-chain replay passes for treatment
```

A **unit-level treatment/control pair** is valid only if both arm jobs for the same holdout unit and master seed are valid.

A **master-seed population cluster** is valid only if all five unit-level treatment/control pairs for that master seed are valid.

The complete variance pilot is scientifically admissible for `sigma_500` only if:

```text
planned arm jobs             = 100
completed valid arm jobs     = 100
planned unit pairs           = 50
valid unit pairs             = 50
planned population clusters  = 10
valid population clusters    = 10
invalid population clusters  = 0
```

If any planned population cluster is invalid:

1. its original seed identity is retained;
2. it is not replaced;
3. the remaining clusters may be reported only as operational diagnostics;
4. `sigma_500` is not admitted as the Spec-35 variance basis under this protocol;
5. `n_final` is not frozen from a reduced complete-case sample;
6. no additional seed is appended;
7. a new versioned variance-pilot decision is required before final H4 planning continues.

This strict rule prevents failures from creating a data-dependent variance-pilot sample size.

---

## 37.12 Replay and Analysis Admission

Replay is mandatory before any arm, pair, or population cluster is admitted to the variance calculation.

At minimum, replay/admission must verify:

```text
exact immutable job identity
exact master seed
exact holdout unit identity in controller-side provenance
exact frozen holdout parameter vector
exact population hash
exact holdout-acceptance hash
exact game config hash
exact treatment/control versions
exact adaptive dependency hashes
public-action reconstruction
adaptive prior reset at each positional boundary
adaptive before/after state hashes
learning-transition count and ordering
no forbidden holdout metadata in hero evidence
match result / snapshot / SQLite identity agreement
```

Variance analysis MUST consume only replay-admitted immutable trajectory evidence. It MUST NOT read a mutable live agent state or infer results directly from scheduler state.

---

## 37.13 Variance-Only Report Contract

The immutable post-run variance report must contain at minimum:

```text
report version
protocol id / version / protocol hash
execution-contract id / hash
binding hash
population id / version / population hash
holdout_acceptance_hash
game config hash
exact master-seed vector
planned/completed/failed arm-job counts
planned/valid/invalid unit-pair counts
planned/valid/invalid population-cluster counts
planned and recorded engine-hand counts
per-seed population contrasts for audit
variance estimator id
sigma_500
n_power_raw
n_power_ceil
n_precision_raw
n_precision_ceil
n_final
planning_half_width_bb100
runtime / storage / replay diagnostics
scientific-use restrictions
```

The report MUST explicitly encode that:

```text
confirmatory_h4_inference_permitted = false
variance_pilot_mean_design_use      = prohibited
pilot_final_pooling                  = false
replacement_seeds                    = forbidden
```

If the campaign is not completely valid under Section 37.11, the report must be marked incomplete/invalid for final-H4 sample-size planning and the scientific `sigma_500` / `n_final` fields must remain unset.

The report itself must be canonical, hash-bound, immutable once complete, and reproducible from admitted arm evidence.

---

## 37.14 Execution and Provenance Boundary

Spec 37 freezes scientific design only. It does not authorize execution.

Before the first variance-pilot hand, a separate execution authority must bind at least:

```text
Spec-37 protocol hash
Spec-36 population hash
holdout acceptance hash
all adaptive dependency hashes
game config hash
Git commit
poetry.lock hash
Python version/runtime
CPU model / architecture / core count
OS identifier
RAM
worker count and process model
RNG configuration
exact master-seed vector
persistence paths
replay versions
```

The execution layer may use parallel workers only if it preserves the same scientific identities and isolation guarantees already established by the adaptive-pilot parallel execution stack:

- each worker owns fresh hero/opponent/adaptive state;
- no mutable model or agent instance is shared across jobs;
- process/PID/queue/completion order does not enter scientific RNG or trajectory identity;
- parent-only durable persistence is preferred for shared campaign state;
- crash/resume behavior is fail-closed and cannot duplicate completed scientific jobs.

Worker count is an execution binding, not a scientific design input, and therefore is not frozen by this protocol.

---

## 37.15 Canonical Job Expansion

A future machine-readable planner must expand the scientific matrix in deterministic canonical order:

```text
holdout unit order -> master-seed order -> arm order
```

with:

```text
holdout unit order =
    h4_holdout_000
    h4_holdout_001
    h4_holdout_002
    h4_holdout_003
    h4_holdout_004

master-seed order = ascending 6,400,000 ... 6,400,009

arm order =
    control
    treatment
```

This produces exactly 100 arm trajectory jobs and 50 canonical treatment/control unit pairs.

Canonical planning order is an identity/reproducibility convention. Runtime completion order may differ under parallel execution and MUST NOT alter the final manifest identity, RNG semantics, estimator, or report.

---

## 37.16 Data Separation and Holdout Seal

The following domains remain scientifically separate:

```text
adaptive development pilot       = 6,300,000 ... 6,300,004
H4 holdout variance pilot        = 6,400,000 ... 6,400,009
final H4 confirmatory experiment = future separately frozen disjoint seeds
```

No trajectory from the adaptive development pilot or this variance pilot may enter the final H4 mean, SD, confidence interval, bootstrap, effect size, or p-value.

After the holdout seal, variance-pilot outcomes may not tune:

```text
adaptive agent
control agent
Bayesian model
posterior-to-range mapping
adaptive lifecycle
parametric opponent policy
holdout population
500-hand primary horizon
50 BB/100 MRD
50 BB/100 precision target
alpha / beta
confirmatory decision rule
```

Only `sigma_500`, runtime/completeness evidence, and the deterministic Spec-35 sample-size outputs may flow forward into the final H4 protocol.

---

## 37.17 Required Machine-Readable Validation Gate

Before Spec-37 execution can be authorized, tests-first validation must verify at minimum:

1. exact protocol id, version, status, and purpose;
2. exact binding to Specs 35-36;
3. exact `population_hash` and `holdout_acceptance_hash`;
4. exact five holdout units and equal weights;
5. exact treatment/control and adaptive dependency hashes;
6. exact 500-hand two-position trajectory structure;
7. exact 10-cluster seed schedule `6,400,000 ... 6,400,009`;
8. exact 100-job / 50-pair / 10-population-cluster / 100,000-hand matrix;
9. disjointness from Phase-1 and adaptive-pilot seed domains;
10. exact CRN claim and no stronger full-action-path identity;
11. independent adaptive reset before every positional trajectory;
12. exact within-seed equal-weight population aggregation;
13. exact ddof=1 `sigma_500` estimator over ten complete seed-level population contrasts;
14. exact Spec-35 power/precision formula and planning half-width formula;
15. all-or-nothing 10/10 population-cluster admission;
16. failure cannot trigger replacement or reduced-sample `sigma_500` planning;
17. replay is mandatory before analysis admission;
18. holdout identity/parameters remain outside hero observations and learning evidence;
19. confirmatory inference and observed-mean design use are prohibited;
20. pilot/final pooling is prohibited;
21. Spec 37 itself does not authorize execution.

---

## 37.18 Required Next Steps

After the canonical machine-readable Spec-37 protocol is green, implementation must proceed in this order:

1. variance-pilot analysis primitives for within-seed population contrast, `sigma_500`, and deterministic Spec-35 sample-size planning;
2. variance-pilot execution enforcement and replay admission;
3. binding/provenance and resumable persistence;
4. separately versioned variance-pilot execution authority/readiness gate;
5. actual 100,000-hand variance-pilot execution;
6. immutable post-run variance report;
7. only then, a separate final H4 confirmatory experiment protocol consuming the frozen variance-report hash and `n_final`.

No implementation may skip from this design directly to final H4 execution.

---

## 37.19 Frozen Boundary

Once the first holdout-variance hand is generated, the following are immutable under this protocol:

```text
protocol_id                     = phase2-h4-holdout-variance-pilot-v1
primary horizon                 = 500 hands per positional trajectory
holdout population hash         = b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3
holdout acceptance hash         = 9ba347fb1d615cec1d6458944df3c293dcac630564e8847a407e987bbb98af84
independent population clusters = 10
master seeds                    = 6,400,000 ... 6,400,009
holdout units per seed          = 5
unit weights                    = 0.2 each
arms                            = control + adaptive treatment
positions per arm               = 2
planned arm jobs                = 100
planned matched unit pairs      = 50
planned engine hands            = 100,000
variance estimator              = sample-sd-matched-population-trajectory-contrast-bb100-ddof1-v1
population aggregation          = within-seed first
complete-campaign requirement   = 10 / 10 valid population clusters
replacement seeds               = forbidden
optional stopping               = prohibited
confirmatory inference          = prohibited
observed-mean design use        = prohibited
final-data pooling              = prohibited
```

Changing any of these after observing holdout variance outcomes requires a new versioned protocol. Results produced under a changed scientific configuration cannot be silently combined with this variance study.
