# Phase-2 Adaptive Pilot Protocol v1

## 32. Status and Scope

This file is normative for the first end-to-end pilot of the frozen Phase-2 adaptive hand-strength branch.

```text
protocol_id       = phase2-adaptive-pilot-v1
protocol_version  = phase2-adaptive-pilot-protocol-v1
status            = design_frozen_execution_not_yet_authorized
purpose           = adaptive_learning_curve_variance_runtime_and_h4_planning_only
```

The pilot is **not** the final H4 experiment and is not eligible for confirmatory inference. Pilot observations MUST NOT be pooled into the later final H4 dataset.

The scientific behavior of the treatment is already frozen by Specs 27-31. This protocol does not change Bayesian priors, evidence classification, posterior-to-range mapping, equity semantics, action thresholds, or lifecycle timing. It freezes only the first pilot experiment design.

A machine-readable companion does not yet exist. The next implementation step MUST create and validate:

```text
configs/experiments/phase2-adaptive-pilot-v1.json
```

from this document using tests first. No adaptive pilot hand may be executed until that machine-readable protocol is canonical, hash-verified, semantically validated, and the validation gate in Section 32.15 is green.

## 32.1 Scientific Purpose

The pilot has four permitted scientific purposes:

1. verify end-to-end execution of the already-frozen adaptive lifecycle over non-trivial trajectories;
2. characterize the variance of matched adaptive-versus-control trajectory contrasts at the 10, 20, 50, and 100 hand cumulative horizons;
3. measure runtime, storage, failure behavior, and replay cost for adaptive trajectories;
4. provide non-confirmatory learning-curve and model-stability evidence that may inform a separately frozen final H4 design.

The pilot MUST NOT be used to claim that H4 is supported or rejected.

## 32.2 Entry Gate and Frozen Dependencies

Pilot implementation may begin only after the Spec-31 validation gate is green.

The treatment depends on the following frozen scientific definitions:

```text
bayesian-opponent-model-v1
adaptive-learning-protocol-v1
posterior-to-range-v1
adaptive-hand-strength-policy-v1
adaptive-lifecycle-v1
adaptive-hand-strength-agent-v1
calibration-opportunities-v2
public-action-trace-v1
```

The control is the exact control already frozen by Spec 30:

```text
hand-strength-baseline-v1
uniform-random-range-v1
```

The opponent dependencies are:

```text
static-profiles-v2
parametric-policy-v1
BASELINE_GAME_CONFIG / baseline-v1
```

A scientifically passing Gate-6 report is mandatory. Its exact report hash MUST be bound to pilot execution metadata before the first pilot hand. If a calibration, policy, equity, game, adaptive-policy, or lifecycle dependency changes scientifically, the pilot requires a new protocol version rather than silent reuse of `phase2-adaptive-pilot-v1`.

## 32.3 Treatment, Control, and Information Boundary

The two hero arms are exactly:

```text
control:
    hand-strength-baseline-v1
    agent_is_adaptive = false
    opponent range = uniform-random-range-v1

treatment:
    adaptive-hand-strength-agent-v1
    agent_is_adaptive = true
    opponent range = posterior-to-range-v1
    lifecycle = adaptive-lifecycle-v1
```

This is the controlled H4 treatment/control relationship frozen by Spec 30: the adaptive treatment changes the opponent range used by the passive equity/pot-odds policy; it does not add betting, raising, jamming, or additional behavioral heuristics.

The experiment scheduler knows the opponent profile identity so that cells can be stratified and reported. The hero agents MUST NOT receive:

- profile ID;
- the opponent's true parameter vector;
- population weights;
- hidden opponent cards;
- hidden deck state;
- opponent RNG state.

For learning, the adaptive hero receives only the public evidence already permitted by Specs 28 and 31. Therefore the opponents are **unknown to the agent** even though their identities are known to the experiment controller.

## 32.4 Development Opponents and Populations

The pilot uses exactly the five calibrated stationary development profiles:

```text
nit
tag
lag
calling_station
maniac
```

No randomized, stress, validation-only, or holdout/test population is consumed by this pilot.

The primary descriptive population weighting is the existing Baseline Synthetic Population:

```text
Nit               10%
TAG               35%
LAG               30%
Calling Station   15%
Maniac            10%
```

The Balanced Population remains a secondary descriptive weighting:

```text
Nit               20%
TAG               20%
LAG               20%
Calling Station   20%
Maniac            20%
```

These weights are used only for deterministic aggregation of the five already-executed profile cells. They do not alter the opponent seen within a trajectory.

The final H4 population remains unset. Pilot evidence MUST NOT be interpreted as holdout evidence.

## 32.5 Trajectory and Position Structure

The adaptive statistical unit remains:

```text
statistical_unit = trajectory_cluster
```

For each arm, opponent profile, and master seed, one trajectory cluster contains two symmetric positional trajectories:

```text
Position run 1: hero BTN, opponent BB
Position run 2: opponent BTN, hero BB
```

The same defined card realizations are used in both positional runs. The adaptive treatment starts from the frozen independent `Beta(1,1)` prior vector before each positional run. State learned in the first positional run MUST NOT enter the second positional run.

The static control is executed through the same full-trajectory position-pair structure so that treatment and control are aggregated at the same trajectory level. The control remains non-adaptive and carries no Bayesian learning state. The execution layer MUST explicitly use trajectory-cluster/full-match mode for the control arm; it MUST NOT infer a paired-deal statistical mode merely from `control.is_adaptive == false`.

No individual hand or decision is treated as an independent inferential observation.

## 32.6 Fixed Two-Tier Pilot Matrix

The pilot uses a predeclared two-tier schedule. It is **not** a sequential stopping rule.

### Short-horizon tier

```text
master seeds                  = 6,300,000 ... 6,300,004 inclusive
opponent profiles             = 5
hero arms                     = 2
hands per positional run      = 100
positions per arm cluster     = 2
```

All five seeds contribute cumulative endpoints at:

```text
10
20
50
100 hands
```

### Long-horizon extension

The following seed is predeclared before execution to continue to the full 500-hand horizon:

```text
long_horizon_master_seed = 6,300,000
hands per positional run = 500
```

For seed `6,300,000`, the 10/20/50/100 endpoints are taken from the same 500-hand trajectories; they are not rerun separately.

Seeds `6,300,001` through `6,300,004` stop at the already-fixed 100-hand trajectory length regardless of observed EV, posterior state, variance, runtime, or confidence intervals.

### Matrix size

The resulting fixed pilot contains:

```text
5 opponent profiles
x 5 matched treatment/control seed identities through hand 100
= 25 matched profile-seed trajectory contrasts through hand 100

1 predeclared long seed
x 5 opponent profiles
= 5 matched profile-seed trajectory contrasts through hand 500
```

Counting engine hands:

```text
seed 6,300,000:
    5 profiles x 2 arms x 2 positions x 500 hands = 10,000 engine hands

seeds 6,300,001 ... 6,300,004:
    5 profiles x 2 arms x 2 positions x 100 hands x 4 seeds = 8,000 engine hands

fixed total = 18,000 simulated engine hands
```

The generic 1,000-5,000-hand pilot guidance in Spec 03 is explicitly non-universal. Adaptive trajectory clustering and the already-frozen 500-hand learning horizon make a larger engine-hand count necessary even for a small number of independent trajectory clusters.

## 32.7 Seed Schedule and CRN Relationship

The pilot seed schedule is:

```text
seed_schedule_version = phase2-adaptive-pilot-seeds-v1
master seeds           = 6,300,000 ... 6,300,004 inclusive
```

The identical ordered seed schedule is reused across:

- all five opponent profiles;
- treatment and control arms;
- the two positional conditions within each arm.

For a fixed profile, seed, and hand index, treatment and control use the same deterministic deal schedule. The two positional conditions within one arm also use the identical defined card realizations required by Spec 28.

The cross-arm CRN claim is intentionally limited:

```text
deal_relation              = identical_defined_card_realisations
agent_rng_relation         = event_indexed_common_identity_where_events_correspond
full_action_path_identity  = not_claimed
```

Treatment and control may choose different actions and therefore create different later action opportunities. The experiment MUST NOT claim that their complete action-RNG paths or public histories are identical merely because their master seed is shared.

No failed trajectory is replaced with a different seed.

## 32.8 Cumulative Arm Estimator and Matched Contrast

For arm $a \in \{adaptive, control\}$, opponent profile $p$, seed $s$, and cumulative horizon $h$, let

- $G^{BTN}_{a,p,s}(h)$ be the hero's total profit in BB over hands $1,\ldots,h$ of the hero-BTN positional trajectory;
- $G^{BB}_{a,p,s}(h)$ be the hero's total profit in BB over hands $1,\ldots,h$ of the hero-BB positional trajectory.

The position-compensated cumulative arm estimator is

$$
Y_{a,p,s}(h)
=
100\cdot\frac{G^{BTN}_{a,p,s}(h)+G^{BB}_{a,p,s}(h)}{2h}.
$$

The matched adaptive-control trajectory contrast is

$$
D_{p,s}(h)=Y_{adaptive,p,s}(h)-Y_{control,p,s}(h).
$$

This is the pilot form of the H4 adaptive-minus-nonadaptive comparison. It is measured in BB/100.

At $h\in\{10,20,50,100\}$, each profile has five predeclared seed-level matched contrasts. The profile planning standard deviation is the ordinary sample standard deviation with denominator $n-1$:

$$
s_{D,p}(h)=sd\left(D_{p,s}(h):s\in S_{100}\right),
$$

where

```text
S_100 = {6300000, 6300001, 6300002, 6300003, 6300004}
```

and the estimator version is:

```text
sample-sd-matched-trajectory-contrast-bb100-ddof1-v1
```

At the 500-hand endpoint only seed `6,300,000` is present. Therefore **no profile-level or population-level variance, standard error, confidence interval, or hypothesis test is defined at 500 hands in this pilot**.

## 32.9 Population-Weighted Pilot Contrasts

Because the same seed identities are reused across all five profiles, population aggregation is performed **within seed first**.

For a frozen population with profile weights $w_p$:

$$
D_{population,s}(h)=\sum_p w_p D_{p,s}(h).
$$

At horizons 10/20/50/100, the sample standard deviation is then computed over the five independent master-seed identities:

$$
s_{population}(h)
=
sd\left(D_{population,s}(h):s\in S_{100}\right).
$$

The five profile cells MUST NOT be incorrectly treated as five independent replicates of one common opponent when estimating population uncertainty.

The Baseline Synthetic weighting is primary descriptive pilot aggregation. The Balanced weighting is secondary descriptive aggregation.

## 32.10 Frozen Horizons and Segments

The pilot preserves the adaptation horizons frozen by Specs 05 and 28:

```text
10
20
50
100
500 hands
```

It also preserves the three reporting segments:

```text
Early Adaptation   hands 1-20
Main Adaptation    hands 21-100
Steady State       hands 101-500
```

Segment arm EV and matched adaptive-control contrasts use the same position-compensated normalization as Section 32.8, replacing the cumulative prefix with the corresponding inclusive hand interval.

The Early and Main segments use all five seeds. The Steady-State segment exists only for the predeclared long seed and is therefore descriptive only.

The 100-hand result remains a cumulative hands-1-100 learning-curve endpoint as required by H4. It is not the 100th hand and is not the final confirmatory H4 endpoint.

## 32.11 Learning and Model Diagnostics

For the adaptive treatment, the pilot report must reconstruct from persisted public learning transitions, separately for each positional trajectory and reporting horizon, at least:

```text
trajectory_hand_count
public evidence opportunity/event totals for all six frozen metrics
posterior alpha/beta state for all six frozen metrics
posterior state hash
valid learning transition count
```

The report may additionally summarize already-frozen posterior-to-range or decision provenance when available, but it MUST NOT introduce a new scientific action rule or new learned metric.

Profile target parameters may be used by the offline evaluator for labeled diagnostic plots or tables because the experiment controller already knows the development cell. They remain unavailable to the adaptive agent itself.

## 32.12 Failure Handling and Validity

The pilot freezes:

```text
illegal_action_policy     = exclude_run
replacement_seed_policy   = forbidden
matched_contrast_policy   = complete_pair_only
adaptive_timeout_policy   = invalidate_trajectory_cluster
```

A profile-seed matched contrast is valid only if **both treatment and control trajectory clusters are scientifically valid** for the required horizon.

For the adaptive treatment, Spec 31 already makes a timeout-altered, illegal-action, agent-error, or engine-error trajectory invalid and prevents silent continuation of the learning history.

If either arm is invalid for a profile-seed identity, that profile-seed identity is excluded from matched analysis rather than replaced. The original failed records and seed identities remain durable.

For seeds scheduled to 100 hands, validity is all-or-nothing through hand 100. For seed `6,300,000`, validity is all-or-nothing through hand 500; a later failure does not retroactively create a scientifically valid shorter matched unit from that long trajectory.

Pilot diagnostics may still describe failures. If any planned profile-seed matched contrast is invalid, the final H4 protocol MUST explicitly determine whether the remaining variance evidence is sufficient or whether a separately versioned additional pilot is required. No replacement seed may be inserted silently.

## 32.13 Permitted and Prohibited Pilot Analysis

Permitted scientific use:

```text
end_to_end_adaptive_validation
learning_curve_description
posterior_state_diagnostics
variance_estimation_at_10_20_50_100
runtime_and_storage_estimation
final_h4_sample_size_planning_input
final_h4_horizon_planning_from_non_outcome_diagnostics
```

Prohibited scientific use:

```text
confirmatory_h4_inference
final_h4_effect_estimate
pilot_plus_final_data_pooling
optional_stopping
seed_replacement_after_failure
adaptive_policy_tuning_from_pilot_ev
opponent_profile_tuning_from_pilot_ev
primary_h4_horizon_selection_from_observed_delta_direction_or_p_value
holdout_population_inspection
mcts_evaluation
```

Pilot arm means and adaptive-minus-control means may be reported descriptively for transparency, but they MUST NOT be used to select the final primary H4 horizon based on the sign, magnitude, statistical significance, or apparent peak of the observed treatment effect.

## 32.14 Final-H4 Design Boundary

The following fields are deliberately and normatively unset in this pilot:

```text
primary_h4_horizon                    = null
final_h4_population                   = null
minimum_relevant_difference_bb100     = null
precision_target_half_width_bb100     = null
final_independent_trajectory_clusters = null
confirmatory_alpha                    = null
confirmatory_beta                     = null
multiplicity_binding                  = null
final_h4_seed_schedule                = null
```

Before any final H4 hand is executed, a separate frozen H4 protocol must define at least:

- the primary confirmatory horizon, which must satisfy the active H4 requirement that the primary comparison use a longer horizon than the 100-hand cumulative endpoint;
- the final opponent population and whether it is validation or holdout/test evidence;
- the minimum practically relevant adaptive-control EV difference if a powered hypothesis test is used;
- the applicable trajectory-level variance input and deterministic power/sample-size rule;
- any estimator-specific precision target;
- the fixed number of independent trajectory clusters;
- the final master-seed schedule, disjoint from all adaptive pilot seeds;
- the primary endpoint and statistical test;
- multiplicity membership and Holm handling if H4 is in the confirmatory family.

Pilot variance may inform the final sample-size calculation. The observed pilot mean treatment effect MUST NOT define the minimum relevant difference.

If the selected final H4 primary horizon exceeds 100 hands, the final protocol must explicitly justify the applicable variance estimate. Because this pilot provides replicate variance only through 100 hands, a separately versioned additional long-horizon pilot may be required rather than silently treating the single 500-hand seed as a variance estimate.

## 32.15 Validation Gate Before First Pilot Hand

Before pilot execution begins, tests must verify at minimum:

1. a canonical machine-readable `phase2-adaptive-pilot-v1` protocol, exact protocol identity, payload hash, and semantic tamper rejection;
2. exact treatment/control agent versions and frozen dependencies;
3. exact five-profile development opponent matrix and Gate-6 binding requirement;
4. the fixed seed schedule and two-tier 100/500 trajectory lengths;
5. exact deterministic job expansion and the total of 18,000 planned engine hands;
6. treatment reset before both positional runs, no adaptive state in the control arm, and trajectory-cluster/full-match execution for both arms;
7. no profile ID or true parameter vector is exposed through the hero observation/learning interface;
8. identical card realizations for matched profile/seed/hand identities without falsely asserting identical full action paths;
9. correct cumulative and segment extraction at 10/20/50/100/500 without promoting hands to independent observations;
10. correct within-seed population weighting before variance is taken across seeds;
11. the 500-hand endpoint is rejected as a source of variance/CI/inferential statistics in v1;
12. complete-pair failure handling, adaptive trajectory invalidation, durable failure records, and forbidden replacement seeds;
13. pilot/final data separation and the deliberately unset final-H4 fields;
14. replay of every adaptive learning chain used by the pilot report succeeds from public evidence and frozen state hashes.

Only after this gate is green may pilot execution tooling be considered scientifically ready to run the first adaptive pilot hand.

## 32.16 Required Execution Provenance

Every pilot trajectory must preserve the reproducibility metadata required by Spec 14, including at least:

```text
experiment_id
pilot_protocol_id
pilot_protocol_hash
game_config_hash
hero_arm
bot_version
opponent_version
opponent_profile_id
opponent_parameters
population_versions
master_seed
planned_trajectory_horizon
rng_configuration
engine_version
python_version
poetry_lock_hash
git_commit
cpu_model
cpu_core_count
cpu_architecture
os_identifier
statistical_unit
gate6_report_hash
adaptive_learning_protocol_hash
posterior_to_range_protocol_hash
adaptive_policy_protocol_hash
adaptive_lifecycle_protocol_hash
```

Environment-specific values are execution bindings, not researcher degrees of freedom. They are captured before or during execution as appropriate and must remain traceable without modifying the scientific protocol.

## 32.17 Change Control

`phase2-adaptive-pilot-v1` is immutable once committed as an active specification. Any change to its treatment/control versions, opponents, populations, seed schedule, trajectory lengths, horizon/segment extraction, statistical contrast, failure handling, permitted analysis, or final-design boundary requires a new protocol version.

The forthcoming machine-readable protocol and execution layer must implement this design exactly; they may not silently reinterpret it.
