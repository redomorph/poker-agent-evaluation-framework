# Phase-1 Pilot Protocol v1

## 23. Status and Scope

This file is normative for the first Phase-1 variance/runtime pilot.

```text
protocol_id       = phase1-pilot-v1
protocol_version  = phase1-pilot-protocol-v1
status            = frozen
purpose           = variance_and_runtime_estimation_only
```

The pilot is **not** the final Phase-1 benchmark and is not eligible for confirmatory inference. Pilot observations MUST NOT be pooled into the later final benchmark dataset.

The machine-readable protocol is:

```text
configs/experiments/phase1-pilot-v1.json
```

Its stored `protocol_hash` is the SHA-256 digest of the canonical, sorted, compact JSON representation of the nested `protocol` object. A mismatch is an integrity failure.

## 23.1 Game and Statistical Unit

The pilot uses `BASELINE_GAME_CONFIG` without modification:

```text
game abstraction = baseline-v1
statistical unit = paired_deal
CRN mode = identical_realisations
```

One independent `paired_deal` unit executes the same deal twice with the positions exchanged. It therefore represents exactly two engine-hand simulations.

The approved paired statistic remains:

$$
d_i = \frac{d_{BTN,i} + d_{BB,i}}{2}.
$$

For pilot variance planning, the cell-level paired-difference standard deviation is reported on the BB/100 scale as:

$$
\widehat{\sigma}_{d,BB/100}=100\,s(d_1,\ldots,d_n),
$$

where $s$ is the ordinary sample standard deviation with denominator $n-1$. The version identifier is:

```text
sample-sd-paired-difference-bb100-ddof1-v1
```

No lower-level engine hand is promoted to an independent statistical observation.

## 23.2 Hero Baselines

The pilot includes exactly two transparent, static hero agents:

```text
random          -> random-baseline-v1
hand_strength   -> hand-strength-baseline-v1
```

Both are non-adaptive. The deterministic hand-strength baseline retains the frozen `uniform-random-equity-v1` behavior from its own specification; the pilot does not introduce a faster equity approximation.

## 23.3 Opponents and Gate-6 Binding

Each hero is evaluated against exactly the five calibrated static profiles:

```text
nit
tag
lag
calling_station
maniac
```

with:

```text
static_profile_version      = static-profiles-v2
parametric_policy_version   = parametric-policy-v1
```

A scientifically passing Gate-6 report is mandatory. Its exact report hash MUST be bound to all pilot run metadata **before the first pilot hand is executed**. All ten cells use the same Gate-6 evidence identity. A changed calibration/policy/equity dependency requires a new pilot protocol version rather than silent reuse of `phase1-pilot-v1`.

## 23.4 Frozen Populations

The primary Phase-1 reference population is the Baseline Synthetic Population because Section 7.7 defines it as the reproducible reference population:

```text
Nit               10%
TAG               35%
LAG               30%
Calling Station   15%
Maniac            10%
```

The Balanced Population is frozen as a secondary descriptive population:

```text
Nit               20%
TAG               20%
LAG               20%
Calling Station   20%
Maniac            20%
```

The machine-readable population files are:

```text
configs/populations/baseline-synthetic-v1.json
configs/populations/balanced-v1.json
```

Each file stores a canonical payload and its SHA-256 population hash. The payload MUST exactly reproduce the corresponding `population-evaluation-v1` definition. Because both populations have the same five-profile support, the Balanced Population requires no additional pilot hand simulations; only the later aggregation weights differ.

Pilot results remain variance/runtime evidence regardless of population weighting. Descriptive population estimates from the pilot MUST NOT be promoted to final benchmark estimates.

## 23.5 Pilot Matrix and Size

The pilot contains:

```text
2 hero agents
x 5 opponent profiles
x 100 paired_deal units per cell
= 1,000 independent paired units
= 2,000 simulated engine hands
```

This is deliberately inside the 1,000–5,000-hand pilot guidance in Section 3.5 while distributing variance evidence equally over all ten baseline cells.

The pilot size is fixed. It MUST NOT be increased after inspecting pilot EV, variance, or confidence intervals. A separately versioned pilot is required for additional variance evidence.

## 23.6 Seed Schedule and CRN

The pilot seed schedule is:

```text
seed_schedule_version = phase1-pilot-seeds-v1
master seeds           = 6,100,000 ... 6,100,099 inclusive
```

The identical ordered set of 100 master seeds is reused for every hero/opponent cell. The RNG contract is:

```text
rng_contract   = event-indexed-crn-v1
DEAL_SCHEDULE
RNG_BOT_A
RNG_BOT_B
(bot_id,hand_id,decision_index,purpose)
```

The common seed schedule is intentional CRN coupling across conditions. Agent random-number consumption MUST NOT alter the deal schedule or another condition's RNG identity.

No failed unit is replaced with a different seed.

## 23.7 Failure Handling

The pilot freezes:

```text
illegal_action_policy   = exclude_run
replacement_seed_policy = forbidden
```

Failures are retained and reported with their original seed/cell identity. Replacement seeds are prohibited.

A pilot with failures may still provide descriptive diagnostic information, but the final sample-size planning record MUST state how many valid paired observations contributed to each cell variance estimate. Missing cells or materially incomplete variance evidence require a separately versioned pilot decision before final-data collection.

## 23.8 Permitted Pilot Analysis

The only scientific uses of `phase1-pilot-v1` are:

```text
variance_estimation
runtime_estimation
final_sample_size_planning
```

The following uses are prohibited:

```text
confirmatory_inference
final_benchmark_estimate
optional_stopping
```

Pilot means may be mechanically produced by lower-level tooling but are not treated as final scientific benchmark evidence.

The pilot freezes cell-specific variance estimation, not a final cross-cell sample allocation rule. The final protocol must deterministically state how the complete vector of pilot variances is converted into its fixed per-cell/family sample sizes before any final hand is executed.

## 23.9 Final-Design Boundary

The following fields are deliberately and normatively unset in the pilot protocol:

```text
minimum_relevant_difference_bb100 = null
precision_target_half_width_bb100 = null
final_independent_units           = null
```

They MUST remain unset in `phase1-pilot-v1`. Filling them in does not update the pilot; it creates a different protocol and therefore requires a different version/hash.

After the pilot is complete, the final Phase-1 protocol will freeze, before final data collection:

- the practical effect threshold $\delta$ where inferential power is required;
- the estimator-specific precision target;
- the pilot-derived variance inputs;
- the deterministic sample-size/allocation rule;
- the resulting fixed independent-unit counts;
- the final master-seed schedule, disjoint from the pilot schedule;
- confirmatory versus descriptive status;
- multiplicity handling if a confirmatory family is declared.

The pilot observations themselves are excluded from that final dataset.

## 23.10 Required Execution Provenance

Every executed pilot run must persist the reproducibility metadata required by Section 16, including at least:

```text
experiment_id
game_config_hash
bot_version
opponent_version
opponent_parameters
population_version
master_seed
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
```

The protocol intentionally does not hard-code machine-specific values such as Git commit, Python build, CPU model, or the Gate-6 report hash before execution. Those values are **bindings**, not researcher degrees of freedom: they are captured and attached before the first pilot hand and must remain consistent across one execution campaign. Changing a scientific dependency requires a new protocol version; changing only the execution machine must be explicitly recorded and treated as a reproducibility/environment distinction.

## 23.11 Canonical Job Expansion

`iter_pilot_jobs(...)` expands the frozen matrix in this deterministic order:

```text
hero order -> opponent order -> seed order
```

with hero order:

```text
random
hand_strength
```

and opponent order:

```text
nit
tag
lag
calling_station
maniac
```

The resulting 1,000 jobs are a deterministic consequence of the protocol. Job ordering is an execution convention and does not alter the scientific seed identities.

## 23.12 Change Control

`phase1-pilot-v1` is immutable once committed. Any change to its game, agent versions, opponent versions, population definitions, seed schedule, matrix size, statistical unit, variance estimator, failure handling, or permitted analysis requires a new protocol version.

The final benchmark is a separate experiment protocol and MUST NOT overwrite or mutate this pilot specification.
