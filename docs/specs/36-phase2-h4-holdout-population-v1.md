# Phase-2 H4 Holdout Population v1

## 36. Status and Scope

This file is normative for the fixed unseen parametric opponent population used by the future H4 500-hand variance pilot and the final H4 confirmatory experiment.

Population specification identifier:

```text
phase2-h4-holdout-population-v1
```

Population version:

```text
phase2-h4-holdout-parametric-population-v1
```

Population role:

```text
holdout_test
```

Status:

```text
population_frozen_machine_readable_acceptance_not_yet_closed
```

This specification freezes the population definition, deterministic generation procedure, exact five opponent units, equal weights, calibration/acceptance boundary, information boundary, and the no-optimization rule required by Spec 35.

It does **not** authorize the 500-hand holdout variance pilot or final H4 data generation. Specs 27–35 remain unchanged.

---

## 36.1 Scientific Role

The final H4 question remains whether:

> `adaptive-hand-strength-agent-v1` outperforms the otherwise identical `hand-strength-baseline-v1` against unknown parametric opponents after sufficient observation time.

The primary H4 population must therefore test generalisation beyond the five development profiles used by `phase2-adaptive-pilot-v1`.

The H4 holdout population is frozen as a fixed finite equal-weight sample from a predeclared continuous parametric support:

```text
population_kind = fixed-finite-equal-weight-parametric-v1
num_units       = 5
unit_weight     = 0.2
```

The same five units and the same weights are used for:

1. the future replicated 500-hand holdout variance pilot;
2. the final H4 confirmatory experiment.

The population is **not resampled per master seed**. Each independent master seed evaluates the complete five-unit population, and population aggregation occurs inside seed before across-seed variance or inference.

The five units are not named after the development archetypes. Their identifiers are neutral:

```text
h4_holdout_000
h4_holdout_001
h4_holdout_002
h4_holdout_003
h4_holdout_004
```

---

## 36.2 Binding Existing Frozen Mechanisms

Every holdout opponent uses exactly:

```text
parameter_schema_version       = opponent-parameters-v1
parametric_policy_version      = parametric-policy-v1
calibration_environment_version = policy-calibration-environment-v1
calibration_version            = policy-calibration-v1
opportunity_definition_version = calibration-opportunities-v2
tolerance_version              = calibration-tolerances-v1
```

The population is bound to the already-passing Gate-6 artifacts:

```text
gate6_report_hash        = ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a
reference_strength_hash  = d2552b7ffa6aca5e7712b2ddef0dce8805f911cad5eb4c0e6b39fb96a06a6143
validation_strength_hash = d7ed741ec44627e49d034d87de36f636a1463aa7a6804656c594189bf25c54e2
```

No holdout-specific policy mechanism, sizing rule, threshold family, equity estimator, calibration opportunity definition, or tolerance is introduced here.

For each frozen target vector $\theta_p$, `policy-strength-gates-v1` is constructed from the existing fixed Gate-6 strength reference. The validation sample is used only for acceptance and must never fit or tune a gate.

---

## 36.3 Holdout Support

The holdout generator operates on six latent dimensions:

```text
vpip
pfr_ratio
three_bet_freq
cbet_freq
aggression
fold_to_cbet
```

The frozen support is:

| Latent dimension | Lower | Upper |
|---|---:|---:|
| `vpip` | 0.08 | 0.60 |
| `pfr_ratio` | 0.05 | 0.90 |
| `three_bet_freq` | 0.01 | 0.20 |
| `cbet_freq` | 0.30 | 0.85 |
| `aggression` | 0.20 | 1.80 |
| `fold_to_cbet` | 0.10 | 0.60 |

`pfr_ratio` is not itself an opponent-policy field. It is a structural generation variable used to guarantee:

$$
0 \le PFR \le VPIP \le 1.
$$

For each generated unit:

$$
PFR = \operatorname{round}_{6,\mathrm{half\ even}}(VPIP\cdot pfr\_ratio).
$$

All five holdout units specify a numeric Aggression target. No holdout metric is declared `don't-care`.

The support was fixed from the pre-existing parametric-policy domain and the already-specified static/stress benchmark envelope. It is not derived from the adaptive-pilot treatment-effect means, signs, profile outcomes, or apparent best-performing regions.

---

## 36.4 Deterministic Generation: `sha256-latin-hypercube-fixedpoint-v1`

Generation algorithm:

```text
sha256-latin-hypercube-fixedpoint-v1
```

Number of units:

```text
N = 5
```

The generation seed material is not researcher-selected after inspecting outcomes. It is derived deterministically from the population identifier:

```text
seed_material = SHA256(
    "phase2-h4-holdout-population-v1:sealed-generation-v1"
)

seed_material =
    a1f17ebdb8504b9c5b59e70439fbe4eaf4a6de559b6436e996f8a0c1c9647d75
```

The seed material is a population-generation namespace only. It is **not** an engine master seed and does not enter card, action, equity, or bootstrap RNG domains.

For each latent dimension `d` and stratum `k in {0,1,2,3,4}`, define the permutation key:

```text
SHA256(
    population_id
    + ":"
    + seed_material
    + ":"
    + d
    + ":perm:"
    + str(k)
)
```

The five strata are ordered by the unsigned 256-bit integer value of these hashes, with the numeric stratum index as the deterministic tie-breaker.

For unit index `i`, define the within-stratum jitter:

```text
jitter(d, i) = uint256(
    SHA256(
        population_id
        + ":"
        + seed_material
        + ":"
        + d
        + ":jitter:"
        + str(i)
    )
) / 2**256
```

If `perm_d[i]` is the selected stratum, then:

$$
u_{d,i}=\frac{perm_d[i]+jitter(d,i)}{5}.
$$

Each latent value is mapped linearly onto the support in Section 36.3 and quantized to exactly six decimal places using decimal round-half-even semantics.

The frozen stratum permutations are:

```text
vpip              = [2, 1, 4, 3, 0]
pfr_ratio         = [4, 3, 2, 1, 0]
three_bet_freq    = [1, 0, 2, 4, 3]
cbet_freq         = [4, 3, 1, 0, 2]
aggression        = [4, 3, 1, 2, 0]
fold_to_cbet      = [3, 2, 4, 0, 1]
```

There is no rejection sampling, no search for favourable adaptive outcomes, and no replacement candidate stream.

---

## 36.5 Exact Frozen Holdout Units

The generator in Section 36.4 yields the following exact target vectors:

| Unit | Weight | VPIP | PFR | 3BetFreq | CBetFreq | Aggression | FoldToCBet |
|---|---:|---:|---:|---:|---:|---:|---:|
| `h4_holdout_000` | 0.2 | 0.363630 | 0.291240 | 0.051497 | 0.746223 | 1.515065 | 0.406611 |
| `h4_holdout_001` | 0.2 | 0.184105 | 0.113241 | 0.018270 | 0.690724 | 1.426503 | 0.383963 |
| `h4_holdout_002` | 0.2 | 0.583082 | 0.227593 | 0.117397 | 0.447809 | 0.670847 | 0.592238 |
| `h4_holdout_003` | 0.2 | 0.460487 | 0.111152 | 0.171613 | 0.343563 | 0.885767 | 0.131947 |
| `h4_holdout_004` | 0.2 | 0.128226 | 0.007052 | 0.159610 | 0.531827 | 0.417032 | 0.200850 |

The weights sum exactly to 1.0.

The canonical population commitment payload contains:

```text
population_id
population_version
population_role
population_kind
parameter_schema_version
parametric_policy_version
calibration_environment_version
calibration_version
opportunity_definition_version
tolerance_version
gate6_report_hash
reference_strength_hash
validation_strength_hash
generation definition
exact five weighted units
```

Canonical JSON uses sorted keys, compact separators, UTF-8, `ensure_ascii = true`, and `allow_nan = false`.

The frozen population hash is:

```text
population_hash = b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3
```

A future machine-readable companion must reproduce this exact population hash. Any different unit, parameter, weight, generation rule, dependency hash, or calibration binding is a different population and cannot be substituted under this version.

### 36.5.1 Exact Population-Hash Payload

`population_hash` is the SHA-256 digest of the canonical JSON encoding of the following object **without** a `population_hash` field:

```json
{
  "calibration_environment_version": "policy-calibration-environment-v1",
  "calibration_version": "policy-calibration-v1",
  "gate6_report_hash": "ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a",
  "generation": {
    "algorithm": "sha256-latin-hypercube-fixedpoint-v1",
    "num_units": 5,
    "pfr_construction": "pfr=round_half_even(vpip*pfr_ratio,6)",
    "quantization_decimals": 6,
    "seed_material": "a1f17ebdb8504b9c5b59e70439fbe4eaf4a6de559b6436e996f8a0c1c9647d75",
    "support": {
      "aggression": [0.2, 1.8],
      "cbet_freq": [0.3, 0.85],
      "fold_to_cbet": [0.1, 0.6],
      "pfr_ratio": [0.05, 0.9],
      "three_bet_freq": [0.01, 0.2],
      "vpip": [0.08, 0.6]
    }
  },
  "opportunity_definition_version": "calibration-opportunities-v2",
  "parameter_schema_version": "opponent-parameters-v1",
  "parametric_policy_version": "parametric-policy-v1",
  "population_id": "phase2-h4-holdout-population-v1",
  "population_kind": "fixed-finite-equal-weight-parametric-v1",
  "population_role": "holdout_test",
  "population_version": "phase2-h4-holdout-parametric-population-v1",
  "reference_strength_hash": "d2552b7ffa6aca5e7712b2ddef0dce8805f911cad5eb4c0e6b39fb96a06a6143",
  "tolerance_version": "calibration-tolerances-v1",
  "units": [
    {
      "parameters": {
        "aggression": 1.515065,
        "cbet_freq": 0.746223,
        "fold_to_cbet": 0.406611,
        "pfr": 0.29124,
        "three_bet_freq": 0.051497,
        "vpip": 0.36363
      },
      "unit_id": "h4_holdout_000",
      "weight": 0.2
    },
    {
      "parameters": {
        "aggression": 1.426503,
        "cbet_freq": 0.690724,
        "fold_to_cbet": 0.383963,
        "pfr": 0.113241,
        "three_bet_freq": 0.01827,
        "vpip": 0.184105
      },
      "unit_id": "h4_holdout_001",
      "weight": 0.2
    },
    {
      "parameters": {
        "aggression": 0.670847,
        "cbet_freq": 0.447809,
        "fold_to_cbet": 0.592238,
        "pfr": 0.227593,
        "three_bet_freq": 0.117397,
        "vpip": 0.583082
      },
      "unit_id": "h4_holdout_002",
      "weight": 0.2
    },
    {
      "parameters": {
        "aggression": 0.885767,
        "cbet_freq": 0.343563,
        "fold_to_cbet": 0.131947,
        "pfr": 0.111152,
        "three_bet_freq": 0.171613,
        "vpip": 0.460487
      },
      "unit_id": "h4_holdout_003",
      "weight": 0.2
    },
    {
      "parameters": {
        "aggression": 0.417032,
        "cbet_freq": 0.531827,
        "fold_to_cbet": 0.20085,
        "pfr": 0.007052,
        "three_bet_freq": 0.15961,
        "vpip": 0.128226
      },
      "unit_id": "h4_holdout_004",
      "weight": 0.2
    }
  ],
  "validation_strength_hash": "d7ed741ec44627e49d034d87de36f636a1463aa7a6804656c594189bf25c54e2"
}
```

The canonical serializer is exactly:

```text
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)
```

encoded as UTF-8 with **no trailing LF included in the hashed bytes**. This yields exactly:

```text
b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3
```

---

## 36.6 Calibration and Acceptance Boundary

Every one of the five frozen target vectors must pass the existing independent validation environment before any H4 holdout variance hand is executed.

For every holdout unit, acceptance requires all six metrics:

```text
VPIP
PFR
3BetFreq
CBetFreq
Aggression
FoldToCBet
```

All six are required. There is no holdout `don't-care` metric.

The frozen acceptance tolerances remain:

```text
VPIP        +/- 0.03
PFR         +/- 0.03
3BetFreq    +/- 0.02
CBetFreq    +/- 0.05
Aggression  +/- 0.20
FoldToCBet  +/- 0.05
```

The acceptance environment remains:

```text
5,000 designated validation opportunities per metric
validation_base_seed = 5,100,002
calibration-opportunities-v2
```

Acceptance must bind the exact Gate-6 reference and validation hashes from Section 36.2.

No target, support boundary, unit weight, seed material, gate threshold, tie probability, action precedence, or sizing rule may be changed because a holdout unit is inconvenient to calibrate.

If **any** of the five frozen units fails official holdout acceptance:

1. the complete `phase2-h4-holdout-population-v1` population is invalid for H4;
2. no failed unit is dropped;
3. no replacement unit is drawn;
4. no support is narrowed;
5. no alternate generation seed is tried;
6. no H4 variance or final hand may be generated;
7. a new versioned holdout-population specification is required.

A pre-freeze feasibility audit against the already-frozen Gate-6 fit/validation snapshots found all five vectors within all six existing tolerances. That audit is implementation-feasibility evidence only; the future machine-readable holdout acceptance artifact and its hashes remain mandatory before execution.

---

## 36.7 Holdout Seal and No-Optimization Rule

For this project, `holdout` means **frozen before H4 holdout outcomes and excluded from subsequent agent optimisation**. It does not depend on the parameter table being cryptographically secret from the repository owner.

The seal point is the activation of this versioned population specification together with its future machine-readable companion and passing acceptance artifact.

After that seal, no H4 holdout evidence may be used to modify:

```text
adaptive-hand-strength-agent-v1
hand-strength-baseline-v1
bayesian-opponent-model-v1
adaptive-learning-protocol-v1
posterior-to-range-v1
adaptive-hand-strength-policy-v1
adaptive-lifecycle-v1
parametric-policy-v1
H4 primary horizon
H4 holdout population
minimum relevant difference
alpha / beta
precision target
H4 decision rule
```

The future 500-hand variance pilot may influence only the long-horizon variance estimate, runtime/completeness evidence, and the deterministic sample-size calculation already frozen in Spec 35.

If a scientific code change affecting treatment, control, opponent policy, RNG semantics, game rules, or estimand is required after holdout variance outcomes are observed, those outcomes cannot remain the variance basis for a confirmatory experiment under the changed scientific configuration. A new versioned protocol/data-separation decision is required.

---

## 36.8 Information Boundary

The acting adaptive or control hero must never receive:

```text
holdout unit id
population id
population weight
true opponent parameter vector
generation seed material
latent pfr_ratio
calibration targets
calibration acceptance result
controller-side classification labels
```

The hero receives only the already-frozen legal observation and public-action evidence contracts.

The opponent controller may know its own exact target vector because those parameters define the static parametric opponent policy. Controller-side true parameters and holdout identities are provenance only and must not be injected into the hero observation, learned evidence, action policy, or adaptive posterior state.

This boundary must be enforced at runtime and replayed/validated before any holdout evidence is admitted to variance or final H4 analysis.

---

## 36.9 Population Estimand

For arm `a`, holdout unit `p`, master seed `s`, and the frozen H4 primary horizon of 500 hands per positional trajectory:

$$
Y_{a,p,s}(500)
=
100\cdot\frac{G^{BTN}_{a,p,s}(500)+G^{BB}_{a,p,s}(500)}{1000}.
$$

The matched unit contrast is:

$$
D_{p,s}(500)=Y_{adaptive,p,s}(500)-Y_{control,p,s}(500).
$$

The frozen primary population contrast for master seed `s` is:

$$
D_{population,s}(500)
=
\sum_{p=0}^{4}0.2\,D_{p,s}(500).
$$

The independent scientific observations for long-horizon variance estimation and final H4 inference are the complete master-seed population contrasts:

```text
D_population,s(500)
```

The following are **not** independent H4 replicates:

```text
hands
individual decisions
BTN and BB positional halves
holdout units within one master seed
posterior updates
learning transitions
```

This preserves the trajectory-cluster hierarchy frozen in Specs 4, 10, 28, 32, and 35.

---

## 36.10 Relationship to Development and Pilot Populations

The following remain development/validation evidence and are not members of the H4 holdout population:

```text
Nit
TAG
LAG
Calling Station
Maniac
Baseline Synthetic population
Balanced population
phase2-adaptive-pilot-v1 seeds 6,300,000 ... 6,300,004
```

The five H4 holdout vectors in Section 36.5 were not used to tune the adaptive policy, Bayesian prior, posterior-to-range mapping, action thresholds, or the Spec-35 confirmatory decision rule.

No adaptive-pilot trajectory is pooled with holdout-variance or final-H4 trajectories.

The holdout population is used for the H4 variance pilot because Spec 35 requires the variance estimate to come from the **same exact population** as the final H4 primary estimand. Use of the holdout for variance planning does not convert its observed treatment-effect mean into development evidence.

---

## 36.11 Reproducibility and Persistence Requirements

A machine-readable holdout-population artifact must be created tests-first before any H4 holdout variance execution is authorized.

That artifact must bind at minimum:

```text
population_id
population_version
population_hash
population_role
population_kind
generation algorithm and seed material
support bounds
quantization rule
exact five unit ids
exact five parameter vectors
exact weights
parameter schema version
parametric-policy version
Gate-6 report hash
reference strength hash
validation strength hash
calibration environment/version
tolerance version
per-unit acceptance hashes
complete acceptance status
```

The artifact must use canonical JSON and semantic tamper rejection rather than trusting only an outer SHA-256 envelope.

Every later H4 variance/final trajectory config must carry the exact frozen population hash and exact unit identity/parameters in controller-side provenance.

---

## 36.12 Execution Boundary

This specification does **not** authorize:

```text
H4 holdout variance-pilot hands
final H4 confirmatory hands
H4 confirmatory inference
```

Before the first 500-hand holdout variance trajectory begins, all of the following must exist and be green:

1. canonical machine-readable `phase2-h4-holdout-population-v1` artifact;
2. exact population-hash validation;
3. deterministic generation tests;
4. official all-six-metric acceptance for all five holdout units;
5. information-boundary tests;
6. the separately versioned H4 500-hand variance-pilot protocol required by Spec 35;
7. its own execution authority/readiness gate.

Specs 33 and 34 authorize only the completed adaptive pilot and do not authorize these H4 holdout runs.

---

## 36.13 Required Next Specification

After the machine-readable holdout population and acceptance gate are closed, the next normative step is:

```text
Phase-2 H4 500-Hand Holdout Variance Pilot v1
```

That protocol must freeze, before its first hand:

```text
exact replicated cluster count
exact disjoint master-seed schedule
500 hands per positional trajectory
same five holdout units and weights
same treatment/control pair
matched CRN relation
failure/complete-pair rule
execution binding/provenance
variance-only analysis and report schema
```

Its observed mean adaptive-control difference must not alter any decision already frozen in Specs 35–36.

---

## 36.14 Validation Gate

Before this holdout-population specification is treated as closed, validation tests must verify at minimum:

1. exact Spec-36 identity, version, status, and specification-index entry;
2. exact binding to Spec 35's `holdout_test` role and 500-hand primary H4 endpoint;
3. exact binding to `opponent-parameters-v1` and `parametric-policy-v1`;
4. exact Gate-6 report, reference-strength, and validation-strength hashes;
5. exact six-dimensional support bounds;
6. deterministic seed-material derivation from the population identifier;
7. exact `sha256-latin-hypercube-fixedpoint-v1` stratum/permutation/jitter semantics;
8. exact six-decimal round-half-even quantization and PFR construction;
9. exact five generated parameter vectors and equal 0.2 weights;
10. exact canonical `population_hash = b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3`;
11. all five units require all six calibration metrics with no `don't-care` exceptions;
12. all five units pass the frozen independent acceptance tolerances before execution;
13. acceptance failure invalidates the entire population and cannot trigger replacement/regeneration;
14. no development/pilot population is silently promoted into H4 holdout evidence;
15. holdout identity/parameters remain controller-side and absent from hero observations/evidence;
16. population aggregation occurs within master seed before across-seed variance/inference;
17. holdout variance/final seed domains remain disjoint from the completed adaptive pilot;
18. no holdout result may tune the agent or alter Spec-35 decision parameters;
19. this specification does not itself authorize any H4 holdout hand.

---

## 36.15 Frozen Population Boundary

After this file is active, the following may not be changed because of any subsequently observed holdout calibration, variance-pilot, or final-H4 treatment effect:

```text
population_id                 = phase2-h4-holdout-population-v1
population_version            = phase2-h4-holdout-parametric-population-v1
population_role               = holdout_test
population_kind               = fixed-finite-equal-weight-parametric-v1
num_units                     = 5
unit_weight                   = 0.2
generation_algorithm          = sha256-latin-hypercube-fixedpoint-v1
seed_material                 = a1f17ebdb8504b9c5b59e70439fbe4eaf4a6de559b6436e996f8a0c1c9647d75
parameter support             = Section 36.3
quantization                  = six-decimal round-half-even
exact five target vectors     = Section 36.5
population_hash               = b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3
calibration environment       = policy-calibration-environment-v1
calibration tolerances        = calibration-tolerances-v1
population aggregation        = within-seed equal-weight first
replacement units             = forbidden
post-seal agent optimization  = forbidden
```

Changing any of these requires a new versioned holdout-population protocol. Already-inspected H4 holdout outcomes may not be used as the justification for selecting a more favourable replacement population.
