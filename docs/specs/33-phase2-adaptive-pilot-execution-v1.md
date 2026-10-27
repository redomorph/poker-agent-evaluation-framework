# Phase-2 Adaptive Pilot Execution v1

## 33. Status and Scope

This file is normative for authorizing, planning, launching, resuming, and preserving execution evidence for the already-frozen `phase2-adaptive-pilot-v1` design in Spec 32.

```text
execution_contract_id       = phase2-adaptive-pilot-execution-v1
execution_contract_version  = phase2-adaptive-pilot-execution-contract-v1
status                      = frozen
authorized_protocol_id      = phase2-adaptive-pilot-v1
authorized_protocol_hash    = ef86d3da94369a86ac1c496ddf981b39d30919e086b35e026039bac3afc8ccb4
purpose                     = operational_execution_authorization_and_reproducibility_only
```

This execution contract changes **no scientific choice** in Spec 32. Treatment/control definitions, opponents, populations, seeds, trajectory lengths, horizons, estimators, failure policy, permitted analysis, and the final-H4 boundary remain exactly those of the frozen Spec-32 payload.

Spec 32 was intentionally frozen while execution was not yet authorized. Its `design_frozen_execution_not_yet_authorized` status records that design-freeze stage and MUST NOT be rewritten after the fact. This later Spec 33 is the sole normative execution authorization for that unchanged payload once the Spec-32 validation gate is green.

The machine-readable execution contract is:

```text
configs/experiments/phase2-adaptive-pilot-execution-v1.json
```

No real adaptive-pilot hand may execute until that file is canonical, hash-verified, semantically frozen, and enforced by the launch layer.

## 33.1 Entry Gate

Before execution authorization is used on a real campaign:

1. Spec 31 lifecycle validation is green;
2. every minimum item in Spec 32 Section 32.15 is covered by passing tests;
3. the complete repository test suite is green on the committed scientific state used for planning;
4. `src/`, `configs/`, `docs/specs/`, `pyproject.toml`, and `poetry.lock` are clean and committed before binding capture;
5. the exact Spec-32 protocol payload hash equals the authorized hash in Section 33.

The test-suite result is process evidence, not a scientific parameter and not an RNG input. The runtime binding remains responsible for proving the exact committed source/configuration/environment identity used for the campaign.

## 33.2 Frozen Execution Components

The authorized v1 execution layer is exactly:

```text
execution_enforcement_version = phase2-adaptive-pilot-execution-enforcement-v1
execution_binding_version     = phase2-adaptive-pilot-execution-binding-v1
trajectory_config_version     = phase2-adaptive-pilot-trajectory-config-v1
run_manifest_version          = phase2-adaptive-pilot-run-manifest-v1
arm_snapshot_version          = phase2-adaptive-pilot-arm-snapshot-v1
pair_checkpoint_version       = phase2-adaptive-pilot-pair-checkpoint-v1
launch_version                = phase2-adaptive-pilot-launch-v1
internal_cli_module           = pokerbench.experiments.adaptive_pilot_cli
workers                       = 1
```

The serial worker count is frozen only for this v1 execution implementation so recorded runtime provenance matches actual scheduling. It does not alter cards, agent RNG identities, the statistical unit, or scientific estimators. Parallel execution requires a separately versioned execution contract, not a silent v1 change.

## 33.3 Plan, Status, Readiness, Run, and Resume

The internal CLI exposes exactly:

```text
plan
status
readiness
run
resume
```

Their meanings are:

- `plan`: verify frozen assets, capture one immutable execution binding, initialize persistence, and materialize the exact 50-job plan without executing a hand;
- `status`: read persisted state without recapturing the current environment and without executing a hand;
- `readiness`: recapture the live execution binding and fail unless it exactly matches the persisted binding and the frozen 50-job / 25-pair / 18,000-hand plan;
- `run`: execute a virgin plan only after readiness succeeds;
- `resume`: continue the same bound campaign only after readiness succeeds, using the frozen crash-consistency rules.

`run` MUST refuse a campaign that already contains execution state and direct the operator to `resume`. Neither command may accept alternate arms, profiles, seeds, horizons, worker counts, or replacement seeds.

## 33.4 Execution Authorization Binding

Every persisted execution binding MUST additionally bind:

```text
execution_contract_id
execution_contract_version
execution_contract_hash
```

The execution-contract hash is the SHA-256 of the canonical nested payload in the machine-readable Spec-33 companion.

Planning, readiness, run, and resume MUST fail closed if the execution contract is missing, non-canonical, hash-invalid, semantically altered, or authorizes a different Spec-32 protocol hash.

A previously planned campaign cannot be resumed under a different execution-contract hash. Such a change requires a new work root/campaign identity.

## 33.5 Frozen Asset and Information Checks

Before plan/readiness succeeds, the launch layer MUST verify at least:

- the passing Gate-6 report and its exact bound hash;
- the Gate-6 reference-strength identity;
- the frozen adaptive preflop strength reference;
- the Baseline Synthetic and Balanced population identities/hashes;
- the exact Spec-32 protocol hash;
- all adaptive dependency hashes already required by Spec 32;
- the current committed Git/lockfile/software/hardware binding.

Opponent profile identity and true parameters remain controller-side provenance only. This execution contract does not widen the hero information boundary from Specs 28, 31, or 32.

## 33.6 Job Identity, Persistence, and Resume

The job matrix is obtained exclusively from the validated Spec-32 protocol. The canonical expansion remains exactly 50 arm/profile/seed trajectory-cluster jobs and 25 matched treatment/control pairs representing 18,000 engine hands.

Each job has one immutable trajectory configuration and one deterministic match identity derived from the frozen execution binding and trajectory configuration.

The v1 work root preserves:

```text
binding.json
run-manifest.json
trajectory-configs/
arm-snapshots/
pair-checkpoints/
```

Canonical JSON artifacts are written atomically. Completed or scientifically failed jobs are terminal and MUST NOT be rerun or replaced.

Crash handling is fail-closed:

- stale `running` with no durable MatchRecord and no arm snapshot may return to `planned` and retry the same canonical job;
- an immutable arm snapshot may restore a missing store copy after hash verification without rerunning the trajectory;
- a durable MatchRecord without its required arm snapshot is an integrity error and MUST NOT be guessed/reconstructed as runtime evidence;
- pair checkpoints are created only when both canonical arms are terminal and existing complete-pair/replay validation succeeds.

## 33.7 CLI Defaults

The internal command remains:

```text
python -m pokerbench.experiments.adaptive_pilot_cli
```

The v1 defaults are:

```text
--protocol             configs/experiments/phase2-adaptive-pilot-v1.json
--execution-contract   configs/experiments/phase2-adaptive-pilot-execution-v1.json
--primary-population   configs/populations/baseline-synthetic-v1.json
--secondary-population configs/populations/balanced-v1.json
--snapshot-root        data/gate6/snapshots
--gate6-report         data/gate6/gate6-report.json
--reference            configs/experiments/preflop-range-strength-reference-v1.json
--database             data/phase2/adaptive-pilot/phase2-adaptive-pilot-v1.sqlite
--work-root            data/phase2/adaptive-pilot/work
--repo-root            .
--poetry-lock          poetry.lock
--workers              1
```

`--ram-bytes` remains mandatory for `plan` because the existing reproducibility environment capture records RAM explicitly.

## 33.8 Authorization Boundary

Spec 33 authorizes execution of the **pilot only** after the machine-readable execution contract and validation gate below are green. It does not authorize:

```text
confirmatory_h4_inference
final_h4_data_collection
pilot_plus_final_pooling
h4_primary_horizon_selection_from_observed_pilot_delta
adaptive_policy_tuning_from_pilot_ev
mcts_evaluation
```

The final H4 design fields in Spec 32 remain null and require a separate frozen protocol before any final H4 hand.

## 33.9 Validation Gate Before First Pilot Hand

Before the first real pilot hand, tests MUST verify at minimum:

1. canonical machine-readable execution-contract identity, payload hash, and semantic tamper rejection;
2. exact authorization of Spec-32 payload hash `ef86d3da94369a86ac1c496ddf981b39d30919e086b35e026039bac3afc8ccb4`;
3. exact v1 execution component versions and serial `workers = 1`;
4. execution binding and every trajectory configuration include the exact execution-contract identity/hash;
5. planning/readiness reject a missing, altered, or differently authorizing execution contract;
6. resume rejects execution-contract drift from the campaign originally planned;
7. CLI exposes the frozen `--execution-contract` path and all five commands use it;
8. no command capable of executing a hand bypasses execution-contract validation;
9. the complete Spec-32 50-job / 25-pair / 18,000-hand readiness checks remain intact.

Only after this validation gate is green and the scientific tree is committed/clean may the first real adaptive-pilot hand be executed.

## 33.10 Change Control

`phase2-adaptive-pilot-execution-v1` is frozen once active. A change to authorization semantics, execution component versions, worker model, persistence/recovery semantics, CLI execution boundary, or the authorized Spec-32 protocol hash requires a new execution-contract version.

Operational bug fixes that do not change these semantics still require tests first and must preserve replay/traceability of any existing campaign.
