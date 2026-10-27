# Phase-2 H4 Holdout Variance Pilot Execution v1

## 38. Status and Scope

This file is normative for authorizing, planning, launching, resuming, and preserving execution evidence for the already-frozen `phase2-h4-holdout-variance-pilot-v1` scientific design in Spec 37.

```text
execution_contract_id       = phase2-h4-holdout-variance-execution-v1
execution_contract_version  = phase2-h4-holdout-variance-execution-contract-v1
status                      = frozen
authorized_protocol_id      = phase2-h4-holdout-variance-pilot-v1
authorized_protocol_version = phase2-h4-holdout-variance-pilot-protocol-v1
authorized_protocol_hash    = d2ad5569a82308a61390ccdc2212ac3b2790fa369e96506a418bac4a7045aac0
purpose                     = operational_execution_authorization_and_reproducibility_only
workers                     = 4
max_in_flight_jobs          = 4
process_model               = spawn-compatible-process-pool-v1
```

Spec 38 changes **no scientific choice** in Specs 35-37. The holdout population, acceptance gate, treatment/control definitions, master seeds, 500-hand positional trajectories, CRN relationship, 100-job matrix, 10/10 population-cluster rule, replay admission, `sigma_500` estimator, and deterministic Spec-35 sample-size formulas remain unchanged.

Spec 37 intentionally remains marked `design_frozen_execution_not_yet_authorized`; that historical design-stage marker MUST NOT be rewritten. This later Spec 38 is the sole v1 execution authority for the unchanged Spec-37 payload once its own machine-readable contract and validation/readiness gate are green.

The machine-readable execution contract is:

```text
configs/experiments/phase2-h4-holdout-variance-execution-v1.json
```

No real holdout-variance hand may execute until that file is canonical, hash-verified, semantically validated, bound into the campaign, and enforced by every command capable of executing a hand.

---

## 38.1 Entry Gate

Before a real variance-pilot campaign may be planned for execution:

1. Specs 35-37 are frozen and green;
2. the canonical machine-readable Spec-37 protocol hash equals the authorized hash in Section 38;
3. the Spec-36 holdout population is canonical and has the exact frozen population hash;
4. the official holdout acceptance artifact is canonical, complete, and 5/5 accepted;
5. Spec-37 execution-enforcement and replay-admission tests are green;
6. Spec-37 binding/provenance and resumable-persistence tests are green;
7. the complete repository test suite is green on the committed scientific state used for binding capture;
8. `src/`, `configs/`, `docs/specs/`, `pyproject.toml`, and `poetry.lock` are clean and committed before live binding capture;
9. the host exposes at least four logical CPU cores;
10. execution provenance records the actual four-worker contract and exact runtime environment.

The test-suite result is process evidence, not a scientific parameter and not an RNG input.

---

## 38.2 Frozen Execution Components

The authorized v1 execution layer is exactly:

```text
execution_enforcement_version = phase2-h4-holdout-variance-execution-enforcement-v1
execution_binding_version     = phase2-h4-holdout-variance-execution-binding-v1
trajectory_config_version     = phase2-h4-holdout-variance-trajectory-config-v1
persistence_layout_version    = phase2-h4-holdout-variance-persistence-layout-v1
run_manifest_version          = phase2-h4-holdout-variance-run-manifest-v1
arm_snapshot_version          = phase2-h4-holdout-variance-arm-snapshot-v1
pair_checkpoint_version       = phase2-h4-holdout-variance-pair-checkpoint-v1
cluster_checkpoint_version    = phase2-h4-holdout-variance-cluster-checkpoint-v1
parallel_run_version          = phase2-h4-holdout-variance-parallel-run-v1
parallel_launch_version       = phase2-h4-holdout-variance-parallel-launch-v1
internal_cli_module           = pokerbench.experiments.h4_holdout_variance_parallel_cli
workers                       = 4
max_in_flight_jobs            = 4
process_model                 = spawn-compatible-process-pool-v1
worker_model                  = process_pool_spawn_compatible_top_level_task
parent_commit_model           = single_parent_serial_commit
```

Worker count and process placement are execution provenance only. They MUST NOT enter scientific trajectory identity or change any estimator/admission rule.

---

## 38.3 Worker Isolation and RNG Invariance

Every submitted worker task receives exactly one already-frozen canonical Spec-37 arm job, its bound trajectory configuration, and deterministic match identity.

Each worker MUST construct fresh instances of:

```text
hero agent
holdout opponent policy
runner / engine state
adaptive Bayesian state for treatment
isolated capture store
```

No mutable agent, opponent, Bayesian model, posterior state, RNG object, runner state, or capture store may be shared between jobs or worker processes.

The following execution metadata MUST NOT enter any scientific identity or RNG derivation:

```text
worker number
PID
process creation order
queue position
submission order
completion order
wall-clock timing
parent commit order
```

In particular, they MUST NOT enter:

- master seed;
- holdout-unit identity;
- deterministic deal schedule;
- `AgentRNG` namespace or event index;
- trajectory configuration hash;
- deterministic match ID;
- adaptive prior or posterior state;
- treatment/control pairing;
- population-cluster identity.

The Spec-37 CRN contract remains exactly:

```text
deal_relation             = identical_defined_card_realisations
agent_rng_relation        = event_indexed_common_identity_where_events_correspond
full_action_path_identity = not_claimed
```

Parallel execution MUST NOT strengthen or weaken this claim.

---

## 38.4 Worker Persistence Boundary

Workers MUST NOT write to shared campaign persistence, including:

```text
campaign SQLite database
run-manifest.json
trajectory-configs/
arm-snapshots/
pair-checkpoints/
cluster-checkpoints/
```

A worker executes one guarded arm job against an isolated in-memory capture store and returns a complete result and its hash-verifiable `MatchRecord` to the parent.

No partial arm evidence is scientific campaign evidence. If a worker terminates before returning a complete result, the affected job remains non-terminal and may later be recomputed only under the same canonical job identity. No replacement seed or alternate holdout unit is permitted.

---

## 38.5 Parent-Only Durable Commit and Crash Consistency

Only the parent process may mutate campaign persistence. Worker results may complete in arbitrary order, but the parent is a single serialized writer.

For each accepted completed arm result, the mandatory parent commit order is:

```text
1. validate canonical job / config / match identity
2. build and atomically persist the immutable arm snapshot
3. save or verify the exact MatchRecord in the campaign SQLite store
4. atomically mark the canonical manifest job terminal
5. create or verify a newly complete treatment/control pair checkpoint
6. create or verify a newly complete five-unit seed-cluster checkpoint
```

Snapshot-before-database ordering is mandatory.

Crash semantics remain fail-closed:

- snapshot present + missing SQLite copy: recover SQLite from the immutable snapshot after full hash/identity validation;
- SQLite record present + missing required snapshot: fail closed as orphaned durable state;
- stale `running` with neither snapshot nor SQLite record: return to `planned` on readiness/resume;
- completed snapshot/record/checkpoint identity conflict: fail closed;
- no terminal job may be executed twice.

A parent crash may lose a worker result that existed only in memory. Such a job may be recomputed under its unchanged canonical identity because no durable scientific evidence had yet been committed.

---

## 38.6 Scheduler and Scientific Cluster Separation

The runtime scheduler manages only the 100 canonical arm jobs with states:

```text
planned
running
completed
failed
```

At most four non-terminal jobs may be physically in flight at once.

The scheduler submits `planned` jobs in canonical Spec-37 order and preserves the immutable 100-job manifest ordering regardless of runtime completion order.

Scheduler terminality is **not** scientific population-cluster admission.

A seed-level cluster becomes scientifically admissible only through the existing Spec-37 replay/enforcement chain:

```text
2 valid arms
-> 1 replay-valid treatment/control unit pair
-> all 5 canonical unit pairs for one seed
-> 1 valid seed-level population cluster
```

The execution layer MUST NOT calculate `sigma_500` or final H4 sample size from scheduler state. Those outputs belong only to the post-run analysis layer after exact 10/10 cluster admission.

---

## 38.7 Plan, Status, Readiness, Run, and Resume

The internal CLI exposes exactly:

```text
plan
status
readiness
run
resume
```

Their meanings are:

- `plan`: validate frozen assets, capture one immutable execution binding, initialize the isolated campaign layout, and materialize the exact 100-job plan without executing a hand;
- `status`: read persisted campaign state without recapturing the current environment and without executing a hand;
- `readiness`: recapture the live committed environment and fail unless it exactly matches the persisted binding, execution authority, and frozen 100-job / 50-pair / 10-cluster / 100,000-hand plan;
- `run`: execute a virgin plan only after readiness succeeds;
- `resume`: continue the same bound campaign only after readiness succeeds and after fail-closed durable-state reconciliation.

`run` MUST refuse any non-virgin campaign and direct the operator to `resume`.

`plan`, `status`, and `readiness` MUST NOT invoke the scientific arm executor. `readiness` MUST be byte-nonmutating with respect to the campaign work root and database.

No command may accept alternate seeds, alternate holdout units, alternate arms, altered horizon, replacement seeds, or a different worker count under this execution-contract version.

---

## 38.8 Frozen CLI Defaults and Persistence Roots

The v1 defaults are:

```text
--protocol             configs/experiments/phase2-h4-holdout-variance-pilot-v1.json
--execution-contract   configs/experiments/phase2-h4-holdout-variance-execution-v1.json
--population           configs/populations/phase2-h4-holdout-population-v1.json
--acceptance           data/gate6/h4-holdout-acceptance-v1.json
--snapshot-root        data/gate6/snapshots
--gate6-report         data/gate6/gate6-report.json
--reference            configs/experiments/preflop-range-strength-reference-v1.json
--database             data/phase2/h4-holdout-variance/variance-pilot.sqlite
--work-root            data/phase2/h4-holdout-variance/work
--repo-root            .
--poetry-lock          poetry.lock
--workers              4
```

The persistence layout derived from the work root is exactly:

```text
data/phase2/h4-holdout-variance/work/run-manifest.json
data/phase2/h4-holdout-variance/work/trajectory-configs/
data/phase2/h4-holdout-variance/work/arm-snapshots/
data/phase2/h4-holdout-variance/work/pair-checkpoints/
data/phase2/h4-holdout-variance/work/cluster-checkpoints/
```

`--ram-bytes` is mandatory for `plan` so the planned binding records explicit RAM rather than inferring a possibly unstable value later.

Generated SQLite/WAL/SHM and `work/` artifacts remain runtime evidence, not source files, and are covered by the existing Phase-2 generated-storage ignore policy.

---

## 38.9 Execution Authorization Binding

Every persisted execution binding MUST bind exactly:

```text
execution_contract_id
execution_contract_version
execution_contract_hash
```

plus every scientific/runtime field already required by the Spec-37 binding layer.

The execution-contract hash is the SHA-256 of the canonical machine-readable Spec-38 payload.

Planning, readiness, run, and resume MUST fail closed if the execution contract is:

```text
missing
non-canonical
hash-invalid
semantically altered
authorizing a different protocol id/version/hash
using workers != 4
using a different process model
using different persistence roots
```

A previously planned campaign cannot be resumed under a different execution-contract hash. Any such change requires a distinct work root/campaign identity and a new versioned execution authority when semantics change.

---

## 38.10 Frozen Asset and Holdout-Seal Checks

Before `plan` or `readiness` succeeds, the launch layer MUST verify at minimum:

- exact Spec-37 canonical protocol hash `d2ad5569a82308a61390ccdc2212ac3b2790fa369e96506a418bac4a7045aac0`;
- exact Spec-36 population hash `b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3`;
- exact holdout acceptance hash `9ba347fb1d615cec1d6458944df3c293dcac630564e8847a407e987bbb98af84` and 5/5 acceptance;
- exact five holdout unit identities and 0.2 weights;
- exact adaptive dependency hashes and versions bound by Spec 37;
- exact game-config hash;
- exact seeds `6,400,000 ... 6,400,009`;
- exact `500 + 500` positional horizon per arm job;
- exact execution-contract identity/hash;
- clean committed scientific tree;
- exact lockfile and live environment binding.

Holdout identity, population weight, true parameters, generation material, and acceptance metadata remain controller-side provenance and MUST NOT enter hero observations or adaptive learning evidence.

---

## 38.11 Failure Semantics

A Python exception, worker-process failure, broken process pool, parent persistence failure, illegal action, disqualifying timeout, engine error, agent error, replay failure, or identity/hash mismatch MUST NOT be silently converted into valid variance evidence.

Scientific failures represented by a complete guarded result may be durably recorded under the existing arm/pair/cluster rules. Infrastructure failures before a complete result is returned stop the active launch and leave affected non-terminal jobs resumable under their unchanged canonical identities.

This execution authority MUST NOT:

```text
append an 11th seed
replace a failed seed
replace a holdout unit
accept 9/10 clusters for sigma_500
change worker count to finish a running campaign
convert infrastructure failure into scientific completion
```

Spec 37's all-or-nothing 10/10 scientific admission remains authoritative.

---

## 38.12 Preflight Before the First Real Hand

A real campaign may not start immediately after applying this human-readable specification.

The following non-executing sequence is mandatory after the machine-readable Spec-38 contract and launch layer are green:

```text
1. complete repository test suite green
2. commit all scientific state
3. scientific-tree cleanliness check returns no changes
4. plan the exact campaign
5. inspect status
6. execute readiness
7. verify binding hash is identical across plan/readiness
8. verify 100 planned jobs, 50 planned pairs, 10 planned clusters, 100,000 planned engine hands
9. verify zero MatchRecords, zero arm snapshots, zero pair checkpoints, and zero cluster checkpoints before run
10. only then invoke run
```

The preflight must prove that `plan` and `readiness` cannot call the scientific executor and that `readiness` leaves campaign bytes unchanged.

---

## 38.13 Authorization Boundary

Spec 38 authorizes only operational execution of the unchanged Spec-37 **variance pilot** after its machine-readable contract and validation/readiness gate are green.

It does not authorize:

```text
confirmatory H4 inference
final H4 data collection
final H4 seed selection
policy or opponent tuning
holdout-population tuning
variance-pilot mean as a design input
pilot/final pooling
reduced-sample sigma_500
replacement seeds
MCTS evaluation
```

Only a complete replay-admitted Spec-37 post-run variance report may later supply `sigma_500` and deterministic sample-size outputs to the separate final H4 protocol.

---

## 38.14 Validation Gate Before First Variance-Pilot Hand

Before the first real variance-pilot hand, tests MUST verify at minimum:

1. canonical machine-readable execution-contract identity, payload hash, and semantic tamper rejection;
2. exact authorization of Spec-37 payload hash `d2ad5569a82308a61390ccdc2212ac3b2790fa369e96506a418bac4a7045aac0`;
3. exact `workers = 4`, `max_in_flight_jobs = 4`, and spawn-compatible process model;
4. exact frozen execution component versions in Section 38.2;
5. every persisted binding and trajectory config contains the exact execution-contract identity/hash;
6. planning/readiness reject a missing, altered, or differently authorizing execution contract;
7. resume rejects execution-contract, Git, lockfile, hardware-binding, worker-count, process-model, or persistence-path drift;
8. workers construct fresh hero/opponent/runner/adaptive state and cannot write campaign persistence;
9. process/PID/queue/completion metadata is absent from scientific RNG and job identity;
10. parent-only commit order is snapshot -> SQLite -> manifest -> pair checkpoint -> cluster checkpoint;
11. snapshot-to-SQLite recovery works and orphan SQLite records fail closed;
12. stale non-durable `running` jobs reset to `planned` without duplicate terminal execution;
13. plan creates exactly 100 jobs / 50 pairs / 10 clusters / 100,000 hands and zero scientific evidence;
14. readiness is byte-nonmutating and cannot invoke the executor;
15. run accepts only a virgin plan; resume accepts only the same bound non-virgin campaign;
16. pair and cluster checkpoints are created only through replay-valid Spec-37 admission;
17. scheduler state cannot directly produce `sigma_500` or `n_final`;
18. the exact five-unit/ten-seed all-or-nothing scientific rule remains enforced;
19. no command can inject replacement seeds, alternate holdout units, or an alternate worker count;
20. the complete repository test suite is green on the committed state used for live planning.

Only after this validation gate is green and the scientific tree is committed/clean may the first real holdout-variance hand be generated.

---

## 38.15 Change Control

`phase2-h4-holdout-variance-execution-v1` is frozen once active.

A change to any of the following requires a new versioned execution authority:

```text
authorized Spec-37 protocol hash
workers / max_in_flight_jobs
process or worker model
parent persistence boundary
commit/recovery ordering
persistence layout/default campaign roots
plan/readiness/run/resume semantics
execution component versions
execution-contract payload semantics
```

Operational bug fixes that preserve all frozen semantics still require tests first and MUST preserve replay/traceability for any existing campaign.

Spec 38 MUST NOT be edited to explain or accommodate observed variance-pilot outcomes.
