# Phase-2 Adaptive Pilot Parallel Execution v1

## 34. Status and Scope

This file is normative for the parallel execution authorization of the already-frozen `phase2-adaptive-pilot-v1` design from Spec 32.

```text
execution_contract_id       = phase2-adaptive-pilot-parallel-execution-v1
execution_contract_version  = phase2-adaptive-pilot-parallel-execution-contract-v1
status                      = frozen
authorized_protocol_id      = phase2-adaptive-pilot-v1
authorized_protocol_hash    = ef86d3da94369a86ac1c496ddf981b39d30919e086b35e026039bac3afc8ccb4
purpose                     = parallel_operational_execution_authorization_only
workers                     = 4
```

Spec 34 changes **no scientific choice** in Specs 28-32. Treatment/control definitions, opponents, populations, master seeds, deterministic deal schedules, trajectory lengths, positional resets, statistical units, estimators, failure handling, analysis permissions, and the final-H4 boundary remain unchanged.

Spec 33 remains the frozen serial execution authority with `workers = 1`. Spec 34 is a separate execution authority for the same Spec-32 payload and MUST NOT be interpreted as an amendment of Spec 33.

The machine-readable parallel execution contract is:

```text
configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json
```

## 34.1 Entry Gate

Before a real parallel pilot launch:

1. Specs 31-33 and the Spec-32 preflight gate are green;
2. the complete repository test suite is green on the committed scientific state;
3. no serial or parallel pilot trajectory is already terminal for the campaign being planned;
4. `src/`, `configs/`, `docs/specs/`, `pyproject.toml`, and `poetry.lock` are clean and committed before binding capture;
5. the exact authorized Spec-32 payload hash matches Section 34;
6. the host reports at least four logical CPU cores;
7. execution provenance records the actual four-worker parallel contract.

The previous aborted serial launch with zero terminal jobs is not pilot evidence and MUST NOT be imported into the parallel campaign.

## 34.2 Frozen Parallel Components

The authorized v1 parallel execution layer is exactly:

```text
execution_enforcement_version = phase2-adaptive-pilot-execution-enforcement-v1
execution_binding_version     = phase2-adaptive-pilot-execution-binding-v1
trajectory_config_version     = phase2-adaptive-pilot-trajectory-config-v1
run_manifest_version          = phase2-adaptive-pilot-run-manifest-v1
arm_snapshot_version          = phase2-adaptive-pilot-arm-snapshot-v1
pair_checkpoint_version       = phase2-adaptive-pilot-pair-checkpoint-v1
parallel_run_version          = phase2-adaptive-pilot-parallel-run-v1
parallel_launch_version       = phase2-adaptive-pilot-parallel-launch-v1
internal_cli_module           = pokerbench.experiments.adaptive_pilot_parallel_cli
workers                       = 4
max_in_flight_jobs            = 4
worker_model                  = process_pool_spawn_compatible_top_level_task
parent_commit_model           = single_parent_serial_commit
```

The scientific trajectory configuration version remains unchanged because process placement is execution provenance, not a trajectory-level scientific parameter.

## 34.3 Isolation and RNG Invariance

Every worker receives exactly one already-frozen canonical job, trajectory configuration, and deterministic match ID. Worker/process identity, scheduling order, PID, completion order, wall-clock timing, and queue position MUST NOT enter:

- the master seed;
- the deal schedule;
- any `AgentRNG` namespace or event index;
- the trajectory configuration hash;
- the deterministic match ID;
- adaptive priors or posterior state;
- opponent parameters.

Each job constructs fresh hero, opponent, runner, and adaptive state inside its worker. No agent or Bayesian model instance may be shared between jobs or processes.

The Spec-32 CRN statement remains exactly:

```text
deal_relation              = identical_defined_card_realisations
agent_rng_relation         = event_indexed_common_identity_where_events_correspond
full_action_path_identity  = not_claimed
```

Parallel scheduling MUST NOT strengthen or weaken that claim.

## 34.4 Worker Persistence Boundary

Workers MUST NOT write to the campaign SQLite database, `run-manifest.json`, `arm-snapshots/`, or `pair-checkpoints/`.

A worker executes the guarded runner against an isolated in-memory capture store and returns one complete `PairedResult` with its hash-verified `MatchRecord` to the parent. No partial scientific evidence is committed from a running worker.

If a worker process terminates before returning a complete result, its job remains non-terminal and may be retried under the same canonical identity. No replacement seed is permitted.

## 34.5 Parent Commit Order and Crash Consistency

Only the parent process may mutate campaign persistence. Results may finish in any order. Parent persistence is single-writer and serialized one completed result at a time; commit order may follow worker completion order. The manifest itself retains canonical job ordering and immutable per-job identities.

For each accepted job, the parent commit order is:

```text
1. validate returned job/config/match identity and build immutable arm snapshot
2. atomically persist the arm snapshot
3. save or verify the exact MatchRecord in the campaign SQLite store
4. atomically mark the canonical manifest job terminal
5. create/verify any newly complete matched-pair checkpoint
```

Snapshot-before-database ordering is mandatory so a crash after snapshot creation can use the already-frozen snapshot-to-store recovery rule. A database MatchRecord without its required arm snapshot remains a fail-closed integrity error.

## 34.6 Scheduling and Resume

At most four non-terminal jobs may be `running` concurrently. The scheduler submits canonical planned jobs in canonical job order, keeps at most four jobs physically in flight, and preserves deterministic job identities regardless of completion or parent-commit order.

Completed or scientifically failed jobs are terminal and MUST NOT be resubmitted. On resume:

- a stale `running` job with neither arm snapshot nor durable campaign MatchRecord returns to `planned`;
- an immutable snapshot may restore a missing campaign-store copy after hash verification;
- an orphan campaign MatchRecord without its arm snapshot fails closed;
- no replacement or opportunistic seed is introduced.

The parent may lose completed-but-not-yet-committed in-memory worker results in a process crash; such jobs are safely recomputed from their unchanged canonical identities on resume.

## 34.7 Plan, Status, Readiness, Run, and Resume

The parallel internal CLI exposes exactly:

```text
plan
status
readiness
run
resume
```

The meanings remain those of Spec 33, except that the persisted binding and live readiness check must authorize this Spec-34 contract and exactly four workers.

Defaults are isolated from the historical serial work root:

```text
--protocol             configs/experiments/phase2-adaptive-pilot-v1.json
--execution-contract   configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json
--primary-population   configs/populations/baseline-synthetic-v1.json
--secondary-population configs/populations/balanced-v1.json
--snapshot-root        data/gate6/snapshots
--gate6-report         data/gate6/gate6-report.json
--reference            configs/experiments/preflop-range-strength-reference-v1.json
--database             data/phase2/adaptive-pilot-parallel/phase2-adaptive-pilot-v1.sqlite
--work-root            data/phase2/adaptive-pilot-parallel/work
--repo-root            .
--poetry-lock          poetry.lock
--workers              4
```

`--ram-bytes` remains mandatory for `plan`.

`run` MUST refuse any non-virgin plan and direct the operator to `resume`. `readiness`, `run`, and `resume` recapture the live committed environment and fail on binding drift.

## 34.8 Scientific Equivalence Requirement

Parallel execution is scientifically admissible only because scheduling is outside all frozen RNG and model-state identities.

Tests MUST establish that, for the same canonical job/config/match identity:

- serial and parallel worker construction use the same guarded runner inputs;
- process/scheduling metadata is absent from trajectory provenance and match identity;
- deterministic deal-schedule proof is unchanged;
- worker completion order cannot change final manifest job ordering, snapshot identity, record identity, or pair identity;
- treatment/control jobs never share mutable agent/adaptive state.

This specification does not claim equal wall-clock timing or identical OS process behavior.

## 34.9 Failure Semantics

A Python exception, worker-process failure, broken process pool, parent persistence failure, timeout-altered adaptive trajectory, illegal action, agent error, or engine error MUST NOT be silently converted into valid pilot evidence.

Scientific failures already represented by a complete guarded `PairedResult` are durably committed as failed terminal jobs under the existing Spec-32 complete-pair rules.

Infrastructure failures before a complete result is returned stop the launch and leave the affected non-terminal jobs resumable under their exact canonical identities. The system MUST NOT mark an infrastructure-failed job scientifically failed merely to continue the matrix.

## 34.10 Authorization Boundary

Spec 34 authorizes only parallel execution of the unchanged adaptive pilot. It does not authorize:

```text
confirmatory_h4_inference
final_h4_data_collection
pilot_plus_final_pooling
adaptive_policy_tuning_from_pilot_ev
h4_primary_horizon_selection_from_observed_pilot_delta
mcts_evaluation
```

## 34.11 Validation Gate Before First Parallel Pilot Hand

Before the first real Spec-34 hand, tests MUST verify at minimum:

1. canonical parallel execution-contract identity, exact payload hash, and semantic tamper rejection;
2. exact authorization of the frozen Spec-32 payload and `workers = 4`;
3. Spec-33 serial contract remains valid and unchanged;
4. binding/config round trips support both frozen execution authorities while real launch validates the selected exact contract;
5. planning produces exactly 50 jobs, 25 pairs, and 18,000 hands with a Spec-34 binding;
6. readiness rejects fewer/more than four workers and hosts with fewer than four logical cores;
7. workers cannot write campaign SQLite/manifest/snapshots/pair checkpoints;
8. worker tasks are spawn/pickle compatible and construct fresh per-job runner/agent state;
9. scheduler submits no more than four jobs concurrently and uses canonical job identities only;
10. out-of-order worker completion preserves canonical final manifest ordering and the same final job identities;
11. snapshot is durably written before campaign SQLite/manifest terminal commit;
12. worker infrastructure failure leaves the job resumable and does not create scientific failure evidence;
13. stale-running, snapshot recovery, orphan-record fail-closed, duplicate protection, and pair-finalization invariants remain intact;
14. process/scheduler identity is absent from RNG/provenance/match-ID inputs;
15. `plan` and `readiness` remain non-executing under the parallel CLI;
16. `run`/`resume` cannot bypass Spec-34 contract/readiness validation;
17. connected serial Spec-33 tests and the complete repository suite remain green.

Only after this gate is green and the scientific tree is committed/clean may the first real parallel adaptive-pilot hand execute.

## 34.12 Change Control

`phase2-adaptive-pilot-parallel-execution-v1` is immutable once active. A change to worker count, scheduling model, worker persistence boundary, parent commit semantics, process model, or authorized Spec-32 payload requires a new execution-contract version.
