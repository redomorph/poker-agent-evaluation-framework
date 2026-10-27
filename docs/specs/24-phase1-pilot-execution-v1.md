# Phase-1 Pilot Execution v1

## 24. Status and Scope

This file is normative for executing and resuming the already-frozen
`phase1-pilot-v1` protocol from Section 23.

```text
execution_binding_version = phase1-pilot-execution-binding-v1
job_snapshot_version      = phase1-pilot-job-snapshot-v1
job_config_version        = phase1-pilot-job-config-v1
execution_report_version  = phase1-pilot-execution-report-v1
internal_cli_version      = phase1-pilot-execution-cli-v1
```

This layer does **not** change the pilot design. It binds execution provenance,
materializes the exact frozen job matrix, executes missing jobs, persists each
attempt, and emits variance/runtime evidence only.

The command is internal rather than part of the public `poker` CLI surface:

```text
python -m pokerbench.experiments.pilot_execution_cli
```

No public `poker pilot` or `poker calibrate` command is introduced.

## 24.1 Binding Before the First Pilot Hand

Before any pilot hand executes, one immutable `binding.json` is written under
the pilot work root. The binding contains:

- protocol identity and protocol hash;
- primary and secondary population identities/hashes;
- passing Gate-6 report hash;
- Gate-6 reference-strength hash;
- Git commit;
- Python version;
- Poetry lockfile name and SHA-256;
- CPU model, core count, architecture, RAM bytes, and OS identifier;
- a hardware fingerprint derived canonically from CPU model, core count, and
  architecture;
- the number of worker processes used for the campaign.

`engine_version` is not an additional researcher-selected label in v1. It is
bound deterministically as:

```text
engine_version = "git:" + git_commit
```

This makes the engine identity traceable to the exact source revision while
retaining the separately persisted `git_commit` field required by the data
model.

The worker count is execution provenance because runtime estimation is an
allowed pilot use. It is not a poker/scientific RNG input and MUST NOT affect
cards, agent RNG identity, or the paired statistic.

A campaign can resume only when the newly captured binding is byte-equivalent
in canonical content to the existing binding. A different Git commit, lockfile,
Gate-6 report, machine binding, RAM value, or worker count requires a new work
root/campaign identity rather than silent continuation.

## 24.2 Git Cleanliness

The binding is captured only from committed scientific source/configuration
state. The following paths must contain no tracked or untracked changes at
binding time:

```text
src/
configs/
docs/specs/
pyproject.toml
poetry.lock
```

Operational outputs under `data/` are not part of this cleanliness check so a
campaign can resume after producing checkpoints and databases.

The execution patch itself therefore MUST be committed before the real pilot is
started.

## 24.3 Exact Job Matrix

The job set is obtained exclusively from `iter_pilot_jobs(...)` applied to the
validated `phase1-pilot-v1` protocol. Execution code MUST NOT accept alternate
hero lists, opponent lists, seed lists, or sample sizes.

The resulting set remains exactly:

```text
2 heroes x 5 opponents x 100 seeds = 1,000 paired_deal jobs
```

Each job contains exactly one `paired_deal`, hence two requested engine hands.
The frozen seed itself is the master seed for that job. No replacement seed is
created for a failed job.

## 24.4 Per-Job Experiment Configuration

Every job gets a complete immutable configuration:

```text
{
  frozen protocol payload,
  execution binding,
  exact job identity,
  required run provenance,
  permitted/prohibited scientific-use flags
}
```

Its canonical SHA-256 is the job `experiment_config_hash` stored in the
`MatchRecord` and the SQLite experiment configuration. This is intentionally
more specific than the protocol hash: the protocol identifies the research
design, while the per-job configuration additionally identifies the exact
machine/software binding and seed/cell.

The run-provenance block contains at least every field required by Section
23.10:

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

The five opponent gate maps are revalidated against the bound Gate-6 PASS
before jobs are launched.

## 24.5 Persistence and Crash Consistency

Every completed job produces two durable forms of evidence:

1. a canonical, hash-verified job checkpoint under the work root; and
2. the complete immutable `MatchRecord` plus normalized metadata in SQLite.

The checkpoint contains the complete `MatchRecord` payload, its hash, the
per-job configuration hash, the paired difference for a valid job or retained
failure evidence for a failed job, and elapsed job runtime in nanoseconds.

Persistence order is:

```text
run job
-> atomically persist checkpoint
-> persist/verify SQLite MatchRecord + metadata
```

Therefore, if execution stops after the checkpoint but before SQLite commit,
resume can reconstruct and persist the exact recorded match without rerunning
the seed or inventing runtime evidence.

If SQLite contains a pilot match but the required runtime checkpoint is absent,
v1 refuses to invent the missing runtime measurement and stops with an
integrity error.

## 24.6 Resume Semantics

A valid job checkpoint is a completed frozen seed regardless of scientific
outcome:

```text
status = valid
or
status = failed
```

A `failed` checkpoint is **not** a missing job. Resume MUST NOT rerun it and
MUST NOT schedule a substitute seed.

Only jobs with no valid checkpoint are submitted for execution. Checkpoint
identity includes the binding hash, cell identity, profile/agent versions,
master seed, experiment-config hash, and MatchRecord hash.

Parallel execution is permitted at the job level. Worker completion order has
no statistical meaning; reports are canonicalized independently of completion
order.

## 24.7 Pilot Variance Estimation

The execution report intentionally does not contain a pilot population EV,
paired mean, confidence interval, effect-size test, or confirmatory p-value.

For each of the ten cells, valid paired differences are used only for:

$$
\widehat{\sigma}_{d,BB/100}
=100\,s(d_1,\ldots,d_n),
$$

with the frozen estimator:

```text
sample-sd-paired-difference-bb100-ddof1-v1
```

The report states, per cell:

- requested units;
- completed units;
- valid units;
- failed units;
- requested and actually recorded engine hands;
- $\widehat{\sigma}_{d,BB/100}$ when at least two valid units exist;
- total job runtime;
- median job runtime.

No mean statistic is promoted into this report.

## 24.8 Runtime Evidence

Runtime is measured with `time.perf_counter_ns()` around each complete
`PairedRunner.run()` job. Checkpointed per-job runtime is retained across
resume. The report aggregates the recorded per-job runtimes; idle time between
separate invocations is not treated as benchmark computation time.

Runtime evidence is descriptive engineering evidence only and is never used as
an agent-quality metric.

## 24.9 Report Status and Exit Codes

The final report is written only after all 1,000 frozen jobs have checkpoints.
It contains a canonical SHA-256 report hash.

```text
exit 0 = all frozen jobs completed and no scientific job failures
exit 1 = all frozen jobs completed, but one or more frozen jobs failed
exit 2 = execution, integrity, provenance, configuration, or persistence error
```

Exit code 1 does not trigger replacement seeds. The report retains the exact
failed cells/seeds via their checkpoints and reports valid/failed counts.
Whether incomplete variance evidence is sufficient for final sample-size
planning is a subsequent explicit scientific decision; this execution layer
does not silently make that decision.

## 24.10 CLI Defaults

The internal v1 command defaults are:

```text
--protocol             configs/experiments/phase1-pilot-v1.json
--primary-population   configs/populations/baseline-synthetic-v1.json
--secondary-population configs/populations/balanced-v1.json
--snapshot-root        data/gate6/snapshots
--gate6-report         data/gate6/gate6-report.json
--database             data/phase1/pilot/phase1-pilot-v1.sqlite
--work-root            data/phase1/pilot/work
--report               data/phase1/pilot/phase1-pilot-report.json
--repo-root            .
--poetry-lock          poetry.lock
--workers              1
--progress-every       10
```

`--ram-bytes` is mandatory because Section 14's existing execution-environment
capture deliberately requires RAM to be supplied explicitly rather than guessed
through platform-specific heuristics.

## 24.11 Final-Experiment Boundary

Nothing produced by this execution layer changes the prohibition in Section
23:

```text
confirmatory_inference      = prohibited
final_benchmark_estimate    = prohibited
optional_stopping           = prohibited
```

The only downstream scientific input is the frozen vector of cell-specific
variance estimates, valid-unit counts, and runtime evidence used to design a
separately versioned final Phase-1 protocol before final-data collection.
