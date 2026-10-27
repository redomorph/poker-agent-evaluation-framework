# Phase-1 Final Execution v1

## 26. Status and scope

This document is normative for executing and resuming the already-frozen
`phase1-baseline-experiment-v1` protocol from Section 25.

```text
execution_binding_version     = phase1-final-execution-binding-v1
job_snapshot_version          = phase1-final-job-snapshot-v1
job_config_version            = phase1-final-job-config-v1
execution_report_version      = phase1-final-execution-report-v1
internal_cli_version          = phase1-final-execution-cli-v1
precision_diagnostic_version  = fixed-profile-normal-mean-ci-v1
```

This layer does **not** change any scientific choice from Section 25. In
particular it does not change the ±50 BB/100 primary precision target, the
2,769/361 fixed allocations, the final seed schedules, the five static
opponents, the primary/secondary population weights, or the exclusion of pilot
observations.

The command is internal:

```text
python -m pokerbench.experiments.final_execution_cli
```

No additional public `poker` subcommand is introduced.

## 26.1 Mandatory evidence before the first final hand

Before execution, the runner MUST verify:

1. canonical `phase1-baseline-experiment-v1` protocol bytes/hash;
2. canonical primary and secondary population definitions;
3. the exact completed pilot report frozen by Section 25;
4. the exact pilot report hash
   `088977f01dff06dc1d64146be35ab2b04f8246e30f7c0130e398c29d6cf24fd1`;
5. the exact Gate-6 report hash
   `ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a`;
6. all five selected static-profile gate maps against that Gate-6 PASS;
7. a clean committed scientific source/configuration state.

Pilot evidence is used only as frozen planning provenance. No pilot hand,
paired difference, EV estimate, or bootstrap observation may enter the final
estimators.

## 26.2 Execution binding

Before the first final hand, one immutable `binding.json` is atomically written
under the final work root. It contains at least:

- final protocol id/version/hash;
- bound pilot report hash;
- primary/secondary population identities and hashes;
- Gate-6 report and reference-strength hashes;
- Git commit;
- `engine_version = "git:" + git_commit`;
- Python version;
- Poetry lockfile name and SHA-256;
- CPU model, physical/logical core identity as provided by the existing
  environment capture, architecture, RAM bytes, and OS identifier;
- canonical hardware fingerprint;
- worker-process count.

A campaign may resume only with the same canonical binding hash. Changing the
Git commit, lockfile, machine binding, Gate-6 evidence, population identities,
or worker count requires a different work root/campaign identity.

Worker count is execution/runtime provenance only. It MUST NOT affect the deal
schedule, action RNG identity, equity RNG identity, or paired statistic.

## 26.3 Git cleanliness

The following scientific paths MUST be clean and committed when the final
binding is captured:

```text
src/
configs/
docs/specs/
pyproject.toml
poetry.lock
```

Generated data under `data/` is operational evidence and is not part of this
cleanliness gate.

The execution implementation itself MUST therefore be committed before final
hands are generated.

## 26.4 Exact final job matrix

The job matrix is obtained only through `iter_final_jobs(...)` from the
validated Section-25 protocol.

The exact workload remains:

```text
Random baseline:
  2,769 paired_deal units/profile × 5 profiles = 13,845 units

Hand-strength baseline:
    361 paired_deal units/profile × 5 profiles =  1,805 units

Total:
  15,650 paired_deal units
  31,300 requested engine hands
```

Every job contains exactly one paired-deal statistical unit and therefore two
requested engine hands.

No alternate hero list, opponent list, seed list, sample size, precision target,
or replacement-seed policy is accepted by the executor.

## 26.5 Final seed contract

The final seed schedule remains `phase1-final-seeds-v1`:

```text
Random:        6,200,000 ... 6,202,768
Hand Strength: 6,200,000 ... 6,200,360
```

The common prefix across heroes is intentional CRN reuse. The final schedules
remain disjoint from pilot seeds `6,100,000 ... 6,100,099`.

A failed final seed is retained as failed evidence and is never replaced.

## 26.6 Per-job configuration identity

Each final job receives a complete immutable experiment configuration containing:

```text
frozen final protocol payload
+ final execution binding
+ exact hero/profile/seed job identity
+ Gate-6 gate-map identity
+ opponent parameters
+ RNG configuration
+ execution provenance
+ scientific-use flags
```

Its canonical SHA-256 is persisted as the job `experiment_config_hash` in the
`MatchRecord` and SQLite experiment metadata.

The persisted precision-target metadata is a structured object, not a scalar:

```text
version       = fixed-profile-normal-mean-ci-v1
confidence    = 0.95
target        = 50 BB/100
estimand      = baseline_synthetic_population_mean_ev_bb100
```

## 26.7 Crash consistency and resume

Each completed job is persisted in this order:

```text
run frozen job
-> atomically write hash-verified checkpoint
-> persist/verify MatchRecord + metadata in SQLite
```

The checkpoint contains the complete MatchRecord payload, MatchRecord hash,
experiment-config hash, paired difference for a valid unit or retained failure
evidence for a failed unit, runtime, hand count, and binding identity.

On resume:

```text
valid checkpoint   -> completed frozen seed; do not rerun
failed checkpoint  -> completed failed seed; do not rerun or replace
missing checkpoint -> execute exactly that frozen seed
```

If SQLite contains a final match but the required runtime checkpoint is absent,
the executor MUST stop rather than invent missing runtime evidence.

Parallel worker completion order has no statistical meaning. Final scientific
records and reports are canonicalized independently of completion order.

## 26.8 Final population estimands

For a hero whose complete frozen allocation is failure-free, the final executor
constructs run-level hero EV evidence from each paired difference using the
already-frozen zero-sum conversion:

$$
\widehat{EV}_{hero,100}=50d.
$$

It then runs `population-evaluation-v1` separately for:

1. the primary Baseline Synthetic population; and
2. the secondary Balanced population.

The final report therefore includes the frozen population metrics:

- weighted mean EV/100;
- weighted median EV/100;
- `R10`;
- `Q25`;
- weighted population standard deviation;
- empirical minimum;
- nested-bootstrap 95% interval for population mean;
- nested-bootstrap 95% interval for `R10`;
- five-profile tail-precision warning.

The five static profiles MUST NOT be represented as the complete `D_test`
population benchmark or as precise lower-tail evidence.

## 26.9 Bootstrap seed derivation

Population-bootstrap randomness is deterministic and domain-separated from all
poker RNG.

For each `(hero, population)` pair, its bootstrap seed is derived
algorithmically from canonical SHA-256 evidence containing:

```text
domain        = phase1-final-population-bootstrap-v1
protocol_hash
hero_id
population_id
```

The first 64 digest bits interpreted as an unsigned integer form the local
bootstrap seed.

This removes a post-data researcher choice. The bootstrap remains at the frozen
`population-evaluation-v1` default of 10,000 resamples.

## 26.10 Primary precision diagnostic

The Section-25 ±50 BB/100 target was planned for the **fixed five-profile
weighted mean**, using simulation variance within each profile. It is distinct
from the `population-evaluation-v1` nested bootstrap, which additionally
resamples opponent-profile observations and is primarily required to respect
population/robustness hierarchy.

Therefore the final report MUST retain both quantities and MUST NOT substitute
one for the other.

For the frozen Baseline Synthetic weights $w_j$, let the valid final unit-level
hero EV observations in profile $j$ have sample variance $s_j^2$ and count
$n_j$. The fixed-profile standard error is:

$$
SE_{fixed} = \sqrt{\sum_j w_j^2\frac{s_j^2}{n_j}}.
$$

The final precision diagnostic is:

$$
\widehat P \pm z_{0.975} SE_{fixed}.
$$

The final half-width is:

$$
h_{final}=z_{0.975}SE_{fixed}.
$$

and the report records:

```text
target_met = (h_final <= 50 BB/100)
```

This diagnostic is **reporting only**. The frozen sample size is never extended
or stopped based on the observed final width.

If `target_met = false`, that outcome is reported transparently. No additional
seed may be appended under v1.

The separate nested-bootstrap mean CI from `population-evaluation-v1` is also
reported and may be wider because it answers a different uncertainty question.

## 26.11 Precision disclosure

Every final report MUST repeat:

```text
Frozen primary target: ±50 BB/100 at 95% confidence
General project-document example: approximately ±1 BB/100
The v1 experiment intentionally uses materially coarser precision.
```

The report MUST NOT describe the present experiment as having ±1 BB/100
precision and MUST NOT use it to support fine-grained claims that require such
precision.

## 26.12 Scientific failures

A completed failed checkpoint remains part of the frozen campaign evidence.
There are no replacement seeds.

For any hero whose frozen allocation contains one or more failed jobs, v1:

- retains all valid and failed job evidence;
- marks that hero as not failure-free;
- withholds final population estimands/precision diagnostics for that hero;
- does not silently analyze a reduced convenience sample.

The overall report remains hash-verifiable and records the campaign as
`INCOMPLETE_FAILURES` for scientific interpretation.

## 26.13 Final report and exit codes

The final report is written only after all frozen jobs have hash-verified
checkpoints.

It contains:

- protocol, pilot, Gate-6, and execution-binding identities;
- requested/completed/valid/failed unit counts;
- requested and recorded engine-hand counts;
- per-hero population evaluations when failure-free;
- per-hero fixed-profile precision diagnostics when failure-free;
- runtime totals;
- precision disclosure;
- five-profile scope/tail warnings;
- explicit confirmation that pilot observations were excluded;
- explicit confirmation that no confirmatory inference or optional stopping was
  performed;
- canonical report SHA-256.

Exit codes are:

```text
0 = all frozen jobs checkpointed, no scientific failures
1 = all frozen jobs checkpointed, one or more scientific failures
2 = execution/integrity/provenance/configuration/persistence error
```

Exit code 1 does not trigger replacement seeds.

## 26.14 Version-change rule

Any change to sample sizes, final seeds, precision target, primary population,
hero policy versions, opponent policy/gate/equity versions, Gate-6 evidence,
bootstrap algorithm, precision diagnostic, or failure-handling semantics requires
a new versioned protocol/execution contract. Existing v1 artifacts are immutable
scientific evidence.
