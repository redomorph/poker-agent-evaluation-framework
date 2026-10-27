# Public Battle CLI v1

## Normative identities

```text
public_battle_cli_version = public-battle-cli-v1
random_baseline_version  = random-baseline-v1
```

This specification freezes the first public `poker battle` command and the
minimal random baseline needed by the roadmap deliverable. It does not change
poker rules, RNG namespaces, position compensation, Gate-6 calibration,
statistical-unit semantics, persistence, or replay semantics.

## Random baseline

`random-baseline-v1` is a static, non-adaptive agent. At every decision it:

1. receives only the normal `AgentObservation`, legal actions, and `AgentRNG`;
2. performs exactly one `AgentRNG.choice(...)` over the supplied legal-action
   sequence;
3. returns the selected legal action.

The selection is uniform over the supplied legal-action sequence. The agent
performs no equity calculation, opponent modelling, hidden-state inspection,
or additional random draw. An empty legal-action sequence is an agent error.

`random-baseline-v1` is a public measurement baseline. It is distinct from any
special low-variance policy used only by the Phase-0 symmetry gate.

## Command contract

The roadmap example is valid as written:

```bash
poker battle \
  --bot-a random \
  --bot-b tag \
  --seeds 10000
```

The complete v1 interface is:

```bash
poker battle \
  --bot-a random \
  --bot-b tag \
  --seeds 10000 \
  --seed-start 0 \
  --hands 1 \
  --snapshot-root data/gate6/snapshots \
  --gate6-report data/gate6/gate6-report.json \
  --database data/benchmark.sqlite \
  --bootstrap-seed 0 \
  [--report data/reports/random-vs-tag.json]
```

`--bot-a` and `--bot-b` accept `random` plus the five frozen
`static-profiles-v2` profile names.

`--seeds N` means exactly `N` independent master-seed jobs. With
`--seed-start S`, the deterministic seed schedule is:

```text
S, S + 1, ..., S + N - 1
```

`--seed-start` defaults to `0`. `--hands` is the number of paired deals per
seed job and defaults to `1`. Both `--seeds` and `--hands` must be positive,
`--seed-start` must be non-negative, and the requested number of paired-deal
statistical units (`seeds * hands`) must be at least two so a standard
deviation and confidence interval are defined.

Each master seed is executed as one separate `PairedRunner` invocation with
`adaptive=False` and the baseline HU NLHE game configuration. v1 executes jobs
synchronously in seed order. Later match-level parallelisation may change only
runtime scheduling, not seed assignment, RNG, results, or aggregation.

## Agent construction and Gate-6 provenance

`random` constructs `random-baseline-v1` and does not require Gate-6 evidence.

Every static profile constructs the existing `ParametricPolicyAgent` from the
FIT calibration-strength snapshots. When either selected bot is a static
profile, the command MUST load Gate-6 evidence once and refuse static-profile
execution unless:

1. the Gate-6 report is integrity-valid and scientifically PASS;
2. the FIT snapshot reference hash equals the report reference hash;
3. each selected static profile gate-map hash equals the hash validated by
   Gate 6.

No calibration or tuning occurs inside `poker battle`.

## Persistence and failures

Every seed job is persisted through the existing `SQLiteBenchmarkStore`.
`--database` defaults to `data/benchmark.sqlite`. The store may initialise a
new/empty schema but MUST NOT silently migrate an incompatible schema.
Persisted matches remain immutable; an existing deterministic match ID is not
overwritten.

The experiment identity for each seed job contains the battle CLI version,
ordered bot identities, seed, hands-per-seed, and Gate-6 report hash when
applicable.

The frozen benchmark illegal-action policy remains `exclude_run`. Failed units
are retained and reported. `poker battle` MUST NOT replace a failed seed or
append a new seed automatically. The battle is complete only when the number
of valid independent units equals the pre-requested `seeds * hands` target.

## Statistical aggregation

All successfully emitted `PairedResult` objects are aggregated only through
the existing cluster-aware `evaluate_paired_results(...)` entry point.
For v1 static agents the statistical unit is `paired_deal`.

The standard report uses:

```text
confidence_level     = 0.95
bootstrap_resamples  = 10000
bootstrap_seed       = --bootstrap-seed  (default 0)
```

The report distinguishes requested engine simulations from analyzed engine
simulations and reports at least:

```text
Bot A
Bot B
Bot versions
N seed jobs
Seed start / seed count
Hands per seed
Requested paired-deal units
Valid independent units
Requested engine hands
Analyzed engine hands
Statistical unit
Mean paired difference (BB/100)
Normal 95% CI
Bootstrap 95% CI
Effect size d_z
Failed runs
Failure policy
Game configuration hash
Gate-6 report hash when applicable
Evaluation version
Battle CLI version
Report hash
Complete / incomplete status
```

A raw per-hand paired difference is converted to BB/100 only by the existing
statistical evaluator; the battle command does not invent a second EV scaling
path.

## Reproducible report

The text report is always printed to stdout. When `--report PATH` is supplied,
a canonical JSON report is written atomically. `report_hash` is SHA-256 over
the canonical report payload excluding the `report_hash` field itself.

## Exit status

Exit code `0` means all requested independent units were valid, persisted, and
the statistical report completed.

Exit code `1` means configuration, Gate-6 integrity, persistence, execution, or
scientific completeness failure. If execution produced enough valid units for
statistics but some requested units failed, the incomplete report is still
rendered (and written when requested) before exit code `1` is returned.

Argument-parser errors use the normal CLI parser exit behavior.
