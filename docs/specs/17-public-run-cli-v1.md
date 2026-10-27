# Public Run CLI v1

## Normative identity

```text
public_run_cli_version = public-run-cli-v1
```

This specification freezes the first public `poker run` command without changing poker, RNG, calibration, statistical, persistence, or replay semantics.

## Command contract

```bash
poker run \
  --bot-a tag \
  --bot-b lag \
  --seed 12345 \
  --hands 1000 \
  --snapshot-root data/gate6/snapshots \
  --gate6-report data/gate6/gate6-report.json \
  --database data/benchmark.sqlite
```

`--bot-a` and `--bot-b` select one of the five frozen `static-profiles-v2` profiles: `nit`, `tag`, `lag`, `calling_station`, or `maniac`.

`poker run` executes exactly one `PairedRunner` invocation with `adaptive=False`, the baseline HU NLHE game configuration, the requested deterministic master seed, and the requested number of paired deals.

The command MUST construct each `ParametricPolicyAgent` from the FIT calibration-strength snapshots. It MUST load the persisted Gate-6 report and refuse execution unless:

1. the Gate-6 report is integrity-valid and scientifically PASS;
2. the FIT snapshot reference hash equals the report reference hash;
3. the selected profile gate-map hash equals the gate-map hash validated by Gate 6.

No calibration or tuning occurs inside `poker run`.

## Persistence

Every successful run is persisted through `SQLiteBenchmarkStore`. The command may initialize an empty/new SQLite database using the already-versioned schema initializer. It MUST NOT silently migrate an incompatible database.

`--match-id` is optional. If omitted, the runner's deterministic match-id derivation is used. The run metadata MUST include an `experiment_id` containing the CLI version, both ordered profile identities, and the Gate-6 report hash so distinct profile pairings cannot collide merely because they use the same agent class, seed, and hand count.

## Output and exit status

The text result reports at least match id, bot identities, seed, requested paired deals, valid statistical units, statistical-unit type, failed-run count, database path, and Gate-6 report hash.

The command does not relabel a raw paired-difference mean as `BB/100`. Formal EV100 reporting remains the responsibility of the statistical evaluation/reporting layer.

Exit code `0` means the command executed and persisted the run. Exit code `1` means configuration, calibration-integrity, persistence, or execution failure. Argument-parser errors use the normal CLI parser exit behavior.
