# Poker Agent Evaluation Framework

A Python research framework for reproducible evaluation of poker agents in **heads-up No-Limit Texas Hold'em (HU NLHE)**. It focuses on controlled opponent populations, deterministic experiments, statistical evaluation and replayability rather than maximising a single bot's win rate.

## Scope

- Heads-up NLHE with a frozen 100-BB cash-game baseline and discrete betting abstraction.
- Deterministic deal scheduling, paired positional comparisons and recorded experiment metadata.
- Random, static-profile and hand-strength baselines; experimental adaptive-opponent modelling components.
- Validation gates, SQLite persistence, replay, export and reproducible reporting.

The framework is a scientific testbed. Benchmark conclusions apply to their defined game abstraction and experimental design, not automatically to unrestricted real-money poker.

## Requirements

- **Python 3.14 or newer** (as declared in `pyproject.toml`).
- [Poetry](https://python-poetry.org/) for the lockfile-based development setup.

## Installation

From the repository root:

```bash
poetry install
poetry run poker --help
```

The CLI is also available as `pokerbench`. For command-specific options, use `poetry run poker <command> --help`.

## Quick start

Run a small deterministic battle between two random agents:

```bash
poetry run poker battle --bot-a random --bot-b random --seeds 2 --hands 10 --database data/benchmark.sqlite
```

Run a paired match between pre-calibrated static profiles (the repository includes the required Gate-6 evidence and snapshots):

```bash
poetry run poker run --bot-a nit --bot-b tag --seed 1234 --hands 100 --database data/benchmark.sqlite
```

The CLI exposes `run`, `battle`, `inspect`, `replay`, `export` and `plot`. See each command's `--help` for required arguments and output formats.

## Tests

```bash
poetry run pytest
```

The test suite includes unit, integration, regression and scientific validation tests. Regenerated benchmark results should be interpreted together with their seeds, configuration hashes, statistical units and validation evidence.

## Documentation and evidence

- [Documentation overview](docs/README.md)
- [Versioned scientific and technical specifications](docs/specs/README.md)
- [Validation and experiment reports](docs/reports/README.md)
- [Archived scientific project document](docs/specs/archive/README.md)

Reference calibration snapshots and selected experiment reports are stored under `data/`. Documentation distinguishes frozen design decisions, pilot evidence and confirmatory analysis boundaries; the presence of a report alone is not a claim of general poker-agent superiority.

## License

[MIT](LICENSE) © 2026 redomorph.
