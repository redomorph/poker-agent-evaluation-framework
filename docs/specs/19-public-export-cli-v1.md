# Public Export CLI v1

## Status

This document freezes the first public export command required by the software deliverables.

```text
public_export_cli_version = public-export-cli-v1
export_bundle_version     = benchmark-export-bundle-v1
```

This specification is normative for `poker export` v1.

## 1. Purpose

`poker export` is a read-only transformation of already-persisted benchmark data. It MUST NOT run agents, replay a benchmark for statistical recomputation, tune a policy, replace failed seeds, or modify the source database.

The source of truth for v1 is a valid `SQLiteBenchmarkStore` database. Every selected match is loaded through the existing integrity-checked persistence API before it is exported.

## 2. Command

```bash
poker export \
  --database data/benchmark.sqlite \
  --output data/exports/benchmark-v1
```

An optional `--match-id` argument may be supplied multiple times:

```bash
poker export \
  --database data/benchmark.sqlite \
  --output data/exports/selected \
  --match-id match-a \
  --match-id match-b
```

If no match IDs are supplied, all persisted matches are exported.

All selected match IDs are canonicalized into ascending lexical order. Duplicate explicitly requested match IDs are rejected. Every explicitly requested match MUST exist; selection failure aborts the export before the destination is committed.

The output path is a directory and MUST NOT already exist. Existing evidence is never silently overwritten.

An export with zero selected matches is rejected.

## 3. Bundle contents

A successful export contains exactly these files:

```text
manifest.json
records.jsonl
matches.csv
hands.csv
decisions.csv
timeouts.csv
```

### 3.1 `records.jsonl`

One canonical JSON line per selected `MatchRecord`, ordered by `match_id`. Each line is exactly the deterministic `MatchRecord.to_dict()` payload serialized with sorted keys and compact separators.

### 3.2 `matches.csv`

One row per selected match, ordered by `match_id`. It contains match identity, seed, game/configuration hashes, record hash, game parameters, and the normalized optional scientific metadata already persisted by the SQLite store. Structured metadata values are stored as canonical JSON strings.

### 3.3 `hands.csv`

One row per persisted hand, ordered by `(match_id, hand_ordinal)`. It contains stable seat/bot identity, hand/configuration identity, deterministic cards/runout, submitted agent actions, termination/failure information, terminal profit, timeout count, and a canonical hand-record hash.

### 3.4 `decisions.csv`

One row per persisted agent-returned decision, ordered by `(match_id, hand_ordinal, action_index)`. Decision state is reconstructed through the existing validated replay/inspection path and includes actor seat/bot identity, street, position, pot, stacks, current bets, full and visible board, actor and opponent cards, prior action history, legal actions, and chosen action.

Framework-applied timeout resolution actions are not misrepresented as agent-returned decisions; timeout events are exported separately.

### 3.5 `timeouts.csv`

One row per persisted timeout event, ordered by `(match_id, hand_ordinal, timeout_ordinal)`. It preserves seat identity, stable bot identity, action/decision index, elapsed time, budget, timeout policy/version, persisted resolution, and applied action when present.

## 4. Deterministic serialization

CSV files use UTF-8, a single header row, RFC-4180-compatible quoting through Python's standard `csv` module, and `\n` line endings on every platform. `None` is serialized as an empty CSV field. Structured values inside CSV cells use canonical JSON.

JSON uses UTF-8 and:

```python
json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

`manifest.json` ends with one newline. Every `records.jsonl` record ends with one newline.

Two exports of the same selected persisted source data under the same software contract MUST be byte-identical, independent of destination path.

## 5. Manifest and integrity

`manifest.json` records at minimum:

```text
public_export_cli_version
export_bundle_version
sqlite_schema_version
sqlite_schema_migration_id
selected_match_ids
match_count
hand_count
decision_count
timeout_count
files -> SHA-256 for each non-manifest file
bundle_hash
```

`bundle_hash` is SHA-256 over the canonical JSON representation of the manifest identity/count/hash payload before the `bundle_hash` field is inserted. The manifest does not attempt to contain a recursive hash of itself.

Every selected `MatchRecord` and normalized hand row MUST pass the existing SQLite integrity checks before any final bundle is committed.

## 6. Failure and atomicity

The source database is opened only for reads during export. Unknown, uninitialized, unsupported, or integrity-failing databases cause a non-zero command result.

The bundle is built in a temporary sibling directory and atomically committed only after all selected records have loaded, replay-derived decision rows have been reconstructed, all files have been written, and all file hashes have been calculated. A failed export MUST NOT leave the requested destination as a partially valid bundle.

## 7. Parquet boundary

The project architecture permits SQLite or Parquet benchmark persistence. `public-export-cli-v1` does not define a Parquet serialization format because the current frozen Poetry environment has no Parquet serialization dependency. Adding such a dependency changes the reproducibility surface (`pyproject.toml` / `poetry.lock`) and therefore requires a separately versioned export/storage contract.

This restriction does not prohibit a future `public-export-cli-v2` from adding Parquet once the serializer, dependency version, schema mapping, and deterministic metadata rules are frozen.
