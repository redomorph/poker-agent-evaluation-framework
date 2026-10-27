"""Integration tests for the public deterministic export bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from pokerbench.cli.main import build_parser, main
from pokerbench.engine.config import Action, BASELINE_GAME_CONFIG
from pokerbench.persistence import MatchPersistenceMetadata, SQLiteBenchmarkStore
from pokerbench.replay import (
    HandTermination,
    MatchRecord,
    RecordedHand,
    RecordedTimeoutEvent,
)


MASTER_SEED = 123


def _normal_hand(*, config_id: str = "A_BTN") -> RecordedHand:
    return RecordedHand(
        config_id=config_id,
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=0,
        seat1_bot_id=1,
        actions=(Action.FOLD,),
        termination=HandTermination.NORMAL,
        terminal_profit_bb=(-0.5, 0.5),
    )


def _timeout_hand() -> RecordedHand:
    return RecordedHand(
        config_id="B_BTN",
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=1,
        seat1_bot_id=0,
        actions=(),
        termination=HandTermination.TIMEOUT_FORFEIT,
        terminal_profit_bb=(-0.5, 0.5),
        timeout_events=(
            RecordedTimeoutEvent(
                player=0,
                bot_id=1,
                action_index=0,
                decision_index=0,
                elapsed_ns=2_000_000,
                budget_ms=1.0,
                timeout_policy="fold",
                resolution="forfeit",
            ),
        ),
        acting_player_at_termination=0,
    )


def _record(match_id: str) -> MatchRecord:
    return MatchRecord(
        match_id=match_id,
        master_seed=MASTER_SEED,
        game_config=BASELINE_GAME_CONFIG,
        hands=(
            _normal_hand(),
            _timeout_hand(),
        ),
        experiment_id=f"experiment:{match_id}",
    )


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "benchmark.sqlite3"
    store = SQLiteBenchmarkStore(database)
    store.initialize()

    metadata = MatchPersistenceMetadata(
        bot_version="bot-v1",
        opponent_hash="opponent-hash",
        ev_bb_100=1.25,
        ci_low=0.5,
        ci_high=2.0,
        engine_version="engine-v1",
        python_version="3.14",
        rng_version="agent-rng-v1",
        git_commit="abc123",
        population_id="population-v1",
        population_seed=999,
        opponent_parameters={"vpip": 0.22},
        bot_configuration={"budget": 100},
        statistical_method="paired-bootstrap",
        sample_size=5000,
        precision_target={"ci_half_width_bb_100": 1.0},
        experiment_config={"seed": MASTER_SEED, "name": "fixture"},
        poetry_lock_hash="lock-hash",
        hardware_fingerprint="hardware-hash",
        os_identifier="test-os",
    )

    # MatchRecord.experiment_config_hash must agree with the supplied payload.
    canonical = json.dumps(
        metadata.experiment_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    first = _record("match-b")
    first = MatchRecord(
        match_id=first.match_id,
        master_seed=first.master_seed,
        game_config=first.game_config,
        hands=first.hands,
        experiment_id=first.experiment_id,
        experiment_config_hash=config_hash,
    )
    second = _record("match-a")
    second = MatchRecord(
        match_id=second.match_id,
        master_seed=second.master_seed,
        game_config=second.game_config,
        hands=second.hands,
        experiment_id=second.experiment_id,
        experiment_config_hash=config_hash,
    )

    store.save_match(first, metadata=metadata)
    store.save_match(second, metadata=metadata)
    return database


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_parser_exposes_export_contract() -> None:
    args = build_parser().parse_args(
        [
            "export",
            "--database",
            "bench.sqlite",
            "--output",
            "bundle",
            "--match-id",
            "b",
            "--match-id",
            "a",
        ]
    )

    assert args.command == "export"
    assert args.database == Path("bench.sqlite")
    assert args.output == Path("bundle")
    assert args.match_ids == ["b", "a"]


def test_export_all_matches_writes_complete_deterministic_bundle(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    output = tmp_path / "export"

    exit_code = main(
        [
            "export",
            "--database",
            str(database),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "records.jsonl",
        "matches.csv",
        "hands.csv",
        "decisions.csv",
        "timeouts.csv",
    }

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["public_export_cli_version"] == "public-export-cli-v1"
    assert manifest["export_bundle_version"] == "benchmark-export-bundle-v1"
    assert manifest["selected_match_ids"] == ["match-a", "match-b"]
    assert manifest["match_count"] == 2
    assert manifest["hand_count"] == 4
    assert manifest["decision_count"] == 2
    assert manifest["timeout_count"] == 2
    assert len(manifest["bundle_hash"]) == 64

    match_rows = _csv_rows(output / "matches.csv")
    assert [row["match_id"] for row in match_rows] == ["match-a", "match-b"]
    assert match_rows[0]["experiment_config_json"] == '{"name":"fixture","seed":123}'
    assert match_rows[0]["opponent_parameters_json"] == '{"vpip":0.22}'

    hand_rows = _csv_rows(output / "hands.csv")
    assert [(row["match_id"], row["hand_ordinal"]) for row in hand_rows] == [
        ("match-a", "0"),
        ("match-a", "1"),
        ("match-b", "0"),
        ("match-b", "1"),
    ]
    assert hand_rows[1]["termination"] == "timeout_forfeit"
    assert hand_rows[1]["timeout_count"] == "1"

    decision_rows = _csv_rows(output / "decisions.csv")
    assert len(decision_rows) == 2
    assert all(row["chosen_action"] == "fold" for row in decision_rows)

    timeout_rows = _csv_rows(output / "timeouts.csv")
    assert len(timeout_rows) == 2
    assert all(row["bot_id"] == "1" for row in timeout_rows)
    assert all(row["resolution"] == "forfeit" for row in timeout_rows)


def test_selected_match_ids_are_sorted_and_missing_match_aborts_atomically(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    selected = tmp_path / "selected"

    ok = main(
        [
            "export",
            "--database",
            str(database),
            "--output",
            str(selected),
            "--match-id",
            "match-b",
            "--match-id",
            "match-a",
        ]
    )
    assert ok == 0
    manifest = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_match_ids"] == ["match-a", "match-b"]

    failed_output = tmp_path / "failed"
    failed = main(
        [
            "export",
            "--database",
            str(database),
            "--output",
            str(failed_output),
            "--match-id",
            "missing",
        ]
    )
    assert failed == 1
    assert not failed_output.exists()


def test_same_database_exports_byte_identical_bundles(tmp_path: Path) -> None:
    database = _database(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert main(["export", "--database", str(database), "--output", str(first)]) == 0
    assert main(["export", "--database", str(database), "--output", str(second)]) == 0

    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_existing_destination_and_duplicate_selection_are_rejected(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    assert main(["export", "--database", str(database), "--output", str(output)]) == 1

    duplicate = tmp_path / "duplicate"
    assert main(
        [
            "export",
            "--database",
            str(database),
            "--output",
            str(duplicate),
            "--match-id",
            "match-a",
            "--match-id",
            "match-a",
        ]
    ) == 1
    assert not duplicate.exists()


def test_integrity_failure_does_not_commit_destination(tmp_path: Path) -> None:
    database = _database(tmp_path)
    output = tmp_path / "tampered-export"

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE matches SET record_json = '{}' WHERE id = 'match-a'"
        )
        connection.commit()

    assert main(["export", "--database", str(database), "--output", str(output)]) == 1
    assert not output.exists()


def test_manifest_hashes_match_exported_files_and_bundle_identity(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    output = tmp_path / "export"
    assert main(["export", "--database", str(database), "--output", str(output)]) == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["files"].items():
        actual_hash = hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert actual_hash == expected_hash

    identity = dict(manifest)
    stored_bundle_hash = identity.pop("bundle_hash")
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == stored_bundle_hash

    records = (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["match_id"] for line in records] == ["match-a", "match-b"]


def test_empty_database_is_rejected_without_committing_output(
    tmp_path: Path,
) -> None:
    database = tmp_path / "empty.sqlite3"
    SQLiteBenchmarkStore(database).initialize()
    output = tmp_path / "empty-export"

    assert main(["export", "--database", str(database), "--output", str(output)]) == 1
    assert not output.exists()
