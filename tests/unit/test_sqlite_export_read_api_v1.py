"""Read-only SQLite APIs required by public-export-cli-v1."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pokerbench.engine.config import Action, BASELINE_GAME_CONFIG
from pokerbench.persistence import MatchPersistenceMetadata, SQLiteBenchmarkStore
from pokerbench.replay import HandTermination, MatchRecord, RecordedHand


def _record(match_id: str) -> MatchRecord:
    return MatchRecord(
        match_id=match_id,
        master_seed=123,
        game_config=BASELINE_GAME_CONFIG,
        hands=(
            RecordedHand(
                config_id="A_BTN",
                master_seed=123,
                hand_index=0,
                dealer=0,
                seat0_bot_id=0,
                seat1_bot_id=1,
                actions=(Action.FOLD,),
                termination=HandTermination.NORMAL,
                terminal_profit_bb=(-0.5, 0.5),
            ),
        ),
    )


def test_list_match_ids_is_lexically_sorted(tmp_path: Path) -> None:
    store = SQLiteBenchmarkStore(tmp_path / "benchmark.sqlite3")
    store.initialize()
    store.save_match(_record("z-match"))
    store.save_match(_record("a-match"))

    assert store.list_match_ids() == ("a-match", "z-match")


def test_load_match_metadata_round_trips_structured_values(tmp_path: Path) -> None:
    store = SQLiteBenchmarkStore(tmp_path / "benchmark.sqlite3")
    store.initialize()
    metadata = MatchPersistenceMetadata(
        bot_version="bot-v1",
        opponent_parameters={"vpip": 0.22},
        bot_configuration={"budget": 100},
        precision_target={"width": 1.0},
    )
    record = _record("match")
    store.save_match(record, metadata=metadata)

    restored = store.load_match_metadata("match")

    assert restored.bot_version == "bot-v1"
    assert restored.opponent_parameters == {"vpip": 0.22}
    assert restored.bot_configuration == {"budget": 100}
    assert restored.precision_target == {"width": 1.0}
    assert restored.experiment_config is None


def test_load_match_metadata_rejects_tampered_structured_json(
    tmp_path: Path,
) -> None:
    from pokerbench.persistence import PersistenceIntegrityError

    store = SQLiteBenchmarkStore(tmp_path / "benchmark.sqlite3")
    store.initialize()
    store.save_match(
        _record("match"),
        metadata=MatchPersistenceMetadata(
            opponent_parameters={"vpip": 0.22},
        ),
    )

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE matches SET opponent_parameters_json = ? WHERE id = ?",
            ('{ "vpip": 0.22 }', "match"),
        )
        connection.commit()

    with pytest.raises(
        PersistenceIntegrityError,
        match="not canonical JSON",
    ):
        store.load_match_metadata("match")
