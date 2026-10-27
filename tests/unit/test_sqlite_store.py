"""Unit tests for SQLiteBenchmarkStore."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.persistence import (
    MatchAlreadyExistsError,
    MatchNotFoundError,
    MatchPersistenceMetadata,
    PersistenceIntegrityError,
    SQLiteBenchmarkStore,
)
from pokerbench.replay import (
    HandTermination,
    MatchRecord,
    RecordedHand,
    RecordedTimeoutEvent,
)


MASTER_SEED = 123


def _normal_hand(
    *,
    config_id: str,
    seat0_bot_id: int,
    seat1_bot_id: int,
) -> RecordedHand:
    return RecordedHand(
        config_id=config_id,
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=seat0_bot_id,
        seat1_bot_id=seat1_bot_id,
        actions=(
            Action.FOLD,
        ),
        termination=(
            HandTermination.NORMAL
        ),
        terminal_profit_bb=(
            -0.5,
            0.5,
        ),
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
        termination=(
            HandTermination.TIMEOUT_FORFEIT
        ),
        terminal_profit_bb=(
            -0.5,
            0.5,
        ),
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


def _record() -> MatchRecord:
    return MatchRecord(
        match_id="match-123",
        master_seed=MASTER_SEED,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        hands=(
            _normal_hand(
                config_id="A_BTN",
                seat0_bot_id=0,
                seat1_bot_id=1,
            ),
            _timeout_hand(),
        ),
        experiment_id="experiment-1",
    )


def _store(
    tmp_path: Path,
) -> SQLiteBenchmarkStore:
    store = SQLiteBenchmarkStore(
        tmp_path
        / "benchmark.sqlite3"
    )
    store.initialize()
    return store


def test_save_and_load_round_trip_exact_match_record(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )
    record = _record()

    store.save_match(
        record
    )

    restored = store.load_match(
        record.match_id
    )

    assert restored == record
    assert (
        restored.record_hash
        == record.record_hash
    )

    replayed = (
        restored.replay_all()
    )

    assert len(replayed) == 2


def test_save_persists_normalized_match_and_hand_rows(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )
    record = _record()

    store.save_match(
        record
    )

    with sqlite3.connect(
        store.path
    ) as connection:
        match_row = (
            connection.execute(
                """
                SELECT
                    seed,
                    hands_played,
                    game_config_hash,
                    abstraction_version
                FROM matches
                WHERE id = ?
                """,
                (
                    record.match_id,
                ),
            )
            .fetchone()
        )

        hand_count = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM hands
                WHERE match_id = ?
                """,
                (
                    record.match_id,
                ),
            )
            .fetchone()[0]
        )

    assert match_row == (
        MASTER_SEED,
        2,
        record.game_config_hash,
        record.game_config
        .abstraction_version,
    )

    assert hand_count == 2


def test_hero_cards_follow_stable_bot_zero_not_seat_zero(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )
    record = _record()

    store.save_match(
        record
    )

    replay = (
        record.hands[1]
        .replay(
            record.game_config
        )
    )

    expected_hero_cards = list(
        replay.initial_state
        .hole_cards[1]
    )

    expected_villain_cards = list(
        replay.initial_state
        .hole_cards[0]
    )

    with sqlite3.connect(
        store.path
    ) as connection:
        row = connection.execute(
            """
            SELECT
                hero_cards,
                villain_cards
            FROM hands
            WHERE
                match_id = ?
                AND hand_ordinal = 1
            """,
            (
                record.match_id,
            ),
        ).fetchone()

    assert json.loads(
        row[0]
    ) == expected_hero_cards

    assert json.loads(
        row[1]
    ) == expected_villain_cards


def test_timeout_actions_json_contains_framework_applied_fold(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )
    record = _record()

    store.save_match(
        record
    )

    with sqlite3.connect(
        store.path
    ) as connection:
        actions_json = (
            connection.execute(
                """
                SELECT actions_json
                FROM hands
                WHERE
                    match_id = ?
                    AND hand_ordinal = 1
                """,
                (
                    record.match_id,
                ),
            )
            .fetchone()[0]
        )

    assert json.loads(
        actions_json
    ) == [
        "fold"
    ]


def test_match_id_is_immutable_and_not_overwritten(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )
    record = _record()

    store.save_match(
        record
    )

    with pytest.raises(
        MatchAlreadyExistsError,
        match=record.match_id,
    ):
        store.save_match(
            record
        )


def test_missing_match_raises_specific_error(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )

    with pytest.raises(
        MatchNotFoundError,
        match="missing",
    ):
        store.load_match(
            "missing"
        )


def test_tampered_match_payload_is_rejected(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )
    record = _record()

    store.save_match(
        record
    )

    with sqlite3.connect(
        store.path
    ) as connection:
        connection.execute(
            """
            UPDATE matches
            SET record_json = ?
            WHERE id = ?
            """,
            (
                "{}",
                record.match_id,
            ),
        )
        connection.commit()

    with pytest.raises(
        PersistenceIntegrityError,
        match="record_hash mismatch",
    ):
        store.load_match(
            record.match_id
        )


def test_tampered_normalized_hand_row_is_rejected(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )
    record = _record()

    store.save_match(
        record
    )

    with sqlite3.connect(
        store.path
    ) as connection:
        connection.execute(
            """
            UPDATE hands
            SET hand_record_json = ?
            WHERE
                match_id = ?
                AND hand_ordinal = 0
            """,
            (
                "{}",
                record.match_id,
            ),
        )
        connection.commit()

    with pytest.raises(
        PersistenceIntegrityError,
        match="hand record hash mismatch",
    ):
        store.load_match(
            record.match_id
        )


def test_optional_scientific_metadata_is_persisted(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )
    record = _record()

    metadata = MatchPersistenceMetadata(
        bot_version="bot-v1",
        opponent_hash="opp-hash",
        ev_bb_100=1.25,
        ci_low=0.50,
        ci_high=2.00,
        engine_version="engine-v1",
        python_version="3.14",
        rng_version="rng-v1",
        git_commit="abc123",
        population_id="population-v1",
        population_seed=999,
        opponent_parameters={
            "vpip": 0.22,
        },
        bot_configuration={
            "budget": 100,
        },
        statistical_method=(
            "paired-bootstrap"
        ),
        sample_size=5000,
        precision_target={
            "ci_half_width_bb_100": 1.0,
        },
        poetry_lock_hash="lock-hash",
        hardware_fingerprint="hardware-hash",
        os_identifier="test-os",
    )

    store.save_match(
        record,
        metadata=metadata,
    )

    with sqlite3.connect(
        store.path
    ) as connection:
        row = connection.execute(
            """
            SELECT
                bot_version,
                opponent_hash,
                ev_bb_100,
                statistical_method,
                sample_size,
                opponent_parameters_json
            FROM matches
            WHERE id = ?
            """,
            (
                record.match_id,
            ),
        ).fetchone()

    assert row[:5] == (
        "bot-v1",
        "opp-hash",
        1.25,
        "paired-bootstrap",
        5000,
    )

    assert json.loads(
        row[5]
    ) == {
        "vpip": 0.22,
    }


def test_experiment_config_hash_is_verified_when_payload_is_supplied(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path
    )

    record = MatchRecord(
        match_id="hash-mismatch",
        master_seed=MASTER_SEED,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        hands=(
            _normal_hand(
                config_id="A_BTN",
                seat0_bot_id=0,
                seat1_bot_id=1,
            ),
        ),
        experiment_config_hash=(
            "not-the-real-hash"
        ),
    )

    metadata = MatchPersistenceMetadata(
        experiment_config={
            "seed": MASTER_SEED,
        }
    )

    with pytest.raises(
        PersistenceIntegrityError,
        match="Experiment configuration hash",
    ):
        store.save_match(
            record,
            metadata=metadata,
        )
