"""Compatibility regression tests for replay archive updates."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.replay import (
    HandTermination,
    MatchRecord,
    RecordedHand,
    RecordedTimeoutEvent,
    ReplayArchive,
)


MASTER_SEED = 42


def _record() -> MatchRecord:
    return MatchRecord(
        match_id="compat-match",
        master_seed=MASTER_SEED,
        game_config=BASELINE_GAME_CONFIG,
        hands=(
            RecordedHand(
                config_id="A_BTN",
                master_seed=MASTER_SEED,
                hand_index=0,
                dealer=0,
                seat0_bot_id=0,
                seat1_bot_id=1,
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
            ),
        ),
        experiment_id="experiment-1",
        experiment_config_hash="config-hash",
    )


def test_timeout_event_accepts_runner_nanosecond_fields() -> None:
    event = RecordedTimeoutEvent(
        player=0,
        action_index=7,
        elapsed_ns=12_500_000,
        budget_ms=10.0,
        timeout_policy="fold",
        resolution="fold",
    )

    assert event.player == 0
    assert event.action_index == 7
    assert event.elapsed_ns == 12_500_000
    assert event.elapsed_ms == 12.5
    assert event.budget_ms == 10.0
    assert event.resolution == "fold"


def test_timeout_event_elapsed_ms_remains_supported() -> None:
    event = RecordedTimeoutEvent(
        player=0,
        action_index=1,
        elapsed_ms=2.5,
        budget_ms=2.0,
        timeout_policy="fold",
        resolution="fold",
    )

    assert event.elapsed_ns == 2_500_000
    assert event.elapsed_ms == 2.5


def test_timeout_event_rejects_conflicting_elapsed_units() -> None:
    with pytest.raises(
        ValueError,
        match="elapsed_ns and elapsed_ms disagree",
    ):
        RecordedTimeoutEvent(
            player=0,
            elapsed_ns=1_000_000,
            elapsed_ms=2.0,
            timeout_policy="fold",
        )


def test_timeout_event_round_trip_preserves_resolution() -> None:
    original = RecordedTimeoutEvent(
        player=1,
        action_index=3,
        elapsed_ns=8_000_000,
        budget_ms=5.0,
        timeout_policy="fold",
        resolution="fold",
        applied_action=Action.FOLD,
    )

    restored = RecordedTimeoutEvent.from_dict(
        original.to_dict()
    )

    assert restored == original


def test_archive_replay_delegates_to_match_record_replay_all(
    tmp_path: Path,
) -> None:
    archive = ReplayArchive(
        tmp_path
    )

    record = _record()

    archive.save(
        record
    )

    replayed = archive.replay(
        record.match_id
    )

    assert len(replayed) == 1

    assert replayed[0].actions == (
        Action.FOLD,
    )

    assert replayed[0].final_profit_bb == (
        -0.5,
        0.5,
    )


def test_archive_replay_missing_match_preserves_file_not_found(
    tmp_path: Path,
) -> None:
    archive = ReplayArchive(
        tmp_path
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        archive.replay(
            "missing"
        )