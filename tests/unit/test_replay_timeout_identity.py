"""Regression tests for timeout seat/bot identity persistence."""

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


MASTER_SEED = 123


def _timeout_record() -> MatchRecord:
    timeout = RecordedTimeoutEvent(
        player=0,
        bot_id=1,
        action_index=0,
        decision_index=0,
        elapsed_ns=2_000_000,
        budget_ms=1.0,
        timeout_policy="fold",
        resolution="fold",
        applied_action=Action.FOLD,
    )

    hand = RecordedHand(
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
            timeout,
        ),
        acting_player_at_termination=0,
    )

    return MatchRecord(
        match_id="timeout-identity-match",
        master_seed=MASTER_SEED,
        game_config=BASELINE_GAME_CONFIG,
        hands=(
            hand,
        ),
    )


def test_timeout_event_allows_player_and_bot_id_to_differ() -> None:
    event = RecordedTimeoutEvent(
        player=0,
        bot_id=1,
        action_index=2,
        decision_index=7,
        elapsed_ns=2_000_000,
        budget_ms=1.0,
        timeout_policy="fold",
        resolution="fold",
        applied_action=Action.FOLD,
    )

    assert event.player == 0
    assert event.bot_id == 1
    assert event.action_index == 2
    assert event.decision_index == 7


def test_timeout_event_falls_back_for_legacy_single_identity() -> None:
    event = RecordedTimeoutEvent(
        player=1,
        action_index=4,
        elapsed_ns=1_000_000,
        budget_ms=1.0,
        timeout_policy="fold",
    )

    assert event.player == 1
    assert event.bot_id == 1
    assert event.action_index == 4
    assert event.decision_index == 4


def test_timeout_event_round_trip_preserves_distinct_identities() -> None:
    original = RecordedTimeoutEvent(
        player=0,
        bot_id=1,
        action_index=3,
        decision_index=9,
        elapsed_ns=2_000_000,
        budget_ms=1.0,
        timeout_policy="fold",
        resolution="fold",
        applied_action=Action.FOLD,
    )

    restored = (
        RecordedTimeoutEvent.from_dict(
            original.to_dict()
        )
    )

    assert restored == original
    assert restored.player == 0
    assert restored.bot_id == 1
    assert restored.action_index == 3
    assert restored.decision_index == 9


def test_archive_round_trip_preserves_mirrored_timeout_identity(
    tmp_path: Path,
) -> None:
    archive = ReplayArchive(
        tmp_path
    )

    original = (
        _timeout_record()
    )

    archive.save(
        original
    )

    restored = archive.load(
        original.match_id
    )

    event = (
        restored
        .hands[0]
        .timeout_events[0]
    )

    assert event.player == 0
    assert event.bot_id == 1

    replayed = archive.replay(
        original.match_id
    )

    assert len(replayed) == 1

    assert replayed[0].actions == (
        Action.FOLD,
    )

    assert (
        replayed[0]
        .final_state
        .is_terminal
    )


def test_timeout_identity_validation_is_independent() -> None:
    with pytest.raises(
        ValueError,
        match="player must be 0 or 1",
    ):
        RecordedTimeoutEvent(
            player=2,
            bot_id=1,
            timeout_policy="fold",
        )

    with pytest.raises(
        ValueError,
        match="bot_id must be 0 or 1",
    ):
        RecordedTimeoutEvent(
            player=0,
            bot_id=2,
            timeout_policy="fold",
        )