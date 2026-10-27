"""Regression tests for timeout-forfeit replay reconstruction."""

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
    ReplayIntegrityError,
)


MASTER_SEED = 123


def _timeout_forfeit_hand(
    *,
    config_id: str,
    seat0_bot_id: int,
    seat1_bot_id: int,
    player: int,
    bot_id: int,
) -> RecordedHand:
    return RecordedHand(
        config_id=config_id,
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=seat0_bot_id,
        seat1_bot_id=seat1_bot_id,
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
                player=player,
                bot_id=bot_id,
                action_index=0,
                decision_index=0,
                elapsed_ns=2_000_000,
                budget_ms=1.0,
                timeout_policy="fold",
                resolution="forfeit",
                applied_action=None,
            ),
        ),
        acting_player_at_termination=player,
    )


def test_fold_policy_forfeit_resolution_replays_as_fold() -> None:
    hand = _timeout_forfeit_hand(
        config_id="A_BTN",
        seat0_bot_id=0,
        seat1_bot_id=1,
        player=0,
        bot_id=0,
    )

    replayed = hand.replay(
        BASELINE_GAME_CONFIG
    )

    assert replayed.actions == (
        Action.FOLD,
    )

    assert (
        replayed.final_state
        .is_terminal
    )

    assert (
        replayed.final_profit_bb
        == (
            -0.5,
            0.5,
        )
    )


def test_fold_policy_forfeit_replay_preserves_mirrored_bot_identity() -> None:
    hand = _timeout_forfeit_hand(
        config_id="B_BTN",
        seat0_bot_id=1,
        seat1_bot_id=0,
        player=0,
        bot_id=1,
    )

    replayed = hand.replay(
        BASELINE_GAME_CONFIG
    )

    assert replayed.actions == (
        Action.FOLD,
    )

    event = hand.timeout_events[0]

    assert event.player == 0
    assert event.bot_id == 1
    assert event.resolution == "forfeit"
    assert event.applied_action is None


def test_archive_replays_real_runner_timeout_shape(
    tmp_path: Path,
) -> None:
    record = MatchRecord(
        match_id="timeout-match",
        master_seed=MASTER_SEED,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        hands=(
            _timeout_forfeit_hand(
                config_id="A_BTN",
                seat0_bot_id=0,
                seat1_bot_id=1,
                player=0,
                bot_id=0,
            ),
            _timeout_forfeit_hand(
                config_id="B_BTN",
                seat0_bot_id=1,
                seat1_bot_id=0,
                player=0,
                bot_id=1,
            ),
        ),
    )

    archive = ReplayArchive(
        tmp_path
    )

    archive.save(
        record
    )

    replayed = archive.replay(
        "timeout-match"
    )

    assert len(replayed) == 2

    assert all(
        result.actions
        == (
            Action.FOLD,
        )
        for result
        in replayed
    )

    assert all(
        result.final_state.is_terminal
        for result
        in replayed
    )


def test_explicit_applied_action_has_priority_over_resolution() -> None:
    event = RecordedTimeoutEvent(
        player=0,
        bot_id=0,
        action_index=0,
        decision_index=0,
        elapsed_ns=2_000_000,
        budget_ms=1.0,
        timeout_policy="fold",
        resolution="forfeit",
        applied_action=Action.FOLD,
    )

    hand = RecordedHand(
        config_id="A_BTN",
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=0,
        seat1_bot_id=1,
        actions=(),
        termination=(
            HandTermination.TIMEOUT_FORFEIT
        ),
        terminal_profit_bb=(
            -0.5,
            0.5,
        ),
        timeout_events=(
            event,
        ),
        acting_player_at_termination=0,
    )

    assert hand.replay(
        BASELINE_GAME_CONFIG
    ).actions == (
        Action.FOLD,
    )


def test_unknown_timeout_outcome_without_applied_action_is_rejected() -> None:
    event = RecordedTimeoutEvent(
        player=0,
        bot_id=0,
        action_index=0,
        decision_index=0,
        elapsed_ns=2_000_000,
        budget_ms=1.0,
        timeout_policy="custom-policy",
        resolution="forfeit",
        applied_action=None,
    )

    hand = RecordedHand(
        config_id="A_BTN",
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=0,
        seat1_bot_id=1,
        actions=(),
        termination=(
            HandTermination.TIMEOUT_FORFEIT
        ),
        terminal_profit_bb=(
            -0.5,
            0.5,
        ),
        timeout_events=(
            event,
        ),
        acting_player_at_termination=0,
    )

    with pytest.raises(
        ReplayIntegrityError,
        match="does not contain a replayable engine action",
    ):
        hand.replay(
            BASELINE_GAME_CONFIG
        )