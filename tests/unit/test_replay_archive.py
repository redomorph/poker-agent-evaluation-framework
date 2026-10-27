"""Unit tests for durable match replay archives."""

from __future__ import annotations

import json
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
    RecordedReplayResult,
    RecordedTimeoutEvent,
    ReplayArchive,
    ReplayIntegrityError,
    load_match_record,
    save_match_record,
)


MASTER_SEED = 42


def _normal_fold_hand(
    *,
    hand_index: int = 0,
) -> RecordedHand:
    return RecordedHand(
        config_id="A_BTN",
        master_seed=MASTER_SEED,
        hand_index=hand_index,
        dealer=0,
        seat0_bot_id=0,
        seat1_bot_id=1,
        actions=(
            Action.FOLD,
        ),
        termination=HandTermination.NORMAL,
        terminal_profit_bb=(
            -0.5,
            0.5,
        ),
    )


def _match_record() -> MatchRecord:
    return MatchRecord(
        match_id="match-42",
        experiment_id="experiment-1",
        experiment_config_hash="experiment-config-hash",
        master_seed=MASTER_SEED,
        game_config=BASELINE_GAME_CONFIG,
        hands=(
            _normal_fold_hand(),
        ),
    )


def test_match_record_retains_experiment_metadata() -> None:
    record = _match_record()

    assert record.experiment_id == "experiment-1"

    assert (
        record.experiment_config_hash
        == "experiment-config-hash"
    )


def test_match_record_replay_all_returns_recorded_results() -> None:
    record = _match_record()

    results = record.replay_all()

    assert len(results) == 1

    assert isinstance(
        results[0],
        RecordedReplayResult,
    )


def test_recorded_replay_result_exposes_actions() -> None:
    record = _match_record()

    replay = record.replay_all()[0]

    assert replay.actions == (
        Action.FOLD,
    )


def test_recorded_replay_result_exposes_final_profit_bb() -> None:
    record = _match_record()

    replay = record.replay_all()[0]

    assert replay.final_profit_bb == (
        -0.5,
        0.5,
    )


def test_timeout_event_accepts_runner_facing_names() -> None:
    event = RecordedTimeoutEvent(
        player=0,
        action_index=3,
        elapsed_ms=12.5,
        budget_ms=10.0,
        timeout_policy="fold",
        applied_action=Action.FOLD,
    )

    assert event.player == 0
    assert event.action_index == 3
    assert event.elapsed_ms == 12.5
    assert event.budget_ms == 10.0
    assert event.timeout_policy == "fold"
    assert event.applied_action is Action.FOLD


def test_timeout_event_exposes_compatibility_aliases() -> None:
    event = RecordedTimeoutEvent(
        player=1,
        action_index=4,
        elapsed_ms=3.0,
        budget_ms=2.0,
        timeout_policy="fold",
        applied_action=Action.FOLD,
    )

    assert event.bot_id == 1
    assert event.decision_index == 4
    assert event.time_budget_ms == 2.0
    assert event.policy == "fold"
    assert event.action is Action.FOLD


def test_timeout_event_accepts_compatibility_constructor_names() -> None:
    event = RecordedTimeoutEvent(
        bot_id=1,
        decision_index=5,
        elapsed_ms=3.5,
        time_budget_ms=3.0,
        policy="fold",
        action=Action.FOLD,
    )

    assert event.player == 1
    assert event.action_index == 5
    assert event.budget_ms == 3.0
    assert event.timeout_policy == "fold"
    assert event.applied_action is Action.FOLD


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

    assert event.elapsed_ns == 2_000_000
    assert event.elapsed_ms == 2.0

    assert event.budget_ms == 1.0
    assert event.timeout_policy == "fold"
    assert event.resolution == "fold"
    assert event.applied_action is Action.FOLD


def test_timeout_event_rejects_conflicting_budget_alias_values() -> None:
    with pytest.raises(
        ValueError,
        match="budget_ms and time_budget_ms disagree",
    ):
        RecordedTimeoutEvent(
            player=0,
            bot_id=1,
            budget_ms=1.0,
            time_budget_ms=2.0,
            timeout_policy="fold",
        )


def test_timeout_event_round_trip_preserves_details() -> None:
    original = RecordedTimeoutEvent(
        player=0,
        action_index=2,
        elapsed_ms=8.25,
        budget_ms=5.0,
        timeout_policy="fold",
        applied_action=Action.FOLD,
        timeout_policy_version="timeout-policy-v1",
    )

    restored = (
        RecordedTimeoutEvent.from_dict(
            original.to_dict()
        )
    )

    assert restored == original


def test_recorded_hand_retains_acting_player_at_termination() -> None:
    hand = RecordedHand(
        config_id="A_BTN",
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=0,
        seat1_bot_id=1,
        actions=(),
        termination=HandTermination.ILLEGAL_ACTION,
        terminal_profit_bb=None,
        requested_action=Action.BET_33,
        legal_actions_at_termination=(
            Action.FOLD,
            Action.CALL,
        ),
        acting_player_at_termination=0,
        failure_message="illegal action",
    )

    assert (
        hand.acting_player_at_termination
        == 0
    )


def test_recorded_hand_round_trip_preserves_exception_metadata() -> None:
    timeout = RecordedTimeoutEvent(
        player=0,
        action_index=1,
        elapsed_ms=11.0,
        budget_ms=10.0,
        timeout_policy="fold",
        applied_action=Action.FOLD,
    )

    original = RecordedHand(
        config_id="A_BTN",
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=0,
        seat1_bot_id=1,
        actions=(),
        termination=HandTermination.AGENT_ERROR,
        terminal_profit_bb=None,
        timeout_events=(
            timeout,
        ),
        requested_action=Action.BET_33,
        legal_actions_at_termination=(
            Action.FOLD,
            Action.CALL,
        ),
        acting_player_at_termination=0,
        failure_message="agent failed",
    )

    restored = RecordedHand.from_dict(
        original.to_dict()
    )

    assert restored == original


def test_save_and_load_match_record_round_trip(
    tmp_path: Path,
) -> None:
    record = _match_record()

    destination = (
        tmp_path
        / "match.json"
    )

    save_match_record(
        record,
        destination,
    )

    loaded = load_match_record(
        destination
    )

    assert loaded == record

    assert (
        loaded.record_hash
        == record.record_hash
    )


def test_save_match_record_accepts_legacy_argument_order(
    tmp_path: Path,
) -> None:
    record = _match_record()

    destination = (
        tmp_path
        / "legacy-order.json"
    )

    save_match_record(
        destination,
        record,
    )

    assert (
        load_match_record(
            destination
        )
        == record
    )


def test_replay_archive_save_load_and_exists(
    tmp_path: Path,
) -> None:
    record = _match_record()

    archive = ReplayArchive(
        tmp_path
    )

    assert not archive.exists(
        record.match_id
    )

    destination = archive.save(
        record
    )

    assert destination == (
        tmp_path
        / "match-42.json"
    )

    assert archive.exists(
        record.match_id
    )

    assert (
        archive.load(
            record.match_id
        )
        == record
    )


def test_missing_match_record_preserves_file_not_found(
    tmp_path: Path,
) -> None:
    missing = (
        tmp_path
        / "does-not-exist.json"
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        load_match_record(
            missing
        )


def test_archive_missing_match_preserves_file_not_found(
    tmp_path: Path,
) -> None:
    archive = ReplayArchive(
        tmp_path
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        archive.load(
            "does-not-exist"
        )


def test_invalid_json_is_replay_integrity_error(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "invalid.json"
    )

    path.write_text(
        "{not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ReplayIntegrityError,
        match="valid match-record JSON",
    ):
        load_match_record(
            path
        )


def test_tampered_record_is_rejected_before_dataclass_construction(
    tmp_path: Path,
) -> None:
    record = _match_record()

    path = (
        tmp_path
        / "tampered.json"
    )

    save_match_record(
        record,
        path,
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    payload["record"][
        "master_seed"
    ] = 999

    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ReplayIntegrityError,
        match="integrity hash mismatch",
    ):
        load_match_record(
            path
        )


def test_archive_rejects_path_traversal_match_id(
    tmp_path: Path,
) -> None:
    archive = ReplayArchive(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="path separators",
    ):
        archive.path_for(
            "../outside"
        )