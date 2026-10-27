"""Tests for calibration-opportunities-v2 replay measurement."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokerbench.agents.profiles import (
    TAG_PROFILE,
)
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.evaluation.calibration import (
    OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
    BehaviorCounts,
    calibrate_profile,
)
from pokerbench.evaluation.opportunities import (
    ActionEffect,
    CalibrationTraceError,
    classify_action_effect,
    measure_behavior_counts,
)
from pokerbench.replay import (
    HandReplay,
    HandTermination,
    RecordedHand,
    RecordedTimeoutEvent,
)


MASTER_SEED = 123


def _normal_hand(
    actions: tuple[Action, ...],
    *,
    hand_index: int = 0,
    seat0_bot_id: int = 0,
    seat1_bot_id: int = 1,
    config_id: str = "CALIBRATION_TEST",
) -> RecordedHand:
    replay = HandReplay(
        config=BASELINE_GAME_CONFIG,
        master_seed=MASTER_SEED,
        hand_index=hand_index,
        dealer=0,
        actions=actions,
    ).replay()

    assert replay.final_state.is_terminal
    assert replay.final_state.terminal_profit is not None

    profit = replay.final_state.terminal_profit

    return RecordedHand(
        config_id=config_id,
        master_seed=MASTER_SEED,
        hand_index=hand_index,
        dealer=0,
        seat0_bot_id=seat0_bot_id,
        seat1_bot_id=seat1_bot_id,
        actions=actions,
        termination=HandTermination.NORMAL,
        terminal_profit_bb=(
            profit[0] / 100.0,
            profit[1] / 100.0,
        ),
    )


def test_opportunity_contract_version_is_v2() -> None:
    assert (
        OPPORTUNITY_DEFINITION_CONTRACT_VERSION
        == "calibration-opportunities-v2"
    )


def test_vpip_is_one_dealt_hand_and_fold_is_not_event() -> None:
    hand = _normal_hand(
        (
            Action.FOLD,
        )
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=0,
    )

    assert counts.vpip_opportunities == 1
    assert counts.vpip_events == 0
    assert counts.pfr_opportunities == 1
    assert counts.pfr_events == 0


def test_vpip_counts_voluntary_call_once_per_hand() -> None:
    hand = _normal_hand(
        (
            Action.CALL,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
        )
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=0,
    )

    assert counts.vpip_opportunities == 1
    assert counts.vpip_events == 1
    assert counts.pfr_events == 0


def test_multiple_preflop_raises_still_count_one_vpip_and_pfr_event() -> None:
    hand = _normal_hand(
        (
            Action.BET_100,
            Action.BET_200,
            Action.BET_200,
            Action.FOLD,
        )
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=0,
    )

    assert counts.vpip_opportunities == 1
    assert counts.vpip_events == 1
    assert counts.pfr_opportunities == 1
    assert counts.pfr_events == 1

    # The target's second raise is a 4-bet, not a 3-bet.
    assert counts.three_bet_opportunities == 0
    assert counts.three_bet_events == 0


def test_exactly_one_prior_raise_creates_three_bet_opportunity() -> None:
    hand = _normal_hand(
        (
            Action.BET_100,
            Action.BET_200,
            Action.FOLD,
        )
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=1,
    )

    assert counts.three_bet_opportunities == 1
    assert counts.three_bet_events == 1


def test_three_bet_measurement_uses_stable_bot_identity_after_seat_swap() -> None:
    hand = _normal_hand(
        (
            Action.BET_100,
            Action.BET_200,
            Action.FOLD,
        ),
        seat0_bot_id=1,
        seat1_bot_id=0,
        config_id="B_BTN",
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=0,
    )

    assert counts.three_bet_opportunities == 1
    assert counts.three_bet_events == 1


def test_checked_to_preflop_aggressor_has_cbet_opportunity() -> None:
    hand = _normal_hand(
        (
            Action.BET_200,
            Action.CALL,
            Action.CHECK,
            Action.BET_100,
            Action.FOLD,
        )
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=0,
    )

    assert counts.cbet_opportunities == 1
    assert counts.cbet_events == 1


def test_donk_bet_before_preflop_aggressor_removes_cbet_opportunity() -> None:
    hand = _normal_hand(
        (
            Action.BET_200,
            Action.CALL,
            Action.BET_100,
            Action.FOLD,
        )
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=0,
    )

    assert counts.cbet_opportunities == 0
    assert counts.cbet_events == 0


def test_direct_fold_to_qualifying_cbet_is_counted() -> None:
    hand = _normal_hand(
        (
            Action.BET_200,
            Action.CALL,
            Action.CHECK,
            Action.BET_100,
            Action.FOLD,
        )
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=1,
    )

    assert counts.fold_to_cbet_opportunities == 1
    assert counts.fold_to_cbet_events == 1


def test_call_to_qualifying_cbet_is_opportunity_but_not_fold_event() -> None:
    hand = _normal_hand(
        (
            Action.BET_200,
            Action.CALL,
            Action.CHECK,
            Action.BET_100,
            Action.CALL,
            Action.BET_100,
            Action.FOLD,
        )
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=1,
    )

    assert counts.fold_to_cbet_opportunities == 1
    assert counts.fold_to_cbet_events == 0

    # Postflop for bot 1: check + call are passive, turn bet is aggressive.
    assert counts.postflop_passive_actions == 2
    assert counts.postflop_aggressive_actions == 1


def test_postflop_folds_are_excluded_from_aggression_denominator() -> None:
    hand = _normal_hand(
        (
            Action.BET_200,
            Action.CALL,
            Action.CHECK,
            Action.BET_100,
            Action.FOLD,
        )
    )

    counts = measure_behavior_counts(
        [hand],
        BASELINE_GAME_CONFIG,
        bot_id=1,
    )

    assert counts.postflop_passive_actions == 1
    assert counts.postflop_aggressive_actions == 0


def test_all_in_is_aggressive_when_it_increases_wager() -> None:
    replay = HandReplay(
        config=BASELINE_GAME_CONFIG,
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        actions=(
            Action.CALL,
            Action.CHECK,
        ),
    ).replay()

    flop_state = replay.final_state

    assert Action.ALL_IN in flop_state.legal_actions()
    assert (
        classify_action_effect(
            flop_state,
            Action.ALL_IN,
        )
        == ActionEffect.AGGRESSIVE
    )


def test_all_in_is_passive_when_it_is_call_equivalent() -> None:
    replay = HandReplay(
        config=BASELINE_GAME_CONFIG,
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        actions=(
            Action.CALL,
            Action.CHECK,
        ),
    ).replay()

    passive_all_in_state = replace(
        replay.final_state,
        current_player=1,
        current_bets=(900, 0),
        stacks=(9000, 500),
        pot=900,
        last_raise=100,
    )

    assert (
        Action.ALL_IN
        in passive_all_in_state.legal_actions()
    )

    assert (
        classify_action_effect(
            passive_all_in_state,
            Action.ALL_IN,
        )
        == ActionEffect.PASSIVE
    )


def test_abnormal_hand_is_rejected_from_calibration() -> None:
    hand = RecordedHand(
        config_id="AGENT_ERROR",
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=0,
        seat1_bot_id=1,
        actions=(),
        termination=HandTermination.AGENT_ERROR,
        terminal_profit_bb=None,
        acting_player_at_termination=0,
        failure_message="agent failed",
    )

    with pytest.raises(
        CalibrationTraceError,
        match="normal hand termination",
    ):
        measure_behavior_counts(
            [hand],
            BASELINE_GAME_CONFIG,
            bot_id=0,
        )


def test_timeout_altered_normal_hand_is_rejected() -> None:
    base = _normal_hand(
        (
            Action.CALL,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
        )
    )

    timeout_altered = replace(
        base,
        timeout_events=(
            RecordedTimeoutEvent(
                player=1,
                bot_id=1,
                action_index=2,
                decision_index=0,
                elapsed_ns=2_000_000,
                budget_ms=1.0,
                timeout_policy=(
                    "check_if_legal_else_fold"
                ),
                resolution="check",
                applied_action=Action.CHECK,
            ),
        ),
    )

    with pytest.raises(
        CalibrationTraceError,
        match="timeout",
    ):
        measure_behavior_counts(
            [timeout_altered],
            BASELINE_GAME_CONFIG,
            bot_id=0,
        )


def test_behavior_counts_aggregate_across_hands() -> None:
    fold_hand = _normal_hand(
        (
            Action.FOLD,
        ),
        hand_index=0,
    )

    vpip_hand = _normal_hand(
        (
            Action.CALL,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
            Action.CHECK,
        ),
        hand_index=1,
    )

    counts = measure_behavior_counts(
        [
            fold_hand,
            vpip_hand,
        ],
        BASELINE_GAME_CONFIG,
        bot_id=0,
    )

    assert counts.vpip_opportunities == 2
    assert counts.vpip_events == 1


def test_active_calibration_rejects_old_opportunity_version() -> None:
    counts = BehaviorCounts(
        opportunity_definition_version=(
            "calibration-opportunities-v1"
        )
    )

    with pytest.raises(
        ValueError,
        match="unsupported opportunity-definition version",
    ):
        calibrate_profile(
            TAG_PROFILE,
            counts,
        )
