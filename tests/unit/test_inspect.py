from __future__ import annotations

from pathlib import Path

import pytest

from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.replay import (
    HandReplay,
    HandTermination,
    MatchRecord,
    RecordedHand,
)
from pokerbench.cli.inspect import (
    InspectOptions,
    inspect_match,
    render_inspection_html,
    render_inspection_text,
)


MASTER_SEED = 42


def _recorded_hand(
    *,
    actions: tuple[Action, ...],
    hand_index: int = 0,
    dealer: int = 0,
    seat0_bot_id: int = 0,
    seat1_bot_id: int = 1,
) -> RecordedHand:
    replay = HandReplay(
        config=BASELINE_GAME_CONFIG,
        master_seed=MASTER_SEED,
        hand_index=hand_index,
        dealer=dealer,
        actions=actions,
    ).replay()

    assert replay.final_state.is_terminal
    assert replay.final_state.terminal_profit is not None

    profit = replay.final_state.terminal_profit

    return RecordedHand(
        config_id="INSPECT_TEST",
        master_seed=MASTER_SEED,
        hand_index=hand_index,
        dealer=dealer,
        seat0_bot_id=seat0_bot_id,
        seat1_bot_id=seat1_bot_id,
        actions=actions,
        termination=HandTermination.NORMAL,
        terminal_profit_bb=(
            profit[0] / 100.0,
            profit[1] / 100.0,
        ),
    )


def _river_loss_hand() -> RecordedHand:
    """
    Bot 0 loses more than 50 BB after reaching the river.

    Approximate betting line:

    preflop:
        bot 0 raises
        bot 1 calls

    flop:
        bot 1 checks
        bot 0 bets
        bot 1 calls

    turn:
        bot 1 checks
        bot 0 bets
        bot 1 calls

    river:
        bot 1 bets
        bot 0 folds
    """
    return _recorded_hand(
        actions=(
            Action.BET_200,
            Action.CALL,
            Action.CHECK,
            Action.BET_100,
            Action.CALL,
            Action.CHECK,
            Action.BET_200,
            Action.CALL,
            Action.BET_33,
            Action.FOLD,
        ),
    )


def _simple_fold_hand(
    hand_index: int = 1,
) -> RecordedHand:
    return _recorded_hand(
        actions=(
            Action.FOLD,
        ),
        hand_index=hand_index,
    )


def _match_record() -> MatchRecord:
    return MatchRecord(
        match_id="inspect-test-match",
        master_seed=MASTER_SEED,
        game_config=BASELINE_GAME_CONFIG,
        hands=(
            _river_loss_hand(),
            _simple_fold_hand(),
        ),
    )


def test_inspect_without_filters_returns_all_hands() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(),
    )

    assert len(inspected) == 2


def test_street_filter_selects_hand_that_reached_river() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(
            street="river",
        ),
    )

    assert len(inspected) == 1

    assert (
        inspected[0]
        .record
        .hand_index
        == 0
    )


def test_action_filter_selects_hand_containing_action() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(
            action=Action.BET_100,
        ),
    )

    assert len(inspected) == 1

    assert (
        inspected[0]
        .record
        .hand_index
        == 0
    )


def test_min_pot_filter_uses_bb() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(
            min_pot_bb=50.0,
        ),
    )

    assert len(inspected) == 1

    assert (
        inspected[0]
        .record
        .hand_index
        == 0
    )


def test_compound_filter_matches_required_example_semantics() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(
            hero_bot_id=0,
            filter_expression=(
                "hero_loss > 50 "
                "AND street == 'river'"
            ),
        ),
    )

    assert len(inspected) == 1

    assert (
        inspected[0]
        .record
        .hand_index
        == 0
    )


def test_compound_filter_is_case_insensitive_for_and() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(
            filter_expression=(
                "street == 'river' "
                "and action == 'fold'"
            ),
        ),
    )

    assert len(inspected) == 1


def test_invalid_filter_field_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported filter field",
    ):
        inspect_match(
            _match_record(),
            InspectOptions(
                filter_expression=(
                    "unknown_field == 1"
                ),
            ),
        )


def test_or_filter_is_rejected_in_v1() -> None:
    with pytest.raises(
        ValueError,
        match="AND",
    ):
        inspect_match(
            _match_record(),
            InspectOptions(
                filter_expression=(
                    "street == 'river' "
                    "OR action == 'fold'"
                ),
            ),
        )


def test_inspection_preserves_complete_decision_chain() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(
            street="river",
        ),
    )

    hand = inspected[0]

    assert (
        tuple(
            decision.chosen_action
            for decision
            in hand.decisions
        )
        == hand.record.actions
    )

    assert (
        len(hand.decisions)
        == len(
            hand.record.actions
        )
    )


def test_decision_snapshot_contains_legal_actions() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(
            street="river",
        ),
    )

    for decision in (
        inspected[0].decisions
    ):
        assert (
            decision.chosen_action
            in decision.legal_actions
        )


def test_decision_snapshot_contains_agent_visible_board() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(
            street="river",
        ),
    )

    hand = inspected[0]

    preflop = hand.decisions[0]

    assert (
        preflop.visible_board
        == ()
    )

    river_decisions = [
        decision
        for decision
        in hand.decisions
        if decision.street
        == "river"
    ]

    assert river_decisions

    assert all(
        len(
            decision.visible_board
        )
        == 5
        for decision
        in river_decisions
    )


def test_text_renderer_contains_required_replay_information() -> None:
    inspected = inspect_match(
        _match_record(),
        InspectOptions(
            street="river",
        ),
    )

    output = render_inspection_text(
        _match_record(),
        inspected,
        hero_bot_id=0,
    )

    assert "Hole cards" in output
    assert "Board" in output
    assert "Position" in output
    assert "Stacks" in output
    assert "Pot" in output
    assert "Action history" in output
    assert "Available actions" in output
    assert "Chosen action" in output

    assert (
        "Estimated hand strength: N/A"
        in output
    )

    assert (
        "Estimated decision EV: N/A"
        in output
    )

    assert "Resulting profit" in output


def test_html_renderer_contains_required_replay_information() -> None:
    record = _match_record()

    inspected = inspect_match(
        record,
        InspectOptions(
            street="river",
        ),
    )

    html = render_inspection_html(
        record,
        inspected,
        hero_bot_id=0,
    )

    assert "<!DOCTYPE html>" in html
    assert "Hole cards" in html
    assert "Available actions" in html
    assert "Chosen action" in html

    assert (
        "Estimated hand strength"
        in html
    )

    assert (
        "Estimated decision EV"
        in html
    )


def test_invalid_hero_bot_id_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="hero_bot_id",
    ):
        InspectOptions(
            hero_bot_id=2,
        )


def test_negative_min_pot_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="min_pot_bb",
    ):
        InspectOptions(
            min_pot_bb=-1.0,
        )