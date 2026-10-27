from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from pokerbench.agents.observation import (
    AGENT_OBSERVATION_VERSION,
    AgentObservation,
)
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.engine.public_action import (
    PUBLIC_ACTION_TRACE_VERSION,
    PublicActionEvent,
)
from pokerbench.engine.state import (
    GameState,
    Street,
)


HOLE_CARDS = (
    (0, 1),
    (2, 3),
)
BOARD = (4, 5, 6, 7, 8)


def _new_state(
    *,
    dealer: int = 0,
    hole_cards: tuple[
        tuple[int, int],
        tuple[int, int],
    ] = HOLE_CARDS,
    board: tuple[int, int, int, int, int] = BOARD,
) -> GameState:
    return GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=hole_cards,
        board=board,
        dealer=dealer,
    )


def test_observation_v2_version_constants_are_frozen() -> None:
    assert (
        AGENT_OBSERVATION_VERSION
        == "agent-observation-v2"
    )
    assert (
        PUBLIC_ACTION_TRACE_VERSION
        == "public-action-trace-v1"
    )


def test_initial_observation_contains_public_position_and_empty_trace() -> None:
    state = _new_state(
        dealer=1,
    )

    observation = state.observation(
        1
    )

    assert isinstance(
        observation,
        AgentObservation,
    )
    assert observation.dealer == 1
    assert observation.player_id == 1
    assert observation.public_action_history == ()
    assert observation.board == ()


def test_public_action_event_is_frozen() -> None:
    event = PublicActionEvent(
        actor=0,
        street=Street.PREFLOP,
        action=Action.CALL,
        wager_increase=False,
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        event.actor = 1  # type: ignore[misc]


def test_public_trace_records_actor_street_action_and_wager_effect() -> None:
    state = _new_state()

    after_raise = state.apply_action(
        Action.BET_200
    )
    after_call = after_raise.apply_action(
        Action.CALL
    )
    after_check = after_call.apply_action(
        Action.CHECK
    )
    after_bet = after_check.apply_action(
        Action.BET_66
    )

    assert after_bet.public_action_history == (
        PublicActionEvent(
            actor=0,
            street=Street.PREFLOP,
            action=Action.BET_200,
            wager_increase=True,
        ),
        PublicActionEvent(
            actor=1,
            street=Street.PREFLOP,
            action=Action.CALL,
            wager_increase=False,
        ),
        PublicActionEvent(
            actor=1,
            street=Street.FLOP,
            action=Action.CHECK,
            wager_increase=False,
        ),
        PublicActionEvent(
            actor=0,
            street=Street.FLOP,
            action=Action.BET_66,
            wager_increase=True,
        ),
    )

    observation = after_bet.observation(
        after_bet.current_player
    )

    assert (
        observation.public_action_history
        == after_bet.public_action_history
    )


def test_fold_is_present_in_public_trace_without_changing_legacy_history() -> None:
    state = _new_state()

    terminal = state.apply_action(
        Action.FOLD
    )

    assert terminal.is_terminal

    # Preserve the existing replay-sensitive legacy behavior.
    assert terminal.action_history == ()

    assert terminal.public_action_history == (
        PublicActionEvent(
            actor=0,
            street=Street.PREFLOP,
            action=Action.FOLD,
            wager_increase=False,
        ),
    )


def test_all_in_raise_is_marked_as_wager_increase() -> None:
    state = _new_state()

    after_all_in = state.apply_action(
        Action.ALL_IN
    )

    assert (
        after_all_in.public_action_history[-1]
        == PublicActionEvent(
            actor=0,
            street=Street.PREFLOP,
            action=Action.ALL_IN,
            wager_increase=True,
        )
    )


def test_short_all_in_call_is_not_marked_as_wager_increase() -> None:
    state = replace(
        _new_state(),
        stacks=(25, 9_900),
    )

    assert Action.ALL_IN in state.legal_actions()

    after_all_in = state.apply_action(
        Action.ALL_IN
    )

    assert (
        after_all_in.public_action_history[-1]
        == PublicActionEvent(
            actor=0,
            street=Street.PREFLOP,
            action=Action.ALL_IN,
            wager_increase=False,
        )
    )


def test_trace_survives_street_advance() -> None:
    state = _new_state()

    flop = (
        state
        .apply_action(Action.CALL)
        .apply_action(Action.CHECK)
    )

    assert flop.street is Street.FLOP
    assert tuple(
        event.street
        for event in flop.public_action_history
    ) == (
        Street.PREFLOP,
        Street.PREFLOP,
    )


def test_observation_v2_does_not_reveal_opponent_or_future_cards() -> None:
    first = _new_state(
        hole_cards=(
            (0, 1),
            (2, 3),
        ),
        board=(4, 5, 6, 7, 8),
    )

    second = _new_state(
        hole_cards=(
            (9, 10),
            (2, 3),
        ),
        board=(4, 5, 6, 11, 12),
    )

    first_flop = (
        first
        .apply_action(Action.CALL)
        .apply_action(Action.CHECK)
    )
    second_flop = (
        second
        .apply_action(Action.CALL)
        .apply_action(Action.CHECK)
    )

    first_observation = (
        first_flop.observation(1)
    )
    second_observation = (
        second_flop.observation(1)
    )

    assert first_observation == second_observation
    assert first_observation.hole_cards == (2, 3)
    assert first_observation.board == (4, 5, 6)

@pytest.mark.parametrize(
    ("kwargs", "expected_exception"),
    [
        (
            {
                "actor": 2,
                "street": Street.PREFLOP,
                "action": Action.CALL,
                "wager_increase": False,
            },
            ValueError,
        ),
        (
            {
                "actor": 0,
                "street": Street.PREFLOP,
                "action": Action.CALL,
                "wager_increase": 1,
            },
            TypeError,
        ),
    ],
)
def test_public_action_event_rejects_invalid_contract_values(
    kwargs: dict[str, object],
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(
        expected_exception,
    ):
        PublicActionEvent(**kwargs)  # type: ignore[arg-type]
