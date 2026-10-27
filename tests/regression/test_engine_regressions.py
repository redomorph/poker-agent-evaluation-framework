from decimal import Decimal

import pytest

from pokerbench.engine.config import Action, GameConfig, GameType
from pokerbench.engine.constants import CHIP_UNIT
from pokerbench.engine.state import GameState, Street


@pytest.fixture
def config() -> GameConfig:
    return GameConfig(
        game_type=GameType.NLHE,
        player_count=2,
        starting_stack_bb=Decimal("100"),
        small_blind_bb=Decimal("0.5"),
        big_blind_bb=Decimal("1"),
        ante_bb=Decimal("0"),
        rake=Decimal("0"),
        stack_reset_bb=Decimal("100"),
        bet_abstraction=(
            Decimal("0.33"),
            Decimal("0.66"),
            Decimal("1.0"),
            Decimal("2.0"),
        ),
        allowed_actions=(
            Action.FOLD,
            Action.CHECK,
            Action.CALL,
            Action.BET_33,
            Action.BET_66,
            Action.BET_100,
            Action.BET_200,
            Action.ALL_IN,
        ),
        abstraction_version="baseline-v1",
    )


def create_state(
    config: GameConfig,
    hole_cards: tuple[tuple[int, int], tuple[int, int]] = (
        (1, 2),
        (3, 4),
    ),
    board: tuple[int, int, int, int, int] = (
        5,
        6,
        7,
        8,
        9,
    ),
    dealer: int = 0,
) -> GameState:
    return GameState.new_hand(
        config=config,
        hole_cards=hole_cards,
        board=board,
        dealer=dealer,
    )


def test_regression_fold_profit_is_net_profit(
    config: GameConfig,
) -> None:
    state = create_state(config)

    terminal = state.apply_action(Action.FOLD)

    assert terminal.is_terminal
    assert terminal.terminal_profit == (-50, 50)


def test_regression_position_rotation_between_dealers(
    config: GameConfig,
) -> None:
    state_dealer_zero = create_state(
        config,
        dealer=0,
    )
    state_dealer_one = create_state(
        config,
        dealer=1,
    )

    assert state_dealer_zero.current_player == 0
    assert state_dealer_one.current_player == 1

    sb = int(config.small_blind_bb * CHIP_UNIT)
    bb = int(config.big_blind_bb * CHIP_UNIT)
    starting_stack = int(config.starting_stack_bb * CHIP_UNIT)

    assert state_dealer_zero.current_bets == (sb, bb)
    assert state_dealer_one.current_bets == (bb, sb)

    assert state_dealer_zero.stacks == (
        starting_stack - sb,
        starting_stack - bb,
    )
    assert state_dealer_one.stacks == (
        starting_stack - bb,
        starting_stack - sb,
    )


def test_regression_minimum_raise_rejects_insufficient_raise(
    config: GameConfig,
) -> None:
    state = create_state(config)

    legal_actions = state.legal_actions()

    assert Action.BET_33 not in legal_actions
    assert Action.BET_200 in legal_actions


def test_regression_stack_limiting_for_all_in(
    config: GameConfig,
) -> None:
    state = create_state(config)

    initial_stack = state.stacks[state.current_player]

    next_state = state.apply_action(Action.ALL_IN)

    assert next_state.stacks[state.current_player] == 0
    assert (
        next_state.pot
        == state.pot + initial_stack
    )


def test_regression_pot_accounting_after_call(
    config: GameConfig,
) -> None:
    state = create_state(config)

    initial_pot = state.pot
    call_amount = (
        state.current_bets[1]
        - state.current_bets[0]
    )

    next_state = state.apply_action(Action.CALL)

    assert next_state.pot == initial_pot + call_amount
    assert next_state.current_bets == (
        state.current_bets[1],
        state.current_bets[1],
    )


def test_regression_street_transition_after_preflop_call_and_check(
    config: GameConfig,
) -> None:
    state = create_state(config)

    state = state.apply_action(Action.CALL)
    state = state.apply_action(Action.CHECK)

    assert state.street == Street.FLOP
    assert state.current_bets == (0, 0)


def test_regression_split_pot_has_zero_net_profit(
    config: GameConfig,
) -> None:
    state = create_state(
        config,
        hole_cards=(
            (12, 11),
            (0, 1),
        ),
        board=(2, 3, 4, 5, 6),
    )

    state = state.apply_action(Action.CALL)
    state = state.apply_action(Action.CHECK)

    for _ in range(2):
        state = state.apply_action(Action.CHECK)
        state = state.apply_action(Action.CHECK)

    state = state.apply_action(Action.CHECK)
    state = state.apply_action(Action.CHECK)

    assert state.is_terminal
    assert state.terminal_profit is not None
    assert state.terminal_profit == (0, 0)

def test_regression_all_in_call_goes_directly_to_showdown(
    config: GameConfig,
) -> None:
    state = create_state(config)

    # Player 0 goes all-in.
    all_in_state = state.apply_action(Action.ALL_IN)

    # The opponent must still be allowed to respond.
    assert not all_in_state.is_terminal
    assert all_in_state.current_player == 1
    assert Action.CALL in all_in_state.legal_actions()
    assert Action.FOLD in all_in_state.legal_actions()

    # Calling the all-in must finish the hand.
    terminal = all_in_state.apply_action(Action.CALL)

    assert terminal.is_terminal
    assert terminal.terminal_profit is not None
    assert terminal.terminal_profit[0] + terminal.terminal_profit[1] == 0