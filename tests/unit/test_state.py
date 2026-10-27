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
    hole_cards=((1, 2), (3, 4)),
    board=(5, 6, 7, 8, 9),
    dealer=0,
) -> GameState:
    return GameState.new_hand(
        config,
        hole_cards,
        board,
        dealer,
    )


def test_initial_state_preflop(config: GameConfig) -> None:
    state = create_state(config)

    sb = int(config.small_blind_bb * CHIP_UNIT)
    bb = int(config.big_blind_bb * CHIP_UNIT)
    start = int(config.starting_stack_bb * CHIP_UNIT)

    assert state.street == Street.PREFLOP
    assert state.dealer == 0

    assert state.stacks == (
        start - sb,
        start - bb,
    )

    assert state.current_bets == (sb, bb)
    assert state.pot == sb + bb

    # Heads-up preflop:
    # Button / Small Blind acts first.
    assert state.current_player == 0

    assert not state.is_terminal


def test_initial_state_with_player_one_as_dealer(
    config: GameConfig,
) -> None:
    state = create_state(config, dealer=1)

    sb = int(config.small_blind_bb * CHIP_UNIT)
    bb = int(config.big_blind_bb * CHIP_UNIT)
    start = int(config.starting_stack_bb * CHIP_UNIT)

    assert state.dealer == 1

    assert state.stacks == (
        start - bb,
        start - sb,
    )

    assert state.current_bets == (bb, sb)

    # Heads-up preflop:
    # dealer/Button/Small Blind acts first.
    assert state.current_player == 1


def test_legal_actions_preflop_big_blind_has_fold_call_raise(
    config: GameConfig,
) -> None:
    state = create_state(config)
    actions = state.legal_actions()

    assert Action.FOLD in actions
    assert Action.CALL in actions
    assert Action.CHECK not in actions
    assert Action.BET_33 not in actions
    assert Action.ALL_IN in actions


def test_legal_actions_after_call(config: GameConfig) -> None:
    state = create_state(config)
    state2 = state.apply_action(Action.CALL)

    actions = state2.legal_actions()

    assert Action.CHECK in actions
    assert Action.FOLD not in actions
    assert Action.CALL not in actions
    assert Action.BET_100 in actions
    assert Action.ALL_IN in actions


def test_bet_amount_calculation(config: GameConfig) -> None:
    state = create_state(config)

    amount = state._bet_amount(Action.BET_33)

    assert amount == 99


def test_raise_legal_when_enough_stack(
    config: GameConfig,
) -> None:
    state = create_state(config)

    assert Action.BET_33 not in state.legal_actions()
    assert Action.BET_200 in state.legal_actions()


def test_all_in_legal(config: GameConfig) -> None:
    state = create_state(config)

    assert Action.ALL_IN in state.legal_actions()


def test_apply_all_in_keeps_opponent_facing_action(
    config: GameConfig,
) -> None:
    state = create_state(config)
    state2 = state.apply_action(Action.ALL_IN)

    assert state2.current_player == 1

    actions = state2.legal_actions()

    assert Action.CALL in actions
    assert Action.FOLD in actions
    assert Action.CHECK not in actions


def test_fold_profit_is_net_profit(
    config: GameConfig,
) -> None:
    state = create_state(config)

    terminal = state.apply_action(Action.FOLD)

    assert terminal.is_terminal
    assert terminal.terminal_profit == (-50, 50)


def test_check_after_call_advances_street(
    config: GameConfig,
) -> None:
    state = create_state(config, dealer=0)

    # Preflop:
    # Button / Small Blind acts first.
    state2 = state.apply_action(Action.CALL)

    # Big Blind checks.
    state3 = state2.apply_action(Action.CHECK)

    assert state3.street == Street.FLOP
    assert state3.current_bets == (0, 0)
    assert state3.pot == 200
    assert state3.pot_at_street_start == 200
    assert state3.street_actions == ()

    # Postflop:
    # Big Blind acts first.
    assert state3.current_player == 1 - state3.dealer
    assert state3.current_player == 1


def test_postflop_action_order_rotates_with_dealer(
    config: GameConfig,
) -> None:
    state = create_state(config, dealer=1)

    # Preflop:
    # Player 1 is Button / Small Blind and therefore acts first.
    assert state.current_player == 1

    state2 = state.apply_action(Action.CALL)
    state3 = state2.apply_action(Action.CHECK)

    assert state3.street == Street.FLOP
    assert state3.dealer == 1

    # Postflop:
    # Player 0 is Big Blind and therefore acts first.
    assert state3.current_player == 0
    assert state3.current_player == 1 - state3.dealer


def test_postflop_first_actor_is_big_blind_on_all_streets(
    config: GameConfig,
) -> None:
    state = create_state(config, dealer=0)

    # Preflop:
    # Player 0 = Button / Small Blind.
    assert state.street == Street.PREFLOP
    assert state.current_player == 0

    # Finish preflop: SB calls, BB checks.
    state = state.apply_action(Action.CALL)
    state = state.apply_action(Action.CHECK)

    # Flop:
    # Player 1 = Big Blind acts first.
    assert state.street == Street.FLOP
    assert state.current_player == 1

    # Finish flop: BB checks, SB checks.
    state = state.apply_action(Action.CHECK)
    state = state.apply_action(Action.CHECK)

    # Turn:
    # Big Blind still acts first.
    assert state.street == Street.TURN
    assert state.current_player == 1

    # Finish turn.
    state = state.apply_action(Action.CHECK)
    state = state.apply_action(Action.CHECK)

    # River:
    # Big Blind still acts first.
    assert state.street == Street.RIVER
    assert state.current_player == 1


def test_apply_raise_updates_last_raise(
    config: GameConfig,
) -> None:
    state = create_state(config)
    state2 = state.apply_action(Action.BET_200)

    assert state2.last_raise == 300
    assert state2.last_aggressor == 0


def test_round_complete_after_call_to_raise(
    config: GameConfig,
) -> None:
    state = create_state(config)

    state2 = state.apply_action(Action.BET_200)
    state3 = state2.apply_action(Action.CALL)

    assert state3.street == Street.FLOP
    assert state3.current_bets == (0, 0)
    assert state3.pot == 800
    assert state3.pot_at_street_start == 800
    assert state3.street_actions == ()

    # dealer=0 => player 1 is Big Blind.
    assert state3.current_player == 1


def test_terminal_showdown_profit_is_net_profit(
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

    assert state.street == Street.FLOP

    for _ in range(3):
        state = state.apply_action(Action.CHECK)
        state = state.apply_action(Action.CHECK)

    assert state.is_terminal
    assert state.terminal_profit is not None
    assert (
        state.terminal_profit[0]
        + state.terminal_profit[1]
        == 0
    )


def test_next_hand_resets_stacks_and_rotates_dealer(
    config: GameConfig,
) -> None:
    state = create_state(config, dealer=0)
    terminal = state.apply_action(Action.FOLD)

    next_state = terminal.next_hand(
        hole_cards=((10, 11), (20, 21)),
        board=(22, 23, 24, 25, 26),
    )

    start = int(config.starting_stack_bb * CHIP_UNIT)
    sb = int(config.small_blind_bb * CHIP_UNIT)
    bb = int(config.big_blind_bb * CHIP_UNIT)

    assert next_state.dealer == 1
    assert next_state.street == Street.PREFLOP

    assert next_state.stacks == (
        start - bb,
        start - sb,
    )

    assert next_state.current_bets == (bb, sb)
    assert next_state.pot == sb + bb
    assert not next_state.is_terminal

    # New hand is preflop, so new Button/SB acts first.
    assert next_state.current_player == 1


def test_next_hand_requires_terminal_state(
    config: GameConfig,
) -> None:
    state = create_state(config)

    with pytest.raises(ValueError):
        state.next_hand(
            hole_cards=((10, 11), (20, 21)),
            board=(22, 23, 24, 25, 26),
        )


def test_terminal_profit_is_zero_sum(
    config: GameConfig,
) -> None:
    state = create_state(config)
    terminal = state.apply_action(Action.FOLD)

    assert terminal.terminal_profit is not None
    assert sum(terminal.terminal_profit) == 0


def test_terminal_state_rejects_further_actions(
    config: GameConfig,
) -> None:
    state = create_state(config)
    terminal = state.apply_action(Action.FOLD)

    with pytest.raises(ValueError):
        terminal.apply_action(Action.CHECK)