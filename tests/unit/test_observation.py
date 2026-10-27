from decimal import Decimal

import pytest

from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import Action, GameConfig, GameType
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


@pytest.fixture
def state(config: GameConfig) -> GameState:
    return GameState.new_hand(
        config=config,
        hole_cards=((0, 12), (25, 38)),
        board=(1, 2, 3, 4, 5),
        dealer=0,
    )


def test_observation_returns_correct_type(
    state: GameState,
) -> None:
    observation = state.observation(0)

    assert isinstance(observation, AgentObservation)


def test_player_observation_contains_own_hole_cards(
    state: GameState,
) -> None:
    player_zero = state.observation(0)
    player_one = state.observation(1)

    assert player_zero.hole_cards == (0, 12)
    assert player_one.hole_cards == (25, 38)


def test_player_observation_does_not_expose_opponent_hole_cards(
    state: GameState,
) -> None:
    player_zero = state.observation(0)
    player_one = state.observation(1)

    assert player_zero.hole_cards == state.hole_cards[0]
    assert player_one.hole_cards == state.hole_cards[1]

    assert state.hole_cards[1] != player_zero.hole_cards
    assert state.hole_cards[0] != player_one.hole_cards


def test_observation_contains_only_visible_board(
    state: GameState,
) -> None:
    observation = state.observation(0)

    assert observation.board == ()
    assert observation.board != state.board


def test_observation_contains_visible_flop(
    state: GameState,
) -> None:
    flop_state = state.apply_action(Action.CALL)
    flop_state = flop_state.apply_action(Action.CHECK)

    observation = flop_state.observation(0)

    assert flop_state.street == Street.FLOP
    assert observation.board == flop_state.board[:3]
    assert observation.board != flop_state.board


def test_observation_contains_public_game_information(
    state: GameState,
) -> None:
    observation = state.observation(0)

    assert observation.stacks == state.stacks
    assert observation.current_bets == state.current_bets
    assert observation.pot == state.pot
    assert observation.current_player == state.current_player
    assert observation.street == state.street
    assert observation.action_history == state.action_history


def test_observation_contains_legal_actions(
    state: GameState,
) -> None:
    observation = state.observation(0)

    assert observation.legal_actions == tuple(
        state.legal_actions()
    )


def test_invalid_player_id_is_rejected(
    state: GameState,
) -> None:
    with pytest.raises(ValueError):
        state.observation(2)

    with pytest.raises(ValueError):
        state.observation(-1)


def test_observation_is_immutable(
    state: GameState,
) -> None:
    observation = state.observation(0)

    with pytest.raises(AttributeError):
        observation.player_id = 1  # type: ignore


def test_observations_for_players_have_different_private_information(
    state: GameState,
) -> None:
    player_zero = state.observation(0)
    player_one = state.observation(1)

    assert player_zero.hole_cards == state.hole_cards[0]
    assert player_one.hole_cards == state.hole_cards[1]
    assert player_zero.hole_cards != player_one.hole_cards


def test_hidden_hole_cards_are_not_part_of_observation_state(
    state: GameState,
) -> None:
    player_zero = state.observation(0)

    observation_values = (
        player_zero.player_id,
        player_zero.hole_cards,
        player_zero.board,
        player_zero.stacks,
        player_zero.current_bets,
        player_zero.pot,
        player_zero.current_player,
        player_zero.street,
        player_zero.action_history,
        player_zero.legal_actions,
    )

    assert state.hole_cards[1] not in observation_values


def test_same_public_state_with_different_hidden_cards_has_same_observation(
    config: GameConfig,
) -> None:
    first = GameState.new_hand(
        config=config,
        hole_cards=((0, 12), (25, 38)),
        board=(1, 2, 3, 4, 5),
        dealer=0,
    )

    second = GameState.new_hand(
        config=config,
        hole_cards=((0, 12), (26, 39)),
        board=(1, 2, 3, 4, 5),
        dealer=0,
    )

    assert first.hole_cards != second.hole_cards
    assert first.observation(0) == second.observation(0)