from decimal import Decimal

import pytest

from pokerbench.engine.config import Action, GameConfig, GameType
from pokerbench.engine.deal_schedule import DealSchedule
from pokerbench.engine.state import Street
from pokerbench.replay import HandReplay, ReplayResult


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


def test_replay_returns_replay_result(config: GameConfig) -> None:
    replay = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=0,
        dealer=0,
        actions=(
            Action.CALL,
            Action.CHECK,
        ),
    )

    result = replay.replay()

    assert isinstance(result, ReplayResult)


def test_replay_uses_deterministic_deal(config: GameConfig) -> None:
    replay = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=3,
        dealer=0,
        actions=(),
    )

    expected = DealSchedule(
        master_seed=12345,
        num_hands=4,
    ).get_hand(3)

    assert replay.deal == expected


def test_replay_initial_state_matches_deal(config: GameConfig) -> None:
    replay = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=0,
        dealer=0,
        actions=(),
    )

    result = replay.replay()
    hole_a, hole_b, board = replay.deal

    assert result.initial_state.hole_cards == (hole_a, hole_b)
    assert result.initial_state.board == board
    assert result.initial_state.street == Street.PREFLOP


def test_replay_contains_initial_and_resulting_states(
    config: GameConfig,
) -> None:
    replay = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=0,
        dealer=0,
        actions=(
            Action.CALL,
            Action.CHECK,
        ),
    )

    result = replay.replay()

    assert len(result.states) == 3
    assert result.states[0] == result.initial_state
    assert result.states[-1] == result.final_state


def test_replay_applies_actions_in_order(config: GameConfig) -> None:
    replay = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=0,
        dealer=0,
        actions=(
            Action.CALL,
            Action.CHECK,
        ),
    )

    result = replay.replay()

    assert result.actions == (
        Action.CALL,
        Action.CHECK,
    )

    assert result.final_state.street == Street.FLOP


def test_replay_is_deterministic(config: GameConfig) -> None:
    replay_a = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=7,
        dealer=1,
        actions=(
            Action.CALL,
            Action.CHECK,
        ),
    )

    replay_b = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=7,
        dealer=1,
        actions=(
            Action.CALL,
            Action.CHECK,
        ),
    )

    result_a = replay_a.replay()
    result_b = replay_b.replay()

    assert result_a == result_b
    assert result_a.initial_state == result_b.initial_state
    assert result_a.final_state == result_b.final_state


def test_different_hand_index_changes_deal(
    config: GameConfig,
) -> None:
    first = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=0,
        dealer=0,
        actions=(),
    )

    second = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=1,
        dealer=0,
        actions=(),
    )

    assert first.deal != second.deal


def test_different_seed_changes_deal(config: GameConfig) -> None:
    first = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=0,
        dealer=0,
        actions=(),
    )

    second = HandReplay(
        config=config,
        master_seed=54321,
        hand_index=0,
        dealer=0,
        actions=(),
    )

    assert first.deal != second.deal


def test_different_actions_change_final_state(
    config: GameConfig,
) -> None:
    check_line = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=0,
        dealer=0,
        actions=(
            Action.CALL,
            Action.CHECK,
        ),
    )

    fold_line = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=0,
        dealer=0,
        actions=(
            Action.FOLD,
        ),
    )

    assert (
        check_line.replay().final_state
        != fold_line.replay().final_state
    )


def test_replay_rejects_invalid_hand_index(
    config: GameConfig,
) -> None:
    with pytest.raises(ValueError):
        HandReplay(
            config=config,
            master_seed=12345,
            hand_index=-1,
            dealer=0,
            actions=(),
        )


def test_replay_rejects_invalid_dealer(
    config: GameConfig,
) -> None:
    with pytest.raises(ValueError):
        HandReplay(
            config=config,
            master_seed=12345,
            hand_index=0,
            dealer=2,
            actions=(),
        )


def test_replay_rejects_non_integer_seed(
    config: GameConfig,
) -> None:
    with pytest.raises(TypeError):
        HandReplay(
            config=config,
            master_seed="12345",  # type: ignore
            hand_index=0,
            dealer=0,
            actions=(),
        )


def test_replay_rejects_non_integer_hand_index(
    config: GameConfig,
) -> None:
    with pytest.raises(TypeError):
        HandReplay(
            config=config,
            master_seed=12345,
            hand_index="0",  # type: ignore
            dealer=0,
            actions=(),
        )

def test_replay_preserves_terminal_fold_action(
    config: GameConfig,
) -> None:
    """
    ReplayResult must retain the complete replayed decision sequence even
    when the engine's terminal GameState does not retain the fold inside
    action_history.
    """
    replay = HandReplay(
        config=config,
        master_seed=12345,
        hand_index=0,
        dealer=0,
        actions=(
            Action.FOLD,
        ),
    )

    result = replay.replay()

    assert result.final_state.is_terminal

    assert result.actions == (
        Action.FOLD,
    )

    assert len(result.states) == 2
