from __future__ import annotations

from dataclasses import replace

import pytest

from pokerbench.agents.equity import (
    EQUITY_RANGE_VERSION,
    EQUITY_SAMPLES,
    EQUITY_VERSION,
    EXACT_OUTCOME_LIMIT,
    EquityMethod,
    estimate_uniform_random_equity,
    joint_outcome_count,
    select_equity_method,
)
from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import Action
from pokerbench.engine.public_action import PublicActionEvent
from pokerbench.engine.state import Street
from pokerbench.rng import AgentRNG


def _observation(
    *,
    street: Street,
    hole_cards: tuple[int, int],
    board: tuple[int, ...],
) -> AgentObservation:
    return AgentObservation(
        player_id=0,
        hole_cards=hole_cards,
        board=board,
        stacks=(10_000, 10_000),
        current_bets=(0, 0),
        pot=200,
        current_player=0,
        street=street,
        action_history=(),
        legal_actions=(Action.CHECK, Action.BET_100),
        dealer=0,
        public_action_history=(),
    )


def test_equity_versions_are_frozen() -> None:
    assert EQUITY_VERSION == "uniform-random-equity-v1"
    assert EQUITY_RANGE_VERSION == "uniform-random-range-v1"
    assert EXACT_OUTCOME_LIMIT == 500_000
    assert EQUITY_SAMPLES == 10_000


def test_joint_outcome_counts_match_frozen_definition() -> None:
    river = _observation(
        street=Street.RIVER,
        hole_cards=(0, 1),
        board=(2, 3, 4, 5, 6),
    )
    turn = replace(
        river,
        street=Street.TURN,
        board=(2, 3, 4, 5),
    )
    flop = replace(
        river,
        street=Street.FLOP,
        board=(2, 3, 4),
    )

    assert joint_outcome_count(river) == 990
    assert joint_outcome_count(turn) == 45_540
    assert joint_outcome_count(flop) == 1_070_190


def test_method_selection_uses_500k_joint_outcome_threshold() -> None:
    turn = _observation(
        street=Street.TURN,
        hole_cards=(0, 1),
        board=(2, 3, 4, 5),
    )
    flop = replace(
        turn,
        street=Street.FLOP,
        board=(2, 3, 4),
    )

    assert select_equity_method(turn) is EquityMethod.EXACT
    assert select_equity_method(flop) is EquityMethod.MONTE_CARLO


def test_exact_river_equity_is_one_for_unbeatable_royal_flush() -> None:
    # Spades: T=47, J=48, Q=49, K=50, A=51.
    observation = _observation(
        street=Street.RIVER,
        hole_cards=(51, 50),
        board=(49, 48, 47, 0, 14),
    )

    result = estimate_uniform_random_equity(
        observation,
        master_seed=42,
        bot_id=0,
        hand_id=7,
        decision_index=3,
    )

    assert result.method is EquityMethod.EXACT
    assert result.total_outcomes == 990
    assert result.wins == 990
    assert result.ties == 0
    assert result.losses == 0
    assert result.equity_ppm == 1_000_000


def test_exact_river_equity_is_half_when_board_forces_royal_flush_tie() -> None:
    observation = _observation(
        street=Street.RIVER,
        hole_cards=(0, 1),
        board=(47, 48, 49, 50, 51),
    )

    result = estimate_uniform_random_equity(
        observation,
        master_seed=1,
        bot_id=0,
        hand_id=0,
        decision_index=0,
    )

    assert result.method is EquityMethod.EXACT
    assert result.wins == 0
    assert result.ties == 990
    assert result.losses == 0
    assert result.equity_ppm == 500_000


def test_monte_carlo_uses_exactly_10k_trials_and_is_deterministic() -> None:
    observation = _observation(
        street=Street.FLOP,
        hole_cards=(51, 50),
        board=(0, 14, 28),
    )

    first = estimate_uniform_random_equity(
        observation,
        master_seed=12345,
        bot_id=1,
        hand_id=99,
        decision_index=4,
    )
    second = estimate_uniform_random_equity(
        observation,
        master_seed=12345,
        bot_id=1,
        hand_id=99,
        decision_index=4,
    )

    assert first == second
    assert first.method is EquityMethod.MONTE_CARLO
    assert first.total_outcomes == EQUITY_SAMPLES
    assert first.wins + first.ties + first.losses == EQUITY_SAMPLES
    assert 0 <= first.equity_ppm <= 1_000_000


def test_equity_estimation_does_not_consume_action_rng() -> None:
    observation = _observation(
        street=Street.FLOP,
        hole_cards=(51, 50),
        board=(0, 14, 28),
    )
    action_rng = AgentRNG(
        master_seed=888,
        bot_id=0,
        hand_id=12,
        purpose="decision",
    )

    decision_index = action_rng.decision_index

    estimate_uniform_random_equity(
        observation,
        master_seed=action_rng.master_seed,
        bot_id=action_rng.bot_id,
        hand_id=action_rng.hand_id,
        decision_index=decision_index,
    )

    assert action_rng.decision_index == decision_index


def test_equity_depends_only_on_hero_visible_cards_and_seed_inputs() -> None:
    first = _observation(
        street=Street.RIVER,
        hole_cards=(51, 50),
        board=(49, 48, 47, 0, 14),
    )
    second = replace(
        first,
        stacks=(9_000, 11_000),
        pot=777,
        dealer=1,
        action_history=(Action.CALL,),
        legal_actions=(Action.FOLD, Action.CALL),
        public_action_history=(
            PublicActionEvent(
                actor=1,
                street=Street.RIVER,
                action=Action.BET_66,
                wager_increase=True,
            ),
        ),
    )

    result_a = estimate_uniform_random_equity(
        first,
        master_seed=5,
        bot_id=0,
        hand_id=3,
        decision_index=2,
    )
    result_b = estimate_uniform_random_equity(
        second,
        master_seed=5,
        bot_id=0,
        hand_id=3,
        decision_index=2,
    )

    assert result_a == result_b


def test_invalid_duplicate_visible_cards_are_rejected() -> None:
    observation = _observation(
        street=Street.FLOP,
        hole_cards=(0, 1),
        board=(1, 2, 3),
    )

    with pytest.raises(ValueError, match="duplicate"):
        joint_outcome_count(observation)


def test_board_length_must_match_street() -> None:
    observation = _observation(
        street=Street.TURN,
        hole_cards=(0, 1),
        board=(2, 3, 4),
    )

    with pytest.raises(ValueError, match="board length"):
        joint_outcome_count(observation)


def test_showdown_is_not_a_policy_equity_decision_street() -> None:
    observation = _observation(
        street=Street.SHOWDOWN,
        hole_cards=(0, 1),
        board=(2, 3, 4, 5, 6),
    )

    with pytest.raises(ValueError, match="Unsupported policy decision street"):
        joint_outcome_count(observation)
