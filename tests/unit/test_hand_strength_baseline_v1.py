from __future__ import annotations

import pytest

from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import Action
from pokerbench.engine.state import Street
from pokerbench.rng import AgentRNG

from pokerbench.agents.hand_strength_baseline import (
    HAND_STRENGTH_BASELINE_VERSION,
    HandStrengthBaselineAgent,
    HandStrengthBaselineDecisionError,
)


def _river_observation(
    *,
    hole_cards: tuple[int, int],
    board: tuple[int, int, int, int, int],
    pot: int,
    current_bets: tuple[int, int],
    legal_actions: tuple[Action, ...],
) -> AgentObservation:
    return AgentObservation(
        player_id=0,
        hole_cards=hole_cards,
        board=board,
        stacks=(9_000, 9_000),
        current_bets=current_bets,
        pot=pot,
        current_player=0,
        street=Street.RIVER,
        action_history=(),
        legal_actions=legal_actions,
        dealer=0,
        public_action_history=(),
    )


def test_hand_strength_baseline_identity_and_static_contract() -> None:
    agent = HandStrengthBaselineAgent()

    assert HAND_STRENGTH_BASELINE_VERSION == "hand-strength-baseline-v1"
    assert agent.name == HAND_STRENGTH_BASELINE_VERSION
    assert not agent.is_adaptive


def test_unbeatable_river_hand_calls_when_equity_exceeds_pot_odds() -> None:
    # AC KC on QC JC TC 2D 3H is an unbeatable royal flush.
    observation = _river_observation(
        hole_cards=(12, 11),
        board=(10, 9, 8, 13, 27),
        pot=1_000,
        current_bets=(0, 1_000),
        legal_actions=(Action.FOLD, Action.CALL),
    )
    rng = AgentRNG(master_seed=7, bot_id=0, hand_id=0)

    action = HandStrengthBaselineAgent().act(
        observation,
        list(observation.legal_actions),
        rng,
    )

    assert action is Action.CALL
    assert action in observation.legal_actions
    assert rng.decision_index == 0


def test_weak_river_hand_folds_when_equity_is_below_pot_odds() -> None:
    observation = _river_observation(
        hole_cards=(0, 14),  # 2C 3D
        board=(31, 46, 9, 23, 37),
        pot=1_000,
        current_bets=(0, 1_000),
        legal_actions=(Action.FOLD, Action.CALL),
    )
    rng = AgentRNG(master_seed=7, bot_id=0, hand_id=0)

    action = HandStrengthBaselineAgent().act(
        observation,
        list(observation.legal_actions),
        rng,
    )

    assert action is Action.FOLD
    assert action in observation.legal_actions
    assert rng.decision_index == 0


def test_no_wager_is_checked_without_consuming_action_rng() -> None:
    observation = _river_observation(
        hole_cards=(0, 14),
        board=(31, 46, 9, 23, 37),
        pot=1_000,
        current_bets=(0, 0),
        legal_actions=(Action.CHECK, Action.BET_100),
    )
    rng = AgentRNG(master_seed=99, bot_id=0, hand_id=3)

    action = HandStrengthBaselineAgent().act(
        observation,
        list(observation.legal_actions),
        rng,
    )

    assert action is Action.CHECK
    assert rng.decision_index == 0


def test_inconsistent_no_wager_state_does_not_invent_aggression() -> None:
    observation = _river_observation(
        hole_cards=(0, 14),
        board=(31, 46, 9, 23, 37),
        pot=1_000,
        current_bets=(0, 0),
        legal_actions=(Action.BET_100,),
    )

    with pytest.raises(
        HandStrengthBaselineDecisionError,
        match="CHECK",
    ):
        HandStrengthBaselineAgent().act(
            observation,
            list(observation.legal_actions),
            AgentRNG(master_seed=1, bot_id=0, hand_id=0),
        )


def test_agent_rejects_observation_for_nonacting_player() -> None:
    observation = _river_observation(
        hole_cards=(0, 14),
        board=(31, 46, 9, 23, 37),
        pot=1_000,
        current_bets=(0, 0),
        legal_actions=(Action.CHECK,),
    )
    wrong_player = AgentObservation(
        **{
            **observation.__dict__,
            "current_player": 1,
        }
    )

    with pytest.raises(
        HandStrengthBaselineDecisionError,
        match="current player",
    ):
        HandStrengthBaselineAgent().act(
            wrong_player,
            [Action.CHECK],
            AgentRNG(master_seed=1, bot_id=0, hand_id=0),
        )
