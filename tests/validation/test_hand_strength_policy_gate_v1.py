from __future__ import annotations

from pokerbench.agents.hand_strength_baseline import HandStrengthBaselineAgent
from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import Action
from pokerbench.engine.state import Street
from pokerbench.rng import AgentRNG


def test_fully_specified_hand_strength_policy_situation_selects_expected_action() -> None:
    observation = AgentObservation(
        player_id=0,
        hole_cards=(12, 11),  # AC KC
        board=(10, 9, 8, 13, 27),  # QC JC TC 2D 3H
        stacks=(9_000, 9_000),
        current_bets=(0, 1_000),
        pot=1_000,
        current_player=0,
        street=Street.RIVER,
        action_history=(),
        legal_actions=(Action.FOLD, Action.CALL),
        dealer=0,
        public_action_history=(),
    )

    action = HandStrengthBaselineAgent().act(
        observation,
        [Action.FOLD, Action.CALL],
        AgentRNG(master_seed=123, bot_id=0, hand_id=0),
    )

    assert action is Action.CALL
