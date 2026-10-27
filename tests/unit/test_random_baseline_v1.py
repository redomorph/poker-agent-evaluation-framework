from __future__ import annotations

import pytest

from pokerbench.agents.random_baseline import (
    RANDOM_BASELINE_VERSION,
    RandomBaselineAgent,
)
from pokerbench.engine.config import Action
from pokerbench.rng import AgentRNG


def test_random_baseline_identity_and_static_status() -> None:
    agent = RandomBaselineAgent()
    assert agent.name == RANDOM_BASELINE_VERSION
    assert agent.is_adaptive is False


def test_random_baseline_uses_exactly_one_agent_rng_event() -> None:
    agent = RandomBaselineAgent()
    rng = AgentRNG(master_seed=123, bot_id=0, hand_id=4)
    legal_actions = [Action.FOLD, Action.CALL, Action.ALL_IN]

    action = agent.act(object(), legal_actions, rng)  # type: ignore[arg-type]

    assert action in legal_actions
    assert rng.decision_index == 1


def test_random_baseline_is_deterministic_for_same_rng_identity() -> None:
    agent = RandomBaselineAgent()
    actions = [Action.FOLD, Action.CALL, Action.ALL_IN]
    first = AgentRNG(master_seed=77, bot_id=1, hand_id=9)
    second = AgentRNG(master_seed=77, bot_id=1, hand_id=9)

    assert agent.act(object(), actions, first) == agent.act(  # type: ignore[arg-type]
        object(), actions, second  # type: ignore[arg-type]
    )


def test_random_baseline_rejects_empty_legal_actions() -> None:
    agent = RandomBaselineAgent()
    rng = AgentRNG(master_seed=1, bot_id=0, hand_id=0)

    with pytest.raises(ValueError, match="legal_actions"):
        agent.act(object(), [], rng)  # type: ignore[arg-type]


def test_random_baseline_requires_agent_rng() -> None:
    agent = RandomBaselineAgent()

    with pytest.raises(TypeError, match="AgentRNG"):
        agent.act(object(), [Action.CHECK], object())  # type: ignore[arg-type]
