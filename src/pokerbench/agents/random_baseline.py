"""Frozen public random baseline agent."""

from __future__ import annotations

from pokerbench.agents.base import Agent
from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import Action
from pokerbench.rng import AgentRNG


RANDOM_BASELINE_VERSION = "random-baseline-v1"


class RandomBaselineAgent(Agent):
    """Static baseline selecting uniformly from the supplied legal actions."""

    name = RANDOM_BASELINE_VERSION

    @property
    def is_adaptive(self) -> bool:
        return False

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        del observation

        if not isinstance(rng, AgentRNG):
            raise TypeError("rng must be an AgentRNG")

        if not isinstance(legal_actions, list):
            raise TypeError("legal_actions must be a list")

        if not legal_actions:
            raise ValueError("legal_actions must not be empty")

        if not all(isinstance(action, Action) for action in legal_actions):
            raise TypeError("legal_actions must contain only Action values")

        return rng.choice(legal_actions)
