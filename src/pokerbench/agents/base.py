# pokerbench/agents/base.py
"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import List

from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import Action
from pokerbench.rng import AgentRNG


class Agent(ABC):
    """Abstract base class for all poker agents."""

    @abstractmethod
    def act(self, observation: AgentObservation, legal_actions: List[Action], rng: AgentRNG) -> Action:
        """
        Choose an action given the current observation and legal actions.

        Args:
            observation: Information observable by this player.
            legal_actions: List of legal actions.
            rng: Event-indexed random number generator.

        Returns:
            The chosen action (must be in legal_actions).
        """
        ...

    def reset(self) -> None:
        """
        Reset any internal state (e.g., for adaptive agents) before a new run.

        The default implementation does nothing.
        """
        pass

    @property
    def is_adaptive(self) -> bool:
        """Return True if the agent maintains state across hands; default False."""
        return False