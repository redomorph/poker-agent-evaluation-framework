from __future__ import annotations

from dataclasses import dataclass

from pokerbench.engine.config import Action
from pokerbench.engine.public_action import PublicActionEvent
from pokerbench.engine.state import Street


AGENT_OBSERVATION_VERSION = "agent-observation-v2"


@dataclass(frozen=True)
class AgentObservation:
    """Information available to one player at a specific game state."""

    player_id: int
    hole_cards: tuple[int, int]
    board: tuple[int, ...]
    stacks: tuple[int, int]
    current_bets: tuple[int, int]
    pot: int
    current_player: int
    street: Street
    action_history: tuple[Action, ...]
    legal_actions: tuple[Action, ...]

    # Additive v2 public information. Defaults preserve direct construction
    # compatibility for existing tests and third-party agents.
    dealer: int = 0
    public_action_history: tuple[
        PublicActionEvent,
        ...,
    ] = ()
