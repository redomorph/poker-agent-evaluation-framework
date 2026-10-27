"""Deterministic equity/pot-odds baseline for Phase-1 measurement."""

from __future__ import annotations

from collections.abc import Sequence

from pokerbench.agents.base import Agent
from pokerbench.agents.equity import estimate_uniform_random_equity
from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import Action
from pokerbench.engine.state import Street
from pokerbench.rng import AgentRNG


HAND_STRENGTH_BASELINE_VERSION = "hand-strength-baseline-v1"
HAND_STRENGTH_POT_ODDS_VERSION = "hand-strength-pot-odds-v1"
_PPM_SCALE = 1_000_000


class HandStrengthBaselineDecisionError(ValueError):
    """Raised when the frozen baseline cannot select its passive action."""


def _validate_observation(observation: AgentObservation) -> None:
    if not isinstance(observation, AgentObservation):
        raise TypeError("observation must be an AgentObservation")
    if observation.player_id not in (0, 1):
        raise HandStrengthBaselineDecisionError(
            "observation.player_id must be 0 or 1"
        )
    if observation.current_player != observation.player_id:
        raise HandStrengthBaselineDecisionError(
            "hand-strength-baseline-v1 may act only for the current player"
        )
    if observation.street not in (
        Street.PREFLOP,
        Street.FLOP,
        Street.TURN,
        Street.RIVER,
    ):
        raise HandStrengthBaselineDecisionError(
            "hand-strength-baseline-v1 cannot act outside a betting street"
        )


def _validate_legal_actions(legal_actions: Sequence[Action]) -> tuple[Action, ...]:
    if isinstance(legal_actions, (str, bytes)):
        raise TypeError("legal_actions must be a sequence of Action values")
    actions = tuple(legal_actions)
    if not actions:
        raise HandStrengthBaselineDecisionError(
            "legal_actions must not be empty"
        )
    if not all(isinstance(action, Action) for action in actions):
        raise TypeError("legal_actions must contain only Action values")
    return actions


def _faces_wager(observation: AgentObservation) -> bool:
    hero = observation.player_id
    opponent = 1 - hero
    return observation.current_bets[opponent] > observation.current_bets[hero]


def _pot_odds_ppm(observation: AgentObservation) -> int:
    hero = observation.player_id
    opponent = 1 - hero
    call_cost = min(
        observation.current_bets[opponent] - observation.current_bets[hero],
        observation.stacks[hero],
    )
    if call_cost <= 0:
        raise HandStrengthBaselineDecisionError(
            "pot odds require a positive call cost"
        )
    denominator = observation.pot + call_cost
    if denominator <= 0:
        raise HandStrengthBaselineDecisionError(
            "pot plus call cost must be positive"
        )
    return (
        call_cost * _PPM_SCALE
        + denominator // 2
    ) // denominator


class HandStrengthBaselineAgent(Agent):
    """Static non-aggressive baseline driven only by equity and pot odds."""

    name = HAND_STRENGTH_BASELINE_VERSION

    @property
    def is_adaptive(self) -> bool:
        return False

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        _validate_observation(observation)
        actions = _validate_legal_actions(legal_actions)
        if not isinstance(rng, AgentRNG):
            raise TypeError("rng must be an AgentRNG")

        if not _faces_wager(observation):
            if Action.CHECK in actions:
                return Action.CHECK
            raise HandStrengthBaselineDecisionError(
                "CHECK must be legal when no wager is faced; "
                "v1 does not invent an aggressive substitute"
            )

        decision_index = rng.decision_index
        equity = estimate_uniform_random_equity(
            observation,
            master_seed=rng.master_seed,
            bot_id=rng.bot_id,
            hand_id=rng.hand_id,
            decision_index=decision_index,
        )
        pot_odds_ppm = _pot_odds_ppm(observation)

        if equity.equity_ppm >= pot_odds_ppm and Action.CALL in actions:
            return Action.CALL
        if Action.FOLD in actions:
            return Action.FOLD

        raise HandStrengthBaselineDecisionError(
            "Frozen passive action is unavailable; v1 does not substitute aggression"
        )
