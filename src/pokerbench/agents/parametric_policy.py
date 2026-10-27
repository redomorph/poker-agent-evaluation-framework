from __future__ import annotations

from typing import Sequence

from pokerbench.agents.base import Agent
from pokerbench.agents.equity import estimate_uniform_random_equity
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.strength_gates import (
    PPM_SCALE,
    PolicyCalibrationMap,
    StrengthGate,
)
from pokerbench.engine.config import Action
from pokerbench.engine.public_action import PublicActionEvent
from pokerbench.engine.state import Street
from pokerbench.rng import AgentRNG


ACTION_SIZING_VERSION = "strength-sizing-v1"

_AGGRESSIVE_RANKS: tuple[Action, ...] = (
    Action.BET_33,
    Action.BET_66,
    Action.BET_100,
    Action.BET_200,
    Action.ALL_IN,
)
class ParametricPolicyDecisionError(ValueError):
    """Raised when parametric-policy-v1 cannot return a legal action."""


def _validate_observation(
    observation: AgentObservation,
) -> None:
    if not isinstance(observation, AgentObservation):
        raise TypeError("observation must be an AgentObservation")

    if observation.player_id not in (0, 1):
        raise ParametricPolicyDecisionError(
            "observation.player_id must be 0 or 1"
        )

    if observation.current_player != observation.player_id:
        raise ParametricPolicyDecisionError(
            "parametric-policy-v1 may act only for the current player"
        )

    if observation.street not in (
        Street.PREFLOP,
        Street.FLOP,
        Street.TURN,
        Street.RIVER,
    ):
        raise ParametricPolicyDecisionError(
            "parametric-policy-v1 cannot act outside a betting street"
        )


def _validate_legal_actions(
    legal_actions: Sequence[Action],
) -> tuple[Action, ...]:
    if isinstance(legal_actions, (str, bytes)):
        raise TypeError("legal_actions must be a sequence of Action values")

    actions = tuple(legal_actions)

    if not actions:
        raise ParametricPolicyDecisionError(
            "legal_actions must not be empty"
        )

    for action in actions:
        if not isinstance(action, Action):
            raise TypeError("legal_actions must contain only Action values")

    return actions


def _opponent(observation: AgentObservation) -> int:
    return 1 - observation.player_id


def _faces_wager(observation: AgentObservation) -> bool:
    hero = observation.player_id
    opponent = _opponent(observation)
    return (
        observation.current_bets[opponent]
        > observation.current_bets[hero]
    )


def _all_in_increases_wager(
    observation: AgentObservation,
) -> bool:
    hero = observation.player_id
    opponent = _opponent(observation)
    return (
        observation.current_bets[hero]
        + observation.stacks[hero]
        > observation.current_bets[opponent]
    )


def _aggressive_legal_actions(
    observation: AgentObservation,
    legal_actions: Sequence[Action],
) -> tuple[Action, ...]:
    legal = set(legal_actions)
    actions: list[Action] = [
        action
        for action in _AGGRESSIVE_RANKS[:-1]
        if action in legal
    ]

    if (
        Action.ALL_IN in legal
        and _all_in_increases_wager(observation)
    ):
        actions.append(Action.ALL_IN)

    return tuple(actions)


def _passive_call_equivalent(
    observation: AgentObservation,
    legal_actions: Sequence[Action],
) -> Action | None:
    if Action.CALL in legal_actions:
        return Action.CALL

    if (
        Action.ALL_IN in legal_actions
        and not _all_in_increases_wager(observation)
    ):
        return Action.ALL_IN

    return None


def _decline_action(
    observation: AgentObservation,
    legal_actions: Sequence[Action],
) -> Action:
    if Action.CHECK in legal_actions:
        return Action.CHECK

    if Action.FOLD in legal_actions:
        return Action.FOLD

    passive = _passive_call_equivalent(
        observation,
        legal_actions,
    )

    if passive is not None:
        return passive

    raise ParametricPolicyDecisionError(
        "No legal decline/passive action exists"
    )


def _preferred_aggressive_action(
    equity_ppm: int,
) -> Action:
    if equity_ppm < 600_000:
        return Action.BET_33
    if equity_ppm < 750_000:
        return Action.BET_66
    if equity_ppm < 900_000:
        return Action.BET_100
    if equity_ppm < 980_000:
        return Action.BET_200
    return Action.ALL_IN


def _choose_aggressive_action(
    observation: AgentObservation,
    legal_actions: Sequence[Action],
    *,
    equity_ppm: int,
) -> Action | None:
    aggressive = _aggressive_legal_actions(
        observation,
        legal_actions,
    )

    if not aggressive:
        return None

    preferred = _preferred_aggressive_action(
        equity_ppm
    )
    preferred_rank = _AGGRESSIVE_RANKS.index(
        preferred
    )

    return min(
        aggressive,
        key=lambda action: (
            abs(
                _AGGRESSIVE_RANKS.index(action)
                - preferred_rank
            ),
            _AGGRESSIVE_RANKS.index(action),
        ),
    )


def _gate_passes(
    gate: StrengthGate,
    *,
    equity_ppm: int,
    decision_draw_ppm: int,
) -> bool:
    return gate.passes(
        equity_ppm=equity_ppm,
        decision_draw_ppm=decision_draw_ppm,
    )


def _preflop_increases(
    observation: AgentObservation,
) -> tuple[PublicActionEvent, ...]:
    return tuple(
        event
        for event in observation.public_action_history
        if (
            event.street is Street.PREFLOP
            and event.wager_increase
        )
    )


def _qualifying_three_bet_opportunity(
    observation: AgentObservation,
    legal_actions: Sequence[Action],
) -> bool:
    if observation.street is not Street.PREFLOP:
        return False

    if not _faces_wager(observation):
        return False

    increases = _preflop_increases(observation)

    if len(increases) != 1:
        return False

    if increases[0].actor != _opponent(observation):
        return False

    return bool(
        _aggressive_legal_actions(
            observation,
            legal_actions,
        )
    )


def _final_preflop_aggressor(
    observation: AgentObservation,
) -> int | None:
    for event in reversed(
        observation.public_action_history
    ):
        if (
            event.street is Street.PREFLOP
            and event.wager_increase
        ):
            return event.actor

    return None


def _qualifying_cbet_opportunity(
    observation: AgentObservation,
    legal_actions: Sequence[Action],
) -> bool:
    if observation.street is not Street.FLOP:
        return False

    hero = observation.player_id
    opponent = _opponent(observation)

    if _final_preflop_aggressor(observation) != hero:
        return False

    flop_events = tuple(
        event
        for event in observation.public_action_history
        if event.street is Street.FLOP
    )

    if any(
        event.actor == hero
        for event in flop_events
    ):
        return False

    if any(
        event.actor == opponent
        and event.wager_increase
        for event in flop_events
    ):
        return False

    return bool(
        _aggressive_legal_actions(
            observation,
            legal_actions,
        )
    )


def _directly_faces_qualifying_cbet(
    observation: AgentObservation,
) -> bool:
    if observation.street is not Street.FLOP:
        return False

    if not _faces_wager(observation):
        return False

    hero = observation.player_id
    opponent = _opponent(observation)

    if _final_preflop_aggressor(observation) != opponent:
        return False

    flop_events_with_index = tuple(
        (index, event)
        for index, event in enumerate(
            observation.public_action_history
        )
        if event.street is Street.FLOP
    )
    opponent_events = tuple(
        (index, event)
        for index, event in flop_events_with_index
        if event.actor == opponent
    )

    if not opponent_events:
        return False

    cbet_index, cbet_event = opponent_events[0]

    if not cbet_event.wager_increase:
        return False

    if any(
        event.street is Street.FLOP
        and event.actor == hero
        and event.wager_increase
        and index < cbet_index
        for index, event in enumerate(
            observation.public_action_history
        )
    ):
        return False

    return (
        cbet_index
        == len(observation.public_action_history) - 1
    )


def _pot_odds_ppm(
    observation: AgentObservation,
) -> int:
    if not _faces_wager(observation):
        raise ParametricPolicyDecisionError(
            "Pot odds are defined only when facing a wager"
        )

    hero = observation.player_id
    opponent = _opponent(observation)
    call_cost = min(
        (
            observation.current_bets[opponent]
            - observation.current_bets[hero]
        ),
        observation.stacks[hero],
    )

    if call_cost <= 0:
        raise ParametricPolicyDecisionError(
            "Call cost must be positive when facing a wager"
        )

    denominator = observation.pot + call_cost

    if denominator <= 0:
        raise ParametricPolicyDecisionError(
            "Pot plus call cost must be positive"
        )

    return (
        call_cost * PPM_SCALE
        + denominator // 2
    ) // denominator


def _require_fold(
    observation: AgentObservation,
    legal_actions: Sequence[Action],
) -> Action:
    if Action.FOLD in legal_actions:
        return Action.FOLD

    return _decline_action(
        observation,
        legal_actions,
    )


def _preflop_single_raise_without_reraise(
    observation: AgentObservation,
    legal_actions: Sequence[Action],
) -> bool:
    if observation.street is not Street.PREFLOP:
        return False

    if not _faces_wager(observation):
        return False

    increases = _preflop_increases(observation)

    if len(increases) != 1:
        return False

    if increases[0].actor != _opponent(observation):
        return False

    return not bool(
        _aggressive_legal_actions(
            observation,
            legal_actions,
        )
    )


def _continue_single_raise_passively(
    calibration_map: PolicyCalibrationMap,
    observation: AgentObservation,
    legal_actions: Sequence[Action],
    *,
    equity_ppm: int,
    decision_draw_ppm: int,
) -> Action:
    passive = _passive_call_equivalent(
        observation,
        legal_actions,
    )

    if (
        passive is not None
        and _gate_passes(
            calibration_map.vpip_gate,
            equity_ppm=equity_ppm,
            decision_draw_ppm=decision_draw_ppm,
        )
        and equity_ppm >= _pot_odds_ppm(observation)
    ):
        return passive

    return _require_fold(
        observation,
        legal_actions,
    )


def _decide_preflop(
    calibration_map: PolicyCalibrationMap,
    observation: AgentObservation,
    legal_actions: Sequence[Action],
    *,
    equity_ppm: int,
    decision_draw_ppm: int,
) -> Action:
    aggressive = _choose_aggressive_action(
        observation,
        legal_actions,
        equity_ppm=equity_ppm,
    )
    increases = _preflop_increases(observation)

    if _qualifying_three_bet_opportunity(
        observation,
        legal_actions,
    ):
        if (
            aggressive is not None
            and _gate_passes(
                calibration_map.three_bet_gate,
                equity_ppm=equity_ppm,
                decision_draw_ppm=decision_draw_ppm,
            )
        ):
            return aggressive

        return _continue_single_raise_passively(
            calibration_map,
            observation,
            legal_actions,
            equity_ppm=equity_ppm,
            decision_draw_ppm=decision_draw_ppm,
        )

    if _preflop_single_raise_without_reraise(
        observation,
        legal_actions,
    ):
        return _continue_single_raise_passively(
            calibration_map,
            observation,
            legal_actions,
            equity_ppm=equity_ppm,
            decision_draw_ppm=decision_draw_ppm,
        )

    if not increases:
        if (
            aggressive is not None
            and _gate_passes(
                calibration_map.pfr_gate,
                equity_ppm=equity_ppm,
                decision_draw_ppm=decision_draw_ppm,
            )
        ):
            return aggressive

        if (
            Action.CALL in legal_actions
            and _gate_passes(
                calibration_map.vpip_gate,
                equity_ppm=equity_ppm,
                decision_draw_ppm=decision_draw_ppm,
            )
        ):
            return Action.CALL

        if Action.CHECK in legal_actions:
            return Action.CHECK

        return _require_fold(
            observation,
            legal_actions,
        )

    if len(increases) >= 2:
        return _continue_single_raise_passively(
            calibration_map,
            observation,
            legal_actions,
            equity_ppm=equity_ppm,
            decision_draw_ppm=decision_draw_ppm,
        )

    raise ParametricPolicyDecisionError(
        "Unclassified preflop decision state"
    )


def _decide_postflop(
    calibration_map: PolicyCalibrationMap,
    observation: AgentObservation,
    legal_actions: Sequence[Action],
    *,
    equity_ppm: int,
    decision_draw_ppm: int,
) -> Action:
    aggressive = _choose_aggressive_action(
        observation,
        legal_actions,
        equity_ppm=equity_ppm,
    )

    if _directly_faces_qualifying_cbet(
        observation
    ):
        if _gate_passes(
            calibration_map.fold_to_cbet_gate,
            equity_ppm=equity_ppm,
            decision_draw_ppm=decision_draw_ppm,
        ):
            return _require_fold(
                observation,
                legal_actions,
            )

        if (
            calibration_map.aggression_gate is not None
            and aggressive is not None
            and _gate_passes(
                calibration_map.aggression_gate,
                equity_ppm=equity_ppm,
                decision_draw_ppm=decision_draw_ppm,
            )
        ):
            return aggressive

        passive = _passive_call_equivalent(
            observation,
            legal_actions,
        )

        if passive is not None:
            return passive

        return _decline_action(
            observation,
            legal_actions,
        )

    if _qualifying_cbet_opportunity(
        observation,
        legal_actions,
    ):
        if (
            aggressive is not None
            and _gate_passes(
                calibration_map.cbet_gate,
                equity_ppm=equity_ppm,
                decision_draw_ppm=decision_draw_ppm,
            )
        ):
            return aggressive

        return _decline_action(
            observation,
            legal_actions,
        )

    if _faces_wager(observation):
        if equity_ppm < _pot_odds_ppm(observation):
            return _require_fold(
                observation,
                legal_actions,
            )

        if (
            calibration_map.aggression_gate is not None
            and aggressive is not None
            and _gate_passes(
                calibration_map.aggression_gate,
                equity_ppm=equity_ppm,
                decision_draw_ppm=decision_draw_ppm,
            )
        ):
            return aggressive

        passive = _passive_call_equivalent(
            observation,
            legal_actions,
        )

        if passive is not None:
            return passive

        return _decline_action(
            observation,
            legal_actions,
        )

    if (
        calibration_map.aggression_gate is not None
        and aggressive is not None
        and _gate_passes(
            calibration_map.aggression_gate,
            equity_ppm=equity_ppm,
            decision_draw_ppm=decision_draw_ppm,
        )
    ):
        return aggressive

    return _decline_action(
        observation,
        legal_actions,
    )


def decide_parametric_action(
    calibration_map: PolicyCalibrationMap,
    observation: AgentObservation,
    legal_actions: Sequence[Action],
    *,
    equity_ppm: int,
    decision_draw_ppm: int,
) -> Action:
    """Return the frozen parametric-policy-v1 action for supplied inputs.

    This pure decision entry point is used both by ``ParametricPolicyAgent``
    and by the frozen calibration-acceptance harness. Equity computation and
    action-RNG consumption remain outside this function.
    """
    if not isinstance(calibration_map, PolicyCalibrationMap):
        raise TypeError(
            "calibration_map must be a PolicyCalibrationMap"
        )

    _validate_observation(observation)
    actions = _validate_legal_actions(legal_actions)

    if (
        not isinstance(equity_ppm, int)
        or isinstance(equity_ppm, bool)
    ):
        raise TypeError("equity_ppm must be an integer")

    if not 0 <= equity_ppm <= PPM_SCALE:
        raise ValueError("equity_ppm must be within [0, 1000000]")

    if (
        not isinstance(decision_draw_ppm, int)
        or isinstance(decision_draw_ppm, bool)
    ):
        raise TypeError("decision_draw_ppm must be an integer")

    if not 0 <= decision_draw_ppm < PPM_SCALE:
        raise ValueError(
            "decision_draw_ppm must be within [0, 999999]"
        )

    if observation.street is Street.PREFLOP:
        action = _decide_preflop(
            calibration_map,
            observation,
            actions,
            equity_ppm=equity_ppm,
            decision_draw_ppm=decision_draw_ppm,
        )
    else:
        action = _decide_postflop(
            calibration_map,
            observation,
            actions,
            equity_ppm=equity_ppm,
            decision_draw_ppm=decision_draw_ppm,
        )

    if action not in actions:
        raise ParametricPolicyDecisionError(
            "parametric-policy-v1 selected an illegal action"
        )

    return action


class ParametricPolicyAgent(Agent):
    """Static opponent implementing the frozen parametric-policy-v1 map."""

    def __init__(
        self,
        calibration_map: PolicyCalibrationMap,
    ) -> None:
        if not isinstance(
            calibration_map,
            PolicyCalibrationMap,
        ):
            raise TypeError(
                "calibration_map must be a PolicyCalibrationMap"
            )

        self._calibration_map = calibration_map

    @property
    def calibration_map(self) -> PolicyCalibrationMap:
        return self._calibration_map

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
        actions = _validate_legal_actions(
            legal_actions
        )

        if not isinstance(rng, AgentRNG):
            raise TypeError("rng must be an AgentRNG")

        decision_index = rng.decision_index
        decision_draw_ppm = rng.randint(
            0,
            PPM_SCALE - 1,
        )
        equity = estimate_uniform_random_equity(
            observation,
            master_seed=rng.master_seed,
            bot_id=rng.bot_id,
            hand_id=rng.hand_id,
            decision_index=decision_index,
        )

        return decide_parametric_action(
            self._calibration_map,
            observation,
            actions,
            equity_ppm=equity.equity_ppm,
            decision_draw_ppm=decision_draw_ppm,
        )
