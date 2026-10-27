from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterator

from pokerbench.agents.equity import (
    estimate_uniform_random_equity,
)
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.strength_gates import (
    PolicyStrengthReference,
    REFERENCE_OPPORTUNITIES_PER_SCENARIO,
)
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
    GameConfig,
)
from pokerbench.engine.deal_schedule import DealSchedule, HandDeal
from pokerbench.engine.state import GameState, Street


CALIBRATION_ENVIRONMENT_VERSION = (
    "policy-calibration-environment-v1"
)
FIT_BASE_SEED = 5_100_001
VALIDATION_BASE_SEED = 5_100_002
CALIBRATION_DECISION_INDEX = 0


class CalibrationPhase(StrEnum):
    FIT = "fit"
    VALIDATION = "validation"


class CalibrationScenario(StrEnum):
    OPEN_BTN = "OPEN_BTN"
    THREE_BET_BB = "THREE_BET_BB"
    CBET_BTN = "CBET_BTN"
    FOLD_TO_CBET_BB = "FOLD_TO_CBET_BB"
    POSTFLOP_AGGRESSION_BTN = "POSTFLOP_AGGRESSION_BTN"


class CalibrationEnvironmentError(ValueError):
    """Raised when a frozen calibration probe cannot be constructed."""


@dataclass(frozen=True)
class CalibrationProbeObservation:
    phase: CalibrationPhase
    scenario: CalibrationScenario
    opportunity_index: int
    deal_seed: int
    hero_seat: int
    dealer: int
    decision_index: int
    observation: AgentObservation


@dataclass(frozen=True)
class _ScriptedAction:
    actor: int
    action: Action


@dataclass(frozen=True)
class _ScenarioDefinition:
    hero_seat: int
    dealer: int
    prefix: tuple[_ScriptedAction, ...]


_OPEN_BTN = _ScenarioDefinition(
    hero_seat=0,
    dealer=0,
    prefix=(),
)
_THREE_BET_BB = _ScenarioDefinition(
    hero_seat=1,
    dealer=0,
    prefix=(
        _ScriptedAction(0, Action.BET_200),
    ),
)
_CBET_BTN = _ScenarioDefinition(
    hero_seat=0,
    dealer=0,
    prefix=(
        _ScriptedAction(0, Action.BET_200),
        _ScriptedAction(1, Action.CALL),
        _ScriptedAction(1, Action.CHECK),
    ),
)
_FOLD_TO_CBET_BB = _ScenarioDefinition(
    hero_seat=1,
    dealer=0,
    prefix=(
        _ScriptedAction(0, Action.BET_200),
        _ScriptedAction(1, Action.CALL),
        _ScriptedAction(1, Action.CHECK),
        _ScriptedAction(0, Action.BET_66),
    ),
)
_POSTFLOP_AGGRESSION_BTN = _ScenarioDefinition(
    hero_seat=0,
    dealer=0,
    prefix=(
        _ScriptedAction(0, Action.CALL),
        _ScriptedAction(1, Action.CHECK),
        _ScriptedAction(1, Action.CHECK),
    ),
)


def _validate_phase(phase: CalibrationPhase) -> None:
    if not isinstance(phase, CalibrationPhase):
        raise TypeError("phase must be a CalibrationPhase")


def _validate_scenario(scenario: CalibrationScenario) -> None:
    if not isinstance(scenario, CalibrationScenario):
        raise TypeError("scenario must be a CalibrationScenario")


def _scenario_definition(
    scenario: CalibrationScenario,
) -> _ScenarioDefinition:
    _validate_scenario(scenario)

    definitions = {
        CalibrationScenario.OPEN_BTN: _OPEN_BTN,
        CalibrationScenario.THREE_BET_BB: _THREE_BET_BB,
        CalibrationScenario.CBET_BTN: _CBET_BTN,
        CalibrationScenario.FOLD_TO_CBET_BB: _FOLD_TO_CBET_BB,
        CalibrationScenario.POSTFLOP_AGGRESSION_BTN: (
            _POSTFLOP_AGGRESSION_BTN
        ),
    }
    return definitions[scenario]


def scenario_hero_seat(
    scenario: CalibrationScenario,
) -> int:
    return _scenario_definition(scenario).hero_seat


def _base_seed(phase: CalibrationPhase) -> int:
    _validate_phase(phase)

    if phase is CalibrationPhase.FIT:
        return FIT_BASE_SEED

    return VALIDATION_BASE_SEED


def scenario_deal_seed(
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
) -> int:
    _validate_scenario(scenario)
    base_seed = _base_seed(phase)
    material = (
        f"{CALIBRATION_ENVIRONMENT_VERSION}:"
        f"{base_seed}:"
        f"{scenario.value}"
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )


def _validate_game_config(game_config: GameConfig) -> None:
    if not isinstance(game_config, GameConfig):
        raise TypeError("game_config must be a GameConfig")

    if game_config != BASELINE_GAME_CONFIG:
        raise ValueError(
            "policy-calibration-environment-v1 requires "
            "BASELINE_GAME_CONFIG"
        )


def _validate_opportunity_index(opportunity_index: int) -> None:
    if (
        not isinstance(opportunity_index, int)
        or isinstance(opportunity_index, bool)
    ):
        raise TypeError("opportunity_index must be an integer")

    if not (
        0
        <= opportunity_index
        < REFERENCE_OPPORTUNITIES_PER_SCENARIO
    ):
        raise ValueError(
            "opportunity_index must be within [0, 4999]"
        )


def _apply_scripted_prefix(
    state: GameState,
    definition: _ScenarioDefinition,
    scenario: CalibrationScenario,
) -> GameState:
    current = state

    for scripted in definition.prefix:
        if current.is_terminal:
            raise CalibrationEnvironmentError(
                "Calibration probe terminated before scripted prefix "
                f"completed: {scenario.value}"
            )

        if current.current_player != scripted.actor:
            raise CalibrationEnvironmentError(
                "Unexpected actor while constructing calibration probe: "
                f"scenario={scenario.value}, "
                f"expected={scripted.actor}, "
                f"actual={current.current_player}"
            )

        if scripted.action not in current.legal_actions():
            raise CalibrationEnvironmentError(
                "Scripted calibration action is not legal: "
                f"scenario={scenario.value}, action={scripted.action.value}"
            )

        current = current.apply_action(scripted.action)

    return current


def _event_signature(
    state: GameState,
) -> tuple[tuple[int, Street, Action, bool], ...]:
    return tuple(
        (
            event.actor,
            event.street,
            event.action,
            event.wager_increase,
        )
        for event in state.public_action_history
    )


def _validate_probe_state(
    state: GameState,
    scenario: CalibrationScenario,
    definition: _ScenarioDefinition,
) -> None:
    if state.is_terminal:
        raise CalibrationEnvironmentError(
            f"Calibration probe is terminal: {scenario.value}"
        )

    if state.current_player != definition.hero_seat:
        raise CalibrationEnvironmentError(
            "Calibration probe does not end on hero decision: "
            f"scenario={scenario.value}, "
            f"hero={definition.hero_seat}, "
            f"current_player={state.current_player}"
        )

    expected: dict[
        CalibrationScenario,
        tuple[tuple[int, Street, Action, bool], ...],
    ] = {
        CalibrationScenario.OPEN_BTN: (),
        CalibrationScenario.THREE_BET_BB: (
            (0, Street.PREFLOP, Action.BET_200, True),
        ),
        CalibrationScenario.CBET_BTN: (
            (0, Street.PREFLOP, Action.BET_200, True),
            (1, Street.PREFLOP, Action.CALL, False),
            (1, Street.FLOP, Action.CHECK, False),
        ),
        CalibrationScenario.FOLD_TO_CBET_BB: (
            (0, Street.PREFLOP, Action.BET_200, True),
            (1, Street.PREFLOP, Action.CALL, False),
            (1, Street.FLOP, Action.CHECK, False),
            (0, Street.FLOP, Action.BET_66, True),
        ),
        CalibrationScenario.POSTFLOP_AGGRESSION_BTN: (
            (0, Street.PREFLOP, Action.CALL, False),
            (1, Street.PREFLOP, Action.CHECK, False),
            (1, Street.FLOP, Action.CHECK, False),
        ),
    }

    if _event_signature(state) != expected[scenario]:
        raise CalibrationEnvironmentError(
            "Calibration probe public trace does not match frozen "
            f"scenario: {scenario.value}"
        )

    expected_street = (
        Street.PREFLOP
        if scenario in (
            CalibrationScenario.OPEN_BTN,
            CalibrationScenario.THREE_BET_BB,
        )
        else Street.FLOP
    )

    if state.street is not expected_street:
        raise CalibrationEnvironmentError(
            "Calibration probe ended on unexpected street: "
            f"scenario={scenario.value}, street={state.street.name}"
        )


def _probe_from_deal(
    *,
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    opportunity_index: int,
    deal_seed: int,
    deal: HandDeal,
    game_config: GameConfig,
) -> CalibrationProbeObservation:
    definition = _scenario_definition(scenario)
    state = GameState.new_hand(
        game_config,
        (deal[0], deal[1]),
        deal[2],
        dealer=definition.dealer,
    )
    state = _apply_scripted_prefix(
        state,
        definition,
        scenario,
    )
    _validate_probe_state(
        state,
        scenario,
        definition,
    )

    return CalibrationProbeObservation(
        phase=phase,
        scenario=scenario,
        opportunity_index=opportunity_index,
        deal_seed=deal_seed,
        hero_seat=definition.hero_seat,
        dealer=definition.dealer,
        decision_index=CALIBRATION_DECISION_INDEX,
        observation=state.observation(definition.hero_seat),
    )


def calibration_probe_observation(
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    opportunity_index: int,
    *,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> CalibrationProbeObservation:
    _validate_phase(phase)
    _validate_scenario(scenario)
    _validate_opportunity_index(opportunity_index)
    _validate_game_config(game_config)

    deal_seed = scenario_deal_seed(
        phase,
        scenario,
    )
    deal = DealSchedule.hand_for_index(
        deal_seed,
        opportunity_index,
    )

    return _probe_from_deal(
        phase=phase,
        scenario=scenario,
        opportunity_index=opportunity_index,
        deal_seed=deal_seed,
        deal=deal,
        game_config=game_config,
    )


def iter_calibration_probe_observations(
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    *,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> Iterator[CalibrationProbeObservation]:
    _validate_phase(phase)
    _validate_scenario(scenario)
    _validate_game_config(game_config)

    deal_seed = scenario_deal_seed(
        phase,
        scenario,
    )
    schedule = DealSchedule(
        deal_seed,
        REFERENCE_OPPORTUNITIES_PER_SCENARIO,
    )

    for opportunity_index, deal in enumerate(schedule):
        yield _probe_from_deal(
            phase=phase,
            scenario=scenario,
            opportunity_index=opportunity_index,
            deal_seed=deal_seed,
            deal=deal,
            game_config=game_config,
        )


def scenario_strengths(
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    *,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> tuple[int, ...]:
    strengths: list[int] = []

    for probe in iter_calibration_probe_observations(
        phase,
        scenario,
        game_config=game_config,
    ):
        result = estimate_uniform_random_equity(
            probe.observation,
            master_seed=probe.deal_seed,
            bot_id=probe.hero_seat,
            hand_id=probe.opportunity_index,
            decision_index=probe.decision_index,
        )
        strengths.append(result.equity_ppm)

    return tuple(strengths)


def generate_policy_strength_reference(
    *,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> PolicyStrengthReference:
    _validate_game_config(game_config)

    return PolicyStrengthReference(
        open_btn=scenario_strengths(
            CalibrationPhase.FIT,
            CalibrationScenario.OPEN_BTN,
            game_config=game_config,
        ),
        three_bet_bb=scenario_strengths(
            CalibrationPhase.FIT,
            CalibrationScenario.THREE_BET_BB,
            game_config=game_config,
        ),
        cbet_btn=scenario_strengths(
            CalibrationPhase.FIT,
            CalibrationScenario.CBET_BTN,
            game_config=game_config,
        ),
        fold_to_cbet_bb=scenario_strengths(
            CalibrationPhase.FIT,
            CalibrationScenario.FOLD_TO_CBET_BB,
            game_config=game_config,
        ),
        postflop_aggression_btn=scenario_strengths(
            CalibrationPhase.FIT,
            CalibrationScenario.POSTFLOP_AGGRESSION_BTN,
            game_config=game_config,
        ),
    )
