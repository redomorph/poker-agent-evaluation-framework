"""Frozen Gate-6 policy-calibration acceptance for parametric-policy-v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from pokerbench.agents.calibration_environment import (
    CALIBRATION_ENVIRONMENT_VERSION,
    FIT_BASE_SEED,
    VALIDATION_BASE_SEED,
    CalibrationPhase,
    CalibrationScenario,
    generate_policy_strength_reference,
    iter_calibration_probe_observations,
    scenario_strengths,
)
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.parametric_policy import decide_parametric_action
from pokerbench.agents.profiles import OpponentParameters, OpponentProfile, STATIC_PROFILES
from pokerbench.agents.strength_gates import (
    PPM_SCALE,
    REFERENCE_OPPORTUNITIES_PER_SCENARIO,
    PolicyStrengthReference,
    build_policy_calibration_map,
)
from pokerbench.engine.config import BASELINE_GAME_CONFIG, Action, GameConfig
from pokerbench.evaluation.calibration import (
    OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
    CALIBRATION_VERSION,
    BehaviorCounts,
    CalibrationMetric,
    CalibrationResult,
    calibrate_parameters,
)
from pokerbench.evaluation.opportunities import (
    ActionEffect,
    classify_observation_action_effect,
    is_voluntary_preflop_contribution,
)
from pokerbench.rng import AgentRNG


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _validate_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a SHA-256 hex digest"
        ) from exc


def _validate_game_config(game_config: GameConfig) -> None:
    if not isinstance(game_config, GameConfig):
        raise TypeError("game_config must be a GameConfig")
    if game_config != BASELINE_GAME_CONFIG:
        raise ValueError(
            "policy-calibration-acceptance-v1 requires BASELINE_GAME_CONFIG"
        )


def _validate_strength_tuple(
    values: tuple[int, ...],
    *,
    field_name: str,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")

    if len(values) != REFERENCE_OPPORTUNITIES_PER_SCENARIO:
        raise ValueError(
            f"{field_name} must contain exactly "
            f"{REFERENCE_OPPORTUNITIES_PER_SCENARIO} strengths"
        )

    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field_name} strengths must be integers")
        if not 0 <= value <= PPM_SCALE:
            raise ValueError(f"{field_name} strengths must be ppm values")


@dataclass(frozen=True)
class PolicyValidationStrengths:
    """Independent validation strengths for the five frozen probe scenarios."""

    open_btn: tuple[int, ...]
    three_bet_bb: tuple[int, ...]
    cbet_btn: tuple[int, ...]
    fold_to_cbet_bb: tuple[int, ...]
    postflop_aggression_btn: tuple[int, ...]
    calibration_environment_version: str = CALIBRATION_ENVIRONMENT_VERSION
    validation_base_seed: int = VALIDATION_BASE_SEED

    def __post_init__(self) -> None:
        if self.calibration_environment_version != CALIBRATION_ENVIRONMENT_VERSION:
            raise ValueError("Unsupported calibration-environment version")
        if self.validation_base_seed != VALIDATION_BASE_SEED:
            raise ValueError("Unsupported validation base seed")

        for field_name in (
            "open_btn",
            "three_bet_bb",
            "cbet_btn",
            "fold_to_cbet_bb",
            "postflop_aggression_btn",
        ):
            _validate_strength_tuple(
                getattr(self, field_name),
                field_name=field_name,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_environment_version": self.calibration_environment_version,
            "validation_base_seed": self.validation_base_seed,
            "open_btn": list(self.open_btn),
            "three_bet_bb": list(self.three_bet_bb),
            "cbet_btn": list(self.cbet_btn),
            "fold_to_cbet_bb": list(self.fold_to_cbet_bb),
            "postflop_aggression_btn": list(self.postflop_aggression_btn),
        }

    @property
    def validation_strength_hash(self) -> str:
        return _sha256_payload(self.to_dict())

    def for_scenario(
        self,
        scenario: CalibrationScenario,
    ) -> tuple[int, ...]:
        mapping = {
            CalibrationScenario.OPEN_BTN: self.open_btn,
            CalibrationScenario.THREE_BET_BB: self.three_bet_bb,
            CalibrationScenario.CBET_BTN: self.cbet_btn,
            CalibrationScenario.FOLD_TO_CBET_BB: self.fold_to_cbet_bb,
            CalibrationScenario.POSTFLOP_AGGRESSION_BTN: (
                self.postflop_aggression_btn
            ),
        }
        return mapping[scenario]


def generate_policy_validation_strengths(
    *,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> PolicyValidationStrengths:
    """Compute the independent validation-strength snapshot once."""
    _validate_game_config(game_config)
    return PolicyValidationStrengths(
        open_btn=scenario_strengths(
            CalibrationPhase.VALIDATION,
            CalibrationScenario.OPEN_BTN,
            game_config=game_config,
        ),
        three_bet_bb=scenario_strengths(
            CalibrationPhase.VALIDATION,
            CalibrationScenario.THREE_BET_BB,
            game_config=game_config,
        ),
        cbet_btn=scenario_strengths(
            CalibrationPhase.VALIDATION,
            CalibrationScenario.CBET_BTN,
            game_config=game_config,
        ),
        fold_to_cbet_bb=scenario_strengths(
            CalibrationPhase.VALIDATION,
            CalibrationScenario.FOLD_TO_CBET_BB,
            game_config=game_config,
        ),
        postflop_aggression_btn=scenario_strengths(
            CalibrationPhase.VALIDATION,
            CalibrationScenario.POSTFLOP_AGGRESSION_BTN,
            game_config=game_config,
        ),
    )


def _counts_dict(counts: BehaviorCounts) -> dict[str, Any]:
    return {
        "vpip_opportunities": counts.vpip_opportunities,
        "vpip_events": counts.vpip_events,
        "pfr_opportunities": counts.pfr_opportunities,
        "pfr_events": counts.pfr_events,
        "three_bet_opportunities": counts.three_bet_opportunities,
        "three_bet_events": counts.three_bet_events,
        "cbet_opportunities": counts.cbet_opportunities,
        "cbet_events": counts.cbet_events,
        "postflop_aggressive_actions": counts.postflop_aggressive_actions,
        "postflop_passive_actions": counts.postflop_passive_actions,
        "fold_to_cbet_opportunities": counts.fold_to_cbet_opportunities,
        "fold_to_cbet_events": counts.fold_to_cbet_events,
        "opportunity_definition_version": counts.opportunity_definition_version,
    }


@dataclass(frozen=True)
class PolicyCalibrationAcceptance:
    profile_name: str
    profile_version: str
    parameter_schema_version: str
    reference_strength_hash: str
    gate_map_hash: str
    validation_strength_hash: str
    behavior_counts: BehaviorCounts
    designated_opportunities: Mapping[str, int]
    calibration_result: CalibrationResult
    calibration_version: str = CALIBRATION_VERSION
    calibration_environment_version: str = CALIBRATION_ENVIRONMENT_VERSION
    fit_base_seed: int = FIT_BASE_SEED
    validation_base_seed: int = VALIDATION_BASE_SEED
    opportunities_per_metric: int = REFERENCE_OPPORTUNITIES_PER_SCENARIO

    def __post_init__(self) -> None:
        if self.calibration_version != CALIBRATION_VERSION:
            raise ValueError("Unsupported calibration version")
        if self.calibration_environment_version != CALIBRATION_ENVIRONMENT_VERSION:
            raise ValueError("Unsupported calibration-environment version")
        if self.fit_base_seed != FIT_BASE_SEED:
            raise ValueError("Unsupported fit base seed")
        if self.validation_base_seed != VALIDATION_BASE_SEED:
            raise ValueError("Unsupported validation base seed")
        if self.opportunities_per_metric != REFERENCE_OPPORTUNITIES_PER_SCENARIO:
            raise ValueError("Unexpected opportunities_per_metric")
        if self.calibration_result.profile_name != self.profile_name:
            raise ValueError("Calibration result/profile mismatch")
        if self.calibration_result.profile_version != self.profile_version:
            raise ValueError("Calibration result/profile-version mismatch")
        if (
            self.calibration_result.parameter_schema_version
            != self.parameter_schema_version
        ):
            raise ValueError(
                "Calibration result/parameter-schema mismatch"
            )

        _validate_sha256(
            self.reference_strength_hash,
            field_name="reference_strength_hash",
        )
        _validate_sha256(
            self.gate_map_hash,
            field_name="gate_map_hash",
        )
        _validate_sha256(
            self.validation_strength_hash,
            field_name="validation_strength_hash",
        )

        normalized = dict(self.designated_opportunities)
        expected = {metric.value for metric in CalibrationMetric}
        if set(normalized) != expected:
            raise ValueError("designated_opportunities must cover all metrics")
        if any(
            value != self.opportunities_per_metric
            for value in normalized.values()
        ):
            raise ValueError("Every designated metric must have 5000 opportunities")
        object.__setattr__(
            self,
            "designated_opportunities",
            MappingProxyType(normalized),
        )

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_result.is_calibrated

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_version": self.calibration_version,
            "calibration_environment_version": self.calibration_environment_version,
            "fit_base_seed": self.fit_base_seed,
            "validation_base_seed": self.validation_base_seed,
            "opportunities_per_metric": self.opportunities_per_metric,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "parameter_schema_version": self.parameter_schema_version,
            "reference_strength_hash": self.reference_strength_hash,
            "gate_map_hash": self.gate_map_hash,
            "validation_strength_hash": self.validation_strength_hash,
            "designated_opportunities": dict(self.designated_opportunities),
            "behavior_counts": _counts_dict(self.behavior_counts),
            "calibration_result": self.calibration_result.to_dict(),
        }

    @property
    def acceptance_hash(self) -> str:
        return _sha256_payload(self.to_dict())


@dataclass(frozen=True)
class StaticProfileGate6Result:
    results: tuple[PolicyCalibrationAcceptance, ...]
    calibration_version: str = CALIBRATION_VERSION

    def __post_init__(self) -> None:
        if self.calibration_version != CALIBRATION_VERSION:
            raise ValueError("Unsupported calibration version")
        expected_names = {
            profile.name.value
            for profile in STATIC_PROFILES.values()
        }
        actual_names = {
            result.profile_name
            for result in self.results
        }
        if (
            actual_names != expected_names
            or len(self.results) != len(expected_names)
        ):
            raise ValueError(
                "Gate-6 result must contain exactly all static profiles"
            )
        object.__setattr__(
            self,
            "results",
            tuple(
                sorted(
                    self.results,
                    key=lambda result: result.profile_name,
                )
            ),
        )

    @property
    def passed(self) -> bool:
        return all(result.is_calibrated for result in self.results)

    @property
    def profile_names(self) -> tuple[str, ...]:
        return tuple(result.profile_name for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_version": self.calibration_version,
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
        }

    @property
    def gate6_hash(self) -> str:
        return _sha256_payload(self.to_dict())


@dataclass
class _MutableDesignatedCounts:
    vpip_events: int = 0
    pfr_events: int = 0
    three_bet_events: int = 0
    cbet_events: int = 0
    aggressive_actions: int = 0
    passive_actions: int = 0
    fold_to_cbet_events: int = 0

    def freeze(self) -> BehaviorCounts:
        n = REFERENCE_OPPORTUNITIES_PER_SCENARIO
        if self.aggressive_actions + self.passive_actions != n:
            raise RuntimeError(
                "POSTFLOP_AGGRESSION_BTN must produce exactly one aggressive/passive action per opportunity"
            )
        return BehaviorCounts(
            vpip_opportunities=n,
            vpip_events=self.vpip_events,
            pfr_opportunities=n,
            pfr_events=self.pfr_events,
            three_bet_opportunities=n,
            three_bet_events=self.three_bet_events,
            cbet_opportunities=n,
            cbet_events=self.cbet_events,
            postflop_aggressive_actions=self.aggressive_actions,
            postflop_passive_actions=self.passive_actions,
            fold_to_cbet_opportunities=n,
            fold_to_cbet_events=self.fold_to_cbet_events,
            opportunity_definition_version=OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
        )


def _decision_draw_ppm(*, master_seed: int, bot_id: int, hand_id: int) -> int:
    rng = AgentRNG(
        master_seed=master_seed,
        bot_id=bot_id,
        hand_id=hand_id,
    )
    return rng.randint(0, PPM_SCALE - 1)


def _record_designated_action(
    counts: _MutableDesignatedCounts,
    scenario: CalibrationScenario,
    *,
    observation: AgentObservation,
    action: Action,
) -> None:
    effect = classify_observation_action_effect(observation, action)

    if scenario is CalibrationScenario.OPEN_BTN:
        if is_voluntary_preflop_contribution(action):
            counts.vpip_events += 1
        if effect is ActionEffect.AGGRESSIVE:
            counts.pfr_events += 1
        return

    if scenario is CalibrationScenario.THREE_BET_BB:
        if effect is ActionEffect.AGGRESSIVE:
            counts.three_bet_events += 1
        return

    if scenario is CalibrationScenario.CBET_BTN:
        if effect is ActionEffect.AGGRESSIVE:
            counts.cbet_events += 1
        return

    if scenario is CalibrationScenario.FOLD_TO_CBET_BB:
        if action is Action.FOLD:
            counts.fold_to_cbet_events += 1
        return

    if scenario is CalibrationScenario.POSTFLOP_AGGRESSION_BTN:
        if effect is ActionEffect.AGGRESSIVE:
            counts.aggressive_actions += 1
        elif effect is ActionEffect.PASSIVE:
            counts.passive_actions += 1
        else:
            raise RuntimeError(
                "POSTFLOP_AGGRESSION_BTN produced a fold, which is outside its frozen designated denominator"
            )
        return

    raise ValueError(f"Unsupported calibration scenario: {scenario}")


def evaluate_parameter_acceptance(
    *,
    profile_name: str,
    profile_version: str,
    parameters: OpponentParameters,
    reference: PolicyStrengthReference,
    validation_strengths: PolicyValidationStrengths,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> PolicyCalibrationAcceptance:
    """Evaluate one explicit parameter vector on the validation sample."""
    _validate_game_config(game_config)
    if not isinstance(parameters, OpponentParameters):
        raise TypeError("parameters must be OpponentParameters")
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError("profile_name must be a non-empty string")
    if not isinstance(profile_version, str) or not profile_version:
        raise ValueError("profile_version must be a non-empty string")

    calibration_map = build_policy_calibration_map(
        parameters,
        reference,
        profile_name=profile_name,
        profile_version=profile_version,
    )
    counts = _MutableDesignatedCounts()

    for scenario in CalibrationScenario:
        strengths = validation_strengths.for_scenario(scenario)
        probes = iter_calibration_probe_observations(
            CalibrationPhase.VALIDATION,
            scenario,
            game_config=game_config,
        )
        seen = 0
        for probe, equity_ppm in zip(probes, strengths, strict=True):
            action = decide_parametric_action(
                calibration_map,
                probe.observation,
                probe.observation.legal_actions,
                equity_ppm=equity_ppm,
                decision_draw_ppm=_decision_draw_ppm(
                    master_seed=probe.deal_seed,
                    bot_id=probe.hero_seat,
                    hand_id=probe.opportunity_index,
                ),
            )
            _record_designated_action(
                counts,
                scenario,
                observation=probe.observation,
                action=action,
            )
            seen += 1

        if seen != REFERENCE_OPPORTUNITIES_PER_SCENARIO:
            raise RuntimeError(
                f"Validation scenario {scenario.value} produced {seen} opportunities"
            )

    behavior_counts = counts.freeze()
    result = calibrate_parameters(
        profile_name=profile_name,
        profile_version=profile_version,
        parameters=parameters,
        counts=behavior_counts,
    )
    designated = {
        metric.value: REFERENCE_OPPORTUNITIES_PER_SCENARIO
        for metric in CalibrationMetric
    }
    return PolicyCalibrationAcceptance(
        profile_name=profile_name,
        profile_version=profile_version,
        parameter_schema_version=parameters.schema_version,
        reference_strength_hash=reference.reference_strength_hash,
        gate_map_hash=calibration_map.gate_map_hash,
        validation_strength_hash=validation_strengths.validation_strength_hash,
        behavior_counts=behavior_counts,
        designated_opportunities=designated,
        calibration_result=result,
    )


def evaluate_profile_acceptance(
    profile: OpponentProfile,
    reference: PolicyStrengthReference,
    validation_strengths: PolicyValidationStrengths,
    *,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> PolicyCalibrationAcceptance:
    """Evaluate one frozen static profile on the independent validation sample."""
    if not isinstance(profile, OpponentProfile):
        raise TypeError("profile must be an OpponentProfile")
    return evaluate_parameter_acceptance(
        profile_name=profile.name.value,
        profile_version=profile.version,
        parameters=profile.parameters,
        reference=reference,
        validation_strengths=validation_strengths,
        game_config=game_config,
    )


def evaluate_static_profile_gate6(
    reference: PolicyStrengthReference | None = None,
    validation_strengths: PolicyValidationStrengths | None = None,
    *,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> StaticProfileGate6Result:
    """Evaluate all five frozen static profiles for Phase-0 Gate 6."""
    _validate_game_config(game_config)
    fitted_reference = (
        generate_policy_strength_reference(game_config=game_config)
        if reference is None
        else reference
    )
    validation = (
        generate_policy_validation_strengths(game_config=game_config)
        if validation_strengths is None
        else validation_strengths
    )

    results = tuple(
        evaluate_profile_acceptance(
            profile,
            fitted_reference,
            validation,
            game_config=game_config,
        )
        for profile in STATIC_PROFILES.values()
    )
    return StaticProfileGate6Result(results=results)
