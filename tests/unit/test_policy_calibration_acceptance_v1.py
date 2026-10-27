from __future__ import annotations

from dataclasses import replace

import pytest

from pokerbench.agents.calibration_environment import (
    CALIBRATION_ENVIRONMENT_VERSION,
    CalibrationPhase,
    CalibrationScenario,
    VALIDATION_BASE_SEED,
    calibration_probe_observation,
)
from pokerbench.agents.profiles import (
    CALLING_STATION_PROFILE,
    STATIC_PROFILES,
    TAG_PROFILE,
)
from pokerbench.agents.strength_gates import (
    PolicyStrengthReference,
    REFERENCE_OPPORTUNITIES_PER_SCENARIO,
    build_profile_calibration_map,
)
from pokerbench.engine.config import Action
from pokerbench.evaluation.calibration import CalibrationMetric
from pokerbench.evaluation.policy_acceptance import (
    PolicyValidationStrengths,
    evaluate_profile_acceptance,
    evaluate_static_profile_gate6,
    generate_policy_validation_strengths,
)


def _strengths(value: int) -> tuple[int, ...]:
    return (value,) * REFERENCE_OPPORTUNITIES_PER_SCENARIO


def _reference() -> PolicyStrengthReference:
    return PolicyStrengthReference(
        open_btn=_strengths(500_000),
        three_bet_bb=_strengths(500_000),
        cbet_btn=_strengths(500_000),
        fold_to_cbet_bb=_strengths(500_000),
        postflop_aggression_btn=_strengths(500_000),
    )


def _validation_strengths() -> PolicyValidationStrengths:
    return PolicyValidationStrengths(
        open_btn=_strengths(500_000),
        three_bet_bb=_strengths(500_000),
        cbet_btn=_strengths(500_000),
        fold_to_cbet_bb=_strengths(500_000),
        postflop_aggression_btn=_strengths(500_000),
    )


def _metric(result, metric: CalibrationMetric):
    return next(item for item in result.metrics if item.metric is metric)


@pytest.fixture(autouse=True)
def _fast_validation_probes(monkeypatch: pytest.MonkeyPatch):
    probes = {
        scenario: calibration_probe_observation(
            CalibrationPhase.VALIDATION,
            scenario,
            0,
        )
        for scenario in CalibrationScenario
    }

    def fake_iter(phase, scenario, **kwargs):
        assert phase is CalibrationPhase.VALIDATION
        probe = probes[scenario]
        for index in range(REFERENCE_OPPORTUNITIES_PER_SCENARIO):
            yield replace(
                probe,
                opportunity_index=index,
            )

    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.iter_calibration_probe_observations",
        fake_iter,
    )


def test_validation_strengths_are_frozen_and_hashable() -> None:
    strengths = _validation_strengths()

    assert strengths.calibration_environment_version == CALIBRATION_ENVIRONMENT_VERSION
    assert strengths.validation_base_seed == VALIDATION_BASE_SEED
    assert len(strengths.validation_strength_hash) == 64
    assert strengths.to_dict()["open_btn"] == [500_000] * 5_000

    with pytest.raises(Exception):
        strengths.open_btn = ()  # type: ignore[misc]


def test_validation_strengths_require_exactly_5000_per_scenario() -> None:
    with pytest.raises(ValueError):
        PolicyValidationStrengths(
            open_btn=(500_000,),
            three_bet_bb=_strengths(500_000),
            cbet_btn=_strengths(500_000),
            fold_to_cbet_bb=_strengths(500_000),
            postflop_aggression_btn=_strengths(500_000),
        )


def test_validation_strength_generation_uses_only_validation_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[CalibrationPhase, CalibrationScenario]] = []

    def fake_strengths(phase, scenario, **kwargs):
        calls.append((phase, scenario))
        return _strengths(500_000)

    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.scenario_strengths",
        fake_strengths,
    )

    result = generate_policy_validation_strengths()

    assert len(calls) == 5
    assert {phase for phase, _ in calls} == {CalibrationPhase.VALIDATION}
    assert {scenario for _, scenario in calls} == set(CalibrationScenario)
    assert len(result.validation_strength_hash) == 64


def test_profile_acceptance_uses_validation_phase_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases: list[CalibrationPhase] = []
    probes = {
        scenario: calibration_probe_observation(
            CalibrationPhase.VALIDATION,
            scenario,
            0,
        )
        for scenario in CalibrationScenario
    }

    def wrapped(phase, scenario, **kwargs):
        phases.append(phase)
        probe = probes[scenario]
        for index in range(REFERENCE_OPPORTUNITIES_PER_SCENARIO):
            yield replace(probe, opportunity_index=index)

    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.iter_calibration_probe_observations",
        wrapped,
    )

    evaluate_profile_acceptance(
        TAG_PROFILE,
        _reference(),
        _validation_strengths(),
    )

    assert phases
    assert set(phases) == {CalibrationPhase.VALIDATION}


def test_profile_acceptance_has_5000_designated_opportunities(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make the gate decisions deterministic and exactly match TAG targets.
    target_actions = {
        CalibrationScenario.OPEN_BTN: Action.CALL,
        CalibrationScenario.THREE_BET_BB: Action.CALL,
        CalibrationScenario.CBET_BTN: Action.CHECK,
        CalibrationScenario.FOLD_TO_CBET_BB: Action.CALL,
        CalibrationScenario.POSTFLOP_AGGRESSION_BTN: Action.CHECK,
    }

    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.decide_parametric_action",
        lambda calibration_map, observation, legal_actions, **kwargs: target_actions[
            _scenario_from_observation(observation)
        ],
    )

    acceptance = evaluate_profile_acceptance(
        TAG_PROFILE,
        _reference(),
        _validation_strengths(),
    )

    assert acceptance.designated_opportunities == {
        "vpip": 5_000,
        "pfr": 5_000,
        "three_bet_freq": 5_000,
        "cbet_freq": 5_000,
        "aggression": 5_000,
        "fold_to_cbet": 5_000,
    }


def _scenario_from_observation(observation):
    if observation.street.value == "preflop":
        if observation.public_action_history:
            return CalibrationScenario.THREE_BET_BB
        return CalibrationScenario.OPEN_BTN
    history = observation.public_action_history
    if len(history) == 4:
        return CalibrationScenario.FOLD_TO_CBET_BB
    if history and history[0].action is Action.BET_200:
        return CalibrationScenario.CBET_BTN
    return CalibrationScenario.POSTFLOP_AGGRESSION_BTN


def test_acceptance_counts_follow_calibration_opportunities_v2_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def action_for(calibration_map, observation, legal_actions, **kwargs):
        scenario = _scenario_from_observation(observation)
        if scenario is CalibrationScenario.OPEN_BTN:
            return Action.BET_200
        if scenario is CalibrationScenario.THREE_BET_BB:
            return Action.ALL_IN
        if scenario is CalibrationScenario.CBET_BTN:
            return Action.BET_66
        if scenario is CalibrationScenario.FOLD_TO_CBET_BB:
            return Action.FOLD
        return Action.BET_33

    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.decide_parametric_action",
        action_for,
    )

    acceptance = evaluate_profile_acceptance(
        TAG_PROFILE,
        _reference(),
        _validation_strengths(),
    )
    counts = acceptance.behavior_counts

    assert counts.vpip_opportunities == 5_000
    assert counts.vpip_events == 5_000
    assert counts.pfr_opportunities == 5_000
    assert counts.pfr_events == 5_000
    assert counts.three_bet_opportunities == 5_000
    assert counts.three_bet_events == 5_000
    assert counts.cbet_opportunities == 5_000
    assert counts.cbet_events == 5_000
    assert counts.fold_to_cbet_opportunities == 5_000
    assert counts.fold_to_cbet_events == 5_000
    assert counts.postflop_aggressive_actions == 5_000
    assert counts.postflop_passive_actions == 0


def test_calling_station_aggression_is_reported_but_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.decide_parametric_action",
        lambda calibration_map, observation, legal_actions, **kwargs: (
            Action.BET_33
            if _scenario_from_observation(observation)
            is CalibrationScenario.POSTFLOP_AGGRESSION_BTN
            else _passive_action(observation, legal_actions)
        ),
    )

    acceptance = evaluate_profile_acceptance(
        CALLING_STATION_PROFILE,
        _reference(),
        _validation_strengths(),
    )
    aggression = _metric(
        acceptance.calibration_result,
        CalibrationMetric.AGGRESSION,
    )

    assert not aggression.required
    assert aggression.passed is None
    assert acceptance.designated_opportunities["aggression"] == 5_000


def _passive_action(observation, legal_actions):
    if Action.CHECK in legal_actions:
        return Action.CHECK
    if Action.CALL in legal_actions:
        return Action.CALL
    return Action.FOLD


def test_acceptance_result_serialization_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.decide_parametric_action",
        lambda calibration_map, observation, legal_actions, **kwargs: _passive_action(
            observation, legal_actions
        ),
    )

    first = evaluate_profile_acceptance(
        TAG_PROFILE,
        _reference(),
        _validation_strengths(),
    )
    second = evaluate_profile_acceptance(
        TAG_PROFILE,
        _reference(),
        _validation_strengths(),
    )

    assert first.to_dict() == second.to_dict()
    assert first.acceptance_hash == second.acceptance_hash
    assert first.calibration_version == "policy-calibration-v1"


def test_ideal_boundary_snapshot_passes_all_static_profiles() -> None:
    gate = evaluate_static_profile_gate6(
        _reference(),
        _validation_strengths(),
    )

    assert gate.passed
    assert all(result.is_calibrated for result in gate.results)


def test_static_gate6_contains_all_five_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_acceptance(profile, reference, validation_strengths, **kwargs):
        real_map = build_profile_calibration_map(profile, reference)
        from pokerbench.evaluation.calibration import BehaviorCounts, calibrate_profile
        p = profile.parameters
        counts = BehaviorCounts(
            vpip_opportunities=5_000,
            vpip_events=round(p.vpip * 5_000),
            pfr_opportunities=5_000,
            pfr_events=round(p.pfr * 5_000),
            three_bet_opportunities=5_000,
            three_bet_events=round((p.three_bet_freq or 0) * 5_000),
            cbet_opportunities=5_000,
            cbet_events=round((p.cbet_freq or 0) * 5_000),
            postflop_aggressive_actions=(
                0 if p.aggression is None else round((p.aggression / (1 + p.aggression)) * 5_000)
            ),
            postflop_passive_actions=(
                5_000 if p.aggression is None else 5_000 - round((p.aggression / (1 + p.aggression)) * 5_000)
            ),
            fold_to_cbet_opportunities=5_000,
            fold_to_cbet_events=round((p.fold_to_cbet or 0) * 5_000),
        )
        result = calibrate_profile(profile, counts)
        from pokerbench.evaluation.policy_acceptance import PolicyCalibrationAcceptance
        return PolicyCalibrationAcceptance(
            profile_name=profile.name.value,
            profile_version=profile.version,
            parameter_schema_version=profile.parameters.schema_version,
            reference_strength_hash=reference.reference_strength_hash,
            gate_map_hash=real_map.gate_map_hash,
            validation_strength_hash=validation_strengths.validation_strength_hash,
            behavior_counts=counts,
            designated_opportunities={m.value: 5_000 for m in CalibrationMetric},
            calibration_result=result,
        )

    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.evaluate_profile_acceptance",
        fake_acceptance,
    )

    gate = evaluate_static_profile_gate6(
        _reference(),
        _validation_strengths(),
    )

    assert set(gate.profile_names) == {name.value for name in STATIC_PROFILES}
    assert len(gate.results) == 5
    assert gate.passed
    assert len(gate.gate6_hash) == 64


def test_static_gate6_fails_if_one_profile_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_acceptance(profile, reference, validation_strengths, **kwargs):
        base = evaluate_profile_acceptance(
            profile,
            reference,
            validation_strengths,
        )
        if profile.name.value != "tag":
            return base
        bad = replace(
            base.calibration_result,
            metrics=tuple(
                replace(item, passed=False, deviation=1.0)
                if item.metric is CalibrationMetric.VPIP
                else item
                for item in base.calibration_result.metrics
            ),
            is_calibrated=False,
        )
        return replace(base, calibration_result=bad)

    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.decide_parametric_action",
        lambda calibration_map, observation, legal_actions, **kwargs: _passive_action(
            observation, legal_actions
        ),
    )
    original = evaluate_profile_acceptance
    monkeypatch.setattr(
        "pokerbench.evaluation.policy_acceptance.evaluate_profile_acceptance",
        fake_acceptance,
    )

    gate = evaluate_static_profile_gate6(
        _reference(),
        _validation_strengths(),
    )

    assert not gate.passed
