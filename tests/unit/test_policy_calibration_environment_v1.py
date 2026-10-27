from __future__ import annotations

import hashlib

import pytest

from pokerbench.agents.calibration_environment import (
    CALIBRATION_ENVIRONMENT_VERSION,
    FIT_BASE_SEED,
    VALIDATION_BASE_SEED,
    CalibrationPhase,
    CalibrationScenario,
    calibration_probe_observation,
    generate_policy_strength_reference,
    iter_calibration_probe_observations,
    scenario_deal_seed,
)
from pokerbench.agents.equity import EquityMethod, EquityResult
from pokerbench.agents.strength_gates import (
    REFERENCE_OPPORTUNITIES_PER_SCENARIO,
    STRENGTH_REFERENCE_VERSION,
)
from pokerbench.engine.config import Action
from pokerbench.engine.state import Street


def _event_signature(probe) -> tuple[tuple[int, Street, Action, bool], ...]:
    return tuple(
        (
            event.actor,
            event.street,
            event.action,
            event.wager_increase,
        )
        for event in probe.observation.public_action_history
    )


def test_environment_constants_match_frozen_contract() -> None:
    assert (
        CALIBRATION_ENVIRONMENT_VERSION
        == "policy-calibration-environment-v1"
    )
    assert FIT_BASE_SEED == 5_100_001
    assert VALIDATION_BASE_SEED == 5_100_002
    assert REFERENCE_OPPORTUNITIES_PER_SCENARIO == 5_000


def test_scenario_seed_uses_frozen_sha256_domain_separation() -> None:
    material = (
        f"{CALIBRATION_ENVIRONMENT_VERSION}:"
        f"{FIT_BASE_SEED}:OPEN_BTN"
    ).encode("utf-8")
    expected = int.from_bytes(
        hashlib.sha256(material).digest(),
        byteorder="big",
        signed=False,
    )

    assert scenario_deal_seed(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
    ) == expected

    assert scenario_deal_seed(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
    ) != scenario_deal_seed(
        CalibrationPhase.VALIDATION,
        CalibrationScenario.OPEN_BTN,
    )

    assert scenario_deal_seed(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
    ) != scenario_deal_seed(
        CalibrationPhase.FIT,
        CalibrationScenario.THREE_BET_BB,
    )


def test_open_btn_probe_is_initial_button_decision() -> None:
    probe = calibration_probe_observation(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
        0,
    )

    observation = probe.observation

    assert probe.hero_seat == 0
    assert probe.dealer == 0
    assert probe.decision_index == 0
    assert observation.player_id == 0
    assert observation.current_player == 0
    assert observation.dealer == 0
    assert observation.street is Street.PREFLOP
    assert observation.public_action_history == ()
    assert Action.CALL in observation.legal_actions
    assert Action.BET_200 in observation.legal_actions


def test_three_bet_bb_probe_faces_exactly_one_raise() -> None:
    probe = calibration_probe_observation(
        CalibrationPhase.FIT,
        CalibrationScenario.THREE_BET_BB,
        0,
    )

    observation = probe.observation

    assert probe.hero_seat == 1
    assert observation.current_player == 1
    assert observation.street is Street.PREFLOP
    assert _event_signature(probe) == (
        (0, Street.PREFLOP, Action.BET_200, True),
    )
    assert observation.current_bets[0] > observation.current_bets[1]
    assert Action.CALL in observation.legal_actions
    assert Action.BET_200 in observation.legal_actions


def test_cbet_btn_probe_preserves_checked_to_preflop_aggressor() -> None:
    probe = calibration_probe_observation(
        CalibrationPhase.FIT,
        CalibrationScenario.CBET_BTN,
        0,
    )

    observation = probe.observation

    assert probe.hero_seat == 0
    assert observation.current_player == 0
    assert observation.street is Street.FLOP
    assert _event_signature(probe) == (
        (0, Street.PREFLOP, Action.BET_200, True),
        (1, Street.PREFLOP, Action.CALL, False),
        (1, Street.FLOP, Action.CHECK, False),
    )
    assert Action.CHECK in observation.legal_actions
    assert Action.BET_66 in observation.legal_actions


def test_fold_to_cbet_bb_probe_directly_faces_qualifying_cbet() -> None:
    probe = calibration_probe_observation(
        CalibrationPhase.FIT,
        CalibrationScenario.FOLD_TO_CBET_BB,
        0,
    )

    observation = probe.observation

    assert probe.hero_seat == 1
    assert observation.current_player == 1
    assert observation.street is Street.FLOP
    assert _event_signature(probe) == (
        (0, Street.PREFLOP, Action.BET_200, True),
        (1, Street.PREFLOP, Action.CALL, False),
        (1, Street.FLOP, Action.CHECK, False),
        (0, Street.FLOP, Action.BET_66, True),
    )
    assert observation.current_bets[0] > observation.current_bets[1]
    assert Action.FOLD in observation.legal_actions
    assert Action.CALL in observation.legal_actions


def test_postflop_aggression_btn_probe_is_unopened_non_cbet_flop() -> None:
    probe = calibration_probe_observation(
        CalibrationPhase.FIT,
        CalibrationScenario.POSTFLOP_AGGRESSION_BTN,
        0,
    )

    observation = probe.observation

    assert probe.hero_seat == 0
    assert observation.current_player == 0
    assert observation.street is Street.FLOP
    assert _event_signature(probe) == (
        (0, Street.PREFLOP, Action.CALL, False),
        (1, Street.PREFLOP, Action.CHECK, False),
        (1, Street.FLOP, Action.CHECK, False),
    )
    assert not any(
        event.street is Street.PREFLOP and event.wager_increase
        for event in observation.public_action_history
    )
    assert Action.CHECK in observation.legal_actions
    assert Action.BET_66 in observation.legal_actions


def test_probe_index_is_restricted_to_frozen_opportunity_count() -> None:
    with pytest.raises(ValueError):
        calibration_probe_observation(
            CalibrationPhase.FIT,
            CalibrationScenario.OPEN_BTN,
            -1,
        )

    with pytest.raises(ValueError):
        calibration_probe_observation(
            CalibrationPhase.FIT,
            CalibrationScenario.OPEN_BTN,
            REFERENCE_OPPORTUNITIES_PER_SCENARIO,
        )


def test_iterator_produces_exactly_5000_domain_separated_probes() -> None:
    probes = tuple(
        iter_calibration_probe_observations(
            CalibrationPhase.VALIDATION,
            CalibrationScenario.OPEN_BTN,
        )
    )

    assert len(probes) == REFERENCE_OPPORTUNITIES_PER_SCENARIO
    assert probes[0].opportunity_index == 0
    assert probes[-1].opportunity_index == 4_999
    assert all(
        probe.deal_seed == probes[0].deal_seed
        for probe in probes
    )
    assert probes[0].deal_seed == scenario_deal_seed(
        CalibrationPhase.VALIDATION,
        CalibrationScenario.OPEN_BTN,
    )


def test_reference_generation_uses_fit_only_and_frozen_equity_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pokerbench.agents.calibration_environment as module

    calls: list[tuple[int, int, int, int, CalibrationScenario]] = []

    def fake_equity(
        observation,
        *,
        master_seed: int,
        bot_id: int,
        hand_id: int,
        decision_index: int,
    ) -> EquityResult:
        del observation
        seed_to_scenario = {
            scenario_deal_seed(CalibrationPhase.FIT, item): item
            for item in CalibrationScenario
        }
        scenario = seed_to_scenario[master_seed]
        calls.append(
            (
                master_seed,
                bot_id,
                hand_id,
                decision_index,
                scenario,
            )
        )
        equity_ppm = (
            hand_id * 31
            + list(CalibrationScenario).index(scenario) * 997
        ) % 1_000_001
        return EquityResult(
            method=EquityMethod.MONTE_CARLO,
            equity_ppm=equity_ppm,
            wins=equity_ppm,
            ties=0,
            losses=1_000_000 - equity_ppm,
            joint_outcomes=1_000_001,
            sample_count=10_000,
        )

    monkeypatch.setattr(
        module,
        "estimate_uniform_random_equity",
        fake_equity,
    )

    reference = generate_policy_strength_reference()

    assert reference.version == STRENGTH_REFERENCE_VERSION
    assert len(reference.open_btn) == 5_000
    assert len(reference.three_bet_bb) == 5_000
    assert len(reference.cbet_btn) == 5_000
    assert len(reference.fold_to_cbet_bb) == 5_000
    assert len(reference.postflop_aggression_btn) == 5_000
    assert len(calls) == 25_000

    for (
        master_seed,
        bot_id,
        hand_id,
        decision_index,
        scenario,
    ) in calls:
        assert master_seed == scenario_deal_seed(
            CalibrationPhase.FIT,
            scenario,
        )
        assert bot_id in (0, 1)
        assert bot_id == module.scenario_hero_seat(scenario)
        assert 0 <= hand_id < 5_000
        assert decision_index == 0


def test_reference_generation_is_deterministic_with_same_fake_equity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pokerbench.agents.calibration_environment as module

    def fake_equity(
        observation,
        *,
        master_seed: int,
        bot_id: int,
        hand_id: int,
        decision_index: int,
    ) -> EquityResult:
        del observation, master_seed, bot_id, decision_index
        value = hand_id % 1_000_001
        return EquityResult(
            method=EquityMethod.MONTE_CARLO,
            equity_ppm=value,
            wins=value,
            ties=0,
            losses=1_000_000 - value,
            joint_outcomes=1_000_001,
            sample_count=10_000,
        )

    monkeypatch.setattr(
        module,
        "estimate_uniform_random_equity",
        fake_equity,
    )

    first = generate_policy_strength_reference()
    second = generate_policy_strength_reference()

    assert first == second
    assert first.reference_strength_hash == second.reference_strength_hash
