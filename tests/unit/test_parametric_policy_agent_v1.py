from __future__ import annotations

from dataclasses import replace

import pytest

import pokerbench.agents.parametric_policy as policy_module
from pokerbench.agents.equity import EquityMethod, EquityResult
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.parametric_policy import (
    ParametricPolicyAgent,
    ParametricPolicyDecisionError,
)
from pokerbench.agents.profiles import OpponentParameters
from pokerbench.agents.strength_gates import (
    GateDirection,
    PolicyCalibrationMap,
    StrengthGate,
)
from pokerbench.engine.config import Action
from pokerbench.engine.public_action import PublicActionEvent
from pokerbench.engine.state import Street
from pokerbench.rng import AgentRNG


REFERENCE_HASH = "0" * 64


def _high(threshold: int, tie: int = 0) -> StrengthGate:
    return StrengthGate(
        direction=GateDirection.HIGH,
        threshold_ppm=threshold,
        tie_probability_ppm=tie,
    )


def _low(threshold: int, tie: int = 0) -> StrengthGate:
    return StrengthGate(
        direction=GateDirection.LOW,
        threshold_ppm=threshold,
        tie_probability_ppm=tie,
    )


def _map(
    *,
    aggression: float | None = 0.8,
    vpip_gate: StrengthGate | None = None,
    pfr_gate: StrengthGate | None = None,
    three_bet_gate: StrengthGate | None = None,
    cbet_gate: StrengthGate | None = None,
    aggression_gate: StrengthGate | None | object = object(),
    fold_to_cbet_gate: StrengthGate | None = None,
) -> PolicyCalibrationMap:
    parameters = OpponentParameters(
        vpip=0.50,
        pfr=0.25,
        three_bet_freq=0.10,
        cbet_freq=0.60,
        aggression=aggression,
        fold_to_cbet=0.40,
    )

    if isinstance(aggression_gate, object) and type(aggression_gate) is object:
        resolved_aggression_gate = (
            None
            if aggression is None
            else _high(600_000)
        )
    else:
        resolved_aggression_gate = aggression_gate

    return PolicyCalibrationMap(
        parameters=parameters,
        vpip_gate=vpip_gate or _high(400_000),
        pfr_gate=pfr_gate or _high(700_000),
        three_bet_gate=three_bet_gate or _high(700_000),
        cbet_gate=cbet_gate or _high(500_000),
        aggression_gate=resolved_aggression_gate,  # type: ignore[arg-type]
        fold_to_cbet_gate=fold_to_cbet_gate or _low(300_000),
        reference_strength_hash=REFERENCE_HASH,
    )


def _event(
    actor: int,
    street: Street,
    action: Action,
    wager_increase: bool,
) -> PublicActionEvent:
    return PublicActionEvent(
        actor=actor,
        street=street,
        action=action,
        wager_increase=wager_increase,
    )


def _observation(
    *,
    player_id: int = 0,
    street: Street = Street.PREFLOP,
    stacks: tuple[int, int] = (9_950, 9_900),
    current_bets: tuple[int, int] = (50, 100),
    pot: int = 150,
    legal_actions: tuple[Action, ...] = (
        Action.FOLD,
        Action.CALL,
        Action.BET_200,
        Action.ALL_IN,
    ),
    history: tuple[PublicActionEvent, ...] = (),
    dealer: int = 0,
) -> AgentObservation:
    board_by_street = {
        Street.PREFLOP: (),
        Street.FLOP: (2, 6, 10),
        Street.TURN: (2, 6, 10, 14),
        Street.RIVER: (2, 6, 10, 14, 18),
    }

    return AgentObservation(
        player_id=player_id,
        hole_cards=(40, 44),
        board=board_by_street[street],
        stacks=stacks,
        current_bets=current_bets,
        pot=pot,
        current_player=player_id,
        street=street,
        action_history=tuple(event.action for event in history),
        legal_actions=legal_actions,
        dealer=dealer,
        public_action_history=history,
    )


def _equity_result(equity_ppm: int) -> EquityResult:
    return EquityResult(
        method=EquityMethod.MONTE_CARLO,
        equity_ppm=equity_ppm,
        wins=0,
        ties=0,
        losses=10_000,
        joint_outcomes=1_070_190,
        sample_count=10_000,
    )


def _stub_equity(
    monkeypatch: pytest.MonkeyPatch,
    equity_ppm: int,
    *,
    captured: dict[str, int] | None = None,
) -> None:
    def fake_estimate(
        observation: AgentObservation,
        *,
        master_seed: int,
        bot_id: int,
        hand_id: int,
        decision_index: int,
    ) -> EquityResult:
        if captured is not None:
            captured.update(
                master_seed=master_seed,
                bot_id=bot_id,
                hand_id=hand_id,
                decision_index=decision_index,
            )
        return _equity_result(equity_ppm)

    monkeypatch.setattr(
        policy_module,
        "estimate_uniform_random_equity",
        fake_estimate,
    )


def test_act_consumes_exactly_one_action_rng_event_and_uses_pre_draw_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}
    _stub_equity(
        monkeypatch,
        800_000,
        captured=captured,
    )
    agent = ParametricPolicyAgent(_map())
    rng = AgentRNG(
        master_seed=1234,
        bot_id=1,
        hand_id=17,
    )

    before = rng.decision_index
    action = agent.act(
        _observation(),
        [
            Action.FOLD,
            Action.CALL,
            Action.BET_100,
            Action.ALL_IN,
        ],
        rng,
    )

    assert action is Action.BET_100
    assert before == 0
    assert rng.decision_index == 1
    assert captured == {
        "master_seed": 1234,
        "bot_id": 1,
        "hand_id": 17,
        "decision_index": 0,
    }


def test_preflop_opening_pfr_gate_selects_strength_sized_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    agent = ParametricPolicyAgent(_map())
    legal = [
        Action.FOLD,
        Action.CALL,
        Action.BET_66,
        Action.BET_100,
        Action.BET_200,
        Action.ALL_IN,
    ]

    assert agent.act(
        _observation(legal_actions=tuple(legal)),
        legal,
        AgentRNG(1, 0, 0),
    ) is Action.BET_100


def test_preflop_opening_vpip_without_pfr_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 600_000)
    agent = ParametricPolicyAgent(_map())
    legal = [Action.FOLD, Action.CALL, Action.BET_200, Action.ALL_IN]

    assert agent.act(
        _observation(legal_actions=tuple(legal)),
        legal,
        AgentRNG(2, 0, 0),
    ) is Action.CALL


def test_preflop_opening_below_vpip_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 200_000)
    agent = ParametricPolicyAgent(_map())
    legal = [Action.FOLD, Action.CALL, Action.BET_200, Action.ALL_IN]

    assert agent.act(
        _observation(legal_actions=tuple(legal)),
        legal,
        AgentRNG(3, 0, 0),
    ) is Action.FOLD


def test_big_blind_after_limp_checks_when_pfr_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 600_000)
    agent = ParametricPolicyAgent(_map())
    history = (
        _event(0, Street.PREFLOP, Action.CALL, False),
    )
    legal = [Action.CHECK, Action.BET_66, Action.BET_100, Action.ALL_IN]
    observation = _observation(
        player_id=1,
        stacks=(9_900, 9_900),
        current_bets=(100, 100),
        pot=200,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(4, 1, 0),
    ) is Action.CHECK


def test_qualifying_three_bet_gate_selects_aggression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    agent = ParametricPolicyAgent(_map())
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
    )
    legal = [Action.FOLD, Action.CALL, Action.BET_100, Action.ALL_IN]
    observation = _observation(
        player_id=1,
        stacks=(9_700, 9_900),
        current_bets=(300, 100),
        pot=400,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(5, 1, 0),
    ) is Action.BET_100


def test_failed_three_bet_gate_uses_vpip_and_pot_odds_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 600_000)
    agent = ParametricPolicyAgent(_map())
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
    )
    legal = [Action.FOLD, Action.CALL, Action.BET_200, Action.ALL_IN]
    observation = _observation(
        player_id=1,
        stacks=(9_700, 9_900),
        current_bets=(300, 100),
        pot=400,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(6, 1, 0),
    ) is Action.CALL


def test_single_raise_without_legal_reraise_uses_passive_all_in_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 600_000)
    agent = ParametricPolicyAgent(_map())
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
    )
    legal = [Action.FOLD, Action.ALL_IN]
    observation = _observation(
        player_id=1,
        stacks=(9_700, 150),
        current_bets=(300, 100),
        pot=400,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(7, 1, 0),
    ) is Action.ALL_IN


def test_two_prior_raises_never_initiate_four_bet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 990_000)
    agent = ParametricPolicyAgent(_map())
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
        _event(1, Street.PREFLOP, Action.BET_200, True),
    )
    legal = [Action.FOLD, Action.CALL, Action.BET_200, Action.ALL_IN]
    observation = _observation(
        player_id=0,
        stacks=(9_300, 9_300),
        current_bets=(300, 700),
        pot=1_000,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(8, 0, 0),
    ) is Action.CALL


def test_checked_to_preflop_aggressor_is_qualifying_cbet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    agent = ParametricPolicyAgent(_map())
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
        _event(1, Street.PREFLOP, Action.CALL, False),
        _event(1, Street.FLOP, Action.CHECK, False),
    )
    legal = [Action.CHECK, Action.BET_66, Action.BET_100, Action.ALL_IN]
    observation = _observation(
        street=Street.FLOP,
        stacks=(9_700, 9_700),
        current_bets=(0, 0),
        pot=600,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(9, 0, 0),
    ) is Action.BET_100


def test_cbet_gate_failure_checks_even_when_generic_aggression_would_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    calibration_map = _map(
        cbet_gate=_high(900_000),
        aggression_gate=_high(700_000),
    )
    agent = ParametricPolicyAgent(calibration_map)
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
        _event(1, Street.PREFLOP, Action.CALL, False),
        _event(1, Street.FLOP, Action.CHECK, False),
    )
    legal = [Action.CHECK, Action.BET_100, Action.ALL_IN]
    observation = _observation(
        street=Street.FLOP,
        stacks=(9_700, 9_700),
        current_bets=(0, 0),
        pot=600,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(10, 0, 0),
    ) is Action.CHECK


def test_donk_before_preflop_aggressor_disqualifies_cbet_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    calibration_map = _map(
        cbet_gate=_high(100_000),
        aggression_gate=_high(900_000),
    )
    agent = ParametricPolicyAgent(calibration_map)
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
        _event(1, Street.PREFLOP, Action.CALL, False),
        _event(1, Street.FLOP, Action.BET_66, True),
    )
    legal = [Action.FOLD, Action.CALL, Action.BET_200, Action.ALL_IN]
    observation = _observation(
        street=Street.FLOP,
        stacks=(9_700, 9_300),
        current_bets=(0, 400),
        pot=1_000,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(11, 0, 0),
    ) is Action.CALL


def test_direct_fold_to_cbet_low_tail_gate_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 200_000)
    agent = ParametricPolicyAgent(_map())
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
        _event(1, Street.PREFLOP, Action.CALL, False),
        _event(1, Street.FLOP, Action.CHECK, False),
        _event(0, Street.FLOP, Action.BET_66, True),
    )
    legal = [Action.FOLD, Action.CALL, Action.BET_200, Action.ALL_IN]
    observation = _observation(
        player_id=1,
        street=Street.FLOP,
        stacks=(9_300, 9_700),
        current_bets=(400, 0),
        pot=1_000,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(12, 1, 0),
    ) is Action.FOLD


def test_direct_fold_to_cbet_continue_can_raise_via_aggression_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    agent = ParametricPolicyAgent(_map())
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
        _event(1, Street.PREFLOP, Action.CALL, False),
        _event(1, Street.FLOP, Action.CHECK, False),
        _event(0, Street.FLOP, Action.BET_66, True),
    )
    legal = [Action.FOLD, Action.CALL, Action.BET_100, Action.ALL_IN]
    observation = _observation(
        player_id=1,
        street=Street.FLOP,
        stacks=(9_300, 9_700),
        current_bets=(400, 0),
        pot=1_000,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(13, 1, 0),
    ) is Action.BET_100


def test_calling_station_direct_cbet_continue_is_passive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    agent = ParametricPolicyAgent(_map(aggression=None))
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
        _event(1, Street.PREFLOP, Action.CALL, False),
        _event(1, Street.FLOP, Action.CHECK, False),
        _event(0, Street.FLOP, Action.BET_66, True),
    )
    legal = [Action.FOLD, Action.CALL, Action.BET_100, Action.ALL_IN]
    observation = _observation(
        player_id=1,
        street=Street.FLOP,
        stacks=(9_300, 9_700),
        current_bets=(400, 0),
        pot=1_000,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(14, 1, 0),
    ) is Action.CALL


def test_generic_postflop_facing_wager_below_pot_odds_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 100_000)
    agent = ParametricPolicyAgent(_map())
    history = (
        _event(1, Street.FLOP, Action.BET_100, True),
    )
    legal = [Action.FOLD, Action.CALL, Action.BET_200, Action.ALL_IN]
    observation = _observation(
        street=Street.TURN,
        stacks=(9_000, 9_000),
        current_bets=(0, 500),
        pot=1_000,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(15, 0, 0),
    ) is Action.FOLD


def test_generic_postflop_not_facing_wager_uses_aggression_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    agent = ParametricPolicyAgent(_map())
    legal = [Action.CHECK, Action.BET_66, Action.BET_100, Action.ALL_IN]
    observation = _observation(
        street=Street.TURN,
        stacks=(9_000, 9_000),
        current_bets=(0, 0),
        pot=1_000,
        legal_actions=tuple(legal),
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(16, 0, 0),
    ) is Action.BET_100


def test_calling_station_generic_postflop_not_facing_wager_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 990_000)
    agent = ParametricPolicyAgent(_map(aggression=None))
    legal = [Action.CHECK, Action.BET_200, Action.ALL_IN]
    observation = _observation(
        street=Street.TURN,
        stacks=(9_000, 9_000),
        current_bets=(0, 0),
        pot=1_000,
        legal_actions=tuple(legal),
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(17, 0, 0),
    ) is Action.CHECK


def test_strength_sizing_fallback_uses_closest_lower_rank_on_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    agent = ParametricPolicyAgent(_map())
    legal = [Action.CHECK, Action.BET_66, Action.BET_200]
    observation = _observation(
        street=Street.TURN,
        stacks=(9_000, 9_000),
        current_bets=(0, 0),
        pot=1_000,
        legal_actions=tuple(legal),
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(18, 0, 0),
    ) is Action.BET_66


def test_all_in_is_aggressive_only_when_it_increases_wager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 990_000)
    agent = ParametricPolicyAgent(_map())
    legal = [Action.CHECK, Action.ALL_IN]
    observation = _observation(
        street=Street.TURN,
        stacks=(2_000, 9_000),
        current_bets=(0, 0),
        pot=1_000,
        legal_actions=tuple(legal),
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(19, 0, 0),
    ) is Action.ALL_IN


def test_policy_rejects_observation_for_nonacting_player_without_rng_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    agent = ParametricPolicyAgent(_map())
    observation = replace(
        _observation(),
        current_player=1,
    )
    rng = AgentRNG(20, 0, 0)

    with pytest.raises(ParametricPolicyDecisionError):
        agent.act(
            observation,
            list(observation.legal_actions),
            rng,
        )

    assert rng.decision_index == 0


def test_same_seed_state_and_map_produce_same_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 700_000)
    calibration_map = _map(
        vpip_gate=_high(700_000, 500_000),
        pfr_gate=_high(700_000, 250_000),
    )
    agent = ParametricPolicyAgent(calibration_map)
    observation = _observation()
    legal = list(observation.legal_actions)

    first = agent.act(
        observation,
        legal,
        AgentRNG(777, 0, 3),
    )
    second = agent.act(
        observation,
        legal,
        AgentRNG(777, 0, 3),
    )

    assert first is second


def test_calling_station_cbet_gate_remains_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    calibration_map = _map(
        aggression=None,
        cbet_gate=_high(500_000),
    )
    agent = ParametricPolicyAgent(calibration_map)
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
        _event(1, Street.PREFLOP, Action.CALL, False),
        _event(1, Street.FLOP, Action.CHECK, False),
    )
    legal = [Action.CHECK, Action.BET_100, Action.ALL_IN]
    observation = _observation(
        street=Street.FLOP,
        stacks=(9_700, 9_700),
        current_bets=(0, 0),
        pot=600,
        legal_actions=tuple(legal),
        history=history,
    )

    assert agent.act(
        observation,
        legal,
        AgentRNG(21, 0, 0),
    ) is Action.BET_100


def test_later_reraise_is_not_misclassified_as_direct_cbet_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_equity(monkeypatch, 800_000)
    calibration_map = _map(
        fold_to_cbet_gate=_low(900_000),
        aggression_gate=_high(900_000),
    )
    agent = ParametricPolicyAgent(calibration_map)
    history = (
        _event(0, Street.PREFLOP, Action.BET_200, True),
        _event(1, Street.PREFLOP, Action.CALL, False),
        _event(1, Street.FLOP, Action.CHECK, False),
        _event(0, Street.FLOP, Action.BET_66, True),
        _event(1, Street.FLOP, Action.BET_100, True),
        _event(0, Street.FLOP, Action.BET_66, True),
    )
    legal = [Action.FOLD, Action.CALL, Action.BET_200, Action.ALL_IN]
    observation = _observation(
        player_id=1,
        street=Street.FLOP,
        stacks=(8_500, 8_500),
        current_bets=(1_500, 1_000),
        pot=3_000,
        legal_actions=tuple(legal),
        history=history,
    )

    # This is a generic response to a later re-raise, not a direct response
    # to the original qualifying continuation bet. With 80% equity and the
    # generic aggression gate failing, the correct action is CALL.
    assert agent.act(
        observation,
        legal,
        AgentRNG(22, 1, 0),
    ) is Action.CALL


def test_calibration_map_records_frozen_equity_versions() -> None:
    calibration_map = _map()
    persisted = calibration_map.to_dict()

    assert persisted["equity_version"] == "uniform-random-equity-v1"
    assert persisted["equity_range_version"] == "uniform-random-range-v1"


def test_parametric_policy_agent_is_exported_from_agents_package() -> None:
    from pokerbench.agents import (
        ACTION_SIZING_VERSION,
        ParametricPolicyAgent as ExportedAgent,
        ParametricPolicyDecisionError as ExportedError,
    )

    assert ExportedAgent is ParametricPolicyAgent
    assert ExportedError is ParametricPolicyDecisionError
    assert ACTION_SIZING_VERSION == "strength-sizing-v1"
