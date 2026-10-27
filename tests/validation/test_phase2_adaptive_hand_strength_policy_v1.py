from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import pokerbench.agents.adaptive_hand_strength as adaptive_module
import pokerbench.agents.range_equity as range_equity_module
from pokerbench.agents.adaptive_hand_strength import (
    ADAPTIVE_HAND_STRENGTH_POLICY_VERSION,
    AdaptiveHandStrengthDecisionError,
    decide_adaptive_hand_strength_action,
)
from pokerbench.agents.equity import EquityMethod, estimate_uniform_random_equity
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.opponent_model import (
    BayesianMetricPosterior,
    BayesianOpponentModel,
    BetaPrior,
    OpponentModelMetric,
    OpponentModelPrior,
)
from pokerbench.agents.opponent_range import (
    PreflopRangeSignal,
    construct_opponent_range,
    load_preflop_strength_reference,
)
from pokerbench.agents.range_equity import (
    RANGE_CONDITIONED_EQUITY_VERSION,
    RANGE_EQUITY_SAMPLES,
    RangeConditionedEquityResult,
    estimate_range_conditioned_equity,
    range_joint_outcome_count,
)
from pokerbench.engine.config import Action
from pokerbench.engine.public_action import PublicActionEvent
from pokerbench.engine.state import Street
from pokerbench.experiments.adaptive_policy_protocol import (
    ADAPTIVE_POLICY_PROTOCOL_VERSION,
    load_adaptive_policy_protocol,
)
from pokerbench.rng import AgentRNG

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/experiments/phase2-adaptive-hand-strength-policy-v1.json"
REFERENCE_PATH = ROOT / "configs/experiments/preflop-range-strength-reference-v1.json"


def _prior(shape: int = 1) -> OpponentModelPrior:
    value = BetaPrior(shape, shape)
    return OpponentModelPrior(
        vpip=value,
        pfr=value,
        three_bet=value,
        cbet=value,
        aggressive_share=value,
        fold_to_cbet=value,
    )


def _posterior(
    metric: OpponentModelMetric,
    prior: BetaPrior,
    evidence: tuple[int, int],
) -> BayesianMetricPosterior:
    return BayesianMetricPosterior(
        metric=metric,
        prior=prior,
        events=evidence[0],
        opportunities=evidence[1],
    )


def _model(
    *,
    vpip: tuple[int, int] = (20, 100),
    pfr: tuple[int, int] = (10, 100),
    three_bet: tuple[int, int] = (5, 100),
    cbet: tuple[int, int] = (50, 100),
    aggressive: tuple[int, int] = (50, 100),
    fold_to_cbet: tuple[int, int] = (50, 100),
    prior_shape: int = 1,
) -> BayesianOpponentModel:
    prior = _prior(prior_shape)
    return BayesianOpponentModel(
        prior=prior,
        hands_observed=vpip[1],
        vpip=_posterior(OpponentModelMetric.VPIP, prior.vpip, vpip),
        pfr=_posterior(OpponentModelMetric.PFR, prior.pfr, pfr),
        three_bet=_posterior(
            OpponentModelMetric.THREE_BET,
            prior.three_bet,
            three_bet,
        ),
        cbet=_posterior(OpponentModelMetric.CBET, prior.cbet, cbet),
        aggressive_share=_posterior(
            OpponentModelMetric.AGGRESSIVE_SHARE,
            prior.aggressive_share,
            aggressive,
        ),
        fold_to_cbet=_posterior(
            OpponentModelMetric.FOLD_TO_CBET,
            prior.fold_to_cbet,
            fold_to_cbet,
        ),
    )


def _event(
    *,
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
    street: Street,
    hole_cards: tuple[int, int],
    board: tuple[int, ...],
    current_bets: tuple[int, int],
    stacks: tuple[int, int] = (9000, 9000),
    pot: int = 400,
    history: tuple[PublicActionEvent, ...] = (),
    legal_actions: tuple[Action, ...] = (Action.FOLD, Action.CALL),
) -> AgentObservation:
    return AgentObservation(
        player_id=0,
        hole_cards=hole_cards,
        board=board,
        stacks=stacks,
        current_bets=current_bets,
        pot=pot,
        current_player=0,
        street=street,
        action_history=tuple(event.action for event in history),
        legal_actions=legal_actions,
        dealer=0,
        public_action_history=history,
    )


def _reference():
    return load_preflop_strength_reference(REFERENCE_PATH)


def test_protocol_file_is_canonical_and_semantically_frozen() -> None:
    protocol = load_adaptive_policy_protocol(PROTOCOL_PATH)
    assert protocol.protocol_version == ADAPTIVE_POLICY_PROTOCOL_VERSION
    assert protocol.payload_hash == (
        "76b6c561c2cb3715cc6352520dfbd3b0b01ce219a1b37d1a0acb62bb6ce10e4a"
    )
    assert ADAPTIVE_HAND_STRENGTH_POLICY_VERSION == "adaptive-hand-strength-policy-v1"


def test_full_uninformative_range_matches_uniform_random_exact_equity_on_river() -> None:
    observation = _observation(
        street=Street.RIVER,
        hole_cards=(12, 25),
        board=(0, 14, 28, 42, 4),
        current_bets=(100, 300),
    )
    snapshot = construct_opponent_range(
        _model(),
        PreflopRangeSignal.UNINFORMATIVE,
        _reference(),
    )
    range_result = estimate_range_conditioned_equity(
        observation,
        snapshot,
        master_seed=123,
        bot_id=0,
        hand_id=7,
        decision_index=0,
    )
    uniform_result = estimate_uniform_random_equity(
        observation,
        master_seed=123,
        bot_id=0,
        hand_id=7,
        decision_index=0,
    )
    assert range_result.method is EquityMethod.EXACT
    assert uniform_result.method is EquityMethod.EXACT
    assert range_result.joint_outcomes == uniform_result.joint_outcomes == 990
    assert range_result.equity_ppm == uniform_result.equity_ppm
    assert (range_result.wins, range_result.ties, range_result.losses) == (
        uniform_result.wins,
        uniform_result.ties,
        uniform_result.losses,
    )


def test_preflop_range_equity_uses_frozen_10000_sample_mc_without_agent_rng_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Make showdown comparison cheap while exercising the frozen MC sampler.
    monkeypatch.setattr(
        range_equity_module,
        "_compare_showdown",
        lambda hero, opponent, board: 1 if sum(board) % 2 == 0 else -1,
    )
    observation = _observation(
        street=Street.PREFLOP,
        hole_cards=(0, 1),
        board=(),
        current_bets=(50, 100),
        pot=150,
    )
    model = _model(vpip=(0, 2000), pfr=(0, 100), three_bet=(0, 100))
    snapshot = construct_opponent_range(
        model,
        PreflopRangeSignal.PASSIVE_ENTRY,
        _reference(),
    )
    assert snapshot.selected_combo_count == 1
    assert range_joint_outcome_count(observation, snapshot) > 500_000
    rng = AgentRNG(master_seed=99, bot_id=0, hand_id=3)
    before = rng.decision_index
    first = estimate_range_conditioned_equity(
        observation,
        snapshot,
        master_seed=rng.master_seed,
        bot_id=rng.bot_id,
        hand_id=rng.hand_id,
        decision_index=rng.decision_index,
    )
    second = estimate_range_conditioned_equity(
        observation,
        snapshot,
        master_seed=rng.master_seed,
        bot_id=rng.bot_id,
        hand_id=rng.hand_id,
        decision_index=rng.decision_index,
    )
    assert first == second
    assert first.method is EquityMethod.MONTE_CARLO
    assert first.sample_count == RANGE_EQUITY_SAMPLES == 10_000
    assert first.equity_version == RANGE_CONDITIONED_EQUITY_VERSION
    assert rng.decision_index == before


def test_range_equity_rejects_tampered_range_hash() -> None:
    observation = _observation(
        street=Street.RIVER,
        hole_cards=(0, 1),
        board=(2, 3, 4, 5, 6),
        current_bets=(100, 300),
    )
    snapshot = construct_opponent_range(
        _model(),
        PreflopRangeSignal.FIRST_RAISE,
        _reference(),
    )
    with pytest.raises(ValueError, match="range snapshot hash mismatch"):
        estimate_range_conditioned_equity(
            observation,
            replace(snapshot, range_hash="0" * 64),
            master_seed=1,
            bot_id=0,
            hand_id=0,
            decision_index=0,
        )


def test_policy_checks_when_free_and_does_not_run_equity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adaptive_module,
        "estimate_range_conditioned_equity",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("equity called")),
    )
    history = (
        _event(
            actor=1,
            street=Street.PREFLOP,
            action=Action.BET_100,
            wager_increase=True,
        ),
    )
    observation = _observation(
        street=Street.FLOP,
        hole_cards=(0, 1),
        board=(2, 3, 4),
        current_bets=(0, 0),
        pot=300,
        history=history,
        legal_actions=(Action.CHECK, Action.BET_100),
    )
    rng = AgentRNG(master_seed=11, bot_id=0, hand_id=1)
    decision = decide_adaptive_hand_strength_action(
        observation,
        observation.legal_actions,
        rng,
        model=_model(),
        reference=_reference(),
    )
    assert decision.action is Action.CHECK
    assert decision.signal is PreflopRangeSignal.FIRST_RAISE
    assert decision.equity is None
    assert decision.pot_odds_ppm is None
    assert rng.decision_index == 0


def _fake_equity(snapshot, equity_ppm: int) -> RangeConditionedEquityResult:
    return RangeConditionedEquityResult(
        method=EquityMethod.EXACT,
        equity_ppm=equity_ppm,
        wins=1,
        ties=0,
        losses=0,
        joint_outcomes=1,
        sample_count=1,
        legal_combo_count=1,
        range_hash=snapshot.range_hash,
    )


def test_policy_calls_on_inclusive_pot_odds_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation(
        street=Street.RIVER,
        hole_cards=(0, 1),
        board=(2, 3, 4, 5, 6),
        current_bets=(100, 300),
        pot=400,
    )
    # call_cost=200; 200/(400+200) -> 333333 ppm after half-up rounding.
    def estimator(obs, snapshot, **kwargs):
        return _fake_equity(snapshot, 333333)

    monkeypatch.setattr(adaptive_module, "estimate_range_conditioned_equity", estimator)
    decision = decide_adaptive_hand_strength_action(
        observation,
        observation.legal_actions,
        AgentRNG(1, 0, 0),
        model=_model(),
        reference=_reference(),
    )
    assert decision.pot_odds_ppm == 333333
    assert decision.equity is not None
    assert decision.equity.equity_ppm == decision.pot_odds_ppm
    assert decision.action is Action.CALL


def test_policy_folds_below_pot_odds_and_never_substitutes_aggression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation(
        street=Street.RIVER,
        hole_cards=(0, 1),
        board=(2, 3, 4, 5, 6),
        current_bets=(100, 300),
        pot=400,
    )
    monkeypatch.setattr(
        adaptive_module,
        "estimate_range_conditioned_equity",
        lambda obs, snapshot, **kwargs: _fake_equity(snapshot, 333332),
    )
    decision = decide_adaptive_hand_strength_action(
        observation,
        (Action.FOLD, Action.CALL, Action.ALL_IN),
        AgentRNG(1, 0, 0),
        model=_model(),
        reference=_reference(),
    )
    assert decision.action is Action.FOLD

    no_passive = replace(
        observation,
        legal_actions=(Action.BET_100, Action.ALL_IN),
    )
    with pytest.raises(AdaptiveHandStrengthDecisionError, match="does not substitute aggression"):
        decide_adaptive_hand_strength_action(
            no_passive,
            no_passive.legal_actions,
            AgentRNG(1, 0, 0),
            model=_model(),
            reference=_reference(),
        )


def test_postflop_behavior_posteriors_are_action_inactive_in_v1() -> None:
    history = (
        _event(
            actor=1,
            street=Street.PREFLOP,
            action=Action.BET_100,
            wager_increase=True,
        ),
        _event(
            actor=1,
            street=Street.FLOP,
            action=Action.BET_100,
            wager_increase=True,
        ),
    )
    observation = _observation(
        street=Street.RIVER,
        hole_cards=(0, 1),
        board=(2, 3, 4, 5, 6),
        current_bets=(100, 300),
        pot=400,
        history=history,
    )
    low = _model(cbet=(1, 100), aggressive=(1, 100), fold_to_cbet=(1, 100))
    high = _model(cbet=(99, 100), aggressive=(99, 100), fold_to_cbet=(99, 100))
    a = decide_adaptive_hand_strength_action(
        observation,
        observation.legal_actions,
        AgentRNG(5, 0, 0),
        model=low,
        reference=_reference(),
    )
    b = decide_adaptive_hand_strength_action(
        observation,
        observation.legal_actions,
        AgentRNG(5, 0, 0),
        model=high,
        reference=_reference(),
    )
    assert a.signal is b.signal is PreflopRangeSignal.FIRST_RAISE
    assert a.range_snapshot.selected_combinations == b.range_snapshot.selected_combinations
    assert a.range_snapshot.width_ppm == b.range_snapshot.width_ppm
    assert a.equity is not None and b.equity is not None
    assert a.equity.equity_ppm == b.equity.equity_ppm
    assert a.action is b.action
    # Provenance still binds the different complete model states.
    assert a.model_state_hash != b.model_state_hash
    assert a.decision_hash != b.decision_hash


def test_policy_requires_the_frozen_beta11_prior_vector() -> None:
    observation = _observation(
        street=Street.FLOP,
        hole_cards=(0, 1),
        board=(2, 3, 4),
        current_bets=(0, 0),
        legal_actions=(Action.CHECK,),
    )
    with pytest.raises(ValueError, match=r"Beta\(1,1\)"):
        decide_adaptive_hand_strength_action(
            observation,
            observation.legal_actions,
            AgentRNG(1, 0, 0),
            model=_model(prior_shape=2),
            reference=_reference(),
        )


def test_decision_hash_is_deterministic_and_replay_payload_is_complete() -> None:
    history = (
        _event(
            actor=1,
            street=Street.PREFLOP,
            action=Action.CALL,
            wager_increase=False,
        ),
    )
    observation = _observation(
        street=Street.RIVER,
        hole_cards=(0, 1),
        board=(2, 3, 4, 5, 6),
        current_bets=(100, 300),
        history=history,
    )
    kwargs = dict(
        observation=observation,
        legal_actions=observation.legal_actions,
        model=_model(),
        reference=_reference(),
    )
    first = decide_adaptive_hand_strength_action(
        rng=AgentRNG(77, 0, 9),
        **kwargs,
    )
    second = decide_adaptive_hand_strength_action(
        rng=AgentRNG(77, 0, 9),
        **kwargs,
    )
    assert first == second
    assert len(first.decision_hash) == 64
    payload = first.to_dict()
    assert payload["model_state_hash"] == first.model_state_hash
    assert payload["range_hash"] == first.range_snapshot.range_hash
    assert payload["signal"] == "passive_entry"
    assert payload["legal_combo_count"] > 0
    assert payload["pot_odds_ppm"] is not None
    assert payload["equity"] is not None
    assert payload["decision_hash"] == first.decision_hash


def test_policy_rejects_a_different_strength_reference_even_with_same_version() -> None:
    observation = _observation(
        street=Street.FLOP,
        hole_cards=(0, 1),
        board=(2, 3, 4),
        current_bets=(0, 0),
        legal_actions=(Action.CHECK,),
    )
    reference = replace(_reference(), payload_hash="0" * 64)
    with pytest.raises(ValueError, match="reference hash"):
        decide_adaptive_hand_strength_action(
            observation,
            observation.legal_actions,
            AgentRNG(1, 0, 0),
            model=_model(),
            reference=reference,
        )


def test_protocol_rejects_semantic_tampering_even_with_recomputed_hash(
    tmp_path: Path,
) -> None:
    document = json.loads(PROTOCOL_PATH.read_bytes())
    document["payload"]["action_policy"]["aggressive_actions"] = True
    canonical = json.dumps(
        document["payload"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    document["payload_hash"] = hashlib.sha256(canonical).hexdigest()
    path = tmp_path / "tampered.json"
    path.write_bytes(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )
    with pytest.raises(ValueError, match="semantics differ"):
        load_adaptive_policy_protocol(path)


def test_empty_live_range_remains_an_error_without_backfill() -> None:
    model = _model(vpip=(0, 2000), pfr=(0, 100), three_bet=(0, 100))
    snapshot = construct_opponent_range(
        model,
        PreflopRangeSignal.PASSIVE_ENTRY,
        _reference(),
    )
    assert snapshot.selected_combo_count == 1
    blocked = snapshot.selected_combinations[0]
    observation = _observation(
        street=Street.PREFLOP,
        hole_cards=blocked,
        board=(),
        current_bets=(50, 100),
        pot=150,
    )
    with pytest.raises(ValueError, match="forbids range backfill"):
        estimate_range_conditioned_equity(
            observation,
            snapshot,
            master_seed=1,
            bot_id=0,
            hand_id=0,
            decision_index=0,
        )
