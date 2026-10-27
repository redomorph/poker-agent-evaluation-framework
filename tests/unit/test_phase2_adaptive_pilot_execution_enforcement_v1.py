from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from pokerbench.agents.adaptive_lifecycle import (
    ADAPTIVE_HAND_STRENGTH_AGENT_VERSION,
    AdaptiveHandStrengthAgent,
)
from pokerbench.agents.base import Agent
from pokerbench.agents.budget import AgentExecutionBudget, TimeoutPolicy
from pokerbench.agents.hand_strength_baseline import (
    HAND_STRENGTH_BASELINE_VERSION,
    HandStrengthBaselineAgent,
)
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.opponent_model import BayesianOpponentModel
from pokerbench.agents.opponent_range import load_preflop_strength_reference
from pokerbench.engine.config import Action, BASELINE_GAME_CONFIG
from pokerbench.engine.state import GameState
from pokerbench.evaluation.calibration import BehaviorCounts
from pokerbench.experiments.adaptive_pilot_binding import (
    build_adaptive_pilot_execution_binding,
    build_adaptive_pilot_trajectory_config,
)
from pokerbench.experiments.adaptive_pilot_execution import (
    AdaptivePilotExecutionError,
    PilotHeroBoundaryAgent,
    build_adaptive_pilot_hero,
    build_adaptive_pilot_runner,
    finalize_adaptive_pilot_pair,
    prove_matched_deal_schedule,
    validate_matched_adaptive_pilot_jobs,
    validate_pilot_hero_observation,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotJob,
    AdaptivePilotProtocol,
    iter_adaptive_pilot_jobs,
    load_adaptive_pilot_protocol,
)
from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.frozen_protocol import load_frozen_population
from pokerbench.replay.archive import MatchRecord, RecordedAdaptiveLearningEvent
from pokerbench.rng import AgentRNG
from pokerbench.runner import PairedResult, PairedRunner, StatisticalUnit
import pokerbench.runner as runner_module

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-v1.json"
REFERENCE_PATH = ROOT / "configs" / "experiments" / "preflop-range-strength-reference-v1.json"
PRIMARY_PATH = ROOT / "configs" / "populations" / "baseline-synthetic-v1.json"
SECONDARY_PATH = ROOT / "configs" / "populations" / "balanced-v1.json"
GATE6_PATH = ROOT / "data" / "gate6" / "gate6-report.json"


class _CallCheckAgent(Agent):
    name = "test-static-opponent-v1"

    def act(self, observation, legal_actions, rng):  # type: ignore[no-untyped-def]
        del observation, rng
        if Action.CHECK in legal_actions:
            return Action.CHECK
        if Action.CALL in legal_actions:
            return Action.CALL
        return Action.FOLD


class _FrozenControlProbe(_CallCheckAgent):
    name = HAND_STRENGTH_BASELINE_VERSION


class _LifecycleProbeAgent(_CallCheckAgent):
    version = ADAPTIVE_HAND_STRENGTH_AGENT_VERSION

    def __init__(self) -> None:
        self._model = AdaptiveHandStrengthAgent.initial_model()

    @property
    def is_adaptive(self) -> bool:
        return True

    @property
    def model(self) -> BayesianOpponentModel:
        return self._model

    def reset(self) -> None:
        self._model = AdaptiveHandStrengthAgent.initial_model()

    def observe_opponent_hand(
        self,
        counts: BehaviorCounts,
        *,
        learner_bot_id: int,
        opponent_bot_id: int,
        trajectory_hand_index: int,
    ) -> RecordedAdaptiveLearningEvent:
        before = self._model
        after = before.updated(counts, first_hand_index=before.hands_observed)
        event = RecordedAdaptiveLearningEvent.from_transition(
            learner_bot_id=learner_bot_id,
            opponent_bot_id=opponent_bot_id,
            trajectory_hand_index=trajectory_hand_index,
            before=before,
            evidence=counts,
            after=after,
        )
        self._model = after
        return event


class _MemoryMatchStore:
    def __init__(self) -> None:
        self.records: dict[str, MatchRecord] = {}

    def save_match(self, record: MatchRecord) -> None:
        if record.match_id in self.records:
            raise ValueError("duplicate match")
        self.records[record.match_id] = record

    def load_match(self, match_id: str) -> MatchRecord:
        return self.records[match_id]


def _protocol() -> AdaptivePilotProtocol:
    return load_adaptive_pilot_protocol(PROTOCOL_PATH)


def _job(
    protocol: AdaptivePilotProtocol,
    *,
    arm_id: str,
    profile_id: str = "nit",
    seed: int = 6_300_001,
) -> AdaptivePilotJob:
    return next(
        job
        for job in iter_adaptive_pilot_jobs(protocol)
        if job.arm_id == arm_id
        and job.opponent_profile_id == profile_id
        and job.master_seed == seed
    )


def _trajectory_config(protocol: AdaptivePilotProtocol, job: AdaptivePilotJob):
    commit = "a" * 40
    environment = ExecutionEnvironment(
        python_version="3.14.0",
        engine_version=f"git:{commit}",
        dependency_lockfile="poetry.lock",
        poetry_lock_hash="b" * 64,
        git_commit=commit,
        cpu_model="Test CPU",
        cpu_core_count=8,
        cpu_architecture="AMD64",
        ram_bytes=16 * 1024**3,
        os_identifier="TestOS",
    )
    binding = build_adaptive_pilot_execution_binding(
        protocol=protocol,
        gate6_report=load_gate6_calibration_report(GATE6_PATH),
        primary_population=load_frozen_population(PRIMARY_PATH),
        secondary_populations=(load_frozen_population(SECONDARY_PATH),),
        environment=environment,
        workers=1,
    )
    return build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=binding,
        job=job,
    )


def _matched_jobs(
    *,
    profile_id: str = "nit",
    seed: int = 6_300_001,
) -> tuple[AdaptivePilotProtocol, AdaptivePilotJob, AdaptivePilotJob]:
    protocol = _protocol()
    return (
        protocol,
        _job(protocol, arm_id="treatment", profile_id=profile_id, seed=seed),
        _job(protocol, arm_id="control", profile_id=profile_id, seed=seed),
    )


def _run_fast_valid_pair(
    *,
    profile_id: str = "nit",
    seed: int = 6_300_001,
) -> tuple[
    AdaptivePilotProtocol,
    AdaptivePilotJob,
    AdaptivePilotJob,
    PairedResult,
    PairedResult,
    _MemoryMatchStore,
]:
    protocol, treatment_job, control_job = _matched_jobs(
        profile_id=profile_id,
        seed=seed,
    )
    store = _MemoryMatchStore()
    treatment = PairedRunner(
        agent_a=_LifecycleProbeAgent(),
        agent_b=_CallCheckAgent(),
        num_hands=treatment_job.hands_per_position,
        seed=seed,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
        match_id=f"treatment-{profile_id}-{seed}",
        benchmark_store=store,
    ).run()
    control = PairedRunner(
        agent_a=_FrozenControlProbe(),
        agent_b=_CallCheckAgent(),
        num_hands=control_job.hands_per_position,
        seed=seed,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
        match_id=f"control-{profile_id}-{seed}",
        benchmark_store=store,
    ).run()
    return protocol, treatment_job, control_job, treatment, control, store


def test_matched_jobs_must_be_exact_frozen_control_treatment_cell() -> None:
    protocol, treatment, control = _matched_jobs()
    matched = validate_matched_adaptive_pilot_jobs(
        protocol=protocol,
        treatment_job=treatment,
        control_job=control,
    )
    assert matched.profile_id == "nit"
    assert matched.master_seed == 6_300_001
    assert matched.hands_per_position == 100

    wrong_profile = _job(protocol, arm_id="control", profile_id="tag", seed=6_300_001)
    with pytest.raises(AdaptivePilotExecutionError, match="same profile"):
        validate_matched_adaptive_pilot_jobs(
            protocol=protocol,
            treatment_job=treatment,
            control_job=wrong_profile,
        )


def test_noncanonical_or_replacement_seed_job_is_rejected() -> None:
    protocol, treatment, control = _matched_jobs()
    replacement = replace(treatment, master_seed=6_399_999)
    with pytest.raises(AdaptivePilotExecutionError, match="canonical frozen pilot job"):
        validate_matched_adaptive_pilot_jobs(
            protocol=protocol,
            treatment_job=replacement,
            control_job=control,
        )


def test_frozen_hero_factory_builds_only_spec32_control_and_treatment() -> None:
    protocol, treatment, control = _matched_jobs()
    reference = load_preflop_strength_reference(REFERENCE_PATH)

    control_hero = build_adaptive_pilot_hero(control, reference=reference)
    treatment_hero = build_adaptive_pilot_hero(treatment, reference=reference)

    assert type(control_hero) is HandStrengthBaselineAgent
    assert control_hero.is_adaptive is False
    assert not hasattr(control_hero, "model")

    assert type(treatment_hero) is AdaptiveHandStrengthAgent
    assert treatment_hero.is_adaptive is True
    assert treatment_hero.version == ADAPTIVE_HAND_STRENGTH_AGENT_VERSION
    assert treatment_hero.model.state_hash == AdaptiveHandStrengthAgent.initial_model().state_hash


@pytest.mark.parametrize("arm_id", ["control", "treatment"])
def test_runner_builder_forces_full_match_trajectory_cluster_for_both_arms(
    arm_id: str,
) -> None:
    protocol = _protocol()
    job = _job(protocol, arm_id=arm_id)
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    store = _MemoryMatchStore()

    runner = build_adaptive_pilot_runner(
        protocol=protocol,
        job=job,
        reference=reference,
        opponent=_CallCheckAgent(),
        game_config=BASELINE_GAME_CONFIG,
        match_id=f"guarded-{arm_id}",
        benchmark_store=store,
        trajectory_config=_trajectory_config(protocol, job),
    )

    assert runner.adaptive is True
    assert runner.statistical_unit is StatisticalUnit.TRAJECTORY_CLUSTER
    assert runner.num_hands == job.hands_per_position
    assert isinstance(runner.agent_a, PilotHeroBoundaryAgent)
    assert runner.agent_a.is_adaptive is job.adaptive
    assert runner.experiment_metadata["experiment_config_hash"]
    assert runner.experiment_metadata["hero_arm"] == arm_id
    assert runner.experiment_metadata["master_seed"] == job.master_seed
    assert runner.experiment_metadata["planned_trajectory_horizon"] == job.hands_per_position


def test_runner_builder_rejects_provenance_for_a_different_frozen_job() -> None:
    protocol = _protocol()
    treatment_job = _job(protocol, arm_id="treatment")
    control_job = _job(protocol, arm_id="control")
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    store = _MemoryMatchStore()

    with pytest.raises(AdaptivePilotExecutionError, match="trajectory config"):
        build_adaptive_pilot_runner(
            protocol=protocol,
            job=treatment_job,
            reference=reference,
            opponent=_CallCheckAgent(),
            game_config=BASELINE_GAME_CONFIG,
            match_id="wrong-provenance",
            benchmark_store=store,
            trajectory_config=_trajectory_config(protocol, control_job),
        )


def test_treatment_runner_resets_frozen_prior_before_both_position_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    job = _job(protocol, arm_id="treatment")
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    store = _MemoryMatchStore()
    runner = build_adaptive_pilot_runner(
        protocol=protocol,
        job=job,
        reference=reference,
        opponent=_CallCheckAgent(),
        game_config=BASELINE_GAME_CONFIG,
        match_id="reset-proof",
        benchmark_store=store,
        trajectory_config=_trajectory_config(protocol, job),
    )
    assert isinstance(runner.agent_a, PilotHeroBoundaryAgent)
    delegate = runner.agent_a.delegate
    assert isinstance(delegate, AdaptiveHandStrengthAgent)
    initial_hash = AdaptiveHandStrengthAgent.initial_model().state_hash
    reset_hashes: list[str] = []
    original_reset = delegate.reset

    def counted_reset() -> None:
        original_reset()
        reset_hashes.append(delegate.model.state_hash)

    monkeypatch.setattr(delegate, "reset", counted_reset)
    monkeypatch.setattr(
        runner,
        "_run_full_match",
        lambda **kwargs: (0.0, 0.0, []),
    )
    runner.run()
    assert reset_hashes == [initial_hash, initial_hash]



def test_information_boundary_rejects_controller_only_fields() -> None:
    state = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=((0, 1), (2, 3)),
        board=(4, 5, 6, 7, 8),
        dealer=0,
    )
    observation = state.observation(state.current_player)
    validate_pilot_hero_observation(observation)

    leaked = replace(observation)
    object.__setattr__(leaked, "profile_id", "nit")
    with pytest.raises(AdaptivePilotExecutionError, match="forbidden or unknown"):
        validate_pilot_hero_observation(leaked)

    leaked_parameters = replace(observation)
    object.__setattr__(leaked_parameters, "true_opponent_parameter_vector", {"vpip": 0.1})
    with pytest.raises(AdaptivePilotExecutionError, match="forbidden or unknown"):
        validate_pilot_hero_observation(leaked_parameters)


def test_boundary_wrapper_forwards_only_public_learning_counts() -> None:
    probe = _LifecycleProbeAgent()
    guarded = PilotHeroBoundaryAgent(probe)
    counts = BehaviorCounts(
        vpip_opportunities=1,
        vpip_events=0,
        pfr_opportunities=1,
        pfr_events=0,
        three_bet_opportunities=0,
        three_bet_events=0,
        cbet_opportunities=0,
        cbet_events=0,
        postflop_aggressive_actions=0,
        postflop_passive_actions=0,
        fold_to_cbet_opportunities=0,
        fold_to_cbet_events=0,
    )
    event = guarded.observe_opponent_hand(
        counts,
        learner_bot_id=0,
        opponent_bot_id=1,
        trajectory_hand_index=0,
    )
    assert event.public_evidence_counts["vpip_opportunities"] == 1
    assert probe.model.hands_observed == 1


def test_matched_crn_proof_uses_identical_deals_without_action_path_claim() -> None:
    protocol, treatment, control = _matched_jobs(seed=6_300_000)
    proof = prove_matched_deal_schedule(
        protocol=protocol,
        treatment_job=treatment,
        control_job=control,
    )
    assert proof.master_seed == 6_300_000
    assert proof.hands_per_position == 500
    assert len(proof.deal_schedule_hash) == 64
    assert proof.deal_relation == "identical_defined_card_realisations"
    assert proof.agent_rng_relation == "event_indexed_common_identity_where_events_correspond"
    assert proof.full_action_path_identity == "not_claimed"


def test_valid_complete_pair_is_replayed_and_admitted_as_cluster_evidence() -> None:
    protocol, treatment_job, control_job, treatment, control, store = _run_fast_valid_pair()
    finalized = finalize_adaptive_pilot_pair(
        protocol=protocol,
        treatment_job=treatment_job,
        control_job=control_job,
        treatment_result=treatment,
        control_result=control,
        benchmark_store=store,
    )

    assert finalized.scientifically_valid is True
    assert finalized.treatment_evidence is not None
    assert finalized.control_evidence is not None
    assert len(finalized.treatment_evidence.hero_btn_profit_bb) == 100
    assert len(finalized.treatment_evidence.hero_bb_profit_bb) == 100
    assert len(finalized.control_evidence.hero_btn_profit_bb) == 100
    assert len(finalized.control_evidence.hero_bb_profit_bb) == 100
    assert finalized.treatment_learning_hands_by_position == (("A_BTN", 100), ("B_BTN", 100))


def test_incomplete_pair_remains_durable_but_cannot_enter_analysis() -> None:
    protocol, treatment_job, control_job, treatment, control, store = _run_fast_valid_pair()
    failed_control = replace(
        control,
        paired_differences=[],
        num_valid_pairs=0,
        failed_runs=[{"hand": 12, "config": "A_BTN", "details": "failed"}],
    )
    finalized = finalize_adaptive_pilot_pair(
        protocol=protocol,
        treatment_job=treatment_job,
        control_job=control_job,
        treatment_result=treatment,
        control_result=failed_control,
        benchmark_store=store,
    )
    assert finalized.scientifically_valid is False
    assert finalized.treatment_evidence is None
    assert finalized.control_evidence is None
    assert finalized.control_failed_runs == (
        {"hand": 12, "config": "A_BTN", "details": "failed"},
    )
    assert finalized.treatment_match_id in store.records
    assert finalized.control_match_id in store.records


def test_adaptive_timeout_invalidates_complete_pair_but_failure_record_is_durable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, treatment_job, control_job = _matched_jobs()
    store = _MemoryMatchStore()
    clock_values = iter(range(0, 20_000_000_000, 2_000_000))
    monkeypatch.setattr(runner_module.time, "perf_counter_ns", lambda: next(clock_values))
    budget = AgentExecutionBudget(
        time_budget_ms=1,
        timeout_policy=TimeoutPolicy.FOLD,
    )
    treatment = PairedRunner(
        agent_a=_LifecycleProbeAgent(),
        agent_b=_CallCheckAgent(),
        num_hands=treatment_job.hands_per_position,
        seed=treatment_job.master_seed,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
        agent_a_budget=budget,
        match_id="timeout-treatment",
        benchmark_store=store,
    ).run()
    control = PairedRunner(
        agent_a=_FrozenControlProbe(),
        agent_b=_CallCheckAgent(),
        num_hands=control_job.hands_per_position,
        seed=control_job.master_seed,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
        match_id="timeout-control",
        benchmark_store=store,
    ).run()
    assert treatment.num_valid_pairs == 0
    assert treatment.failed_runs[0]["details"] == "adaptive_learning_evidence_invalid"

    finalized = finalize_adaptive_pilot_pair(
        protocol=protocol,
        treatment_job=treatment_job,
        control_job=control_job,
        treatment_result=treatment,
        control_result=control,
        benchmark_store=store,
    )
    assert finalized.scientifically_valid is False
    assert finalized.treatment_evidence is None
    assert finalized.control_evidence is None
    assert finalized.treatment_match_id in store.records



def test_finalization_rejects_non_durable_match_record() -> None:
    protocol, treatment_job, control_job, treatment, control, store = _run_fast_valid_pair()
    del store.records[control.match_record.match_id]  # type: ignore[union-attr]
    with pytest.raises(AdaptivePilotExecutionError, match="durably persisted"):
        finalize_adaptive_pilot_pair(
            protocol=protocol,
            treatment_job=treatment_job,
            control_job=control_job,
            treatment_result=treatment,
            control_result=control,
            benchmark_store=store,
        )


def test_finalization_rejects_tampered_adaptive_learning_chain() -> None:
    protocol, treatment_job, control_job, treatment, control, store = _run_fast_valid_pair()
    assert treatment.match_record is not None
    hands = list(treatment.match_record.hands)
    hand = hands[1]
    event = hand.adaptive_learning_events[0]
    core = {
        "version": event.version,
        "learner_bot_id": event.learner_bot_id,
        "opponent_bot_id": event.opponent_bot_id,
        "trajectory_hand_index": event.trajectory_hand_index,
        "model_state_before_hash": "0" * 64,
        "public_evidence_counts": dict(event.public_evidence_counts),
        "model_state_after_hash": event.model_state_after_hash,
    }
    transition_hash = hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    tampered_event = RecordedAdaptiveLearningEvent(
        **core,
        transition_hash=transition_hash,
    )
    hands[1] = replace(hand, adaptive_learning_events=(tampered_event,))
    tampered_record = replace(treatment.match_record, hands=tuple(hands))
    tampered_result = replace(treatment, match_record=tampered_record)
    store.records[tampered_record.match_id] = tampered_record

    with pytest.raises(ValueError, match="before-state hash mismatch"):
        finalize_adaptive_pilot_pair(
            protocol=protocol,
            treatment_job=treatment_job,
            control_job=control_job,
            treatment_result=tampered_result,
            control_result=control,
            benchmark_store=store,
        )


def test_control_record_must_never_contain_adaptive_learning_events() -> None:
    protocol, treatment_job, control_job, treatment, control, store = _run_fast_valid_pair()
    assert treatment.match_record is not None
    assert control.match_record is not None
    source_event = treatment.match_record.hands[0].adaptive_learning_events[0]
    control_hands = list(control.match_record.hands)
    control_hands[0] = replace(control_hands[0], adaptive_learning_events=(source_event,))
    contaminated = replace(control.match_record, hands=tuple(control_hands))
    contaminated_result = replace(control, match_record=contaminated)
    store.records[contaminated.match_id] = contaminated

    with pytest.raises(AdaptivePilotExecutionError, match="control arm.*adaptive learning"):
        finalize_adaptive_pilot_pair(
            protocol=protocol,
            treatment_job=treatment_job,
            control_job=control_job,
            treatment_result=treatment,
            control_result=contaminated_result,
            benchmark_store=store,
        )
