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
from pokerbench.agents.hand_strength_baseline import (
    HAND_STRENGTH_BASELINE_VERSION,
    HandStrengthBaselineAgent,
)
from pokerbench.agents.opponent_model import BayesianOpponentModel
from pokerbench.agents.opponent_range import load_preflop_strength_reference
from pokerbench.engine.config import Action, BASELINE_GAME_CONFIG
from pokerbench.engine.state import GameState
from pokerbench.evaluation.calibration import BehaviorCounts
from pokerbench.experiments.h4_holdout_variance_execution import (
    H4HoldoutVarianceArmEvidence,
    H4HoldoutVarianceExecutionError,
    H4HoldoutVarianceHeroBoundaryAgent,
    H4HoldoutVariancePairExecution,
    admit_h4_holdout_variance_campaign,
    admit_h4_holdout_variance_population_cluster,
    build_h4_holdout_variance_hero,
    build_h4_holdout_variance_runner,
    finalize_h4_holdout_variance_pair,
    prove_h4_holdout_variance_matched_deal_schedule,
    validate_h4_holdout_variance_hero_observation,
    validate_matched_h4_holdout_variance_jobs,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4_HOLDOUT_UNIT_IDS,
    H4_VARIANCE_MASTER_SEEDS,
    H4HoldoutVarianceJob,
    H4HoldoutVarianceProtocol,
    iter_h4_holdout_variance_jobs,
    load_h4_holdout_variance_protocol,
)
from pokerbench.replay.archive import MatchRecord, RecordedAdaptiveLearningEvent
from pokerbench.runner import PairedResult, PairedRunner, StatisticalUnit


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-h4-holdout-variance-pilot-v1.json"
REFERENCE_PATH = ROOT / "configs" / "experiments" / "preflop-range-strength-reference-v1.json"


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
    def __init__(self, records: dict[str, MatchRecord] | None = None) -> None:
        self.records = dict(records or {})

    def save_match(self, record: MatchRecord) -> None:
        if record.match_id in self.records:
            raise ValueError("duplicate match")
        self.records[record.match_id] = record

    def load_match(self, match_id: str) -> MatchRecord:
        return self.records[match_id]


def _protocol() -> H4HoldoutVarianceProtocol:
    return load_h4_holdout_variance_protocol(PROTOCOL_PATH)


def _job(
    protocol: H4HoldoutVarianceProtocol,
    *,
    arm_id: str,
    unit_id: str = "h4_holdout_000",
    seed: int = 6_400_000,
) -> H4HoldoutVarianceJob:
    return next(
        job
        for job in iter_h4_holdout_variance_jobs(protocol)
        if job.arm_id == arm_id
        and job.holdout_unit_id == unit_id
        and job.master_seed == seed
    )


def _matched_jobs(
    *,
    unit_id: str = "h4_holdout_000",
    seed: int = 6_400_000,
) -> tuple[H4HoldoutVarianceProtocol, H4HoldoutVarianceJob, H4HoldoutVarianceJob]:
    protocol = _protocol()
    return (
        protocol,
        _job(protocol, arm_id="treatment", unit_id=unit_id, seed=seed),
        _job(protocol, arm_id="control", unit_id=unit_id, seed=seed),
    )


@pytest.fixture(scope="module")
def valid_pair_run() -> tuple[
    H4HoldoutVarianceProtocol,
    H4HoldoutVarianceJob,
    H4HoldoutVarianceJob,
    PairedResult,
    PairedResult,
    _MemoryMatchStore,
]:
    protocol, treatment_job, control_job = _matched_jobs()
    store = _MemoryMatchStore()
    treatment = PairedRunner(
        agent_a=_LifecycleProbeAgent(),
        agent_b=_CallCheckAgent(),
        num_hands=500,
        seed=treatment_job.master_seed,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
        match_id="h4-variance-treatment-fast",
        benchmark_store=store,
    ).run()
    control = PairedRunner(
        agent_a=_FrozenControlProbe(),
        agent_b=_CallCheckAgent(),
        num_hands=500,
        seed=control_job.master_seed,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
        match_id="h4-variance-control-fast",
        benchmark_store=store,
    ).run()
    return protocol, treatment_job, control_job, treatment, control, store


def _fake_pair(
    *,
    unit_id: str,
    seed: int,
    contrast: float,
    valid: bool = True,
) -> H4HoldoutVariancePairExecution:
    treatment_evidence = None
    control_evidence = None
    unit_contrast = None
    if valid:
        treatment_evidence = H4HoldoutVarianceArmEvidence(
            "treatment", unit_id, seed, contrast * 5.0, contrast * 5.0
        )
        control_evidence = H4HoldoutVarianceArmEvidence(
            "control", unit_id, seed, 0.0, 0.0
        )
        unit_contrast = contrast
    return H4HoldoutVariancePairExecution(
        holdout_unit_id=unit_id,
        master_seed=seed,
        hands_per_position=500,
        scientifically_valid=valid,
        treatment_match_id=f"t-{unit_id}-{seed}",
        control_match_id=f"c-{unit_id}-{seed}",
        treatment_record_hash="a" * 64,
        control_record_hash="b" * 64,
        treatment_failed_runs=() if valid else ({"details": "failed"},),
        control_failed_runs=(),
        treatment_learning_hands_by_position=(("A_BTN", 500), ("B_BTN", 500))
        if valid
        else (),
        treatment_evidence=treatment_evidence,
        control_evidence=control_evidence,
        unit_contrast_bb100=unit_contrast,
    )


def _cluster(
    protocol: H4HoldoutVarianceProtocol,
    *,
    seed: int,
    contrast: float,
):
    return admit_h4_holdout_variance_population_cluster(
        protocol=protocol,
        master_seed=seed,
        pair_executions=tuple(
            _fake_pair(unit_id=unit, seed=seed, contrast=contrast)
            for unit in H4_HOLDOUT_UNIT_IDS
        ),
    )


def test_matched_jobs_must_be_exact_frozen_holdout_unit_seed_cell() -> None:
    protocol, treatment, control = _matched_jobs()
    matched = validate_matched_h4_holdout_variance_jobs(
        protocol=protocol,
        treatment_job=treatment,
        control_job=control,
    )
    assert matched.holdout_unit_id == "h4_holdout_000"
    assert matched.master_seed == 6_400_000
    assert matched.hands_per_position == 500

    wrong_unit = _job(
        protocol,
        arm_id="control",
        unit_id="h4_holdout_001",
        seed=6_400_000,
    )
    with pytest.raises(H4HoldoutVarianceExecutionError, match="same holdout unit"):
        validate_matched_h4_holdout_variance_jobs(
            protocol=protocol,
            treatment_job=treatment,
            control_job=wrong_unit,
        )


def test_noncanonical_or_replacement_seed_job_is_rejected() -> None:
    protocol, treatment, control = _matched_jobs()
    replacement = replace(treatment, master_seed=6_400_010)
    with pytest.raises(H4HoldoutVarianceExecutionError, match="canonical frozen Spec-37 job"):
        validate_matched_h4_holdout_variance_jobs(
            protocol=protocol,
            treatment_job=replacement,
            control_job=control,
        )


def test_frozen_hero_factory_builds_only_spec37_control_and_treatment() -> None:
    protocol, treatment, control = _matched_jobs()
    reference = load_preflop_strength_reference(REFERENCE_PATH)

    control_hero = build_h4_holdout_variance_hero(control, reference=reference)
    treatment_hero = build_h4_holdout_variance_hero(treatment, reference=reference)

    assert type(control_hero) is HandStrengthBaselineAgent
    assert control_hero.is_adaptive is False
    assert type(treatment_hero) is AdaptiveHandStrengthAgent
    assert treatment_hero.is_adaptive is True
    assert treatment_hero.model.state_hash == AdaptiveHandStrengthAgent.initial_model().state_hash


@pytest.mark.parametrize("arm_id", ["control", "treatment"])
def test_runner_builder_forces_500_hand_full_match_trajectory_cluster(
    arm_id: str,
) -> None:
    protocol = _protocol()
    job = _job(protocol, arm_id=arm_id)
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    store = _MemoryMatchStore()

    runner = build_h4_holdout_variance_runner(
        protocol=protocol,
        job=job,
        reference=reference,
        opponent=_CallCheckAgent(),
        game_config=BASELINE_GAME_CONFIG,
        match_id=f"guarded-h4-{arm_id}",
        benchmark_store=store,
    )

    assert runner.adaptive is True
    assert runner.statistical_unit is StatisticalUnit.TRAJECTORY_CLUSTER
    assert runner.num_hands == 500
    assert isinstance(runner.agent_a, H4HoldoutVarianceHeroBoundaryAgent)
    assert runner.agent_a.is_adaptive is job.adaptive


def test_treatment_runner_resets_frozen_prior_before_both_position_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    job = _job(protocol, arm_id="treatment")
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    runner = build_h4_holdout_variance_runner(
        protocol=protocol,
        job=job,
        reference=reference,
        opponent=_CallCheckAgent(),
        game_config=BASELINE_GAME_CONFIG,
        match_id="h4-reset-proof",
        benchmark_store=_MemoryMatchStore(),
    )
    assert isinstance(runner.agent_a, H4HoldoutVarianceHeroBoundaryAgent)
    delegate = runner.agent_a.delegate
    assert isinstance(delegate, AdaptiveHandStrengthAgent)
    initial_hash = AdaptiveHandStrengthAgent.initial_model().state_hash
    reset_hashes: list[str] = []
    original_reset = delegate.reset

    def counted_reset() -> None:
        original_reset()
        reset_hashes.append(delegate.model.state_hash)

    monkeypatch.setattr(delegate, "reset", counted_reset)
    monkeypatch.setattr(runner, "_run_full_match", lambda **kwargs: (0.0, 0.0, []))
    runner.run()
    assert reset_hashes == [initial_hash, initial_hash]


def test_information_boundary_rejects_holdout_controller_fields() -> None:
    state = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=((0, 1), (2, 3)),
        board=(4, 5, 6, 7, 8),
        dealer=0,
    )
    observation = state.observation(state.current_player)
    validate_h4_holdout_variance_hero_observation(observation)

    leaked_id = replace(observation)
    object.__setattr__(leaked_id, "holdout_unit_id", "h4_holdout_000")
    with pytest.raises(H4HoldoutVarianceExecutionError, match="forbidden or unknown"):
        validate_h4_holdout_variance_hero_observation(leaked_id)

    leaked_parameters = replace(observation)
    object.__setattr__(
        leaked_parameters,
        "true_opponent_parameter_vector",
        {"vpip": 0.36363},
    )
    with pytest.raises(H4HoldoutVarianceExecutionError, match="forbidden or unknown"):
        validate_h4_holdout_variance_hero_observation(leaked_parameters)


def test_matched_crn_proof_is_identical_across_arms_and_holdout_units() -> None:
    protocol, treatment_0, control_0 = _matched_jobs(unit_id="h4_holdout_000")
    proof_0 = prove_h4_holdout_variance_matched_deal_schedule(
        protocol=protocol,
        treatment_job=treatment_0,
        control_job=control_0,
    )
    _, treatment_4, control_4 = _matched_jobs(unit_id="h4_holdout_004")
    proof_4 = prove_h4_holdout_variance_matched_deal_schedule(
        protocol=protocol,
        treatment_job=treatment_4,
        control_job=control_4,
    )
    assert proof_0.master_seed == 6_400_000
    assert proof_0.hands_per_position == 500
    assert proof_0.deal_schedule_hash == proof_4.deal_schedule_hash
    assert proof_0.deal_relation == "identical_defined_card_realisations"
    assert proof_0.full_action_path_identity == "not_claimed"


def test_valid_complete_pair_is_durable_replayed_and_admitted(
    valid_pair_run,
) -> None:  # type: ignore[no-untyped-def]
    protocol, treatment_job, control_job, treatment, control, store = valid_pair_run
    finalized = finalize_h4_holdout_variance_pair(
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
    assert finalized.treatment_evidence.hands_per_position == 500
    assert finalized.control_evidence.hands_per_position == 500
    assert finalized.treatment_learning_hands_by_position == (
        ("A_BTN", 500),
        ("B_BTN", 500),
    )
    assert finalized.unit_contrast_bb100 is not None


def test_incomplete_arm_remains_durable_but_cannot_yield_unit_contrast(
    valid_pair_run,
) -> None:  # type: ignore[no-untyped-def]
    protocol, treatment_job, control_job, treatment, control, store = valid_pair_run
    failed_control = replace(
        control,
        paired_differences=[],
        num_valid_pairs=0,
        failed_runs=[{"hand": 233, "config": "A_BTN", "details": "failed"}],
    )
    finalized = finalize_h4_holdout_variance_pair(
        protocol=protocol,
        treatment_job=treatment_job,
        control_job=control_job,
        treatment_result=treatment,
        control_result=failed_control,
        benchmark_store=store,
    )
    assert finalized.scientifically_valid is False
    assert finalized.unit_contrast_bb100 is None
    assert finalized.treatment_evidence is None
    assert finalized.control_evidence is None
    assert finalized.control_failed_runs == (
        {"hand": 233, "config": "A_BTN", "details": "failed"},
    )


def test_finalization_rejects_non_durable_match_record(valid_pair_run) -> None:  # type: ignore[no-untyped-def]
    protocol, treatment_job, control_job, treatment, control, store = valid_pair_run
    reduced = _MemoryMatchStore(store.records)
    assert control.match_record is not None
    del reduced.records[control.match_record.match_id]
    with pytest.raises(H4HoldoutVarianceExecutionError, match="durably persisted"):
        finalize_h4_holdout_variance_pair(
            protocol=protocol,
            treatment_job=treatment_job,
            control_job=control_job,
            treatment_result=treatment,
            control_result=control,
            benchmark_store=reduced,
        )


def test_finalization_rejects_tampered_adaptive_learning_chain(valid_pair_run) -> None:  # type: ignore[no-untyped-def]
    protocol, treatment_job, control_job, treatment, control, store = valid_pair_run
    assert treatment.match_record is not None
    assert control.match_record is not None
    hands = list(treatment.match_record.hands)
    event = hands[1].adaptive_learning_events[0]
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
    tampered_event = RecordedAdaptiveLearningEvent(**core, transition_hash=transition_hash)
    hands[1] = replace(hands[1], adaptive_learning_events=(tampered_event,))
    tampered_record = replace(treatment.match_record, hands=tuple(hands))
    tampered_result = replace(treatment, match_record=tampered_record)
    tampered_store = _MemoryMatchStore(store.records)
    tampered_store.records[tampered_record.match_id] = tampered_record

    with pytest.raises(ValueError, match="before-state hash mismatch"):
        finalize_h4_holdout_variance_pair(
            protocol=protocol,
            treatment_job=treatment_job,
            control_job=control_job,
            treatment_result=tampered_result,
            control_result=control,
            benchmark_store=tampered_store,
        )


def test_control_record_must_never_contain_adaptive_learning_events(valid_pair_run) -> None:  # type: ignore[no-untyped-def]
    protocol, treatment_job, control_job, treatment, control, store = valid_pair_run
    assert treatment.match_record is not None
    assert control.match_record is not None
    source_event = treatment.match_record.hands[0].adaptive_learning_events[0]
    control_hands = list(control.match_record.hands)
    control_hands[0] = replace(
        control_hands[0],
        adaptive_learning_events=(source_event,),
    )
    contaminated = replace(control.match_record, hands=tuple(control_hands))
    contaminated_result = replace(control, match_record=contaminated)
    contaminated_store = _MemoryMatchStore(store.records)
    contaminated_store.records[contaminated.match_id] = contaminated

    with pytest.raises(H4HoldoutVarianceExecutionError, match="control arm.*adaptive learning"):
        finalize_h4_holdout_variance_pair(
            protocol=protocol,
            treatment_job=treatment_job,
            control_job=control_job,
            treatment_result=treatment,
            control_result=contaminated_result,
            benchmark_store=contaminated_store,
        )


def test_population_cluster_requires_exactly_all_five_units() -> None:
    protocol = _protocol()
    seed = H4_VARIANCE_MASTER_SEEDS[0]
    pairs = tuple(
        _fake_pair(unit_id=unit, seed=seed, contrast=float(index + 1))
        for index, unit in enumerate(H4_HOLDOUT_UNIT_IDS)
    )
    cluster = admit_h4_holdout_variance_population_cluster(
        protocol=protocol,
        master_seed=seed,
        pair_executions=pairs,
    )
    assert tuple(unit for unit, _ in cluster.unit_contrasts_bb100) == H4_HOLDOUT_UNIT_IDS
    assert cluster.population_contrast_bb100 == pytest.approx(3.0)

    with pytest.raises(H4HoldoutVarianceExecutionError, match="exactly five"):
        admit_h4_holdout_variance_population_cluster(
            protocol=protocol,
            master_seed=seed,
            pair_executions=pairs[:-1],
        )


def test_one_invalid_pair_invalidates_entire_seed_population_cluster() -> None:
    protocol = _protocol()
    seed = H4_VARIANCE_MASTER_SEEDS[0]
    pairs = [
        _fake_pair(unit_id=unit, seed=seed, contrast=1.0)
        for unit in H4_HOLDOUT_UNIT_IDS
    ]
    pairs[2] = _fake_pair(
        unit_id=H4_HOLDOUT_UNIT_IDS[2],
        seed=seed,
        contrast=1.0,
        valid=False,
    )
    with pytest.raises(H4HoldoutVarianceExecutionError, match="invalid unit pair"):
        admit_h4_holdout_variance_population_cluster(
            protocol=protocol,
            master_seed=seed,
            pair_executions=pairs,
        )


def test_population_cluster_rejects_duplicate_unit_or_mixed_seed() -> None:
    protocol = _protocol()
    seed = H4_VARIANCE_MASTER_SEEDS[0]
    pairs = [
        _fake_pair(unit_id=unit, seed=seed, contrast=1.0)
        for unit in H4_HOLDOUT_UNIT_IDS
    ]
    duplicate = list(pairs)
    duplicate[-1] = _fake_pair(
        unit_id=H4_HOLDOUT_UNIT_IDS[0], seed=seed, contrast=1.0
    )
    with pytest.raises(H4HoldoutVarianceExecutionError, match="duplicate holdout unit"):
        admit_h4_holdout_variance_population_cluster(
            protocol=protocol,
            master_seed=seed,
            pair_executions=duplicate,
        )

    mixed = list(pairs)
    mixed[-1] = _fake_pair(
        unit_id=H4_HOLDOUT_UNIT_IDS[-1],
        seed=H4_VARIANCE_MASTER_SEEDS[1],
        contrast=1.0,
    )
    with pytest.raises(H4HoldoutVarianceExecutionError, match="same master seed"):
        admit_h4_holdout_variance_population_cluster(
            protocol=protocol,
            master_seed=seed,
            pair_executions=mixed,
        )


def test_campaign_admission_releases_sigma_and_plan_only_after_exact_ten_clusters() -> None:
    protocol = _protocol()
    clusters = tuple(
        _cluster(protocol, seed=seed, contrast=float(index))
        for index, seed in enumerate(H4_VARIANCE_MASTER_SEEDS)
    )
    admitted = admit_h4_holdout_variance_campaign(
        protocol=protocol,
        clusters=clusters,
    )
    assert len(admitted.clusters) == 10
    assert tuple(seed for seed, _ in admitted.seed_contrasts_bb100) == H4_VARIANCE_MASTER_SEEDS
    assert admitted.sigma_500 > 0.0
    assert admitted.sample_size_plan.n_final >= 10
    assert not hasattr(admitted, "mean_difference_bb100")


def test_campaign_rejects_nine_of_ten_even_when_all_nine_are_valid() -> None:
    protocol = _protocol()
    clusters = tuple(
        _cluster(protocol, seed=seed, contrast=float(index))
        for index, seed in enumerate(H4_VARIANCE_MASTER_SEEDS[:-1])
    )
    with pytest.raises(H4HoldoutVarianceExecutionError, match="exactly ten valid"):
        admit_h4_holdout_variance_campaign(protocol=protocol, clusters=clusters)


def test_campaign_rejects_replacement_seed_instead_of_missing_frozen_seed() -> None:
    protocol = _protocol()
    clusters = [
        _cluster(protocol, seed=seed, contrast=float(index))
        for index, seed in enumerate(H4_VARIANCE_MASTER_SEEDS)
    ]
    clusters[-1] = replace(clusters[-1], master_seed=6_400_010)
    with pytest.raises(H4HoldoutVarianceExecutionError, match="frozen Spec-37 seeds"):
        admit_h4_holdout_variance_campaign(protocol=protocol, clusters=clusters)


def test_boundary_wrapper_forwards_only_public_learning_counts() -> None:
    probe = _LifecycleProbeAgent()
    guarded = H4HoldoutVarianceHeroBoundaryAgent(probe)
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


def test_runner_builder_rejects_adaptive_holdout_opponent() -> None:
    protocol = _protocol()
    job = _job(protocol, arm_id="control")
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    with pytest.raises(H4HoldoutVarianceExecutionError, match="must be stationary"):
        build_h4_holdout_variance_runner(
            protocol=protocol,
            job=job,
            reference=reference,
            opponent=_LifecycleProbeAgent(),
            game_config=BASELINE_GAME_CONFIG,
            match_id="adaptive-opponent-forbidden",
            benchmark_store=_MemoryMatchStore(),
        )


def test_population_cluster_recomputes_and_rejects_tampered_unit_contrast() -> None:
    protocol = _protocol()
    seed = H4_VARIANCE_MASTER_SEEDS[0]
    pairs = [
        _fake_pair(unit_id=unit, seed=seed, contrast=1.0)
        for unit in H4_HOLDOUT_UNIT_IDS
    ]
    pairs[0] = replace(pairs[0], unit_contrast_bb100=999.0)
    with pytest.raises(H4HoldoutVarianceExecutionError, match="differs from replay-admitted"):
        admit_h4_holdout_variance_population_cluster(
            protocol=protocol,
            master_seed=seed,
            pair_executions=pairs,
        )


def test_campaign_recomputes_and_rejects_tampered_population_contrast() -> None:
    protocol = _protocol()
    clusters = [
        _cluster(protocol, seed=seed, contrast=float(index))
        for index, seed in enumerate(H4_VARIANCE_MASTER_SEEDS)
    ]
    clusters[0] = replace(clusters[0], population_contrast_bb100=999.0)
    with pytest.raises(H4HoldoutVarianceExecutionError, match="differs from replay-admitted"):
        admit_h4_holdout_variance_campaign(protocol=protocol, clusters=clusters)
