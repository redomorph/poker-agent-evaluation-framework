"""Execution enforcement and replay admission for frozen Spec-37 variance evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pokerbench.agents.adaptive_lifecycle import (
    ADAPTIVE_HAND_STRENGTH_AGENT_VERSION,
    AdaptiveHandStrengthAgent,
    replay_adaptive_learning_for_bot,
)
from pokerbench.agents.base import Agent
from pokerbench.agents.hand_strength_baseline import (
    HAND_STRENGTH_BASELINE_VERSION,
    HandStrengthBaselineAgent,
)
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.opponent_range import PreflopStrengthReference
from pokerbench.engine.config import Action, GameConfig, config_hash
from pokerbench.engine.deal_schedule import DealSchedule
from pokerbench.evaluation.calibration import BehaviorCounts
from pokerbench.experiments.h4_holdout_variance_analysis import (
    H4FinalSampleSizePlan,
    matched_unit_contrast_bb100,
    population_contrast_for_seed,
    variance_pilot_plan_from_complete_seed_contrasts,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4_HOLDOUT_UNIT_IDS,
    H4_VARIANCE_MASTER_SEEDS,
    H4HoldoutVarianceJob,
    H4HoldoutVarianceProtocol,
    expected_h4_holdout_variance_payload,
    iter_h4_holdout_variance_jobs,
)
from pokerbench.replay.archive import MatchRecord, RecordedAdaptiveLearningEvent
from pokerbench.rng import AgentRNG
from pokerbench.runner import (
    IllegalActionPolicy,
    PairedResult,
    PairedRunner,
    StatisticalUnit,
)


H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION = (
    "phase2-h4-holdout-variance-execution-enforcement-v1"
)

_ALLOWED_OBSERVATION_FIELDS = frozenset(
    {
        "player_id",
        "hole_cards",
        "board",
        "stacks",
        "current_bets",
        "pot",
        "current_player",
        "street",
        "action_history",
        "legal_actions",
        "dealer",
        "public_action_history",
    }
)


class H4HoldoutVarianceExecutionError(ValueError):
    """Raised when Spec-37 execution/replay evidence violates a frozen invariant."""


@runtime_checkable
class DurableMatchStore(Protocol):
    """Minimal durability contract required before replay admission."""

    def save_match(self, record: MatchRecord) -> None: ...

    def load_match(self, match_id: str) -> MatchRecord: ...


@dataclass(frozen=True)
class MatchedH4HoldoutVarianceJobs:
    holdout_unit_id: str
    master_seed: int
    hands_per_position: int
    treatment_job: H4HoldoutVarianceJob
    control_job: H4HoldoutVarianceJob


@dataclass(frozen=True)
class H4HoldoutVarianceDealScheduleProof:
    master_seed: int
    hands_per_position: int
    deal_schedule_hash: str
    deal_relation: str = "identical_defined_card_realisations"
    agent_rng_relation: str = "event_indexed_common_identity_where_events_correspond"
    full_action_path_identity: str = "not_claimed"


@dataclass(frozen=True)
class H4HoldoutVarianceArmEvidence:
    arm_id: str
    holdout_unit_id: str
    master_seed: int
    hero_btn_gain_bb: float
    hero_bb_gain_bb: float
    hands_per_position: int = 500


@dataclass(frozen=True)
class H4HoldoutVariancePairExecution:
    holdout_unit_id: str
    master_seed: int
    hands_per_position: int
    scientifically_valid: bool
    treatment_match_id: str
    control_match_id: str
    treatment_record_hash: str
    control_record_hash: str
    treatment_failed_runs: tuple[Mapping[str, Any], ...]
    control_failed_runs: tuple[Mapping[str, Any], ...]
    treatment_learning_hands_by_position: tuple[tuple[str, int], ...]
    treatment_evidence: H4HoldoutVarianceArmEvidence | None
    control_evidence: H4HoldoutVarianceArmEvidence | None
    unit_contrast_bb100: float | None
    enforcement_version: str = H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION


@dataclass(frozen=True)
class H4HoldoutVariancePopulationCluster:
    master_seed: int
    pair_executions: tuple[H4HoldoutVariancePairExecution, ...]
    unit_contrasts_bb100: tuple[tuple[str, float], ...]
    population_contrast_bb100: float
    enforcement_version: str = H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION


@dataclass(frozen=True)
class H4HoldoutVarianceCampaignAdmission:
    clusters: tuple[H4HoldoutVariancePopulationCluster, ...]
    seed_contrasts_bb100: tuple[tuple[int, float], ...]
    sample_size_plan: H4FinalSampleSizePlan
    enforcement_version: str = H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION

    @property
    def sigma_500(self) -> float:
        return self.sample_size_plan.sigma_500


class H4HoldoutVarianceHeroBoundaryAgent(Agent):
    """Guard hero decision/learning interfaces from controller-only holdout data."""

    def __init__(self, delegate: Agent) -> None:
        if not isinstance(delegate, Agent):
            raise TypeError("delegate must be an Agent")
        self._delegate = delegate

    @property
    def delegate(self) -> Agent:
        return self._delegate

    @property
    def is_adaptive(self) -> bool:
        return self._delegate.is_adaptive

    def reset(self) -> None:
        self._delegate.reset()

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        validate_h4_holdout_variance_hero_observation(observation)
        return self._delegate.act(observation, legal_actions, rng)

    def observe_opponent_hand(
        self,
        counts: BehaviorCounts,
        *,
        learner_bot_id: int,
        opponent_bot_id: int,
        trajectory_hand_index: int,
    ) -> RecordedAdaptiveLearningEvent:
        if not isinstance(counts, BehaviorCounts):
            raise TypeError("adaptive learning input must be public BehaviorCounts")
        if not self._delegate.is_adaptive:
            raise H4HoldoutVarianceExecutionError(
                "control arm cannot receive adaptive learning updates"
            )
        observe = getattr(self._delegate, "observe_opponent_hand", None)
        if not callable(observe):
            raise H4HoldoutVarianceExecutionError(
                "adaptive treatment must implement the frozen lifecycle learning hook"
            )
        event = observe(
            counts,
            learner_bot_id=learner_bot_id,
            opponent_bot_id=opponent_bot_id,
            trajectory_hand_index=trajectory_hand_index,
        )
        if not isinstance(event, RecordedAdaptiveLearningEvent):
            raise H4HoldoutVarianceExecutionError(
                "adaptive lifecycle hook must return RecordedAdaptiveLearningEvent"
            )
        return event


def validate_h4_holdout_variance_hero_observation(
    observation: AgentObservation,
) -> None:
    """Reject any controller-only or newly injected hero-observation field."""
    if type(observation) is not AgentObservation:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot hero observation must be the frozen AgentObservation type"
        )
    if frozenset(vars(observation)) != _ALLOWED_OBSERVATION_FIELDS:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot hero observation contains a forbidden or unknown field"
        )


def _require_protocol(protocol: H4HoldoutVarianceProtocol) -> None:
    if not isinstance(protocol, H4HoldoutVarianceProtocol):
        raise TypeError("protocol must be H4HoldoutVarianceProtocol")


def _require_canonical_job(
    protocol: H4HoldoutVarianceProtocol,
    job: H4HoldoutVarianceJob,
) -> None:
    if not isinstance(job, H4HoldoutVarianceJob):
        raise TypeError("job must be H4HoldoutVarianceJob")
    if job not in iter_h4_holdout_variance_jobs(protocol):
        raise H4HoldoutVarianceExecutionError(
            "job must be an exact canonical frozen Spec-37 job; replacement seeds are forbidden"
        )


def validate_matched_h4_holdout_variance_jobs(
    *,
    protocol: H4HoldoutVarianceProtocol,
    treatment_job: H4HoldoutVarianceJob,
    control_job: H4HoldoutVarianceJob,
) -> MatchedH4HoldoutVarianceJobs:
    """Require one exact frozen treatment/control holdout-unit/seed pair."""
    _require_protocol(protocol)
    _require_canonical_job(protocol, treatment_job)
    _require_canonical_job(protocol, control_job)
    if treatment_job.arm_id != "treatment" or not treatment_job.adaptive:
        raise H4HoldoutVarianceExecutionError(
            "treatment_job must be the frozen adaptive arm"
        )
    if control_job.arm_id != "control" or control_job.adaptive:
        raise H4HoldoutVarianceExecutionError(
            "control_job must be the frozen static arm"
        )
    if treatment_job.holdout_unit_id != control_job.holdout_unit_id:
        raise H4HoldoutVarianceExecutionError(
            "matched jobs must use the same holdout unit"
        )
    if treatment_job.master_seed != control_job.master_seed:
        raise H4HoldoutVarianceExecutionError(
            "matched jobs must use the same master seed"
        )
    if treatment_job.hands_per_position != control_job.hands_per_position:
        raise H4HoldoutVarianceExecutionError(
            "matched jobs must use the same trajectory horizon"
        )
    if treatment_job.hands_per_position != 500:
        raise H4HoldoutVarianceExecutionError(
            "Spec-37 matched jobs require exactly 500 hands per position"
        )
    if treatment_job.position_run_count != 2 or control_job.position_run_count != 2:
        raise H4HoldoutVarianceExecutionError(
            "matched jobs must contain exactly two position runs"
        )
    return MatchedH4HoldoutVarianceJobs(
        holdout_unit_id=treatment_job.holdout_unit_id,
        master_seed=treatment_job.master_seed,
        hands_per_position=treatment_job.hands_per_position,
        treatment_job=treatment_job,
        control_job=control_job,
    )


def build_h4_holdout_variance_hero(
    job: H4HoldoutVarianceJob,
    *,
    reference: PreflopStrengthReference,
) -> Agent:
    """Build exactly the frozen Spec-37 control or treatment hero."""
    if not isinstance(job, H4HoldoutVarianceJob):
        raise TypeError("job must be H4HoldoutVarianceJob")
    if job.arm_id == "control":
        if job.adaptive or job.agent_version != HAND_STRENGTH_BASELINE_VERSION:
            raise H4HoldoutVarianceExecutionError("invalid frozen control-arm binding")
        return HandStrengthBaselineAgent()
    if job.arm_id == "treatment":
        if not job.adaptive or job.agent_version != ADAPTIVE_HAND_STRENGTH_AGENT_VERSION:
            raise H4HoldoutVarianceExecutionError("invalid frozen treatment-arm binding")
        return AdaptiveHandStrengthAgent(reference)
    raise H4HoldoutVarianceExecutionError("unknown Spec-37 arm")


def _require_durable_store(store: object) -> None:
    if not isinstance(store, DurableMatchStore):
        raise H4HoldoutVarianceExecutionError(
            "Spec-37 execution requires a durable save/load match store"
        )


def build_h4_holdout_variance_runner(
    *,
    protocol: H4HoldoutVarianceProtocol,
    job: H4HoldoutVarianceJob,
    reference: PreflopStrengthReference,
    opponent: Agent,
    game_config: GameConfig,
    match_id: str,
    benchmark_store: DurableMatchStore,
    experiment_metadata: Mapping[str, Any] | None = None,
) -> PairedRunner:
    """Build a guarded 500+500 arm runner without authorizing campaign execution."""
    _require_protocol(protocol)
    _require_canonical_job(protocol, job)
    if not isinstance(opponent, Agent):
        raise TypeError("opponent must be an Agent")
    if opponent.is_adaptive:
        raise H4HoldoutVarianceExecutionError(
            "Spec-37 holdout opponents must be stationary"
        )
    expected_game_hash = str(
        expected_h4_holdout_variance_payload()["bindings"]["game"]["game_config_hash"]
    )
    if config_hash(game_config) != expected_game_hash:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot game configuration differs from frozen Spec 37"
        )
    _require_durable_store(benchmark_store)

    hero = H4HoldoutVarianceHeroBoundaryAgent(
        build_h4_holdout_variance_hero(job, reference=reference)
    )
    runner = PairedRunner(
        agent_a=hero,
        agent_b=opponent,
        num_hands=job.hands_per_position,
        seed=job.master_seed,
        game_config=game_config,
        experiment_metadata=dict(experiment_metadata or {}),
        adaptive=True,
        illegal_action_policy=IllegalActionPolicy.EXCLUDE_RUN,
        match_id=match_id,
        benchmark_store=benchmark_store,
    )
    if runner.statistical_unit is not StatisticalUnit.TRAJECTORY_CLUSTER:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot arm runner must use trajectory-cluster statistical units"
        )
    return runner


def _deal_schedule_hash(job: H4HoldoutVarianceJob) -> str:
    schedule = DealSchedule(
        master_seed=job.master_seed,
        num_hands=job.hands_per_position,
    )
    payload = [
        [list(hole_a), list(hole_b), list(board)]
        for hole_a, hole_b, board in schedule
    ]
    canonical = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def prove_h4_holdout_variance_matched_deal_schedule(
    *,
    protocol: H4HoldoutVarianceProtocol,
    treatment_job: H4HoldoutVarianceJob,
    control_job: H4HoldoutVarianceJob,
) -> H4HoldoutVarianceDealScheduleProof:
    """Prove the defined-card CRN relation for one matched Spec-37 unit pair."""
    matched = validate_matched_h4_holdout_variance_jobs(
        protocol=protocol,
        treatment_job=treatment_job,
        control_job=control_job,
    )
    treatment_hash = _deal_schedule_hash(treatment_job)
    control_hash = _deal_schedule_hash(control_job)
    if treatment_hash != control_hash:
        raise H4HoldoutVarianceExecutionError(
            "matched treatment/control jobs do not share identical card realisations"
        )
    return H4HoldoutVarianceDealScheduleProof(
        master_seed=matched.master_seed,
        hands_per_position=matched.hands_per_position,
        deal_schedule_hash=treatment_hash,
    )


def _require_arm_result(
    *,
    job: H4HoldoutVarianceJob,
    result: PairedResult,
) -> tuple[bool, MatchRecord]:
    if not isinstance(result, PairedResult):
        raise TypeError("result must be PairedResult")
    if result.statistical_unit is not StatisticalUnit.TRAJECTORY_CLUSTER:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot arm result has wrong statistical unit"
        )
    if result.seed != job.master_seed or result.num_hands != 500:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot arm result does not match its frozen job"
        )
    record = result.match_record
    if record is None:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot arm result must retain a MatchRecord"
        )
    if record.master_seed != job.master_seed:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot MatchRecord seed differs from frozen job"
        )

    valid_shape = (
        result.num_valid_pairs == 1
        and len(result.paired_differences) == 1
        and not result.failed_runs
    )
    failed_shape = (
        result.num_valid_pairs == 0
        and not result.paired_differences
        and bool(result.failed_runs)
    )
    if not (valid_shape or failed_shape):
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot arm result has inconsistent validity fields"
        )
    if valid_shape and len(record.hands) != 1000:
        raise H4HoldoutVarianceExecutionError(
            "valid variance-pilot arm must persist exactly 500+500 position hands"
        )
    if len(record.hands) > 1000:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot MatchRecord contains too many hands"
        )
    return valid_shape, record


def _require_persisted_record(store: DurableMatchStore, record: MatchRecord) -> None:
    _require_durable_store(store)
    try:
        persisted = store.load_match(record.match_id)
    except Exception as exc:
        raise H4HoldoutVarianceExecutionError(
            "variance-pilot MatchRecord is not durably persisted"
        ) from exc
    if not isinstance(persisted, MatchRecord) or persisted.record_hash != record.record_hash:
        raise H4HoldoutVarianceExecutionError(
            "durably persisted variance-pilot MatchRecord differs from runner evidence"
        )


def _control_has_learning_events(record: MatchRecord) -> bool:
    return any(hand.adaptive_learning_events for hand in record.hands)


def _extract_hero_gains(record: MatchRecord) -> tuple[float, float]:
    by_config: dict[str, list[tuple[int, float]]] = {"A_BTN": [], "B_BTN": []}
    for hand in record.hands:
        if hand.config_id not in by_config:
            raise H4HoldoutVarianceExecutionError(
                "unexpected position config in variance-pilot MatchRecord"
            )
        if hand.terminal_profit_bb is None:
            raise H4HoldoutVarianceExecutionError(
                "valid variance-pilot hand must retain terminal profit"
            )
        if hand.seat0_bot_id == 0:
            hero_profit = float(hand.terminal_profit_bb[0])
        elif hand.seat1_bot_id == 0:
            hero_profit = float(hand.terminal_profit_bb[1])
        else:
            raise H4HoldoutVarianceExecutionError(
                "variance-pilot hand does not contain stable hero bot_id=0"
            )
        by_config[hand.config_id].append((hand.hand_index, hero_profit))

    gains: list[float] = []
    for config_id in ("A_BTN", "B_BTN"):
        items = sorted(by_config[config_id])
        if tuple(index for index, _ in items) != tuple(range(500)):
            raise H4HoldoutVarianceExecutionError(
                "valid variance-pilot position run must contain hands 0..499 exactly once"
            )
        gains.append(sum(value for _, value in items))
    return gains[0], gains[1]


def _learning_hand_counts(record: MatchRecord) -> tuple[tuple[str, int], ...]:
    models = replay_adaptive_learning_for_bot(record, bot_id=0)
    return tuple(
        (config_id, models[config_id].hands_observed)
        for config_id in ("A_BTN", "B_BTN")
        if config_id in models
    )


def finalize_h4_holdout_variance_pair(
    *,
    protocol: H4HoldoutVarianceProtocol,
    treatment_job: H4HoldoutVarianceJob,
    control_job: H4HoldoutVarianceJob,
    treatment_result: PairedResult,
    control_result: PairedResult,
    benchmark_store: DurableMatchStore,
) -> H4HoldoutVariancePairExecution:
    """Require durability + replay before a matched unit pair can yield analysis evidence."""
    matched = validate_matched_h4_holdout_variance_jobs(
        protocol=protocol,
        treatment_job=treatment_job,
        control_job=control_job,
    )
    treatment_valid, treatment_record = _require_arm_result(
        job=treatment_job,
        result=treatment_result,
    )
    control_valid, control_record = _require_arm_result(
        job=control_job,
        result=control_result,
    )
    _require_persisted_record(benchmark_store, treatment_record)
    _require_persisted_record(benchmark_store, control_record)

    if _control_has_learning_events(control_record):
        raise H4HoldoutVarianceExecutionError(
            "control arm MatchRecord must not contain adaptive learning events"
        )

    # Replay is mandatory even though the variance-pilot mean is not a design input.
    learning_counts = _learning_hand_counts(treatment_record)
    if treatment_valid and learning_counts != (("A_BTN", 500), ("B_BTN", 500)):
        raise H4HoldoutVarianceExecutionError(
            "valid adaptive treatment must replay exactly 500 learning transitions per position"
        )

    scientifically_valid = treatment_valid and control_valid
    treatment_evidence: H4HoldoutVarianceArmEvidence | None = None
    control_evidence: H4HoldoutVarianceArmEvidence | None = None
    unit_contrast: float | None = None
    if scientifically_valid:
        treatment_btn, treatment_bb = _extract_hero_gains(treatment_record)
        control_btn, control_bb = _extract_hero_gains(control_record)
        treatment_evidence = H4HoldoutVarianceArmEvidence(
            arm_id="treatment",
            holdout_unit_id=matched.holdout_unit_id,
            master_seed=matched.master_seed,
            hero_btn_gain_bb=treatment_btn,
            hero_bb_gain_bb=treatment_bb,
        )
        control_evidence = H4HoldoutVarianceArmEvidence(
            arm_id="control",
            holdout_unit_id=matched.holdout_unit_id,
            master_seed=matched.master_seed,
            hero_btn_gain_bb=control_btn,
            hero_bb_gain_bb=control_bb,
        )
        unit_contrast = matched_unit_contrast_bb100(
            control_btn_gain_bb=control_btn,
            control_bb_gain_bb=control_bb,
            treatment_btn_gain_bb=treatment_btn,
            treatment_bb_gain_bb=treatment_bb,
        )

    return H4HoldoutVariancePairExecution(
        holdout_unit_id=matched.holdout_unit_id,
        master_seed=matched.master_seed,
        hands_per_position=matched.hands_per_position,
        scientifically_valid=scientifically_valid,
        treatment_match_id=treatment_record.match_id,
        control_match_id=control_record.match_id,
        treatment_record_hash=treatment_record.record_hash,
        control_record_hash=control_record.record_hash,
        treatment_failed_runs=tuple(dict(item) for item in treatment_result.failed_runs),
        control_failed_runs=tuple(dict(item) for item in control_result.failed_runs),
        treatment_learning_hands_by_position=learning_counts,
        treatment_evidence=treatment_evidence,
        control_evidence=control_evidence,
        unit_contrast_bb100=unit_contrast,
    )



def _validate_pair_execution_for_cluster(
    pair: H4HoldoutVariancePairExecution,
    *,
    expected_unit_id: str,
    expected_seed: int,
) -> None:
    if pair.enforcement_version != H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION:
        raise H4HoldoutVarianceExecutionError(
            "population cluster pair uses an unknown enforcement version"
        )
    if pair.holdout_unit_id != expected_unit_id or pair.master_seed != expected_seed:
        raise H4HoldoutVarianceExecutionError(
            "population cluster pair identity differs from the frozen cell"
        )
    if (
        not pair.scientifically_valid
        or pair.unit_contrast_bb100 is None
        or pair.treatment_evidence is None
        or pair.control_evidence is None
    ):
        raise H4HoldoutVarianceExecutionError(
            "one invalid unit pair invalidates the entire population cluster"
        )
    if pair.hands_per_position != 500:
        raise H4HoldoutVarianceExecutionError(
            "population cluster pair has a non-frozen horizon"
        )
    if pair.treatment_failed_runs or pair.control_failed_runs:
        raise H4HoldoutVarianceExecutionError(
            "scientifically valid population-cluster pairs cannot contain failed runs"
        )
    if pair.treatment_learning_hands_by_position != (("A_BTN", 500), ("B_BTN", 500)):
        raise H4HoldoutVarianceExecutionError(
            "population-cluster treatment pair lacks complete replay admission"
        )
    for evidence, arm_id in (
        (pair.treatment_evidence, "treatment"),
        (pair.control_evidence, "control"),
    ):
        if (
            evidence.arm_id != arm_id
            or evidence.holdout_unit_id != expected_unit_id
            or evidence.master_seed != expected_seed
            or evidence.hands_per_position != 500
        ):
            raise H4HoldoutVarianceExecutionError(
                "population-cluster arm evidence identity differs from the admitted pair"
            )
    recomputed = matched_unit_contrast_bb100(
        control_btn_gain_bb=pair.control_evidence.hero_btn_gain_bb,
        control_bb_gain_bb=pair.control_evidence.hero_bb_gain_bb,
        treatment_btn_gain_bb=pair.treatment_evidence.hero_btn_gain_bb,
        treatment_bb_gain_bb=pair.treatment_evidence.hero_bb_gain_bb,
    )
    if recomputed != float(pair.unit_contrast_bb100):
        raise H4HoldoutVarianceExecutionError(
            "population-cluster unit contrast differs from replay-admitted arm evidence"
        )


def _validate_cluster_for_campaign(
    protocol: H4HoldoutVarianceProtocol,
    cluster: H4HoldoutVariancePopulationCluster,
) -> H4HoldoutVariancePopulationCluster:
    rebuilt = admit_h4_holdout_variance_population_cluster(
        protocol=protocol,
        master_seed=cluster.master_seed,
        pair_executions=cluster.pair_executions,
    )
    if (
        rebuilt.unit_contrasts_bb100 != cluster.unit_contrasts_bb100
        or rebuilt.population_contrast_bb100 != cluster.population_contrast_bb100
        or cluster.enforcement_version
        != H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION
    ):
        raise H4HoldoutVarianceExecutionError(
            "variance campaign cluster differs from replay-admitted population evidence"
        )
    return rebuilt

def admit_h4_holdout_variance_population_cluster(
    *,
    protocol: H4HoldoutVarianceProtocol,
    master_seed: int,
    pair_executions: Sequence[H4HoldoutVariancePairExecution],
) -> H4HoldoutVariancePopulationCluster:
    """Admit one seed only when all five frozen holdout unit pairs are replay-valid."""
    _require_protocol(protocol)
    if master_seed not in H4_VARIANCE_MASTER_SEEDS:
        raise H4HoldoutVarianceExecutionError(
            "population cluster seed is not in the frozen Spec-37 seed domain"
        )
    if len(pair_executions) != len(H4_HOLDOUT_UNIT_IDS):
        raise H4HoldoutVarianceExecutionError(
            "population cluster requires exactly five holdout unit pairs"
        )
    by_unit: dict[str, H4HoldoutVariancePairExecution] = {}
    for pair in pair_executions:
        if not isinstance(pair, H4HoldoutVariancePairExecution):
            raise TypeError("pair_executions must contain H4HoldoutVariancePairExecution")
        if pair.master_seed != master_seed:
            raise H4HoldoutVarianceExecutionError(
                "all unit pairs in a population cluster must use the same master seed"
            )
        if pair.holdout_unit_id in by_unit:
            raise H4HoldoutVarianceExecutionError(
                "population cluster contains a duplicate holdout unit"
            )
        by_unit[pair.holdout_unit_id] = pair
    if set(by_unit) != set(H4_HOLDOUT_UNIT_IDS):
        raise H4HoldoutVarianceExecutionError(
            "population cluster must contain exactly all five frozen holdout units"
        )

    unit_contrasts: dict[str, float] = {}
    ordered_pairs: list[H4HoldoutVariancePairExecution] = []
    for unit_id in H4_HOLDOUT_UNIT_IDS:
        pair = by_unit[unit_id]
        _validate_pair_execution_for_cluster(
            pair,
            expected_unit_id=unit_id,
            expected_seed=master_seed,
        )
        ordered_pairs.append(pair)
        assert pair.unit_contrast_bb100 is not None
        unit_contrasts[unit_id] = float(pair.unit_contrast_bb100)

    population_contrast = population_contrast_for_seed(unit_contrasts)
    return H4HoldoutVariancePopulationCluster(
        master_seed=master_seed,
        pair_executions=tuple(ordered_pairs),
        unit_contrasts_bb100=tuple(
            (unit_id, unit_contrasts[unit_id]) for unit_id in H4_HOLDOUT_UNIT_IDS
        ),
        population_contrast_bb100=population_contrast,
    )


def admit_h4_holdout_variance_campaign(
    *,
    protocol: H4HoldoutVarianceProtocol,
    clusters: Sequence[H4HoldoutVariancePopulationCluster],
) -> H4HoldoutVarianceCampaignAdmission:
    """Release sigma_500/sample-size planning only after exact 10-of-10 admission."""
    _require_protocol(protocol)
    if len(clusters) != len(H4_VARIANCE_MASTER_SEEDS):
        raise H4HoldoutVarianceExecutionError(
            "sigma_500 requires exactly ten valid population clusters; reduced complete-case planning is forbidden"
        )
    by_seed: dict[int, H4HoldoutVariancePopulationCluster] = {}
    for cluster in clusters:
        if not isinstance(cluster, H4HoldoutVariancePopulationCluster):
            raise TypeError("clusters must contain H4HoldoutVariancePopulationCluster")
        if cluster.master_seed in by_seed:
            raise H4HoldoutVarianceExecutionError(
                "variance campaign contains a duplicate population-cluster seed"
            )
        by_seed[cluster.master_seed] = cluster
    if set(by_seed) != set(H4_VARIANCE_MASTER_SEEDS):
        raise H4HoldoutVarianceExecutionError(
            "variance campaign must contain exactly the ten frozen Spec-37 seeds; replacements are forbidden"
        )

    rebuilt_by_seed = {
        seed: _validate_cluster_for_campaign(protocol, by_seed[seed])
        for seed in H4_VARIANCE_MASTER_SEEDS
    }
    seed_contrasts = {
        seed: rebuilt_by_seed[seed].population_contrast_bb100
        for seed in H4_VARIANCE_MASTER_SEEDS
    }
    plan = variance_pilot_plan_from_complete_seed_contrasts(seed_contrasts)
    return H4HoldoutVarianceCampaignAdmission(
        clusters=tuple(rebuilt_by_seed[seed] for seed in H4_VARIANCE_MASTER_SEEDS),
        seed_contrasts_bb100=tuple(
            (seed, seed_contrasts[seed]) for seed in H4_VARIANCE_MASTER_SEEDS
        ),
        sample_size_plan=plan,
    )
