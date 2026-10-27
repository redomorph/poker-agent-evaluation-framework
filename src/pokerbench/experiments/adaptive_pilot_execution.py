"""Runtime enforcement primitives for the frozen Phase-2 adaptive pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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
from pokerbench.experiments.adaptive_pilot_analysis import TrajectoryArmEvidence
from pokerbench.experiments.adaptive_pilot_binding import (
    AdaptivePilotBindingError,
    AdaptivePilotTrajectoryConfig,
    validate_adaptive_pilot_trajectory_config,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotJob,
    AdaptivePilotProtocol,
    expected_adaptive_pilot_payload,
    iter_adaptive_pilot_jobs,
)
from pokerbench.replay.archive import MatchRecord, RecordedAdaptiveLearningEvent
from pokerbench.rng import AgentRNG
from pokerbench.runner import (
    IllegalActionPolicy,
    PairedResult,
    PairedRunner,
    StatisticalUnit,
)


ADAPTIVE_PILOT_EXECUTION_ENFORCEMENT_VERSION = (
    "phase2-adaptive-pilot-execution-enforcement-v1"
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


@runtime_checkable
class DurableMatchStore(Protocol):
    """Read/write durability contract required before pilot analysis admission."""

    def save_match(self, record: MatchRecord) -> None: ...

    def load_match(self, match_id: str) -> MatchRecord: ...


class AdaptivePilotExecutionError(ValueError):
    """Raised when Spec-32 execution evidence violates a frozen invariant."""


@dataclass(frozen=True)
class MatchedAdaptivePilotJobs:
    """One canonical treatment/control profile-seed identity."""

    profile_id: str
    master_seed: int
    hands_per_position: int
    treatment_job: AdaptivePilotJob
    control_job: AdaptivePilotJob


@dataclass(frozen=True)
class MatchedDealScheduleProof:
    """Pre-execution proof of the limited cross-arm CRN relationship."""

    master_seed: int
    hands_per_position: int
    deal_schedule_hash: str
    deal_relation: str = "identical_defined_card_realisations"
    agent_rng_relation: str = "event_indexed_common_identity_where_events_correspond"
    full_action_path_identity: str = "not_claimed"


@dataclass(frozen=True)
class AdaptivePilotPairExecution:
    """Durable complete-pair status and analysis admission evidence."""

    profile_id: str
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
    treatment_evidence: TrajectoryArmEvidence | None
    control_evidence: TrajectoryArmEvidence | None
    enforcement_version: str = ADAPTIVE_PILOT_EXECUTION_ENFORCEMENT_VERSION


class PilotHeroBoundaryAgent(Agent):
    """Guard the hero-facing decision and learning interfaces at runtime."""

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
        validate_pilot_hero_observation(observation)
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
            raise AdaptivePilotExecutionError(
                "control arm cannot receive adaptive learning updates"
            )
        observe = getattr(self._delegate, "observe_opponent_hand", None)
        if not callable(observe):
            raise AdaptivePilotExecutionError(
                "adaptive treatment must implement the frozen lifecycle learning hook"
            )
        event = observe(
            counts,
            learner_bot_id=learner_bot_id,
            opponent_bot_id=opponent_bot_id,
            trajectory_hand_index=trajectory_hand_index,
        )
        if not isinstance(event, RecordedAdaptiveLearningEvent):
            raise AdaptivePilotExecutionError(
                "adaptive lifecycle hook must return RecordedAdaptiveLearningEvent"
            )
        return event


def validate_pilot_hero_observation(observation: AgentObservation) -> None:
    """Reject any controller-only or newly injected hero-observation field."""
    if type(observation) is not AgentObservation:
        raise AdaptivePilotExecutionError(
            "pilot hero observation must be the frozen AgentObservation type"
        )
    actual_fields = frozenset(vars(observation))
    if actual_fields != _ALLOWED_OBSERVATION_FIELDS:
        raise AdaptivePilotExecutionError(
            "pilot hero observation contains a forbidden or unknown field"
        )


def _require_protocol(protocol: AdaptivePilotProtocol) -> None:
    if not isinstance(protocol, AdaptivePilotProtocol):
        raise TypeError("protocol must be an AdaptivePilotProtocol")


def _require_canonical_job(
    protocol: AdaptivePilotProtocol,
    job: AdaptivePilotJob,
) -> None:
    if not isinstance(job, AdaptivePilotJob):
        raise TypeError("job must be an AdaptivePilotJob")
    if job not in iter_adaptive_pilot_jobs(protocol):
        raise AdaptivePilotExecutionError(
            "job must be an exact canonical frozen pilot job; replacement seeds are forbidden"
        )


def validate_matched_adaptive_pilot_jobs(
    *,
    protocol: AdaptivePilotProtocol,
    treatment_job: AdaptivePilotJob,
    control_job: AdaptivePilotJob,
) -> MatchedAdaptivePilotJobs:
    """Require one exact frozen treatment/control pair for a profile and seed."""
    _require_protocol(protocol)
    _require_canonical_job(protocol, treatment_job)
    _require_canonical_job(protocol, control_job)
    if treatment_job.arm_id != "treatment" or not treatment_job.adaptive:
        raise AdaptivePilotExecutionError("treatment_job must be the frozen adaptive arm")
    if control_job.arm_id != "control" or control_job.adaptive:
        raise AdaptivePilotExecutionError("control_job must be the frozen static arm")
    if treatment_job.opponent_profile_id != control_job.opponent_profile_id:
        raise AdaptivePilotExecutionError("matched jobs must use the same profile")
    if treatment_job.master_seed != control_job.master_seed:
        raise AdaptivePilotExecutionError("matched jobs must use the same master seed")
    if treatment_job.hands_per_position != control_job.hands_per_position:
        raise AdaptivePilotExecutionError("matched jobs must use the same trajectory horizon")
    if (
        treatment_job.position_run_count != control_job.position_run_count
        or treatment_job.position_run_count != 2
    ):
        raise AdaptivePilotExecutionError("matched jobs must contain exactly two position runs")
    for job in (treatment_job, control_job):
        if job.statistical_unit != "trajectory_cluster" or job.execution_mode != "full-match":
            raise AdaptivePilotExecutionError(
                "both pilot arms must use trajectory_cluster/full-match execution"
            )
    return MatchedAdaptivePilotJobs(
        profile_id=treatment_job.opponent_profile_id,
        master_seed=treatment_job.master_seed,
        hands_per_position=treatment_job.hands_per_position,
        treatment_job=treatment_job,
        control_job=control_job,
    )


def build_adaptive_pilot_hero(
    job: AdaptivePilotJob,
    *,
    reference: PreflopStrengthReference,
) -> Agent:
    """Build exactly the Spec-32 control or treatment hero implementation."""
    if not isinstance(job, AdaptivePilotJob):
        raise TypeError("job must be an AdaptivePilotJob")
    if job.arm_id == "control":
        if job.adaptive or job.agent_version != HAND_STRENGTH_BASELINE_VERSION:
            raise AdaptivePilotExecutionError("invalid frozen control-arm binding")
        return HandStrengthBaselineAgent()
    if job.arm_id == "treatment":
        if not job.adaptive or job.agent_version != ADAPTIVE_HAND_STRENGTH_AGENT_VERSION:
            raise AdaptivePilotExecutionError("invalid frozen treatment-arm binding")
        return AdaptiveHandStrengthAgent(reference)
    raise AdaptivePilotExecutionError("unknown adaptive-pilot arm")


def _require_durable_store(store: object) -> None:
    if not isinstance(store, DurableMatchStore):
        raise AdaptivePilotExecutionError(
            "adaptive pilot execution requires a durable save/load match store"
        )


def build_adaptive_pilot_runner(
    *,
    protocol: AdaptivePilotProtocol,
    job: AdaptivePilotJob,
    reference: PreflopStrengthReference,
    opponent: Agent,
    game_config: GameConfig,
    match_id: str,
    benchmark_store: DurableMatchStore,
    trajectory_config: AdaptivePilotTrajectoryConfig,
) -> PairedRunner:
    """Build a guarded full-match runner without authorizing matrix execution."""
    _require_protocol(protocol)
    _require_canonical_job(protocol, job)
    if not isinstance(opponent, Agent):
        raise TypeError("opponent must be an Agent")
    if opponent.is_adaptive:
        raise AdaptivePilotExecutionError("Spec-32 development opponents must be stationary")
    expected_game_hash = str(expected_adaptive_pilot_payload()["game"]["game_config_hash"])
    if config_hash(game_config) != expected_game_hash:
        raise AdaptivePilotExecutionError("pilot game configuration differs from frozen Spec-32")
    _require_durable_store(benchmark_store)
    try:
        validate_adaptive_pilot_trajectory_config(
            protocol=protocol,
            config=trajectory_config,
            expected_job=job,
        )
    except AdaptivePilotBindingError as exc:
        raise AdaptivePilotExecutionError(
            "trajectory config does not match the guarded pilot job"
        ) from exc

    hero = PilotHeroBoundaryAgent(
        build_adaptive_pilot_hero(job, reference=reference)
    )
    runner = PairedRunner(
        agent_a=hero,
        agent_b=opponent,
        num_hands=job.hands_per_position,
        seed=job.master_seed,
        game_config=game_config,
        experiment_metadata=trajectory_config.runner_metadata,
        adaptive=True,
        illegal_action_policy=IllegalActionPolicy.EXCLUDE_RUN,
        match_id=match_id,
        benchmark_store=benchmark_store,
    )
    if runner.statistical_unit is not StatisticalUnit.TRAJECTORY_CLUSTER:
        raise AdaptivePilotExecutionError(
            "pilot runner must use trajectory-cluster statistical units"
        )
    return runner


def _deal_schedule_hash(job: AdaptivePilotJob) -> str:
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


def prove_matched_deal_schedule(
    *,
    protocol: AdaptivePilotProtocol,
    treatment_job: AdaptivePilotJob,
    control_job: AdaptivePilotJob,
) -> MatchedDealScheduleProof:
    """Prove the Spec-32 defined-card CRN relation without path overclaiming."""
    matched = validate_matched_adaptive_pilot_jobs(
        protocol=protocol,
        treatment_job=treatment_job,
        control_job=control_job,
    )
    treatment_hash = _deal_schedule_hash(treatment_job)
    control_hash = _deal_schedule_hash(control_job)
    if treatment_hash != control_hash:
        raise AdaptivePilotExecutionError(
            "matched treatment/control jobs do not share identical card realisations"
        )
    return MatchedDealScheduleProof(
        master_seed=matched.master_seed,
        hands_per_position=matched.hands_per_position,
        deal_schedule_hash=treatment_hash,
    )


def _require_arm_result(
    *,
    job: AdaptivePilotJob,
    result: PairedResult,
) -> tuple[bool, MatchRecord]:
    if not isinstance(result, PairedResult):
        raise TypeError("result must be a PairedResult")
    if result.statistical_unit is not StatisticalUnit.TRAJECTORY_CLUSTER:
        raise AdaptivePilotExecutionError("pilot arm result has wrong statistical unit")
    if result.seed != job.master_seed or result.num_hands != job.hands_per_position:
        raise AdaptivePilotExecutionError("pilot arm result does not match its frozen job")
    record = result.match_record
    if record is None:
        raise AdaptivePilotExecutionError("pilot arm result must retain a MatchRecord")
    if record.master_seed != job.master_seed:
        raise AdaptivePilotExecutionError("pilot MatchRecord seed differs from frozen job")

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
        raise AdaptivePilotExecutionError("pilot arm result has inconsistent validity fields")
    if valid_shape and len(record.hands) != 2 * job.hands_per_position:
        raise AdaptivePilotExecutionError(
            "valid pilot trajectory cluster must persist both complete position runs"
        )
    if len(record.hands) > 2 * job.hands_per_position:
        raise AdaptivePilotExecutionError("pilot MatchRecord contains too many hands")
    return valid_shape, record


def _require_persisted_record(store: DurableMatchStore, record: MatchRecord) -> None:
    _require_durable_store(store)
    try:
        persisted = store.load_match(record.match_id)
    except Exception as exc:
        raise AdaptivePilotExecutionError(
            "pilot MatchRecord is not durably persisted"
        ) from exc
    if not isinstance(persisted, MatchRecord) or persisted.record_hash != record.record_hash:
        raise AdaptivePilotExecutionError(
            "durably persisted pilot MatchRecord differs from runner evidence"
        )


def _control_has_learning_events(record: MatchRecord) -> bool:
    return any(hand.adaptive_learning_events for hand in record.hands)


def _extract_hero_profit_series(
    record: MatchRecord,
    *,
    hands_per_position: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    by_config: dict[str, list[tuple[int, float]]] = {"A_BTN": [], "B_BTN": []}
    for hand in record.hands:
        if hand.config_id not in by_config:
            raise AdaptivePilotExecutionError("unexpected position config in pilot MatchRecord")
        if hand.terminal_profit_bb is None:
            raise AdaptivePilotExecutionError("valid pilot hand must retain terminal profit")
        if hand.seat0_bot_id == 0:
            hero_profit = float(hand.terminal_profit_bb[0])
        elif hand.seat1_bot_id == 0:
            hero_profit = float(hand.terminal_profit_bb[1])
        else:
            raise AdaptivePilotExecutionError("pilot hand does not contain stable hero bot_id=0")
        by_config[hand.config_id].append((hand.hand_index, hero_profit))

    series: list[tuple[float, ...]] = []
    for config_id in ("A_BTN", "B_BTN"):
        items = sorted(by_config[config_id])
        if tuple(index for index, _ in items) != tuple(range(hands_per_position)):
            raise AdaptivePilotExecutionError(
                "valid pilot position run must contain every planned hand exactly once"
            )
        series.append(tuple(value for _, value in items))
    return series[0], series[1]


def _learning_hand_counts(record: MatchRecord) -> tuple[tuple[str, int], ...]:
    models = replay_adaptive_learning_for_bot(record, bot_id=0)
    return tuple(
        (config_id, models[config_id].hands_observed)
        for config_id in ("A_BTN", "B_BTN")
        if config_id in models
    )


def finalize_adaptive_pilot_pair(
    *,
    protocol: AdaptivePilotProtocol,
    treatment_job: AdaptivePilotJob,
    control_job: AdaptivePilotJob,
    treatment_result: PairedResult,
    control_result: PairedResult,
    benchmark_store: DurableMatchStore,
) -> AdaptivePilotPairExecution:
    """Verify durability/replay and admit only a complete valid pair to analysis."""
    matched = validate_matched_adaptive_pilot_jobs(
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
        raise AdaptivePilotExecutionError(
            "control arm MatchRecord must not contain adaptive learning events"
        )

    learning_counts = _learning_hand_counts(treatment_record)
    if treatment_valid:
        expected_counts = (("A_BTN", matched.hands_per_position), ("B_BTN", matched.hands_per_position))
        if learning_counts != expected_counts:
            raise AdaptivePilotExecutionError(
                "valid adaptive treatment must replay one learning transition per hand in both positions"
            )

    scientifically_valid = treatment_valid and control_valid
    treatment_evidence: TrajectoryArmEvidence | None = None
    control_evidence: TrajectoryArmEvidence | None = None
    if scientifically_valid:
        treatment_btn, treatment_bb = _extract_hero_profit_series(
            treatment_record,
            hands_per_position=matched.hands_per_position,
        )
        control_btn, control_bb = _extract_hero_profit_series(
            control_record,
            hands_per_position=matched.hands_per_position,
        )
        treatment_evidence = TrajectoryArmEvidence(
            arm_id="treatment",
            profile_id=matched.profile_id,
            master_seed=matched.master_seed,
            hero_btn_profit_bb=treatment_btn,
            hero_bb_profit_bb=treatment_bb,
            scientifically_valid=True,
        )
        control_evidence = TrajectoryArmEvidence(
            arm_id="control",
            profile_id=matched.profile_id,
            master_seed=matched.master_seed,
            hero_btn_profit_bb=control_btn,
            hero_bb_profit_bb=control_bb,
            scientifically_valid=True,
        )

    return AdaptivePilotPairExecution(
        profile_id=matched.profile_id,
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
    )
