"""Post-run integrity audit and descriptive analysis for the frozen Spec-32 pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pokerbench.agents.adaptive_lifecycle import (
    AdaptiveHandStrengthAgent,
    replay_adaptive_learning_for_bot,
)
from pokerbench.agents.opponent_model import OpponentModelMetric
from pokerbench.evaluation.calibration import BehaviorCounts
from pokerbench.evaluation.population import DiscretePopulationDefinition
from pokerbench.experiments.adaptive_pilot_analysis import (
    FROZEN_CUMULATIVE_HORIZONS,
    FROZEN_SEGMENTS,
    MatchedTrajectoryContrast,
    cumulative_arm_estimate,
    matched_trajectory_contrast,
    population_contrast_sample_sd,
    population_weighted_seed_contrast,
    profile_contrast_sample_sd,
    segment_arm_estimate,
)
from pokerbench.experiments.adaptive_pilot_binding import (
    AdaptivePilotExecutionBinding,
    load_adaptive_pilot_execution_binding,
)
from pokerbench.experiments.adaptive_pilot_execution import (
    AdaptivePilotPairExecution,
    finalize_adaptive_pilot_pair,
)
from pokerbench.experiments.adaptive_pilot_parallel_execution_contract import (
    AdaptivePilotParallelExecutionContract,
    load_adaptive_pilot_parallel_execution_contract,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotProtocol,
    iter_adaptive_pilot_jobs,
    load_adaptive_pilot_protocol,
)
from pokerbench.experiments.adaptive_pilot_run import (
    AdaptivePilotArmSnapshot,
    _load_pair_checkpoint,
    adaptive_pilot_job_key,
    load_adaptive_pilot_arm_snapshot,
    load_adaptive_pilot_run_manifest,
)
from pokerbench.experiments.frozen_protocol import load_frozen_population
from pokerbench.persistence.sqlite import SQLiteBenchmarkStore
from pokerbench.replay.archive import MatchRecord


ADAPTIVE_PILOT_POSTRUN_ANALYSIS_VERSION = "phase2-adaptive-pilot-postrun-analysis-v1"
ADAPTIVE_PILOT_POSTRUN_REPORT_VERSION = "phase2-adaptive-pilot-postrun-report-v1"

_METRIC_ORDER = (
    OpponentModelMetric.VPIP,
    OpponentModelMetric.PFR,
    OpponentModelMetric.THREE_BET,
    OpponentModelMetric.CBET,
    OpponentModelMetric.AGGRESSIVE_SHARE,
    OpponentModelMetric.FOLD_TO_CBET,
)
_FINAL_H4_DESIGN = {
    "primary_h4_horizon": None,
    "final_h4_population": None,
    "minimum_relevant_difference_bb100": None,
    "precision_target_half_width_bb100": None,
    "final_independent_trajectory_clusters": None,
    "confirmatory_alpha": None,
    "confirmatory_beta": None,
    "multiplicity_binding": None,
    "final_h4_seed_schedule": None,
}
_SCIENTIFIC_USE = {
    "confirmatory_h4_inference": False,
    "final_h4_effect_estimate": False,
    "pilot_plus_final_data_pooling": False,
    "optional_stopping": False,
    "seed_replacement_after_failure": False,
    "adaptive_policy_tuning_from_pilot_ev": False,
    "opponent_profile_tuning_from_pilot_ev": False,
    "primary_h4_horizon_selection_from_observed_delta": False,
    "variance_estimation_horizons": [10, 20, 50, 100],
    "variance_at_500": False,
}


class AdaptivePilotPostRunError(ValueError):
    """Raised when completed pilot evidence cannot enter the frozen analysis."""


def _sha256_canonical(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_file_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _validate_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise AdaptivePilotPostRunError(f"{field_name} must be a lowercase SHA-256")
    return value


def _finite(value: float, field_name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise AdaptivePilotPostRunError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True)
class AdaptivePilotLearningMetricDiagnostic:
    metric: str
    opportunities: int
    events: int
    alpha: int
    beta: int

    def __post_init__(self) -> None:
        if self.metric not in tuple(metric.value for metric in _METRIC_ORDER):
            raise AdaptivePilotPostRunError("learning diagnostic metric is not frozen")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.opportunities, self.events, self.alpha, self.beta)
        ):
            raise AdaptivePilotPostRunError("learning diagnostic counts must be non-negative integers")
        if self.events > self.opportunities or self.alpha <= 0 or self.beta <= 0:
            raise AdaptivePilotPostRunError("learning diagnostic counts are inconsistent")
        if self.alpha != 1 + self.events or self.beta != 1 + self.opportunities - self.events:
            raise AdaptivePilotPostRunError("learning diagnostic posterior does not match frozen Beta(1,1) prior")

    def payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "opportunities": self.opportunities,
            "events": self.events,
            "alpha": self.alpha,
            "beta": self.beta,
        }


@dataclass(frozen=True)
class AdaptivePilotLearningHorizonDiagnostic:
    profile_id: str
    master_seed: int
    config_id: str
    horizon: int
    trajectory_hand_count: int
    valid_learning_transition_count: int
    posterior_state_hash: str
    metrics: tuple[AdaptivePilotLearningMetricDiagnostic, ...]

    def __post_init__(self) -> None:
        if not self.profile_id or self.config_id not in {"A_BTN", "B_BTN"}:
            raise AdaptivePilotPostRunError("learning diagnostic identity is invalid")
        if self.horizon not in FROZEN_CUMULATIVE_HORIZONS:
            raise AdaptivePilotPostRunError("learning diagnostic horizon is not frozen")
        if self.trajectory_hand_count != self.horizon:
            raise AdaptivePilotPostRunError("learning diagnostic trajectory count must equal horizon")
        if self.valid_learning_transition_count != self.horizon:
            raise AdaptivePilotPostRunError("learning diagnostic transition count must equal horizon")
        _validate_sha256(self.posterior_state_hash, "posterior_state_hash")
        if tuple(metric.metric for metric in self.metrics) != tuple(
            metric.value for metric in _METRIC_ORDER
        ):
            raise AdaptivePilotPostRunError("learning diagnostic must contain the six frozen metrics in order")

    def payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "master_seed": self.master_seed,
            "config_id": self.config_id,
            "horizon": self.horizon,
            "trajectory_hand_count": self.trajectory_hand_count,
            "valid_learning_transition_count": self.valid_learning_transition_count,
            "posterior_state_hash": self.posterior_state_hash,
            "metrics": [item.payload() for item in self.metrics],
        }


@dataclass(frozen=True)
class AdaptivePilotPairAuditEvidence:
    profile_id: str
    master_seed: int
    hands_per_position: int
    treatment_match_id: str
    control_match_id: str
    treatment_record_hash: str
    control_record_hash: str
    treatment_snapshot_hash: str
    control_snapshot_hash: str
    pair_checkpoint_hash: str

    def __post_init__(self) -> None:
        if not self.profile_id or not self.treatment_match_id or not self.control_match_id:
            raise AdaptivePilotPostRunError("pair audit identity must be non-empty")
        for field_name in (
            "treatment_record_hash",
            "control_record_hash",
            "treatment_snapshot_hash",
            "control_snapshot_hash",
            "pair_checkpoint_hash",
        ):
            _validate_sha256(str(getattr(self, field_name)), field_name)

    def payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "master_seed": self.master_seed,
            "hands_per_position": self.hands_per_position,
            "treatment_match_id": self.treatment_match_id,
            "control_match_id": self.control_match_id,
            "treatment_record_hash": self.treatment_record_hash,
            "control_record_hash": self.control_record_hash,
            "treatment_snapshot_hash": self.treatment_snapshot_hash,
            "control_snapshot_hash": self.control_snapshot_hash,
            "pair_checkpoint_hash": self.pair_checkpoint_hash,
        }


@dataclass(frozen=True)
class AdaptivePilotCampaignAudit:
    binding_hash: str
    manifest_hash: str
    terminal_jobs: int
    completed_jobs: int
    failed_jobs: int
    valid_pairs: int
    invalid_pairs: int
    recorded_engine_hands: int
    total_arm_runtime_ns: int
    median_arm_runtime_ns: float
    database_bytes: int
    work_root_bytes: int
    replayed_treatment_match_records: int
    replayed_learning_transitions: int
    pair_evidence: tuple[AdaptivePilotPairAuditEvidence, ...]
    pair_executions: tuple[AdaptivePilotPairExecution, ...]
    learning_diagnostics: tuple[AdaptivePilotLearningHorizonDiagnostic, ...]

    def __post_init__(self) -> None:
        _validate_sha256(self.binding_hash, "binding_hash")
        _validate_sha256(self.manifest_hash, "manifest_hash")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.terminal_jobs,
                self.completed_jobs,
                self.failed_jobs,
                self.valid_pairs,
                self.invalid_pairs,
                self.recorded_engine_hands,
                self.total_arm_runtime_ns,
                self.database_bytes,
                self.work_root_bytes,
                self.replayed_treatment_match_records,
                self.replayed_learning_transitions,
            )
        ):
            raise AdaptivePilotPostRunError("campaign audit counts must be non-negative integers")
        _finite(self.median_arm_runtime_ns, "median_arm_runtime_ns")


@dataclass(frozen=True)
class AdaptivePilotWindowSummary:
    scope_kind: str
    scope_id: str
    population_version: str | None
    population_hash: str | None
    window_kind: str
    window_id: str
    start_hand: int
    end_hand: int
    num_seed_contrasts: int
    master_seeds: tuple[int, ...]
    mean_treatment_bb100: float
    mean_control_bb100: float
    mean_difference_bb100: float
    std_difference_bb100: float | None
    variance_estimator: str | None
    inferential_statistics_permitted: bool = False

    def __post_init__(self) -> None:
        if self.scope_kind not in {"profile", "population"} or not self.scope_id:
            raise AdaptivePilotPostRunError("window summary scope is invalid")
        if self.num_seed_contrasts != len(self.master_seeds) or self.num_seed_contrasts <= 0:
            raise AdaptivePilotPostRunError("window summary seed count is inconsistent")
        if len(set(self.master_seeds)) != len(self.master_seeds):
            raise AdaptivePilotPostRunError("window summary master seeds must be unique")
        if self.scope_kind == "profile":
            if self.population_version is not None or self.population_hash is not None:
                raise AdaptivePilotPostRunError("profile summary cannot carry population provenance")
        else:
            if self.population_version is None or self.population_hash is None:
                raise AdaptivePilotPostRunError("population summary must carry frozen population provenance")
            _validate_sha256(self.population_hash, "population_hash")
        variance_defined = (
            self.window_kind == "cumulative" and self.start_hand == 1 and self.end_hand in (10, 20, 50, 100)
        )
        if variance_defined:
            if self.std_difference_bb100 is None or self.variance_estimator is None:
                raise AdaptivePilotPostRunError("frozen cumulative planning window requires sample-SD evidence")
        elif self.std_difference_bb100 is not None or self.variance_estimator is not None:
            raise AdaptivePilotPostRunError("variance is undefined for this pilot reporting window")
        for value in (
            self.mean_treatment_bb100,
            self.mean_control_bb100,
            self.mean_difference_bb100,
        ):
            _finite(value, "window summary mean")
        if self.std_difference_bb100 is not None:
            _finite(self.std_difference_bb100, "std_difference_bb100")
        if self.inferential_statistics_permitted:
            raise AdaptivePilotPostRunError("pilot report must not permit inferential statistics")

    def payload(self) -> dict[str, Any]:
        return {
            "scope_kind": self.scope_kind,
            "scope_id": self.scope_id,
            "population_version": self.population_version,
            "population_hash": self.population_hash,
            "window_kind": self.window_kind,
            "window_id": self.window_id,
            "start_hand": self.start_hand,
            "end_hand": self.end_hand,
            "num_seed_contrasts": self.num_seed_contrasts,
            "master_seeds": list(self.master_seeds),
            "mean_treatment_bb100": self.mean_treatment_bb100,
            "mean_control_bb100": self.mean_control_bb100,
            "mean_difference_bb100": self.mean_difference_bb100,
            "std_difference_bb100": self.std_difference_bb100,
            "variance_estimator": self.variance_estimator,
            "inferential_statistics_permitted": self.inferential_statistics_permitted,
        }


@dataclass(frozen=True)
class AdaptivePilotPostRunReport:
    protocol_id: str
    protocol_version: str
    protocol_hash: str
    execution_contract_id: str
    execution_contract_version: str
    execution_contract_hash: str
    binding_hash: str
    manifest_hash: str
    completed_jobs: int
    failed_jobs: int
    valid_pairs: int
    invalid_pairs: int
    recorded_engine_hands: int
    total_arm_runtime_ns: int
    median_arm_runtime_ns: float
    database_bytes: int
    work_root_bytes: int
    replayed_treatment_match_records: int
    replayed_learning_transitions: int
    pair_evidence: tuple[AdaptivePilotPairAuditEvidence, ...]
    profile_summaries: tuple[AdaptivePilotWindowSummary, ...]
    population_summaries: tuple[AdaptivePilotWindowSummary, ...]
    learning_diagnostics: tuple[AdaptivePilotLearningHorizonDiagnostic, ...]
    complete: bool
    failure_free: bool
    final_h4_design: Mapping[str, Any]
    scientific_use: Mapping[str, Any]
    analysis_version: str = ADAPTIVE_PILOT_POSTRUN_ANALYSIS_VERSION
    report_version: str = ADAPTIVE_PILOT_POSTRUN_REPORT_VERSION

    def __post_init__(self) -> None:
        if self.analysis_version != ADAPTIVE_PILOT_POSTRUN_ANALYSIS_VERSION:
            raise AdaptivePilotPostRunError("unsupported adaptive-pilot post-run analysis version")
        if self.report_version != ADAPTIVE_PILOT_POSTRUN_REPORT_VERSION:
            raise AdaptivePilotPostRunError("unsupported adaptive-pilot post-run report version")
        for name in ("protocol_hash", "execution_contract_hash", "binding_hash", "manifest_hash"):
            _validate_sha256(str(getattr(self, name)), name)
        if dict(self.final_h4_design) != _FINAL_H4_DESIGN:
            raise AdaptivePilotPostRunError("final H4 design fields must remain unset in the pilot report")
        if dict(self.scientific_use) != _SCIENTIFIC_USE:
            if self.scientific_use.get("confirmatory_h4_inference") is not False:
                raise AdaptivePilotPostRunError("pilot report cannot contain confirmatory H4 inference")
            raise AdaptivePilotPostRunError("pilot report scientific-use boundary differs from Spec 32")
        if self.complete != (self.completed_jobs == 50):
            raise AdaptivePilotPostRunError("post-run complete flag is inconsistent")
        if self.failure_free != (self.complete and self.failed_jobs == 0 and self.invalid_pairs == 0):
            raise AdaptivePilotPostRunError("post-run failure-free flag is inconsistent")
        if self.complete:
            if self.valid_pairs != 25 or self.invalid_pairs != 0:
                raise AdaptivePilotPostRunError("complete pilot report requires exactly 25 valid pairs")
            if self.recorded_engine_hands != 18_000:
                raise AdaptivePilotPostRunError("complete pilot report requires exactly 18,000 engine hands")
            if self.replayed_treatment_match_records != 25 or self.replayed_learning_transitions != 9_000:
                raise AdaptivePilotPostRunError("complete pilot report replay coverage is inconsistent")
            if len(self.pair_evidence) != 25 or len(self.learning_diagnostics) != 210:
                raise AdaptivePilotPostRunError("complete pilot report evidence cardinality is inconsistent")
            if len(self.profile_summaries) != 40 or len(self.population_summaries) != 16:
                raise AdaptivePilotPostRunError("complete pilot report window-summary cardinality is inconsistent")

    def payload(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "analysis_version": self.analysis_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "execution_contract_id": self.execution_contract_id,
            "execution_contract_version": self.execution_contract_version,
            "execution_contract_hash": self.execution_contract_hash,
            "binding_hash": self.binding_hash,
            "manifest_hash": self.manifest_hash,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "valid_pairs": self.valid_pairs,
            "invalid_pairs": self.invalid_pairs,
            "recorded_engine_hands": self.recorded_engine_hands,
            "total_arm_runtime_ns": self.total_arm_runtime_ns,
            "median_arm_runtime_ns": self.median_arm_runtime_ns,
            "database_bytes": self.database_bytes,
            "work_root_bytes": self.work_root_bytes,
            "replayed_treatment_match_records": self.replayed_treatment_match_records,
            "replayed_learning_transitions": self.replayed_learning_transitions,
            "pair_evidence": [item.payload() for item in self.pair_evidence],
            "profile_summaries": [item.payload() for item in self.profile_summaries],
            "population_summaries": [item.payload() for item in self.population_summaries],
            "learning_diagnostics": [item.payload() for item in self.learning_diagnostics],
            "complete": self.complete,
            "failure_free": self.failure_free,
            "final_h4_design": dict(self.final_h4_design),
            "scientific_use": dict(self.scientific_use),
        }

    @property
    def report_hash(self) -> str:
        return _sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"report": self.payload(), "report_hash": self.report_hash}


def _metric_diagnostics(model) -> tuple[AdaptivePilotLearningMetricDiagnostic, ...]:  # type: ignore[no-untyped-def]
    return tuple(
        AdaptivePilotLearningMetricDiagnostic(
            metric=metric.value,
            opportunities=model.posterior(metric).opportunities,
            events=model.posterior(metric).events,
            alpha=model.posterior(metric).alpha,
            beta=model.posterior(metric).beta,
        )
        for metric in _METRIC_ORDER
    )


def learning_horizon_diagnostics(
    record: MatchRecord,
    *,
    profile_id: str,
    master_seed: int,
    horizons: Sequence[int],
) -> tuple[AdaptivePilotLearningHorizonDiagnostic, ...]:
    """Replay and summarize the frozen learner at cumulative trajectory horizons."""
    if record.master_seed != master_seed:
        raise AdaptivePilotPostRunError("learning diagnostic seed differs from MatchRecord")
    frozen_horizons = tuple(sorted(set(int(value) for value in horizons)))
    if not frozen_horizons or any(value not in FROZEN_CUMULATIVE_HORIZONS for value in frozen_horizons):
        raise AdaptivePilotPostRunError("learning diagnostics require frozen cumulative horizons")

    try:
        replay_adaptive_learning_for_bot(record, bot_id=0)
    except (TypeError, ValueError) as exc:
        raise AdaptivePilotPostRunError(f"adaptive learning replay failed: {exc}") from exc

    diagnostics: list[AdaptivePilotLearningHorizonDiagnostic] = []
    expected = 0
    for config_id in ("A_BTN", "B_BTN"):
        hands = sorted(
            (hand for hand in record.hands if hand.config_id == config_id),
            key=lambda hand: hand.hand_index,
        )
        if not hands:
            continue
        expected += sum(horizon <= len(hands) for horizon in frozen_horizons)
        if tuple(hand.hand_index for hand in hands) != tuple(range(len(hands))):
            raise AdaptivePilotPostRunError("learning trajectory hand indexes are not contiguous")
        model = AdaptiveHandStrengthAgent.initial_model()
        for position_index, hand in enumerate(hands, start=1):
            events = tuple(
                event for event in hand.adaptive_learning_events if event.learner_bot_id == 0
            )
            if len(events) != 1:
                raise AdaptivePilotPostRunError("treatment hand must contain exactly one learner transition")
            event = events[0]
            counts = BehaviorCounts(**dict(event.public_evidence_counts))
            model = model.updated(counts, first_hand_index=model.hands_observed)
            if model.state_hash != event.model_state_after_hash:
                raise AdaptivePilotPostRunError("diagnostic replay state hash differs from persisted transition")
            if position_index in frozen_horizons:
                diagnostics.append(
                    AdaptivePilotLearningHorizonDiagnostic(
                        profile_id=profile_id,
                        master_seed=master_seed,
                        config_id=config_id,
                        horizon=position_index,
                        trajectory_hand_count=position_index,
                        valid_learning_transition_count=position_index,
                        posterior_state_hash=model.state_hash,
                        metrics=_metric_diagnostics(model),
                    )
                )
    if len(diagnostics) != expected:
        raise AdaptivePilotPostRunError("learning diagnostics do not cover both positional trajectories")
    return tuple(diagnostics)


def _window_estimates(pair: AdaptivePilotPairExecution):
    if not pair.scientifically_valid or pair.treatment_evidence is None or pair.control_evidence is None:
        raise AdaptivePilotPostRunError("post-run analysis requires scientifically valid complete pairs")
    rows: list[tuple[Any, Any, MatchedTrajectoryContrast]] = []
    for horizon in FROZEN_CUMULATIVE_HORIZONS:
        if horizon > pair.hands_per_position:
            continue
        treatment = cumulative_arm_estimate(pair.treatment_evidence, horizon=horizon)
        control = cumulative_arm_estimate(pair.control_evidence, horizon=horizon)
        rows.append((treatment, control, matched_trajectory_contrast(treatment=treatment, control=control)))
    for segment_id, (_, end_hand) in FROZEN_SEGMENTS.items():
        if end_hand > pair.hands_per_position:
            continue
        treatment = segment_arm_estimate(pair.treatment_evidence, segment_id=segment_id)
        control = segment_arm_estimate(pair.control_evidence, segment_id=segment_id)
        rows.append((treatment, control, matched_trajectory_contrast(treatment=treatment, control=control)))
    return tuple(rows)


def _profile_summaries(pairs: Sequence[AdaptivePilotPairExecution]) -> tuple[AdaptivePilotWindowSummary, ...]:
    rows: list[tuple[Any, Any, MatchedTrajectoryContrast]] = []
    for pair in pairs:
        rows.extend(_window_estimates(pair))
    summaries: list[AdaptivePilotWindowSummary] = []
    profiles = tuple(dict.fromkeys(pair.profile_id for pair in pairs))
    window_order = (
        "cumulative_10", "cumulative_20", "cumulative_50", "cumulative_100", "cumulative_500",
        "early_adaptation", "main_adaptation", "steady_state",
    )
    for profile_id in profiles:
        for window_id in window_order:
            selected = [row for row in rows if row[2].profile_id == profile_id and row[2].window_id == window_id]
            if not selected:
                continue
            contrasts = tuple(row[2] for row in selected)
            first = contrasts[0]
            variance = None
            estimator = None
            if first.window_kind == "cumulative" and first.end_hand in (10, 20, 50, 100):
                estimate = profile_contrast_sample_sd(contrasts)
                variance = estimate.std_difference_bb100
                estimator = estimate.estimator_version
            summaries.append(
                AdaptivePilotWindowSummary(
                    scope_kind="profile",
                    scope_id=profile_id,
                    population_version=None,
                    population_hash=None,
                    window_kind=first.window_kind,
                    window_id=first.window_id,
                    start_hand=first.start_hand,
                    end_hand=first.end_hand,
                    num_seed_contrasts=len(contrasts),
                    master_seeds=tuple(sorted(item.master_seed for item in contrasts)),
                    mean_treatment_bb100=statistics.fmean(row[0].bb100 for row in selected),
                    mean_control_bb100=statistics.fmean(row[1].bb100 for row in selected),
                    mean_difference_bb100=statistics.fmean(item.difference_bb100 for item in contrasts),
                    std_difference_bb100=variance,
                    variance_estimator=estimator,
                )
            )
    return tuple(summaries)


def _population_summaries(
    pairs: Sequence[AdaptivePilotPairExecution],
    populations: Sequence[DiscretePopulationDefinition],
) -> tuple[AdaptivePilotWindowSummary, ...]:
    all_rows: list[tuple[Any, Any, MatchedTrajectoryContrast]] = []
    for pair in pairs:
        all_rows.extend(_window_estimates(pair))
    summaries: list[AdaptivePilotWindowSummary] = []
    window_order = (
        "cumulative_10", "cumulative_20", "cumulative_50", "cumulative_100", "cumulative_500",
        "early_adaptation", "main_adaptation", "steady_state",
    )
    for population in populations:
        weight_map = dict(population.weights_ppm)
        for window_id in window_order:
            window_rows = [row for row in all_rows if row[2].window_id == window_id]
            if not window_rows:
                continue
            seed_values: list[tuple[int, float, float, Any]] = []
            for seed in sorted({row[2].master_seed for row in window_rows}):
                seed_rows = [row for row in window_rows if row[2].master_seed == seed]
                contrasts = tuple(row[2] for row in seed_rows)
                population_difference = population_weighted_seed_contrast(
                    contrasts, population=population
                )
                by_profile = {row[2].profile_id: row for row in seed_rows}
                if set(by_profile) != set(weight_map):
                    raise AdaptivePilotPostRunError("population window is missing a frozen profile cell")
                treatment_mean = sum(
                    by_profile[profile][0].bb100 * weight for profile, weight in weight_map.items()
                ) / 1_000_000
                control_mean = sum(
                    by_profile[profile][1].bb100 * weight for profile, weight in weight_map.items()
                ) / 1_000_000
                seed_values.append((seed, treatment_mean, control_mean, population_difference))
            first = seed_values[0][3]
            variance = None
            estimator = None
            if first.window_kind == "cumulative" and first.end_hand in (10, 20, 50, 100):
                estimate = population_contrast_sample_sd(tuple(item[3] for item in seed_values))
                variance = estimate.std_difference_bb100
                estimator = estimate.estimator_version
            summaries.append(
                AdaptivePilotWindowSummary(
                    scope_kind="population",
                    scope_id=population.population_id,
                    population_version=population.version,
                    population_hash=population.population_hash,
                    window_kind=first.window_kind,
                    window_id=first.window_id,
                    start_hand=first.start_hand,
                    end_hand=first.end_hand,
                    num_seed_contrasts=len(seed_values),
                    master_seeds=tuple(item[0] for item in seed_values),
                    mean_treatment_bb100=statistics.fmean(item[1] for item in seed_values),
                    mean_control_bb100=statistics.fmean(item[2] for item in seed_values),
                    mean_difference_bb100=statistics.fmean(item[3].difference_bb100 for item in seed_values),
                    std_difference_bb100=variance,
                    variance_estimator=estimator,
                )
            )
    return tuple(summaries)


def _validate_complete_audit(audit: AdaptivePilotCampaignAudit) -> None:
    if audit.terminal_jobs != 50 or audit.completed_jobs != 50 or audit.failed_jobs != 0:
        raise AdaptivePilotPostRunError("post-run report requires exactly 50 completed jobs and zero failures")
    if audit.valid_pairs != 25 or audit.invalid_pairs != 0 or len(audit.pair_executions) != 25:
        raise AdaptivePilotPostRunError("post-run report requires exactly 25 valid pairs and zero invalid pairs")
    if audit.recorded_engine_hands != 18_000:
        raise AdaptivePilotPostRunError("post-run report requires exactly 18,000 recorded engine hands")
    if audit.replayed_treatment_match_records != 25 or audit.replayed_learning_transitions != 9_000:
        raise AdaptivePilotPostRunError("post-run replay audit must cover all 25 treatment records and 9,000 transitions")
    if len(audit.pair_evidence) != 25:
        raise AdaptivePilotPostRunError("post-run audit must retain exactly 25 pair evidence rows")
    if len(audit.learning_diagnostics) != 210:
        raise AdaptivePilotPostRunError("post-run report requires all 210 frozen learning-horizon diagnostics")


def build_adaptive_pilot_postrun_report(
    *,
    protocol: AdaptivePilotProtocol,
    execution_contract_id: str,
    execution_contract_version: str,
    execution_contract_hash: str,
    audit: AdaptivePilotCampaignAudit,
    primary_population: DiscretePopulationDefinition,
    secondary_population: DiscretePopulationDefinition,
) -> AdaptivePilotPostRunReport:
    """Build the frozen descriptive report without introducing H4 inference."""
    if not isinstance(protocol, AdaptivePilotProtocol):
        raise TypeError("protocol must be AdaptivePilotProtocol")
    _validate_sha256(execution_contract_hash, "execution_contract_hash")
    _validate_complete_audit(audit)
    profile_summaries = _profile_summaries(audit.pair_executions)
    population_summaries = _population_summaries(
        audit.pair_executions, (primary_population, secondary_population)
    )
    return AdaptivePilotPostRunReport(
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_hash=protocol.payload_hash,
        execution_contract_id=execution_contract_id,
        execution_contract_version=execution_contract_version,
        execution_contract_hash=execution_contract_hash,
        binding_hash=audit.binding_hash,
        manifest_hash=audit.manifest_hash,
        completed_jobs=audit.completed_jobs,
        failed_jobs=audit.failed_jobs,
        valid_pairs=audit.valid_pairs,
        invalid_pairs=audit.invalid_pairs,
        recorded_engine_hands=audit.recorded_engine_hands,
        total_arm_runtime_ns=audit.total_arm_runtime_ns,
        median_arm_runtime_ns=audit.median_arm_runtime_ns,
        database_bytes=audit.database_bytes,
        work_root_bytes=audit.work_root_bytes,
        replayed_treatment_match_records=audit.replayed_treatment_match_records,
        replayed_learning_transitions=audit.replayed_learning_transitions,
        pair_evidence=audit.pair_evidence,
        profile_summaries=profile_summaries,
        population_summaries=population_summaries,
        learning_diagnostics=audit.learning_diagnostics,
        complete=True,
        failure_free=True,
        final_h4_design=dict(_FINAL_H4_DESIGN),
        scientific_use=dict(_SCIENTIFIC_USE),
    )


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _audit_completed_campaign(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    database: Path,
    work_root: Path,
) -> AdaptivePilotCampaignAudit:
    store = SQLiteBenchmarkStore(database)
    manifest = load_adaptive_pilot_run_manifest(
        work_root / "run-manifest.json", protocol=protocol, expected_binding=binding
    )
    if manifest.terminal_jobs != 50 or any(state.status != "completed" for state in manifest.jobs):
        raise AdaptivePilotPostRunError("campaign integrity requires all 50 jobs completed")
    expected_ids = {state.match_id for state in manifest.jobs}
    actual_ids = set(store.list_match_ids())
    if actual_ids != expected_ids:
        raise AdaptivePilotPostRunError("campaign SQLite MatchRecord identities differ from the canonical 50-job manifest")

    jobs = iter_adaptive_pilot_jobs(protocol)
    snapshot_dir = work_root / "arm-snapshots"
    expected_snapshot_names = {f"{adaptive_pilot_job_key(job)}.json" for job in jobs}
    if {path.name for path in snapshot_dir.glob("*.json")} != expected_snapshot_names:
        raise AdaptivePilotPostRunError("campaign arm snapshots do not exactly match the canonical 50-job plan")
    checkpoint_dir = work_root / "pair-checkpoints"
    expected_checkpoint_names = {
        f"{profile_id}__{seed}.json" for profile_id in protocol.opponent_profile_ids for seed in protocol.master_seeds
    }
    if {path.name for path in checkpoint_dir.glob("*.json")} != expected_checkpoint_names:
        raise AdaptivePilotPostRunError("campaign pair checkpoints do not exactly match the canonical 25-pair plan")

    snapshots: dict[tuple[str, str, int], AdaptivePilotArmSnapshot] = {}
    runtimes: list[int] = []
    recorded_engine_hands = 0
    for job in jobs:
        snapshot = load_adaptive_pilot_arm_snapshot(
            snapshot_dir / f"{adaptive_pilot_job_key(job)}.json",
            protocol=protocol,
            expected_binding=binding,
            expected_job=job,
        )
        if snapshot.status != "completed":
            raise AdaptivePilotPostRunError("campaign audit found a non-completed arm snapshot")
        persisted = store.load_match(snapshot.match_id)
        if persisted.record_hash != snapshot.match_record_hash:
            raise AdaptivePilotPostRunError("campaign SQLite record differs from immutable arm snapshot")
        snapshots[(job.arm_id, job.opponent_profile_id, job.master_seed)] = snapshot
        runtimes.append(snapshot.elapsed_ns)
        recorded_engine_hands += snapshot.recorded_engine_hands

    pair_evidence: list[AdaptivePilotPairAuditEvidence] = []
    pair_executions: list[AdaptivePilotPairExecution] = []
    diagnostics: list[AdaptivePilotLearningHorizonDiagnostic] = []
    replayed_transitions = 0
    by_identity = {(job.arm_id, job.opponent_profile_id, job.master_seed): job for job in jobs}
    for profile_id in protocol.opponent_profile_ids:
        for seed in protocol.master_seeds:
            treatment_job = by_identity[("treatment", profile_id, seed)]
            control_job = by_identity[("control", profile_id, seed)]
            treatment = snapshots[("treatment", profile_id, seed)]
            control = snapshots[("control", profile_id, seed)]
            checkpoint = _load_pair_checkpoint(
                checkpoint_dir / f"{profile_id}__{seed}.json",
                binding=binding,
                treatment=treatment,
                control=control,
            )
            if not checkpoint.scientifically_valid:
                raise AdaptivePilotPostRunError("campaign audit found an invalid matched-pair checkpoint")
            try:
                execution = finalize_adaptive_pilot_pair(
                    protocol=protocol,
                    treatment_job=treatment_job,
                    control_job=control_job,
                    treatment_result=treatment.to_paired_result(),
                    control_result=control.to_paired_result(),
                    benchmark_store=store,
                )
            except (TypeError, ValueError) as exc:
                raise AdaptivePilotPostRunError(f"matched-pair replay audit failed: {exc}") from exc
            if not execution.scientifically_valid:
                raise AdaptivePilotPostRunError("replayed matched pair is scientifically invalid")
            horizons = tuple(
                horizon for horizon in FROZEN_CUMULATIVE_HORIZONS if horizon <= execution.hands_per_position
            )
            pair_diagnostics = learning_horizon_diagnostics(
                treatment.match_record,
                profile_id=profile_id,
                master_seed=seed,
                horizons=horizons,
            )
            diagnostics.extend(pair_diagnostics)
            replayed_transitions += sum(
                len(tuple(event for event in hand.adaptive_learning_events if event.learner_bot_id == 0))
                for hand in treatment.match_record.hands
            )
            pair_executions.append(execution)
            pair_evidence.append(
                AdaptivePilotPairAuditEvidence(
                    profile_id=profile_id,
                    master_seed=seed,
                    hands_per_position=execution.hands_per_position,
                    treatment_match_id=execution.treatment_match_id,
                    control_match_id=execution.control_match_id,
                    treatment_record_hash=execution.treatment_record_hash,
                    control_record_hash=execution.control_record_hash,
                    treatment_snapshot_hash=treatment.snapshot_hash,
                    control_snapshot_hash=control.snapshot_hash,
                    pair_checkpoint_hash=checkpoint.checkpoint_hash,
                )
            )

    audit = AdaptivePilotCampaignAudit(
        binding_hash=binding.binding_hash,
        manifest_hash=manifest.manifest_hash,
        terminal_jobs=manifest.terminal_jobs,
        completed_jobs=sum(state.status == "completed" for state in manifest.jobs),
        failed_jobs=sum(state.status == "failed" for state in manifest.jobs),
        valid_pairs=len(pair_executions),
        invalid_pairs=0,
        recorded_engine_hands=recorded_engine_hands,
        total_arm_runtime_ns=sum(runtimes),
        median_arm_runtime_ns=float(statistics.median(runtimes)),
        database_bytes=database.stat().st_size,
        work_root_bytes=_directory_bytes(work_root),
        replayed_treatment_match_records=len(pair_executions),
        replayed_learning_transitions=replayed_transitions,
        pair_evidence=tuple(pair_evidence),
        pair_executions=tuple(pair_executions),
        learning_diagnostics=tuple(diagnostics),
    )
    _validate_complete_audit(audit)
    return audit


def _validate_binding_contract(
    binding: AdaptivePilotExecutionBinding,
    contract: AdaptivePilotParallelExecutionContract,
) -> None:
    if (
        binding.execution_contract_id != contract.execution_contract_id
        or binding.execution_contract_version != contract.execution_contract_version
        or binding.execution_contract_hash != contract.payload_hash
        or binding.workers != contract.workers
    ):
        raise AdaptivePilotPostRunError("persisted execution binding differs from frozen Spec-34 contract")


def _validate_population_binding(
    binding: AdaptivePilotExecutionBinding,
    population: DiscretePopulationDefinition,
) -> None:
    identity = (population.population_id, population.version, population.population_hash)
    if identity not in binding.population_versions:
        raise AdaptivePilotPostRunError("post-run population differs from persisted execution binding")


def _analyze_from_paths(
    *,
    protocol_path: str | Path,
    execution_contract_path: str | Path,
    primary_population_path: str | Path,
    secondary_population_path: str | Path,
    database: str | Path,
    work_root: str | Path,
) -> AdaptivePilotPostRunReport:
    protocol = load_adaptive_pilot_protocol(protocol_path)
    contract = load_adaptive_pilot_parallel_execution_contract(execution_contract_path)
    root = Path(work_root)
    binding = load_adaptive_pilot_execution_binding(root / "binding.json")
    _validate_binding_contract(binding, contract)
    if binding.pilot_protocol_hash != protocol.payload_hash:
        raise AdaptivePilotPostRunError("persisted binding differs from frozen Spec-32 protocol")
    primary = load_frozen_population(primary_population_path)
    secondary = load_frozen_population(secondary_population_path)
    _validate_population_binding(binding, primary)
    _validate_population_binding(binding, secondary)
    audit = _audit_completed_campaign(
        protocol=protocol,
        binding=binding,
        database=Path(database),
        work_root=root,
    )
    return build_adaptive_pilot_postrun_report(
        protocol=protocol,
        execution_contract_id=contract.execution_contract_id,
        execution_contract_version=contract.execution_contract_version,
        execution_contract_hash=contract.payload_hash,
        audit=audit,
        primary_population=primary,
        secondary_population=secondary,
    )


def _atomic_write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_file_bytes(value)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def save_adaptive_pilot_postrun_report(
    report: AdaptivePilotPostRunReport,
    path: str | Path,
) -> Path:
    if not isinstance(report, AdaptivePilotPostRunReport):
        raise TypeError("report must be AdaptivePilotPostRunReport")
    destination = Path(path)
    if destination.exists():
        existing = load_adaptive_pilot_postrun_report(destination)
        if existing.report_hash != report.report_hash:
            raise AdaptivePilotPostRunError("existing post-run report differs from current audited evidence")
        return destination
    return _atomic_write_json(destination, report.to_dict())


def _metric_from_payload(payload: Mapping[str, Any]) -> AdaptivePilotLearningMetricDiagnostic:
    return AdaptivePilotLearningMetricDiagnostic(
        metric=str(payload["metric"]), opportunities=int(payload["opportunities"]),
        events=int(payload["events"]), alpha=int(payload["alpha"]), beta=int(payload["beta"]),
    )


def _diagnostic_from_payload(payload: Mapping[str, Any]) -> AdaptivePilotLearningHorizonDiagnostic:
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, list):
        raise AdaptivePilotPostRunError("learning diagnostic metrics must be an array")
    return AdaptivePilotLearningHorizonDiagnostic(
        profile_id=str(payload["profile_id"]), master_seed=int(payload["master_seed"]),
        config_id=str(payload["config_id"]), horizon=int(payload["horizon"]),
        trajectory_hand_count=int(payload["trajectory_hand_count"]),
        valid_learning_transition_count=int(payload["valid_learning_transition_count"]),
        posterior_state_hash=str(payload["posterior_state_hash"]),
        metrics=tuple(_metric_from_payload(item) for item in raw_metrics),
    )


def _pair_evidence_from_payload(payload: Mapping[str, Any]) -> AdaptivePilotPairAuditEvidence:
    return AdaptivePilotPairAuditEvidence(
        profile_id=str(payload["profile_id"]), master_seed=int(payload["master_seed"]),
        hands_per_position=int(payload["hands_per_position"]),
        treatment_match_id=str(payload["treatment_match_id"]), control_match_id=str(payload["control_match_id"]),
        treatment_record_hash=str(payload["treatment_record_hash"]), control_record_hash=str(payload["control_record_hash"]),
        treatment_snapshot_hash=str(payload["treatment_snapshot_hash"]), control_snapshot_hash=str(payload["control_snapshot_hash"]),
        pair_checkpoint_hash=str(payload["pair_checkpoint_hash"]),
    )


def _window_from_payload(payload: Mapping[str, Any]) -> AdaptivePilotWindowSummary:
    return AdaptivePilotWindowSummary(
        scope_kind=str(payload["scope_kind"]), scope_id=str(payload["scope_id"]),
        population_version=None if payload.get("population_version") is None else str(payload["population_version"]),
        population_hash=None if payload.get("population_hash") is None else str(payload["population_hash"]),
        window_kind=str(payload["window_kind"]), window_id=str(payload["window_id"]),
        start_hand=int(payload["start_hand"]), end_hand=int(payload["end_hand"]),
        num_seed_contrasts=int(payload["num_seed_contrasts"]),
        master_seeds=tuple(int(seed) for seed in payload["master_seeds"]),
        mean_treatment_bb100=float(payload["mean_treatment_bb100"]),
        mean_control_bb100=float(payload["mean_control_bb100"]),
        mean_difference_bb100=float(payload["mean_difference_bb100"]),
        std_difference_bb100=None if payload.get("std_difference_bb100") is None else float(payload["std_difference_bb100"]),
        variance_estimator=None if payload.get("variance_estimator") is None else str(payload["variance_estimator"]),
        inferential_statistics_permitted=bool(payload["inferential_statistics_permitted"]),
    )


def load_adaptive_pilot_postrun_report(path: str | Path) -> AdaptivePilotPostRunReport:
    source = Path(path)
    raw = source.read_bytes()
    document = json.loads(raw)
    if raw != _canonical_file_bytes(document):
        raise AdaptivePilotPostRunError("post-run report file is not canonical JSON/LF")
    if not isinstance(document, Mapping) or not isinstance(document.get("report"), Mapping):
        raise AdaptivePilotPostRunError("post-run report file is malformed")
    payload = document["report"]
    stored_hash = _validate_sha256(str(document.get("report_hash")), "report_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise AdaptivePilotPostRunError("report_hash does not match post-run report payload")
    scientific_use = payload.get("scientific_use")
    if not isinstance(scientific_use, Mapping):
        raise AdaptivePilotPostRunError("post-run report scientific-use block is missing")
    if scientific_use.get("confirmatory_h4_inference") is not False:
        raise AdaptivePilotPostRunError("pilot report cannot contain confirmatory H4 inference")
    final_h4 = payload.get("final_h4_design")
    if not isinstance(final_h4, Mapping):
        raise AdaptivePilotPostRunError("post-run report final-H4 boundary is missing")
    report = AdaptivePilotPostRunReport(
        report_version=str(payload["report_version"]), analysis_version=str(payload["analysis_version"]),
        protocol_id=str(payload["protocol_id"]), protocol_version=str(payload["protocol_version"]),
        protocol_hash=str(payload["protocol_hash"]), execution_contract_id=str(payload["execution_contract_id"]),
        execution_contract_version=str(payload["execution_contract_version"]),
        execution_contract_hash=str(payload["execution_contract_hash"]), binding_hash=str(payload["binding_hash"]),
        manifest_hash=str(payload["manifest_hash"]), completed_jobs=int(payload["completed_jobs"]),
        failed_jobs=int(payload["failed_jobs"]), valid_pairs=int(payload["valid_pairs"]), invalid_pairs=int(payload["invalid_pairs"]),
        recorded_engine_hands=int(payload["recorded_engine_hands"]), total_arm_runtime_ns=int(payload["total_arm_runtime_ns"]),
        median_arm_runtime_ns=float(payload["median_arm_runtime_ns"]), database_bytes=int(payload["database_bytes"]),
        work_root_bytes=int(payload["work_root_bytes"]), replayed_treatment_match_records=int(payload["replayed_treatment_match_records"]),
        replayed_learning_transitions=int(payload["replayed_learning_transitions"]),
        pair_evidence=tuple(_pair_evidence_from_payload(item) for item in payload["pair_evidence"]),
        profile_summaries=tuple(_window_from_payload(item) for item in payload["profile_summaries"]),
        population_summaries=tuple(_window_from_payload(item) for item in payload["population_summaries"]),
        learning_diagnostics=tuple(_diagnostic_from_payload(item) for item in payload["learning_diagnostics"]),
        complete=bool(payload["complete"]), failure_free=bool(payload["failure_free"]),
        final_h4_design=dict(final_h4), scientific_use=dict(scientific_use),
    )
    if report.report_hash != stored_hash:
        raise AdaptivePilotPostRunError("loaded post-run report is not semantically canonical")
    return report


def analyze_adaptive_pilot_postrun(
    *,
    protocol_path: str | Path,
    execution_contract_path: str | Path,
    primary_population_path: str | Path,
    secondary_population_path: str | Path,
    database: str | Path,
    work_root: str | Path,
    report_path: str | Path,
) -> AdaptivePilotPostRunReport:
    """Audit completed evidence, build the descriptive report, and save it immutably."""
    report = _analyze_from_paths(
        protocol_path=protocol_path,
        execution_contract_path=execution_contract_path,
        primary_population_path=primary_population_path,
        secondary_population_path=secondary_population_path,
        database=database,
        work_root=work_root,
    )
    save_adaptive_pilot_postrun_report(report, report_path)
    return report


def render_adaptive_pilot_postrun_report_text(report: AdaptivePilotPostRunReport) -> str:
    primary = [
        item for item in report.population_summaries
        if item.scope_id == "baseline-synthetic-static-profiles" and item.window_kind == "cumulative"
    ]
    lines = [
        "Phase-2 adaptive pilot post-run report",
        f"  status: {'COMPLETE' if report.complete else 'INCOMPLETE'}",
        f"  failure free: {'yes' if report.failure_free else 'no'}",
        f"  completed jobs: {report.completed_jobs}/50",
        f"  valid pairs: {report.valid_pairs}/25",
        f"  recorded engine hands: {report.recorded_engine_hands:,}",
        f"  replayed learning transitions: {report.replayed_learning_transitions:,}",
        "  Baseline Synthetic adaptive-minus-control means (BB/100):",
    ]
    for item in primary:
        variance = "descriptive-only" if item.std_difference_bb100 is None else f"sd={item.std_difference_bb100:.6f}"
        lines.append(f"    h={item.end_hand}: delta={item.mean_difference_bb100:.6f}, {variance}")
    lines.extend(
        [
            "  confirmatory H4 inference: prohibited",
            "  final H4 effect estimate: prohibited",
            f"  report hash: {report.report_hash}",
        ]
    )
    return "\n".join(lines) + "\n"
