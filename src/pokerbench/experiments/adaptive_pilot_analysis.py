"""Spec-32 analysis primitives for the Phase-2 adaptive pilot.

This module operates only on complete trajectory-cluster evidence. It never
promotes dependent hands inside one adaptive trajectory to independent
statistical observations.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from pokerbench.evaluation.population import (
    POPULATION_WEIGHT_SCALE,
    DiscretePopulationDefinition,
)


ADAPTIVE_PILOT_VARIANCE_ESTIMATOR = (
    "sample-sd-matched-trajectory-contrast-bb100-ddof1-v1"
)

FROZEN_CUMULATIVE_HORIZONS = (10, 20, 50, 100, 500)
VARIANCE_CUMULATIVE_HORIZONS = (10, 20, 50, 100)
FROZEN_SEGMENTS = {
    "early_adaptation": (1, 20),
    "main_adaptation": (21, 100),
    "steady_state": (101, 500),
}
FROZEN_MASTER_SEEDS = (
    6_300_000,
    6_300_001,
    6_300_002,
    6_300_003,
    6_300_004,
)
LONG_HORIZON_MASTER_SEED = 6_300_000


class AdaptivePilotAnalysisError(ValueError):
    """Raised when Spec-32 analysis evidence or aggregation is invalid."""


class AdaptivePilotInferenceProhibitedError(AdaptivePilotAnalysisError):
    """Raised when v1 analysis attempts prohibited inferential use."""


@dataclass(frozen=True)
class TrajectoryArmEvidence:
    """Per-hand hero profit for one complete arm/profile/seed trajectory cluster."""

    arm_id: str
    profile_id: str
    master_seed: int
    hero_btn_profit_bb: tuple[float, ...]
    hero_bb_profit_bb: tuple[float, ...]
    scientifically_valid: bool = True

    def __post_init__(self) -> None:
        if self.arm_id not in {"treatment", "control"}:
            raise AdaptivePilotAnalysisError(
                "arm_id must be 'treatment' or 'control'"
            )
        if not self.profile_id:
            raise AdaptivePilotAnalysisError("profile_id must not be empty")
        if isinstance(self.master_seed, bool) or not isinstance(self.master_seed, int):
            raise AdaptivePilotAnalysisError("master_seed must be an integer")
        if self.master_seed not in FROZEN_MASTER_SEEDS:
            raise AdaptivePilotAnalysisError(
                "master_seed must be a predeclared pilot seed; replacement seeds are forbidden"
            )
        if not isinstance(self.scientifically_valid, bool):
            raise AdaptivePilotAnalysisError("scientifically_valid must be boolean")
        if len(self.hero_btn_profit_bb) != len(self.hero_bb_profit_bb):
            raise AdaptivePilotAnalysisError(
                "the two positional profit trajectories must have equal length"
            )
        for field_name, values in (
            ("hero_btn_profit_bb", self.hero_btn_profit_bb),
            ("hero_bb_profit_bb", self.hero_bb_profit_bb),
        ):
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise AdaptivePilotAnalysisError(
                        f"{field_name} values must be numeric"
                    )
                if not math.isfinite(float(value)):
                    raise AdaptivePilotAnalysisError(
                        f"{field_name} values must be finite"
                    )


@dataclass(frozen=True)
class ArmWindowEstimate:
    """One position-compensated BB/100 estimate from one trajectory cluster."""

    arm_id: str
    profile_id: str
    master_seed: int
    window_kind: str
    window_id: str
    start_hand: int
    end_hand: int
    bb100: float


@dataclass(frozen=True)
class MatchedTrajectoryContrast:
    """One independent adaptive-minus-control contrast for a profile and seed."""

    profile_id: str
    master_seed: int
    window_kind: str
    window_id: str
    start_hand: int
    end_hand: int
    treatment_bb100: float
    control_bb100: float
    difference_bb100: float


@dataclass(frozen=True)
class PopulationSeedContrast:
    """Within-seed population-weighted contrast across the frozen profiles."""

    population_id: str
    population_version: str
    population_hash: str
    master_seed: int
    window_kind: str
    window_id: str
    start_hand: int
    end_hand: int
    num_profile_cells: int
    difference_bb100: float


@dataclass(frozen=True)
class PilotVarianceEstimate:
    """Sample SD over independent seed-level trajectory contrasts."""

    estimator_version: str
    scope_kind: str
    scope_id: str
    horizon: int
    num_independent_clusters: int
    master_seeds: tuple[int, ...]
    std_difference_bb100: float


def _require_valid_evidence(evidence: TrajectoryArmEvidence) -> None:
    if not isinstance(evidence, TrajectoryArmEvidence):
        raise TypeError("evidence must be TrajectoryArmEvidence")
    if not evidence.scientifically_valid:
        raise AdaptivePilotAnalysisError(
            "scientifically invalid trajectory evidence cannot enter matched analysis"
        )


def _require_long_seed_for_window(evidence: TrajectoryArmEvidence, end_hand: int) -> None:
    if end_hand > 100 and evidence.master_seed != LONG_HORIZON_MASTER_SEED:
        raise AdaptivePilotAnalysisError(
            "hands beyond 100 are defined only for the predeclared long-horizon seed"
        )


def _arm_window_estimate(
    evidence: TrajectoryArmEvidence,
    *,
    window_kind: str,
    window_id: str,
    start_hand: int,
    end_hand: int,
) -> ArmWindowEstimate:
    _require_valid_evidence(evidence)
    _require_long_seed_for_window(evidence, end_hand)
    if len(evidence.hero_btn_profit_bb) < end_hand:
        raise AdaptivePilotAnalysisError(
            f"analysis window requires {end_hand} hands per positional trajectory"
        )

    start_index = start_hand - 1
    num_hands = end_hand - start_hand + 1
    btn_profit = sum(evidence.hero_btn_profit_bb[start_index:end_hand])
    bb_profit = sum(evidence.hero_bb_profit_bb[start_index:end_hand])
    bb100 = 100.0 * (btn_profit + bb_profit) / (2.0 * num_hands)

    return ArmWindowEstimate(
        arm_id=evidence.arm_id,
        profile_id=evidence.profile_id,
        master_seed=evidence.master_seed,
        window_kind=window_kind,
        window_id=window_id,
        start_hand=start_hand,
        end_hand=end_hand,
        bb100=bb100,
    )


def cumulative_arm_estimate(
    evidence: TrajectoryArmEvidence,
    *,
    horizon: int,
) -> ArmWindowEstimate:
    """Compute Spec-32 Y(a,p,s,h) for one frozen cumulative horizon."""
    if horizon not in FROZEN_CUMULATIVE_HORIZONS:
        raise AdaptivePilotAnalysisError(
            "horizon must be a frozen cumulative horizon: 10, 20, 50, 100, or 500"
        )
    return _arm_window_estimate(
        evidence,
        window_kind="cumulative",
        window_id=f"cumulative_{horizon}",
        start_hand=1,
        end_hand=horizon,
    )


def segment_arm_estimate(
    evidence: TrajectoryArmEvidence,
    *,
    segment_id: str,
) -> ArmWindowEstimate:
    """Compute the position-compensated BB/100 estimate for a frozen segment."""
    interval = FROZEN_SEGMENTS.get(segment_id)
    if interval is None:
        raise AdaptivePilotAnalysisError(
            "segment_id must identify a frozen segment"
        )
    start_hand, end_hand = interval
    return _arm_window_estimate(
        evidence,
        window_kind="segment",
        window_id=segment_id,
        start_hand=start_hand,
        end_hand=end_hand,
    )


def matched_trajectory_contrast(
    *,
    treatment: ArmWindowEstimate | None,
    control: ArmWindowEstimate | None,
) -> MatchedTrajectoryContrast:
    """Return one complete-pair treatment-minus-control trajectory contrast."""
    if treatment is None or control is None:
        raise AdaptivePilotAnalysisError(
            "matched analysis requires a complete treatment/control pair"
        )
    if not isinstance(treatment, ArmWindowEstimate) or not isinstance(
        control, ArmWindowEstimate
    ):
        raise TypeError("treatment and control must be ArmWindowEstimate objects")
    if treatment.arm_id != "treatment" or control.arm_id != "control":
        raise AdaptivePilotAnalysisError(
            "matched contrast requires treatment and control arm identities"
        )
    treatment_identity = (
        treatment.profile_id,
        treatment.master_seed,
        treatment.window_kind,
        treatment.window_id,
        treatment.start_hand,
        treatment.end_hand,
    )
    control_identity = (
        control.profile_id,
        control.master_seed,
        control.window_kind,
        control.window_id,
        control.start_hand,
        control.end_hand,
    )
    if treatment_identity != control_identity:
        raise AdaptivePilotAnalysisError(
            "matched treatment/control estimates must use the same profile, seed, and window"
        )

    return MatchedTrajectoryContrast(
        profile_id=treatment.profile_id,
        master_seed=treatment.master_seed,
        window_kind=treatment.window_kind,
        window_id=treatment.window_id,
        start_hand=treatment.start_hand,
        end_hand=treatment.end_hand,
        treatment_bb100=treatment.bb100,
        control_bb100=control.bb100,
        difference_bb100=treatment.bb100 - control.bb100,
    )


def _require_variance_window(
    *,
    window_kind: str,
    start_hand: int,
    end_hand: int,
) -> int:
    if window_kind == "cumulative" and start_hand == 1 and end_hand == 500:
        raise AdaptivePilotInferenceProhibitedError(
            "Spec-32 v1 prohibits variance, CI, and inferential statistics at 500 hands"
        )
    if (
        window_kind != "cumulative"
        or start_hand != 1
        or end_hand not in VARIANCE_CUMULATIVE_HORIZONS
    ):
        raise AdaptivePilotInferenceProhibitedError(
            "Spec-32 v1 defines variance only for cumulative horizons 10, 20, 50, and 100"
        )
    return end_hand


def _validate_unique_seeds(master_seeds: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(master_seeds)
    if len(set(seeds)) != len(seeds):
        raise AdaptivePilotAnalysisError(
            "variance evidence must contain unique independent master seeds"
        )
    for seed in seeds:
        if seed not in FROZEN_MASTER_SEEDS:
            raise AdaptivePilotAnalysisError(
                "variance evidence contains a non-predeclared replacement seed"
            )
    return tuple(sorted(seeds))


def profile_contrast_sample_sd(
    contrasts: Sequence[MatchedTrajectoryContrast],
) -> PilotVarianceEstimate:
    """Compute Spec-32 sample SD across seed-level profile contrasts."""
    items = tuple(contrasts)
    if not items:
        raise AdaptivePilotAnalysisError("at least one profile contrast is required")
    first = items[0]
    horizon = _require_variance_window(
        window_kind=first.window_kind,
        start_hand=first.start_hand,
        end_hand=first.end_hand,
    )
    identity = (
        first.profile_id,
        first.window_kind,
        first.window_id,
        first.start_hand,
        first.end_hand,
    )
    for item in items:
        if (
            item.profile_id,
            item.window_kind,
            item.window_id,
            item.start_hand,
            item.end_hand,
        ) != identity:
            raise AdaptivePilotAnalysisError(
                "profile variance requires one profile and one cumulative window"
            )
    seeds = _validate_unique_seeds(item.master_seed for item in items)
    if len(items) < 2:
        raise AdaptivePilotAnalysisError(
            "sample standard deviation requires at least two independent seed contrasts"
        )
    std_bb100 = statistics.stdev(item.difference_bb100 for item in items)
    return PilotVarianceEstimate(
        estimator_version=ADAPTIVE_PILOT_VARIANCE_ESTIMATOR,
        scope_kind="profile",
        scope_id=first.profile_id,
        horizon=horizon,
        num_independent_clusters=len(items),
        master_seeds=seeds,
        std_difference_bb100=std_bb100,
    )


def population_weighted_seed_contrast(
    contrasts: Sequence[MatchedTrajectoryContrast],
    *,
    population: DiscretePopulationDefinition,
) -> PopulationSeedContrast:
    """Aggregate profile contrasts within one seed before any across-seed variance."""
    if not isinstance(population, DiscretePopulationDefinition):
        raise TypeError("population must be a DiscretePopulationDefinition")
    items = tuple(contrasts)
    if not items:
        raise AdaptivePilotAnalysisError("population weighting requires profile contrasts")

    profile_ids = tuple(item.profile_id for item in items)
    if len(set(profile_ids)) != len(profile_ids):
        raise AdaptivePilotAnalysisError(
            "population weighting rejects duplicate profile contrasts"
        )
    expected_profile_ids = tuple(profile_id for profile_id, _ in population.weights_ppm)
    if set(profile_ids) != set(expected_profile_ids):
        raise AdaptivePilotAnalysisError(
            "population contrast profiles must exactly match the frozen population profiles"
        )

    first = items[0]
    identity = (
        first.master_seed,
        first.window_kind,
        first.window_id,
        first.start_hand,
        first.end_hand,
    )
    for item in items:
        if (
            item.master_seed,
            item.window_kind,
            item.window_id,
            item.start_hand,
            item.end_hand,
        ) != identity:
            raise AdaptivePilotAnalysisError(
                "population weighting requires all profile contrasts from the same seed and window"
            )

    by_profile = {item.profile_id: item for item in items}
    weighted_difference = sum(
        by_profile[profile_id].difference_bb100 * weight_ppm
        for profile_id, weight_ppm in population.weights_ppm
    ) / POPULATION_WEIGHT_SCALE

    return PopulationSeedContrast(
        population_id=population.population_id,
        population_version=population.version,
        population_hash=population.population_hash,
        master_seed=first.master_seed,
        window_kind=first.window_kind,
        window_id=first.window_id,
        start_hand=first.start_hand,
        end_hand=first.end_hand,
        num_profile_cells=len(items),
        difference_bb100=weighted_difference,
    )


def population_contrast_sample_sd(
    contrasts: Sequence[PopulationSeedContrast],
) -> PilotVarianceEstimate:
    """Compute sample SD across within-seed population-weighted contrasts."""
    items = tuple(contrasts)
    if not items:
        raise AdaptivePilotAnalysisError(
            "at least one population seed contrast is required"
        )
    first = items[0]
    horizon = _require_variance_window(
        window_kind=first.window_kind,
        start_hand=first.start_hand,
        end_hand=first.end_hand,
    )
    identity = (
        first.population_id,
        first.population_version,
        first.population_hash,
        first.window_kind,
        first.window_id,
        first.start_hand,
        first.end_hand,
    )
    for item in items:
        if (
            item.population_id,
            item.population_version,
            item.population_hash,
            item.window_kind,
            item.window_id,
            item.start_hand,
            item.end_hand,
        ) != identity:
            raise AdaptivePilotAnalysisError(
                "population variance requires one population and one cumulative window"
            )
    seeds = _validate_unique_seeds(item.master_seed for item in items)
    if len(items) < 2:
        raise AdaptivePilotAnalysisError(
            "sample standard deviation requires at least two independent seed contrasts"
        )
    std_bb100 = statistics.stdev(item.difference_bb100 for item in items)
    return PilotVarianceEstimate(
        estimator_version=ADAPTIVE_PILOT_VARIANCE_ESTIMATOR,
        scope_kind="population",
        scope_id=first.population_id,
        horizon=horizon,
        num_independent_clusters=len(items),
        master_seeds=seeds,
        std_difference_bb100=std_bb100,
    )
