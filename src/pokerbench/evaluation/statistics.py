# src/pokerbench/evaluation/statistics.py
"""
Statistical evaluation for paired poker-agent experiments.

The module operates on independent statistical clusters.

For static policies:
    one paired deal may be one independent cluster.

For adaptive/stateful policies:
    one complete paired trajectory is one independent cluster.

Dependent hands inside an adaptive trajectory are never bootstrap-resampled
individually.

All public performance statistics are reported in BB/100.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

from pokerbench.runner import (
    PairedResult,
    StatisticalUnit,
)


EVALUATION_VERSION = "paired-statistics-v1"
BOOTSTRAP_PERCENTILE_METHOD = "linear-n-minus-1-v1"

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95

BB100_SCALE = 100.0


@dataclass(frozen=True)
class PairedCluster:
    """
    One independent statistical cluster.

    mean_difference_bb_per_hand is already normalised to a per-hand
    paired performance difference in big blinds.

    num_simulated_hands records the number of actual engine hand
    simulations represented by this cluster.

    For a static paired deal:
        num_simulated_hands = 2

    because the same deal is simulated once in each positional condition.

    For an adaptive trajectory containing N hands per positional condition:
        num_simulated_hands = 2 * N
    """

    mean_difference_bb_per_hand: float
    num_simulated_hands: int

    def __post_init__(self) -> None:
        if not math.isfinite(
            self.mean_difference_bb_per_hand
        ):
            raise ValueError(
                "mean_difference_bb_per_hand must be finite"
            )

        if self.num_simulated_hands <= 0:
            raise ValueError(
                "num_simulated_hands must be positive"
            )


@dataclass(frozen=True)
class ClusterDataset:
    """
    Independent cluster dataset prepared for statistical evaluation.
    """

    statistical_unit: StatisticalUnit
    clusters: tuple[PairedCluster, ...]
    num_failed_runs: int

    def __post_init__(self) -> None:
        if self.num_failed_runs < 0:
            raise ValueError(
                "num_failed_runs must be non-negative"
            )


@dataclass(frozen=True)
class StatisticalEvaluation:
    """
    Complete paired statistical summary.

    Mean, standard deviation, and confidence intervals are expressed
    in BB/100.

    effect_size_dz is dimensionless.

    effect_size_dz is None when the sample standard deviation is zero,
    because d_z = mean / SD is undefined in that case.
    """

    evaluation_version: str
    percentile_method: str

    statistical_unit: StatisticalUnit

    num_independent_units: int
    num_analyzed_simulated_hands: int
    num_failed_runs: int

    mean_difference_bb100: float
    std_difference_bb100: float

    normal_ci_low_bb100: float
    normal_ci_high_bb100: float

    bootstrap_ci_low_bb100: float
    bootstrap_ci_high_bb100: float

    effect_size_dz: float | None

    confidence_level: float

    bootstrap_resamples: int
    bootstrap_seed: int


def clusters_from_paired_results(
    results: Sequence[PairedResult],
) -> ClusterDataset:
    """
    Convert runner results into independent statistical clusters.

    Static policies
    ---------------
    Every paired difference is already one per-hand paired observation.
    Each successful paired deal corresponds to two actual engine hand
    simulations because both position assignments are executed.

    Adaptive policies
    -----------------
    Every PairedResult must contain at most one complete paired trajectory.

    The runner stores the adaptive paired difference as the total difference
    over num_hands hands per positional trajectory. It is therefore divided
    by num_hands only after the complete trajectory has been formed.

    This conversion is deliberately performed at trajectory level so that
    individual dependent hands never become independent statistical units.
    """
    if not results:
        raise ValueError(
            "At least one PairedResult is required"
        )

    statistical_unit = results[0].statistical_unit

    for result in results:
        if result.statistical_unit != statistical_unit:
            raise ValueError(
                "All PairedResult objects must use the same "
                "statistical unit"
            )

        if (
            result.num_valid_pairs
            != len(result.paired_differences)
        ):
            raise ValueError(
                "num_valid_pairs must match the number of "
                "paired_differences"
            )

    clusters: list[PairedCluster] = []
    num_failed_runs = sum(
        len(result.failed_runs)
        for result in results
    )

    if statistical_unit == StatisticalUnit.PAIRED_DEAL:
        for result in results:
            for difference_bb in result.paired_differences:
                clusters.append(
                    PairedCluster(
                        mean_difference_bb_per_hand=(
                            difference_bb
                        ),
                        num_simulated_hands=2,
                    )
                )

    elif (
        statistical_unit
        == StatisticalUnit.TRAJECTORY_CLUSTER
    ):
        for result in results:
            if len(result.paired_differences) > 1:
                raise ValueError(
                    "A trajectory-cluster PairedResult must not "
                    "contain multiple independent observations. "
                    "Dependent hands inside one trajectory must not "
                    "be treated as independent clusters."
                )

            if not result.paired_differences:
                continue

            if result.num_hands <= 0:
                raise ValueError(
                    "Adaptive trajectory results must have "
                    "num_hands > 0"
                )

            total_difference_bb = (
                result.paired_differences[0]
            )

            mean_difference_bb_per_hand = (
                total_difference_bb
                / result.num_hands
            )

            clusters.append(
                PairedCluster(
                    mean_difference_bb_per_hand=(
                        mean_difference_bb_per_hand
                    ),
                    num_simulated_hands=(
                        2 * result.num_hands
                    ),
                )
            )

    elif (
        statistical_unit
        == StatisticalUnit.INDEPENDENT_RUN
    ):
        raise ValueError(
            "Automatic conversion of INDEPENDENT_RUN "
            "PairedResult objects is not defined because the "
            "runner does not currently emit that design. "
            "Construct PairedCluster objects explicitly for "
            "independent-run experiments."
        )

    else:
        raise ValueError(
            f"Unsupported statistical unit: {statistical_unit}"
        )

    return ClusterDataset(
        statistical_unit=statistical_unit,
        clusters=tuple(clusters),
        num_failed_runs=num_failed_runs,
    )


def evaluate_paired_results(
    results: Sequence[PairedResult],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int = (
        DEFAULT_BOOTSTRAP_RESAMPLES
    ),
    confidence_level: float = (
        DEFAULT_CONFIDENCE_LEVEL
    ),
) -> StatisticalEvaluation:
    """
    Evaluate one or more runner results.

    This is the primary integration entry point for PairedRunner output.
    """
    dataset = clusters_from_paired_results(
        results
    )

    return evaluate_clusters(
        clusters=dataset.clusters,
        statistical_unit=dataset.statistical_unit,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        num_failed_runs=dataset.num_failed_runs,
    )


def evaluate_clusters(
    clusters: Sequence[PairedCluster],
    *,
    statistical_unit: StatisticalUnit,
    bootstrap_seed: int,
    bootstrap_resamples: int = (
        DEFAULT_BOOTSTRAP_RESAMPLES
    ),
    confidence_level: float = (
        DEFAULT_CONFIDENCE_LEVEL
    ),
    num_failed_runs: int = 0,
) -> StatisticalEvaluation:
    """
    Evaluate independent paired clusters.

    Exactly the objects supplied through `clusters` are resampled by the
    bootstrap. The evaluator never looks inside a cluster.

    Consequently:

    - static paired deals can be supplied individually;
    - adaptive trajectories remain intact;
    - temporally dependent hands cannot accidentally be resampled as
      independent observations.
    """
    _validate_evaluation_inputs(
        clusters=clusters,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        num_failed_runs=num_failed_runs,
    )

    values_bb = tuple(
        cluster.mean_difference_bb_per_hand
        for cluster in clusters
    )

    num_independent_units = len(values_bb)

    mean_bb = statistics.fmean(values_bb)
    std_bb = statistics.stdev(values_bb)

    mean_bb100 = (
        BB100_SCALE * mean_bb
    )
    std_bb100 = (
        BB100_SCALE * std_bb
    )

    normal_ci_low_bb100, normal_ci_high_bb100 = (
        _normal_confidence_interval(
            mean=mean_bb100,
            sample_std=std_bb100,
            sample_size=num_independent_units,
            confidence_level=confidence_level,
        )
    )

    (
        bootstrap_ci_low_bb100,
        bootstrap_ci_high_bb100,
    ) = _bootstrap_confidence_interval(
        values_bb=values_bb,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
    )

    effect_size_dz: float | None

    if std_bb == 0.0:
        effect_size_dz = None
    else:
        effect_size_dz = (
            mean_bb / std_bb
        )

    num_analyzed_simulated_hands = sum(
        cluster.num_simulated_hands
        for cluster in clusters
    )

    return StatisticalEvaluation(
        evaluation_version=EVALUATION_VERSION,
        percentile_method=(
            BOOTSTRAP_PERCENTILE_METHOD
        ),
        statistical_unit=statistical_unit,
        num_independent_units=(
            num_independent_units
        ),
        num_analyzed_simulated_hands=(
            num_analyzed_simulated_hands
        ),
        num_failed_runs=num_failed_runs,
        mean_difference_bb100=mean_bb100,
        std_difference_bb100=std_bb100,
        normal_ci_low_bb100=(
            normal_ci_low_bb100
        ),
        normal_ci_high_bb100=(
            normal_ci_high_bb100
        ),
        bootstrap_ci_low_bb100=(
            bootstrap_ci_low_bb100
        ),
        bootstrap_ci_high_bb100=(
            bootstrap_ci_high_bb100
        ),
        effect_size_dz=effect_size_dz,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def _validate_evaluation_inputs(
    *,
    clusters: Sequence[PairedCluster],
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence_level: float,
    num_failed_runs: int,
) -> None:
    """
    Validate statistical evaluation inputs.
    """
    if len(clusters) < 2:
        raise ValueError(
            "At least two independent statistical clusters "
            "are required to estimate SD and a confidence interval"
        )

    if not isinstance(
        bootstrap_seed,
        int,
    ):
        raise TypeError(
            "bootstrap_seed must be an integer"
        )

    if not isinstance(
        bootstrap_resamples,
        int,
    ):
        raise TypeError(
            "bootstrap_resamples must be an integer"
        )

    if bootstrap_resamples < 2:
        raise ValueError(
            "bootstrap_resamples must be at least 2"
        )

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level must satisfy "
            "0 < confidence_level < 1"
        )

    if num_failed_runs < 0:
        raise ValueError(
            "num_failed_runs must be non-negative"
        )


def _normal_confidence_interval(
    *,
    mean: float,
    sample_std: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Return the normal-approximation confidence interval.

    The normal CI is retained as the framework's analytical interval.
    Bootstrap percentiles are reported separately as the robust
    sensitivity analysis required by the specification.
    """
    if sample_size < 2:
        raise ValueError(
            "sample_size must be at least 2"
        )

    alpha = (
        1.0 - confidence_level
    )

    critical_value = NormalDist().inv_cdf(
        1.0 - alpha / 2.0
    )

    standard_error = (
        sample_std
        / math.sqrt(sample_size)
    )

    half_width = (
        critical_value
        * standard_error
    )

    return (
        mean - half_width,
        mean + half_width,
    )


def _bootstrap_confidence_interval(
    *,
    values_bb: tuple[float, ...],
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Return the percentile bootstrap CI of the cluster-level mean.

    Every resample draws complete independent clusters with replacement.

    The standard experiment uses 10,000 resamples.

    A local RNG is used so bootstrap randomness cannot affect game,
    agent, or deal RNG state.
    """
    rng = random.Random(
        bootstrap_seed
    )

    sample_size = len(values_bb)

    bootstrap_means_bb100: list[float] = []

    for _ in range(
        bootstrap_resamples
    ):
        sampled_clusters = rng.choices(
            values_bb,
            k=sample_size,
        )

        bootstrap_mean_bb = (
            statistics.fmean(
                sampled_clusters
            )
        )

        bootstrap_means_bb100.append(
            BB100_SCALE
            * bootstrap_mean_bb
        )

    bootstrap_means_bb100.sort()

    alpha = (
        1.0 - confidence_level
    )

    lower_probability = (
        alpha / 2.0
    )
    upper_probability = (
        1.0 - alpha / 2.0
    )

    lower = _linear_percentile(
        sorted_values=bootstrap_means_bb100,
        probability=lower_probability,
    )

    upper = _linear_percentile(
        sorted_values=bootstrap_means_bb100,
        probability=upper_probability,
    )

    return (
        lower,
        upper,
    )


def _linear_percentile(
    *,
    sorted_values: Sequence[float],
    probability: float,
) -> float:
    """
    Return a percentile using the versioned linear-n-minus-1 convention.

    For n ordered observations, the fractional index is:

        h = (n - 1) * p

    Values between adjacent indices are linearly interpolated.

    This convention is explicitly versioned because the project
    specification fixes the 2.5% / 97.5% percentile targets but does
    not prescribe a finite-sample interpolation rule.
    """
    if not sorted_values:
        raise ValueError(
            "sorted_values must not be empty"
        )

    if not (
        0.0
        <= probability
        <= 1.0
    ):
        raise ValueError(
            "probability must satisfy "
            "0 <= probability <= 1"
        )

    if len(sorted_values) == 1:
        return float(
            sorted_values[0]
        )

    position = (
        len(sorted_values) - 1
    ) * probability

    lower_index = math.floor(
        position
    )
    upper_index = math.ceil(
        position
    )

    lower_value = float(
        sorted_values[lower_index]
    )

    upper_value = float(
        sorted_values[upper_index]
    )

    if lower_index == upper_index:
        return lower_value

    weight = (
        position - lower_index
    )

    return (
        lower_value
        + weight
        * (
            upper_value
            - lower_value
        )
    )