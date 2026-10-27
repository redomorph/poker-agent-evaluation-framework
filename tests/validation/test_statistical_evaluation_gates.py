# tests/validation/test_statistical_evaluation_gates.py
"""
Phase-0 statistical-evaluation validation gates.

These tests validate:

1. BB/100 reporting.
2. Mean / SD / normal CI reporting.
3. Effect size d_z.
4. The mandatory 10,000-resample paired bootstrap.
5. Deterministic bootstrap reproducibility.
6. Cluster-aware treatment of adaptive trajectories.
7. Protection against hand-level pseudo-replication.
"""

from __future__ import annotations

import pytest

from pokerbench.evaluation import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    evaluate_paired_results,
)
from pokerbench.runner import (
    PairedResult,
    StatisticalUnit,
)


BOOTSTRAP_SEED = 20260810


def _result(
    *,
    difference: float,
    statistical_unit: StatisticalUnit,
    num_hands: int,
    seed: int,
) -> PairedResult:
    return PairedResult(
        paired_differences=[
            difference
        ],
        statistical_unit=statistical_unit,
        num_hands=num_hands,
        num_valid_pairs=1,
        failed_runs=[],
        experiment_metadata={},
        seed=seed,
    )


def test_bootstrap_standard_is_10000_resamples(
) -> None:
    """
    Section 4.5 defines 10,000 bootstrap resamples as the standard.
    """
    assert (
        DEFAULT_BOOTSTRAP_RESAMPLES
        == 10_000
    )


def test_static_statistical_evaluation_reports_bb100_and_bootstrap_ci(
) -> None:
    """
    Static paired deals are resampled at paired-deal level.
    """
    result = PairedResult(
        paired_differences=[
            0.01,
            0.01,
            0.01,
            0.01,
        ],
        statistical_unit=(
            StatisticalUnit.PAIRED_DEAL
        ),
        num_hands=4,
        num_valid_pairs=4,
        failed_runs=[],
        experiment_metadata={},
        seed=42,
    )

    evaluation = evaluate_paired_results(
        [result],
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    assert (
        evaluation.bootstrap_resamples
        == 10_000
    )

    assert (
        evaluation.statistical_unit
        == StatisticalUnit.PAIRED_DEAL
    )

    assert (
        evaluation.num_independent_units
        == 4
    )

    # Four paired deals represent eight actual engine hand simulations.
    assert (
        evaluation.num_analyzed_simulated_hands
        == 8
    )

    # 0.01 BB/hand == 1 BB/100.
    assert (
        evaluation.mean_difference_bb100
        == pytest.approx(1.0)
    )

    assert (
        evaluation.std_difference_bb100
        == pytest.approx(0.0)
    )

    assert (
        evaluation.normal_ci_low_bb100
        == pytest.approx(1.0)
    )

    assert (
        evaluation.normal_ci_high_bb100
        == pytest.approx(1.0)
    )

    assert (
        evaluation.bootstrap_ci_low_bb100
        == pytest.approx(1.0)
    )

    assert (
        evaluation.bootstrap_ci_high_bb100
        == pytest.approx(1.0)
    )


def test_adaptive_bootstrap_uses_complete_trajectory_clusters(
) -> None:
    """
    Adaptive/stateful inference uses complete trajectories.

    Each result below represents one complete independent paired
    trajectory with 100 hands in each positional condition.

    The three total paired differences:

        1 BB / 100 hands
        2 BB / 100 hands
        3 BB / 100 hands

    correspond to cluster-level rates:

        1 BB/100
        2 BB/100
        3 BB/100

    The independent sample size must therefore be 3, never 300.
    """
    results = [
        _result(
            difference=1.0,
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=101,
        ),
        _result(
            difference=2.0,
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=102,
        ),
        _result(
            difference=3.0,
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=103,
        ),
    ]

    evaluation = evaluate_paired_results(
        results,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    assert (
        evaluation.statistical_unit
        == StatisticalUnit.TRAJECTORY_CLUSTER
    )

    # Critical anti-pseudoreplication assertion:
    assert (
        evaluation.num_independent_units
        == 3
    )

    # Three paired 100-hand trajectories:
    #
    # 3 clusters
    # * 100 hands
    # * 2 positional conditions
    # = 600 actual engine simulations.
    assert (
        evaluation.num_analyzed_simulated_hands
        == 600
    )

    assert (
        evaluation.mean_difference_bb100
        == pytest.approx(2.0)
    )

    assert (
        evaluation.std_difference_bb100
        == pytest.approx(1.0)
    )

    assert (
        evaluation.effect_size_dz
        == pytest.approx(2.0)
    )

    assert (
        evaluation.bootstrap_resamples
        == 10_000
    )


def test_bootstrap_is_reproducible_with_fixed_seed(
) -> None:
    """
    Evaluation randomness must itself be reproducible.
    """
    results = [
        _result(
            difference=-1.0,
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=201,
        ),
        _result(
            difference=0.5,
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=202,
        ),
        _result(
            difference=2.0,
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=203,
        ),
        _result(
            difference=4.0,
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=204,
        ),
    ]

    first = evaluate_paired_results(
        results,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    second = evaluate_paired_results(
        results,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    assert (
        first.bootstrap_ci_low_bb100
        == second.bootstrap_ci_low_bb100
    )

    assert (
        first.bootstrap_ci_high_bb100
        == second.bootstrap_ci_high_bb100
    )


def test_adaptive_hand_level_pseudoreplication_is_rejected(
) -> None:
    """
    One adaptive trajectory may not expose multiple within-run hand
    observations as independent bootstrap clusters.
    """
    invalid = PairedResult(
        paired_differences=[
            0.1,
            0.2,
            0.3,
        ],
        statistical_unit=(
            StatisticalUnit.TRAJECTORY_CLUSTER
        ),
        num_hands=100,
        num_valid_pairs=3,
        failed_runs=[],
        experiment_metadata={},
        seed=999,
    )

    with pytest.raises(
        ValueError,
        match="Dependent hands",
    ):
        evaluate_paired_results(
            [invalid],
            bootstrap_seed=BOOTSTRAP_SEED,
        )