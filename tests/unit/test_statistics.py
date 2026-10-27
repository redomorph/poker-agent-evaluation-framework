# tests/unit/test_statistics.py
"""Unit tests for paired statistical evaluation."""

from __future__ import annotations

import pytest

from pokerbench.evaluation import (
    BOOTSTRAP_PERCENTILE_METHOD,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    EVALUATION_VERSION,
    PairedCluster,
    clusters_from_paired_results,
    evaluate_clusters,
    evaluate_paired_results,
)
from pokerbench.runner import (
    PairedResult,
    StatisticalUnit,
)


def _paired_result(
    *,
    differences: list[float],
    statistical_unit: StatisticalUnit,
    num_hands: int,
    seed: int = 42,
    failed_runs: list[dict[str, object]] | None = None,
) -> PairedResult:
    return PairedResult(
        paired_differences=differences,
        statistical_unit=statistical_unit,
        num_hands=num_hands,
        num_valid_pairs=len(
            differences
        ),
        failed_runs=(
            failed_runs
            if failed_runs is not None
            else []
        ),
        experiment_metadata={},
        seed=seed,
    )


def test_evaluate_clusters_reports_bb100_statistics(
) -> None:
    clusters = (
        PairedCluster(
            mean_difference_bb_per_hand=0.01,
            num_simulated_hands=2,
        ),
        PairedCluster(
            mean_difference_bb_per_hand=0.02,
            num_simulated_hands=2,
        ),
        PairedCluster(
            mean_difference_bb_per_hand=0.03,
            num_simulated_hands=2,
        ),
    )

    result = evaluate_clusters(
        clusters=clusters,
        statistical_unit=(
            StatisticalUnit.PAIRED_DEAL
        ),
        bootstrap_seed=123,
        bootstrap_resamples=500,
    )

    assert (
        result.evaluation_version
        == EVALUATION_VERSION
    )
    assert (
        result.percentile_method
        == BOOTSTRAP_PERCENTILE_METHOD
    )

    assert (
        result.statistical_unit
        == StatisticalUnit.PAIRED_DEAL
    )

    assert (
        result.num_independent_units
        == 3
    )

    assert (
        result.num_analyzed_simulated_hands
        == 6
    )

    # Raw differences:
    #
    # 0.01, 0.02, 0.03 BB/hand
    #
    # Mean = 0.02 BB/hand = 2 BB/100.
    # Sample SD = 0.01 BB/hand = 1 BB/100.
    assert (
        result.mean_difference_bb100
        == pytest.approx(2.0)
    )

    assert (
        result.std_difference_bb100
        == pytest.approx(1.0)
    )

    # d_z = mean / sample SD = 2 / 1.
    assert (
        result.effect_size_dz
        == pytest.approx(2.0)
    )

    # 95% normal approximation:
    #
    # 2 ± z_0.975 * 1 / sqrt(3)
    assert (
        result.normal_ci_low_bb100
        == pytest.approx(
            0.8684142659238285
        )
    )

    assert (
        result.normal_ci_high_bb100
        == pytest.approx(
            3.1315857340761717
        )
    )


def test_constant_clusters_have_exact_zero_variance_result(
) -> None:
    clusters = tuple(
        PairedCluster(
            mean_difference_bb_per_hand=0.01,
            num_simulated_hands=2,
        )
        for _ in range(4)
    )

    result = evaluate_clusters(
        clusters=clusters,
        statistical_unit=(
            StatisticalUnit.PAIRED_DEAL
        ),
        bootstrap_seed=999,
        bootstrap_resamples=500,
    )

    assert (
        result.mean_difference_bb100
        == pytest.approx(1.0)
    )

    assert (
        result.std_difference_bb100
        == pytest.approx(0.0)
    )

    assert (
        result.normal_ci_low_bb100
        == pytest.approx(1.0)
    )

    assert (
        result.normal_ci_high_bb100
        == pytest.approx(1.0)
    )

    assert (
        result.bootstrap_ci_low_bb100
        == pytest.approx(1.0)
    )

    assert (
        result.bootstrap_ci_high_bb100
        == pytest.approx(1.0)
    )

    # d_z is undefined for SD == 0.
    assert result.effect_size_dz is None


def test_bootstrap_is_bit_reproducible_for_same_seed(
) -> None:
    clusters = (
        PairedCluster(
            mean_difference_bb_per_hand=-0.02,
            num_simulated_hands=2,
        ),
        PairedCluster(
            mean_difference_bb_per_hand=0.00,
            num_simulated_hands=2,
        ),
        PairedCluster(
            mean_difference_bb_per_hand=0.01,
            num_simulated_hands=2,
        ),
        PairedCluster(
            mean_difference_bb_per_hand=0.04,
            num_simulated_hands=2,
        ),
    )

    first = evaluate_clusters(
        clusters=clusters,
        statistical_unit=(
            StatisticalUnit.PAIRED_DEAL
        ),
        bootstrap_seed=12345,
        bootstrap_resamples=1_000,
    )

    second = evaluate_clusters(
        clusters=clusters,
        statistical_unit=(
            StatisticalUnit.PAIRED_DEAL
        ),
        bootstrap_seed=12345,
        bootstrap_resamples=1_000,
    )

    assert (
        first.bootstrap_ci_low_bb100
        == second.bootstrap_ci_low_bb100
    )

    assert (
        first.bootstrap_ci_high_bb100
        == second.bootstrap_ci_high_bb100
    )


def test_default_bootstrap_resample_count_is_10000(
) -> None:
    assert (
        DEFAULT_BOOTSTRAP_RESAMPLES
        == 10_000
    )


def test_static_paired_results_become_one_cluster_per_pair(
) -> None:
    source = _paired_result(
        differences=[
            0.01,
            -0.02,
            0.03,
        ],
        statistical_unit=(
            StatisticalUnit.PAIRED_DEAL
        ),
        num_hands=3,
    )

    dataset = clusters_from_paired_results(
        [source]
    )

    assert (
        dataset.statistical_unit
        == StatisticalUnit.PAIRED_DEAL
    )

    assert len(dataset.clusters) == 3

    assert tuple(
        cluster.mean_difference_bb_per_hand
        for cluster in dataset.clusters
    ) == pytest.approx(
        (
            0.01,
            -0.02,
            0.03,
        )
    )

    # Every paired deal executes two positional hand conditions.
    assert (
        sum(
            cluster.num_simulated_hands
            for cluster in dataset.clusters
        )
        == 6
    )


def test_adaptive_results_become_complete_trajectory_clusters(
) -> None:
    first = _paired_result(
        differences=[1.0],
        statistical_unit=(
            StatisticalUnit.TRAJECTORY_CLUSTER
        ),
        num_hands=100,
        seed=1,
    )

    second = _paired_result(
        differences=[2.0],
        statistical_unit=(
            StatisticalUnit.TRAJECTORY_CLUSTER
        ),
        num_hands=100,
        seed=2,
    )

    third = _paired_result(
        differences=[3.0],
        statistical_unit=(
            StatisticalUnit.TRAJECTORY_CLUSTER
        ),
        num_hands=100,
        seed=3,
    )

    dataset = clusters_from_paired_results(
        [
            first,
            second,
            third,
        ]
    )

    assert (
        dataset.statistical_unit
        == StatisticalUnit.TRAJECTORY_CLUSTER
    )

    assert len(dataset.clusters) == 3

    # Runner totals are divided by the complete 100-hand trajectory
    # length only after trajectory construction.
    assert tuple(
        cluster.mean_difference_bb_per_hand
        for cluster in dataset.clusters
    ) == pytest.approx(
        (
            0.01,
            0.02,
            0.03,
        )
    )

    # Each paired trajectory consists of:
    #
    # 100 hands in position condition A
    # 100 hands in position condition B
    #
    # Three independent clusters therefore represent 600 actual
    # engine hand simulations.
    assert (
        sum(
            cluster.num_simulated_hands
            for cluster in dataset.clusters
        )
        == 600
    )


def test_adaptive_evaluation_uses_trajectory_as_independent_unit(
) -> None:
    results = [
        _paired_result(
            differences=[1.0],
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=1,
        ),
        _paired_result(
            differences=[2.0],
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=2,
        ),
        _paired_result(
            differences=[3.0],
            statistical_unit=(
                StatisticalUnit.TRAJECTORY_CLUSTER
            ),
            num_hands=100,
            seed=3,
        ),
    ]

    evaluation = evaluate_paired_results(
        results,
        bootstrap_seed=999,
        bootstrap_resamples=500,
    )

    assert (
        evaluation.statistical_unit
        == StatisticalUnit.TRAJECTORY_CLUSTER
    )

    assert (
        evaluation.num_independent_units
        == 3
    )

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


def test_adaptive_result_rejects_multiple_observations_inside_one_trajectory(
) -> None:
    invalid = _paired_result(
        differences=[
            0.1,
            0.2,
        ],
        statistical_unit=(
            StatisticalUnit.TRAJECTORY_CLUSTER
        ),
        num_hands=100,
    )

    with pytest.raises(
        ValueError,
        match="Dependent hands",
    ):
        clusters_from_paired_results(
            [invalid]
        )


def test_mixed_statistical_units_are_rejected(
) -> None:
    static_result = _paired_result(
        differences=[0.1],
        statistical_unit=(
            StatisticalUnit.PAIRED_DEAL
        ),
        num_hands=1,
    )

    adaptive_result = _paired_result(
        differences=[1.0],
        statistical_unit=(
            StatisticalUnit.TRAJECTORY_CLUSTER
        ),
        num_hands=100,
    )

    with pytest.raises(
        ValueError,
        match="same statistical unit",
    ):
        clusters_from_paired_results(
            [
                static_result,
                adaptive_result,
            ]
        )


def test_evaluation_requires_at_least_two_independent_clusters(
) -> None:
    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        evaluate_clusters(
            clusters=(
                PairedCluster(
                    mean_difference_bb_per_hand=0.01,
                    num_simulated_hands=2,
                ),
            ),
            statistical_unit=(
                StatisticalUnit.PAIRED_DEAL
            ),
            bootstrap_seed=1,
        )


def test_failed_runs_are_reported_separately(
) -> None:
    source = _paired_result(
        differences=[
            0.01,
            0.02,
        ],
        statistical_unit=(
            StatisticalUnit.PAIRED_DEAL
        ),
        num_hands=3,
        failed_runs=[
            {
                "hand": 2,
                "config": "A_BTN",
                "details": "failed",
            }
        ],
    )

    evaluation = evaluate_paired_results(
        [source],
        bootstrap_seed=42,
        bootstrap_resamples=500,
    )

    assert (
        evaluation.num_independent_units
        == 2
    )

    assert (
        evaluation.num_failed_runs
        == 1
    )


def test_paired_cluster_rejects_non_finite_values(
) -> None:
    with pytest.raises(ValueError):
        PairedCluster(
            mean_difference_bb_per_hand=(
                float("nan")
            ),
            num_simulated_hands=2,
        )

    with pytest.raises(ValueError):
        PairedCluster(
            mean_difference_bb_per_hand=(
                float("inf")
            ),
            num_simulated_hands=2,
        )


def test_paired_cluster_requires_positive_hand_count(
) -> None:
    with pytest.raises(ValueError):
        PairedCluster(
            mean_difference_bb_per_hand=0.0,
            num_simulated_hands=0,
        )