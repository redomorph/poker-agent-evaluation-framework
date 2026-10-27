from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pokerbench.evaluation.population import (
    BALANCED_STATIC_POPULATION,
    BASELINE_SYNTHETIC_STATIC_POPULATION,
    POPULATION_EVALUATION_VERSION,
    ProfilePerformanceEvidence,
    evaluate_population,
    profile_performance_from_paired_results,
    save_population_evaluation_report,
)
from pokerbench.runner import PairedResult, StatisticalUnit


def _evidence(
    profile_id: str,
    *run_ev_bb100: float,
) -> ProfilePerformanceEvidence:
    return ProfilePerformanceEvidence(
        profile_id=profile_id,
        run_ev_bb100=tuple(run_ev_bb100),
        simulated_hands_per_run=tuple(2_000 for _ in run_ev_bb100),
    )


def _five_profile_evidence() -> tuple[ProfilePerformanceEvidence, ...]:
    return (
        _evidence("nit", -20.0),
        _evidence("tag", 0.0),
        _evidence("lag", 10.0),
        _evidence("calling_station", 30.0),
        _evidence("maniac", 50.0),
    )


def test_static_population_weights_are_exactly_frozen() -> None:
    assert BALANCED_STATIC_POPULATION.weights_ppm == (
        ("nit", 200_000),
        ("tag", 200_000),
        ("lag", 200_000),
        ("calling_station", 200_000),
        ("maniac", 200_000),
    )
    assert BASELINE_SYNTHETIC_STATIC_POPULATION.weights_ppm == (
        ("nit", 100_000),
        ("tag", 350_000),
        ("lag", 300_000),
        ("calling_station", 150_000),
        ("maniac", 100_000),
    )
    assert sum(weight for _, weight in BALANCED_STATIC_POPULATION.weights_ppm) == 1_000_000
    assert sum(
        weight
        for _, weight in BASELINE_SYNTHETIC_STATIC_POPULATION.weights_ppm
    ) == 1_000_000
    assert len(BALANCED_STATIC_POPULATION.population_hash) == 64


def test_balanced_population_point_metrics_follow_weighted_inverse_cdf() -> None:
    result = evaluate_population(
        BALANCED_STATIC_POPULATION,
        _five_profile_evidence(),
        bootstrap_seed=5,
        bootstrap_resamples=200,
    )

    assert result.evaluation_version == POPULATION_EVALUATION_VERSION
    assert result.mean_ev_bb100 == pytest.approx(14.0)
    assert result.median_ev_bb100 == pytest.approx(10.0)
    assert result.r10_bb100 == pytest.approx(-20.0)
    assert result.q25_bb100 == pytest.approx(0.0)
    assert result.empirical_min_bb100 == pytest.approx(-20.0)
    expected_std = math.sqrt(
        sum((value - 14.0) ** 2 for value in (-20.0, 0.0, 10.0, 30.0, 50.0))
        / 5.0
    )
    assert result.stddev_bb100 == pytest.approx(expected_std)
    assert result.profile_count == 5
    assert result.run_count == 5
    assert result.total_simulated_hands == 10_000
    assert result.tail_precision_warning


def test_baseline_synthetic_population_uses_specified_profile_weights() -> None:
    result = evaluate_population(
        BASELINE_SYNTHETIC_STATIC_POPULATION,
        _five_profile_evidence(),
        bootstrap_seed=7,
        bootstrap_resamples=200,
    )

    assert result.mean_ev_bb100 == pytest.approx(10.5)
    assert result.median_ev_bb100 == pytest.approx(10.0)
    assert result.r10_bb100 == pytest.approx(-20.0)
    assert result.q25_bb100 == pytest.approx(0.0)


def test_nested_population_bootstrap_is_deterministic() -> None:
    evidence = (
        _evidence("nit", -25.0, -15.0),
        _evidence("tag", -2.0, 2.0),
        _evidence("lag", 8.0, 12.0),
        _evidence("calling_station", 25.0, 35.0),
        _evidence("maniac", 45.0, 55.0),
    )

    first = evaluate_population(
        BALANCED_STATIC_POPULATION,
        evidence,
        bootstrap_seed=991,
        bootstrap_resamples=500,
    )
    second = evaluate_population(
        BALANCED_STATIC_POPULATION,
        tuple(reversed(evidence)),
        bootstrap_seed=991,
        bootstrap_resamples=500,
    )

    assert first == second
    assert first.report_hash == second.report_hash
    assert first.mean_ci_low_bb100 <= first.mean_ev_bb100 <= first.mean_ci_high_bb100
    assert first.r10_ci_low_bb100 <= first.r10_ci_high_bb100


def test_population_evaluator_rejects_missing_or_duplicate_profile_evidence() -> None:
    evidence = _five_profile_evidence()

    with pytest.raises(ValueError, match="exactly match"):
        evaluate_population(
            BALANCED_STATIC_POPULATION,
            evidence[:-1],
            bootstrap_seed=0,
            bootstrap_resamples=20,
        )

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_population(
            BALANCED_STATIC_POPULATION,
            evidence[:-1] + (evidence[0],),
            bootstrap_seed=0,
            bootstrap_resamples=20,
        )


def test_complete_paired_results_convert_pairwise_difference_to_hero_ev() -> None:
    result = PairedResult(
        paired_differences=[2.0, 4.0],
        statistical_unit=StatisticalUnit.PAIRED_DEAL,
        num_hands=2,
        num_valid_pairs=2,
        failed_runs=[],
        experiment_metadata={},
        seed=11,
    )

    evidence = profile_performance_from_paired_results(
        "tag",
        (result,),
    )

    assert evidence.run_ev_bb100 == (150.0,)
    assert evidence.simulated_hands_per_run == (4,)


def test_incomplete_paired_result_is_rejected_for_population_evidence() -> None:
    incomplete = PairedResult(
        paired_differences=[2.0],
        statistical_unit=StatisticalUnit.PAIRED_DEAL,
        num_hands=2,
        num_valid_pairs=1,
        failed_runs=[{"hand": 1}],
        experiment_metadata={},
        seed=11,
    )

    with pytest.raises(ValueError, match="complete"):
        profile_performance_from_paired_results(
            "tag",
            (incomplete,),
        )


def test_population_report_is_canonical_hashed_and_atomic(tmp_path: Path) -> None:
    result = evaluate_population(
        BALANCED_STATIC_POPULATION,
        _five_profile_evidence(),
        bootstrap_seed=8,
        bootstrap_resamples=100,
    )
    destination = tmp_path / "population.json"

    save_population_evaluation_report(result, destination)
    raw = destination.read_bytes()
    envelope = json.loads(raw)

    assert raw.endswith(b"\n")
    assert envelope["report_hash"] == result.report_hash
    assert envelope["report"]["evidence_hash"] == result.evidence_hash
    assert not (tmp_path / "population.json.tmp").exists()
