"""
Validation gate for pre-experiment power and precision planning.

The important invariant is that planning returns counts of independent
statistical units and fixes N before the final experiment.
"""

from __future__ import annotations

from pokerbench.evaluation.planning import (
    COLD_START_SIGMA_D_BB100,
    plan_cold_start_paired_power,
    plan_mean_precision,
    recommend_fixed_sample_size,
)


def test_cold_start_power_assumption_is_conservative_100_bb100(
) -> None:
    plan = (
        plan_cold_start_paired_power(
            delta_bb100=5.0
        )
    )

    assert (
        plan.sigma_d_bb100
        == COLD_START_SIGMA_D_BB100
        == 100.0
    )

    assert (
        plan.required_independent_units
        == 3140
    )


def test_example_95_percent_plus_minus_one_bb100_precision_is_preplanned(
) -> None:
    plan = plan_mean_precision(
        sigma_bb100=100.0,
        target_half_width_bb100=1.0,
        confidence_level=0.95,
        estimand=(
            "mean_paired_difference_bb100"
        ),
        variance_source=(
            "pilot-paired-sd-v1"
        ),
    )

    assert (
        plan.required_independent_units
        == 38415
    )


def test_final_fixed_n_satisfies_power_and_precision(
) -> None:
    power = (
        plan_cold_start_paired_power(
            delta_bb100=5.0
        )
    )

    precision = (
        plan_mean_precision(
            sigma_bb100=100.0,
            target_half_width_bb100=1.0,
            confidence_level=0.95,
            estimand=(
                "mean_paired_difference_bb100"
            ),
            variance_source=(
                "cold-start-conservative"
            ),
        )
    )

    final_plan = (
        recommend_fixed_sample_size(
            power_plan=power,
            precision_plan=precision,
        )
    )

    assert (
        final_plan.required_independent_units
        >= power.required_independent_units
    )

    assert (
        final_plan.required_independent_units
        >= precision.required_independent_units
    )

    assert (
        final_plan.required_independent_units
        == 38415
    )


def test_precision_plan_records_required_scientific_metadata(
) -> None:
    plan = plan_mean_precision(
        sigma_bb100=75.0,
        target_half_width_bb100=1.0,
        confidence_level=0.95,
        estimand=(
            "mean_paired_difference_bb100"
        ),
        variance_source=(
            "pilot-seed-set-v1"
        ),
    )

    assert (
        plan.estimand
        == "mean_paired_difference_bb100"
    )

    assert (
        plan.target_half_width_bb100
        == 1.0
    )

    assert (
        plan.confidence_level
        == 0.95
    )

    assert (
        plan.variance_source
        == "pilot-seed-set-v1"
    )