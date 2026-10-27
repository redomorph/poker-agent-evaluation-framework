"""Unit tests for pre-experiment sample-size planning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from statistics import NormalDist

import pytest

from pokerbench.evaluation.planning import (
    COLD_START_SIGMA_D_BB100,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_CONFIDENCE_LEVEL,
    PLANNING_VERSION,
    plan_cold_start_paired_power,
    plan_mean_precision,
    plan_paired_power,
    recommend_fixed_sample_size,
)


def test_default_power_parameters_match_specification(
) -> None:
    assert DEFAULT_ALPHA == 0.05
    assert DEFAULT_BETA == 0.20


def test_default_precision_confidence_is_95_percent(
) -> None:
    assert (
        DEFAULT_CONFIDENCE_LEVEL
        == 0.95
    )


def test_cold_start_sigma_matches_specification(
) -> None:
    assert (
        COLD_START_SIGMA_D_BB100
        == 100.0
    )


def test_power_plan_matches_paired_formula(
) -> None:
    plan = plan_paired_power(
        delta_bb100=5.0,
        sigma_d_bb100=100.0,
        alpha=0.05,
        beta=0.20,
        variance_source="pilot-v1",
    )

    normal = NormalDist()

    z_alpha = normal.inv_cdf(
        0.975
    )

    z_power = normal.inv_cdf(
        0.80
    )

    expected_raw = (
        (
            z_alpha
            + z_power
        )
        ** 2
        * 100.0
        ** 2
        / 5.0
        ** 2
    )

    assert (
        plan.raw_required_independent_units
        == pytest.approx(
            expected_raw
        )
    )

    assert (
        plan.required_independent_units
        == 3140
    )


def test_power_plan_rounds_up_to_whole_independent_unit(
) -> None:
    plan = plan_paired_power(
        delta_bb100=5.0,
        sigma_d_bb100=100.0,
        variance_source="pilot-v1",
    )

    assert (
        plan.required_independent_units
        >= plan.raw_required_independent_units
    )

    assert (
        plan.required_independent_units
        - plan.raw_required_independent_units
        < 1.0
    )


def test_smaller_relevant_effect_requires_more_units(
) -> None:
    large_effect = plan_paired_power(
        delta_bb100=5.0,
        sigma_d_bb100=100.0,
        variance_source="pilot-v1",
    )

    small_effect = plan_paired_power(
        delta_bb100=2.0,
        sigma_d_bb100=100.0,
        variance_source="pilot-v1",
    )

    assert (
        small_effect.required_independent_units
        > large_effect.required_independent_units
    )


def test_cold_start_power_plan_is_explicitly_marked_provisional(
) -> None:
    plan = (
        plan_cold_start_paired_power(
            delta_bb100=5.0
        )
    )

    assert (
        plan.sigma_d_bb100
        == 100.0
    )

    assert (
        plan.variance_source
        == "cold-start-conservative"
    )


def test_precision_plan_matches_mean_ci_formula(
) -> None:
    plan = plan_mean_precision(
        sigma_bb100=100.0,
        target_half_width_bb100=1.0,
        confidence_level=0.95,
        estimand="mean_paired_difference_bb100",
        variance_source="pilot-v1",
    )

    z = NormalDist().inv_cdf(
        0.975
    )

    expected_raw = (
        z
        * 100.0
        / 1.0
    ) ** 2

    assert (
        plan.raw_required_independent_units
        == pytest.approx(
            expected_raw
        )
    )

    assert (
        plan.required_independent_units
        == 38415
    )


def test_tighter_precision_requires_more_units(
) -> None:
    wide = plan_mean_precision(
        sigma_bb100=50.0,
        target_half_width_bb100=2.0,
        estimand="mean_paired_difference_bb100",
        variance_source="pilot-v1",
    )

    narrow = plan_mean_precision(
        sigma_bb100=50.0,
        target_half_width_bb100=1.0,
        estimand="mean_paired_difference_bb100",
        variance_source="pilot-v1",
    )

    assert (
        narrow.required_independent_units
        > wide.required_independent_units
    )


def test_higher_confidence_requires_more_units(
) -> None:
    plan_95 = plan_mean_precision(
        sigma_bb100=50.0,
        target_half_width_bb100=1.0,
        confidence_level=0.95,
        estimand="mean_paired_difference_bb100",
        variance_source="pilot-v1",
    )

    plan_99 = plan_mean_precision(
        sigma_bb100=50.0,
        target_half_width_bb100=1.0,
        confidence_level=0.99,
        estimand="mean_paired_difference_bb100",
        variance_source="pilot-v1",
    )

    assert (
        plan_99.required_independent_units
        > plan_95.required_independent_units
    )


def test_combined_plan_uses_stricter_requirement(
) -> None:
    power = plan_paired_power(
        delta_bb100=5.0,
        sigma_d_bb100=100.0,
        variance_source="pilot-v1",
    )

    precision = plan_mean_precision(
        sigma_bb100=100.0,
        target_half_width_bb100=1.0,
        estimand="mean_paired_difference_bb100",
        variance_source="pilot-v1",
    )

    result = recommend_fixed_sample_size(
        power_plan=power,
        precision_plan=precision,
    )

    assert (
        result.required_independent_units
        == precision.required_independent_units
    )

    assert (
        result.required_independent_units
        == 38415
    )


def test_combined_plan_requires_at_least_one_constraint(
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        recommend_fixed_sample_size()


@pytest.mark.parametrize(
    (
        "delta",
        "sigma",
    ),
    [
        (0.0, 100.0),
        (-1.0, 100.0),
        (1.0, 0.0),
        (1.0, -100.0),
    ],
)
def test_power_rejects_non_positive_inputs(
    delta: float,
    sigma: float,
) -> None:
    with pytest.raises(
        ValueError,
    ):
        plan_paired_power(
            delta_bb100=delta,
            sigma_d_bb100=sigma,
            variance_source="pilot-v1",
        )


@pytest.mark.parametrize(
    "alpha",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_power_rejects_invalid_alpha(
    alpha: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="alpha",
    ):
        plan_paired_power(
            delta_bb100=5.0,
            sigma_d_bb100=100.0,
            alpha=alpha,
            variance_source="pilot-v1",
        )


@pytest.mark.parametrize(
    "beta",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_power_rejects_invalid_beta(
    beta: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="beta",
    ):
        plan_paired_power(
            delta_bb100=5.0,
            sigma_d_bb100=100.0,
            beta=beta,
            variance_source="pilot-v1",
        )


def test_precision_rejects_zero_half_width(
) -> None:
    with pytest.raises(
        ValueError,
        match="target_half_width_bb100",
    ):
        plan_mean_precision(
            sigma_bb100=100.0,
            target_half_width_bb100=0.0,
            estimand="mean",
            variance_source="pilot-v1",
        )


def test_precision_rejects_empty_estimand(
) -> None:
    with pytest.raises(
        ValueError,
        match="estimand",
    ):
        plan_mean_precision(
            sigma_bb100=100.0,
            target_half_width_bb100=1.0,
            estimand="",
            variance_source="pilot-v1",
        )


def test_plans_are_frozen(
) -> None:
    plan = plan_paired_power(
        delta_bb100=5.0,
        sigma_d_bb100=100.0,
        variance_source="pilot-v1",
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        plan.required_independent_units = 1  # type: ignore[misc]


def test_planning_version_is_explicit(
) -> None:
    plan = plan_cold_start_paired_power(
        delta_bb100=5.0
    )

    assert (
        plan.planning_version
        == PLANNING_VERSION
    )