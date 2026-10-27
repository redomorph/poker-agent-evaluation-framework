from __future__ import annotations

import math
import statistics

import pytest

from pokerbench.experiments.h4_holdout_variance_analysis import (
    H4HoldoutVarianceAnalysisError, matched_unit_contrast_bb100, population_contrast_for_seed,
    sigma_500_from_population_contrasts, plan_final_h4_sample_size, variance_pilot_plan_from_complete_seed_contrasts,
)

def _units(value: float) -> dict[str, float]:
    return {f"h4_holdout_{i:03d}": value + i for i in range(5)}

def test_unit_contrast_uses_two_500_hand_positions() -> None:
    assert matched_unit_contrast_bb100(control_btn_gain_bb=10, control_bb_gain_bb=-20, treatment_btn_gain_bb=30, treatment_bb_gain_bb=-10) == pytest.approx(3.0)

def test_population_contrast_weights_inside_seed_first() -> None:
    contrasts = _units(10.0)
    assert population_contrast_for_seed(contrasts) == pytest.approx(sum(contrasts.values()) / 5)

def test_population_contrast_requires_all_five_units_and_no_others() -> None:
    contrasts = _units(10.0); contrasts.pop("h4_holdout_004")
    with pytest.raises(H4HoldoutVarianceAnalysisError): population_contrast_for_seed(contrasts)
    contrasts = _units(10.0); contrasts["extra"] = 0.0
    with pytest.raises(H4HoldoutVarianceAnalysisError): population_contrast_for_seed(contrasts)

def test_sigma_500_is_sample_sd_ddof1_over_exact_ten_frozen_seeds() -> None:
    values = {6_400_000+i: float(i) for i in range(10)}
    assert sigma_500_from_population_contrasts(values) == pytest.approx(statistics.stdev(range(10)))

def test_sigma_500_rejects_nine_seeds_replacement_and_unknown_seed() -> None:
    values = {6_400_000+i: float(i) for i in range(9)}
    with pytest.raises(H4HoldoutVarianceAnalysisError): sigma_500_from_population_contrasts(values)
    values[6_400_010] = 9.0
    with pytest.raises(H4HoldoutVarianceAnalysisError): sigma_500_from_population_contrasts(values)

def test_sample_size_plan_matches_frozen_spec35_formulas() -> None:
    sigma = 100.0; plan = plan_final_h4_sample_size(sigma)
    expected_power = ((1.9599639845400536 + 0.8416212335729143) ** 2 * sigma**2) / 50.0**2
    expected_precision = ((1.9599639845400536 * sigma) / 50.0) ** 2
    assert plan.n_power_raw == pytest.approx(expected_power)
    assert plan.n_power_ceil == math.ceil(expected_power)
    assert plan.n_precision_raw == pytest.approx(expected_precision)
    assert plan.n_precision_ceil == math.ceil(expected_precision)
    assert plan.n_final == max(10, math.ceil(expected_power), math.ceil(expected_precision))
    assert plan.planning_half_width_bb100 == pytest.approx(1.9599639845400536 * sigma / math.sqrt(plan.n_final))

def test_sample_size_floor_is_ten_and_observed_mean_is_not_an_input() -> None:
    assert plan_final_h4_sample_size(0.0).n_final == 10
    values_a = {6_400_000+i: float(i) for i in range(10)}
    values_b = {seed: value + 10_000 for seed, value in values_a.items()}
    assert variance_pilot_plan_from_complete_seed_contrasts(values_a) == variance_pilot_plan_from_complete_seed_contrasts(values_b)

def test_nonfinite_or_negative_sigma_fails_closed() -> None:
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(H4HoldoutVarianceAnalysisError): plan_final_h4_sample_size(value)
