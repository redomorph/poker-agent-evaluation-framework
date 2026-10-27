"""Unit tests for Spec-32 adaptive-pilot analysis primitives."""

from __future__ import annotations

import math

import pytest

from pokerbench.evaluation.population import (
    BALANCED_STATIC_POPULATION,
    BASELINE_SYNTHETIC_STATIC_POPULATION,
)
from pokerbench.experiments.adaptive_pilot_analysis import (
    ADAPTIVE_PILOT_VARIANCE_ESTIMATOR,
    AdaptivePilotAnalysisError,
    AdaptivePilotInferenceProhibitedError,
    TrajectoryArmEvidence,
    cumulative_arm_estimate,
    matched_trajectory_contrast,
    population_weighted_seed_contrast,
    population_contrast_sample_sd,
    profile_contrast_sample_sd,
    segment_arm_estimate,
)


def _evidence(
    *,
    arm_id: str,
    profile_id: str = "tag",
    master_seed: int = 6_300_000,
    btn: tuple[float, ...] | None = None,
    bb: tuple[float, ...] | None = None,
    scientifically_valid: bool = True,
) -> TrajectoryArmEvidence:
    default = tuple(0.0 for _ in range(500))
    return TrajectoryArmEvidence(
        arm_id=arm_id,
        profile_id=profile_id,
        master_seed=master_seed,
        hero_btn_profit_bb=default if btn is None else btn,
        hero_bb_profit_bb=default if bb is None else bb,
        scientifically_valid=scientifically_valid,
    )


def _constant_evidence(
    *,
    arm_id: str,
    value_bb_per_position_hand: float,
    profile_id: str = "tag",
    master_seed: int = 6_300_000,
    length: int = 500,
) -> TrajectoryArmEvidence:
    values = tuple(value_bb_per_position_hand for _ in range(length))
    return _evidence(
        arm_id=arm_id,
        profile_id=profile_id,
        master_seed=master_seed,
        btn=values,
        bb=values,
    )


def _contrast(
    *,
    profile_id: str,
    master_seed: int,
    treatment_bb_per_position_hand: float,
    control_bb_per_position_hand: float = 0.0,
    horizon: int = 100,
):
    treatment = cumulative_arm_estimate(
        _constant_evidence(
            arm_id="treatment",
            value_bb_per_position_hand=treatment_bb_per_position_hand,
            profile_id=profile_id,
            master_seed=master_seed,
        ),
        horizon=horizon,
    )
    control = cumulative_arm_estimate(
        _constant_evidence(
            arm_id="control",
            value_bb_per_position_hand=control_bb_per_position_hand,
            profile_id=profile_id,
            master_seed=master_seed,
        ),
        horizon=horizon,
    )
    return matched_trajectory_contrast(treatment=treatment, control=control)


def test_cumulative_estimator_uses_full_prefix_and_position_compensation() -> None:
    btn = tuple([1.0] * 10 + [3.0] * 10 + [0.0] * 480)
    bb = tuple(0.0 for _ in range(500))
    evidence = _evidence(arm_id="treatment", btn=btn, bb=bb)

    h10 = cumulative_arm_estimate(evidence, horizon=10)
    h20 = cumulative_arm_estimate(evidence, horizon=20)

    assert h10.window_kind == "cumulative"
    assert h10.start_hand == 1
    assert h10.end_hand == 10
    assert h10.bb100 == pytest.approx(50.0)

    assert h20.start_hand == 1
    assert h20.end_hand == 20
    assert h20.bb100 == pytest.approx(100.0)


def test_segment_estimator_uses_inclusive_frozen_intervals() -> None:
    btn = tuple([1.0] * 20 + [2.0] * 80 + [3.0] * 400)
    bb = tuple(0.0 for _ in range(500))
    evidence = _evidence(arm_id="treatment", btn=btn, bb=bb)

    early = segment_arm_estimate(evidence, segment_id="early_adaptation")
    main = segment_arm_estimate(evidence, segment_id="main_adaptation")
    steady = segment_arm_estimate(evidence, segment_id="steady_state")

    assert (early.start_hand, early.end_hand, early.bb100) == pytest.approx((1, 20, 50.0))
    assert (main.start_hand, main.end_hand, main.bb100) == pytest.approx((21, 100, 100.0))
    assert (steady.start_hand, steady.end_hand, steady.bb100) == pytest.approx((101, 500, 150.0))


def test_estimators_reject_off_protocol_windows_and_short_trajectories() -> None:
    short = _constant_evidence(
        arm_id="treatment",
        value_bb_per_position_hand=0.0,
        length=100,
    )

    with pytest.raises(AdaptivePilotAnalysisError, match="frozen cumulative horizon"):
        cumulative_arm_estimate(short, horizon=25)

    with pytest.raises(AdaptivePilotAnalysisError, match="requires 500 hands"):
        cumulative_arm_estimate(short, horizon=500)

    with pytest.raises(AdaptivePilotAnalysisError, match="frozen segment"):
        segment_arm_estimate(short, segment_id="not-a-segment")



def test_analysis_rejects_replacement_seeds_and_long_windows_on_short_seeds() -> None:
    with pytest.raises(AdaptivePilotAnalysisError, match="predeclared pilot seed"):
        _constant_evidence(
            arm_id="treatment",
            value_bb_per_position_hand=0.0,
            master_seed=6_399_999,
        )

    short_seed_with_long_series = _constant_evidence(
        arm_id="treatment",
        value_bb_per_position_hand=0.0,
        master_seed=6_300_001,
        length=500,
    )
    with pytest.raises(AdaptivePilotAnalysisError, match="long-horizon seed"):
        cumulative_arm_estimate(short_seed_with_long_series, horizon=500)
    with pytest.raises(AdaptivePilotAnalysisError, match="long-horizon seed"):
        segment_arm_estimate(short_seed_with_long_series, segment_id="steady_state")

def test_nonfinite_hand_profit_and_invalid_trajectory_are_rejected() -> None:
    with pytest.raises(AdaptivePilotAnalysisError, match="finite"):
        _evidence(
            arm_id="treatment",
            btn=(math.nan,),
            bb=(0.0,),
        )

    invalid = _constant_evidence(
        arm_id="treatment",
        value_bb_per_position_hand=0.0,
    )
    invalid = TrajectoryArmEvidence(
        arm_id=invalid.arm_id,
        profile_id=invalid.profile_id,
        master_seed=invalid.master_seed,
        hero_btn_profit_bb=invalid.hero_btn_profit_bb,
        hero_bb_profit_bb=invalid.hero_bb_profit_bb,
        scientifically_valid=False,
    )

    with pytest.raises(AdaptivePilotAnalysisError, match="scientifically invalid"):
        cumulative_arm_estimate(invalid, horizon=100)


def test_matched_contrast_is_treatment_minus_control_and_requires_complete_pair() -> None:
    treatment = cumulative_arm_estimate(
        _constant_evidence(
            arm_id="treatment",
            value_bb_per_position_hand=0.03,
        ),
        horizon=100,
    )
    control = cumulative_arm_estimate(
        _constant_evidence(
            arm_id="control",
            value_bb_per_position_hand=0.01,
        ),
        horizon=100,
    )

    contrast = matched_trajectory_contrast(treatment=treatment, control=control)

    assert contrast.treatment_bb100 == pytest.approx(3.0)
    assert contrast.control_bb100 == pytest.approx(1.0)
    assert contrast.difference_bb100 == pytest.approx(2.0)
    assert contrast.master_seed == 6_300_000
    assert contrast.profile_id == "tag"

    with pytest.raises(AdaptivePilotAnalysisError, match="complete treatment/control pair"):
        matched_trajectory_contrast(treatment=treatment, control=None)


def test_matched_contrast_rejects_mismatched_profile_seed_or_window() -> None:
    treatment = cumulative_arm_estimate(
        _constant_evidence(
            arm_id="treatment",
            value_bb_per_position_hand=0.01,
            profile_id="tag",
            master_seed=6_300_000,
        ),
        horizon=100,
    )
    wrong_profile = cumulative_arm_estimate(
        _constant_evidence(
            arm_id="control",
            value_bb_per_position_hand=0.0,
            profile_id="nit",
            master_seed=6_300_000,
        ),
        horizon=100,
    )
    wrong_seed = cumulative_arm_estimate(
        _constant_evidence(
            arm_id="control",
            value_bb_per_position_hand=0.0,
            profile_id="tag",
            master_seed=6_300_001,
        ),
        horizon=100,
    )
    wrong_window = cumulative_arm_estimate(
        _constant_evidence(
            arm_id="control",
            value_bb_per_position_hand=0.0,
            profile_id="tag",
            master_seed=6_300_000,
        ),
        horizon=50,
    )

    for control in (wrong_profile, wrong_seed, wrong_window):
        with pytest.raises(AdaptivePilotAnalysisError, match="same profile, seed, and window"):
            matched_trajectory_contrast(treatment=treatment, control=control)


def test_profile_sample_sd_uses_seed_level_contrasts_with_ddof1() -> None:
    contrasts = tuple(
        _contrast(
            profile_id="tag",
            master_seed=6_300_000 + index,
            treatment_bb_per_position_hand=value / 100.0,
            horizon=100,
        )
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0))
    )

    result = profile_contrast_sample_sd(contrasts)

    assert result.estimator_version == ADAPTIVE_PILOT_VARIANCE_ESTIMATOR
    assert result.scope_kind == "profile"
    assert result.scope_id == "tag"
    assert result.horizon == 100
    assert result.num_independent_clusters == 5
    assert result.master_seeds == (
        6_300_000,
        6_300_001,
        6_300_002,
        6_300_003,
        6_300_004,
    )
    assert result.std_difference_bb100 == pytest.approx(math.sqrt(2.5))


def test_500_hand_endpoint_is_rejected_for_profile_variance() -> None:
    contrast = _contrast(
        profile_id="tag",
        master_seed=6_300_000,
        treatment_bb_per_position_hand=0.01,
        horizon=500,
    )

    with pytest.raises(AdaptivePilotInferenceProhibitedError, match="500"):
        profile_contrast_sample_sd((contrast,))


def test_population_weighting_occurs_within_seed_using_frozen_profile_weights() -> None:
    profile_values = {
        "nit": -20.0,
        "tag": 0.0,
        "lag": 10.0,
        "calling_station": 30.0,
        "maniac": 50.0,
    }
    contrasts = tuple(
        _contrast(
            profile_id=profile_id,
            master_seed=6_300_001,
            treatment_bb_per_position_hand=value / 100.0,
        )
        for profile_id, value in reversed(tuple(profile_values.items()))
    )

    primary = population_weighted_seed_contrast(
        contrasts,
        population=BASELINE_SYNTHETIC_STATIC_POPULATION,
    )
    balanced = population_weighted_seed_contrast(
        contrasts,
        population=BALANCED_STATIC_POPULATION,
    )

    assert primary.master_seed == 6_300_001
    assert primary.num_profile_cells == 5
    assert primary.difference_bb100 == pytest.approx(10.5)
    assert balanced.difference_bb100 == pytest.approx(14.0)


def test_population_weighting_rejects_missing_duplicate_or_cross_seed_profiles() -> None:
    profiles = ("nit", "tag", "lag", "calling_station", "maniac")
    contrasts = tuple(
        _contrast(
            profile_id=profile_id,
            master_seed=6_300_000,
            treatment_bb_per_position_hand=0.01,
        )
        for profile_id in profiles
    )

    with pytest.raises(AdaptivePilotAnalysisError, match="exactly match"):
        population_weighted_seed_contrast(
            contrasts[:-1],
            population=BASELINE_SYNTHETIC_STATIC_POPULATION,
        )

    with pytest.raises(AdaptivePilotAnalysisError, match="duplicate"):
        population_weighted_seed_contrast(
            contrasts[:-1] + (contrasts[0],),
            population=BASELINE_SYNTHETIC_STATIC_POPULATION,
        )

    cross_seed = contrasts[:-1] + (
        _contrast(
            profile_id="maniac",
            master_seed=6_300_001,
            treatment_bb_per_position_hand=0.01,
        ),
    )
    with pytest.raises(AdaptivePilotAnalysisError, match="same seed and window"):
        population_weighted_seed_contrast(
            cross_seed,
            population=BASELINE_SYNTHETIC_STATIC_POPULATION,
        )


def test_population_sample_sd_is_across_seed_aggregates_not_profile_cells() -> None:
    population_seed_contrasts = []
    for seed_offset, seed_value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
        per_profile = tuple(
            _contrast(
                profile_id=profile_id,
                master_seed=6_300_000 + seed_offset,
                treatment_bb_per_position_hand=seed_value / 100.0,
            )
            for profile_id in ("nit", "tag", "lag", "calling_station", "maniac")
        )
        population_seed_contrasts.append(
            population_weighted_seed_contrast(
                per_profile,
                population=BASELINE_SYNTHETIC_STATIC_POPULATION,
            )
        )

    result = population_contrast_sample_sd(tuple(population_seed_contrasts))

    assert result.scope_kind == "population"
    assert result.scope_id == "baseline-synthetic-static-profiles"
    assert result.num_independent_clusters == 5
    assert result.std_difference_bb100 == pytest.approx(math.sqrt(2.5))


def test_500_hand_endpoint_is_rejected_for_population_variance() -> None:
    per_profile = tuple(
        _contrast(
            profile_id=profile_id,
            master_seed=6_300_000,
            treatment_bb_per_position_hand=0.01,
            horizon=500,
        )
        for profile_id in ("nit", "tag", "lag", "calling_station", "maniac")
    )
    population_seed_contrast = population_weighted_seed_contrast(
        per_profile,
        population=BASELINE_SYNTHETIC_STATIC_POPULATION,
    )

    with pytest.raises(AdaptivePilotInferenceProhibitedError, match="500"):
        population_contrast_sample_sd((population_seed_contrast,))
