"""
Validation of the frozen policy-calibration contract.

This file validates the measurement and acceptance framework.

It does NOT claim that the benchmark profiles themselves have been calibrated
against an implemented parametric playing policy. That final acceptance gate
requires the separately frozen policy mechanism and opportunity detector.
"""

from __future__ import annotations

from pokerbench.agents.profiles import (
    CALLING_STATION_PROFILE,
    TAG_PROFILE,
    OpponentParameters,
    OpponentProfile,
    ProfileName,
)
from pokerbench.evaluation.calibration import (
    BehaviorCounts,
    CalibrationMetric,
    calibrate_profile,
)


def _result_for_metric(
    result,
    metric: CalibrationMetric,
):
    return next(
        value
        for value in result.metrics
        if value.metric == metric
    )


def test_tag_calibration_contract_accepts_in_tolerance_observations(
) -> None:
    result = calibrate_profile(
        TAG_PROFILE,
        BehaviorCounts(
            vpip_opportunities=10_000,
            vpip_events=2_200,
            pfr_opportunities=10_000,
            pfr_events=1_800,
            three_bet_opportunities=1_000,
            three_bet_events=70,
            cbet_opportunities=1_000,
            cbet_events=650,
            fold_to_cbet_opportunities=1_000,
            fold_to_cbet_events=450,
            postflop_aggressive_actions=800,
            postflop_passive_actions=1_000,
        ),
    )

    assert result.is_calibrated


def test_calling_station_fold_to_cbet_target_is_enforced(
) -> None:
    result = calibrate_profile(
        CALLING_STATION_PROFILE,
        BehaviorCounts(
            vpip_opportunities=10_000,
            vpip_events=5_000,
            pfr_opportunities=10_000,
            pfr_events=500,
            three_bet_opportunities=1_000,
            three_bet_events=20,
            cbet_opportunities=1_000,
            cbet_events=350,
            fold_to_cbet_opportunities=1_000,
            fold_to_cbet_events=150,
        ),
    )

    assert result.is_calibrated

    fold_result = (
        _result_for_metric(
            result,
            CalibrationMetric.FOLD_TO_CBET,
        )
    )

    assert fold_result.required
    assert fold_result.passed is True


def test_calling_station_aggression_is_dont_care(
) -> None:
    result = calibrate_profile(
        CALLING_STATION_PROFILE,
        BehaviorCounts(
            vpip_opportunities=1_000,
            vpip_events=500,
            pfr_opportunities=1_000,
            pfr_events=50,

            three_bet_opportunities=1_000,
            three_bet_events=20,
            cbet_opportunities=1_000,
            cbet_events=350,
            postflop_aggressive_actions=1_000,
            postflop_passive_actions=1,

            fold_to_cbet_opportunities=100,
            fold_to_cbet_events=15,
        ),
    )

    aggression = (
        _result_for_metric(
            result,
            CalibrationMetric.AGGRESSION,
        )
    )

    assert not aggression.required
    assert aggression.passed is None

    assert result.is_calibrated


def test_contract_supports_all_six_calibration_metrics(
) -> None:
    """
    Synthetic fully specified target vector.

    This validates the calibration machinery for all required observable
    quantities without pretending that Section 7.5 supplies all six targets
    for every static profile.
    """
    profile = OpponentProfile(
        name=ProfileName.TAG,
        parameters=OpponentParameters(
            vpip=0.30,
            pfr=0.20,
            three_bet_freq=0.10,
            cbet_freq=0.60,
            aggression=1.0,
            fold_to_cbet=0.40,
        ),
    )

    result = calibrate_profile(
        profile,
        BehaviorCounts(
            vpip_opportunities=1_000,
            vpip_events=300,

            pfr_opportunities=1_000,
            pfr_events=200,

            three_bet_opportunities=200,
            three_bet_events=20,

            cbet_opportunities=200,
            cbet_events=120,

            postflop_aggressive_actions=100,
            postflop_passive_actions=100,

            fold_to_cbet_opportunities=200,
            fold_to_cbet_events=80,
        ),
    )

    assert result.is_calibrated

    assert all(
        item.passed is True
        for item in result.metrics
    )


def test_targeted_metric_without_opportunities_cannot_pass(
) -> None:
    profile = OpponentProfile(
        name=ProfileName.TAG,
        parameters=OpponentParameters(
            vpip=0.22,
            pfr=0.18,
            cbet_freq=0.60,
            aggression=0.8,
        ),
    )

    result = calibrate_profile(
        profile,
        BehaviorCounts(
            vpip_opportunities=1_000,
            vpip_events=220,

            pfr_opportunities=1_000,
            pfr_events=180,

            cbet_opportunities=0,
            cbet_events=0,

            postflop_aggressive_actions=80,
            postflop_passive_actions=100,
        ),
    )

    assert not result.is_calibrated

    cbet = _result_for_metric(
        result,
        CalibrationMetric.CBET,
    )

    assert cbet.required
    assert cbet.passed is False