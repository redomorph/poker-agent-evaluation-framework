"""Unit tests for policy calibration and opponent profiles."""

from __future__ import annotations
from collections.abc import Callable

import pytest

from pokerbench.agents.profiles import (
    CALLING_STATION_PROFILE,
    LAG_PROFILE,
    MANIAC_PROFILE,
    NIT_PROFILE,
    TAG_PROFILE,
    OpponentParameters,
    OpponentProfile,
    ProfileName,
)
from pokerbench.evaluation.calibration import (
    DEFAULT_CALIBRATION_TOLERANCES,
    BehaviorCounts,
    CalibrationMetric,
    ObservedBehavior,
    calibrate_profile,
)


def _metric(
    result,
    metric: CalibrationMetric,
):
    return next(
        item
        for item in result.metrics
        if item.metric == metric
    )


def test_parameter_schema_rejects_pfr_above_vpip() -> None:
    with pytest.raises(
        ValueError,
        match="pfr <= vpip",
    ):
        OpponentParameters(
            vpip=0.20,
            pfr=0.25,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OpponentParameters(
            vpip=1.1,
            pfr=0.10,
        ),
        lambda: OpponentParameters(
            vpip=1.0,
            pfr=1.1,
        ),
        lambda: OpponentParameters(
            vpip=0.20,
            pfr=0.10,
            three_bet_freq=1.1,
        ),
        lambda: OpponentParameters(
            vpip=0.20,
            pfr=0.10,
            cbet_freq=1.1,
        ),
        lambda: OpponentParameters(
            vpip=0.20,
            pfr=0.10,
            fold_to_cbet=1.1,
        ),
    ],
)
def test_frequency_targets_must_be_bounded(
    factory: Callable[
        [],
        OpponentParameters,
    ],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        factory()


def test_aggression_is_ratio_and_may_exceed_one() -> None:
    parameters = OpponentParameters(
        vpip=0.55,
        pfr=0.45,
        aggression=1.5,
    )

    assert (
        parameters.aggression
        == 1.5
    )


def test_static_profiles_match_frozen_targets() -> None:
    assert (
        NIT_PROFILE.parameters.vpip,
        NIT_PROFILE.parameters.pfr,
        NIT_PROFILE.parameters.aggression,
    ) == (
        0.10,
        0.04,
        0.3,
    )

    assert (
        TAG_PROFILE.parameters.vpip,
        TAG_PROFILE.parameters.pfr,
        TAG_PROFILE.parameters.aggression,
    ) == (
        0.22,
        0.18,
        0.8,
    )

    assert (
        LAG_PROFILE.parameters.vpip,
        LAG_PROFILE.parameters.pfr,
        LAG_PROFILE.parameters.aggression,
    ) == (
        0.32,
        0.26,
        0.9,
    )

    assert (
        CALLING_STATION_PROFILE
        .parameters
        .vpip
        == 0.50
    )

    assert (
        CALLING_STATION_PROFILE
        .parameters
        .pfr
        == 0.05
    )

    assert (
        CALLING_STATION_PROFILE
        .parameters
        .fold_to_cbet
        == 0.15
    )

    assert (
        CALLING_STATION_PROFILE
        .parameters
        .aggression
        is None
    )

    assert (
        MANIAC_PROFILE.parameters.vpip,
        MANIAC_PROFILE.parameters.pfr,
        MANIAC_PROFILE.parameters.aggression,
    ) == (
        0.55,
        0.45,
        1.5,
    )


def test_observed_statistics_use_explicit_denominators() -> None:
    counts = BehaviorCounts(
        vpip_opportunities=100,
        vpip_events=22,
        pfr_opportunities=100,
        pfr_events=18,
        three_bet_opportunities=20,
        three_bet_events=2,
        cbet_opportunities=10,
        cbet_events=6,
        postflop_aggressive_actions=8,
        postflop_passive_actions=10,
        fold_to_cbet_opportunities=20,
        fold_to_cbet_events=3,
    )

    observed = (
        ObservedBehavior.from_counts(
            counts
        )
    )

    assert observed.vpip == pytest.approx(
        0.22
    )

    assert observed.pfr == pytest.approx(
        0.18
    )

    assert (
        observed.three_bet_freq
        == pytest.approx(0.10)
    )

    assert (
        observed.cbet_freq
        == pytest.approx(0.60)
    )

    assert (
        observed.aggression
        == pytest.approx(0.80)
    )

    assert (
        observed.fold_to_cbet
        == pytest.approx(0.15)
    )


def test_zero_denominator_produces_undefined_observation() -> None:
    observed = (
        ObservedBehavior.from_counts(
            BehaviorCounts()
        )
    )

    assert observed.vpip is None
    assert observed.pfr is None
    assert (
        observed.three_bet_freq
        is None
    )
    assert observed.cbet_freq is None
    assert observed.aggression is None
    assert (
        observed.fold_to_cbet
        is None
    )


def test_frequency_events_cannot_exceed_opportunities() -> None:
    with pytest.raises(
        ValueError,
        match="VPIP",
    ):
        BehaviorCounts(
            vpip_opportunities=10,
            vpip_events=11,
        )


def test_tag_passes_at_tolerance_boundary() -> None:
    counts = BehaviorCounts(
        vpip_opportunities=100,
        vpip_events=25,
        pfr_opportunities=100,
        pfr_events=21,
        three_bet_opportunities=100,
        three_bet_events=7,
        cbet_opportunities=100,
        cbet_events=65,

        # Aggression = 1.0.
        # TAG target = 0.8, deviation = 0.2.
        postflop_aggressive_actions=10,
        postflop_passive_actions=10,
        fold_to_cbet_opportunities=100,
        fold_to_cbet_events=45,
    )

    result = calibrate_profile(
        TAG_PROFILE,
        counts,
    )

    assert result.is_calibrated

    assert (
        _metric(
            result,
            CalibrationMetric.VPIP,
        ).passed
        is True
    )

    assert (
        _metric(
            result,
            CalibrationMetric.PFR,
        ).passed
        is True
    )

    assert (
        _metric(
            result,
            CalibrationMetric.AGGRESSION,
        ).passed
        is True
    )


def test_tag_fails_outside_tolerance() -> None:
    counts = BehaviorCounts(
        vpip_opportunities=100,
        vpip_events=26,
        pfr_opportunities=100,
        pfr_events=18,
        postflop_aggressive_actions=8,
        postflop_passive_actions=10,
    )

    result = calibrate_profile(
        TAG_PROFILE,
        counts,
    )

    assert not result.is_calibrated

    vpip = _metric(
        result,
        CalibrationMetric.VPIP,
    )

    assert vpip.required
    assert vpip.passed is False

    assert vpip.deviation == pytest.approx(
        0.04
    )


def test_unspecified_target_is_not_acceptance_criterion() -> None:
    counts = BehaviorCounts(
        vpip_opportunities=100,
        vpip_events=50,
        pfr_opportunities=100,
        pfr_events=5,
        three_bet_opportunities=100,
        three_bet_events=2,
        cbet_opportunities=100,
        cbet_events=35,

        # Deliberately extreme aggression.
        postflop_aggressive_actions=100,
        postflop_passive_actions=1,

        fold_to_cbet_opportunities=100,
        fold_to_cbet_events=15,
    )

    result = calibrate_profile(
        CALLING_STATION_PROFILE,
        counts,
    )

    aggression = _metric(
        result,
        CalibrationMetric.AGGRESSION,
    )

    assert not aggression.required
    assert aggression.target is None
    assert aggression.passed is None

    assert result.is_calibrated


def test_required_metric_with_zero_opportunities_fails() -> None:
    profile = OpponentProfile(
        name=ProfileName.TAG,
        parameters=OpponentParameters(
            vpip=0.22,
            pfr=0.18,
            cbet_freq=0.60,
            aggression=0.8,
        ),
    )

    counts = BehaviorCounts(
        vpip_opportunities=100,
        vpip_events=22,
        pfr_opportunities=100,
        pfr_events=18,
        cbet_opportunities=0,
        cbet_events=0,
        postflop_aggressive_actions=8,
        postflop_passive_actions=10,
    )

    result = calibrate_profile(
        profile,
        counts,
    )

    cbet = _metric(
        result,
        CalibrationMetric.CBET,
    )

    assert cbet.required
    assert cbet.observed is None
    assert cbet.passed is False

    assert not result.is_calibrated


def test_frozen_default_tolerances_match_specification() -> None:
    tolerances = (
        DEFAULT_CALIBRATION_TOLERANCES
    )

    assert tolerances.vpip == 0.03
    assert tolerances.pfr == 0.03

    assert (
        tolerances.three_bet_freq
        == 0.02
    )

    assert (
        tolerances.cbet_freq
        == 0.05
    )

    assert (
        tolerances.aggression
        == 0.20
    )

    assert (
        tolerances.fold_to_cbet
        == 0.05
    )


def test_calibration_result_is_persistence_ready() -> None:
    counts = BehaviorCounts(
        vpip_opportunities=100,
        vpip_events=22,
        pfr_opportunities=100,
        pfr_events=18,
        three_bet_opportunities=100,
        three_bet_events=7,
        cbet_opportunities=100,
        cbet_events=65,
        postflop_aggressive_actions=8,
        postflop_passive_actions=10,
        fold_to_cbet_opportunities=100,
        fold_to_cbet_events=45,
    )

    result = calibrate_profile(
        TAG_PROFILE,
        counts,
    )

    persisted = result.to_dict()

    assert (
        persisted["profile_name"]
        == "tag"
    )

    assert (
        persisted["is_calibrated"]
        is True
    )

    assert len(
        persisted["metrics"]
    ) == 6