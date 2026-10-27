"""Regression tests for the frozen static-profiles-v2 contract."""

from __future__ import annotations

from pokerbench.agents.profiles import (
    CALLING_STATION_PROFILE,
    LAG_PROFILE,
    MANIAC_PROFILE,
    NIT_PROFILE,
    STATIC_PROFILE_VERSION,
    TAG_PROFILE,
)


def _parameter_tuple(profile) -> tuple[float | None, ...]:
    parameters = profile.parameters

    return (
        parameters.vpip,
        parameters.pfr,
        parameters.three_bet_freq,
        parameters.cbet_freq,
        parameters.aggression,
        parameters.fold_to_cbet,
    )


def test_static_profile_version_is_v2() -> None:
    assert STATIC_PROFILE_VERSION == "static-profiles-v2"


def test_nit_profile_matches_frozen_v2_targets() -> None:
    assert _parameter_tuple(
        NIT_PROFILE
    ) == (
        0.10,
        0.04,
        0.02,
        0.45,
        0.30,
        0.55,
    )


def test_tag_profile_matches_frozen_v2_targets() -> None:
    assert _parameter_tuple(
        TAG_PROFILE
    ) == (
        0.22,
        0.18,
        0.07,
        0.65,
        0.80,
        0.45,
    )


def test_lag_profile_matches_frozen_v2_targets() -> None:
    assert _parameter_tuple(
        LAG_PROFILE
    ) == (
        0.32,
        0.26,
        0.10,
        0.70,
        0.90,
        0.38,
    )


def test_calling_station_profile_has_only_explicit_aggression_dont_care() -> None:
    assert _parameter_tuple(
        CALLING_STATION_PROFILE
    ) == (
        0.50,
        0.05,
        0.02,
        0.35,
        None,
        0.15,
    )


def test_maniac_profile_matches_frozen_v2_targets() -> None:
    assert _parameter_tuple(
        MANIAC_PROFILE
    ) == (
        0.55,
        0.45,
        0.18,
        0.80,
        1.50,
        0.25,
    )
