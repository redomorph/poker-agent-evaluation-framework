"""
Frozen parameter definitions for benchmark opponent profiles.

Policy parameters are target behavioural quantities. They are deliberately
separate from the observable frequencies measured during calibration.

The concrete policy mechanism phi that maps these targets and an information
state to actions is versioned separately and is not defined in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


PARAMETER_SCHEMA_VERSION = "opponent-parameters-v1"
STATIC_PROFILE_VERSION = "static-profiles-v2"


class ProfileName(StrEnum):
    """Names of the frozen initial benchmark opponent profiles."""

    NIT = "nit"
    TAG = "tag"
    LAG = "lag"
    CALLING_STATION = "calling_station"
    MANIAC = "maniac"


def _validate_frequency(
    value: float | None,
    field_name: str,
) -> None:
    if value is None:
        return

    if not isinstance(
        value,
        (int, float),
    ) or isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{field_name} must be numeric or None"
        )

    numeric = float(value)

    if not math.isfinite(
        numeric
    ):
        raise ValueError(
            f"{field_name} must be finite"
        )

    if not (
        0.0
        <= numeric
        <= 1.0
    ):
        raise ValueError(
            f"{field_name} must be within [0, 1]"
        )


@dataclass(frozen=True)
class OpponentParameters:
    """
    Frozen target parameter vector for a parametric opponent.

    VPIP and PFR are mandatory because all currently defined static
    benchmark profiles specify both.

    Other targets may be None only when the benchmark profile explicitly
    declares that metric as don't-care. Missing values are never interpreted
    implicitly as don't-care.
    """

    vpip: float
    pfr: float

    three_bet_freq: float | None = None
    cbet_freq: float | None = None
    aggression: float | None = None
    fold_to_cbet: float | None = None

    schema_version: str = (
        PARAMETER_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        if (
            self.schema_version
            != PARAMETER_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported opponent-parameter "
                f"schema version: {self.schema_version}"
            )

        _validate_frequency(
            self.vpip,
            "vpip",
        )

        _validate_frequency(
            self.pfr,
            "pfr",
        )

        _validate_frequency(
            self.three_bet_freq,
            "three_bet_freq",
        )

        _validate_frequency(
            self.cbet_freq,
            "cbet_freq",
        )

        _validate_frequency(
            self.fold_to_cbet,
            "fold_to_cbet",
        )

        if (
            self.pfr
            > self.vpip
        ):
            raise ValueError(
                "pfr must satisfy pfr <= vpip"
            )

        if self.aggression is not None:
            if not isinstance(
                self.aggression,
                (int, float),
            ) or isinstance(
                self.aggression,
                bool,
            ):
                raise TypeError(
                    "aggression must be numeric or None"
                )

            numeric_aggression = float(
                self.aggression
            )

            if not math.isfinite(
                numeric_aggression
            ):
                raise ValueError(
                    "aggression must be finite"
                )

            if numeric_aggression < 0.0:
                raise ValueError(
                    "aggression must be non-negative"
                )


@dataclass(frozen=True)
class OpponentProfile:
    """
    One frozen benchmark opponent profile.

    The profile stores target behaviour only. It does not itself define
    policy mechanics.
    """

    name: ProfileName
    parameters: OpponentParameters

    version: str = (
        STATIC_PROFILE_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        if (
            self.version
            != STATIC_PROFILE_VERSION
        ):
            raise ValueError(
                "Unsupported static-profile version"
            )


NIT_PROFILE = OpponentProfile(
    name=ProfileName.NIT,
    parameters=OpponentParameters(
        vpip=0.10,
        pfr=0.04,
        three_bet_freq=0.02,
        cbet_freq=0.45,
        aggression=0.30,
        fold_to_cbet=0.55,
    ),
)


TAG_PROFILE = OpponentProfile(
    name=ProfileName.TAG,
    parameters=OpponentParameters(
        vpip=0.22,
        pfr=0.18,
        three_bet_freq=0.07,
        cbet_freq=0.65,
        aggression=0.80,
        fold_to_cbet=0.45,
    ),
)


LAG_PROFILE = OpponentProfile(
    name=ProfileName.LAG,
    parameters=OpponentParameters(
        vpip=0.32,
        pfr=0.26,
        three_bet_freq=0.10,
        cbet_freq=0.70,
        aggression=0.90,
        fold_to_cbet=0.38,
    ),
)


CALLING_STATION_PROFILE = OpponentProfile(
    name=ProfileName.CALLING_STATION,
    parameters=OpponentParameters(
        vpip=0.50,
        pfr=0.05,
        three_bet_freq=0.02,
        cbet_freq=0.35,
        aggression=None,
        fold_to_cbet=0.15,
    ),
)


MANIAC_PROFILE = OpponentProfile(
    name=ProfileName.MANIAC,
    parameters=OpponentParameters(
        vpip=0.55,
        pfr=0.45,
        three_bet_freq=0.18,
        cbet_freq=0.80,
        aggression=1.50,
        fold_to_cbet=0.25,
    ),
)


STATIC_PROFILES: Mapping[
    ProfileName,
    OpponentProfile,
] = MappingProxyType(
    {
        ProfileName.NIT: (
            NIT_PROFILE
        ),
        ProfileName.TAG: (
            TAG_PROFILE
        ),
        ProfileName.LAG: (
            LAG_PROFILE
        ),
        ProfileName.CALLING_STATION: (
            CALLING_STATION_PROFILE
        ),
        ProfileName.MANIAC: (
            MANIAC_PROFILE
        ),
    }
)


def get_static_profile(
    name: ProfileName,
) -> OpponentProfile:
    """Return one frozen benchmark profile."""
    return STATIC_PROFILES[
        name
    ]
