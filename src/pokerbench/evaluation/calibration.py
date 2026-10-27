"""
Policy-calibration measurement and acceptance framework.

This module deliberately separates:

    target policy parameters

from:

    observed behavioural frequencies

Calibration arithmetic is performed from explicitly supplied
opportunity/event counts. The operational poker definitions used to derive
those counts from replay traces are frozen separately in
``pokerbench.evaluation.opportunities``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pokerbench.agents.profiles import (
    OpponentParameters,
    OpponentProfile,
)


CALIBRATION_VERSION = (
    "policy-calibration-v1"
)

OPPORTUNITY_DEFINITION_CONTRACT_VERSION = (
    "calibration-opportunities-v2"
)


class CalibrationMetric(StrEnum):
    """Observable metrics required by the calibration specification."""

    VPIP = "vpip"
    PFR = "pfr"
    THREE_BET = "three_bet_freq"
    CBET = "cbet_freq"
    AGGRESSION = "aggression"
    FOLD_TO_CBET = "fold_to_cbet"


@dataclass(frozen=True)
class CalibrationTolerances:
    """
    Frozen practical calibration tolerances.

    Frequency tolerances are expressed as absolute proportions:

        0.03 == 3 percentage points.
    """

    vpip: float = 0.03
    pfr: float = 0.03

    three_bet_freq: float = 0.02
    cbet_freq: float = 0.05

    aggression: float = 0.20

    fold_to_cbet: float = 0.05

    version: str = "calibration-tolerances-v1"

    def __post_init__(
        self,
    ) -> None:
        for field_name in (
            "vpip",
            "pfr",
            "three_bet_freq",
            "cbet_freq",
            "aggression",
            "fold_to_cbet",
        ):
            value = getattr(
                self,
                field_name,
            )

            if not isinstance(
                value,
                (int, float),
            ) or isinstance(
                value,
                bool,
            ):
                raise TypeError(
                    f"{field_name} tolerance must be numeric"
                )

            if (
                not math.isfinite(
                    float(value)
                )
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"{field_name} tolerance must be "
                    "finite and non-negative"
                )


DEFAULT_CALIBRATION_TOLERANCES = (
    CalibrationTolerances()
)


@dataclass(frozen=True)
class BehaviorCounts:
    """
    Explicit opportunity/event counts from one calibration run.

    The event detector producing these counts is intentionally outside this
    class. This prevents the mathematical calibration layer from silently
    inventing poker-specific opportunity definitions.

    Frequency metrics use:

        observed = events / opportunities

    Aggression uses:

        aggressive postflop actions / passive postflop actions

    where aggressive actions are bet/raise and passive actions are
    call/check. Folds are not part of that ratio.
    """

    vpip_opportunities: int = 0
    vpip_events: int = 0

    pfr_opportunities: int = 0
    pfr_events: int = 0

    three_bet_opportunities: int = 0
    three_bet_events: int = 0

    cbet_opportunities: int = 0
    cbet_events: int = 0

    postflop_aggressive_actions: int = 0
    postflop_passive_actions: int = 0

    fold_to_cbet_opportunities: int = 0
    fold_to_cbet_events: int = 0

    opportunity_definition_version: str = (
        OPPORTUNITY_DEFINITION_CONTRACT_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        if not self.opportunity_definition_version:
            raise ValueError(
                "opportunity_definition_version "
                "must not be empty"
            )

        count_fields = (
            "vpip_opportunities",
            "vpip_events",
            "pfr_opportunities",
            "pfr_events",
            "three_bet_opportunities",
            "three_bet_events",
            "cbet_opportunities",
            "cbet_events",
            "postflop_aggressive_actions",
            "postflop_passive_actions",
            "fold_to_cbet_opportunities",
            "fold_to_cbet_events",
        )

        for field_name in count_fields:
            value = getattr(
                self,
                field_name,
            )

            if (
                not isinstance(
                    value,
                    int,
                )
                or isinstance(
                    value,
                    bool,
                )
            ):
                raise TypeError(
                    f"{field_name} must be an integer"
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} must be non-negative"
                )

        self._validate_event_pair(
            events=self.vpip_events,
            opportunities=(
                self.vpip_opportunities
            ),
            metric="VPIP",
        )

        self._validate_event_pair(
            events=self.pfr_events,
            opportunities=(
                self.pfr_opportunities
            ),
            metric="PFR",
        )

        self._validate_event_pair(
            events=(
                self.three_bet_events
            ),
            opportunities=(
                self.three_bet_opportunities
            ),
            metric="3Bet",
        )

        self._validate_event_pair(
            events=self.cbet_events,
            opportunities=(
                self.cbet_opportunities
            ),
            metric="CBetFreq",
        )

        self._validate_event_pair(
            events=(
                self.fold_to_cbet_events
            ),
            opportunities=(
                self.fold_to_cbet_opportunities
            ),
            metric="FoldToCBet",
        )

    @staticmethod
    def _validate_event_pair(
        *,
        events: int,
        opportunities: int,
        metric: str,
    ) -> None:
        if events > opportunities:
            raise ValueError(
                f"{metric} events cannot exceed opportunities"
            )


@dataclass(frozen=True)
class ObservedBehavior:
    """
    Observable calibration statistics.

    None means that the metric is undefined because its denominator was zero.
    """

    vpip: float | None
    pfr: float | None

    three_bet_freq: float | None
    cbet_freq: float | None

    aggression: float | None

    fold_to_cbet: float | None

    @classmethod
    def from_counts(
        cls,
        counts: BehaviorCounts,
    ) -> "ObservedBehavior":
        return cls(
            vpip=_frequency(
                counts.vpip_events,
                counts.vpip_opportunities,
            ),
            pfr=_frequency(
                counts.pfr_events,
                counts.pfr_opportunities,
            ),
            three_bet_freq=_frequency(
                counts.three_bet_events,
                counts.three_bet_opportunities,
            ),
            cbet_freq=_frequency(
                counts.cbet_events,
                counts.cbet_opportunities,
            ),
            aggression=_ratio(
                counts.postflop_aggressive_actions,
                counts.postflop_passive_actions,
            ),
            fold_to_cbet=_frequency(
                counts.fold_to_cbet_events,
                counts.fold_to_cbet_opportunities,
            ),
        )


@dataclass(frozen=True)
class CalibrationMetricResult:
    """
    Acceptance result for one calibration metric.

    required=False means the benchmark profile does not define a target for
    that metric.

    required=True with observed=None always fails because a defined target
    cannot be validated without an observable denominator.
    """

    metric: CalibrationMetric

    target: float | None
    observed: float | None

    tolerance: float

    numerator: int
    denominator: int

    required: bool

    deviation: float | None

    passed: bool | None


@dataclass(frozen=True)
class CalibrationResult:
    """Complete frozen calibration result for one profile."""

    profile_name: str
    profile_version: str

    parameter_schema_version: str

    calibration_version: str

    opportunity_definition_version: str
    tolerance_version: str

    metrics: tuple[
        CalibrationMetricResult,
        ...,
    ]

    is_calibrated: bool

    def __post_init__(
        self,
    ) -> None:
        if (
            self.calibration_version
            != CALIBRATION_VERSION
        ):
            raise ValueError(
                "Unsupported calibration version"
            )

        expected_metrics = set(
            CalibrationMetric
        )

        actual_metrics = {
            result.metric
            for result in self.metrics
        }

        if actual_metrics != expected_metrics:
            raise ValueError(
                "CalibrationResult must contain "
                "exactly one result for every metric"
            )

        calculated = all(
            (
                not result.required
                or result.passed is True
            )
            for result in self.metrics
        )

        if calculated != self.is_calibrated:
            raise ValueError(
                "is_calibrated is inconsistent "
                "with metric results"
            )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        """Return persistence-ready deterministic calibration data."""
        return {
            "profile_name": (
                self.profile_name
            ),
            "profile_version": (
                self.profile_version
            ),
            "parameter_schema_version": (
                self.parameter_schema_version
            ),
            "calibration_version": (
                self.calibration_version
            ),
            "opportunity_definition_version": (
                self.opportunity_definition_version
            ),
            "tolerance_version": (
                self.tolerance_version
            ),
            "is_calibrated": (
                self.is_calibrated
            ),
            "metrics": [
                {
                    "metric": (
                        result.metric.value
                    ),
                    "target": (
                        result.target
                    ),
                    "observed": (
                        result.observed
                    ),
                    "tolerance": (
                        result.tolerance
                    ),
                    "numerator": (
                        result.numerator
                    ),
                    "denominator": (
                        result.denominator
                    ),
                    "required": (
                        result.required
                    ),
                    "deviation": (
                        result.deviation
                    ),
                    "passed": (
                        result.passed
                    ),
                }
                for result in self.metrics
            ],
        }


@dataclass(frozen=True)
class _MetricObservation:
    value: float | None
    numerator: int
    denominator: int


def calibrate_parameters(
    *,
    profile_name: str,
    profile_version: str,
    parameters: OpponentParameters,
    counts: BehaviorCounts,
    tolerances: CalibrationTolerances = (
        DEFAULT_CALIBRATION_TOLERANCES
    ),
) -> CalibrationResult:
    """Compare observed behaviour against one explicit parameter vector."""
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError("profile_name must be a non-empty string")
    if not isinstance(profile_version, str) or not profile_version:
        raise ValueError("profile_version must be a non-empty string")
    if not isinstance(parameters, OpponentParameters):
        raise TypeError("parameters must be OpponentParameters")
    if (
        counts.opportunity_definition_version
        != OPPORTUNITY_DEFINITION_CONTRACT_VERSION
    ):
        raise ValueError(
            "Calibration counts use unsupported "
            "opportunity-definition version: "
            f"{counts.opportunity_definition_version}"
        )

    observed = ObservedBehavior.from_counts(counts)
    results = tuple(
        _evaluate_metric(
            metric=metric,
            parameters=parameters,
            observed=observed,
            counts=counts,
            tolerances=tolerances,
        )
        for metric in CalibrationMetric
    )
    is_calibrated = all(
        not result.required or result.passed is True
        for result in results
    )

    return CalibrationResult(
        profile_name=profile_name,
        profile_version=profile_version,
        parameter_schema_version=parameters.schema_version,
        calibration_version=CALIBRATION_VERSION,
        opportunity_definition_version=(
            counts.opportunity_definition_version
        ),
        tolerance_version=tolerances.version,
        metrics=results,
        is_calibrated=is_calibrated,
    )


def calibrate_profile(
    profile: OpponentProfile,
    counts: BehaviorCounts,
    *,
    tolerances: CalibrationTolerances = (
        DEFAULT_CALIBRATION_TOLERANCES
    ),
) -> CalibrationResult:
    """Compare observed behaviour against one frozen static profile."""
    return calibrate_parameters(
        profile_name=profile.name.value,
        profile_version=profile.version,
        parameters=profile.parameters,
        counts=counts,
        tolerances=tolerances,
    )


def _evaluate_metric(
    *,
    metric: CalibrationMetric,
    parameters: OpponentParameters,
    observed: ObservedBehavior,
    counts: BehaviorCounts,
    tolerances: CalibrationTolerances,
) -> CalibrationMetricResult:
    target = _target_for(
        parameters,
        metric,
    )

    observation = (
        _observation_for(
            observed,
            counts,
            metric,
        )
    )

    tolerance = (
        _tolerance_for(
            tolerances,
            metric,
        )
    )

    if target is None:
        return CalibrationMetricResult(
            metric=metric,
            target=None,
            observed=(
                observation.value
            ),
            tolerance=tolerance,
            numerator=(
                observation.numerator
            ),
            denominator=(
                observation.denominator
            ),
            required=False,
            deviation=None,
            passed=None,
        )

    if observation.value is None:
        return CalibrationMetricResult(
            metric=metric,
            target=float(
                target
            ),
            observed=None,
            tolerance=tolerance,
            numerator=(
                observation.numerator
            ),
            denominator=(
                observation.denominator
            ),
            required=True,
            deviation=None,
            passed=False,
        )

    deviation = abs(
        observation.value
        - float(target)
    )

    passed = (
        deviation
        <= tolerance
        + 1e-12
    )

    return CalibrationMetricResult(
        metric=metric,
        target=float(
            target
        ),
        observed=(
            observation.value
        ),
        tolerance=tolerance,
        numerator=(
            observation.numerator
        ),
        denominator=(
            observation.denominator
        ),
        required=True,
        deviation=deviation,
        passed=passed,
    )


def _target_for(
    parameters: OpponentParameters,
    metric: CalibrationMetric,
) -> float | None:
    if metric == CalibrationMetric.VPIP:
        return parameters.vpip

    if metric == CalibrationMetric.PFR:
        return parameters.pfr

    if (
        metric
        == CalibrationMetric.THREE_BET
    ):
        return (
            parameters.three_bet_freq
        )

    if (
        metric
        == CalibrationMetric.CBET
    ):
        return (
            parameters.cbet_freq
        )

    if (
        metric
        == CalibrationMetric.AGGRESSION
    ):
        return (
            parameters.aggression
        )

    if (
        metric
        == CalibrationMetric.FOLD_TO_CBET
    ):
        return (
            parameters.fold_to_cbet
        )

    raise ValueError(
        f"Unsupported calibration metric: {metric}"
    )


def _observation_for(
    observed: ObservedBehavior,
    counts: BehaviorCounts,
    metric: CalibrationMetric,
) -> _MetricObservation:
    if metric == CalibrationMetric.VPIP:
        return _MetricObservation(
            value=observed.vpip,
            numerator=(
                counts.vpip_events
            ),
            denominator=(
                counts.vpip_opportunities
            ),
        )

    if metric == CalibrationMetric.PFR:
        return _MetricObservation(
            value=observed.pfr,
            numerator=(
                counts.pfr_events
            ),
            denominator=(
                counts.pfr_opportunities
            ),
        )

    if (
        metric
        == CalibrationMetric.THREE_BET
    ):
        return _MetricObservation(
            value=(
                observed.three_bet_freq
            ),
            numerator=(
                counts.three_bet_events
            ),
            denominator=(
                counts.three_bet_opportunities
            ),
        )

    if (
        metric
        == CalibrationMetric.CBET
    ):
        return _MetricObservation(
            value=(
                observed.cbet_freq
            ),
            numerator=(
                counts.cbet_events
            ),
            denominator=(
                counts.cbet_opportunities
            ),
        )

    if (
        metric
        == CalibrationMetric.AGGRESSION
    ):
        return _MetricObservation(
            value=(
                observed.aggression
            ),
            numerator=(
                counts
                .postflop_aggressive_actions
            ),
            denominator=(
                counts
                .postflop_passive_actions
            ),
        )

    if (
        metric
        == CalibrationMetric.FOLD_TO_CBET
    ):
        return _MetricObservation(
            value=(
                observed.fold_to_cbet
            ),
            numerator=(
                counts
                .fold_to_cbet_events
            ),
            denominator=(
                counts
                .fold_to_cbet_opportunities
            ),
        )

    raise ValueError(
        f"Unsupported calibration metric: {metric}"
    )


def _tolerance_for(
    tolerances: CalibrationTolerances,
    metric: CalibrationMetric,
) -> float:
    if metric == CalibrationMetric.VPIP:
        return tolerances.vpip

    if metric == CalibrationMetric.PFR:
        return tolerances.pfr

    if (
        metric
        == CalibrationMetric.THREE_BET
    ):
        return (
            tolerances
            .three_bet_freq
        )

    if (
        metric
        == CalibrationMetric.CBET
    ):
        return (
            tolerances.cbet_freq
        )

    if (
        metric
        == CalibrationMetric.AGGRESSION
    ):
        return (
            tolerances.aggression
        )

    if (
        metric
        == CalibrationMetric.FOLD_TO_CBET
    ):
        return (
            tolerances
            .fold_to_cbet
        )

    raise ValueError(
        f"Unsupported calibration metric: {metric}"
    )


def _frequency(
    events: int,
    opportunities: int,
) -> float | None:
    if opportunities == 0:
        return None

    return (
        events
        / opportunities
    )


def _ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return (
        numerator
        / denominator
    )