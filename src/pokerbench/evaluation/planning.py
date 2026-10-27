"""
Pre-experiment sample-size planning.

Power and precision are separate scientific design questions.

All returned sample sizes are counts of independent statistical units:

- independent paired deals for eligible static policies;
- independent runs / trajectory clusters for adaptive policies.

They are never automatically interpreted as raw simulated hand counts.

The planning approximations in this module operate on BB/100 quantities.
The effect size, standard deviation, and target half-width must therefore
use the same BB/100 scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


PLANNING_VERSION = "sample-size-planning-v1"

POWER_METHOD = "paired-normal-approximation-v1"
PRECISION_METHOD = "mean-normal-ci-approximation-v1"

DEFAULT_ALPHA = 0.05
DEFAULT_BETA = 0.20
DEFAULT_CONFIDENCE_LEVEL = 0.95

COLD_START_SIGMA_D_BB100 = 100.0


@dataclass(frozen=True)
class PowerPlan:
    """
    Frozen paired-comparison power plan.

    required_independent_units is the ceiling of

        ((z_(1-alpha/2) + z_(1-beta))^2 * sigma_d^2) / delta^2

    and therefore represents independent pairs / clusters, not hands.
    """

    delta_bb100: float
    sigma_d_bb100: float

    alpha: float
    beta: float

    z_alpha_two_sided: float
    z_power: float

    raw_required_independent_units: float
    required_independent_units: int

    variance_source: str

    method: str = POWER_METHOD
    planning_version: str = PLANNING_VERSION


@dataclass(frozen=True)
class PrecisionPlan:
    """
    Frozen precision plan for a mean estimate.

    For a desired two-sided confidence interval half-width h,

        h = z * sigma / sqrt(n)

    gives the planning approximation

        n = (z * sigma / h)^2.

    This plan applies to a mean estimand only. It must not be reused for
    quantiles or other estimands without a separately defined procedure.
    """

    estimand: str

    sigma_bb100: float
    target_half_width_bb100: float

    confidence_level: float
    z_confidence: float

    raw_required_independent_units: float
    required_independent_units: int

    variance_source: str

    method: str = PRECISION_METHOD
    planning_version: str = PLANNING_VERSION


@dataclass(frozen=True)
class SampleSizeRecommendation:
    """
    Combined fixed-N recommendation.

    If both power and precision plans apply to the same experiment, the final
    number of independent units must satisfy both requirements and is therefore
    the maximum of the two planned sample sizes.
    """

    required_independent_units: int

    power_plan: PowerPlan | None
    precision_plan: PrecisionPlan | None

    planning_version: str = PLANNING_VERSION


def plan_paired_power(
    *,
    delta_bb100: float,
    sigma_d_bb100: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    variance_source: str,
) -> PowerPlan:
    """
    Plan independent units for a two-sided paired comparison.

    Parameters
    ----------
    delta_bb100:
        Minimum practically relevant paired difference in BB/100.

    sigma_d_bb100:
        Estimated SD of the independent paired differences in BB/100.

    alpha:
        Two-sided type-I error probability.

    beta:
        Type-II error probability. Power is 1 - beta.

    variance_source:
        Versioned description of the source of sigma_d, for example
        "pilot-5000-v1" or "cold-start-conservative".
    """
    _require_positive_finite(
        delta_bb100,
        "delta_bb100",
    )

    _require_positive_finite(
        sigma_d_bb100,
        "sigma_d_bb100",
    )

    _require_probability(
        alpha,
        "alpha",
    )

    _require_probability(
        beta,
        "beta",
    )

    _require_non_empty(
        variance_source,
        "variance_source",
    )

    normal = NormalDist()

    z_alpha = normal.inv_cdf(
        1.0 - alpha / 2.0
    )

    z_power = normal.inv_cdf(
        1.0 - beta
    )

    raw_required = (
        (
            z_alpha
            + z_power
        )
        ** 2
        * sigma_d_bb100
        ** 2
        / delta_bb100
        ** 2
    )

    required = max(
        1,
        math.ceil(
            raw_required
        ),
    )

    return PowerPlan(
        delta_bb100=delta_bb100,
        sigma_d_bb100=sigma_d_bb100,
        alpha=alpha,
        beta=beta,
        z_alpha_two_sided=z_alpha,
        z_power=z_power,
        raw_required_independent_units=(
            raw_required
        ),
        required_independent_units=(
            required
        ),
        variance_source=(
            variance_source
        ),
    )


def plan_cold_start_paired_power(
    *,
    delta_bb100: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> PowerPlan:
    """
    Produce the provisional cold-start power plan from the specification's
    conservative sigma_d ~= 100 BB/100 assumption.

    This result is provisional. It must be replaced by a pilot-based variance
    estimate before the final confirmatory experiment.
    """
    return plan_paired_power(
        delta_bb100=(
            delta_bb100
        ),
        sigma_d_bb100=(
            COLD_START_SIGMA_D_BB100
        ),
        alpha=alpha,
        beta=beta,
        variance_source=(
            "cold-start-conservative"
        ),
    )


def plan_mean_precision(
    *,
    sigma_bb100: float,
    target_half_width_bb100: float,
    confidence_level: float = (
        DEFAULT_CONFIDENCE_LEVEL
    ),
    estimand: str,
    variance_source: str,
) -> PrecisionPlan:
    """
    Plan independent units for a pre-specified mean confidence-interval
    half-width.

    This is a planning approximation using a normal critical value.

    It must not be interpreted as permission for optional stopping after the
    experiment starts.
    """
    _require_positive_finite(
        sigma_bb100,
        "sigma_bb100",
    )

    _require_positive_finite(
        target_half_width_bb100,
        "target_half_width_bb100",
    )

    _require_probability(
        confidence_level,
        "confidence_level",
    )

    _require_non_empty(
        estimand,
        "estimand",
    )

    _require_non_empty(
        variance_source,
        "variance_source",
    )

    normal = NormalDist()

    z_confidence = normal.inv_cdf(
        0.5
        + confidence_level / 2.0
    )

    raw_required = (
        (
            z_confidence
            * sigma_bb100
            / target_half_width_bb100
        )
        ** 2
    )

    required = max(
        1,
        math.ceil(
            raw_required
        ),
    )

    return PrecisionPlan(
        estimand=estimand,
        sigma_bb100=(
            sigma_bb100
        ),
        target_half_width_bb100=(
            target_half_width_bb100
        ),
        confidence_level=(
            confidence_level
        ),
        z_confidence=(
            z_confidence
        ),
        raw_required_independent_units=(
            raw_required
        ),
        required_independent_units=(
            required
        ),
        variance_source=(
            variance_source
        ),
    )


def recommend_fixed_sample_size(
    *,
    power_plan: PowerPlan | None = None,
    precision_plan: PrecisionPlan | None = None,
) -> SampleSizeRecommendation:
    """
    Return a fixed pre-experiment sample-size recommendation.

    If both constraints are present, the larger requirement wins.
    """
    if (
        power_plan is None
        and precision_plan is None
    ):
        raise ValueError(
            "At least one power or precision plan is required"
        )

    requirements: list[int] = []

    if power_plan is not None:
        requirements.append(
            power_plan.required_independent_units
        )

    if precision_plan is not None:
        requirements.append(
            precision_plan.required_independent_units
        )

    return SampleSizeRecommendation(
        required_independent_units=max(
            requirements
        ),
        power_plan=power_plan,
        precision_plan=precision_plan,
    )


def _require_probability(
    value: float,
    field_name: str,
) -> None:
    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise TypeError(
            f"{field_name} must be numeric"
        )

    numeric = float(
        value
    )

    if not math.isfinite(
        numeric
    ):
        raise ValueError(
            f"{field_name} must be finite"
        )

    if not (
        0.0
        < numeric
        < 1.0
    ):
        raise ValueError(
            f"{field_name} must be within (0, 1)"
        )


def _require_positive_finite(
    value: float,
    field_name: str,
) -> None:
    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise TypeError(
            f"{field_name} must be numeric"
        )

    numeric = float(
        value
    )

    if not math.isfinite(
        numeric
    ):
        raise ValueError(
            f"{field_name} must be finite"
        )

    if numeric <= 0.0:
        raise ValueError(
            f"{field_name} must be positive"
        )


def _require_non_empty(
    value: str,
    field_name: str,
) -> None:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field_name} must be a string"
        )

    if not value.strip():
        raise ValueError(
            f"{field_name} must not be empty"
        )