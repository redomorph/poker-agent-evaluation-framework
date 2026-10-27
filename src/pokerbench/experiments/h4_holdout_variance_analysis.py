"""Variance-only analysis primitives for the frozen Spec-37 campaign."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from collections.abc import Mapping

from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4_HOLDOUT_UNIT_IDS, H4_HOLDOUT_UNIT_WEIGHT, H4_VARIANCE_MASTER_SEEDS,
)

H4_VARIANCE_ESTIMATOR_ID = "sample-sd-matched-population-trajectory-contrast-bb100-ddof1-v1"
H4_MINIMUM_RELEVANT_DIFFERENCE_BB100 = 50.0
H4_PRECISION_TARGET_HALF_WIDTH_BB100 = 50.0
H4_Z_0_975 = 1.9599639845400536
H4_Z_0_80 = 0.8416212335729143
H4_MINIMUM_FINAL_CLUSTERS = 10

class H4HoldoutVarianceAnalysisError(ValueError):
    """Raised when incomplete or non-frozen evidence is offered for planning."""

@dataclass(frozen=True)
class H4FinalSampleSizePlan:
    sigma_500: float
    n_power_raw: float
    n_power_ceil: int
    n_precision_raw: float
    n_precision_ceil: int
    n_final: int
    planning_half_width_bb100: float
    variance_estimator: str = H4_VARIANCE_ESTIMATOR_ID

def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise H4HoldoutVarianceAnalysisError(f"{name} must be finite")
    return result

def position_compensated_bb100(btn_gain_bb: float, bb_gain_bb: float, *, hands_per_position: int = 500) -> float:
    if hands_per_position != 500:
        raise H4HoldoutVarianceAnalysisError("Spec 37 requires exactly 500 hands per position")
    return 100.0 * (_finite(btn_gain_bb, "btn_gain_bb") + _finite(bb_gain_bb, "bb_gain_bb")) / (2 * hands_per_position)

def matched_unit_contrast_bb100(*, control_btn_gain_bb: float, control_bb_gain_bb: float, treatment_btn_gain_bb: float, treatment_bb_gain_bb: float) -> float:
    return position_compensated_bb100(treatment_btn_gain_bb, treatment_bb_gain_bb) - position_compensated_bb100(control_btn_gain_bb, control_bb_gain_bb)

def population_contrast_for_seed(unit_contrasts: Mapping[str, float]) -> float:
    if set(unit_contrasts) != set(H4_HOLDOUT_UNIT_IDS):
        raise H4HoldoutVarianceAnalysisError("population contrast requires all five frozen holdout units and no others")
    return sum(H4_HOLDOUT_UNIT_WEIGHT * _finite(unit_contrasts[unit], unit) for unit in H4_HOLDOUT_UNIT_IDS)

def sigma_500_from_population_contrasts(seed_contrasts: Mapping[int, float]) -> float:
    if set(seed_contrasts) != set(H4_VARIANCE_MASTER_SEEDS):
        raise H4HoldoutVarianceAnalysisError("sigma_500 requires exactly the ten frozen Spec-37 seeds")
    ordered = [_finite(seed_contrasts[seed], f"seed {seed}") for seed in H4_VARIANCE_MASTER_SEEDS]
    return statistics.stdev(ordered)

def plan_final_h4_sample_size(sigma_500: float) -> H4FinalSampleSizePlan:
    sigma = _finite(sigma_500, "sigma_500")
    if sigma < 0.0:
        raise H4HoldoutVarianceAnalysisError("sigma_500 must be non-negative")
    n_power_raw = ((H4_Z_0_975 + H4_Z_0_80) ** 2 * sigma**2) / H4_MINIMUM_RELEVANT_DIFFERENCE_BB100**2
    n_precision_raw = ((H4_Z_0_975 * sigma) / H4_PRECISION_TARGET_HALF_WIDTH_BB100) ** 2
    n_power_ceil = math.ceil(n_power_raw)
    n_precision_ceil = math.ceil(n_precision_raw)
    n_final = max(H4_MINIMUM_FINAL_CLUSTERS, n_power_ceil, n_precision_ceil)
    half_width = H4_Z_0_975 * sigma / math.sqrt(n_final)
    return H4FinalSampleSizePlan(sigma, n_power_raw, n_power_ceil, n_precision_raw, n_precision_ceil, n_final, half_width)

def variance_pilot_plan_from_complete_seed_contrasts(seed_contrasts: Mapping[int, float]) -> H4FinalSampleSizePlan:
    """Admit only a complete 10/10 variance campaign; observed mean is never an input."""
    return plan_final_h4_sample_size(sigma_500_from_population_contrasts(seed_contrasts))
