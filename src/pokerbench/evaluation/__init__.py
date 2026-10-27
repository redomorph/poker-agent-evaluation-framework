"""Statistical evaluation and policy-calibration utilities."""

from pokerbench.evaluation.calibration import (
    CALIBRATION_VERSION,
    DEFAULT_CALIBRATION_TOLERANCES,
    OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
    BehaviorCounts,
    CalibrationMetric,
    CalibrationMetricResult,
    CalibrationResult,
    CalibrationTolerances,
    ObservedBehavior,
    calibrate_profile,
)
from pokerbench.evaluation.opportunities import (
    ActionEffect,
    CalibrationTraceError,
    classify_action_effect,
    measure_behavior_counts,
    measure_match_behavior,
)
from pokerbench.evaluation.statistics import (
    BOOTSTRAP_PERCENTILE_METHOD,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    EVALUATION_VERSION,
    ClusterDataset,
    PairedCluster,
    StatisticalEvaluation,
    clusters_from_paired_results,
    evaluate_clusters,
    evaluate_paired_results,
)


__all__ = [
    "BOOTSTRAP_PERCENTILE_METHOD",
    "CALIBRATION_VERSION",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CALIBRATION_TOLERANCES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "EVALUATION_VERSION",
    "OPPORTUNITY_DEFINITION_CONTRACT_VERSION",
    "BehaviorCounts",
    "CalibrationTraceError",
    "ActionEffect",
    "CalibrationMetric",
    "CalibrationMetricResult",
    "CalibrationResult",
    "CalibrationTolerances",
    "ClusterDataset",
    "ObservedBehavior",
    "PairedCluster",
    "StatisticalEvaluation",
    "calibrate_profile",
    "classify_action_effect",
    "clusters_from_paired_results",
    "evaluate_clusters",
    "evaluate_paired_results",
    "measure_behavior_counts",
    "measure_match_behavior",
]