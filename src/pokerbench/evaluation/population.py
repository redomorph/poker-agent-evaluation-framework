"""Population-performance and robustness evaluation for Phase-1 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pokerbench.runner import PairedResult, StatisticalUnit


POPULATION_EVALUATION_VERSION = "population-evaluation-v1"
POPULATION_WEIGHT_SCALE = 1_000_000
POPULATION_QUANTILE_VERSION = "weighted-inverse-cdf-v1"
POPULATION_BOOTSTRAP_VERSION = "nested-profile-run-bootstrap-v1"
POPULATION_PERCENTILE_METHOD = "linear-n-minus-1-v1"
POPULATION_CONFIDENCE_LEVEL = 0.95
DEFAULT_POPULATION_BOOTSTRAP_RESAMPLES = 10_000
MIN_TAIL_PROFILE_COUNT = 10


@dataclass(frozen=True)
class DiscretePopulationDefinition:
    """Versioned finite opponent-population distribution."""

    population_id: str
    version: str
    weights_ppm: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.population_id:
            raise ValueError("population_id must not be empty")
        if not self.version:
            raise ValueError("population version must not be empty")
        if not self.weights_ppm:
            raise ValueError("weights_ppm must not be empty")

        profile_ids = tuple(profile_id for profile_id, _ in self.weights_ppm)
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("population profile ids must be unique")
        for profile_id, weight in self.weights_ppm:
            if not profile_id:
                raise ValueError("population profile id must not be empty")
            if isinstance(weight, bool) or not isinstance(weight, int):
                raise TypeError("population weights must be integers")
            if weight <= 0:
                raise ValueError("population weights must be positive")
        if sum(weight for _, weight in self.weights_ppm) != POPULATION_WEIGHT_SCALE:
            raise ValueError(
                "population weights must sum exactly to POPULATION_WEIGHT_SCALE"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "population_id": self.population_id,
            "version": self.version,
            "weight_scale": POPULATION_WEIGHT_SCALE,
            "weights_ppm": [
                {"profile_id": profile_id, "weight_ppm": weight}
                for profile_id, weight in self.weights_ppm
            ],
        }

    @property
    def population_hash(self) -> str:
        return _sha256_canonical(self.payload())


BALANCED_STATIC_POPULATION = DiscretePopulationDefinition(
    population_id="balanced-static-profiles",
    version="balanced-static-profiles-v1",
    weights_ppm=(
        ("nit", 200_000),
        ("tag", 200_000),
        ("lag", 200_000),
        ("calling_station", 200_000),
        ("maniac", 200_000),
    ),
)

BASELINE_SYNTHETIC_STATIC_POPULATION = DiscretePopulationDefinition(
    population_id="baseline-synthetic-static-profiles",
    version="baseline-synthetic-static-profiles-v1",
    weights_ppm=(
        ("nit", 100_000),
        ("tag", 350_000),
        ("lag", 300_000),
        ("calling_station", 150_000),
        ("maniac", 100_000),
    ),
)


@dataclass(frozen=True)
class ProfilePerformanceEvidence:
    """Independent run-level EV evidence for one opponent profile."""

    profile_id: str
    run_ev_bb100: tuple[float, ...]
    simulated_hands_per_run: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must not be empty")
        if not self.run_ev_bb100:
            raise ValueError("run_ev_bb100 must contain at least one run")
        if len(self.run_ev_bb100) != len(self.simulated_hands_per_run):
            raise ValueError(
                "run_ev_bb100 and simulated_hands_per_run must have equal length"
            )
        for value in self.run_ev_bb100:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("run EV values must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("run EV values must be finite")
        for hands in self.simulated_hands_per_run:
            if isinstance(hands, bool) or not isinstance(hands, int):
                raise TypeError("simulated hand counts must be integers")
            if hands <= 0:
                raise ValueError("simulated hand counts must be positive")

    @property
    def mean_ev_bb100(self) -> float:
        return statistics.fmean(self.run_ev_bb100)

    @property
    def total_simulated_hands(self) -> int:
        return sum(self.simulated_hands_per_run)

    def payload(self, *, weight_ppm: int) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "weight_ppm": weight_ppm,
            "run_ev_bb100": list(self.run_ev_bb100),
            "simulated_hands_per_run": list(self.simulated_hands_per_run),
            "mean_ev_bb100": self.mean_ev_bb100,
        }


@dataclass(frozen=True)
class PopulationProfileEvidenceSummary:
    """Immutable self-contained profile evidence embedded in a report."""

    profile_id: str
    weight_ppm: int
    run_ev_bb100: tuple[float, ...]
    simulated_hands_per_run: tuple[int, ...]
    mean_ev_bb100: float

    def payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "weight_ppm": self.weight_ppm,
            "run_ev_bb100": list(self.run_ev_bb100),
            "simulated_hands_per_run": list(self.simulated_hands_per_run),
            "mean_ev_bb100": self.mean_ev_bb100,
        }


@dataclass(frozen=True)
class PopulationEvaluation:
    """Complete population point estimates, uncertainty, and provenance."""

    population_id: str
    population_version: str
    population_hash: str
    evidence_hash: str
    profile_evidence: tuple[PopulationProfileEvidenceSummary, ...]
    profile_count: int
    run_count: int
    total_simulated_hands: int
    mean_ev_bb100: float
    mean_ci_low_bb100: float
    mean_ci_high_bb100: float
    median_ev_bb100: float
    r10_bb100: float
    r10_ci_low_bb100: float
    r10_ci_high_bb100: float
    q25_bb100: float
    stddev_bb100: float
    empirical_min_bb100: float
    tail_precision_warning: bool
    bootstrap_seed: int
    bootstrap_resamples: int
    evaluation_version: str = POPULATION_EVALUATION_VERSION
    quantile_version: str = POPULATION_QUANTILE_VERSION
    bootstrap_version: str = POPULATION_BOOTSTRAP_VERSION
    percentile_method: str = POPULATION_PERCENTILE_METHOD
    confidence_level: float = POPULATION_CONFIDENCE_LEVEL

    def payload(self) -> dict[str, Any]:
        return {
            "evaluation_version": self.evaluation_version,
            "quantile_version": self.quantile_version,
            "bootstrap_version": self.bootstrap_version,
            "percentile_method": self.percentile_method,
            "confidence_level": self.confidence_level,
            "population_id": self.population_id,
            "population_version": self.population_version,
            "population_hash": self.population_hash,
            "evidence_hash": self.evidence_hash,
            "profile_evidence": [
                item.payload()
                for item in self.profile_evidence
            ],
            "profile_count": self.profile_count,
            "run_count": self.run_count,
            "total_simulated_hands": self.total_simulated_hands,
            "mean_ev_bb100": self.mean_ev_bb100,
            "mean_ci_low_bb100": self.mean_ci_low_bb100,
            "mean_ci_high_bb100": self.mean_ci_high_bb100,
            "median_ev_bb100": self.median_ev_bb100,
            "r10_bb100": self.r10_bb100,
            "r10_ci_low_bb100": self.r10_ci_low_bb100,
            "r10_ci_high_bb100": self.r10_ci_high_bb100,
            "q25_bb100": self.q25_bb100,
            "stddev_bb100": self.stddev_bb100,
            "empirical_min_bb100": self.empirical_min_bb100,
            "tail_precision_warning": self.tail_precision_warning,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_resamples": self.bootstrap_resamples,
        }

    @property
    def report_hash(self) -> str:
        return _sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.payload(),
            "report_hash": self.report_hash,
        }


def profile_performance_from_paired_results(
    profile_id: str,
    results: Sequence[PairedResult],
) -> ProfilePerformanceEvidence:
    """Convert complete static paired results to player-A run-level EV/100."""
    if not profile_id:
        raise ValueError("profile_id must not be empty")
    if not results:
        raise ValueError("results must contain at least one PairedResult")

    run_ev_bb100: list[float] = []
    simulated_hands_per_run: list[int] = []

    for result in results:
        if not isinstance(result, PairedResult):
            raise TypeError("results must contain only PairedResult values")
        if result.statistical_unit is not StatisticalUnit.PAIRED_DEAL:
            raise ValueError(
                "Population conversion requires paired_deal statistical units"
            )
        if result.num_hands <= 0:
            raise ValueError("PairedResult num_hands must be positive")
        if (
            result.failed_runs
            or result.num_valid_pairs != result.num_hands
            or len(result.paired_differences) != result.num_hands
        ):
            raise ValueError(
                "Population evidence requires complete PairedResult objects"
            )
        if not all(math.isfinite(value) for value in result.paired_differences):
            raise ValueError("paired differences must be finite")

        mean_pairwise_difference_bb = statistics.fmean(result.paired_differences)
        # In the frozen zero-rake HU game, d = profit_A - profit_B = 2*profit_A.
        run_ev_bb100.append(50.0 * mean_pairwise_difference_bb)
        simulated_hands_per_run.append(2 * result.num_hands)

    return ProfilePerformanceEvidence(
        profile_id=profile_id,
        run_ev_bb100=tuple(run_ev_bb100),
        simulated_hands_per_run=tuple(simulated_hands_per_run),
    )


def evaluate_population(
    population: DiscretePopulationDefinition,
    evidence: Sequence[ProfilePerformanceEvidence],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int = DEFAULT_POPULATION_BOOTSTRAP_RESAMPLES,
) -> PopulationEvaluation:
    """Evaluate weighted population performance and robustness."""
    if not isinstance(population, DiscretePopulationDefinition):
        raise TypeError("population must be a DiscretePopulationDefinition")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise TypeError("bootstrap_seed must be an integer")
    if isinstance(bootstrap_resamples, bool) or not isinstance(
        bootstrap_resamples, int
    ):
        raise TypeError("bootstrap_resamples must be an integer")
    if bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be at least 2")

    by_profile: dict[str, ProfilePerformanceEvidence] = {}
    for item in evidence:
        if not isinstance(item, ProfilePerformanceEvidence):
            raise TypeError("evidence must contain ProfilePerformanceEvidence values")
        if item.profile_id in by_profile:
            raise ValueError("duplicate profile evidence is not permitted")
        by_profile[item.profile_id] = item

    expected_ids = tuple(profile_id for profile_id, _ in population.weights_ppm)
    if set(by_profile) != set(expected_ids):
        raise ValueError(
            "Population evidence profile ids must exactly match the population"
        )

    ordered_evidence = tuple(by_profile[profile_id] for profile_id in expected_ids)
    profile_means = tuple(item.mean_ev_bb100 for item in ordered_evidence)
    weights = tuple(weight for _, weight in population.weights_ppm)

    mean_ev = _weighted_mean(profile_means, weights)
    median_ev = _weighted_quantile(profile_means, weights, quantile_ppm=500_000)
    r10 = _weighted_quantile(profile_means, weights, quantile_ppm=100_000)
    q25 = _weighted_quantile(profile_means, weights, quantile_ppm=250_000)
    stddev = math.sqrt(
        sum(
            weight * (value - mean_ev) ** 2
            for value, weight in zip(profile_means, weights, strict=True)
        )
        / POPULATION_WEIGHT_SCALE
    )

    bootstrap_means, bootstrap_r10 = _nested_bootstrap(
        ordered_evidence=ordered_evidence,
        weights=weights,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    alpha = 1.0 - POPULATION_CONFIDENCE_LEVEL
    mean_ci = _percentile_interval(
        bootstrap_means,
        lower_probability=alpha / 2.0,
        upper_probability=1.0 - alpha / 2.0,
    )
    r10_ci = _percentile_interval(
        bootstrap_r10,
        lower_probability=alpha / 2.0,
        upper_probability=1.0 - alpha / 2.0,
    )

    profile_summaries = tuple(
        PopulationProfileEvidenceSummary(
            profile_id=item.profile_id,
            weight_ppm=weight,
            run_ev_bb100=tuple(float(value) for value in item.run_ev_bb100),
            simulated_hands_per_run=item.simulated_hands_per_run,
            mean_ev_bb100=item.mean_ev_bb100,
        )
        for item, weight in zip(ordered_evidence, weights, strict=True)
    )
    evidence_payload = {
        "population": population.payload(),
        "profile_evidence": [
            item.payload()
            for item in profile_summaries
        ],
    }
    evidence_hash = _sha256_canonical(evidence_payload)

    return PopulationEvaluation(
        population_id=population.population_id,
        population_version=population.version,
        population_hash=population.population_hash,
        evidence_hash=evidence_hash,
        profile_evidence=profile_summaries,
        profile_count=len(ordered_evidence),
        run_count=sum(len(item.run_ev_bb100) for item in ordered_evidence),
        total_simulated_hands=sum(
            item.total_simulated_hands for item in ordered_evidence
        ),
        mean_ev_bb100=mean_ev,
        mean_ci_low_bb100=mean_ci[0],
        mean_ci_high_bb100=mean_ci[1],
        median_ev_bb100=median_ev,
        r10_bb100=r10,
        r10_ci_low_bb100=r10_ci[0],
        r10_ci_high_bb100=r10_ci[1],
        q25_bb100=q25,
        stddev_bb100=stddev,
        empirical_min_bb100=min(profile_means),
        tail_precision_warning=len(ordered_evidence) < MIN_TAIL_PROFILE_COUNT,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )


def save_population_evaluation_report(
    evaluation: PopulationEvaluation,
    path: str | Path,
) -> Path:
    """Atomically save one canonical population-evaluation report."""
    if not isinstance(evaluation, PopulationEvaluation):
        raise TypeError("evaluation must be a PopulationEvaluation")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    serialized = _canonical_json(evaluation.to_dict()) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    return sum(
        value * weight
        for value, weight in zip(values, weights, strict=True)
    ) / POPULATION_WEIGHT_SCALE


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[int],
    *,
    quantile_ppm: int,
) -> float:
    if not 0 < quantile_ppm <= POPULATION_WEIGHT_SCALE:
        raise ValueError("quantile_ppm must be within (0, POPULATION_WEIGHT_SCALE]")
    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= quantile_ppm:
            return float(value)
    raise RuntimeError("population weights did not reach the requested quantile")


def _draw_profile_index(rng: random.Random, weights: Sequence[int]) -> int:
    ticket = rng.randrange(POPULATION_WEIGHT_SCALE)
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if ticket < cumulative:
            return index
    raise RuntimeError("population weights did not cover RNG ticket")


def _sample_inverse_cdf(values: Sequence[float], *, quantile_ppm: int) -> float:
    ordered = sorted(values)
    numerator = quantile_ppm * len(ordered)
    rank = max(
        1,
        (numerator + POPULATION_WEIGHT_SCALE - 1) // POPULATION_WEIGHT_SCALE,
    )
    return float(ordered[rank - 1])


def _nested_bootstrap(
    *,
    ordered_evidence: Sequence[ProfilePerformanceEvidence],
    weights: Sequence[int],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> tuple[list[float], list[float]]:
    rng = random.Random(bootstrap_seed)
    sample_size = len(ordered_evidence)
    means: list[float] = []
    r10_values: list[float] = []

    for _ in range(bootstrap_resamples):
        sampled: list[float] = []
        for _ in range(sample_size):
            profile_index = _draw_profile_index(rng, weights)
            profile = ordered_evidence[profile_index]
            run_index = rng.randrange(len(profile.run_ev_bb100))
            sampled.append(float(profile.run_ev_bb100[run_index]))
        means.append(statistics.fmean(sampled))
        r10_values.append(
            _sample_inverse_cdf(sampled, quantile_ppm=100_000)
        )

    means.sort()
    r10_values.sort()
    return means, r10_values


def _percentile_interval(
    sorted_values: Sequence[float],
    *,
    lower_probability: float,
    upper_probability: float,
) -> tuple[float, float]:
    return (
        _linear_percentile(sorted_values, probability=lower_probability),
        _linear_percentile(sorted_values, probability=upper_probability),
    )


def _linear_percentile(
    sorted_values: Sequence[float],
    *,
    probability: float,
) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be within [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    lower = float(sorted_values[lower_index])
    upper = float(sorted_values[upper_index])
    return lower + fraction * (upper - lower)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
