"""Public multi-seed battle command and reproducible statistical report."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokerbench.agents.base import Agent
from pokerbench.agents.profiles import STATIC_PROFILE_VERSION
from pokerbench.agents.random_baseline import (
    RANDOM_BASELINE_VERSION,
    RandomBaselineAgent,
)
from pokerbench.agents.strength_gates import PARAMETRIC_POLICY_VERSION
from pokerbench.cli.run import (
    VerifiedStaticProfileContext,
    build_verified_static_profile_agent,
    load_verified_static_profile_context,
)
from pokerbench.engine.config import BASELINE_GAME_CONFIG, config_hash
from pokerbench.evaluation.statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    EVALUATION_VERSION,
    StatisticalEvaluation,
    evaluate_paired_results,
)
from pokerbench.persistence import SQLiteBenchmarkStore
from pokerbench.runner import (
    IllegalActionPolicy,
    PairedResult,
    PairedRunner,
    StatisticalUnit,
)


PUBLIC_BATTLE_CLI_VERSION = "public-battle-cli-v1"
PUBLIC_BATTLE_CONFIDENCE_LEVEL = 0.95
PUBLIC_BATTLE_DEFAULT_DATABASE = Path("data/benchmark.sqlite")


class PublicBattleError(ValueError):
    """Raised when a requested public battle is not scientifically admissible."""


@dataclass(frozen=True)
class PublicBattleReport:
    bot_a: str
    bot_b: str
    bot_a_version: str
    bot_b_version: str
    seed_start: int
    seed_count: int
    hands_per_seed: int
    seeds: tuple[int, ...]
    requested_independent_units: int
    valid_independent_units: int
    requested_engine_hands: int
    analyzed_engine_hands: int
    failed_runs: int
    match_ids: tuple[str, ...]
    statistical_unit: str
    game_config_hash: str
    gate6_report_hash: str | None
    bootstrap_seed: int
    evaluation: StatisticalEvaluation | None
    complete: bool
    battle_cli_version: str = PUBLIC_BATTLE_CLI_VERSION
    random_baseline_version: str = RANDOM_BASELINE_VERSION
    static_profile_version: str = STATIC_PROFILE_VERSION
    parametric_policy_version: str = PARAMETRIC_POLICY_VERSION
    evaluation_version: str = EVALUATION_VERSION
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    confidence_level: float = PUBLIC_BATTLE_CONFIDENCE_LEVEL
    failure_policy: str = IllegalActionPolicy.EXCLUDE_RUN.value

    def __post_init__(self) -> None:
        if self.seed_count <= 0:
            raise ValueError("seed_count must be positive")
        if self.hands_per_seed <= 0:
            raise ValueError("hands_per_seed must be positive")
        if self.seed_start < 0:
            raise ValueError("seed_start must be non-negative")
        if len(self.seeds) != self.seed_count:
            raise ValueError("seeds length must equal seed_count")
        if self.requested_independent_units != self.seed_count * self.hands_per_seed:
            raise ValueError("requested_independent_units is inconsistent")
        if self.requested_engine_hands != 2 * self.requested_independent_units:
            raise ValueError("requested_engine_hands is inconsistent")
        if self.valid_independent_units < 0:
            raise ValueError("valid_independent_units must be non-negative")
        if self.failed_runs < 0:
            raise ValueError("failed_runs must be non-negative")
        if self.statistical_unit != StatisticalUnit.PAIRED_DEAL.value:
            raise ValueError("public-battle-cli-v1 requires paired_deal units")
        if self.complete and self.valid_independent_units != self.requested_independent_units:
            raise ValueError("complete report must contain every requested unit")
        if self.complete and self.evaluation is None:
            raise ValueError("complete report requires a statistical evaluation")

    def _evaluation_payload(self) -> dict[str, Any] | None:
        evaluation = self.evaluation
        if evaluation is None:
            return None
        return {
            "statistical_unit": evaluation.statistical_unit.value,
            "num_independent_units": evaluation.num_independent_units,
            "num_analyzed_simulated_hands": evaluation.num_analyzed_simulated_hands,
            "num_failed_runs": evaluation.num_failed_runs,
            "mean_difference_bb100": evaluation.mean_difference_bb100,
            "std_difference_bb100": evaluation.std_difference_bb100,
            "normal_ci_low_bb100": evaluation.normal_ci_low_bb100,
            "normal_ci_high_bb100": evaluation.normal_ci_high_bb100,
            "bootstrap_ci_low_bb100": evaluation.bootstrap_ci_low_bb100,
            "bootstrap_ci_high_bb100": evaluation.bootstrap_ci_high_bb100,
            "effect_size_dz": evaluation.effect_size_dz,
            "confidence_level": evaluation.confidence_level,
            "bootstrap_resamples": evaluation.bootstrap_resamples,
            "bootstrap_seed": evaluation.bootstrap_seed,
            "evaluation_version": evaluation.evaluation_version,
            "percentile_method": evaluation.percentile_method,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "battle_cli_version": self.battle_cli_version,
            "random_baseline_version": self.random_baseline_version,
            "static_profile_version": self.static_profile_version,
            "parametric_policy_version": self.parametric_policy_version,
            "evaluation_version": self.evaluation_version,
            "bot_a": self.bot_a,
            "bot_b": self.bot_b,
            "bot_a_version": self.bot_a_version,
            "bot_b_version": self.bot_b_version,
            "seed_start": self.seed_start,
            "seed_count": self.seed_count,
            "hands_per_seed": self.hands_per_seed,
            "seeds": list(self.seeds),
            "requested_independent_units": self.requested_independent_units,
            "valid_independent_units": self.valid_independent_units,
            "requested_engine_hands": self.requested_engine_hands,
            "analyzed_engine_hands": self.analyzed_engine_hands,
            "failed_runs": self.failed_runs,
            "failure_policy": self.failure_policy,
            "match_ids": list(self.match_ids),
            "statistical_unit": self.statistical_unit,
            "game_config_hash": self.game_config_hash,
            "gate6_report_hash": self.gate6_report_hash,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_resamples": self.bootstrap_resamples,
            "confidence_level": self.confidence_level,
            "complete": self.complete,
            "evaluation": self._evaluation_payload(),
        }

    @property
    def report_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.payload(),
            "report_hash": self.report_hash,
        }


@dataclass(frozen=True)
class PublicBattleResult:
    report: PublicBattleReport
    results: tuple[PairedResult, ...]
    database: Path
    report_path: Path | None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _bot_version(bot: str) -> str:
    if bot == "random":
        return RANDOM_BASELINE_VERSION
    return f"{STATIC_PROFILE_VERSION}+{PARAMETRIC_POLICY_VERSION}"


def _build_battle_agent(
    bot: str,
    *,
    context: VerifiedStaticProfileContext | None,
) -> Agent:
    if bot == "random":
        return RandomBaselineAgent()
    if context is None:
        raise PublicBattleError("Static profile requires verified Gate-6 context")
    return build_verified_static_profile_agent(bot, context=context)


def _validate_battle_inputs(
    *,
    seed_start: int,
    seed_count: int,
    hands: int,
    bootstrap_seed: int,
) -> None:
    for name, value in (
        ("seed_start", seed_start),
        ("seed_count", seed_count),
        ("hands", hands),
        ("bootstrap_seed", bootstrap_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if hands <= 0:
        raise ValueError("hands must be positive")
    if seed_count * hands < 2:
        raise ValueError(
            "poker battle requires at least two requested paired-deal units"
        )


def execute_public_battle(
    *,
    bot_a: str,
    bot_b: str,
    seed_count: int,
    seed_start: int,
    hands: int,
    snapshot_root: Path,
    gate6_report_path: Path,
    database: Path,
    bootstrap_seed: int,
    report_path: Path | None = None,
) -> PublicBattleResult:
    """Execute, persist, and statistically aggregate a deterministic seed matrix."""
    _validate_battle_inputs(
        seed_start=seed_start,
        seed_count=seed_count,
        hands=hands,
        bootstrap_seed=bootstrap_seed,
    )

    allowed = {"random", "nit", "tag", "lag", "calling_station", "maniac"}
    if bot_a not in allowed:
        raise PublicBattleError(f"Unknown battle bot: {bot_a}")
    if bot_b not in allowed:
        raise PublicBattleError(f"Unknown battle bot: {bot_b}")

    context: VerifiedStaticProfileContext | None = None
    if bot_a != "random" or bot_b != "random":
        context = load_verified_static_profile_context(
            snapshot_root=snapshot_root,
            gate6_report_path=gate6_report_path,
        )

    gate6_report_hash = (
        context.gate6_report.report_hash
        if context is not None
        else None
    )

    benchmark_store = SQLiteBenchmarkStore(database)
    benchmark_store.initialize()

    seeds = tuple(seed_start + offset for offset in range(seed_count))
    results: list[PairedResult] = []

    for seed in seeds:
        agent_a = _build_battle_agent(bot_a, context=context)
        agent_b = _build_battle_agent(bot_b, context=context)
        gate6_identity = gate6_report_hash if gate6_report_hash is not None else "none"
        experiment_id = (
            f"{PUBLIC_BATTLE_CLI_VERSION}:{bot_a}:{bot_b}:"
            f"seed={seed}:hands={hands}:gate6={gate6_identity}"
        )

        runner = PairedRunner(
            agent_a=agent_a,
            agent_b=agent_b,
            num_hands=hands,
            seed=seed,
            game_config=BASELINE_GAME_CONFIG,
            adaptive=False,
            illegal_action_policy=IllegalActionPolicy.EXCLUDE_RUN,
            benchmark_store=benchmark_store,
            experiment_metadata={
                "experiment_id": experiment_id,
                "public_battle_cli_version": PUBLIC_BATTLE_CLI_VERSION,
                "bot_a": bot_a,
                "bot_b": bot_b,
                "bot_a_version": _bot_version(bot_a),
                "bot_b_version": _bot_version(bot_b),
                "seed_job": seed,
                "hands_per_seed": hands,
                "gate6_report_hash": gate6_report_hash,
            },
        )
        results.append(runner.run())

    result_tuple = tuple(results)
    valid_units = sum(result.num_valid_pairs for result in result_tuple)
    failed_runs = sum(len(result.failed_runs) for result in result_tuple)
    requested_units = seed_count * hands

    evaluation: StatisticalEvaluation | None = None
    if valid_units >= 2:
        evaluation = evaluate_paired_results(
            result_tuple,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            confidence_level=PUBLIC_BATTLE_CONFIDENCE_LEVEL,
        )

    analyzed_engine_hands = (
        evaluation.num_analyzed_simulated_hands
        if evaluation is not None
        else 2 * valid_units
    )
    match_ids = tuple(
        result.match_record.match_id
        for result in result_tuple
        if result.match_record is not None
    )
    complete = valid_units == requested_units and evaluation is not None

    report = PublicBattleReport(
        bot_a=bot_a,
        bot_b=bot_b,
        bot_a_version=_bot_version(bot_a),
        bot_b_version=_bot_version(bot_b),
        seed_start=seed_start,
        seed_count=seed_count,
        hands_per_seed=hands,
        seeds=seeds,
        requested_independent_units=requested_units,
        valid_independent_units=valid_units,
        requested_engine_hands=2 * requested_units,
        analyzed_engine_hands=analyzed_engine_hands,
        failed_runs=failed_runs,
        match_ids=match_ids,
        statistical_unit=StatisticalUnit.PAIRED_DEAL.value,
        game_config_hash=config_hash(BASELINE_GAME_CONFIG),
        gate6_report_hash=gate6_report_hash,
        bootstrap_seed=bootstrap_seed,
        evaluation=evaluation,
        complete=complete,
    )

    destination: Path | None = None
    if report_path is not None:
        destination = save_public_battle_report(report, report_path)

    return PublicBattleResult(
        report=report,
        results=result_tuple,
        database=Path(database),
        report_path=destination,
    )


def save_public_battle_report(report: PublicBattleReport, path: str | Path) -> Path:
    """Atomically persist one canonical battle report."""
    if not isinstance(report, PublicBattleReport):
        raise TypeError("report must be a PublicBattleReport")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    serialized = _canonical_json(report.to_dict()) + "\n"
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


def render_public_battle_text(result: PublicBattleResult) -> str:
    report = result.report
    lines = [
        "PokerBench battle report",
        f"  status: {'COMPLETE' if report.complete else 'INCOMPLETE'}",
        f"  bot A: {report.bot_a} ({report.bot_a_version})",
        f"  bot B: {report.bot_b} ({report.bot_b_version})",
        f"  seed jobs: {report.seed_count}",
        f"  seed start: {report.seed_start}",
        f"  hands per seed: {report.hands_per_seed}",
        f"  requested paired-deal units: {report.requested_independent_units}",
        f"  valid independent units: {report.valid_independent_units}",
        f"  requested engine hands: {report.requested_engine_hands}",
        f"  analyzed engine hands: {report.analyzed_engine_hands}",
        f"  statistical unit: {report.statistical_unit}",
        f"  failed runs: {report.failed_runs}",
        f"  failure policy: {report.failure_policy}",
        f"  database: {result.database}",
        f"  game config hash: {report.game_config_hash}",
        f"  gate6 report hash: {report.gate6_report_hash or 'N/A'}",
        f"  evaluation version: {report.evaluation_version}",
        f"  battle CLI version: {report.battle_cli_version}",
    ]

    evaluation = report.evaluation
    if evaluation is None:
        lines.append("  statistics: unavailable (<2 valid independent units)")
    else:
        lines.extend(
            [
                f"  mean paired difference: {evaluation.mean_difference_bb100:.6f} BB/100",
                (
                    "  normal 95% CI: "
                    f"[{evaluation.normal_ci_low_bb100:.6f}, "
                    f"{evaluation.normal_ci_high_bb100:.6f}] BB/100"
                ),
                (
                    "  bootstrap 95% CI: "
                    f"[{evaluation.bootstrap_ci_low_bb100:.6f}, "
                    f"{evaluation.bootstrap_ci_high_bb100:.6f}] BB/100"
                ),
                (
                    "  effect size d_z: N/A"
                    if evaluation.effect_size_dz is None
                    else f"  effect size d_z: {evaluation.effect_size_dz:.6f}"
                ),
                f"  bootstrap resamples: {evaluation.bootstrap_resamples}",
                f"  bootstrap seed: {evaluation.bootstrap_seed}",
            ]
        )

    if result.report_path is not None:
        lines.append(f"  JSON report: {result.report_path}")
    lines.append(f"  report hash: {report.report_hash}")
    return "\n".join(lines)


def public_battle_exit_code(result: PublicBattleResult) -> int:
    return 0 if result.report.complete else 1
