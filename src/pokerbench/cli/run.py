"""Public single-match execution command for frozen static profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pokerbench.agents.parametric_policy import ParametricPolicyAgent
from pokerbench.agents.profiles import ProfileName, get_static_profile
from pokerbench.agents.strength_gates import (
    PolicyStrengthReference,
    build_profile_calibration_map,
)
from pokerbench.engine.config import BASELINE_GAME_CONFIG
from pokerbench.evaluation.calibration_snapshots import (
    CalibrationStrengthSnapshotStore,
    policy_strength_reference_from_store,
)
from pokerbench.evaluation.gate6_execution import (
    Gate6CalibrationReport,
    load_gate6_calibration_report,
)
from pokerbench.persistence import SQLiteBenchmarkStore
from pokerbench.runner import PairedResult, PairedRunner

PUBLIC_RUN_CLI_VERSION = "public-run-cli-v1"


class PublicRunError(ValueError):
    """Raised when a requested public run is not scientifically admissible."""


@dataclass(frozen=True)
class VerifiedStaticProfileContext:
    """Gate-6 evidence shared by public commands using static profiles."""

    reference: PolicyStrengthReference
    gate6_report: Gate6CalibrationReport


@dataclass(frozen=True)
class PublicRunResult:
    result: PairedResult
    bot_a: str
    bot_b: str
    database: Path
    gate6_report_hash: str
    cli_version: str = PUBLIC_RUN_CLI_VERSION


def load_verified_static_profile_context(
    *,
    snapshot_root: Path,
    gate6_report_path: Path,
) -> VerifiedStaticProfileContext:
    """Load and verify the frozen FIT reference against a Gate-6 PASS report."""
    gate6_report = load_gate6_calibration_report(gate6_report_path)
    if not gate6_report.passed:
        raise PublicRunError("Gate-6 calibration report is not a scientific PASS")

    snapshot_store = CalibrationStrengthSnapshotStore(snapshot_root)
    reference = policy_strength_reference_from_store(snapshot_store)
    if reference.reference_strength_hash != gate6_report.reference_strength_hash:
        raise PublicRunError(
            "FIT snapshots do not match the Gate-6 calibration report"
        )

    return VerifiedStaticProfileContext(
        reference=reference,
        gate6_report=gate6_report,
    )


def build_verified_static_profile_agent(
    profile_name: str,
    *,
    context: VerifiedStaticProfileContext,
) -> ParametricPolicyAgent:
    """Construct one static profile only when its Gate-6 gate map matches."""
    try:
        profile_enum = ProfileName(profile_name)
    except ValueError as exc:
        raise PublicRunError(f"Unknown static profile: {profile_name}") from exc

    profile = get_static_profile(profile_enum)
    calibration_map = build_profile_calibration_map(profile, context.reference)

    acceptance = next(
        (
            item
            for item in context.gate6_report.gate6_result.results
            if item.profile_name == profile.name.value
        ),
        None,
    )
    if acceptance is None:
        raise PublicRunError(
            f"Gate-6 report does not contain profile {profile.name.value}"
        )
    if acceptance.gate_map_hash != calibration_map.gate_map_hash:
        raise PublicRunError(
            f"Gate map for {profile.name.value} does not match Gate-6 report"
        )
    if acceptance.reference_strength_hash != context.reference.reference_strength_hash:
        raise PublicRunError(
            "Calibration snapshot reference does not match Gate-6 report"
        )

    return ParametricPolicyAgent(calibration_map)


def execute_public_run(
    *,
    bot_a: str,
    bot_b: str,
    seed: int,
    hands: int,
    snapshot_root: Path,
    gate6_report_path: Path,
    database: Path,
    match_id: str | None = None,
) -> PublicRunResult:
    """Execute and persist one paired static-profile match."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(hands, bool) or not isinstance(hands, int):
        raise TypeError("hands must be an integer")
    if hands <= 0:
        raise ValueError("hands must be positive")

    context = load_verified_static_profile_context(
        snapshot_root=snapshot_root,
        gate6_report_path=gate6_report_path,
    )

    agent_a = build_verified_static_profile_agent(bot_a, context=context)
    agent_b = build_verified_static_profile_agent(bot_b, context=context)

    benchmark_store = SQLiteBenchmarkStore(database)
    benchmark_store.initialize()

    gate6_report = context.gate6_report
    runner = PairedRunner(
        agent_a=agent_a,
        agent_b=agent_b,
        num_hands=hands,
        seed=seed,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
        match_id=match_id,
        benchmark_store=benchmark_store,
        experiment_metadata={
            "experiment_id": (
                f"{PUBLIC_RUN_CLI_VERSION}:{bot_a}:{bot_b}:"
                f"{gate6_report.report_hash}"
            ),
            "public_run_cli_version": PUBLIC_RUN_CLI_VERSION,
            "bot_a_profile": bot_a,
            "bot_b_profile": bot_b,
            "gate6_report_hash": gate6_report.report_hash,
        },
    )
    result = runner.run()

    return PublicRunResult(
        result=result,
        bot_a=bot_a,
        bot_b=bot_b,
        database=Path(database),
        gate6_report_hash=gate6_report.report_hash,
    )


def render_public_run_text(run: PublicRunResult) -> str:
    result = run.result
    match_id = (
        result.match_record.match_id
        if result.match_record is not None
        else str(result.experiment_metadata.get("match_id", "N/A"))
    )
    lines = [
        "PokerBench run complete",
        f"  match id: {match_id}",
        f"  bot A: {run.bot_a}",
        f"  bot B: {run.bot_b}",
        f"  seed: {result.seed}",
        f"  requested paired deals: {result.num_hands}",
        f"  valid statistical units: {result.num_valid_pairs}",
        f"  statistical unit: {result.statistical_unit.value}",
        f"  failed runs: {len(result.failed_runs)}",
        f"  database: {run.database}",
        f"  gate6 report hash: {run.gate6_report_hash}",
    ]
    if result.mean_d is not None:
        lines.append(f"  mean paired difference: {result.mean_d:.6f} BB")
    if result.ci_low is not None and result.ci_high is not None:
        lines.append(
            "  normal 95% CI (paired difference): "
            f"[{result.ci_low:.6f}, {result.ci_high:.6f}] BB"
        )
    return "\n".join(lines)
