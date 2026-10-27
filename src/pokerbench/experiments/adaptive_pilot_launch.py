"""Launch-readiness and real-worker wiring for the frozen Phase-2 adaptive pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokerbench.agents.opponent_range import PreflopStrengthReference, load_preflop_strength_reference
from pokerbench.cli.run import (
    VerifiedStaticProfileContext,
    build_verified_static_profile_agent,
    load_verified_static_profile_context,
)
from pokerbench.engine.config import BASELINE_GAME_CONFIG
from pokerbench.experiments.adaptive_pilot_binding import (
    AdaptivePilotBindingError,
    AdaptivePilotExecutionBinding,
    AdaptivePilotTrajectoryConfig,
    capture_adaptive_pilot_execution_binding,
    load_adaptive_pilot_execution_binding,
)
from pokerbench.experiments.adaptive_pilot_execution_contract import (
    AdaptivePilotExecutionContract,
    AdaptivePilotExecutionContractError,
    frozen_adaptive_pilot_execution_contract,
    load_adaptive_pilot_execution_contract,
)
from pokerbench.experiments.adaptive_pilot_execution import (
    AdaptivePilotExecutionError,
    build_adaptive_pilot_hero,
    build_adaptive_pilot_runner,
)
from pokerbench.experiments.adaptive_pilot_parallel_execution_contract import (
    AdaptivePilotParallelExecutionContract,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotJob,
    AdaptivePilotProtocol,
    AdaptivePilotProtocolError,
    iter_adaptive_pilot_jobs,
    load_adaptive_pilot_protocol,
)
from pokerbench.experiments.adaptive_pilot_run import (
    AdaptivePilotRunError,
    AdaptivePilotRunReadiness,
    AdaptivePilotRunResult,
    AdaptivePilotRunStore,
    assert_adaptive_pilot_run_ready,
    execute_adaptive_pilot_run,
    initialize_adaptive_pilot_run,
    load_adaptive_pilot_run_manifest,
)
from pokerbench.persistence import SQLiteBenchmarkStore
from pokerbench.runner import PairedResult


ADAPTIVE_PILOT_LAUNCH_VERSION = "phase2-adaptive-pilot-launch-v1"
DEFAULT_ADAPTIVE_PILOT_EXECUTION_CONTRACT_PATH = Path(
    "configs/experiments/phase2-adaptive-pilot-execution-v1.json"
)


class AdaptivePilotLaunchError(ValueError):
    """Raised when the frozen adaptive pilot is not safe to launch or resume."""


@dataclass(frozen=True)
class AdaptivePilotLaunchAssets:
    """Verified immutable assets needed by the concrete per-job worker."""

    protocol: AdaptivePilotProtocol
    binding: AdaptivePilotExecutionBinding
    verified_profiles: VerifiedStaticProfileContext
    adaptive_reference: PreflopStrengthReference
    launch_version: str = ADAPTIVE_PILOT_LAUNCH_VERSION


@dataclass(frozen=True)
class AdaptivePilotLaunchStatus:
    """Non-mutating persisted run-state summary."""

    binding_hash: str
    planned_jobs: int
    planned_pairs: int
    planned_engine_hands: int
    terminal_jobs: int
    planned: int
    running: int
    completed: int
    failed: int


def _load_protocol(path: str | Path) -> AdaptivePilotProtocol:
    try:
        return load_adaptive_pilot_protocol(Path(path))
    except (OSError, ValueError, AdaptivePilotProtocolError) as exc:
        raise AdaptivePilotLaunchError(f"could not load frozen adaptive-pilot protocol: {exc}") from exc


def _load_execution_contract(path: str | Path) -> AdaptivePilotExecutionContract:
    try:
        return load_adaptive_pilot_execution_contract(Path(path))
    except (OSError, ValueError, AdaptivePilotExecutionContractError) as exc:
        raise AdaptivePilotLaunchError(f"could not load frozen execution contract: {exc}") from exc


def _load_persisted_binding(
    work_root: str | Path,
    execution_contract: AdaptivePilotExecutionContract,
) -> AdaptivePilotExecutionBinding:
    path = Path(work_root) / "binding.json"
    if not path.exists():
        raise AdaptivePilotLaunchError("adaptive pilot is not planned; binding.json is missing")
    try:
        binding = load_adaptive_pilot_execution_binding(path)
    except (OSError, ValueError, AdaptivePilotBindingError) as exc:
        raise AdaptivePilotLaunchError(f"could not load persisted execution binding: {exc}") from exc
    if (
        binding.execution_contract_id != execution_contract.execution_contract_id
        or binding.execution_contract_version != execution_contract.execution_contract_version
        or binding.execution_contract_hash != execution_contract.payload_hash
    ):
        raise AdaptivePilotLaunchError("persisted binding execution contract differs from launch contract")
    return binding


def load_adaptive_pilot_launch_assets(
    *,
    protocol_path: str | Path,
    binding: AdaptivePilotExecutionBinding,
    snapshot_root: str | Path,
    gate6_report_path: str | Path,
    reference_path: str | Path,
    execution_contract: AdaptivePilotExecutionContract | AdaptivePilotParallelExecutionContract | None = None,
) -> AdaptivePilotLaunchAssets:
    """Load exact Gate-6/profile/reference assets under the Spec-33 authorization."""
    protocol = _load_protocol(protocol_path)
    contract = execution_contract or frozen_adaptive_pilot_execution_contract()
    if contract.authorized_protocol_id != protocol.protocol_id:
        raise AdaptivePilotLaunchError("execution contract authorizes a different pilot protocol id")
    if contract.authorized_protocol_version != protocol.protocol_version:
        raise AdaptivePilotLaunchError("execution contract authorizes a different pilot protocol version")
    if contract.authorized_protocol_hash != protocol.payload_hash:
        raise AdaptivePilotLaunchError("execution contract authorizes a different pilot protocol hash")
    if binding.execution_contract_hash != contract.payload_hash:
        raise AdaptivePilotLaunchError("execution binding does not bind the supplied execution contract")
    if binding.workers != contract.workers:
        raise AdaptivePilotLaunchError("execution binding worker count differs from execution contract")
    if protocol.payload_hash != binding.pilot_protocol_hash:
        raise AdaptivePilotLaunchError("pilot protocol hash differs from persisted execution binding")
    try:
        context = load_verified_static_profile_context(
            snapshot_root=Path(snapshot_root),
            gate6_report_path=Path(gate6_report_path),
        )
        reference = load_preflop_strength_reference(Path(reference_path))
    except (OSError, ValueError) as exc:
        raise AdaptivePilotLaunchError(f"could not verify adaptive-pilot launch assets: {exc}") from exc
    if context.gate6_report.report_hash != binding.gate6_report_hash:
        raise AdaptivePilotLaunchError("Gate-6 report hash differs from persisted execution binding")
    if context.reference.reference_strength_hash != binding.gate6_reference_strength_hash:
        raise AdaptivePilotLaunchError("Gate-6 reference hash differs from persisted execution binding")

    # Constructing one frozen treatment hero validates the adaptive reference hash/semantics
    # without executing a hand or mutating adaptive state.
    treatment = next(job for job in iter_adaptive_pilot_jobs(protocol) if job.arm_id == "treatment")
    try:
        build_adaptive_pilot_hero(treatment, reference=reference)
    except (TypeError, ValueError, AdaptivePilotExecutionError) as exc:
        raise AdaptivePilotLaunchError(f"adaptive reference is not frozen-v1 compatible: {exc}") from exc
    return AdaptivePilotLaunchAssets(protocol, binding, context, reference)


def build_adaptive_pilot_job_executor(assets: AdaptivePilotLaunchAssets):
    """Return the concrete guarded worker consumed by the resumable scheduler."""
    if not isinstance(assets, AdaptivePilotLaunchAssets):
        raise TypeError("assets must be AdaptivePilotLaunchAssets")

    def execute(
        job: AdaptivePilotJob,
        config: AdaptivePilotTrajectoryConfig,
        match_id: str,
        store: AdaptivePilotRunStore,
    ) -> PairedResult:
        if config.binding != assets.binding:
            raise AdaptivePilotLaunchError("trajectory binding differs from launch binding")
        opponent = build_verified_static_profile_agent(
            job.opponent_profile_id,
            context=assets.verified_profiles,
        )
        runner = build_adaptive_pilot_runner(
            protocol=assets.protocol,
            job=job,
            reference=assets.adaptive_reference,
            opponent=opponent,
            game_config=BASELINE_GAME_CONFIG,
            match_id=match_id,
            benchmark_store=store,
            trajectory_config=config,
        )
        return runner.run()

    return execute


def status_adaptive_pilot_launch(
    *,
    protocol_path: str | Path,
    work_root: str | Path,
    execution_contract_path: str | Path = DEFAULT_ADAPTIVE_PILOT_EXECUTION_CONTRACT_PATH,
) -> AdaptivePilotLaunchStatus:
    """Read persisted plan/manifest status without recapturing the current environment."""
    protocol = _load_protocol(protocol_path)
    execution_contract = _load_execution_contract(execution_contract_path)
    binding = _load_persisted_binding(work_root, execution_contract)
    path = Path(work_root) / "run-manifest.json"
    if not path.exists():
        raise AdaptivePilotLaunchError("adaptive pilot is not planned; run-manifest.json is missing")
    try:
        manifest = load_adaptive_pilot_run_manifest(path, protocol=protocol, expected_binding=binding)
    except (OSError, ValueError, AdaptivePilotRunError) as exc:
        raise AdaptivePilotLaunchError(f"could not load adaptive-pilot run manifest: {exc}") from exc
    counts = {name: sum(item.status == name for item in manifest.jobs) for name in ("planned", "running", "completed", "failed")}
    return AdaptivePilotLaunchStatus(
        binding_hash=binding.binding_hash,
        planned_jobs=manifest.planned_job_count,
        planned_pairs=manifest.planned_pair_count,
        planned_engine_hands=manifest.planned_engine_hands,
        terminal_jobs=manifest.terminal_jobs,
        planned=counts["planned"],
        running=counts["running"],
        completed=counts["completed"],
        failed=counts["failed"],
    )


def plan_adaptive_pilot_launch(
    *,
    protocol_path: str | Path,
    primary_population_path: str | Path,
    secondary_population_path: str | Path,
    snapshot_root: str | Path,
    gate6_report_path: str | Path,
    reference_path: str | Path,
    database: str | Path,
    work_root: str | Path,
    repo_root: str | Path,
    poetry_lock_path: str | Path,
    ram_bytes: int,
    workers: int = 1,
    execution_contract_path: str | Path = DEFAULT_ADAPTIVE_PILOT_EXECUTION_CONTRACT_PATH,
) -> AdaptivePilotLaunchStatus:
    """Capture the live binding and freeze the 50-job plan without executing a hand."""
    if workers != 1:
        raise AdaptivePilotLaunchError("phase2-adaptive-pilot-launch-v1 requires workers=1")
    protocol = _load_protocol(protocol_path)
    execution_contract = _load_execution_contract(execution_contract_path)
    if execution_contract.workers != workers:
        raise AdaptivePilotLaunchError("execution contract worker count differs from launch worker count")
    try:
        binding = capture_adaptive_pilot_execution_binding(
            protocol=protocol,
            execution_contract=execution_contract,
            gate6_report_path=gate6_report_path,
            primary_population_path=primary_population_path,
            secondary_population_paths=(secondary_population_path,),
            repo_root=repo_root,
            poetry_lock_path=poetry_lock_path,
            ram_bytes=ram_bytes,
            workers=workers,
        )
        load_adaptive_pilot_launch_assets(
            protocol_path=protocol_path,
            binding=binding,
            execution_contract=execution_contract,
            snapshot_root=snapshot_root,
            gate6_report_path=gate6_report_path,
            reference_path=reference_path,
        )
        store = SQLiteBenchmarkStore(Path(database))
        store.initialize()
        initialize_adaptive_pilot_run(protocol=protocol, binding=binding, work_root=work_root)
    except (OSError, ValueError, AdaptivePilotBindingError, AdaptivePilotRunError) as exc:
        if isinstance(exc, AdaptivePilotLaunchError):
            raise
        raise AdaptivePilotLaunchError(f"could not plan adaptive pilot: {exc}") from exc
    return status_adaptive_pilot_launch(
        protocol_path=protocol_path,
        work_root=work_root,
        execution_contract_path=execution_contract_path,
    )


def _recapture_current_binding(
    *,
    protocol: AdaptivePilotProtocol,
    execution_contract: AdaptivePilotExecutionContract,
    persisted: AdaptivePilotExecutionBinding,
    gate6_report_path: str | Path,
    primary_population_path: str | Path,
    secondary_population_path: str | Path,
    repo_root: str | Path,
    poetry_lock_path: str | Path,
) -> AdaptivePilotExecutionBinding:
    try:
        return capture_adaptive_pilot_execution_binding(
            protocol=protocol,
            execution_contract=execution_contract,
            gate6_report_path=gate6_report_path,
            primary_population_path=primary_population_path,
            secondary_population_paths=(secondary_population_path,),
            repo_root=repo_root,
            poetry_lock_path=poetry_lock_path,
            ram_bytes=persisted.ram_bytes,
            workers=persisted.workers,
        )
    except (OSError, ValueError, AdaptivePilotBindingError) as exc:
        raise AdaptivePilotLaunchError(f"could not recapture current execution binding: {exc}") from exc


def assert_adaptive_pilot_launch_ready(
    *,
    protocol_path: str | Path,
    primary_population_path: str | Path,
    secondary_population_path: str | Path,
    snapshot_root: str | Path,
    gate6_report_path: str | Path,
    reference_path: str | Path,
    database: str | Path,
    work_root: str | Path,
    repo_root: str | Path,
    poetry_lock_path: str | Path,
    execution_contract_path: str | Path = DEFAULT_ADAPTIVE_PILOT_EXECUTION_CONTRACT_PATH,
) -> AdaptivePilotRunReadiness:
    """Fail closed unless current assets/environment exactly match the persisted plan."""
    protocol = _load_protocol(protocol_path)
    execution_contract = _load_execution_contract(execution_contract_path)
    persisted = _load_persisted_binding(work_root, execution_contract)
    current = _recapture_current_binding(
        protocol=protocol,
        execution_contract=execution_contract,
        persisted=persisted,
        gate6_report_path=gate6_report_path,
        primary_population_path=primary_population_path,
        secondary_population_path=secondary_population_path,
        repo_root=repo_root,
        poetry_lock_path=poetry_lock_path,
    )
    if current != persisted:
        raise AdaptivePilotLaunchError("current execution binding differs from persisted launch binding")
    load_adaptive_pilot_launch_assets(
        protocol_path=protocol_path,
        binding=persisted,
        execution_contract=execution_contract,
        snapshot_root=snapshot_root,
        gate6_report_path=gate6_report_path,
        reference_path=reference_path,
    )
    store = SQLiteBenchmarkStore(Path(database))
    try:
        readiness = assert_adaptive_pilot_run_ready(
            protocol=protocol,
            binding=persisted,
            work_root=work_root,
            benchmark_store=store,
        )
    except (OSError, ValueError, AdaptivePilotRunError) as exc:
        raise AdaptivePilotLaunchError(f"adaptive pilot is not launch-ready: {exc}") from exc
    if (readiness.planned_jobs, readiness.planned_pairs, readiness.planned_engine_hands) != (
        execution_contract.planned_jobs,
        execution_contract.planned_pairs,
        execution_contract.planned_engine_hands,
    ):
        raise AdaptivePilotLaunchError("launch readiness does not reproduce the frozen 50-job/25-pair/18,000-hand matrix")
    return readiness


def run_adaptive_pilot_launch(
    *,
    protocol_path: str | Path,
    primary_population_path: str | Path,
    secondary_population_path: str | Path,
    snapshot_root: str | Path,
    gate6_report_path: str | Path,
    reference_path: str | Path,
    database: str | Path,
    work_root: str | Path,
    repo_root: str | Path,
    poetry_lock_path: str | Path,
    resume: bool,
    execution_contract_path: str | Path = DEFAULT_ADAPTIVE_PILOT_EXECUTION_CONTRACT_PATH,
) -> AdaptivePilotRunResult:
    """Launch or resume only after full live readiness verification succeeds."""
    readiness = assert_adaptive_pilot_launch_ready(
        protocol_path=protocol_path,
        primary_population_path=primary_population_path,
        secondary_population_path=secondary_population_path,
        snapshot_root=snapshot_root,
        gate6_report_path=gate6_report_path,
        reference_path=reference_path,
        database=database,
        work_root=work_root,
        repo_root=repo_root,
        poetry_lock_path=poetry_lock_path,
        execution_contract_path=execution_contract_path,
    )
    status = status_adaptive_pilot_launch(
        protocol_path=protocol_path,
        work_root=work_root,
        execution_contract_path=execution_contract_path,
    )
    if not resume and (status.terminal_jobs != 0 or status.running != 0):
        raise AdaptivePilotLaunchError("pilot already contains execution state; use resume")
    if readiness.binding_hash != status.binding_hash:
        raise AdaptivePilotLaunchError("readiness/status binding mismatch")
    protocol = _load_protocol(protocol_path)
    execution_contract = _load_execution_contract(execution_contract_path)
    binding = _load_persisted_binding(work_root, execution_contract)
    assets = load_adaptive_pilot_launch_assets(
        protocol_path=protocol_path,
        binding=binding,
        execution_contract=execution_contract,
        snapshot_root=snapshot_root,
        gate6_report_path=gate6_report_path,
        reference_path=reference_path,
    )
    store = SQLiteBenchmarkStore(Path(database))
    try:
        return execute_adaptive_pilot_run(
            protocol=protocol,
            binding=binding,
            work_root=work_root,
            benchmark_store=store,
            job_executor=build_adaptive_pilot_job_executor(assets),
        )
    except (OSError, ValueError, AdaptivePilotRunError, AdaptivePilotExecutionError) as exc:
        raise AdaptivePilotLaunchError(f"adaptive pilot execution failed: {exc}") from exc
