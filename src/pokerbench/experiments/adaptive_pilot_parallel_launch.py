"""Launch/readiness wiring for the frozen Spec-34 four-worker adaptive pilot."""

from __future__ import annotations

import os
from pathlib import Path

from pokerbench.experiments.adaptive_pilot_binding import (
    AdaptivePilotBindingError,
    AdaptivePilotExecutionBinding,
    capture_adaptive_pilot_execution_binding,
    load_adaptive_pilot_execution_binding,
)
from pokerbench.experiments.adaptive_pilot_launch import (
    AdaptivePilotLaunchAssets,
    AdaptivePilotLaunchStatus,
    load_adaptive_pilot_launch_assets,
)
from pokerbench.experiments.adaptive_pilot_parallel_execution_contract import (
    AdaptivePilotParallelExecutionContract,
    AdaptivePilotParallelExecutionContractError,
    load_adaptive_pilot_parallel_execution_contract,
)
from pokerbench.experiments.adaptive_pilot_parallel_run import (
    PARALLEL_WORKERS,
    AdaptivePilotParallelRunError,
    AdaptivePilotParallelWorkerTask,
    execute_adaptive_pilot_parallel_run,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotProtocol,
    load_adaptive_pilot_protocol,
)
from pokerbench.experiments.adaptive_pilot_run import (
    AdaptivePilotRunError,
    AdaptivePilotRunReadiness,
    AdaptivePilotRunResult,
    adaptive_pilot_job_key,
    assert_adaptive_pilot_run_ready,
    initialize_adaptive_pilot_run,
    load_adaptive_pilot_run_manifest,
)
from pokerbench.persistence.sqlite import SQLiteBenchmarkStore


ADAPTIVE_PILOT_PARALLEL_LAUNCH_VERSION = "phase2-adaptive-pilot-parallel-launch-v1"
DEFAULT_ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_PATH = Path(
    "configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json"
)


class AdaptivePilotParallelLaunchError(ValueError):
    """Raised when a real Spec-34 launch is not exactly authorized."""


def _require_parallel_host() -> None:
    count = os.cpu_count()
    if count is None or count < PARALLEL_WORKERS:
        raise AdaptivePilotParallelLaunchError("parallel pilot requires at least four logical CPU cores")


def _load_protocol(path: str | Path) -> AdaptivePilotProtocol:
    try:
        return load_adaptive_pilot_protocol(Path(path))
    except (OSError, ValueError) as exc:
        raise AdaptivePilotParallelLaunchError(f"could not load frozen pilot protocol: {exc}") from exc


def _load_contract(path: str | Path) -> AdaptivePilotParallelExecutionContract:
    try:
        return load_adaptive_pilot_parallel_execution_contract(Path(path))
    except (OSError, ValueError, AdaptivePilotParallelExecutionContractError) as exc:
        raise AdaptivePilotParallelLaunchError(f"could not load frozen parallel execution contract: {exc}") from exc


def _load_binding(
    work_root: str | Path,
    contract: AdaptivePilotParallelExecutionContract,
) -> AdaptivePilotExecutionBinding:
    path = Path(work_root) / "binding.json"
    if not path.exists():
        raise AdaptivePilotParallelLaunchError("parallel adaptive pilot is not planned; binding.json is missing")
    try:
        binding = load_adaptive_pilot_execution_binding(path)
    except (OSError, ValueError, AdaptivePilotBindingError) as exc:
        raise AdaptivePilotParallelLaunchError(f"could not load persisted parallel binding: {exc}") from exc
    if (
        binding.execution_contract_id != contract.execution_contract_id
        or binding.execution_contract_version != contract.execution_contract_version
        or binding.execution_contract_hash != contract.payload_hash
        or binding.workers != contract.workers
    ):
        raise AdaptivePilotParallelLaunchError("persisted binding differs from frozen parallel execution contract")
    return binding


def _status(
    *, protocol: AdaptivePilotProtocol, binding: AdaptivePilotExecutionBinding, work_root: str | Path
) -> AdaptivePilotLaunchStatus:
    path = Path(work_root) / "run-manifest.json"
    if not path.exists():
        raise AdaptivePilotParallelLaunchError("parallel adaptive pilot is not planned; run-manifest.json is missing")
    manifest = load_adaptive_pilot_run_manifest(path, protocol=protocol, expected_binding=binding)
    counts = {name: sum(item.status == name for item in manifest.jobs) for name in ("planned", "running", "completed", "failed")}
    return AdaptivePilotLaunchStatus(
        binding_hash=binding.binding_hash,
        planned_jobs=manifest.planned_job_count,
        planned_pairs=manifest.planned_pair_count,
        planned_engine_hands=manifest.planned_engine_hands,
        terminal_jobs=manifest.terminal_jobs,
        planned=counts["planned"], running=counts["running"], completed=counts["completed"], failed=counts["failed"],
    )


def status_adaptive_pilot_parallel_launch(
    *, protocol_path: str | Path, work_root: str | Path,
    execution_contract_path: str | Path = DEFAULT_ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_PATH,
) -> AdaptivePilotLaunchStatus:
    protocol = _load_protocol(protocol_path)
    contract = _load_contract(execution_contract_path)
    binding = _load_binding(work_root, contract)
    return _status(protocol=protocol, binding=binding, work_root=work_root)


def _load_assets(
    *, protocol_path: str | Path, binding: AdaptivePilotExecutionBinding,
    contract: AdaptivePilotParallelExecutionContract, snapshot_root: str | Path,
    gate6_report_path: str | Path, reference_path: str | Path,
) -> AdaptivePilotLaunchAssets:
    try:
        return load_adaptive_pilot_launch_assets(
            protocol_path=protocol_path,
            binding=binding,
            execution_contract=contract,
            snapshot_root=snapshot_root,
            gate6_report_path=gate6_report_path,
            reference_path=reference_path,
        )
    except (OSError, ValueError) as exc:
        raise AdaptivePilotParallelLaunchError(f"could not verify parallel launch assets: {exc}") from exc


def plan_adaptive_pilot_parallel_launch(
    *, protocol_path: str | Path, primary_population_path: str | Path,
    secondary_population_path: str | Path, snapshot_root: str | Path,
    gate6_report_path: str | Path, reference_path: str | Path, database: str | Path,
    work_root: str | Path, repo_root: str | Path, poetry_lock_path: str | Path,
    ram_bytes: int, workers: int = PARALLEL_WORKERS,
    execution_contract_path: str | Path = DEFAULT_ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_PATH,
) -> AdaptivePilotLaunchStatus:
    if workers != PARALLEL_WORKERS:
        raise AdaptivePilotParallelLaunchError("phase2 adaptive pilot parallel launch requires workers=4")
    _require_parallel_host()
    protocol = _load_protocol(protocol_path)
    contract = _load_contract(execution_contract_path)
    if contract.workers != PARALLEL_WORKERS or contract.max_in_flight_jobs != PARALLEL_WORKERS:
        raise AdaptivePilotParallelLaunchError("parallel execution contract does not freeze four workers")
    try:
        binding = capture_adaptive_pilot_execution_binding(
            protocol=protocol, execution_contract=contract,
            gate6_report_path=gate6_report_path,
            primary_population_path=primary_population_path,
            secondary_population_paths=(secondary_population_path,),
            repo_root=repo_root, poetry_lock_path=poetry_lock_path,
            ram_bytes=ram_bytes, workers=workers,
        )
        _load_assets(
            protocol_path=protocol_path, binding=binding, contract=contract,
            snapshot_root=snapshot_root, gate6_report_path=gate6_report_path, reference_path=reference_path,
        )
        store = SQLiteBenchmarkStore(Path(database))
        store.initialize()
        initialize_adaptive_pilot_run(protocol=protocol, binding=binding, work_root=work_root)
    except AdaptivePilotParallelLaunchError:
        raise
    except (OSError, ValueError, AdaptivePilotBindingError, AdaptivePilotRunError) as exc:
        raise AdaptivePilotParallelLaunchError(f"could not plan parallel adaptive pilot: {exc}") from exc
    return _status(protocol=protocol, binding=binding, work_root=work_root)


def assert_adaptive_pilot_parallel_launch_ready(
    *, protocol_path: str | Path, primary_population_path: str | Path,
    secondary_population_path: str | Path, snapshot_root: str | Path,
    gate6_report_path: str | Path, reference_path: str | Path, database: str | Path,
    work_root: str | Path, repo_root: str | Path, poetry_lock_path: str | Path,
    execution_contract_path: str | Path = DEFAULT_ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_PATH,
) -> AdaptivePilotRunReadiness:
    _require_parallel_host()
    protocol = _load_protocol(protocol_path)
    contract = _load_contract(execution_contract_path)
    persisted = _load_binding(work_root, contract)
    try:
        current = capture_adaptive_pilot_execution_binding(
            protocol=protocol, execution_contract=contract,
            gate6_report_path=gate6_report_path,
            primary_population_path=primary_population_path,
            secondary_population_paths=(secondary_population_path,),
            repo_root=repo_root, poetry_lock_path=poetry_lock_path,
            ram_bytes=persisted.ram_bytes, workers=PARALLEL_WORKERS,
        )
    except (OSError, ValueError, AdaptivePilotBindingError) as exc:
        raise AdaptivePilotParallelLaunchError(f"could not recapture current parallel binding: {exc}") from exc
    if current != persisted:
        raise AdaptivePilotParallelLaunchError("current parallel execution binding differs from persisted binding")
    _load_assets(
        protocol_path=protocol_path, binding=persisted, contract=contract,
        snapshot_root=snapshot_root, gate6_report_path=gate6_report_path, reference_path=reference_path,
    )
    store = SQLiteBenchmarkStore(Path(database))
    try:
        readiness = assert_adaptive_pilot_run_ready(
            protocol=protocol, binding=persisted, work_root=work_root,
            benchmark_store=store, expected_workers=PARALLEL_WORKERS,
        )
    except (OSError, ValueError, AdaptivePilotRunError) as exc:
        raise AdaptivePilotParallelLaunchError(f"parallel readiness failed: {exc}") from exc
    if (
        readiness.planned_jobs != contract.planned_jobs
        or readiness.planned_pairs != contract.planned_pairs
        or readiness.planned_engine_hands != contract.planned_engine_hands
    ):
        raise AdaptivePilotParallelLaunchError("parallel readiness matrix differs from frozen contract")
    return readiness


def run_adaptive_pilot_parallel_launch(
    *, protocol_path: str | Path, primary_population_path: str | Path,
    secondary_population_path: str | Path, snapshot_root: str | Path,
    gate6_report_path: str | Path, reference_path: str | Path, database: str | Path,
    work_root: str | Path, repo_root: str | Path, poetry_lock_path: str | Path,
    resume: bool,
    execution_contract_path: str | Path = DEFAULT_ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_PATH,
) -> AdaptivePilotRunResult:
    readiness = assert_adaptive_pilot_parallel_launch_ready(
        protocol_path=protocol_path, primary_population_path=primary_population_path,
        secondary_population_path=secondary_population_path, snapshot_root=snapshot_root,
        gate6_report_path=gate6_report_path, reference_path=reference_path,
        database=database, work_root=work_root, repo_root=repo_root,
        poetry_lock_path=poetry_lock_path, execution_contract_path=execution_contract_path,
    )
    protocol = _load_protocol(protocol_path)
    contract = _load_contract(execution_contract_path)
    binding = _load_binding(work_root, contract)
    status = _status(protocol=protocol, binding=binding, work_root=work_root)
    if not resume and (status.terminal_jobs != 0 or status.running != 0):
        raise AdaptivePilotParallelLaunchError("pilot already contains execution state; use resume")
    if readiness.binding_hash != status.binding_hash:
        raise AdaptivePilotParallelLaunchError("readiness/status binding mismatch")
    _load_assets(
        protocol_path=protocol_path, binding=binding, contract=contract,
        snapshot_root=snapshot_root, gate6_report_path=gate6_report_path, reference_path=reference_path,
    )
    root = Path(work_root).resolve()

    def task_builder(index, job, config, match_id):  # type: ignore[no-untyped-def]
        return AdaptivePilotParallelWorkerTask(
            protocol_path=str(Path(protocol_path).resolve()),
            execution_contract_path=str(Path(execution_contract_path).resolve()),
            snapshot_root=str(Path(snapshot_root).resolve()),
            gate6_report_path=str(Path(gate6_report_path).resolve()),
            reference_path=str(Path(reference_path).resolve()),
            job_index=index,
            trajectory_config_path=str(root / "trajectory-configs" / f"{adaptive_pilot_job_key(job)}.json"),
            match_id=match_id,
            binding_path=str(root / "binding.json"),
        )

    store = SQLiteBenchmarkStore(Path(database))
    try:
        return execute_adaptive_pilot_parallel_run(
            protocol=protocol, binding=binding, work_root=work_root,
            benchmark_store=store, task_builder=task_builder,
        )
    except (OSError, ValueError, AdaptivePilotRunError, AdaptivePilotParallelRunError, AdaptivePilotExecutionError) as exc:
        raise AdaptivePilotParallelLaunchError(f"parallel adaptive pilot execution failed: {exc}") from exc
