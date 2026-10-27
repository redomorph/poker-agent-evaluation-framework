"""Four-process parent-committed execution for the frozen Phase-2 adaptive pilot."""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

from pokerbench.experiments.adaptive_pilot_binding import (
    AdaptivePilotExecutionBinding,
    AdaptivePilotTrajectoryConfig,
    load_adaptive_pilot_execution_binding,
    load_adaptive_pilot_trajectory_config,
)
from pokerbench.experiments.adaptive_pilot_execution import AdaptivePilotExecutionError
from pokerbench.experiments.adaptive_pilot_launch import (
    build_adaptive_pilot_job_executor,
    load_adaptive_pilot_launch_assets,
)
from pokerbench.experiments.adaptive_pilot_parallel_execution_contract import (
    load_adaptive_pilot_parallel_execution_contract,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotJob,
    AdaptivePilotProtocol,
    iter_adaptive_pilot_jobs,
    load_adaptive_pilot_protocol,
)
from pokerbench.experiments.adaptive_pilot_run import (
    AdaptivePilotRunResult,
    AdaptivePilotRunStore,
    _arm_snapshot_from_result,
    _load_exact_config,
    _load_or_finalize_pair_checkpoints,
    _reconcile_manifest,
    _replace_manifest_job,
    _snapshot_path,
    assert_adaptive_pilot_run_ready,
    initialize_adaptive_pilot_run,
    load_adaptive_pilot_run_manifest,
    save_adaptive_pilot_arm_snapshot,
    save_adaptive_pilot_run_manifest,
)
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import PairedResult


ADAPTIVE_PILOT_PARALLEL_RUN_VERSION = "phase2-adaptive-pilot-parallel-run-v1"
PARALLEL_WORKERS = 4


class AdaptivePilotParallelRunError(ValueError):
    """Raised when Spec-34 parallel orchestration violates its frozen boundary."""


@dataclass(frozen=True)
class AdaptivePilotParallelWorkerTask:
    """Spawn-compatible immutable description of one already-frozen trajectory job."""

    protocol_path: str
    execution_contract_path: str
    snapshot_root: str
    gate6_report_path: str
    reference_path: str
    job_index: int
    trajectory_config_path: str
    match_id: str
    binding_path: str = ""


class ParallelTaskBuilder(Protocol):
    def __call__(
        self,
        index: int,
        job: AdaptivePilotJob,
        config: AdaptivePilotTrajectoryConfig,
        match_id: str,
    ) -> AdaptivePilotParallelWorkerTask: ...


class _CaptureStore:
    """Worker-local non-durable store; workers never touch campaign persistence."""

    def __init__(self) -> None:
        self._record: MatchRecord | None = None

    def save_match(self, record: MatchRecord) -> None:
        if self._record is not None:
            raise AdaptivePilotParallelRunError("worker attempted to persist more than one MatchRecord")
        self._record = record

    def load_match(self, match_id: str) -> MatchRecord:
        if self._record is None or self._record.match_id != match_id:
            raise KeyError(match_id)
        return self._record

    def list_match_ids(self) -> tuple[str, ...]:
        return () if self._record is None else (self._record.match_id,)

    @property
    def record(self) -> MatchRecord | None:
        return self._record


def execute_adaptive_pilot_parallel_worker_task(
    task: AdaptivePilotParallelWorkerTask,
) -> PairedResult:
    """Execute one canonical job without any campaign-store write."""
    if not isinstance(task, AdaptivePilotParallelWorkerTask):
        raise TypeError("task must be an AdaptivePilotParallelWorkerTask")
    protocol = load_adaptive_pilot_protocol(Path(task.protocol_path))
    contract = load_adaptive_pilot_parallel_execution_contract(Path(task.execution_contract_path))
    jobs = iter_adaptive_pilot_jobs(protocol)
    if task.job_index < 0 or task.job_index >= len(jobs):
        raise AdaptivePilotParallelRunError("worker task job index is outside the frozen matrix")
    job = jobs[task.job_index]
    if not task.binding_path:
        raise AdaptivePilotParallelRunError("worker task binding path is required")
    binding = load_adaptive_pilot_execution_binding(Path(task.binding_path))
    if (
        binding.execution_contract_id != contract.execution_contract_id
        or binding.execution_contract_version != contract.execution_contract_version
        or binding.execution_contract_hash != contract.payload_hash
        or binding.workers != contract.workers
    ):
        raise AdaptivePilotParallelRunError("worker binding differs from frozen parallel contract")
    config = load_adaptive_pilot_trajectory_config(
        Path(task.trajectory_config_path),
        protocol=protocol,
        expected_binding=binding,
        expected_job=job,
    )
    assets = load_adaptive_pilot_launch_assets(
        protocol_path=task.protocol_path,
        binding=binding,
        execution_contract=contract,  # structural execution-authorization interface
        snapshot_root=task.snapshot_root,
        gate6_report_path=task.gate6_report_path,
        reference_path=task.reference_path,
    )
    capture = _CaptureStore()
    result = build_adaptive_pilot_job_executor(assets)(job, config, task.match_id, capture)
    if result.match_record is None or capture.record is None:
        raise AdaptivePilotParallelRunError("worker returned no complete MatchRecord")
    if result.match_record.record_hash != capture.record.record_hash:
        raise AdaptivePilotParallelRunError("worker capture differs from returned MatchRecord")
    return result


def _commit_result(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    work_root: Path,
    store: AdaptivePilotRunStore,
    manifest,
    index: int,
    job: AdaptivePilotJob,
    config: AdaptivePilotTrajectoryConfig,
    result: PairedResult,
    elapsed_ns: int,
):
    state = manifest.jobs[index]
    if state.status != "running":
        raise AdaptivePilotParallelRunError("parallel parent may commit only a running canonical job")
    snapshot = _arm_snapshot_from_result(
        job=job,
        config=config,
        match_id=state.match_id,
        result=result,
        elapsed_ns=elapsed_ns,
    )
    # Spec 34 deliberately persists the recoverable snapshot before campaign SQLite.
    save_adaptive_pilot_arm_snapshot(snapshot, _snapshot_path(work_root, job))
    try:
        store.save_match(snapshot.match_record)
        persisted = store.load_match(snapshot.match_id)
    except Exception as exc:
        raise AdaptivePilotParallelRunError("parent could not durably commit campaign MatchRecord") from exc
    if persisted.record_hash != snapshot.match_record_hash:
        raise AdaptivePilotParallelRunError("campaign MatchRecord differs from immutable arm snapshot")
    terminal = replace(
        state,
        status=snapshot.status,
        arm_snapshot_hash=snapshot.snapshot_hash,
        match_record_hash=snapshot.match_record_hash,
    )
    manifest = _replace_manifest_job(manifest, index, terminal)
    save_adaptive_pilot_run_manifest(manifest, work_root / "run-manifest.json")
    _load_or_finalize_pair_checkpoints(
        protocol=protocol,
        binding=binding,
        work_root=work_root,
        store=store,
        manifest=manifest,
    )
    return manifest


def _spawn_process_pool(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def execute_adaptive_pilot_parallel_run(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    work_root: str | Path,
    benchmark_store: AdaptivePilotRunStore,
    task_builder: ParallelTaskBuilder,
    worker_function: Callable[[AdaptivePilotParallelWorkerTask], PairedResult] = (
        execute_adaptive_pilot_parallel_worker_task
    ),
    executor_factory: Callable[[int], Executor] = _spawn_process_pool,
) -> AdaptivePilotRunResult:
    """Execute/resume the fixed matrix with four isolated jobs in flight."""
    if binding.workers != PARALLEL_WORKERS:
        raise AdaptivePilotParallelRunError("phase2 parallel run requires provenance workers=4")
    if not callable(task_builder) or not callable(worker_function):
        raise TypeError("task_builder and worker_function must be callable")
    root = Path(work_root)
    initialize_adaptive_pilot_run(protocol=protocol, binding=binding, work_root=root)
    assert_adaptive_pilot_run_ready(
        protocol=protocol,
        binding=binding,
        work_root=root,
        benchmark_store=benchmark_store,
        expected_workers=PARALLEL_WORKERS,
    )
    manifest = load_adaptive_pilot_run_manifest(
        root / "run-manifest.json",
        protocol=protocol,
        expected_binding=binding,
    )
    manifest = _reconcile_manifest(
        protocol=protocol,
        binding=binding,
        work_root=root,
        store=benchmark_store,
        manifest=manifest,
    )
    jobs = iter_adaptive_pilot_jobs(protocol)
    pending = [index for index, state in enumerate(manifest.jobs) if state.status not in {"completed", "failed"}]
    if not pending:
        checkpoints = _load_or_finalize_pair_checkpoints(
            protocol=protocol, binding=binding, work_root=root, store=benchmark_store, manifest=manifest
        )
        return AdaptivePilotRunResult(
            binding_hash=binding.binding_hash,
            manifest_hash=manifest.manifest_hash,
            terminal_jobs=manifest.terminal_jobs,
            completed_jobs=sum(item.status == "completed" for item in manifest.jobs),
            failed_jobs=sum(item.status == "failed" for item in manifest.jobs),
            pair_checkpoints=checkpoints,
        )

    active: dict[Future[PairedResult], tuple[int, AdaptivePilotJob, AdaptivePilotTrajectoryConfig, int]] = {}
    next_pending = 0

    def submit_one(pool: Executor, index: int) -> None:
        nonlocal manifest
        job = jobs[index]
        config = _load_exact_config(protocol=protocol, binding=binding, work_root=root, job=job)
        state = manifest.jobs[index]
        if state.status != "planned":
            raise AdaptivePilotParallelRunError("parallel scheduler may submit only planned jobs")
        manifest = _replace_manifest_job(manifest, index, replace(state, status="running"))
        save_adaptive_pilot_run_manifest(manifest, root / "run-manifest.json")
        task = task_builder(index, job, config, state.match_id)
        if not isinstance(task, AdaptivePilotParallelWorkerTask):
            raise TypeError("task_builder must return AdaptivePilotParallelWorkerTask")
        started_ns = time.perf_counter_ns()
        active[pool.submit(worker_function, task)] = (index, job, config, started_ns)

    try:
        with executor_factory(PARALLEL_WORKERS) as pool:
            while next_pending < len(pending) and len(active) < PARALLEL_WORKERS:
                submit_one(pool, pending[next_pending])
                next_pending += 1
            while active:
                done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in done:
                    index, job, config, started_ns = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        for remaining in active:
                            remaining.cancel()
                        raise AdaptivePilotParallelRunError(
                            f"worker infrastructure failure for canonical job index {index}"
                        ) from exc
                    manifest = _commit_result(
                        protocol=protocol,
                        binding=binding,
                        work_root=root,
                        store=benchmark_store,
                        manifest=manifest,
                        index=index,
                        job=job,
                        config=config,
                        result=result,
                        elapsed_ns=time.perf_counter_ns() - started_ns,
                    )
                    if next_pending < len(pending):
                        submit_one(pool, pending[next_pending])
                        next_pending += 1
    except AdaptivePilotParallelRunError:
        raise
    except Exception as exc:
        raise AdaptivePilotParallelRunError("parallel adaptive-pilot scheduler failed") from exc

    if manifest.terminal_jobs != 50:
        raise AdaptivePilotParallelRunError("parallel adaptive-pilot run ended without all 50 jobs terminal")
    checkpoints = _load_or_finalize_pair_checkpoints(
        protocol=protocol,
        binding=binding,
        work_root=root,
        store=benchmark_store,
        manifest=manifest,
    )
    if len(checkpoints) != 25:
        raise AdaptivePilotParallelRunError("parallel adaptive-pilot run ended without all 25 pairs")
    return AdaptivePilotRunResult(
        binding_hash=binding.binding_hash,
        manifest_hash=manifest.manifest_hash,
        terminal_jobs=manifest.terminal_jobs,
        completed_jobs=sum(item.status == "completed" for item in manifest.jobs),
        failed_jobs=sum(item.status == "failed" for item in manifest.jobs),
        pair_checkpoints=checkpoints,
    )
