"""Four-process parent-committed execution for the frozen Spec-38 variance pilot."""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

from pokerbench.agents.opponent_range import load_preflop_strength_reference
from pokerbench.agents.parametric_policy import ParametricPolicyAgent
from pokerbench.agents.strength_gates import build_policy_calibration_map
from pokerbench.engine.config import BASELINE_GAME_CONFIG
from pokerbench.evaluation.calibration_snapshots import (
    CalibrationStrengthSnapshotStore,
    policy_strength_reference_from_store,
    policy_validation_strengths_from_store,
)
from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.evaluation.h4_holdout_acceptance import (
    H4_HOLDOUT_ACCEPTANCE_HASH,
    load_h4_holdout_acceptance,
)
from pokerbench.experiments.h4_holdout_population import (
    H4HoldoutPopulation,
    load_h4_holdout_population,
)
from pokerbench.experiments.h4_holdout_variance_binding import (
    H4HoldoutVarianceExecutionBinding,
    H4HoldoutVarianceTrajectoryConfig,
    load_h4_holdout_variance_execution_binding,
    load_h4_holdout_variance_trajectory_config,
)
from pokerbench.experiments.h4_holdout_variance_execution import (
    H4HoldoutVarianceExecutionError,
    build_h4_holdout_variance_runner,
    finalize_h4_holdout_variance_pair,
)
from pokerbench.experiments.h4_holdout_variance_execution_contract import (
    H4_HOLDOUT_VARIANCE_MAX_IN_FLIGHT_JOBS,
    H4_HOLDOUT_VARIANCE_PROCESS_MODEL,
    H4_HOLDOUT_VARIANCE_WORKERS,
    load_h4_holdout_variance_execution_contract,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4_HOLDOUT_UNIT_IDS,
    H4HoldoutVarianceJob,
    H4HoldoutVarianceProtocol,
    iter_h4_holdout_variance_jobs,
    load_h4_holdout_variance_protocol,
)
from pokerbench.experiments.h4_holdout_variance_run import (
    H4HoldoutVarianceRunResult,
    H4HoldoutVarianceRunStore,
    _arm_snapshot_from_result,
    _load_exact_config,
    _cluster_checkpoint_from_pairs,
    _cluster_checkpoint_path,
    _load_cluster_checkpoint,
    _load_or_finalize_cluster_checkpoints,
    _load_or_finalize_pair_checkpoints,
    _load_pair_checkpoint,
    _pair_checkpoint_from_execution,
    _pair_checkpoint_path,
    _reconcile_manifest,
    _replace_manifest_job,
    _save_checkpoint,
    _snapshot_path,
    assert_h4_holdout_variance_run_ready,
    h4_holdout_variance_match_id,
    initialize_h4_holdout_variance_run,
    load_h4_holdout_variance_arm_snapshot,
    load_h4_holdout_variance_run_manifest,
    save_h4_holdout_variance_arm_snapshot,
    save_h4_holdout_variance_run_manifest,
)
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import PairedResult


H4_HOLDOUT_VARIANCE_PARALLEL_RUN_VERSION = "phase2-h4-holdout-variance-parallel-run-v1"
PARALLEL_WORKERS = H4_HOLDOUT_VARIANCE_WORKERS


class H4HoldoutVarianceParallelRunError(ValueError):
    """Raised when Spec-38 parallel orchestration violates its frozen boundary."""


@dataclass(frozen=True)
class H4HoldoutVarianceParallelWorkerTask:
    """Spawn-compatible immutable description of one canonical Spec-37 arm job."""

    protocol_path: str
    execution_contract_path: str
    population_path: str
    acceptance_path: str
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
        job: H4HoldoutVarianceJob,
        config: H4HoldoutVarianceTrajectoryConfig,
        match_id: str,
    ) -> H4HoldoutVarianceParallelWorkerTask: ...


class _CaptureStore:
    """Worker-local non-durable store; workers never touch campaign persistence."""

    def __init__(self) -> None:
        self._record: MatchRecord | None = None

    def save_match(self, record: MatchRecord) -> None:
        if self._record is not None:
            raise H4HoldoutVarianceParallelRunError(
                "worker attempted to persist more than one MatchRecord"
            )
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


def _require_worker_binding(
    binding: H4HoldoutVarianceExecutionBinding,
    *,
    contract,
) -> None:
    authority = binding.execution_authority
    if (
        authority.authority_id != contract.execution_contract_id
        or authority.authority_version != contract.execution_contract_version
        or authority.payload_hash != contract.payload_hash
        or authority.authorized_protocol_id != contract.authorized_protocol_id
        or authority.authorized_protocol_version != contract.authorized_protocol_version
        or authority.authorized_protocol_hash != contract.authorized_protocol_hash
        or binding.workers != H4_HOLDOUT_VARIANCE_WORKERS
        or binding.process_model != H4_HOLDOUT_VARIANCE_PROCESS_MODEL
    ):
        raise H4HoldoutVarianceParallelRunError(
            "worker binding differs from frozen Spec-38 execution contract"
        )


def execute_h4_holdout_variance_parallel_worker_task(
    task: H4HoldoutVarianceParallelWorkerTask,
) -> PairedResult:
    """Execute one canonical arm in isolated process state with no campaign-store write."""
    if not isinstance(task, H4HoldoutVarianceParallelWorkerTask):
        raise TypeError("task must be H4HoldoutVarianceParallelWorkerTask")
    if not task.binding_path:
        raise H4HoldoutVarianceParallelRunError("worker task binding path is required")

    protocol = load_h4_holdout_variance_protocol(Path(task.protocol_path))
    contract = load_h4_holdout_variance_execution_contract(Path(task.execution_contract_path))
    population = load_h4_holdout_population(Path(task.population_path))
    jobs = iter_h4_holdout_variance_jobs(protocol)
    if task.job_index < 0 or task.job_index >= len(jobs):
        raise H4HoldoutVarianceParallelRunError(
            "worker task job index is outside the frozen Spec-37 matrix"
        )
    job = jobs[task.job_index]
    binding = load_h4_holdout_variance_execution_binding(Path(task.binding_path))
    _require_worker_binding(binding, contract=contract)
    config = load_h4_holdout_variance_trajectory_config(
        Path(task.trajectory_config_path),
        protocol=protocol,
        expected_binding=binding,
        population=population,
        expected_job=job,
    )
    expected_match_id = h4_holdout_variance_match_id(
        binding=binding, trajectory_config_hash=config.config_hash
    )
    if task.match_id != expected_match_id or config.binding != binding:
        raise H4HoldoutVarianceParallelRunError(
            "worker task identity differs from bound trajectory"
        )

    snapshots = CalibrationStrengthSnapshotStore(Path(task.snapshot_root))
    policy_reference = policy_strength_reference_from_store(snapshots)
    validation_strengths = policy_validation_strengths_from_store(snapshots)
    gate6_report = load_gate6_calibration_report(Path(task.gate6_report_path))
    acceptance = load_h4_holdout_acceptance(
        Path(task.acceptance_path),
        population=population,
        reference=policy_reference,
        validation_strengths=validation_strengths,
        gate6_report_hash=gate6_report.report_hash,
    )
    if (
        acceptance.acceptance_hash != H4_HOLDOUT_ACCEPTANCE_HASH
        or acceptance.acceptance_hash != binding.holdout_acceptance_hash
    ):
        raise H4HoldoutVarianceParallelRunError(
            "worker holdout acceptance differs from frozen binding"
        )

    adaptive_reference = load_preflop_strength_reference(Path(task.reference_path))
    unit = next((item for item in population.units if item.unit_id == job.holdout_unit_id), None)
    if unit is None:
        raise H4HoldoutVarianceParallelRunError("worker holdout unit is missing from frozen population")
    calibration_map = build_policy_calibration_map(
        unit.parameters,
        policy_reference,
        profile_name=unit.unit_id,
        profile_version=population.population_version,
    )
    opponent = ParametricPolicyAgent(calibration_map)
    capture = _CaptureStore()
    runner = build_h4_holdout_variance_runner(
        protocol=protocol,
        job=job,
        reference=adaptive_reference,
        opponent=opponent,
        game_config=BASELINE_GAME_CONFIG,
        match_id=task.match_id,
        benchmark_store=capture,
        experiment_metadata=config.runner_metadata,
    )
    result = runner.run()
    if result.match_record is None or capture.record is None:
        raise H4HoldoutVarianceParallelRunError("worker returned no complete MatchRecord")
    if result.match_record.record_hash != capture.record.record_hash:
        raise H4HoldoutVarianceParallelRunError(
            "worker capture differs from returned MatchRecord"
        )
    return result


def _finalize_related_checkpoints(
    *,
    protocol: H4HoldoutVarianceProtocol,
    binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation,
    root: Path,
    store: H4HoldoutVarianceRunStore,
    manifest,
    completed_job: H4HoldoutVarianceJob,
) -> None:
    jobs = iter_h4_holdout_variance_jobs(protocol)
    pair_items: dict[str, tuple[H4HoldoutVarianceJob, object]] = {}
    for candidate, state in zip(jobs, manifest.jobs, strict=True):
        if (
            candidate.holdout_unit_id == completed_job.holdout_unit_id
            and candidate.master_seed == completed_job.master_seed
            and state.status in {"completed", "failed"}
        ):
            snapshot = load_h4_holdout_variance_arm_snapshot(
                _snapshot_path(root, candidate),
                protocol=protocol,
                expected_binding=binding,
                population=population,
                expected_job=candidate,
            )
            pair_items[candidate.arm_id] = (candidate, snapshot)
    if set(pair_items) != {"control", "treatment"}:
        return

    control_job, control = pair_items["control"]
    treatment_job, treatment = pair_items["treatment"]
    pair_path = _pair_checkpoint_path(
        root, completed_job.holdout_unit_id, completed_job.master_seed
    )
    if pair_path.exists():
        pair = _load_pair_checkpoint(
            pair_path, binding=binding, treatment=treatment, control=control
        )
    else:
        execution = finalize_h4_holdout_variance_pair(
            protocol=protocol,
            treatment_job=treatment_job,
            control_job=control_job,
            treatment_result=treatment.to_paired_result(),
            control_result=control.to_paired_result(),
            benchmark_store=store,
        )
        pair = _pair_checkpoint_from_execution(
            binding=binding,
            execution=execution,
            treatment=treatment,
            control=control,
        )
        _save_checkpoint(pair, pair_path)

    seed = completed_job.master_seed
    pair_paths = [_pair_checkpoint_path(root, unit_id, seed) for unit_id in H4_HOLDOUT_UNIT_IDS]
    if not all(path.exists() for path in pair_paths):
        return

    pairs = []
    for unit_id, path in zip(H4_HOLDOUT_UNIT_IDS, pair_paths, strict=True):
        unit_items: dict[str, tuple[H4HoldoutVarianceJob, object]] = {}
        for candidate, state in zip(jobs, manifest.jobs, strict=True):
            if (
                candidate.holdout_unit_id == unit_id
                and candidate.master_seed == seed
                and state.status in {"completed", "failed"}
            ):
                snapshot = load_h4_holdout_variance_arm_snapshot(
                    _snapshot_path(root, candidate),
                    protocol=protocol,
                    expected_binding=binding,
                    population=population,
                    expected_job=candidate,
                )
                unit_items[candidate.arm_id] = (candidate, snapshot)
        if set(unit_items) != {"control", "treatment"}:
            raise H4HoldoutVarianceParallelRunError(
                "cluster checkpoint inputs lost a terminal unit pair"
            )
        pairs.append(
            _load_pair_checkpoint(
                path,
                binding=binding,
                treatment=unit_items["treatment"][1],
                control=unit_items["control"][1],
            )
        )
    canonical_pairs = tuple(pairs)
    cluster_path = _cluster_checkpoint_path(root, seed)
    if cluster_path.exists():
        _load_cluster_checkpoint(
            cluster_path, binding=binding, pairs=canonical_pairs, seed=seed
        )
    else:
        cluster = _cluster_checkpoint_from_pairs(
            binding=binding, seed=seed, pairs=canonical_pairs
        )
        _save_checkpoint(cluster, cluster_path)


def _commit_result(
    *,
    protocol: H4HoldoutVarianceProtocol,
    binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation,
    work_root: Path,
    store: H4HoldoutVarianceRunStore,
    manifest,
    index: int,
    job: H4HoldoutVarianceJob,
    config: H4HoldoutVarianceTrajectoryConfig,
    result: PairedResult,
    elapsed_ns: int,
):
    state = manifest.jobs[index]
    if state.status != "running":
        raise H4HoldoutVarianceParallelRunError(
            "parallel parent may commit only a running canonical job"
        )
    snapshot = _arm_snapshot_from_result(
        job=job,
        config=config,
        match_id=state.match_id,
        result=result,
        elapsed_ns=elapsed_ns,
    )
    # Spec 38 freezes snapshot-before-database parent persistence.
    save_h4_holdout_variance_arm_snapshot(snapshot, _snapshot_path(work_root, job))
    try:
        store.save_match(snapshot.match_record)
        persisted = store.load_match(snapshot.match_id)
    except Exception as exc:
        raise H4HoldoutVarianceParallelRunError(
            "parent could not durably commit campaign MatchRecord"
        ) from exc
    if persisted.record_hash != snapshot.match_record_hash:
        raise H4HoldoutVarianceParallelRunError(
            "campaign MatchRecord differs from immutable arm snapshot"
        )
    terminal = replace(
        state,
        status=snapshot.status,
        arm_snapshot_hash=snapshot.snapshot_hash,
        match_record_hash=snapshot.match_record_hash,
    )
    manifest = _replace_manifest_job(manifest, index, terminal)
    save_h4_holdout_variance_run_manifest(manifest, work_root / "run-manifest.json")
    _finalize_related_checkpoints(
        protocol=protocol,
        binding=binding,
        population=population,
        root=work_root,
        store=store,
        manifest=manifest,
        completed_job=job,
    )
    return manifest


def _spawn_process_pool(max_workers: int) -> ProcessPoolExecutor:
    if max_workers != H4_HOLDOUT_VARIANCE_MAX_IN_FLIGHT_JOBS:
        raise H4HoldoutVarianceParallelRunError("Spec-38 process pool requires exactly four workers")
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def _result_from_ready(ready) -> H4HoldoutVarianceRunResult:  # type: ignore[no-untyped-def]
    return H4HoldoutVarianceRunResult(
        binding_hash=ready.binding_hash,
        terminal_jobs=ready.terminal_jobs,
        completed_jobs=ready.completed_jobs,
        failed_jobs=ready.failed_jobs,
        pair_count=ready.pair_count,
        valid_pairs=ready.valid_pairs,
        invalid_pairs=ready.invalid_pairs,
        cluster_count=ready.cluster_count,
        valid_clusters=ready.valid_clusters,
        invalid_clusters=ready.invalid_clusters,
        scientifically_complete=ready.scientifically_complete,
    )


def execute_h4_holdout_variance_parallel_run(
    *,
    protocol: H4HoldoutVarianceProtocol,
    binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation,
    work_root: str | Path,
    benchmark_store: H4HoldoutVarianceRunStore,
    task_builder: ParallelTaskBuilder,
    worker_function: Callable[[H4HoldoutVarianceParallelWorkerTask], PairedResult] = (
        execute_h4_holdout_variance_parallel_worker_task
    ),
    executor_factory: Callable[[int], Executor] = _spawn_process_pool,
) -> H4HoldoutVarianceRunResult:
    """Execute/resume the exact 100-job Spec-38 matrix with four isolated jobs in flight."""
    if binding.workers != H4_HOLDOUT_VARIANCE_WORKERS:
        raise H4HoldoutVarianceParallelRunError("Spec-38 parallel run requires provenance workers=4")
    if binding.process_model != H4_HOLDOUT_VARIANCE_PROCESS_MODEL:
        raise H4HoldoutVarianceParallelRunError("Spec-38 parallel run process model drifted")
    if not callable(task_builder) or not callable(worker_function):
        raise TypeError("task_builder and worker_function must be callable")

    root = Path(work_root)
    initialize_h4_holdout_variance_run(
        protocol=protocol,
        binding=binding,
        population=population,
        work_root=root,
    )
    # Actual execution/resume is allowed to reconcile durable state; the standalone
    # first-hand readiness command remains byte-nonmutating.
    assert_h4_holdout_variance_run_ready(
        protocol=protocol,
        binding=binding,
        population=population,
        work_root=root,
        benchmark_store=benchmark_store,
    )
    manifest = load_h4_holdout_variance_run_manifest(
        root / "run-manifest.json",
        protocol=protocol,
        expected_binding=binding,
        population=population,
    )
    manifest = _reconcile_manifest(
        protocol=protocol,
        binding=binding,
        population=population,
        root=root,
        store=benchmark_store,
        manifest=manifest,
    )
    jobs = iter_h4_holdout_variance_jobs(protocol)
    pending = [
        index
        for index, state in enumerate(manifest.jobs)
        if state.status not in {"completed", "failed"}
    ]
    if not pending:
        ready = assert_h4_holdout_variance_run_ready(
            protocol=protocol,
            binding=binding,
            population=population,
            work_root=root,
            benchmark_store=benchmark_store,
        )
        return _result_from_ready(ready)

    active: dict[
        Future[PairedResult],
        tuple[int, H4HoldoutVarianceJob, H4HoldoutVarianceTrajectoryConfig, int],
    ] = {}
    next_pending = 0

    def submit_one(pool: Executor, index: int) -> None:
        nonlocal manifest
        job = jobs[index]
        config = _load_exact_config(
            protocol=protocol,
            binding=binding,
            population=population,
            root=root,
            job=job,
        )
        state = manifest.jobs[index]
        if state.status != "planned":
            raise H4HoldoutVarianceParallelRunError(
                "parallel scheduler may submit only planned jobs"
            )
        manifest = _replace_manifest_job(manifest, index, replace(state, status="running"))
        save_h4_holdout_variance_run_manifest(manifest, root / "run-manifest.json")
        task = task_builder(index, job, config, state.match_id)
        if not isinstance(task, H4HoldoutVarianceParallelWorkerTask):
            raise TypeError("task_builder must return H4HoldoutVarianceParallelWorkerTask")
        started_ns = time.perf_counter_ns()
        active[pool.submit(worker_function, task)] = (index, job, config, started_ns)

    try:
        with executor_factory(H4_HOLDOUT_VARIANCE_WORKERS) as pool:
            while next_pending < len(pending) and len(active) < H4_HOLDOUT_VARIANCE_MAX_IN_FLIGHT_JOBS:
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
                        raise H4HoldoutVarianceParallelRunError(
                            f"worker infrastructure failure for canonical job index {index}"
                        ) from exc
                    manifest = _commit_result(
                        protocol=protocol,
                        binding=binding,
                        population=population,
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
    except H4HoldoutVarianceParallelRunError:
        raise
    except Exception as exc:
        raise H4HoldoutVarianceParallelRunError(
            "parallel H4 holdout-variance scheduler failed"
        ) from exc

    if manifest.terminal_jobs != 100:
        raise H4HoldoutVarianceParallelRunError(
            "parallel variance-pilot run ended without all 100 jobs terminal"
        )
    ready = assert_h4_holdout_variance_run_ready(
        protocol=protocol,
        binding=binding,
        population=population,
        work_root=root,
        benchmark_store=benchmark_store,
    )
    if ready.pair_count != 50 or ready.cluster_count != 10:
        raise H4HoldoutVarianceParallelRunError(
            "parallel variance-pilot run ended without all 50 pairs and 10 clusters"
        )
    return _result_from_ready(ready)
