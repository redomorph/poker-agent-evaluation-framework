from __future__ import annotations

import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pokerbench.engine.config import BASELINE_GAME_CONFIG
from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.experiments.adaptive_pilot_binding import build_adaptive_pilot_execution_binding
from pokerbench.experiments.adaptive_pilot_parallel_execution_contract import load_adaptive_pilot_parallel_execution_contract
from pokerbench.experiments.adaptive_pilot_parallel_run import (
    AdaptivePilotParallelRunError,
    AdaptivePilotParallelWorkerTask,
    execute_adaptive_pilot_parallel_run,
)
from pokerbench.experiments.adaptive_pilot_protocol import iter_adaptive_pilot_jobs, load_adaptive_pilot_protocol
from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.frozen_protocol import load_frozen_population
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import PairedResult, StatisticalUnit

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-v1.json"
CONTRACT_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-parallel-execution-v1.json"
PRIMARY_PATH = ROOT / "configs" / "populations" / "baseline-synthetic-v1.json"
SECONDARY_PATH = ROOT / "configs" / "populations" / "balanced-v1.json"
GATE6_PATH = ROOT / "data" / "gate6" / "gate6-report.json"


class MemoryStore:
    def __init__(self) -> None:
        self.records: dict[str, MatchRecord] = {}
        self.save_order: list[str] = []

    def save_match(self, record: MatchRecord) -> None:
        if record.match_id in self.records:
            raise ValueError("duplicate")
        self.records[record.match_id] = record
        self.save_order.append(record.match_id)

    def load_match(self, match_id: str) -> MatchRecord:
        return self.records[match_id]

    def list_match_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.records))


def binding():
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    contract = load_adaptive_pilot_parallel_execution_contract(CONTRACT_PATH)
    commit = "a" * 40
    env = ExecutionEnvironment(
        python_version="3.14.0", engine_version=f"git:{commit}", dependency_lockfile="poetry.lock",
        poetry_lock_hash="b" * 64, git_commit=commit, cpu_model="Test CPU", cpu_core_count=8,
        cpu_architecture="AMD64", ram_bytes=16 * 1024**3, os_identifier="TestOS",
    )
    return build_adaptive_pilot_execution_binding(
        protocol=protocol, execution_contract=contract,
        gate6_report=load_gate6_calibration_report(GATE6_PATH),
        primary_population=load_frozen_population(PRIMARY_PATH),
        secondary_populations=(load_frozen_population(SECONDARY_PATH),), environment=env, workers=4,
    )


def failed_result(job, config, match_id):  # type: ignore[no-untyped-def]
    record = MatchRecord(
        match_id=match_id, master_seed=job.master_seed, game_config=BASELINE_GAME_CONFIG, hands=(),
        experiment_id=config.run_provenance["experiment_id"], experiment_config_hash=config.config_hash,
    )
    return PairedResult(
        paired_differences=[], statistical_unit=StatisticalUnit.TRAJECTORY_CLUSTER,
        num_hands=job.hands_per_position, num_valid_pairs=0,
        failed_runs=[{"hand": 0, "config": "A_BTN", "details": "synthetic"}],
        experiment_metadata={"experiment_config_hash": config.config_hash}, seed=job.master_seed,
        match_record=record,
    )


def test_parallel_worker_task_is_pickle_compatible_and_contains_no_scheduler_identity() -> None:
    task = AdaptivePilotParallelWorkerTask(
        protocol_path="protocol.json", execution_contract_path="contract.json", snapshot_root="snapshots",
        gate6_report_path="gate6.json", reference_path="reference.json", job_index=7,
        trajectory_config_path="config.json", match_id="match-1",
    )
    assert pickle.loads(pickle.dumps(task)) == task
    payload = vars(task)
    assert "pid" not in payload and "worker_id" not in payload and "completion_order" not in payload


def test_parallel_run_caps_concurrency_at_four_and_preserves_canonical_manifest_order(tmp_path: Path) -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    bind = binding()
    store = MemoryStore()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def worker(task):  # type: ignore[no-untyped-def]
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.002 * (4 - (task.job_index % 4)))
            job = iter_adaptive_pilot_jobs(protocol)[task.job_index]
            from pokerbench.experiments.adaptive_pilot_binding import load_adaptive_pilot_trajectory_config
            config = load_adaptive_pilot_trajectory_config(
                task.trajectory_config_path, protocol=protocol, expected_binding=bind, expected_job=job
            )
            return failed_result(job, config, task.match_id)
        finally:
            with lock:
                active -= 1

    result = execute_adaptive_pilot_parallel_run(
        protocol=protocol, binding=bind, work_root=tmp_path, benchmark_store=store,
        task_builder=lambda index, job, config, match_id: AdaptivePilotParallelWorkerTask(
            protocol_path=str(PROTOCOL_PATH), execution_contract_path=str(CONTRACT_PATH),
            snapshot_root="unused", gate6_report_path="unused", reference_path="unused",
            job_index=index, trajectory_config_path=str(tmp_path / "trajectory-configs" / f"{job.arm_id}__{job.opponent_profile_id}__{job.master_seed}.json"),
            match_id=match_id,
        ),
        worker_function=worker,
        executor_factory=lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
    )
    assert max_active <= 4
    assert result.terminal_jobs == 50 and result.failed_jobs == 50
    manifest = __import__(
        "pokerbench.experiments.adaptive_pilot_run", fromlist=["load_adaptive_pilot_run_manifest"]
    ).load_adaptive_pilot_run_manifest(tmp_path / "run-manifest.json", protocol=protocol, expected_binding=bind)
    assert [state.job_key for state in manifest.jobs] == [
        f"{job.arm_id}__{job.opponent_profile_id}__{job.master_seed}"
        for job in iter_adaptive_pilot_jobs(protocol)
    ]
    assert set(store.save_order) == {state.match_id for state in manifest.jobs}


def test_parallel_infrastructure_failure_leaves_nonterminal_jobs_resumable(tmp_path: Path) -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    bind = binding()
    store = MemoryStore()

    def worker(task):  # type: ignore[no-untyped-def]
        if task.job_index == 1:
            raise RuntimeError("worker crashed")
        job = iter_adaptive_pilot_jobs(protocol)[task.job_index]
        from pokerbench.experiments.adaptive_pilot_binding import load_adaptive_pilot_trajectory_config
        config = load_adaptive_pilot_trajectory_config(
            task.trajectory_config_path, protocol=protocol, expected_binding=bind, expected_job=job
        )
        return failed_result(job, config, task.match_id)

    with pytest.raises(AdaptivePilotParallelRunError, match="worker infrastructure failure"):
        execute_adaptive_pilot_parallel_run(
            protocol=protocol, binding=bind, work_root=tmp_path, benchmark_store=store,
            task_builder=lambda index, job, config, match_id: AdaptivePilotParallelWorkerTask(
                protocol_path=str(PROTOCOL_PATH), execution_contract_path=str(CONTRACT_PATH), snapshot_root="x",
                gate6_report_path="x", reference_path="x", job_index=index,
                trajectory_config_path=str(tmp_path / "trajectory-configs" / f"{job.arm_id}__{job.opponent_profile_id}__{job.master_seed}.json"), match_id=match_id,
            ), worker_function=worker, executor_factory=lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
        )

    from pokerbench.experiments.adaptive_pilot_run import assert_adaptive_pilot_run_ready
    ready = assert_adaptive_pilot_run_ready(
        protocol=protocol, binding=bind, work_root=tmp_path, benchmark_store=store, expected_workers=4
    )
    assert ready.terminal_jobs < 50


def test_parallel_worker_task_exposes_no_campaign_persistence_and_worker_function_is_pickleable() -> None:
    from pokerbench.experiments.adaptive_pilot_parallel_run import execute_adaptive_pilot_parallel_worker_task
    task = AdaptivePilotParallelWorkerTask(
        protocol_path="p", execution_contract_path="c", snapshot_root="s", gate6_report_path="g",
        reference_path="r", job_index=0, trajectory_config_path="t", match_id="m", binding_path="b",
    )
    assert "database" not in vars(task)
    assert "work_root" not in vars(task)
    pickle.dumps(execute_adaptive_pilot_parallel_worker_task)


def test_parallel_parent_writes_snapshot_before_campaign_store(tmp_path: Path) -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    bind = binding()

    class OrderingStore(MemoryStore):
        def save_match(self, record: MatchRecord) -> None:
            assert len(list((tmp_path / "arm-snapshots").glob("*.json"))) == len(self.records) + 1
            super().save_match(record)

    store = OrderingStore()

    def worker(task):  # type: ignore[no-untyped-def]
        job = iter_adaptive_pilot_jobs(protocol)[task.job_index]
        from pokerbench.experiments.adaptive_pilot_binding import load_adaptive_pilot_trajectory_config
        config = load_adaptive_pilot_trajectory_config(
            task.trajectory_config_path, protocol=protocol, expected_binding=bind, expected_job=job
        )
        return failed_result(job, config, task.match_id)

    execute_adaptive_pilot_parallel_run(
        protocol=protocol, binding=bind, work_root=tmp_path, benchmark_store=store,
        task_builder=lambda index, job, config, match_id: AdaptivePilotParallelWorkerTask(
            protocol_path=str(PROTOCOL_PATH), execution_contract_path=str(CONTRACT_PATH), snapshot_root="x",
            gate6_report_path="x", reference_path="x", job_index=index,
            trajectory_config_path=str(tmp_path / "trajectory-configs" / f"{job.arm_id}__{job.opponent_profile_id}__{job.master_seed}.json"), match_id=match_id,
        ), worker_function=worker, executor_factory=lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
    )
    assert len(store.records) == 50
