from __future__ import annotations

import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pokerbench.experiments.h4_holdout_variance_binding import (
    load_h4_holdout_variance_trajectory_config,
)
from pokerbench.experiments.h4_holdout_variance_parallel_run import (
    H4HoldoutVarianceParallelRunError,
    H4HoldoutVarianceParallelWorkerTask,
    _spawn_process_pool,
    execute_h4_holdout_variance_parallel_run,
    execute_h4_holdout_variance_parallel_worker_task,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    iter_h4_holdout_variance_jobs,
)
from pokerbench.experiments.h4_holdout_variance_run import (
    assert_h4_holdout_variance_run_ready,
    h4_holdout_variance_job_key,
    load_h4_holdout_variance_run_manifest,
)
from pokerbench.replay.archive import MatchRecord

from test_phase2_h4_holdout_variance_run_v1 import (
    ACCEPTANCE_PATH,
    GATE6_PATH,
    POPULATION_PATH,
    PROTOCOL_PATH,
    SNAPSHOT_ROOT,
    _MemoryRunStore,
    _binding,
    _failed_result,
    _population,
    _protocol,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs" / "experiments" / "phase2-h4-holdout-variance-execution-v1.json"
REFERENCE_PATH = ROOT / "configs" / "experiments" / "preflop-range-strength-reference-v1.json"


def _task_builder(root: Path, protocol, binding):  # type: ignore[no-untyped-def]
    def build(index, job, config, match_id):  # type: ignore[no-untyped-def]
        return H4HoldoutVarianceParallelWorkerTask(
            protocol_path=str(PROTOCOL_PATH),
            execution_contract_path=str(CONTRACT_PATH),
            population_path=str(POPULATION_PATH),
            acceptance_path=str(ACCEPTANCE_PATH),
            snapshot_root=str(SNAPSHOT_ROOT),
            gate6_report_path=str(GATE6_PATH),
            reference_path=str(REFERENCE_PATH),
            job_index=index,
            trajectory_config_path=str(
                root / "trajectory-configs" / f"{h4_holdout_variance_job_key(job)}.json"
            ),
            match_id=match_id,
            binding_path=str(root / "binding.json"),
        )

    return build


def _failed_worker(protocol, binding):  # type: ignore[no-untyped-def]
    def worker(task: H4HoldoutVarianceParallelWorkerTask):
        job = iter_h4_holdout_variance_jobs(protocol)[task.job_index]
        config = load_h4_holdout_variance_trajectory_config(
            task.trajectory_config_path,
            protocol=protocol,
            expected_binding=binding,
            population=_population(),
            expected_job=job,
        )
        return _failed_result(
            job=job,
            config_hash=config.config_hash,
            experiment_id=config.run_provenance["experiment_id"],
            match_id=task.match_id,
        )

    return worker


def test_parallel_worker_task_is_pickleable_and_has_no_campaign_or_scheduler_identity() -> None:
    task = H4HoldoutVarianceParallelWorkerTask(
        protocol_path="protocol.json",
        execution_contract_path="contract.json",
        population_path="population.json",
        acceptance_path="acceptance.json",
        snapshot_root="snapshots",
        gate6_report_path="gate6.json",
        reference_path="reference.json",
        job_index=7,
        trajectory_config_path="trajectory.json",
        match_id="match-1",
        binding_path="binding.json",
    )
    assert pickle.loads(pickle.dumps(task)) == task
    payload = vars(task)
    for forbidden in (
        "database",
        "work_root",
        "pid",
        "worker_id",
        "queue_position",
        "completion_order",
    ):
        assert forbidden not in payload
    pickle.dumps(execute_h4_holdout_variance_parallel_worker_task)


def test_parallel_run_caps_concurrency_at_four_and_preserves_canonical_manifest_order(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    active = 0
    max_active = 0
    lock = threading.Lock()
    base_worker = _failed_worker(protocol, binding)

    def worker(task: H4HoldoutVarianceParallelWorkerTask):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.001 * (4 - task.job_index % 4))
            return base_worker(task)
        finally:
            with lock:
                active -= 1

    result = execute_h4_holdout_variance_parallel_run(
        protocol=protocol,
        binding=binding,
        population=_population(),
        work_root=tmp_path,
        benchmark_store=store,
        task_builder=_task_builder(tmp_path, protocol, binding),
        worker_function=worker,
        executor_factory=lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
    )

    assert max_active <= 4
    assert result.terminal_jobs == 100
    assert result.failed_jobs == 100
    assert result.pair_count == 50 and result.invalid_pairs == 50
    assert result.cluster_count == 10 and result.invalid_clusters == 10
    manifest = load_h4_holdout_variance_run_manifest(
        tmp_path / "run-manifest.json",
        protocol=protocol,
        expected_binding=binding,
        population=_population(),
    )
    assert [state.job_key for state in manifest.jobs] == [
        h4_holdout_variance_job_key(job) for job in iter_h4_holdout_variance_jobs(protocol)
    ]
    assert len(tuple((tmp_path / "pair-checkpoints").glob("*.json"))) == 50
    assert len(tuple((tmp_path / "cluster-checkpoints").glob("*.json"))) == 10


def test_parallel_parent_writes_snapshot_before_campaign_store(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()

    class OrderingStore(_MemoryRunStore):
        def save_match(self, record: MatchRecord) -> None:
            assert len(tuple((tmp_path / "arm-snapshots").glob("*.json"))) == len(self.records) + 1
            super().save_match(record)

    store = OrderingStore()
    execute_h4_holdout_variance_parallel_run(
        protocol=protocol,
        binding=binding,
        population=_population(),
        work_root=tmp_path,
        benchmark_store=store,
        task_builder=_task_builder(tmp_path, protocol, binding),
        worker_function=_failed_worker(protocol, binding),
        executor_factory=lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
    )
    assert len(store.records) == 100


def test_parallel_worker_infrastructure_failure_leaves_nonterminal_jobs_resumable(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    base_worker = _failed_worker(protocol, binding)

    def worker(task: H4HoldoutVarianceParallelWorkerTask):
        if task.job_index == 1:
            raise RuntimeError("synthetic worker crash")
        return base_worker(task)

    with pytest.raises(H4HoldoutVarianceParallelRunError, match="worker infrastructure failure"):
        execute_h4_holdout_variance_parallel_run(
            protocol=protocol,
            binding=binding,
            population=_population(),
            work_root=tmp_path,
            benchmark_store=store,
            task_builder=_task_builder(tmp_path, protocol, binding),
            worker_function=worker,
            executor_factory=lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
        )

    ready = assert_h4_holdout_variance_run_ready(
        protocol=protocol,
        binding=binding,
        population=_population(),
        work_root=tmp_path,
        benchmark_store=store,
    )
    assert ready.terminal_jobs < 100


def test_parallel_terminal_campaign_is_idempotent_and_never_reexecutes_jobs(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    first = execute_h4_holdout_variance_parallel_run(
        protocol=protocol,
        binding=binding,
        population=_population(),
        work_root=tmp_path,
        benchmark_store=store,
        task_builder=_task_builder(tmp_path, protocol, binding),
        worker_function=_failed_worker(protocol, binding),
        executor_factory=lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
    )

    def forbidden(_task):  # type: ignore[no-untyped-def]
        raise AssertionError("terminal jobs must not be executed again")

    second = execute_h4_holdout_variance_parallel_run(
        protocol=protocol,
        binding=binding,
        population=_population(),
        work_root=tmp_path,
        benchmark_store=store,
        task_builder=_task_builder(tmp_path, protocol, binding),
        worker_function=forbidden,
        executor_factory=lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
    )
    assert second == first


def test_parallel_runtime_requires_exact_four_worker_binding(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    object.__setattr__(binding, "workers", 3)
    with pytest.raises(H4HoldoutVarianceParallelRunError, match="workers=4"):
        execute_h4_holdout_variance_parallel_run(
            protocol=protocol,
            binding=binding,
            population=_population(),
            work_root=tmp_path,
            benchmark_store=_MemoryRunStore(),
            task_builder=_task_builder(tmp_path, protocol, binding),
        )


def test_spawn_pool_uses_spawn_context_and_exact_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_context(name: str):
        calls["context"] = name
        return "spawn-context"

    def fake_pool(*, max_workers: int, mp_context):  # type: ignore[no-untyped-def]
        calls["workers"] = max_workers
        calls["mp_context"] = mp_context
        return "pool"

    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_run.multiprocessing.get_context",
        fake_context,
    )
    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_run.ProcessPoolExecutor",
        fake_pool,
    )
    assert _spawn_process_pool(4) == "pool"
    assert calls == {"context": "spawn", "workers": 4, "mp_context": "spawn-context"}


def test_worker_constructs_fresh_opponent_runner_and_local_capture_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pokerbench.agents.parametric_policy import ParametricPolicyAgent
    from pokerbench.experiments.h4_holdout_variance_parallel_launch import (
        plan_h4_holdout_variance_parallel_launch,
    )
    from test_phase2_h4_holdout_variance_parallel_launch_v1 import _kwargs

    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count",
        lambda: 8,
    )
    kwargs = _kwargs(tmp_path)
    plan_h4_holdout_variance_parallel_launch(
        **kwargs,
        ram_bytes=8 * 1024**3,
        workers=4,
    )
    protocol = _protocol()
    job = iter_h4_holdout_variance_jobs(protocol)[0]
    work_root = Path(kwargs["work_root"])
    task = H4HoldoutVarianceParallelWorkerTask(
        protocol_path=str(PROTOCOL_PATH),
        execution_contract_path=str(CONTRACT_PATH),
        population_path=str(POPULATION_PATH),
        acceptance_path=str(ACCEPTANCE_PATH),
        snapshot_root=str(SNAPSHOT_ROOT),
        gate6_report_path=str(GATE6_PATH),
        reference_path=str(REFERENCE_PATH),
        job_index=0,
        trajectory_config_path=str(
            work_root / "trajectory-configs" / f"{h4_holdout_variance_job_key(job)}.json"
        ),
        match_id=load_h4_holdout_variance_run_manifest(
            work_root / "run-manifest.json",
            protocol=protocol,
            expected_binding=__import__(
                "pokerbench.experiments.h4_holdout_variance_binding",
                fromlist=["load_h4_holdout_variance_execution_binding"],
            ).load_h4_holdout_variance_execution_binding(work_root / "binding.json"),
            population=_population(),
        ).jobs[0].match_id,
        binding_path=str(work_root / "binding.json"),
    )

    opponents: list[object] = []
    captures: list[object] = []

    def fake_runner(**builder_kwargs):  # type: ignore[no-untyped-def]
        opponents.append(builder_kwargs["opponent"])
        captures.append(builder_kwargs["benchmark_store"])

        class Runner:
            def run(self):  # type: ignore[no-untyped-def]
                result = _failed_result(
                    job=builder_kwargs["job"],
                    config_hash=builder_kwargs["experiment_metadata"]["experiment_config_hash"],
                    experiment_id=builder_kwargs["experiment_metadata"]["experiment_id"],
                    match_id=builder_kwargs["match_id"],
                )
                builder_kwargs["benchmark_store"].save_match(result.match_record)
                return result

        return Runner()

    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_run.build_h4_holdout_variance_runner",
        fake_runner,
    )
    first = execute_h4_holdout_variance_parallel_worker_task(task)
    second = execute_h4_holdout_variance_parallel_worker_task(task)
    assert first.match_record is not None and second.match_record is not None
    assert len(opponents) == len(captures) == 2
    assert all(isinstance(opponent, ParametricPolicyAgent) for opponent in opponents)
    assert opponents[0] is not opponents[1]
    assert captures[0] is not captures[1]
