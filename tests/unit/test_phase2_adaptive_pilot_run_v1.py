from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pokerbench.engine.config import BASELINE_GAME_CONFIG
from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.experiments.adaptive_pilot_binding import (
    AdaptivePilotBindingError,
    AdaptivePilotExecutionBinding,
    build_adaptive_pilot_execution_binding,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotJob,
    AdaptivePilotProtocol,
    iter_adaptive_pilot_jobs,
    load_adaptive_pilot_protocol,
)
from pokerbench.experiments.adaptive_pilot_run import (
    ADAPTIVE_PILOT_ARM_SNAPSHOT_VERSION,
    ADAPTIVE_PILOT_PAIR_CHECKPOINT_VERSION,
    ADAPTIVE_PILOT_RUN_MANIFEST_VERSION,
    AdaptivePilotRunError,
    adaptive_pilot_job_key,
    adaptive_pilot_match_id,
    assert_adaptive_pilot_run_ready,
    execute_adaptive_pilot_run,
    initialize_adaptive_pilot_run,
    load_adaptive_pilot_arm_snapshot,
    load_adaptive_pilot_run_manifest,
)
from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.frozen_protocol import load_frozen_population
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import PairedResult, StatisticalUnit


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-v1.json"
PRIMARY_PATH = ROOT / "configs" / "populations" / "baseline-synthetic-v1.json"
SECONDARY_PATH = ROOT / "configs" / "populations" / "balanced-v1.json"
GATE6_PATH = ROOT / "data" / "gate6" / "gate6-report.json"


class _MemoryRunStore:
    def __init__(self) -> None:
        self.records: dict[str, MatchRecord] = {}

    def save_match(self, record: MatchRecord) -> None:
        if record.match_id in self.records:
            raise ValueError("duplicate match")
        self.records[record.match_id] = record

    def load_match(self, match_id: str) -> MatchRecord:
        return self.records[match_id]

    def list_match_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.records))


def _protocol() -> AdaptivePilotProtocol:
    return load_adaptive_pilot_protocol(PROTOCOL_PATH)


def _binding(*, workers: int = 1, commit_char: str = "a") -> AdaptivePilotExecutionBinding:
    commit = commit_char * 40
    environment = ExecutionEnvironment(
        python_version="3.14.0",
        engine_version=f"git:{commit}",
        dependency_lockfile="poetry.lock",
        poetry_lock_hash="b" * 64,
        git_commit=commit,
        cpu_model="Test CPU",
        cpu_core_count=8,
        cpu_architecture="AMD64",
        ram_bytes=16 * 1024**3,
        os_identifier="TestOS",
    )
    return build_adaptive_pilot_execution_binding(
        protocol=_protocol(),
        gate6_report=load_gate6_calibration_report(GATE6_PATH),
        primary_population=load_frozen_population(PRIMARY_PATH),
        secondary_populations=(load_frozen_population(SECONDARY_PATH),),
        environment=environment,
        workers=workers,
    )


def _failed_result(
    *,
    job: AdaptivePilotJob,
    config_hash: str,
    experiment_id: str,
    match_id: str,
) -> PairedResult:
    record = MatchRecord(
        match_id=match_id,
        master_seed=job.master_seed,
        game_config=BASELINE_GAME_CONFIG,
        hands=(),
        experiment_id=experiment_id,
        experiment_config_hash=config_hash,
    )
    return PairedResult(
        paired_differences=[],
        statistical_unit=StatisticalUnit.TRAJECTORY_CLUSTER,
        num_hands=job.hands_per_position,
        num_valid_pairs=0,
        failed_runs=[{"hand": 0, "config": "A_BTN", "details": "synthetic-test-failure"}],
        experiment_metadata={"experiment_config_hash": config_hash},
        seed=job.master_seed,
        match_record=record,
    )


def _persisting_failed_executor(calls: list[str]):
    def execute(job, config, match_id, store):  # type: ignore[no-untyped-def]
        calls.append(adaptive_pilot_job_key(job))
        result = _failed_result(
            job=job,
            config_hash=config.config_hash,
            experiment_id=config.run_provenance["experiment_id"],
            match_id=match_id,
        )
        assert result.match_record is not None
        store.save_match(result.match_record)
        return result

    return execute


def _manifest_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def test_initialize_run_freezes_exact_canonical_50_job_plan_and_18000_hands(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    manifest = initialize_adaptive_pilot_run(
        protocol=protocol,
        binding=binding,
        work_root=tmp_path,
    )
    jobs = iter_adaptive_pilot_jobs(protocol)

    assert manifest.manifest_version == ADAPTIVE_PILOT_RUN_MANIFEST_VERSION
    assert manifest.binding_hash == binding.binding_hash
    assert manifest.protocol_hash == protocol.payload_hash
    assert manifest.planned_job_count == 50
    assert manifest.planned_pair_count == 25
    assert manifest.planned_engine_hands == 18_000
    assert tuple(state.job_key for state in manifest.jobs) == tuple(
        adaptive_pilot_job_key(job) for job in jobs
    )
    assert all(state.status == "planned" for state in manifest.jobs)
    assert len({state.match_id for state in manifest.jobs}) == 50
    assert len(list((tmp_path / "trajectory-configs").glob("*.json"))) == 50
    assert (tmp_path / "binding.json").exists()


def test_manifest_is_canonical_round_trippable_and_semantic_tamper_evident(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    initialize_adaptive_pilot_run(protocol=protocol, binding=binding, work_root=tmp_path)
    path = tmp_path / "run-manifest.json"

    assert path.read_bytes().endswith(b"\n")
    assert not path.read_bytes().endswith(b"\n\n")
    loaded = load_adaptive_pilot_run_manifest(
        path,
        protocol=protocol,
        expected_binding=binding,
    )
    assert loaded.planned_job_count == 50

    raw = _manifest_payload(path)
    raw["manifest"]["jobs"].pop()
    raw["manifest_hash"] = _canonical_hash(raw["manifest"])
    _write_canonical(path, raw)
    with pytest.raises(AdaptivePilotRunError, match="canonical 50-job plan"):
        load_adaptive_pilot_run_manifest(
            path,
            protocol=protocol,
            expected_binding=binding,
        )


def test_readiness_requires_exact_binding_configs_and_serial_worker_binding(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding(workers=1)
    initialize_adaptive_pilot_run(protocol=protocol, binding=binding, work_root=tmp_path)
    readiness = assert_adaptive_pilot_run_ready(
        protocol=protocol,
        binding=binding,
        work_root=tmp_path,
        benchmark_store=_MemoryRunStore(),
    )

    assert readiness.ready
    assert readiness.planned_jobs == 50
    assert readiness.planned_pairs == 25
    assert readiness.planned_engine_hands == 18_000
    assert readiness.binding_hash == binding.binding_hash
    assert readiness.terminal_jobs == 0

    first_config = next((tmp_path / "trajectory-configs").glob("*.json"))
    first_config.unlink()
    with pytest.raises(AdaptivePilotRunError, match="trajectory config"):
        assert_adaptive_pilot_run_ready(
            protocol=protocol,
            binding=binding,
            work_root=tmp_path,
            benchmark_store=_MemoryRunStore(),
        )

    other_root = tmp_path / "workers"
    parallel_binding = _binding(workers=2)
    initialize_adaptive_pilot_run(
        protocol=protocol,
        binding=parallel_binding,
        work_root=other_root,
    )
    with pytest.raises(AdaptivePilotRunError, match="workers=1"):
        assert_adaptive_pilot_run_ready(
            protocol=protocol,
            binding=parallel_binding,
            work_root=other_root,
            benchmark_store=_MemoryRunStore(),
        )


def test_existing_run_cannot_resume_under_changed_execution_binding(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding(commit_char="a")
    initialize_adaptive_pilot_run(protocol=protocol, binding=binding, work_root=tmp_path)

    with pytest.raises(AdaptivePilotBindingError, match="binding"):
        initialize_adaptive_pilot_run(
            protocol=protocol,
            binding=_binding(commit_char="c"),
            work_root=tmp_path,
        )


def test_execute_checkpoints_all_jobs_and_pairs_without_duplicate_resume_evidence(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    calls: list[str] = []

    result = execute_adaptive_pilot_run(
        protocol=protocol,
        binding=binding,
        work_root=tmp_path,
        benchmark_store=store,
        job_executor=_persisting_failed_executor(calls),
    )

    assert len(calls) == 50
    assert result.terminal_jobs == 50
    assert result.completed_jobs == 0
    assert result.failed_jobs == 50
    assert result.pair_count == 25
    assert result.valid_pairs == 0
    assert result.invalid_pairs == 25
    assert len(store.records) == 50
    assert len(list((tmp_path / "arm-snapshots").glob("*.json"))) == 50
    assert len(list((tmp_path / "pair-checkpoints").glob("*.json"))) == 25

    second_calls: list[str] = []
    resumed = execute_adaptive_pilot_run(
        protocol=protocol,
        binding=binding,
        work_root=tmp_path,
        benchmark_store=store,
        job_executor=_persisting_failed_executor(second_calls),
    )
    assert second_calls == []
    assert resumed == result


def test_simulated_crash_resumes_from_stale_running_without_repeating_completed_jobs(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    jobs = iter_adaptive_pilot_jobs(protocol)
    first_calls: list[str] = []

    def crashing_executor(job, config, match_id, durable_store):  # type: ignore[no-untyped-def]
        first_calls.append(adaptive_pilot_job_key(job))
        if len(first_calls) == 4:
            raise RuntimeError("simulated crash")
        result = _failed_result(
            job=job,
            config_hash=config.config_hash,
            experiment_id=config.run_provenance["experiment_id"],
            match_id=match_id,
        )
        assert result.match_record is not None
        durable_store.save_match(result.match_record)
        return result

    with pytest.raises(RuntimeError, match="simulated crash"):
        execute_adaptive_pilot_run(
            protocol=protocol,
            binding=binding,
            work_root=tmp_path,
            benchmark_store=store,
            job_executor=crashing_executor,
        )

    manifest = load_adaptive_pilot_run_manifest(
        tmp_path / "run-manifest.json",
        protocol=protocol,
        expected_binding=binding,
    )
    assert [state.status for state in manifest.jobs[:4]] == ["failed", "failed", "failed", "running"]

    resumed_calls: list[str] = []
    execute_adaptive_pilot_run(
        protocol=protocol,
        binding=binding,
        work_root=tmp_path,
        benchmark_store=store,
        job_executor=_persisting_failed_executor(resumed_calls),
    )
    assert resumed_calls == [adaptive_pilot_job_key(job) for job in jobs[3:]]
    assert len(store.records) == 50


def test_resume_rejects_orphan_durable_match_without_arm_checkpoint(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()

    def crash_after_store(job, config, match_id, durable_store):  # type: ignore[no-untyped-def]
        result = _failed_result(
            job=job,
            config_hash=config.config_hash,
            experiment_id=config.run_provenance["experiment_id"],
            match_id=match_id,
        )
        assert result.match_record is not None
        durable_store.save_match(result.match_record)
        raise RuntimeError("crash after store")

    with pytest.raises(RuntimeError, match="crash after store"):
        execute_adaptive_pilot_run(
            protocol=protocol,
            binding=binding,
            work_root=tmp_path,
            benchmark_store=store,
            job_executor=crash_after_store,
        )

    with pytest.raises(AdaptivePilotRunError, match="orphan durable MatchRecord"):
        assert_adaptive_pilot_run_ready(
            protocol=protocol,
            binding=binding,
            work_root=tmp_path,
            benchmark_store=store,
        )


def test_resume_recovers_missing_store_record_from_immutable_arm_snapshot(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    calls: list[str] = []

    def crash_on_second(job, config, match_id, durable_store):  # type: ignore[no-untyped-def]
        if calls:
            raise RuntimeError("stop after first checkpoint")
        calls.append(adaptive_pilot_job_key(job))
        result = _failed_result(
            job=job,
            config_hash=config.config_hash,
            experiment_id=config.run_provenance["experiment_id"],
            match_id=match_id,
        )
        assert result.match_record is not None
        durable_store.save_match(result.match_record)
        return result

    with pytest.raises(RuntimeError, match="stop after first checkpoint"):
        execute_adaptive_pilot_run(
            protocol=protocol,
            binding=binding,
            work_root=tmp_path,
            benchmark_store=store,
            job_executor=crash_on_second,
        )

    first_match_id = next(iter(store.records))
    store.records.pop(first_match_id)
    resumed_calls: list[str] = []
    execute_adaptive_pilot_run(
        protocol=protocol,
        binding=binding,
        work_root=tmp_path,
        benchmark_store=store,
        job_executor=_persisting_failed_executor(resumed_calls),
    )
    assert first_match_id in store.records
    assert adaptive_pilot_job_key(iter_adaptive_pilot_jobs(protocol)[0]) not in resumed_calls
    assert len(store.records) == 50


def test_terminal_manifest_state_without_snapshot_fails_closed(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    calls: list[str] = []
    execute_adaptive_pilot_run(
        protocol=protocol,
        binding=binding,
        work_root=tmp_path,
        benchmark_store=store,
        job_executor=_persisting_failed_executor(calls),
    )

    first_snapshot = next((tmp_path / "arm-snapshots").glob("*.json"))
    first_snapshot.unlink()
    with pytest.raises(AdaptivePilotRunError, match="terminal manifest state has no arm snapshot"):
        assert_adaptive_pilot_run_ready(
            protocol=protocol,
            binding=binding,
            work_root=tmp_path,
            benchmark_store=store,
        )


def test_self_consistently_rehashed_snapshot_with_wrong_config_is_rejected(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    calls: list[str] = []
    execute_adaptive_pilot_run(
        protocol=protocol,
        binding=binding,
        work_root=tmp_path,
        benchmark_store=store,
        job_executor=_persisting_failed_executor(calls),
    )

    first_job = iter_adaptive_pilot_jobs(protocol)[0]
    path = tmp_path / "arm-snapshots" / f"{adaptive_pilot_job_key(first_job)}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["snapshot"]["trajectory_config_hash"] = "f" * 64
    raw["snapshot_hash"] = _canonical_hash(raw["snapshot"])
    _write_canonical(path, raw)

    with pytest.raises(AdaptivePilotRunError, match="trajectory config hash"):
        load_adaptive_pilot_arm_snapshot(
            path,
            protocol=protocol,
            expected_binding=binding,
            expected_job=first_job,
        )


def test_pair_checkpoint_is_not_created_until_both_arms_are_terminal(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    calls: list[str] = []

    def stop_before_treatment(job, config, match_id, durable_store):  # type: ignore[no-untyped-def]
        if job.arm_id == "treatment":
            raise RuntimeError("stop before treatment")
        return _persisting_failed_executor(calls)(job, config, match_id, durable_store)

    with pytest.raises(RuntimeError, match="stop before treatment"):
        execute_adaptive_pilot_run(
            protocol=protocol,
            binding=binding,
            work_root=tmp_path,
            benchmark_store=store,
            job_executor=stop_before_treatment,
        )

    assert len(calls) == 25
    assert list((tmp_path / "pair-checkpoints").glob("*.json")) == []

    resumed_calls: list[str] = []
    result = execute_adaptive_pilot_run(
        protocol=protocol,
        binding=binding,
        work_root=tmp_path,
        benchmark_store=store,
        job_executor=_persisting_failed_executor(resumed_calls),
    )
    assert len(resumed_calls) == 25
    assert result.pair_count == 25
    assert len(list((tmp_path / "pair-checkpoints").glob("*.json"))) == 25


def test_job_executor_must_durably_persist_exact_runner_evidence_before_checkpoint(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()

    def non_persisting_executor(job, config, match_id, durable_store):  # type: ignore[no-untyped-def]
        del durable_store
        return _failed_result(
            job=job,
            config_hash=config.config_hash,
            experiment_id=config.run_provenance["experiment_id"],
            match_id=match_id,
        )

    with pytest.raises(AdaptivePilotRunError, match="durably persisted"):
        execute_adaptive_pilot_run(
            protocol=protocol,
            binding=binding,
            work_root=tmp_path,
            benchmark_store=store,
            job_executor=non_persisting_executor,
        )
    assert list((tmp_path / "arm-snapshots").glob("*.json")) == []


def test_job_keys_match_ids_and_checkpoint_versions_are_stable_and_unique(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    manifest = initialize_adaptive_pilot_run(protocol=protocol, binding=binding, work_root=tmp_path)
    jobs = iter_adaptive_pilot_jobs(protocol)

    assert len({adaptive_pilot_job_key(job) for job in jobs}) == 50
    assert tuple(state.match_id for state in manifest.jobs) == tuple(
        adaptive_pilot_match_id(
            binding=binding,
            trajectory_config_hash=state.trajectory_config_hash,
        )
        for state in manifest.jobs
    )
    assert ADAPTIVE_PILOT_ARM_SNAPSHOT_VERSION.endswith("-v1")
    assert ADAPTIVE_PILOT_PAIR_CHECKPOINT_VERSION.endswith("-v1")
    assert ADAPTIVE_PILOT_RUN_MANIFEST_VERSION.endswith("-v1")
