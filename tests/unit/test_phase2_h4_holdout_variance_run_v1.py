from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pokerbench.engine.config import BASELINE_GAME_CONFIG
from pokerbench.evaluation.calibration_snapshots import (
    CalibrationStrengthSnapshotStore,
    policy_strength_reference_from_store,
    policy_validation_strengths_from_store,
)
from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.evaluation.h4_holdout_acceptance import load_h4_holdout_acceptance
from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.h4_holdout_population import load_h4_holdout_population
from pokerbench.experiments.h4_holdout_variance_binding import (
    H4HoldoutVarianceExecutionAuthorityReference,
    build_h4_holdout_variance_execution_binding,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4HoldoutVarianceJob,
    iter_h4_holdout_variance_jobs,
    load_h4_holdout_variance_protocol,
)
from pokerbench.experiments.h4_holdout_variance_run import (
    H4_HOLDOUT_VARIANCE_ARM_SNAPSHOT_VERSION,
    H4_HOLDOUT_VARIANCE_CLUSTER_CHECKPOINT_VERSION,
    H4_HOLDOUT_VARIANCE_PAIR_CHECKPOINT_VERSION,
    H4_HOLDOUT_VARIANCE_RUN_MANIFEST_VERSION,
    H4HoldoutVarianceRunError,
    assert_h4_holdout_variance_run_ready,
    execute_h4_holdout_variance_run,
    h4_holdout_variance_job_key,
    initialize_h4_holdout_variance_run,
    load_h4_holdout_variance_run_manifest,
)
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import PairedResult, StatisticalUnit


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-h4-holdout-variance-pilot-v1.json"
POPULATION_PATH = ROOT / "configs" / "populations" / "phase2-h4-holdout-population-v1.json"
ACCEPTANCE_PATH = ROOT / "data" / "gate6" / "h4-holdout-acceptance-v1.json"
GATE6_PATH = ROOT / "data" / "gate6" / "gate6-report.json"
SNAPSHOT_ROOT = ROOT / "data" / "gate6" / "snapshots"


class _MemoryRunStore:
    def __init__(self) -> None:
        self.records: dict[str, MatchRecord] = {}
        self.fail_next_save = False

    def save_match(self, record: MatchRecord) -> None:
        if self.fail_next_save:
            self.fail_next_save = False
            raise RuntimeError("synthetic durable-store outage")
        if record.match_id in self.records:
            raise ValueError("duplicate match")
        self.records[record.match_id] = record

    def load_match(self, match_id: str) -> MatchRecord:
        return self.records[match_id]

    def list_match_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.records))


def _protocol():
    return load_h4_holdout_variance_protocol(PROTOCOL_PATH)


def _population():
    return load_h4_holdout_population(POPULATION_PATH)


def _acceptance():
    population = _population()
    report = load_gate6_calibration_report(GATE6_PATH)
    store = CalibrationStrengthSnapshotStore(SNAPSHOT_ROOT)
    return load_h4_holdout_acceptance(
        ACCEPTANCE_PATH,
        population=population,
        reference=policy_strength_reference_from_store(store),
        validation_strengths=policy_validation_strengths_from_store(store),
        gate6_report_hash=report.report_hash,
    )


def _binding(*, commit_char: str = "a"):
    protocol = _protocol()
    commit = commit_char * 40
    return build_h4_holdout_variance_execution_binding(
        protocol=protocol,
        population=_population(),
        acceptance=_acceptance(),
        environment=ExecutionEnvironment(
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
        ),
        workers=4,
        process_model="spawn-compatible-process-pool-v1",
        persistence_paths={
            "work_root": "data/phase2/h4-holdout-variance/work",
            "database": "data/phase2/h4-holdout-variance/variance-pilot.sqlite",
            "trajectory_configs": "data/phase2/h4-holdout-variance/work/trajectory-configs",
            "arm_snapshots": "data/phase2/h4-holdout-variance/work/arm-snapshots",
            "pair_checkpoints": "data/phase2/h4-holdout-variance/work/pair-checkpoints",
            "cluster_checkpoints": "data/phase2/h4-holdout-variance/work/cluster-checkpoints",
        },
        execution_authority=H4HoldoutVarianceExecutionAuthorityReference(
            authority_id="test-authority",
            authority_version="test-authority-v1",
            payload_hash="c" * 64,
            authorized_protocol_id=protocol.protocol_id,
            authorized_protocol_version=protocol.protocol_version,
            authorized_protocol_hash=protocol.payload_hash,
        ),
    )


def _failed_result(*, job: H4HoldoutVarianceJob, config_hash: str, experiment_id: str, match_id: str) -> PairedResult:
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
        num_hands=500,
        num_valid_pairs=0,
        failed_runs=[{"hand": 0, "config": "A_BTN", "details": "synthetic-test-failure"}],
        experiment_metadata={"experiment_config_hash": config_hash},
        seed=job.master_seed,
        match_record=record,
    )


def _failed_executor(calls: list[str]):
    def execute(job, config, match_id):  # type: ignore[no-untyped-def]
        calls.append(h4_holdout_variance_job_key(job))
        return _failed_result(
            job=job,
            config_hash=config.config_hash,
            experiment_id=config.run_provenance["experiment_id"],
            match_id=match_id,
        )
    return execute


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def test_initialize_run_freezes_exact_100_job_50_pair_10_cluster_100000_hand_plan(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    manifest = initialize_h4_holdout_variance_run(
        protocol=protocol,
        binding=binding,
        population=_population(),
        work_root=tmp_path,
    )
    assert manifest.manifest_version == H4_HOLDOUT_VARIANCE_RUN_MANIFEST_VERSION
    assert manifest.planned_job_count == 100
    assert manifest.planned_pair_count == 50
    assert manifest.planned_cluster_count == 10
    assert manifest.planned_engine_hands == 100_000
    assert tuple(state.job_key for state in manifest.jobs) == tuple(
        h4_holdout_variance_job_key(job) for job in iter_h4_holdout_variance_jobs(protocol)
    )
    assert all(state.status == "planned" for state in manifest.jobs)
    assert len({state.match_id for state in manifest.jobs}) == 100
    assert len(list((tmp_path / "trajectory-configs").glob("*.json"))) == 100


def test_manifest_roundtrip_rejects_rehashed_semantic_plan_drift(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    initialize_h4_holdout_variance_run(protocol=protocol, binding=binding, population=_population(), work_root=tmp_path)
    path = tmp_path / "run-manifest.json"
    manifest = load_h4_holdout_variance_run_manifest(path, protocol=protocol, expected_binding=binding, population=_population())
    assert manifest.planned_job_count == 100

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["manifest"]["jobs"].pop()
    raw["manifest_hash"] = _canonical_hash(raw["manifest"])
    path.write_bytes(_canonical_bytes(raw))
    with pytest.raises(H4HoldoutVarianceRunError, match="canonical 100-job plan"):
        load_h4_holdout_variance_run_manifest(path, protocol=protocol, expected_binding=binding, population=_population())


def test_readiness_is_nonexecuting_and_requires_exact_binding_configs_and_empty_unexpected_artifacts(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    initialize_h4_holdout_variance_run(protocol=protocol, binding=binding, population=_population(), work_root=tmp_path)
    store = _MemoryRunStore()
    readiness = assert_h4_holdout_variance_run_ready(
        protocol=protocol,
        binding=binding,
        population=_population(),
        work_root=tmp_path,
        benchmark_store=store,
    )
    assert readiness.ready
    assert readiness.planned_jobs == 100
    assert readiness.planned_pairs == 50
    assert readiness.planned_clusters == 10
    assert readiness.planned_engine_hands == 100_000
    assert readiness.terminal_jobs == 0
    assert not store.records

    (tmp_path / "arm-snapshots" / "foreign.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(H4HoldoutVarianceRunError, match="unexpected arm snapshot"):
        assert_h4_holdout_variance_run_ready(
            protocol=protocol,
            binding=binding,
            population=_population(),
            work_root=tmp_path,
            benchmark_store=store,
        )


def test_existing_plan_cannot_resume_under_changed_execution_binding(tmp_path: Path) -> None:
    initialize_h4_holdout_variance_run(protocol=_protocol(), binding=_binding(commit_char="a"), population=_population(), work_root=tmp_path)
    with pytest.raises(Exception, match="binding"):
        initialize_h4_holdout_variance_run(protocol=_protocol(), binding=_binding(commit_char="d"), population=_population(), work_root=tmp_path)


def test_execute_checkpoints_all_failed_jobs_pairs_and_clusters_without_duplicate_resume(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    calls: list[str] = []
    result = execute_h4_holdout_variance_run(
        protocol=protocol,
        binding=binding,
        population=_population(),
        work_root=tmp_path,
        benchmark_store=store,
        job_executor=_failed_executor(calls),
    )
    assert len(calls) == 100
    assert result.terminal_jobs == 100
    assert result.completed_jobs == 0
    assert result.failed_jobs == 100
    assert result.pair_count == 50
    assert result.valid_pairs == 0
    assert result.invalid_pairs == 50
    assert result.cluster_count == 10
    assert result.valid_clusters == 0
    assert result.invalid_clusters == 10
    assert result.scientifically_complete is False
    assert len(store.records) == 100
    assert len(list((tmp_path / "arm-snapshots").glob("*.json"))) == 100
    assert len(list((tmp_path / "pair-checkpoints").glob("*.json"))) == 50
    assert len(list((tmp_path / "cluster-checkpoints").glob("*.json"))) == 10

    second_calls: list[str] = []
    resumed = execute_h4_holdout_variance_run(
        protocol=protocol,
        binding=binding,
        population=_population(),
        work_root=tmp_path,
        benchmark_store=store,
        job_executor=_failed_executor(second_calls),
    )
    assert second_calls == []
    assert resumed == result


def test_crash_before_snapshot_resets_stale_running_and_reexecutes_only_uncheckpointed_jobs(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    jobs = iter_h4_holdout_variance_jobs(protocol)
    calls: list[str] = []

    def crashing(job, config, match_id):  # type: ignore[no-untyped-def]
        calls.append(h4_holdout_variance_job_key(job))
        if len(calls) == 4:
            raise RuntimeError("synthetic crash")
        return _failed_result(job=job, config_hash=config.config_hash, experiment_id=config.run_provenance["experiment_id"], match_id=match_id)

    with pytest.raises(RuntimeError, match="synthetic crash"):
        execute_h4_holdout_variance_run(
            protocol=protocol, binding=binding, population=_population(), work_root=tmp_path,
            benchmark_store=store, job_executor=crashing,
        )
    manifest = load_h4_holdout_variance_run_manifest(tmp_path / "run-manifest.json", protocol=protocol, expected_binding=binding, population=_population())
    assert [state.status for state in manifest.jobs[:4]] == ["failed", "failed", "failed", "running"]

    resumed_calls: list[str] = []
    execute_h4_holdout_variance_run(
        protocol=protocol, binding=binding, population=_population(), work_root=tmp_path,
        benchmark_store=store, job_executor=_failed_executor(resumed_calls),
    )
    assert resumed_calls == [h4_holdout_variance_job_key(job) for job in jobs[3:]]


def test_snapshot_before_store_allows_recovery_without_duplicate_reexecution(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    store = _MemoryRunStore()
    store.fail_next_save = True
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="synthetic durable-store outage"):
        execute_h4_holdout_variance_run(
            protocol=protocol, binding=binding, population=_population(), work_root=tmp_path,
            benchmark_store=store, job_executor=_failed_executor(calls),
        )
    assert len(calls) == 1
    assert len(list((tmp_path / "arm-snapshots").glob("*.json"))) == 1
    assert not store.records

    resumed_calls: list[str] = []
    execute_h4_holdout_variance_run(
        protocol=protocol, binding=binding, population=_population(), work_root=tmp_path,
        benchmark_store=store, job_executor=_failed_executor(resumed_calls),
    )
    assert h4_holdout_variance_job_key(iter_h4_holdout_variance_jobs(protocol)[0]) not in resumed_calls
    assert len(store.records) == 100


def test_orphan_durable_match_without_snapshot_fails_closed(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    population = _population()
    manifest = initialize_h4_holdout_variance_run(protocol=protocol, binding=binding, population=population, work_root=tmp_path)
    first_job = iter_h4_holdout_variance_jobs(protocol)[0]
    first_state = manifest.jobs[0]
    config_path = tmp_path / "trajectory-configs" / f"{first_state.job_key}.json"
    config_hash = json.loads(config_path.read_text(encoding="utf-8"))["config_hash"]
    record = MatchRecord(
        match_id=first_state.match_id,
        master_seed=first_job.master_seed,
        game_config=BASELINE_GAME_CONFIG,
        hands=(),
        experiment_id=f"{protocol.protocol_id}:{binding.binding_hash}",
        experiment_config_hash=config_hash,
    )
    store = _MemoryRunStore()
    store.save_match(record)
    with pytest.raises(H4HoldoutVarianceRunError, match="orphan durable MatchRecord"):
        assert_h4_holdout_variance_run_ready(
            protocol=protocol, binding=binding, population=population, work_root=tmp_path, benchmark_store=store,
        )


def test_checkpoint_versions_are_explicit_and_frozen() -> None:
    assert H4_HOLDOUT_VARIANCE_ARM_SNAPSHOT_VERSION == "phase2-h4-holdout-variance-arm-snapshot-v1"
    assert H4_HOLDOUT_VARIANCE_PAIR_CHECKPOINT_VERSION == "phase2-h4-holdout-variance-pair-checkpoint-v1"
    assert H4_HOLDOUT_VARIANCE_CLUSTER_CHECKPOINT_VERSION == "phase2-h4-holdout-variance-cluster-checkpoint-v1"
    assert H4_HOLDOUT_VARIANCE_RUN_MANIFEST_VERSION == "phase2-h4-holdout-variance-run-manifest-v1"
