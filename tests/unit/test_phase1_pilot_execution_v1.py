from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.frozen_protocol import (
    PilotJob,
    load_frozen_population,
    load_phase1_pilot_protocol,
)
from pokerbench.experiments.pilot_execution import (
    PHASE1_PILOT_EXECUTION_BINDING_VERSION,
    PHASE1_PILOT_EXECUTION_REPORT_VERSION,
    PHASE1_PILOT_JOB_SNAPSHOT_VERSION,
    PilotExecutionBinding,
    PilotExecutionError,
    PilotJobExecution,
    build_job_experiment_config,
    build_pilot_execution_binding,
    build_pilot_report,
    hardware_fingerprint,
    load_pilot_execution_binding,
    load_pilot_job_snapshot,
    load_phase1_pilot_execution_report,
    pilot_job_match_id,
    save_pilot_execution_binding,
    save_pilot_job_snapshot,
    save_phase1_pilot_execution_report,
)
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import StatisticalUnit


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase1-pilot-v1.json"
PRIMARY_PATH = ROOT / "configs" / "populations" / "baseline-synthetic-v1.json"
SECONDARY_PATH = ROOT / "configs" / "populations" / "balanced-v1.json"


def _protocol():
    primary = load_frozen_population(PRIMARY_PATH)
    secondary = load_frozen_population(SECONDARY_PATH)
    populations = {
        primary.population_id: primary,
        secondary.population_id: secondary,
    }
    return load_phase1_pilot_protocol(PROTOCOL_PATH, populations=populations)


def _environment() -> ExecutionEnvironment:
    return ExecutionEnvironment(
        python_version="3.14.0",
        engine_version="git:" + "a" * 40,
        dependency_lockfile="poetry.lock",
        poetry_lock_hash="b" * 64,
        git_commit="a" * 40,
        cpu_model="Test CPU",
        cpu_core_count=8,
        cpu_architecture="AMD64",
        ram_bytes=16 * 1024**3,
        os_identifier="TestOS",
    )


def _binding(*, workers: int = 4) -> PilotExecutionBinding:
    protocol = _protocol()
    return build_pilot_execution_binding(
        protocol=protocol,
        gate6_report_hash="c" * 64,
        gate6_reference_strength_hash="d" * 64,
        environment=_environment(),
        workers=workers,
    )


def _job(seed: int = 6_100_000) -> PilotJob:
    return PilotJob(
        cell_id="random-vs-nit",
        hero_id="random",
        hero_version="random-baseline-v1",
        opponent_profile_id="nit",
        opponent_profile_version="static-profiles-v2",
        master_seed=seed,
    )


def _record(job: PilotJob, binding: PilotExecutionBinding, config_hash: str) -> MatchRecord:
    from pokerbench.engine.config import BASELINE_GAME_CONFIG

    return MatchRecord(
        match_id=pilot_job_match_id(binding, job),
        experiment_id=f"phase1-pilot-v1:{binding.binding_hash}",
        experiment_config_hash=config_hash,
        master_seed=job.master_seed,
        game_config=BASELINE_GAME_CONFIG,
        hands=(),
    )


def test_binding_is_canonical_hash_bound_and_workers_are_execution_provenance(tmp_path: Path) -> None:
    first = _binding(workers=4)
    second = _binding(workers=4)
    different_workers = _binding(workers=8)

    assert first.binding_version == PHASE1_PILOT_EXECUTION_BINDING_VERSION
    assert first.binding_hash == second.binding_hash
    assert first.binding_hash != different_workers.binding_hash
    assert first.engine_version == "git:" + "a" * 40
    assert first.hardware_fingerprint == hardware_fingerprint(_environment())

    path = tmp_path / "binding.json"
    save_pilot_execution_binding(first, path)
    assert load_pilot_execution_binding(path) == first

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["binding"]["workers"] = 99
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PilotExecutionError, match="binding_hash"):
        load_pilot_execution_binding(path)


def test_job_experiment_config_contains_every_required_run_provenance() -> None:
    protocol = _protocol()
    binding = _binding()
    job = _job()
    raw_protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["protocol"]

    config = build_job_experiment_config(
        protocol=protocol,
        protocol_payload=raw_protocol,
        binding=binding,
        job=job,
        opponent_gate_map_hash="e" * 64,
        opponent_parameters={
            "vpip": 0.10,
            "pfr": 0.04,
            "schema_version": "opponent-parameters-v1",
        },
    )

    run_provenance = config["run_provenance"]
    assert set(protocol.required_run_provenance) <= set(run_provenance)
    assert run_provenance["master_seed"] == job.master_seed
    assert run_provenance["statistical_unit"] == StatisticalUnit.PAIRED_DEAL.value
    assert run_provenance["gate6_report_hash"] == binding.gate6_report_hash
    assert config["protocol"] == raw_protocol
    assert config["execution_binding"]["binding_hash"] == binding.binding_hash


def test_job_match_id_is_deterministic_and_bound_to_execution_identity() -> None:
    job = _job()
    assert pilot_job_match_id(_binding(workers=4), job) == pilot_job_match_id(
        _binding(workers=4), job
    )
    assert pilot_job_match_id(_binding(workers=4), job) != pilot_job_match_id(
        _binding(workers=8), job
    )


def test_job_snapshot_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    binding = _binding()
    job = _job()
    raw_protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["protocol"]
    config = build_job_experiment_config(
        protocol=_protocol(),
        protocol_payload=raw_protocol,
        binding=binding,
        job=job,
        opponent_gate_map_hash="e" * 64,
        opponent_parameters={"profile": "nit"},
    )
    from pokerbench.experiments.pilot_execution import sha256_canonical

    config_hash = sha256_canonical(config)
    record = _record(job, binding, config_hash)
    execution = PilotJobExecution.from_runner_result(
        job=job,
        binding=binding,
        result_paired_differences=(1.25,),
        num_valid_pairs=1,
        failed_runs=(),
        elapsed_ns=123_456,
        match_record=record,
    )

    path = tmp_path / "job.json"
    save_pilot_job_snapshot(execution, path)
    loaded = load_pilot_job_snapshot(path, binding=binding, expected_job=job)

    assert loaded.snapshot_version == PHASE1_PILOT_JOB_SNAPSHOT_VERSION
    assert loaded.status == "valid"
    assert loaded.paired_difference_bb == 1.25
    assert loaded.match_record_hash == record.record_hash

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["snapshot"]["elapsed_ns"] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PilotExecutionError, match="snapshot_hash"):
        load_pilot_job_snapshot(path, binding=binding, expected_job=job)


def test_failed_job_snapshot_is_completed_evidence_not_a_missing_seed() -> None:
    binding = _binding()
    job = _job()
    config_hash = "f" * 64
    record = _record(job, binding, config_hash)

    execution = PilotJobExecution.from_runner_result(
        job=job,
        binding=binding,
        result_paired_differences=(),
        num_valid_pairs=0,
        failed_runs=({"hand": 0, "config": "A_BTN", "details": "failed"},),
        elapsed_ns=99,
        match_record=record,
    )

    assert execution.status == "failed"
    assert execution.paired_difference_bb is None
    assert execution.completed
    assert execution.master_seed == 6_100_000


def test_pilot_report_computes_only_ddof1_variance_and_runtime_outputs() -> None:
    protocol = replace(
        _protocol(),
        hero_agents=(_protocol().hero_agents[0],),
        opponent_profile_ids=("nit",),
        master_seeds=(1, 2, 3),
        paired_units_per_cell=3,
        cell_count=1,
        planned_independent_units=3,
        planned_simulated_engine_hands=6,
    )
    binding = _binding()
    jobs = tuple(
        replace(_job(seed), master_seed=seed)
        for seed in (1, 2, 3)
    )
    executions = tuple(
        PilotJobExecution(
            cell_id=job.cell_id,
            hero_id=job.hero_id,
            hero_version=job.hero_version,
            opponent_profile_id=job.opponent_profile_id,
            opponent_profile_version=job.opponent_profile_version,
            master_seed=job.master_seed,
            match_id=f"m-{job.master_seed}",
            experiment_config_hash="1" * 64,
            match_record_hash="2" * 64,
            status="valid",
            paired_difference_bb=value,
            failed_runs=(),
            elapsed_ns=elapsed,
            recorded_engine_hands=2,
            binding_hash=binding.binding_hash,
        )
        for job, value, elapsed in zip(jobs, (1.0, 2.0, 3.0), (10, 20, 30), strict=True)
    )

    report = build_pilot_report(protocol=protocol, binding=binding, jobs=jobs, executions=executions)

    assert report.report_version == PHASE1_PILOT_EXECUTION_REPORT_VERSION
    assert report.complete
    assert report.failed_units == 0
    assert len(report.cells) == 1
    cell = report.cells[0]
    assert cell.valid_units == 3
    assert cell.std_difference_bb100 == pytest.approx(100.0)
    assert cell.total_runtime_ns == 60
    assert cell.median_runtime_ns == 20
    payload = report.payload()
    assert "mean_difference" not in json.dumps(payload)
    assert payload["scientific_use"]["contains_final_benchmark_estimate"] is False
    assert payload["scientific_use"]["contains_confirmatory_inference"] is False


def test_pilot_report_retains_failures_without_replacement_and_is_deterministic() -> None:
    protocol = replace(
        _protocol(),
        hero_agents=(_protocol().hero_agents[0],),
        opponent_profile_ids=("nit",),
        master_seeds=(1, 2),
        paired_units_per_cell=2,
        cell_count=1,
        planned_independent_units=2,
        planned_simulated_engine_hands=4,
    )
    binding = _binding()
    jobs = tuple(replace(_job(seed), master_seed=seed) for seed in (1, 2))
    valid = PilotJobExecution(
        cell_id=jobs[0].cell_id,
        hero_id=jobs[0].hero_id,
        hero_version=jobs[0].hero_version,
        opponent_profile_id=jobs[0].opponent_profile_id,
        opponent_profile_version=jobs[0].opponent_profile_version,
        master_seed=1,
        match_id="m-1",
        experiment_config_hash="1" * 64,
        match_record_hash="2" * 64,
        status="valid",
        paired_difference_bb=2.0,
        failed_runs=(),
        elapsed_ns=10,
        recorded_engine_hands=2,
        binding_hash=binding.binding_hash,
    )
    failed = replace(
        valid,
        master_seed=2,
        match_id="m-2",
        status="failed",
        paired_difference_bb=None,
        failed_runs=({"details": "failed"},),
        elapsed_ns=11,
        recorded_engine_hands=1,
    )

    first = build_pilot_report(
        protocol=protocol,
        binding=binding,
        jobs=jobs,
        executions=(valid, failed),
    )
    second = build_pilot_report(
        protocol=protocol,
        binding=binding,
        jobs=tuple(reversed(jobs)),
        executions=(failed, valid),
    )

    assert first.report_hash == second.report_hash
    assert first.complete
    assert first.failed_units == 1
    assert first.valid_units == 1
    assert first.cells[0].std_difference_bb100 is None
    assert first.seeds == (1, 2)


def test_report_round_trip_is_hash_verified(tmp_path: Path) -> None:
    protocol = replace(
        _protocol(),
        hero_agents=(_protocol().hero_agents[0],),
        opponent_profile_ids=("nit",),
        master_seeds=(1,),
        paired_units_per_cell=1,
        cell_count=1,
        planned_independent_units=1,
        planned_simulated_engine_hands=2,
    )
    binding = _binding()
    job = replace(_job(1), master_seed=1)
    execution = PilotJobExecution(
        cell_id=job.cell_id,
        hero_id=job.hero_id,
        hero_version=job.hero_version,
        opponent_profile_id=job.opponent_profile_id,
        opponent_profile_version=job.opponent_profile_version,
        master_seed=1,
        match_id="m-1",
        experiment_config_hash="1" * 64,
        match_record_hash="2" * 64,
        status="failed",
        paired_difference_bb=None,
        failed_runs=({"details": "failed"},),
        elapsed_ns=10,
        recorded_engine_hands=1,
        binding_hash=binding.binding_hash,
    )
    report = build_pilot_report(
        protocol=protocol,
        binding=binding,
        jobs=(job,),
        executions=(execution,),
    )

    path = tmp_path / "report.json"
    save_phase1_pilot_execution_report(report, path)
    loaded = load_phase1_pilot_execution_report(path)
    assert loaded.report_hash == report.report_hash

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["report"]["failed_units"] = 0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PilotExecutionError, match="report_hash"):
        load_phase1_pilot_execution_report(path)
