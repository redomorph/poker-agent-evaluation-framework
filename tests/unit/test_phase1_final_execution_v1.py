from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.final_execution import (
    PHASE1_FINAL_EXECUTION_BINDING_VERSION,
    PHASE1_FINAL_EXECUTION_REPORT_VERSION,
    PHASE1_FINAL_JOB_SNAPSHOT_VERSION,
    FinalExecutionError,
    FinalJobExecution,
    build_final_execution_binding,
    build_final_job_experiment_config,
    build_final_report,
    final_job_match_id,
    load_final_execution_binding,
    load_final_job_snapshot,
    load_phase1_final_execution_report,
    save_final_execution_binding,
    save_final_job_snapshot,
    save_phase1_final_execution_report,
    sha256_canonical,
)
from pokerbench.experiments.final_protocol import (
    FinalAgentPlan,
    FinalJob,
    load_phase1_baseline_protocol,
)
from pokerbench.experiments.frozen_protocol import load_frozen_population
from pokerbench.replay.archive import MatchRecord

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase1-baseline-experiment-v1.json"
PRIMARY_PATH = ROOT / "configs" / "populations" / "baseline-synthetic-v1.json"
SECONDARY_PATH = ROOT / "configs" / "populations" / "balanced-v1.json"


def _inputs():
    primary = load_frozen_population(PRIMARY_PATH)
    secondary = load_frozen_population(SECONDARY_PATH)
    protocol = load_phase1_baseline_protocol(
        PROTOCOL_PATH,
        populations={primary.population_id: primary, secondary.population_id: secondary},
    )
    return protocol, primary, secondary


def _environment() -> ExecutionEnvironment:
    return ExecutionEnvironment(
        python_version="3.14.0", engine_version="git:" + "a" * 40,
        dependency_lockfile="poetry.lock", poetry_lock_hash="b" * 64,
        git_commit="a" * 40, cpu_model="Test CPU", cpu_core_count=8,
        cpu_architecture="AMD64", ram_bytes=16 * 1024**3, os_identifier="TestOS",
    )


def _binding(workers: int = 4):
    protocol, primary, secondary = _inputs()
    return build_final_execution_binding(
        protocol=protocol, primary_population=primary, secondary_population=secondary,
        gate6_report_hash=protocol.gate6_report_hash,
        gate6_reference_strength_hash="d" * 64,
        environment=_environment(), workers=workers,
    )


def _job(profile: str = "nit", seed: int = 6_200_000) -> FinalJob:
    return FinalJob(
        cell_id=f"random-vs-{profile}", hero_id="random",
        hero_version="random-baseline-v1", opponent_profile_id=profile,
        opponent_profile_version="static-profiles-v2", master_seed=seed,
    )


def _record(job: FinalJob, binding, config_hash: str) -> MatchRecord:
    from pokerbench.engine.config import BASELINE_GAME_CONFIG
    return MatchRecord(
        match_id=final_job_match_id(binding, job),
        experiment_id=f"phase1-baseline-experiment-v1:{binding.binding_hash}",
        experiment_config_hash=config_hash, master_seed=job.master_seed,
        game_config=BASELINE_GAME_CONFIG, hands=(),
    )


def test_final_binding_is_hash_bound_and_includes_pilot_identity(tmp_path: Path) -> None:
    first = _binding(4)
    assert first.binding_version == PHASE1_FINAL_EXECUTION_BINDING_VERSION
    assert first.pilot_report_hash == _inputs()[0].pilot_report_hash
    assert first.binding_hash != _binding(8).binding_hash

    path = tmp_path / "binding.json"
    save_final_execution_binding(first, path)
    assert load_final_execution_binding(path) == first

    raw = json.loads(path.read_text())
    raw["binding"]["workers"] = 99
    path.write_text(json.dumps(raw))
    with pytest.raises(FinalExecutionError, match="binding_hash"):
        load_final_execution_binding(path)


def test_final_job_config_embeds_protocol_binding_pilot_and_precision() -> None:
    protocol, _, _ = _inputs()
    raw_protocol = json.loads(PROTOCOL_PATH.read_text())["protocol"]
    config = build_final_job_experiment_config(
        protocol=protocol, protocol_payload=raw_protocol, binding=_binding(),
        job=_job(), opponent_gate_map_hash="e" * 64,
        opponent_parameters={"profile": "nit"},
    )
    assert config["protocol"] == raw_protocol
    assert config["execution_binding"]["pilot_report_hash"] == protocol.pilot_report_hash
    assert config["run_provenance"]["master_seed"] == 6_200_000
    assert config["scientific_use"]["pilot_data_included"] is False
    assert config["scientific_use"]["optional_stopping"] is False


def test_final_checkpoint_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    binding = _binding()
    job = _job()
    config_hash = "f" * 64
    execution = FinalJobExecution.from_runner_result(
        job=job, binding=binding, result_paired_differences=(1.25,),
        num_valid_pairs=1, failed_runs=(), elapsed_ns=123,
        match_record=_record(job, binding, config_hash),
    )
    path = tmp_path / "job.json"
    save_final_job_snapshot(execution, path)
    loaded = load_final_job_snapshot(path, binding=binding, expected_job=job)
    assert loaded.snapshot_version == PHASE1_FINAL_JOB_SNAPSHOT_VERSION
    assert loaded.paired_difference_bb == 1.25

    raw = json.loads(path.read_text())
    raw["snapshot"]["elapsed_ns"] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(FinalExecutionError, match="snapshot_hash"):
        load_final_job_snapshot(path, binding=binding, expected_job=job)


def _small_protocol():
    protocol, _, _ = _inputs()
    plan = FinalAgentPlan(
        agent_id="random", agent_version="random-baseline-v1",
        paired_units_per_profile=2, seed_start=6_200_000, seed_count=2,
        raw_required_units_per_profile=2.0, planned_half_width_bb100=50.0,
    )
    return replace(
        protocol, agent_plans=(plan,), planned_independent_units=10,
        planned_simulated_engine_hands=20,
    )


def _small_jobs() -> tuple[FinalJob, ...]:
    profiles = ("nit", "tag", "lag", "calling_station", "maniac")
    return tuple(_job(profile, 6_200_000 + offset) for profile in profiles for offset in (0, 1))


def _execution_for(job: FinalJob, value: float, *, failed: bool = False) -> FinalJobExecution:
    binding = _binding()
    return FinalJobExecution(
        cell_id=job.cell_id, hero_id=job.hero_id, hero_version=job.hero_version,
        opponent_profile_id=job.opponent_profile_id,
        opponent_profile_version=job.opponent_profile_version,
        master_seed=job.master_seed, match_id=final_job_match_id(binding, job),
        experiment_config_hash="1" * 64, match_record_hash="2" * 64,
        status="failed" if failed else "valid",
        paired_difference_bb=None if failed else value,
        failed_runs=({"details": "failed"},) if failed else (), elapsed_ns=10,
        recorded_engine_hands=1 if failed else 2, binding_hash=binding.binding_hash,
    )


def test_final_report_produces_population_results_and_separate_precision_diagnostic() -> None:
    protocol, primary, secondary = _inputs()
    protocol = _small_protocol()
    binding = _binding()
    jobs = _small_jobs()
    values = {
        "nit": (1.0, 1.2), "tag": (2.0, 2.2), "lag": (3.0, 3.2),
        "calling_station": (4.0, 4.2), "maniac": (5.0, 5.2),
    }
    executions = tuple(
        _execution_for(job, values[job.opponent_profile_id][job.master_seed - 6_200_000])
        for job in jobs
    )
    report = build_final_report(
        protocol=protocol, binding=binding, primary_population=primary,
        secondary_population=secondary, jobs=jobs, executions=executions,
    )
    assert report.report_version == PHASE1_FINAL_EXECUTION_REPORT_VERSION
    assert report.failure_free
    hero = report.heroes[0]
    assert hero.primary_population is not None
    assert hero.secondary_population is not None
    assert hero.precision is not None
    assert hero.primary_population.tail_precision_warning
    assert hero.precision.mean_ev_bb100 == pytest.approx(hero.primary_population.mean_ev_bb100)
    assert hero.precision.estimator_version == "fixed-profile-normal-mean-ci-v1"
    assert report.payload()["precision_disclosure"]["frozen_target_half_width_bb100"] == 50.0
    assert report.payload()["precision_disclosure"]["project_documentation_illustrative_half_width_bb100"] == 1.0


def test_failed_final_seed_is_retained_and_final_estimands_are_withheld() -> None:
    protocol, primary, secondary = _inputs()
    protocol = _small_protocol()
    binding = _binding()
    jobs = _small_jobs()
    executions = tuple(
        _execution_for(job, 1.0, failed=(index == 0))
        for index, job in enumerate(jobs)
    )
    report = build_final_report(
        protocol=protocol, binding=binding, primary_population=primary,
        secondary_population=secondary, jobs=jobs, executions=executions,
    )
    assert report.complete
    assert not report.failure_free
    assert report.failed_units == 1
    assert report.heroes[0].primary_population is None
    assert report.heroes[0].precision is None


def test_final_report_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    protocol, primary, secondary = _inputs()
    protocol = _small_protocol()
    binding = _binding()
    jobs = _small_jobs()
    report = build_final_report(
        protocol=protocol, binding=binding, primary_population=primary,
        secondary_population=secondary, jobs=jobs,
        executions=tuple(_execution_for(job, 1.0 + 0.1 * i) for i, job in enumerate(jobs)),
    )
    path = tmp_path / "report.json"
    save_phase1_final_execution_report(report, path)
    loaded = load_phase1_final_execution_report(path)
    assert loaded.report_hash == report.report_hash

    raw = json.loads(path.read_text())
    raw["report"]["failed_units"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(FinalExecutionError, match="report_hash"):
        load_phase1_final_execution_report(path)


def test_final_job_match_id_is_bound_to_execution_identity() -> None:
    assert final_job_match_id(_binding(4), _job()) == final_job_match_id(_binding(4), _job())
    assert final_job_match_id(_binding(4), _job()) != final_job_match_id(_binding(8), _job())
