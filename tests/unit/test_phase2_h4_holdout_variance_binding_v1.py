from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

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
    H4_HOLDOUT_VARIANCE_EXECUTION_BINDING_VERSION,
    H4_HOLDOUT_VARIANCE_TRAJECTORY_CONFIG_VERSION,
    H4HoldoutVarianceBindingError,
    H4HoldoutVarianceExecutionAuthorityReference,
    build_h4_holdout_variance_execution_binding,
    build_h4_holdout_variance_trajectory_config,
    capture_h4_holdout_variance_execution_binding,
    load_h4_holdout_variance_execution_binding,
    load_h4_holdout_variance_trajectory_config,
    save_h4_holdout_variance_execution_binding,
    save_h4_holdout_variance_trajectory_config,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4HoldoutVarianceJob,
    iter_h4_holdout_variance_jobs,
    load_h4_holdout_variance_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-h4-holdout-variance-pilot-v1.json"
POPULATION_PATH = ROOT / "configs" / "populations" / "phase2-h4-holdout-population-v1.json"
ACCEPTANCE_PATH = ROOT / "data" / "gate6" / "h4-holdout-acceptance-v1.json"
GATE6_PATH = ROOT / "data" / "gate6" / "gate6-report.json"
SNAPSHOT_ROOT = ROOT / "data" / "gate6" / "snapshots"


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


def _environment(*, commit_char: str = "a", engine_version: str | None = None) -> ExecutionEnvironment:
    commit = commit_char * 40
    return ExecutionEnvironment(
        python_version="3.14.0",
        engine_version=engine_version or f"git:{commit}",
        dependency_lockfile="poetry.lock",
        poetry_lock_hash="b" * 64,
        git_commit=commit,
        cpu_model="Test CPU",
        cpu_core_count=8,
        cpu_architecture="AMD64",
        ram_bytes=16 * 1024**3,
        os_identifier="TestOS",
    )


def _authority(*, protocol_hash: str | None = None):
    protocol = _protocol()
    return H4HoldoutVarianceExecutionAuthorityReference(
        authority_id="phase2-h4-holdout-variance-execution-test-authority",
        authority_version="phase2-h4-holdout-variance-execution-test-authority-v1",
        payload_hash="c" * 64,
        authorized_protocol_id=protocol.protocol_id,
        authorized_protocol_version=protocol.protocol_version,
        authorized_protocol_hash=protocol_hash or protocol.payload_hash,
    )


def _persistence_paths() -> dict[str, str]:
    return {
        "work_root": "data/phase2/h4-holdout-variance/work",
        "database": "data/phase2/h4-holdout-variance/variance-pilot.sqlite",
        "trajectory_configs": "data/phase2/h4-holdout-variance/work/trajectory-configs",
        "arm_snapshots": "data/phase2/h4-holdout-variance/work/arm-snapshots",
        "pair_checkpoints": "data/phase2/h4-holdout-variance/work/pair-checkpoints",
        "cluster_checkpoints": "data/phase2/h4-holdout-variance/work/cluster-checkpoints",
    }


def _binding(*, workers: int = 4, commit_char: str = "a"):
    return build_h4_holdout_variance_execution_binding(
        protocol=_protocol(),
        population=_population(),
        acceptance=_acceptance(),
        environment=_environment(commit_char=commit_char),
        workers=workers,
        process_model="spawn-compatible-process-pool-v1",
        persistence_paths=_persistence_paths(),
        execution_authority=_authority(),
    )


def _job(*, arm_id: str = "treatment", unit_id: str = "h4_holdout_003", seed: int = 6_400_007) -> H4HoldoutVarianceJob:
    return next(
        job
        for job in iter_h4_holdout_variance_jobs(_protocol())
        if job.arm_id == arm_id and job.holdout_unit_id == unit_id and job.master_seed == seed
    )


def test_binding_captures_exact_spec37_population_acceptance_rng_dependencies_and_environment() -> None:
    protocol = _protocol()
    binding = _binding()

    assert binding.binding_version == H4_HOLDOUT_VARIANCE_EXECUTION_BINDING_VERSION
    assert binding.protocol_id == protocol.protocol_id
    assert binding.protocol_hash == protocol.payload_hash
    assert binding.population_hash == "b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3"
    assert binding.holdout_acceptance_hash == "9ba347fb1d615cec1d6458944df3c293dcac630564e8847a407e987bbb98af84"
    assert binding.master_seeds == tuple(range(6_400_000, 6_400_010))
    assert binding.holdout_unit_ids == tuple(f"h4_holdout_{i:03d}" for i in range(5))
    assert binding.game_config_hash == "b1916b8c91e234bb0b4d4863aab601db8ccfd8b73e336a924310e13644bae625"
    assert dict(binding.rng_configuration)["full_action_path_identity"] == "not_claimed"
    assert binding.workers == 4
    assert binding.process_model == "spawn-compatible-process-pool-v1"
    assert dict(binding.persistence_paths) == _persistence_paths()
    assert binding.engine_version == "git:" + "a" * 40
    assert binding.execution_authority.payload_hash == "c" * 64


def test_binding_rejects_environment_or_execution_authority_drift() -> None:
    kwargs = dict(
        protocol=_protocol(),
        population=_population(),
        acceptance=_acceptance(),
        workers=4,
        process_model="spawn-compatible-process-pool-v1",
        persistence_paths=_persistence_paths(),
    )
    with pytest.raises(H4HoldoutVarianceBindingError, match="engine_version"):
        build_h4_holdout_variance_execution_binding(
            **kwargs,
            environment=_environment(engine_version="release-build"),
            execution_authority=_authority(),
        )
    with pytest.raises(H4HoldoutVarianceBindingError, match="authorized protocol hash"):
        build_h4_holdout_variance_execution_binding(
            **kwargs,
            environment=_environment(),
            execution_authority=_authority(protocol_hash="d" * 64),
        )


def test_binding_is_canonical_round_trippable_and_tamper_evident(tmp_path: Path) -> None:
    binding = _binding()
    path = tmp_path / "binding.json"
    save_h4_holdout_variance_execution_binding(binding, path)

    assert path.read_bytes().endswith(b"\n")
    assert not path.read_bytes().endswith(b"\r\n")
    assert load_h4_holdout_variance_execution_binding(path) == binding

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["binding"]["workers"] = 99
    path.write_bytes(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    )
    with pytest.raises(H4HoldoutVarianceBindingError, match="binding_hash"):
        load_h4_holdout_variance_execution_binding(path)


def test_trajectory_config_binds_controller_side_holdout_parameters_without_claiming_confirmatory_use() -> None:
    protocol = _protocol()
    binding = _binding()
    job = _job()
    config = build_h4_holdout_variance_trajectory_config(
        protocol=protocol,
        binding=binding,
        population=_population(),
        job=job,
    )

    assert config.config_version == H4_HOLDOUT_VARIANCE_TRAJECTORY_CONFIG_VERSION
    assert config.holdout_unit_id == "h4_holdout_003"
    assert config.master_seed == 6_400_007
    assert config.hands_per_position == 500
    assert config.run_provenance["holdout_unit_id"] == "h4_holdout_003"
    assert config.run_provenance["holdout_population_hash"] == binding.population_hash
    assert config.run_provenance["holdout_acceptance_hash"] == binding.holdout_acceptance_hash
    assert config.run_provenance["true_opponent_parameters"] == dict(config.opponent_parameters)
    assert config.payload()["scientific_use"] == {
        "variance_planning_only": True,
        "confirmatory_h4_inference_permitted": False,
        "variance_pilot_mean_design_use": "prohibited",
        "pilot_final_pooling": False,
        "variance_pilot_final_pooling": False,
    }


def test_all_100_jobs_receive_unique_configs_and_exact_100000_hand_plan() -> None:
    protocol = _protocol()
    binding = _binding()
    population = _population()
    configs = tuple(
        build_h4_holdout_variance_trajectory_config(
            protocol=protocol,
            binding=binding,
            population=population,
            job=job,
        )
        for job in iter_h4_holdout_variance_jobs(protocol)
    )
    assert len(configs) == 100
    assert len({config.config_hash for config in configs}) == 100
    assert sum(config.job_simulated_engine_hands for config in configs) == 100_000
    assert {config.run_provenance["experiment_id"] for config in configs} == {
        f"{protocol.protocol_id}:{binding.binding_hash}"
    }


def test_replacement_seed_cannot_receive_a_trajectory_config() -> None:
    replacement = replace(_job(), master_seed=6_400_010)
    with pytest.raises(H4HoldoutVarianceBindingError, match="canonical frozen Spec-37 job"):
        build_h4_holdout_variance_trajectory_config(
            protocol=_protocol(),
            binding=_binding(),
            population=_population(),
            job=replacement,
        )


def test_trajectory_config_roundtrip_rejects_semantic_tamper_even_after_rehash(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    population = _population()
    job = _job()
    config = build_h4_holdout_variance_trajectory_config(
        protocol=protocol,
        binding=binding,
        population=population,
        job=job,
    )
    path = tmp_path / "config.json"
    save_h4_holdout_variance_trajectory_config(config, path)
    assert load_h4_holdout_variance_trajectory_config(
        path,
        protocol=protocol,
        expected_binding=binding,
        population=population,
        expected_job=job,
    ) == config

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["config"]["run_provenance"]["master_seed"] = 123
    canonical = json.dumps(
        raw["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    raw["config_hash"] = hashlib.sha256(canonical).hexdigest()
    path.write_bytes(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    )
    with pytest.raises(H4HoldoutVarianceBindingError, match="semantic drift"):
        load_h4_holdout_variance_trajectory_config(
            path,
            protocol=protocol,
            expected_binding=binding,
            population=population,
            expected_job=job,
        )


def test_live_capture_requires_clean_committed_scientific_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "docs" / "specs").mkdir(parents=True)
    (repo / "src" / "marker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "configs" / "marker.json").write_text("{}\n", encoding="utf-8")
    (repo / "docs" / "specs" / "marker.md").write_text("# marker\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[tool.poetry]\nname='variance-binding-test'\n", encoding="utf-8")
    lock_path = repo / "poetry.lock"
    lock_path.write_text("# frozen test lock\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "clean"], cwd=repo, check=True)

    binding = capture_h4_holdout_variance_execution_binding(
        protocol=_protocol(),
        population=_population(),
        acceptance=_acceptance(),
        repo_root=repo,
        poetry_lock_path=lock_path,
        ram_bytes=8 * 1024**3,
        workers=2,
        process_model="test-process-model-v1",
        persistence_paths=_persistence_paths(),
        execution_authority=_authority(),
    )
    assert binding.git_commit
    assert binding.engine_version == f"git:{binding.git_commit}"

    (repo / "src" / "drift.py").write_text("DRIFT=True\n", encoding="utf-8")
    with pytest.raises(H4HoldoutVarianceBindingError, match="clean and committed"):
        capture_h4_holdout_variance_execution_binding(
            protocol=_protocol(),
            population=_population(),
            acceptance=_acceptance(),
            repo_root=repo,
            poetry_lock_path=lock_path,
            ram_bytes=8 * 1024**3,
            workers=2,
            process_model="test-process-model-v1",
            persistence_paths=_persistence_paths(),
            execution_authority=_authority(),
        )
