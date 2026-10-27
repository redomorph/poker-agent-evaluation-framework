from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.agents.profiles import get_static_profile, ProfileName
from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.experiments.adaptive_pilot_binding import (
    ADAPTIVE_PILOT_EXECUTION_BINDING_VERSION,
    ADAPTIVE_PILOT_TRAJECTORY_CONFIG_VERSION,
    AdaptivePilotBindingError,
    build_adaptive_pilot_execution_binding,
    capture_adaptive_pilot_execution_binding,
    build_adaptive_pilot_trajectory_config,
    load_adaptive_pilot_execution_binding,
    load_adaptive_pilot_trajectory_config,
    save_adaptive_pilot_execution_binding,
    save_adaptive_pilot_trajectory_config,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotJob,
    AdaptivePilotProtocol,
    expected_adaptive_pilot_payload,
    iter_adaptive_pilot_jobs,
    load_adaptive_pilot_protocol,
)
from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.frozen_protocol import load_frozen_population


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-v1.json"
PRIMARY_PATH = ROOT / "configs" / "populations" / "baseline-synthetic-v1.json"
SECONDARY_PATH = ROOT / "configs" / "populations" / "balanced-v1.json"
GATE6_PATH = ROOT / "data" / "gate6" / "gate6-report.json"


def _protocol() -> AdaptivePilotProtocol:
    return load_adaptive_pilot_protocol(PROTOCOL_PATH)


def _environment(*, engine_version: str | None = None) -> ExecutionEnvironment:
    commit = "a" * 40
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


def _binding(*, workers: int = 4):
    return build_adaptive_pilot_execution_binding(
        protocol=_protocol(),
        gate6_report=load_gate6_calibration_report(GATE6_PATH),
        primary_population=load_frozen_population(PRIMARY_PATH),
        secondary_populations=(load_frozen_population(SECONDARY_PATH),),
        environment=_environment(),
        workers=workers,
    )


def _job(
    *,
    arm_id: str = "treatment",
    profile_id: str = "nit",
    seed: int = 6_300_001,
) -> AdaptivePilotJob:
    protocol = _protocol()
    return next(
        job
        for job in iter_adaptive_pilot_jobs(protocol)
        if job.arm_id == arm_id
        and job.opponent_profile_id == profile_id
        and job.master_seed == seed
    )


def test_execution_binding_binds_exact_protocol_gate6_populations_rng_dependencies_and_environment() -> None:
    protocol = _protocol()
    binding = _binding()
    frozen = expected_adaptive_pilot_payload()

    assert binding.binding_version == ADAPTIVE_PILOT_EXECUTION_BINDING_VERSION
    assert binding.pilot_protocol_id == protocol.protocol_id
    assert binding.pilot_protocol_hash == protocol.payload_hash
    assert binding.game_config_hash == frozen["game"]["game_config_hash"]
    assert binding.gate6_report_hash == "ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a"
    assert binding.gate6_reference_strength_hash == "d2552b7ffa6aca5e7712b2ddef0dce8805f911cad5eb4c0e6b39fb96a06a6143"
    assert binding.population_versions == (
        (
            "baseline-synthetic-static-profiles",
            "baseline-synthetic-static-profiles-v1",
            "ae332873e33e409be48118116c75ecfa44ed612785630c4799f6948902665888",
        ),
        (
            "balanced-static-profiles",
            "balanced-static-profiles-v1",
            "776d1ffb3a2558143d061129df968949e26621cce8595f48e0d1739726d5dc7f",
        ),
    )
    assert dict(binding.rng_configuration) == {
        "agent_rng_relation": "event_indexed_common_identity_where_events_correspond",
        "deal_relation": "identical_defined_card_realisations",
        "full_action_path_identity": "not_claimed",
        "reuse_same_seed_schedule_across_arms": True,
        "reuse_same_seed_schedule_across_positions": True,
        "reuse_same_seed_schedule_across_profiles": True,
        "seed_schedule_version": "phase2-adaptive-pilot-seeds-v1",
    }
    assert binding.adaptive_learning_protocol_hash == frozen["dependencies"]["adaptive_learning_protocol"]["payload_hash"]
    assert binding.posterior_to_range_protocol_hash == frozen["dependencies"]["posterior_to_range"]["payload_hash"]
    assert binding.adaptive_policy_protocol_hash == frozen["dependencies"]["adaptive_policy"]["payload_hash"]
    assert binding.adaptive_lifecycle_protocol_hash == frozen["dependencies"]["adaptive_lifecycle"]["payload_hash"]
    assert binding.engine_version == "git:" + "a" * 40
    assert binding.python_version == "3.14.0"
    assert binding.poetry_lock_hash == "b" * 64
    assert binding.cpu_model == "Test CPU"
    assert binding.cpu_core_count == 8
    assert binding.cpu_architecture == "AMD64"
    assert binding.os_identifier == "TestOS"
    assert binding.workers == 4


def test_execution_binding_rejects_population_or_environment_drift() -> None:
    protocol = _protocol()
    gate6 = load_gate6_calibration_report(GATE6_PATH)
    primary = load_frozen_population(PRIMARY_PATH)
    secondary = load_frozen_population(SECONDARY_PATH)

    with pytest.raises(AdaptivePilotBindingError, match="primary population"):
        build_adaptive_pilot_execution_binding(
            protocol=protocol,
            gate6_report=gate6,
            primary_population=replace(primary, version="drifted-v2"),
            secondary_populations=(secondary,),
            environment=_environment(),
            workers=4,
        )

    with pytest.raises(AdaptivePilotBindingError, match="engine_version"):
        build_adaptive_pilot_execution_binding(
            protocol=protocol,
            gate6_report=gate6,
            primary_population=primary,
            secondary_populations=(secondary,),
            environment=_environment(engine_version="release-build"),
            workers=4,
        )


def test_execution_binding_is_canonical_round_trippable_and_tamper_evident(tmp_path: Path) -> None:
    binding = _binding()
    path = tmp_path / "binding.json"
    save_adaptive_pilot_execution_binding(binding, path)

    assert path.read_bytes().endswith(b"\n")
    assert not path.read_bytes().endswith(b"\n\n")
    assert load_adaptive_pilot_execution_binding(path) == binding

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["binding"]["workers"] = 99
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(AdaptivePilotBindingError, match="binding_hash"):
        load_adaptive_pilot_execution_binding(path)


def test_trajectory_config_contains_every_required_spec32_provenance_field() -> None:
    protocol = _protocol()
    binding = _binding()
    job = _job()
    config = build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=binding,
        job=job,
    )

    provenance = config.run_provenance
    assert config.config_version == ADAPTIVE_PILOT_TRAJECTORY_CONFIG_VERSION
    assert set(protocol.required_run_provenance) <= set(provenance)
    assert provenance["experiment_id"] == f"{protocol.protocol_id}:{binding.binding_hash}"
    assert provenance["pilot_protocol_id"] == protocol.protocol_id
    assert provenance["pilot_protocol_hash"] == protocol.payload_hash
    assert provenance["game_config_hash"] == binding.game_config_hash
    assert provenance["hero_arm"] == "treatment"
    assert provenance["bot_version"] == "adaptive-hand-strength-agent-v1"
    assert provenance["opponent_version"] == "static-profiles-v2+parametric-policy-v1"
    assert provenance["opponent_profile_id"] == "nit"
    assert provenance["master_seed"] == 6_300_001
    assert provenance["planned_trajectory_horizon"] == 100
    assert provenance["statistical_unit"] == "trajectory_cluster"
    assert provenance["gate6_report_hash"] == binding.gate6_report_hash
    assert provenance["adaptive_learning_protocol_hash"] == binding.adaptive_learning_protocol_hash
    assert provenance["posterior_to_range_protocol_hash"] == binding.posterior_to_range_protocol_hash
    assert provenance["adaptive_policy_protocol_hash"] == binding.adaptive_policy_protocol_hash
    assert provenance["adaptive_lifecycle_protocol_hash"] == binding.adaptive_lifecycle_protocol_hash


def test_trajectory_provenance_records_exact_controller_side_opponent_parameters() -> None:
    config = build_adaptive_pilot_trajectory_config(
        protocol=_protocol(),
        binding=_binding(),
        job=_job(profile_id="calling_station"),
    )
    parameters = config.run_provenance["opponent_parameters"]
    frozen = get_static_profile(ProfileName.CALLING_STATION).parameters

    assert parameters == {
        "schema_version": frozen.schema_version,
        "vpip": frozen.vpip,
        "pfr": frozen.pfr,
        "three_bet_freq": frozen.three_bet_freq,
        "cbet_freq": frozen.cbet_freq,
        "aggression": frozen.aggression,
        "fold_to_cbet": frozen.fold_to_cbet,
    }
    assert parameters["aggression"] is None


def test_matched_control_and_treatment_share_binding_profile_seed_and_horizon() -> None:
    protocol = _protocol()
    binding = _binding()
    treatment = build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=binding,
        job=_job(arm_id="treatment"),
    )
    control = build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=binding,
        job=_job(arm_id="control"),
    )

    t = treatment.run_provenance
    c = control.run_provenance
    assert t["experiment_id"] == c["experiment_id"]
    assert t["opponent_profile_id"] == c["opponent_profile_id"] == "nit"
    assert t["master_seed"] == c["master_seed"] == 6_300_001
    assert t["planned_trajectory_horizon"] == c["planned_trajectory_horizon"] == 100
    assert t["hero_arm"] == "treatment"
    assert c["hero_arm"] == "control"
    assert t["bot_version"] == "adaptive-hand-strength-agent-v1"
    assert c["bot_version"] == "hand-strength-baseline-v1"
    assert treatment.config_hash != control.config_hash


def test_long_seed_binds_500_hand_horizon_and_replacement_seed_is_rejected() -> None:
    protocol = _protocol()
    binding = _binding()
    long_config = build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=binding,
        job=_job(seed=6_300_000),
    )
    assert long_config.run_provenance["planned_trajectory_horizon"] == 500

    replacement = replace(_job(), master_seed=6_399_999)
    with pytest.raises(AdaptivePilotBindingError, match="canonical frozen pilot job"):
        build_adaptive_pilot_trajectory_config(
            protocol=protocol,
            binding=binding,
            job=replacement,
        )


def test_binding_mismatch_cannot_be_reused_for_trajectory_config() -> None:
    protocol = _protocol()
    binding = replace(_binding(), pilot_protocol_hash="f" * 64)
    with pytest.raises(AdaptivePilotBindingError, match="pilot protocol hash"):
        build_adaptive_pilot_trajectory_config(
            protocol=protocol,
            binding=binding,
            job=_job(),
        )


def test_all_50_frozen_jobs_receive_unique_hash_bound_configs_under_one_experiment() -> None:
    protocol = _protocol()
    binding = _binding()
    configs = tuple(
        build_adaptive_pilot_trajectory_config(protocol=protocol, binding=binding, job=job)
        for job in iter_adaptive_pilot_jobs(protocol)
    )

    assert len(configs) == 50
    assert len({config.config_hash for config in configs}) == 50
    assert {config.run_provenance["experiment_id"] for config in configs} == {
        f"{protocol.protocol_id}:{binding.binding_hash}"
    }
    assert sum(config.job_simulated_engine_hands for config in configs) == 18_000


def test_trajectory_config_is_canonical_round_trippable_and_tamper_evident(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    job = _job()
    config = build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=binding,
        job=job,
    )
    path = tmp_path / "trajectory-config.json"
    save_adaptive_pilot_trajectory_config(config, path)

    loaded = load_adaptive_pilot_trajectory_config(
        path,
        protocol=protocol,
        expected_binding=binding,
        expected_job=job,
    )
    assert loaded == config

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["config"]["run_provenance"]["master_seed"] = 123
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(AdaptivePilotBindingError, match="config_hash"):
        load_adaptive_pilot_trajectory_config(
            path,
            protocol=protocol,
            expected_binding=binding,
            expected_job=job,
        )


def test_trajectory_config_semantic_tamper_is_rejected_even_after_rehash(tmp_path: Path) -> None:
    protocol = _protocol()
    binding = _binding()
    job = _job()
    config = build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=binding,
        job=job,
    )
    path = tmp_path / "trajectory-config-rehashed.json"
    save_adaptive_pilot_trajectory_config(config, path)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["config"]["run_provenance"]["hero_arm"] = "control"
    canonical = json.dumps(
        tampered["config"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    tampered["config_hash"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(AdaptivePilotBindingError, match="semantic drift"):
        load_adaptive_pilot_trajectory_config(
            path,
            protocol=protocol,
            expected_binding=binding,
            expected_job=job,
        )


def test_live_capture_requires_clean_committed_scientific_tree_and_captures_current_environment(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "docs" / "specs").mkdir(parents=True)
    (repo / "src" / "marker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "configs" / "marker.json").write_text("{}\n", encoding="utf-8")
    (repo / "docs" / "specs" / "marker.md").write_text("# marker\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[tool.poetry]\nname='binding-test'\n", encoding="utf-8")
    lock_path = repo / "poetry.lock"
    lock_path.write_text("# frozen test lock\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "clean scientific state"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    binding = capture_adaptive_pilot_execution_binding(
        protocol=_protocol(),
        gate6_report_path=GATE6_PATH,
        primary_population_path=PRIMARY_PATH,
        secondary_population_paths=(SECONDARY_PATH,),
        repo_root=repo,
        poetry_lock_path=lock_path,
        ram_bytes=8 * 1024**3,
        workers=2,
    )
    assert binding.git_commit == commit
    assert binding.engine_version == f"git:{commit}"
    assert binding.dependency_lockfile == "poetry.lock"
    assert binding.python_version
    assert binding.cpu_model
    assert binding.os_identifier

    (repo / "src" / "uncommitted.py").write_text("DRIFT = True\n", encoding="utf-8")
    with pytest.raises(AdaptivePilotBindingError, match="clean and committed"):
        capture_adaptive_pilot_execution_binding(
            protocol=_protocol(),
            gate6_report_path=GATE6_PATH,
            primary_population_path=PRIMARY_PATH,
            secondary_population_paths=(SECONDARY_PATH,),
            repo_root=repo,
            poetry_lock_path=lock_path,
            ram_bytes=8 * 1024**3,
            workers=2,
        )
