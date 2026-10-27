from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.experiments.adaptive_pilot_binding import (
    AdaptivePilotExecutionBinding,
    build_adaptive_pilot_execution_binding,
    build_adaptive_pilot_trajectory_config,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotProtocol,
    iter_adaptive_pilot_jobs,
    load_adaptive_pilot_protocol,
)
from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.frozen_protocol import load_frozen_population
from pokerbench.persistence import SQLiteBenchmarkStore

from pokerbench.experiments.adaptive_pilot_launch import (
    ADAPTIVE_PILOT_LAUNCH_VERSION,
    AdaptivePilotLaunchError,
    assert_adaptive_pilot_launch_ready,
    build_adaptive_pilot_job_executor,
    load_adaptive_pilot_launch_assets,
    plan_adaptive_pilot_launch,
    run_adaptive_pilot_launch,
    status_adaptive_pilot_launch,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/experiments/phase2-adaptive-pilot-v1.json"
PRIMARY_PATH = ROOT / "configs/populations/baseline-synthetic-v1.json"
SECONDARY_PATH = ROOT / "configs/populations/balanced-v1.json"
GATE6_PATH = ROOT / "data/gate6/gate6-report.json"
SNAPSHOT_ROOT = ROOT / "data/gate6/snapshots"
REFERENCE_PATH = ROOT / "configs/experiments/preflop-range-strength-reference-v1.json"


def _protocol() -> AdaptivePilotProtocol:
    return load_adaptive_pilot_protocol(PROTOCOL_PATH)


def _binding(*, commit_char: str = "a") -> AdaptivePilotExecutionBinding:
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
        workers=1,
    )


def _common_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "protocol_path": PROTOCOL_PATH,
        "primary_population_path": PRIMARY_PATH,
        "secondary_population_path": SECONDARY_PATH,
        "snapshot_root": SNAPSHOT_ROOT,
        "gate6_report_path": GATE6_PATH,
        "reference_path": REFERENCE_PATH,
        "database": tmp_path / "pilot.sqlite",
        "work_root": tmp_path / "work",
        "repo_root": ROOT,
        "poetry_lock_path": ROOT / "poetry.lock",
    }


def test_load_launch_assets_binds_exact_gate6_and_adaptive_reference(tmp_path: Path) -> None:
    binding = _binding()
    assets = load_adaptive_pilot_launch_assets(
        protocol_path=PROTOCOL_PATH,
        binding=binding,
        snapshot_root=SNAPSHOT_ROOT,
        gate6_report_path=GATE6_PATH,
        reference_path=REFERENCE_PATH,
    )

    assert assets.launch_version == ADAPTIVE_PILOT_LAUNCH_VERSION
    assert assets.protocol.payload_hash == binding.pilot_protocol_hash
    assert assets.verified_profiles.gate6_report.report_hash == binding.gate6_report_hash
    assert assets.verified_profiles.reference.reference_strength_hash == binding.gate6_reference_strength_hash
    assert assets.adaptive_reference.payload_hash == "cb110768ad74a9747064ba25e88af635055b1491a58deae8f09112aae801fc1e"


def test_launch_assets_reject_gate6_binding_drift() -> None:
    with pytest.raises(AdaptivePilotLaunchError, match="Gate-6 report hash"):
        load_adaptive_pilot_launch_assets(
            protocol_path=PROTOCOL_PATH,
            binding=replace(_binding(), gate6_report_hash="0" * 64),
            snapshot_root=SNAPSHOT_ROOT,
            gate6_report_path=GATE6_PATH,
            reference_path=REFERENCE_PATH,
        )


def test_guarded_worker_uses_bound_profile_job_config_match_and_store(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = _protocol()
    binding = _binding()
    assets = load_adaptive_pilot_launch_assets(
        protocol_path=PROTOCOL_PATH,
        binding=binding,
        snapshot_root=SNAPSHOT_ROOT,
        gate6_report_path=GATE6_PATH,
        reference_path=REFERENCE_PATH,
    )
    job = iter_adaptive_pilot_jobs(protocol)[0]
    config = build_adaptive_pilot_trajectory_config(protocol=protocol, binding=binding, job=job)
    store = SimpleNamespace()
    calls: dict[str, object] = {}
    sentinel_result = object()

    def fake_opponent(profile_id: str, *, context):  # type: ignore[no-untyped-def]
        calls["profile_id"] = profile_id
        calls["context"] = context
        return SimpleNamespace(is_adaptive=False)

    class FakeRunner:
        def run(self):  # type: ignore[no-untyped-def]
            calls["ran"] = True
            return sentinel_result

    def fake_runner(**kwargs):  # type: ignore[no-untyped-def]
        calls["runner"] = kwargs
        return FakeRunner()

    monkeypatch.setattr("pokerbench.experiments.adaptive_pilot_launch.build_verified_static_profile_agent", fake_opponent)
    monkeypatch.setattr("pokerbench.experiments.adaptive_pilot_launch.build_adaptive_pilot_runner", fake_runner)

    result = build_adaptive_pilot_job_executor(assets)(job, config, "match-123", store)  # type: ignore[arg-type]
    assert result is sentinel_result
    assert calls["profile_id"] == job.opponent_profile_id
    runner_args = calls["runner"]
    assert isinstance(runner_args, dict)
    assert runner_args["protocol"] == protocol
    assert runner_args["job"] == job
    assert runner_args["trajectory_config"] == config
    assert runner_args["match_id"] == "match-123"
    assert runner_args["benchmark_store"] is store
    assert calls["ran"] is True


def test_guarded_worker_rejects_config_from_another_binding() -> None:
    protocol = _protocol()
    binding = _binding(commit_char="a")
    other = _binding(commit_char="c")
    assets = load_adaptive_pilot_launch_assets(
        protocol_path=PROTOCOL_PATH,
        binding=binding,
        snapshot_root=SNAPSHOT_ROOT,
        gate6_report_path=GATE6_PATH,
        reference_path=REFERENCE_PATH,
    )
    job = iter_adaptive_pilot_jobs(protocol)[0]
    config = build_adaptive_pilot_trajectory_config(protocol=protocol, binding=other, job=job)
    with pytest.raises(AdaptivePilotLaunchError, match="trajectory binding"):
        build_adaptive_pilot_job_executor(assets)(job, config, "match-123", SimpleNamespace())  # type: ignore[arg-type]


def test_plan_creates_binding_manifest_configs_and_empty_database_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _common_paths(tmp_path)
    binding = _binding()
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: binding,
    )

    status = plan_adaptive_pilot_launch(**paths, ram_bytes=binding.ram_bytes, workers=1)
    store = SQLiteBenchmarkStore(paths["database"])

    assert status.planned_jobs == 50
    assert status.planned_pairs == 25
    assert status.planned_engine_hands == 18_000
    assert status.terminal_jobs == 0
    assert len(list((paths["work_root"] / "trajectory-configs").glob("*.json"))) == 50
    assert store.list_match_ids() == ()


def test_launch_readiness_recaptures_current_environment_and_checks_exact_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _common_paths(tmp_path)
    binding = _binding()
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: binding,
    )
    plan_adaptive_pilot_launch(**paths, ram_bytes=binding.ram_bytes, workers=1)

    ready = assert_adaptive_pilot_launch_ready(**paths)
    assert ready.ready
    assert ready.planned_jobs == 50
    assert ready.planned_pairs == 25
    assert ready.planned_engine_hands == 18_000
    assert ready.terminal_jobs == 0
    assert ready.binding_hash == binding.binding_hash


def test_launch_readiness_rejects_git_or_environment_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _common_paths(tmp_path)
    binding = _binding(commit_char="a")
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: binding,
    )
    plan_adaptive_pilot_launch(**paths, ram_bytes=binding.ram_bytes, workers=1)
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: _binding(commit_char="c"),
    )

    with pytest.raises(AdaptivePilotLaunchError, match="current execution binding"):
        assert_adaptive_pilot_launch_ready(**paths)


def test_status_reads_persisted_state_without_recapturing_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _common_paths(tmp_path)
    binding = _binding()
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: binding,
    )
    plan_adaptive_pilot_launch(**paths, ram_bytes=binding.ram_bytes, workers=1)
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("status must not recapture")),
    )

    status = status_adaptive_pilot_launch(protocol_path=PROTOCOL_PATH, work_root=paths["work_root"])
    assert status.planned_jobs == 50
    assert status.planned == 50
    assert status.running == 0
    assert status.completed == 0
    assert status.failed == 0


def test_run_requires_virgin_plan_while_resume_accepts_partial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _common_paths(tmp_path)
    binding = _binding()
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: binding,
    )
    plan_adaptive_pilot_launch(**paths, ram_bytes=binding.ram_bytes, workers=1)
    fake_result = SimpleNamespace(terminal_jobs=50, completed_jobs=50, failed_jobs=0, pair_count=25, valid_pairs=25)
    calls: list[str] = []

    def fake_execute(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("execute")
        return fake_result

    monkeypatch.setattr("pokerbench.experiments.adaptive_pilot_launch.execute_adaptive_pilot_run", fake_execute)
    result = run_adaptive_pilot_launch(**paths, resume=False)
    assert result is fake_result
    assert calls == ["execute"]

    # Once status is reported as partial/terminal, a fresh run must refuse and resume is explicit.
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.status_adaptive_pilot_launch",
        lambda **kwargs: SimpleNamespace(terminal_jobs=1, running=0, binding_hash=binding.binding_hash),
    )
    with pytest.raises(AdaptivePilotLaunchError, match="use resume"):
        run_adaptive_pilot_launch(**paths, resume=False)
    assert run_adaptive_pilot_launch(**paths, resume=True) is fake_result


def test_run_refuses_before_readiness_and_never_invokes_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _common_paths(tmp_path)
    called = False

    def should_not_run(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("scheduler must not run")

    monkeypatch.setattr("pokerbench.experiments.adaptive_pilot_launch.execute_adaptive_pilot_run", should_not_run)
    with pytest.raises(AdaptivePilotLaunchError, match="not planned"):
        run_adaptive_pilot_launch(**paths, resume=False)
    assert called is False
