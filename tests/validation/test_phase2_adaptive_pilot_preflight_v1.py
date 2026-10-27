from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.experiments import adaptive_pilot_cli
from pokerbench.experiments.adaptive_pilot_binding import build_adaptive_pilot_execution_binding
from pokerbench.experiments.adaptive_pilot_execution_contract import load_adaptive_pilot_execution_contract
from pokerbench.experiments.adaptive_pilot_protocol import load_adaptive_pilot_protocol
from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.frozen_protocol import load_frozen_population
from pokerbench.persistence import SQLiteBenchmarkStore


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/experiments/phase2-adaptive-pilot-v1.json"
EXECUTION_CONTRACT_PATH = ROOT / "configs/experiments/phase2-adaptive-pilot-execution-v1.json"
PRIMARY_PATH = ROOT / "configs/populations/baseline-synthetic-v1.json"
SECONDARY_PATH = ROOT / "configs/populations/balanced-v1.json"
SNAPSHOT_ROOT = ROOT / "data/gate6/snapshots"
GATE6_PATH = ROOT / "data/gate6/gate6-report.json"
REFERENCE_PATH = ROOT / "configs/experiments/preflop-range-strength-reference-v1.json"


def _binding():
    commit = "a" * 40
    environment = ExecutionEnvironment(
        python_version="3.14.0",
        engine_version=f"git:{commit}",
        dependency_lockfile="poetry.lock",
        poetry_lock_hash="b" * 64,
        git_commit=commit,
        cpu_model="Preflight Test CPU",
        cpu_core_count=8,
        cpu_architecture="AMD64",
        ram_bytes=16 * 1024**3,
        os_identifier="PreflightTestOS",
    )
    return build_adaptive_pilot_execution_binding(
        protocol=load_adaptive_pilot_protocol(PROTOCOL_PATH),
        execution_contract=load_adaptive_pilot_execution_contract(EXECUTION_CONTRACT_PATH),
        gate6_report=load_gate6_calibration_report(GATE6_PATH),
        primary_population=load_frozen_population(PRIMARY_PATH),
        secondary_populations=(load_frozen_population(SECONDARY_PATH),),
        environment=environment,
        workers=1,
    )


def _common_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--protocol",
        str(PROTOCOL_PATH),
        "--execution-contract",
        str(EXECUTION_CONTRACT_PATH),
        "--primary-population",
        str(PRIMARY_PATH),
        "--secondary-population",
        str(SECONDARY_PATH),
        "--snapshot-root",
        str(SNAPSHOT_ROOT),
        "--gate6-report",
        str(GATE6_PATH),
        "--reference",
        str(REFERENCE_PATH),
        "--database",
        str(tmp_path / "pilot.sqlite"),
        "--work-root",
        str(tmp_path / "work"),
        "--repo-root",
        str(ROOT),
        "--poetry-lock",
        str(ROOT / "poetry.lock"),
    ]


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_cli_plan_plus_readiness_preflight_is_non_executing_virgin_and_artifact_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    binding = _binding()
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: binding,
    )

    def forbidden_execution(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("plan/readiness preflight must never execute a pilot job")

    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.execute_adaptive_pilot_run",
        forbidden_execution,
    )
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.build_adaptive_pilot_job_executor",
        forbidden_execution,
    )

    common = _common_cli_args(tmp_path)
    plan_code = adaptive_pilot_cli.main(
        ["plan", *common, "--ram-bytes", str(binding.ram_bytes), "--workers", "1"]
    )
    plan_output = capsys.readouterr()

    work_root = tmp_path / "work"
    database = tmp_path / "pilot.sqlite"
    store = SQLiteBenchmarkStore(database)

    assert plan_code == 0
    assert "50 jobs" in plan_output.out
    assert "25 matched pairs" in plan_output.out
    assert "18,000 engine hands" in plan_output.out
    assert f"binding={binding.binding_hash}" in plan_output.out
    assert store.list_match_ids() == ()
    assert len(list((work_root / "trajectory-configs").glob("*.json"))) == 50
    assert list((work_root / "arm-snapshots").glob("*.json")) == []
    assert list((work_root / "pair-checkpoints").glob("*.json")) == []

    planned_artifacts = _artifact_hashes(work_root)
    planned_database_hash = hashlib.sha256(database.read_bytes()).hexdigest()

    readiness_code = adaptive_pilot_cli.main(["readiness", *common])
    readiness_output = capsys.readouterr()

    assert readiness_code == 0
    assert "adaptive pilot READY" in readiness_output.out
    assert "50 jobs" in readiness_output.out
    assert "25 planned pairs" in readiness_output.out
    assert "18,000 engine hands" in readiness_output.out
    assert "terminal=0" in readiness_output.out
    assert f"binding={binding.binding_hash}" in readiness_output.out
    assert store.list_match_ids() == ()
    assert hashlib.sha256(database.read_bytes()).hexdigest() == planned_database_hash
    assert _artifact_hashes(work_root) == planned_artifacts
    assert list((work_root / "arm-snapshots").glob("*.json")) == []
    assert list((work_root / "pair-checkpoints").glob("*.json")) == []


def test_readiness_preflight_rejects_live_binding_drift_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    binding = _binding()
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: binding,
    )
    common = _common_cli_args(tmp_path)
    assert adaptive_pilot_cli.main(
        ["plan", *common, "--ram-bytes", str(binding.ram_bytes), "--workers", "1"]
    ) == 0
    capsys.readouterr()

    drifted = binding.__class__(
        **{
            **binding.__dict__,
            "git_commit": "c" * 40,
            "engine_version": f"git:{'c' * 40}",
        }
    )
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: drifted,
    )

    called = False

    def forbidden_execution(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("readiness must fail before execution")

    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.execute_adaptive_pilot_run",
        forbidden_execution,
    )

    assert adaptive_pilot_cli.main(["readiness", *common]) == 2
    output = capsys.readouterr()
    assert "current execution binding differs" in output.err
    assert called is False
    assert SQLiteBenchmarkStore(tmp_path / "pilot.sqlite").list_match_ids() == ()
