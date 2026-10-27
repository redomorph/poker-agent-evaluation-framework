from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pokerbench.experiments import adaptive_pilot_parallel_cli
from pokerbench.experiments.adaptive_pilot_parallel_launch import (
    AdaptivePilotParallelLaunchError,
    plan_adaptive_pilot_parallel_launch,
)

ROOT = Path(__file__).resolve().parents[2]


def test_parallel_cli_exposes_five_commands_and_frozen_four_worker_defaults() -> None:
    parser = adaptive_pilot_parallel_cli.build_parser()
    action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(action.choices) == {"plan", "status", "readiness", "run", "resume"}
    args = parser.parse_args(["plan", "--ram-bytes", "17071452160"])
    assert args.workers == 4
    assert args.execution_contract == Path("configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json")
    assert args.work_root == Path("data/phase2/adaptive-pilot-parallel/work")


def test_parallel_plan_rejects_wrong_worker_count_and_low_core_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    common = dict(
        protocol_path=ROOT / "configs/experiments/phase2-adaptive-pilot-v1.json",
        execution_contract_path=ROOT / "configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json",
        primary_population_path=ROOT / "configs/populations/baseline-synthetic-v1.json",
        secondary_population_path=ROOT / "configs/populations/balanced-v1.json",
        snapshot_root=ROOT / "data/gate6/snapshots", gate6_report_path=ROOT / "data/gate6/gate6-report.json",
        reference_path=ROOT / "configs/experiments/preflop-range-strength-reference-v1.json",
        database=tmp_path / "pilot.sqlite", work_root=tmp_path / "work", repo_root=ROOT,
        poetry_lock_path=ROOT / "poetry.lock", ram_bytes=16 * 1024**3,
    )
    with pytest.raises(AdaptivePilotParallelLaunchError, match="workers=4"):
        plan_adaptive_pilot_parallel_launch(**common, workers=3)
    monkeypatch.setattr("pokerbench.experiments.adaptive_pilot_parallel_launch.os.cpu_count", lambda: 2)
    with pytest.raises(AdaptivePilotParallelLaunchError, match="at least four"):
        plan_adaptive_pilot_parallel_launch(**common, workers=4)


def test_parallel_cli_plan_and_readiness_are_nonexecuting(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(adaptive_pilot_parallel_cli, "plan_adaptive_pilot_parallel_launch", lambda **kw: calls.append("plan") or SimpleNamespace(binding_hash="a"*64, planned_jobs=50, planned_pairs=25, planned_engine_hands=18000, terminal_jobs=0, planned=50, running=0, completed=0, failed=0))
    monkeypatch.setattr(adaptive_pilot_parallel_cli, "assert_adaptive_pilot_parallel_launch_ready", lambda **kw: calls.append("ready") or SimpleNamespace(binding_hash="a"*64, planned_jobs=50, planned_pairs=25, planned_engine_hands=18000, terminal_jobs=0))
    monkeypatch.setattr(adaptive_pilot_parallel_cli, "run_adaptive_pilot_parallel_launch", lambda **kw: (_ for _ in ()).throw(AssertionError("must not execute")))
    assert adaptive_pilot_parallel_cli.main(["plan", "--ram-bytes", "17071452160"]) == 0
    assert adaptive_pilot_parallel_cli.main(["readiness"]) == 0
    assert calls == ["plan", "ready"]


def test_parallel_plan_materializes_exact_matrix_without_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pokerbench.experiments.adaptive_pilot_parallel_launch import status_adaptive_pilot_parallel_launch
    from test_phase2_adaptive_pilot_parallel_run_v1 import binding

    bind = binding()
    monkeypatch.setattr("pokerbench.experiments.adaptive_pilot_parallel_launch.os.cpu_count", lambda: 8)
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_parallel_launch.capture_adaptive_pilot_execution_binding",
        lambda **kwargs: bind,
    )
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_parallel_launch._load_assets",
        lambda **kwargs: None,
    )
    status = plan_adaptive_pilot_parallel_launch(
        protocol_path=ROOT / "configs/experiments/phase2-adaptive-pilot-v1.json",
        execution_contract_path=ROOT / "configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json",
        primary_population_path=ROOT / "configs/populations/baseline-synthetic-v1.json",
        secondary_population_path=ROOT / "configs/populations/balanced-v1.json",
        snapshot_root=ROOT / "data/gate6/snapshots", gate6_report_path=ROOT / "data/gate6/gate6-report.json",
        reference_path=ROOT / "configs/experiments/preflop-range-strength-reference-v1.json",
        database=tmp_path / "pilot.sqlite", work_root=tmp_path / "work", repo_root=ROOT,
        poetry_lock_path=ROOT / "poetry.lock", ram_bytes=16 * 1024**3, workers=4,
    )
    assert (status.planned_jobs, status.planned_pairs, status.planned_engine_hands, status.terminal_jobs) == (50, 25, 18_000, 0)
    assert len(list((tmp_path / "work" / "trajectory-configs").glob("*.json"))) == 50
    assert status_adaptive_pilot_parallel_launch(
        protocol_path=ROOT / "configs/experiments/phase2-adaptive-pilot-v1.json",
        execution_contract_path=ROOT / "configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json",
        work_root=tmp_path / "work",
    ).terminal_jobs == 0


@pytest.mark.parametrize("resume", [False, True])
def test_parallel_run_and_resume_fail_before_scheduler_when_contract_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    from pokerbench.experiments.adaptive_pilot_parallel_launch import run_adaptive_pilot_parallel_launch
    called = False

    def forbidden(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("scheduler must not be reached")

    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_parallel_launch.execute_adaptive_pilot_parallel_run",
        forbidden,
    )
    with pytest.raises(AdaptivePilotParallelLaunchError, match="execution contract"):
        run_adaptive_pilot_parallel_launch(
            protocol_path=ROOT / "configs/experiments/phase2-adaptive-pilot-v1.json",
            execution_contract_path=tmp_path / "missing.json",
            primary_population_path=ROOT / "configs/populations/baseline-synthetic-v1.json",
            secondary_population_path=ROOT / "configs/populations/balanced-v1.json",
            snapshot_root=ROOT / "data/gate6/snapshots", gate6_report_path=ROOT / "data/gate6/gate6-report.json",
            reference_path=ROOT / "configs/experiments/preflop-range-strength-reference-v1.json",
            database=tmp_path / "pilot.sqlite", work_root=tmp_path / "work", repo_root=ROOT,
            poetry_lock_path=ROOT / "poetry.lock", resume=resume,
        )
    assert called is False
