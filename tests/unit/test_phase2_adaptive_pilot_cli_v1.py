from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pokerbench.experiments import adaptive_pilot_cli


def test_cli_exposes_only_plan_status_readiness_run_and_resume() -> None:
    parser = adaptive_pilot_cli.build_parser()
    action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(action.choices) == {"plan", "status", "readiness", "run", "resume"}


def test_cli_plan_calls_plan_only_and_reports_frozen_matrix(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        adaptive_pilot_cli,
        "plan_adaptive_pilot_launch",
        lambda **kwargs: calls.append("plan") or SimpleNamespace(
            binding_hash="a" * 64,
            planned_jobs=50,
            planned_pairs=25,
            planned_engine_hands=18_000,
            terminal_jobs=0,
            planned=50,
            running=0,
            completed=0,
            failed=0,
        ),
    )
    monkeypatch.setattr(
        adaptive_pilot_cli,
        "run_adaptive_pilot_launch",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("plan must not execute")),
    )

    code = adaptive_pilot_cli.main(["plan", "--ram-bytes", "17179869184"])
    assert code == 0
    assert calls == ["plan"]
    output = capsys.readouterr().out
    assert "50 jobs" in output
    assert "18,000 engine hands" in output


def test_cli_readiness_fails_closed_with_exit_2(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        adaptive_pilot_cli,
        "assert_adaptive_pilot_launch_ready",
        lambda **kwargs: (_ for _ in ()).throw(adaptive_pilot_cli.AdaptivePilotLaunchError("binding drift")),
    )
    code = adaptive_pilot_cli.main(["readiness"])
    captured = capsys.readouterr()
    assert code == 2
    assert "binding drift" in captured.err


def test_cli_run_and_resume_are_distinct_scheduler_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    modes: list[bool] = []
    monkeypatch.setattr(
        adaptive_pilot_cli,
        "run_adaptive_pilot_launch",
        lambda **kwargs: modes.append(bool(kwargs["resume"])) or SimpleNamespace(
            terminal_jobs=50,
            completed_jobs=50,
            failed_jobs=0,
            pair_count=25,
            valid_pairs=25,
            invalid_pairs=0,
        ),
    )
    assert adaptive_pilot_cli.main(["run"]) == 0
    assert adaptive_pilot_cli.main(["resume"]) == 0
    assert modes == [False, True]


def test_cli_status_does_not_require_ram_or_execution_environment(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        adaptive_pilot_cli,
        "status_adaptive_pilot_launch",
        lambda **kwargs: SimpleNamespace(
            binding_hash="a" * 64,
            planned_jobs=50,
            planned_pairs=25,
            planned_engine_hands=18_000,
            terminal_jobs=7,
            planned=43,
            running=0,
            completed=6,
            failed=1,
        ),
    )
    assert adaptive_pilot_cli.main(["status"]) == 0
    assert "terminal=7/50" in capsys.readouterr().out
