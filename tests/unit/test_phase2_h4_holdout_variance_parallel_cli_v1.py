from __future__ import annotations

import pytest

from pokerbench.experiments import h4_holdout_variance_parallel_cli as cli


def test_cli_exposes_exact_spec38_command_surface() -> None:
    parser = cli.build_parser()
    subparser_action = next(action for action in parser._actions if action.dest == "command")
    assert tuple(subparser_action.choices) == ("plan", "status", "readiness", "run", "resume")


def test_cli_plan_requires_explicit_ram_bytes() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["plan"])
    args = cli.build_parser().parse_args(["plan", "--ram-bytes", "17179869184"])
    assert args.ram_bytes == 17179869184
    assert args.workers == 4


def test_cli_run_and_resume_delegate_to_parallel_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    class Result:
        terminal_jobs = 100
        completed_jobs = 100
        failed_jobs = 0
        pair_count = 50
        valid_pairs = 50
        invalid_pairs = 0
        cluster_count = 10
        valid_clusters = 10
        invalid_clusters = 0
        scientifically_complete = True

    monkeypatch.setattr(
        cli,
        "run_h4_holdout_variance_parallel_launch",
        lambda **kwargs: calls.append(kwargs["resume"]) or Result(),
    )
    assert cli.main(["run"]) == 0
    assert cli.main(["resume"]) == 0
    assert calls == [False, True]


def test_cli_returns_nonzero_for_terminal_but_scientifically_incomplete_campaign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        terminal_jobs = 100
        completed_jobs = 99
        failed_jobs = 1
        pair_count = 50
        valid_pairs = 49
        invalid_pairs = 1
        cluster_count = 10
        valid_clusters = 9
        invalid_clusters = 1
        scientifically_complete = False

    monkeypatch.setattr(cli, "run_h4_holdout_variance_parallel_launch", lambda **kwargs: Result())
    assert cli.main(["run"]) == 1
