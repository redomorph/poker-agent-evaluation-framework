from __future__ import annotations

from pathlib import Path
import importlib

import pytest

from pokerbench.cli.main import build_parser, main

cli_main = importlib.import_module("pokerbench.cli.main")


def test_public_run_parser_exposes_frozen_arguments() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--bot-a", "tag",
            "--bot-b", "lag",
            "--seed", "12345",
            "--hands", "1000",
            "--database", "benchmark.sqlite",
        ]
    )
    assert args.command == "run"
    assert args.bot_a == "tag"
    assert args.bot_b == "lag"
    assert args.seed == 12345
    assert args.hands == 1000
    assert args.database == Path("benchmark.sqlite")


def test_public_run_uses_frozen_gate6_defaults() -> None:
    args = build_parser().parse_args(
        [
            "run", "--bot-a", "nit", "--bot-b", "tag",
            "--seed", "1", "--hands", "2", "--database", "x.sqlite",
        ]
    )
    assert args.snapshot_root == Path("data/gate6/snapshots")
    assert args.gate6_report == Path("data/gate6/gate6-report.json")


def test_public_run_rejects_unknown_profile() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "run", "--bot-a", "unknown", "--bot-b", "tag",
                "--seed", "1", "--hands", "2", "--database", "x.sqlite",
            ]
        )


def test_public_run_dispatches_and_renders(monkeypatch, capsys) -> None:
    sentinel = object()
    captured = {}

    def fake_execute_public_run(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(cli_main, "execute_public_run", fake_execute_public_run)
    monkeypatch.setattr(cli_main, "render_public_run_text", lambda value: "RUN OK")

    code = main(
        [
            "run", "--bot-a", "tag", "--bot-b", "lag",
            "--seed", "7", "--hands", "9", "--database", "bench.sqlite",
            "--match-id", "m-1",
        ]
    )
    assert code == 0
    assert captured["bot_a"] == "tag"
    assert captured["bot_b"] == "lag"
    assert captured["seed"] == 7
    assert captured["hands"] == 9
    assert captured["match_id"] == "m-1"
    assert "RUN OK" in capsys.readouterr().out


def test_public_run_returns_one_on_execution_error(monkeypatch, capsys) -> None:
    def fail(**kwargs):
        raise ValueError("bad gate6")

    monkeypatch.setattr(cli_main, "execute_public_run", fail)
    code = main(
        [
            "run", "--bot-a", "tag", "--bot-b", "lag",
            "--seed", "7", "--hands", "9", "--database", "bench.sqlite",
        ]
    )
    assert code == 1
    assert "Run failed: bad gate6" in capsys.readouterr().err


def test_database_is_required_for_public_run() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "run", "--bot-a", "tag", "--bot-b", "lag",
                "--seed", "7", "--hands", "9",
            ]
        )
