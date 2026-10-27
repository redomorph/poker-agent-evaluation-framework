from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pokerbench.cli.main import build_parser, main
from pokerbench.cli.battle import PublicBattleResult
from pokerbench.runner import PairedResult, StatisticalUnit

cli_main = importlib.import_module("pokerbench.cli.main")


def _paired_result(seed: int, differences: list[float], failed: int = 0) -> PairedResult:
    return PairedResult(
        paired_differences=differences,
        statistical_unit=StatisticalUnit.PAIRED_DEAL,
        num_hands=len(differences) + failed,
        num_valid_pairs=len(differences),
        failed_runs=[{"seed": seed}] * failed,
        experiment_metadata={},
        seed=seed,
    )


def test_battle_parser_accepts_roadmap_example_with_frozen_defaults() -> None:
    args = build_parser().parse_args(
        ["battle", "--bot-a", "random", "--bot-b", "tag", "--seeds", "10000"]
    )
    assert args.command == "battle"
    assert args.bot_a == "random"
    assert args.bot_b == "tag"
    assert args.seeds == 10000
    assert args.seed_start == 0
    assert args.hands == 1
    assert args.database == Path("data/benchmark.sqlite")
    assert args.bootstrap_seed == 0
    assert args.snapshot_root == Path("data/gate6/snapshots")
    assert args.gate6_report == Path("data/gate6/gate6-report.json")


def test_battle_parser_rejects_unknown_bot() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["battle", "--bot-a", "unknown", "--bot-b", "tag", "--seeds", "2"]
        )


def test_battle_parser_rejects_non_positive_seed_count() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["battle", "--bot-a", "random", "--bot-b", "tag", "--seeds", "0"]
        )


def test_battle_dispatches_and_returns_complete_status(monkeypatch, capsys) -> None:
    sentinel = object()
    captured = {}

    def fake_execute_public_battle(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(cli_main, "execute_public_battle", fake_execute_public_battle)
    monkeypatch.setattr(cli_main, "render_public_battle_text", lambda value: "BATTLE OK")
    monkeypatch.setattr(cli_main, "public_battle_exit_code", lambda value: 0)

    code = main(
        [
            "battle", "--bot-a", "random", "--bot-b", "tag", "--seeds", "3",
            "--seed-start", "10", "--hands", "2", "--database", "bench.sqlite",
            "--report", "battle.json", "--bootstrap-seed", "99",
        ]
    )

    assert code == 0
    assert captured["bot_a"] == "random"
    assert captured["bot_b"] == "tag"
    assert captured["seed_count"] == 3
    assert captured["seed_start"] == 10
    assert captured["hands"] == 2
    assert captured["database"] == Path("bench.sqlite")
    assert captured["report_path"] == Path("battle.json")
    assert captured["bootstrap_seed"] == 99
    assert "BATTLE OK" in capsys.readouterr().out


def test_battle_returns_one_on_execution_error(monkeypatch, capsys) -> None:
    def fail(**kwargs):
        raise ValueError("bad battle")

    monkeypatch.setattr(cli_main, "execute_public_battle", fail)
    code = main(
        ["battle", "--bot-a", "random", "--bot-b", "tag", "--seeds", "2"]
    )
    assert code == 1
    assert "Battle failed: bad battle" in capsys.readouterr().err


def test_battle_requires_at_least_two_requested_paired_units(monkeypatch, capsys) -> None:
    code = main(
        ["battle", "--bot-a", "random", "--bot-b", "random", "--seeds", "1"]
    )
    assert code == 1
    assert "at least two" in capsys.readouterr().err.lower()


def test_random_only_battle_executes_without_gate6_and_writes_report(tmp_path) -> None:
    from pokerbench.cli.battle import execute_public_battle

    database = tmp_path / "battle.sqlite"
    report_path = tmp_path / "battle.json"

    battle = execute_public_battle(
        bot_a="random",
        bot_b="random",
        seed_count=2,
        seed_start=20,
        hands=1,
        snapshot_root=tmp_path / "missing-snapshots",
        gate6_report_path=tmp_path / "missing-gate6.json",
        database=database,
        bootstrap_seed=17,
        report_path=report_path,
    )

    assert battle.report.complete
    assert battle.report.gate6_report_hash is None
    assert battle.report.seeds == (20, 21)
    assert battle.report.requested_independent_units == 2
    assert battle.report.valid_independent_units == 2
    assert battle.report.requested_engine_hands == 4
    assert battle.report.analyzed_engine_hands == 4
    assert battle.report.evaluation is not None
    assert battle.report.evaluation.num_independent_units == 2
    assert battle.report.evaluation.bootstrap_resamples == 10_000
    assert battle.report.evaluation.bootstrap_seed == 17
    assert len(battle.report.match_ids) == 2
    assert database.exists()
    assert report_path.exists()

    payload = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_hash"] == battle.report.report_hash
    assert payload["report"]["complete"] is True


def test_battle_report_hash_is_deterministic(tmp_path) -> None:
    from pokerbench.cli.battle import execute_public_battle

    first = execute_public_battle(
        bot_a="random",
        bot_b="random",
        seed_count=2,
        seed_start=100,
        hands=1,
        snapshot_root=tmp_path / "unused-a",
        gate6_report_path=tmp_path / "unused-a.json",
        database=tmp_path / "first.sqlite",
        bootstrap_seed=5,
    )
    second = execute_public_battle(
        bot_a="random",
        bot_b="random",
        seed_count=2,
        seed_start=100,
        hands=1,
        snapshot_root=tmp_path / "unused-b",
        gate6_report_path=tmp_path / "unused-b.json",
        database=tmp_path / "second.sqlite",
        bootstrap_seed=5,
    )

    assert first.report.to_dict() == second.report.to_dict()
    assert first.report.report_hash == second.report.report_hash


def test_static_battle_loads_gate6_context_once_for_all_seed_jobs(
    monkeypatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace
    import pokerbench.cli.battle as battle_module
    from pokerbench.agents.random_baseline import RandomBaselineAgent

    calls = []
    fake_context = SimpleNamespace(
        gate6_report=SimpleNamespace(report_hash="a" * 64)
    )

    def fake_load_context(**kwargs):
        calls.append(kwargs)
        return fake_context

    monkeypatch.setattr(
        battle_module,
        "load_verified_static_profile_context",
        fake_load_context,
    )
    monkeypatch.setattr(
        battle_module,
        "_build_battle_agent",
        lambda bot, context: RandomBaselineAgent(),
    )

    battle = battle_module.execute_public_battle(
        bot_a="tag",
        bot_b="lag",
        seed_count=3,
        seed_start=30,
        hands=1,
        snapshot_root=tmp_path / "snapshots",
        gate6_report_path=tmp_path / "gate6.json",
        database=tmp_path / "battle.sqlite",
        bootstrap_seed=1,
    )

    assert len(calls) == 1
    assert battle.report.gate6_report_hash == "a" * 64
    assert battle.report.complete


def test_incomplete_battle_keeps_failures_and_returns_one(monkeypatch, tmp_path) -> None:
    import pokerbench.cli.battle as battle_module

    outcomes = iter(
        [
            _paired_result(0, [1.0]),
            _paired_result(1, [], failed=1),
            _paired_result(2, [-1.0]),
        ]
    )

    class FakeStore:
        def __init__(self, path):
            self.path = path

        def initialize(self):
            return None

    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            return next(outcomes)

    monkeypatch.setattr(battle_module, "SQLiteBenchmarkStore", FakeStore)
    monkeypatch.setattr(battle_module, "PairedRunner", FakeRunner)

    battle = battle_module.execute_public_battle(
        bot_a="random",
        bot_b="random",
        seed_count=3,
        seed_start=0,
        hands=1,
        snapshot_root=tmp_path / "unused",
        gate6_report_path=tmp_path / "unused.json",
        database=tmp_path / "unused.sqlite",
        bootstrap_seed=7,
    )

    assert not battle.report.complete
    assert battle.report.requested_independent_units == 3
    assert battle.report.valid_independent_units == 2
    assert battle.report.failed_runs == 1
    assert battle.report.evaluation is not None
    assert battle.report.evaluation.num_independent_units == 2
    assert battle_module.public_battle_exit_code(battle) == 1
