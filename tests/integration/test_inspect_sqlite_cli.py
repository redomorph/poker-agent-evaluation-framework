"""End-to-end SQLite -> `poker inspect` CLI integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokerbench.agents.base import Agent
from pokerbench.agents.observation import (
    AgentObservation,
)
from pokerbench.cli.main import main
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.persistence import (
    SQLiteBenchmarkStore,
)
from pokerbench.rng import AgentRNG
from pokerbench.runner import PairedRunner


MATCH_ID = "sqlite-cli-match"


class FoldAgent(Agent):
    """Deterministic agent used only for end-to-end persistence tests."""

    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        del observation
        del rng

        if Action.FOLD not in legal_actions:
            raise AssertionError(
                "FoldAgent expected fold to be legal"
            )

        return Action.FOLD


def _persist_runner_match(
    database: Path,
    *,
    match_id: str = MATCH_ID,
) -> None:
    store = SQLiteBenchmarkStore(
        database
    )
    store.initialize()

    result = PairedRunner(
        agent_a=FoldAgent("A"),
        agent_b=FoldAgent("B"),
        num_hands=2,
        seed=123,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        adaptive=False,
        match_id=match_id,
        benchmark_store=store,
    ).run()

    assert result.match_record is not None
    assert store.match_exists(
        match_id
    )


def test_cli_inspects_real_runner_match_from_sqlite(
    tmp_path: Path,
    capsys,
) -> None:
    database = (
        tmp_path
        / "benchmark.sqlite3"
    )

    _persist_runner_match(
        database
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--database",
            str(database),
        ]
    )

    assert exit_code == 0

    captured = (
        capsys.readouterr()
    )

    assert (
        f"Match: {MATCH_ID}"
        in captured.out
    )

    assert (
        "Chosen action: fold"
        in captured.out
    )

    assert captured.err == ""


def test_cli_sqlite_inspection_supports_existing_filters(
    tmp_path: Path,
    capsys,
) -> None:
    database = (
        tmp_path
        / "benchmark.sqlite3"
    )

    _persist_runner_match(
        database
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--database",
            str(database),
            "--action",
            "fold",
        ]
    )

    assert exit_code == 0

    output = (
        capsys
        .readouterr()
        .out
    )

    assert "Chosen action: fold" in output


def test_cli_sqlite_inspection_writes_html(
    tmp_path: Path,
) -> None:
    database = (
        tmp_path
        / "benchmark.sqlite3"
    )

    _persist_runner_match(
        database
    )

    destination = (
        tmp_path
        / "inspection.html"
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--database",
            str(database),
            "--html",
            str(destination),
        ]
    )

    assert exit_code == 0
    assert destination.is_file()

    html = destination.read_text(
        encoding="utf-8"
    )

    assert "<!DOCTYPE html>" in html
    assert MATCH_ID in html


def test_cli_sqlite_missing_match_reports_not_found(
    tmp_path: Path,
    capsys,
) -> None:
    database = (
        tmp_path
        / "benchmark.sqlite3"
    )

    store = SQLiteBenchmarkStore(
        database
    )
    store.initialize()

    exit_code = main(
        [
            "inspect",
            "--match-id",
            "does-not-exist",
            "--database",
            str(database),
        ]
    )

    assert exit_code == 1

    captured = (
        capsys.readouterr()
    )

    assert (
        "Match not found: does-not-exist"
        in captured.err
    )

    assert (
        "Could not read valid match-record JSON"
        not in captured.err
    )


def test_cli_sqlite_uninitialized_database_fails_explicitly(
    tmp_path: Path,
    capsys,
) -> None:
    database = (
        tmp_path
        / "missing.sqlite3"
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--database",
            str(database),
        ]
    )

    assert exit_code == 1

    captured = (
        capsys.readouterr()
    )

    assert "Inspection failed:" in captured.err
    assert "does not exist" in captured.err
    assert not database.exists()


def test_cli_rejects_database_and_archive_root_together(
    tmp_path: Path,
) -> None:
    database = (
        tmp_path
        / "benchmark.sqlite3"
    )
    archive_root = (
        tmp_path
        / "replays"
    )

    with pytest.raises(
        SystemExit,
    ):
        main(
            [
                "inspect",
                "--match-id",
                MATCH_ID,
                "--database",
                str(database),
                "--archive-root",
                str(archive_root),
            ]
        )