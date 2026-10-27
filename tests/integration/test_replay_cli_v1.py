"""Integration tests for the public ``poker replay`` command."""

from __future__ import annotations

from pathlib import Path

from pokerbench.cli.main import build_parser, main
from pokerbench.engine.config import Action, BASELINE_GAME_CONFIG
from pokerbench.persistence import SQLiteBenchmarkStore
from pokerbench.replay import (
    HandTermination,
    MatchRecord,
    RecordedHand,
    ReplayArchive,
)


MASTER_SEED = 42
MATCH_ID = "cli-replay-match"


def _record() -> MatchRecord:
    return MatchRecord(
        match_id=MATCH_ID,
        master_seed=MASTER_SEED,
        game_config=BASELINE_GAME_CONFIG,
        hands=(
            RecordedHand(
                config_id="A_BTN",
                master_seed=MASTER_SEED,
                hand_index=0,
                dealer=0,
                seat0_bot_id=0,
                seat1_bot_id=1,
                actions=(Action.FOLD,),
                termination=HandTermination.NORMAL,
                terminal_profit_bb=(-0.5, 0.5),
            ),
        ),
    )


def test_parser_exposes_public_replay_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "replay",
            "--match-id",
            MATCH_ID,
        ]
    )

    assert args.command == "replay"
    assert args.match_id == MATCH_ID


def test_replay_from_archive_renders_complete_decision_chain(
    tmp_path: Path,
    capsys,
) -> None:
    archive = ReplayArchive(tmp_path)
    archive.save(_record())

    exit_code = main(
        [
            "replay",
            "--match-id",
            MATCH_ID,
            "--archive-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0

    output = capsys.readouterr().out

    assert f"Match: {MATCH_ID}" in output
    assert "Hand 0" in output
    assert "Hole cards:" in output
    assert "Board:" in output
    assert "Position:" in output
    assert "Stacks:" in output
    assert "Pot:" in output
    assert "Action history:" in output
    assert "Available actions:" in output
    assert "Chosen action: fold" in output
    assert "Resulting profit:" in output
    assert "Estimated hand strength: N/A" in output
    assert "Estimated decision EV: N/A" in output


def test_replay_from_sqlite_uses_persisted_match(
    tmp_path: Path,
    capsys,
) -> None:
    database = tmp_path / "benchmark.sqlite"
    store = SQLiteBenchmarkStore(database)
    store.initialize()
    store.save_match(_record())

    exit_code = main(
        [
            "replay",
            "--match-id",
            MATCH_ID,
            "--database",
            str(database),
        ]
    )

    assert exit_code == 0
    assert f"Match: {MATCH_ID}" in capsys.readouterr().out


def test_replay_writes_minimal_html_viewer(
    tmp_path: Path,
    capsys,
) -> None:
    archive_root = tmp_path / "archive"
    ReplayArchive(archive_root).save(_record())
    html_path = tmp_path / "replay.html"

    exit_code = main(
        [
            "replay",
            "--match-id",
            MATCH_ID,
            "--archive-root",
            str(archive_root),
            "--html",
            str(html_path),
        ]
    )

    assert exit_code == 0
    assert html_path.is_file()

    html = html_path.read_text(encoding="utf-8")

    for label in (
        "Hole cards",
        "Board",
        "Position",
        "Stacks",
        "Pot",
        "Action history",
        "Available actions",
        "Chosen action",
        "Estimated hand strength",
        "Resulting profit",
        "Estimated decision EV",
    ):
        assert label in html

    assert "HTML replay written to:" in capsys.readouterr().out


def test_replay_unknown_match_returns_nonzero(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "replay",
            "--match-id",
            "missing",
            "--archive-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert "Match not found: missing" in capsys.readouterr().err


def test_replay_rejects_multiple_persistence_sources(
    tmp_path: Path,
) -> None:
    parser = build_parser()

    try:
        parser.parse_args(
            [
                "replay",
                "--match-id",
                MATCH_ID,
                "--archive-root",
                str(tmp_path),
                "--database",
                str(tmp_path / "benchmark.sqlite"),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError(
            "replay must reject simultaneous archive/database sources"
        )
