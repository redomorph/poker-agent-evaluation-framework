"""Integration tests for the `poker inspect` CLI."""

from __future__ import annotations

from pathlib import Path

from pokerbench.cli.main import main
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.replay import (
    HandReplay,
    HandTermination,
    MatchRecord,
    RecordedHand,
    ReplayArchive,
)


MASTER_SEED = 42
MATCH_ID = "cli-inspect-match"


def _recorded_hand() -> RecordedHand:
    actions = (
        Action.BET_200,
        Action.CALL,
        Action.CHECK,
        Action.BET_100,
        Action.CALL,
        Action.CHECK,
        Action.BET_200,
        Action.CALL,
        Action.BET_33,
        Action.FOLD,
    )

    replay = HandReplay(
        config=BASELINE_GAME_CONFIG,
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        actions=actions,
    ).replay()

    assert (
        replay.final_state
        .terminal_profit
        is not None
    )

    profit = (
        replay.final_state
        .terminal_profit
    )

    return RecordedHand(
        config_id="CLI_TEST",
        master_seed=MASTER_SEED,
        hand_index=0,
        dealer=0,
        seat0_bot_id=0,
        seat1_bot_id=1,
        actions=actions,
        termination=(
            HandTermination.NORMAL
        ),
        terminal_profit_bb=(
            profit[0] / 100.0,
            profit[1] / 100.0,
        ),
    )


def _archive(
    root: Path,
) -> ReplayArchive:
    archive = ReplayArchive(
        root
    )

    archive.save(
        MatchRecord(
            match_id=MATCH_ID,
            master_seed=MASTER_SEED,
            game_config=(
                BASELINE_GAME_CONFIG
            ),
            hands=(
                _recorded_hand(),
            ),
        )
    )

    return archive


def test_cli_inspect_match(
    tmp_path: Path,
    capsys,
) -> None:
    _archive(
        tmp_path
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--archive-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0

    output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        f"Match: {MATCH_ID}"
        in output
    )

    assert "Chosen action" in output


def test_cli_street_filter(
    tmp_path: Path,
    capsys,
) -> None:
    _archive(
        tmp_path
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--archive-root",
            str(tmp_path),
            "--street",
            "river",
        ]
    )

    assert exit_code == 0

    output = (
        capsys
        .readouterr()
        .out
    )

    assert "river" in output.lower()


def test_cli_action_filter(
    tmp_path: Path,
    capsys,
) -> None:
    _archive(
        tmp_path
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--archive-root",
            str(tmp_path),
            "--action",
            "call",
        ]
    )

    assert exit_code == 0

    output = (
        capsys
        .readouterr()
        .out
    )

    assert "call" in output.lower()


def test_cli_min_pot_filter(
    tmp_path: Path,
    capsys,
) -> None:
    _archive(
        tmp_path
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--archive-root",
            str(tmp_path),
            "--min-pot",
            "50",
        ]
    )

    assert exit_code == 0

    output = (
        capsys
        .readouterr()
        .out
    )

    assert "Hand 0" in output


def test_cli_required_compound_filter(
    tmp_path: Path,
    capsys,
) -> None:
    _archive(
        tmp_path
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--archive-root",
            str(tmp_path),
            "--filter",
            (
                "hero_loss > 50 "
                "AND street == 'river'"
            ),
        ]
    )

    assert exit_code == 0

    output = (
        capsys
        .readouterr()
        .out
    )

    assert "Hand 0" in output


def test_cli_writes_html_replay(
    tmp_path: Path,
) -> None:
    _archive(
        tmp_path
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
            "--archive-root",
            str(tmp_path),
            "--html",
            str(destination),
        ]
    )

    assert exit_code == 0

    assert (
        destination.is_file()
    )

    html = (
        destination.read_text(
            encoding="utf-8"
        )
    )

    assert "<!DOCTYPE html>" in html
    assert MATCH_ID in html


def test_cli_no_matches_is_successful_and_explicit(
    tmp_path: Path,
    capsys,
) -> None:
    _archive(
        tmp_path
    )

    exit_code = main(
        [
            "inspect",
            "--match-id",
            MATCH_ID,
            "--archive-root",
            str(tmp_path),
            "--street",
            "river",
            "--action",
            "all_in",
        ]
    )

    assert exit_code == 0

    output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        "No matching hands."
        in output
    )


def test_cli_missing_match_reports_not_found(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "inspect",
            "--match-id",
            "does-not-exist",
            "--archive-root",
            str(tmp_path),
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