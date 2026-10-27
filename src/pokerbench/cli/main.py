"""Top-level PokerBench command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from pokerbench.cli.battle import (
    PUBLIC_BATTLE_DEFAULT_DATABASE,
    execute_public_battle,
    public_battle_exit_code,
    render_public_battle_text,
)
from pokerbench.cli.export import (
    execute_public_export,
    render_public_export_text,
)
from pokerbench.cli.plot import (
    execute_public_plot,
    render_public_plot_text,
)
from pokerbench.cli.inspect import (
    InspectOptions,
    inspect_match,
    render_inspection_html,
    render_inspection_text,
    write_inspection_html,
)
from pokerbench.cli.run import (
    execute_public_run,
    render_public_run_text,
)
from pokerbench.cli.replay import (
    render_replay_html,
    render_replay_text,
    replay_match,
    write_replay_html,
)
from pokerbench.engine.config import (
    Action,
)
from pokerbench.persistence import (
    MatchNotFoundError,
    PersistenceError,
    SQLiteBenchmarkStore,
)
from pokerbench.replay import (
    MatchRecord,
    ReplayArchive,
    ReplayIntegrityError,
)


STATIC_PROFILE_CHOICES = (
    "nit",
    "tag",
    "lag",
    "calling_station",
    "maniac",
)

BATTLE_BOT_CHOICES = (
    "random",
    *STATIC_PROFILE_CHOICES,
)

DEFAULT_REPLAY_ROOT = Path(
    os.environ.get(
        "POKERBENCH_REPLAY_ROOT",
        "data/replays",
    )
)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI parser."""
    parser = argparse.ArgumentParser(
        prog="poker",
        description=(
            "Scientific poker-agent "
            "evaluation framework"
        ),
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Execute and persist one paired static-profile match",
    )
    run_parser.add_argument("--bot-a", required=True, choices=STATIC_PROFILE_CHOICES)
    run_parser.add_argument("--bot-b", required=True, choices=STATIC_PROFILE_CHOICES)
    run_parser.add_argument("--seed", required=True, type=int)
    run_parser.add_argument("--hands", required=True, type=int)
    run_parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path("data/gate6/snapshots"),
    )
    run_parser.add_argument(
        "--gate6-report",
        type=Path,
        default=Path("data/gate6/gate6-report.json"),
    )
    run_parser.add_argument(
        "--database",
        required=True,
        type=Path,
    )
    run_parser.add_argument("--match-id")


    battle_parser = subparsers.add_parser(
        "battle",
        help="Execute and aggregate a deterministic multi-seed battle",
    )
    battle_parser.add_argument("--bot-a", required=True, choices=BATTLE_BOT_CHOICES)
    battle_parser.add_argument("--bot-b", required=True, choices=BATTLE_BOT_CHOICES)
    battle_parser.add_argument("--seeds", required=True, type=_positive_int)
    battle_parser.add_argument("--seed-start", type=_nonnegative_int, default=0)
    battle_parser.add_argument("--hands", type=_positive_int, default=1)
    battle_parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path("data/gate6/snapshots"),
    )
    battle_parser.add_argument(
        "--gate6-report",
        type=Path,
        default=Path("data/gate6/gate6-report.json"),
    )
    battle_parser.add_argument(
        "--database",
        type=Path,
        default=PUBLIC_BATTLE_DEFAULT_DATABASE,
    )
    battle_parser.add_argument("--bootstrap-seed", type=int, default=0)
    battle_parser.add_argument("--report", type=Path, dest="report_path")

    export_parser = subparsers.add_parser(
        "export",
        help="Export persisted benchmark data as a deterministic bundle",
    )
    export_parser.add_argument(
        "--database",
        required=True,
        type=Path,
        help="SQLite benchmark database to export",
    )
    export_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New destination directory for the export bundle",
    )
    export_parser.add_argument(
        "--match-id",
        action="append",
        dest="match_ids",
        help="Optional persisted match ID; may be repeated",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot complete persisted battle reports as deterministic SVG",
    )
    plot_parser.add_argument(
        "--report",
        required=True,
        action="append",
        type=Path,
        dest="report_paths",
        help="Complete public-battle-cli-v1 JSON report; may be repeated",
    )
    plot_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New destination directory for the plot bundle",
    )
    plot_parser.add_argument(
        "--ci",
        choices=("bootstrap", "normal"),
        default="bootstrap",
        dest="ci_kind",
        help="Already-computed confidence interval to display",
    )

    inspect_parser = (
        subparsers.add_parser(
            "inspect",
            help=(
                "Inspect replayable hands "
                "from a persisted match"
            ),
        )
    )

    inspect_parser.add_argument(
        "--match-id",
        required=True,
        help="Persisted match identifier",
    )

    persistence_source = (
        inspect_parser
        .add_mutually_exclusive_group()
    )

    persistence_source.add_argument(
        "--archive-root",
        type=Path,
        default=(
            DEFAULT_REPLAY_ROOT
        ),
        help=(
            "ReplayArchive directory "
            "(default: data/replays or "
            "POKERBENCH_REPLAY_ROOT)"
        ),
    )

    persistence_source.add_argument(
        "--database",
        type=Path,
        help=(
            "SQLite benchmark database "
            "containing the persisted match"
        ),
    )

    inspect_parser.add_argument(
        "--filter",
        dest=(
            "filter_expression"
        ),
        help=(
            "Conjunctive filter expression, "
            "for example: "
            "\"hero_loss > 50 AND "
            "street == 'river'\""
        ),
    )

    inspect_parser.add_argument(
        "--street",
        choices=(
            "preflop",
            "flop",
            "turn",
            "river",
            "showdown",
        ),
    )

    inspect_parser.add_argument(
        "--action",
        choices=tuple(
            action.value
            for action
            in Action
        ),
    )

    inspect_parser.add_argument(
        "--min-pot",
        type=float,
        dest="min_pot_bb",
        help=(
            "Minimum maximum hand pot "
            "in BB"
        ),
    )

    inspect_parser.add_argument(
        "--hero-bot-id",
        type=int,
        choices=(
            0,
            1,
        ),
        default=0,
        help=(
            "Stable bot identity used for "
            "hero_loss/hero_profit filters"
        ),
    )

    inspect_parser.add_argument(
        "--html",
        type=Path,
        dest="html_output",
        help=(
            "Optional HTML replay-viewer "
            "output path"
        ),
    )

    replay_parser = (
        subparsers.add_parser(
            "replay",
            help=(
                "Deterministically replay all hands "
                "from a persisted match"
            ),
        )
    )

    replay_parser.add_argument(
        "--match-id",
        required=True,
        help="Persisted match identifier",
    )

    replay_source = (
        replay_parser
        .add_mutually_exclusive_group()
    )

    replay_source.add_argument(
        "--archive-root",
        type=Path,
        default=(
            DEFAULT_REPLAY_ROOT
        ),
        help=(
            "ReplayArchive directory "
            "(default: data/replays or "
            "POKERBENCH_REPLAY_ROOT)"
        ),
    )

    replay_source.add_argument(
        "--database",
        type=Path,
        help=(
            "SQLite benchmark database "
            "containing the persisted match"
        ),
    )

    replay_parser.add_argument(
        "--hero-bot-id",
        type=int,
        choices=(
            0,
            1,
        ),
        default=0,
        help=(
            "Stable bot identity used for "
            "hero-oriented replay fields"
        ),
    )

    replay_parser.add_argument(
        "--html",
        type=Path,
        dest="html_output",
        help=(
            "Optional HTML replay-viewer "
            "output path"
        ),
    )

    return parser


def main(
    argv: Sequence[
        str
    ] | None = None,
) -> int:
    """Execute the command-line interface."""
    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    if args.command == "run":
        return _run_public_run(args)

    if args.command == "battle":
        return _run_public_battle(args)

    if args.command == "export":
        return _run_public_export(args)

    if args.command == "plot":
        return _run_public_plot(args)

    if (
        args.command
        == "inspect"
    ):
        return _run_inspect(
            args
        )

    if (
        args.command
        == "replay"
    ):
        return _run_replay(
            args
        )

    parser.error(
        "Unsupported command"
    )

    return 2





def _run_public_run(args: argparse.Namespace) -> int:
    try:
        run = execute_public_run(
            bot_a=args.bot_a,
            bot_b=args.bot_b,
            seed=args.seed,
            hands=args.hands,
            snapshot_root=args.snapshot_root,
            gate6_report_path=args.gate6_report,
            database=args.database,
            match_id=args.match_id,
        )
        print(render_public_run_text(run))
        return 0
    except (OSError, ValueError, TypeError, PersistenceError) as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1


def _run_public_battle(args: argparse.Namespace) -> int:
    try:
        battle = execute_public_battle(
            bot_a=args.bot_a,
            bot_b=args.bot_b,
            seed_count=args.seeds,
            seed_start=args.seed_start,
            hands=args.hands,
            snapshot_root=args.snapshot_root,
            gate6_report_path=args.gate6_report,
            database=args.database,
            bootstrap_seed=args.bootstrap_seed,
            report_path=args.report_path,
        )
        print(render_public_battle_text(battle))
        return public_battle_exit_code(battle)
    except (OSError, ValueError, TypeError, PersistenceError) as exc:
        print(f"Battle failed: {exc}", file=sys.stderr)
        return 1


def _run_public_export(args: argparse.Namespace) -> int:
    try:
        result = execute_public_export(
            database=args.database,
            output=args.output,
            match_ids=args.match_ids,
        )
        print(render_public_export_text(result))
        return 0
    except (OSError, ValueError, TypeError, PersistenceError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1


def _run_public_plot(args: argparse.Namespace) -> int:
    try:
        result = execute_public_plot(
            report_paths=args.report_paths,
            output=args.output,
            ci_kind=args.ci_kind,
        )
        print(render_public_plot_text(result))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"Plot failed: {exc}", file=sys.stderr)
        return 1


def _load_inspect_record(
    args: argparse.Namespace,
) -> MatchRecord:
    """
    Load the requested match from exactly one explicit persistence source.

    SQLite is used only when --database is supplied. Otherwise the existing
    ReplayArchive path remains the default for backward compatibility.
    """
    if args.database is not None:
        store = SQLiteBenchmarkStore(
            args.database
        )

        return store.load_match(
            args.match_id
        )

    archive = ReplayArchive(
        args.archive_root
    )

    return archive.load(
        args.match_id
    )


def _run_inspect(
    args: argparse.Namespace,
) -> int:
    try:
        record = _load_inspect_record(
            args
        )

        action = (
            Action(
                args.action
            )
            if args.action
            is not None
            else None
        )

        options = InspectOptions(
            hero_bot_id=(
                args.hero_bot_id
            ),
            street=(
                args.street
            ),
            action=action,
            min_pot_bb=(
                args.min_pot_bb
            ),
            filter_expression=(
                args.filter_expression
            ),
        )

        hands = inspect_match(
            record,
            options,
        )

        rendered = (
            render_inspection_text(
                record,
                hands,
                hero_bot_id=(
                    args.hero_bot_id
                ),
            )
        )

        print(
            rendered
        )

        if (
            args.html_output
            is not None
        ):
            html_output = (
                render_inspection_html(
                    record,
                    hands,
                    hero_bot_id=(
                        args.hero_bot_id
                    ),
                )
            )

            destination = (
                write_inspection_html(
                    args.html_output,
                    html_output,
                )
            )

            print(
                "\nHTML replay written to: "
                f"{destination}"
            )

        return 0

    except (
        FileNotFoundError,
        MatchNotFoundError,
    ):
        print(
            "Match not found: "
            f"{args.match_id}",
            file=sys.stderr,
        )

        return 1

    except (
        PersistenceError,
        ReplayIntegrityError,
        ValueError,
        TypeError,
    ) as exc:
        print(
            f"Inspection failed: {exc}",
            file=sys.stderr,
        )

        return 1

def _load_replay_record(
    args: argparse.Namespace,
) -> MatchRecord:
    """Load one replayable match from the selected persistence source."""
    if args.database is not None:
        store = SQLiteBenchmarkStore(
            args.database
        )

        return store.load_match(
            args.match_id
        )

    archive = ReplayArchive(
        args.archive_root
    )

    return archive.load(
        args.match_id
    )


def _run_replay(
    args: argparse.Namespace,
) -> int:
    try:
        record = _load_replay_record(
            args
        )

        hands = replay_match(
            record,
            hero_bot_id=(
                args.hero_bot_id
            ),
        )

        rendered = render_replay_text(
            record,
            hands,
            hero_bot_id=(
                args.hero_bot_id
            ),
        )

        print(
            rendered
        )

        if (
            args.html_output
            is not None
        ):
            html_output = (
                render_replay_html(
                    record,
                    hands,
                    hero_bot_id=(
                        args.hero_bot_id
                    ),
                )
            )

            destination = (
                write_replay_html(
                    args.html_output,
                    html_output,
                )
            )

            print(
                "\nHTML replay written to: "
                f"{destination}"
            )

        return 0

    except (
        FileNotFoundError,
        MatchNotFoundError,
    ):
        print(
            "Match not found: "
            f"{args.match_id}",
            file=sys.stderr,
        )

        return 1

    except (
        PersistenceError,
        ReplayIntegrityError,
        ValueError,
        TypeError,
    ) as exc:
        print(
            f"Replay failed: {exc}",
            file=sys.stderr,
        )

        return 1
