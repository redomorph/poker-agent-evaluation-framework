"""Command-line wrapper for the frozen synchronous Phase-0 Gate-6 workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pokerbench.evaluation.calibration_snapshots import (
    CalibrationStrengthSnapshotError,
)
from pokerbench.evaluation.gate6_execution import (
    Gate6CalibrationReportError,
    execute_gate6_calibration,
    render_gate6_calibration_report_text,
)
from pokerbench.evaluation.gate6_progress import Gate6ProgressMonitor


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the dedicated Gate-6 execution parser."""
    parser = argparse.ArgumentParser(
        prog="python -m pokerbench.evaluation.gate6_execution_cli",
        description=(
            "Run or resume the frozen Phase-0 policy-calibration Gate-6 "
            "experiment and persist its immutable report."
        ),
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        required=True,
        help="Directory containing resumable calibration-strength snapshots.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Destination JSON path for the immutable Gate-6 report.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Number of deterministic worker processes (default: 1).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_int,
        default=25,
        help=(
            "Persist a scenario checkpoint after this many completed missing "
            "opportunities (default: 25)."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=_positive_float,
        default=5.0,
        help=(
            "Poll persisted snapshots for visible progress at this interval "
            "in seconds (default: 5)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute/resume Gate 6 and return a process-friendly status code."""
    args = build_parser().parse_args(argv)

    print(
        "Gate-6 calibration execution\n"
        f"  snapshot root: {args.snapshot_root}\n"
        f"  report: {args.report}\n"
        f"  workers: {args.workers}\n"
        f"  checkpoint every: {args.checkpoint_every}\n"
        f"  progress interval: {args.progress_interval:g}s",
        flush=True,
    )

    monitor = Gate6ProgressMonitor(
        args.snapshot_root,
        interval_seconds=args.progress_interval,
    )
    try:
        monitor.start()
        report = execute_gate6_calibration(
            args.snapshot_root,
            args.report,
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
        )
    except (
        CalibrationStrengthSnapshotError,
        Gate6CalibrationReportError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"Gate-6 execution failed: {exc}",
            file=sys.stderr,
        )
        return 2
    finally:
        monitor.stop()

    print()
    print(
        render_gate6_calibration_report_text(report),
        end="",
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
