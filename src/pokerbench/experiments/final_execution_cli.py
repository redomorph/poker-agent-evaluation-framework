"""Internal CLI for executing/resuming the frozen Phase-1 final experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from pokerbench.evaluation.calibration_snapshots import CalibrationStrengthSnapshotError
from pokerbench.evaluation.gate6_execution import Gate6CalibrationReportError
from pokerbench.experiments.final_execution import (
    FinalExecutionError,
    execute_phase1_final,
    render_phase1_final_execution_report_text,
)
from pokerbench.experiments.final_protocol import FinalProtocolError


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pokerbench.experiments.final_execution_cli",
        description="Execute or resume the frozen Phase-1 final static-baseline experiment.",
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/experiments/phase1-baseline-experiment-v1.json"))
    parser.add_argument("--primary-population", type=Path, default=Path("configs/populations/baseline-synthetic-v1.json"))
    parser.add_argument("--secondary-population", type=Path, default=Path("configs/populations/balanced-v1.json"))
    parser.add_argument("--pilot-report", type=Path, default=Path("data/phase1/pilot/phase1-pilot-report.json"))
    parser.add_argument("--snapshot-root", type=Path, default=Path("data/gate6/snapshots"))
    parser.add_argument("--gate6-report", type=Path, default=Path("data/gate6/gate6-report.json"))
    parser.add_argument("--database", type=Path, default=Path("data/phase1/final/phase1-baseline-v1.sqlite"))
    parser.add_argument("--work-root", type=Path, default=Path("data/phase1/final/work"))
    parser.add_argument("--report", type=Path, default=Path("data/phase1/final/phase1-baseline-report.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--poetry-lock", type=Path, default=Path("poetry.lock"))
    parser.add_argument("--ram-bytes", type=_positive_int, required=True)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--progress-every", type=_positive_int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    last_printed = -1

    def progress(completed: int, total: int, resumed: int, failed: int) -> None:
        nonlocal last_printed
        if (completed == total or completed == resumed or completed % args.progress_every == 0) and completed != last_printed:
            print(
                f"final progress: {completed}/{total} checkpointed (resumed={resumed}, failed={failed})",
                flush=True,
            )
            last_printed = completed

    try:
        report = execute_phase1_final(
            protocol_path=args.protocol, primary_population_path=args.primary_population,
            secondary_population_path=args.secondary_population, pilot_report_path=args.pilot_report,
            snapshot_root=args.snapshot_root, gate6_report_path=args.gate6_report,
            database=args.database, work_root=args.work_root, report_path=args.report,
            repo_root=args.repo_root, poetry_lock_path=args.poetry_lock,
            ram_bytes=args.ram_bytes, workers=args.workers, progress_callback=progress,
        )
    except (
        CalibrationStrengthSnapshotError, Gate6CalibrationReportError, FinalProtocolError,
        FinalExecutionError, OSError, subprocess.SubprocessError, TypeError, ValueError,
    ) as exc:
        print(f"Phase-1 final execution failed: {exc}", file=sys.stderr)
        return 2

    print(render_phase1_final_execution_report_text(report), end="")
    return 0 if report.failure_free else 1


if __name__ == "__main__":
    raise SystemExit(main())
