"""CLI for integrity-audited descriptive analysis of the completed adaptive pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pokerbench.experiments.adaptive_pilot_postrun import (
    AdaptivePilotPostRunError,
    analyze_adaptive_pilot_postrun,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pokerbench.experiments.adaptive_pilot_postrun_cli",
        description="Audit and descriptively analyze the completed frozen Phase-2 adaptive pilot.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--protocol", type=Path, default=Path("configs/experiments/phase2-adaptive-pilot-v1.json"))
    analyze.add_argument("--execution-contract", type=Path, default=Path("configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json"))
    analyze.add_argument("--primary-population", type=Path, default=Path("configs/populations/baseline-synthetic-v1.json"))
    analyze.add_argument("--secondary-population", type=Path, default=Path("configs/populations/balanced-v1.json"))
    analyze.add_argument("--database", type=Path, default=Path("data/phase2/adaptive-pilot-parallel/phase2-adaptive-pilot-v1.sqlite"))
    analyze.add_argument("--work-root", type=Path, default=Path("data/phase2/adaptive-pilot-parallel/work"))
    analyze.add_argument("--report", type=Path, default=Path("data/phase2/adaptive-pilot-parallel/phase2-adaptive-pilot-report-v1.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze_adaptive_pilot_postrun(
            protocol_path=args.protocol,
            execution_contract_path=args.execution_contract,
            primary_population_path=args.primary_population,
            secondary_population_path=args.secondary_population,
            database=args.database,
            work_root=args.work_root,
            report_path=args.report,
        )
    except (AdaptivePilotPostRunError, OSError, TypeError, ValueError) as exc:
        print(f"Adaptive pilot post-run analyze failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"adaptive pilot post-run COMPLETE: {report.completed_jobs} jobs, {report.valid_pairs} valid pairs, "
        f"{report.invalid_pairs} invalid pairs, {report.recorded_engine_hands:,} engine hands; "
        f"report_hash={report.report_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
