"""Internal CLI for the frozen Spec-34 four-worker adaptive pilot execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pokerbench.experiments.adaptive_pilot_parallel_launch import (
    AdaptivePilotParallelLaunchError,
    assert_adaptive_pilot_parallel_launch_ready,
    plan_adaptive_pilot_parallel_launch,
    run_adaptive_pilot_parallel_launch,
    status_adaptive_pilot_parallel_launch,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=Path("configs/experiments/phase2-adaptive-pilot-v1.json"))
    parser.add_argument("--execution-contract", type=Path, default=Path("configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json"))
    parser.add_argument("--primary-population", type=Path, default=Path("configs/populations/baseline-synthetic-v1.json"))
    parser.add_argument("--secondary-population", type=Path, default=Path("configs/populations/balanced-v1.json"))
    parser.add_argument("--snapshot-root", type=Path, default=Path("data/gate6/snapshots"))
    parser.add_argument("--gate6-report", type=Path, default=Path("data/gate6/gate6-report.json"))
    parser.add_argument("--reference", type=Path, default=Path("configs/experiments/preflop-range-strength-reference-v1.json"))
    parser.add_argument("--database", type=Path, default=Path("data/phase2/adaptive-pilot-parallel/phase2-adaptive-pilot-v1.sqlite"))
    parser.add_argument("--work-root", type=Path, default=Path("data/phase2/adaptive-pilot-parallel/work"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--poetry-lock", type=Path, default=Path("poetry.lock"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pokerbench.experiments.adaptive_pilot_parallel_cli",
        description="Plan/check/execute the frozen four-worker Phase-2 adaptive pilot.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "status", "readiness", "run", "resume"):
        command = sub.add_parser(name)
        _add_paths(command)
        if name == "plan":
            command.add_argument("--ram-bytes", type=_positive_int, required=True)
            command.add_argument("--workers", type=_positive_int, default=4)
    return parser


def _kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "protocol_path": args.protocol,
        "execution_contract_path": args.execution_contract,
        "primary_population_path": args.primary_population,
        "secondary_population_path": args.secondary_population,
        "snapshot_root": args.snapshot_root,
        "gate6_report_path": args.gate6_report,
        "reference_path": args.reference,
        "database": args.database,
        "work_root": args.work_root,
        "repo_root": args.repo_root,
        "poetry_lock_path": args.poetry_lock,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            status = plan_adaptive_pilot_parallel_launch(
                **_kwargs(args), ram_bytes=args.ram_bytes, workers=args.workers
            )
            print(
                f"parallel adaptive pilot planned: {status.planned_jobs} jobs, {status.planned_pairs} matched pairs, "
                f"{status.planned_engine_hands:,} engine hands; workers=4; binding={status.binding_hash}"
            )
            return 0
        if args.command == "status":
            status = status_adaptive_pilot_parallel_launch(
                protocol_path=args.protocol, work_root=args.work_root,
                execution_contract_path=args.execution_contract,
            )
            print(
                f"parallel adaptive pilot status: terminal={status.terminal_jobs}/{status.planned_jobs}, "
                f"planned={status.planned}, running={status.running}, completed={status.completed}, failed={status.failed}; "
                f"binding={status.binding_hash}"
            )
            return 0
        if args.command == "readiness":
            ready = assert_adaptive_pilot_parallel_launch_ready(**_kwargs(args))
            print(
                f"parallel adaptive pilot READY: {ready.planned_jobs} jobs, {ready.planned_pairs} planned pairs, "
                f"{ready.planned_engine_hands:,} engine hands; terminal={ready.terminal_jobs}; workers=4; binding={ready.binding_hash}"
            )
            return 0
        result = run_adaptive_pilot_parallel_launch(**_kwargs(args), resume=args.command == "resume")
        print(
            f"parallel adaptive pilot terminal: jobs={result.terminal_jobs}, completed={result.completed_jobs}, "
            f"failed={result.failed_jobs}, pairs={result.pair_count}, valid_pairs={result.valid_pairs}, "
            f"invalid_pairs={result.invalid_pairs}"
        )
        return 0 if result.failed_jobs == 0 and result.invalid_pairs == 0 else 1
    except (AdaptivePilotParallelLaunchError, OSError, TypeError, ValueError) as exc:
        print(f"Parallel adaptive pilot {args.command} failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
