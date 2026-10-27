"""Internal CLI for the Spec-38 H4 holdout-variance execution gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pokerbench.experiments.h4_holdout_variance_parallel_launch import (
    DEFAULT_H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_PATH,
    H4HoldoutVarianceParallelLaunchError,
    assert_h4_holdout_variance_parallel_launch_ready,
    plan_h4_holdout_variance_parallel_launch,
    run_h4_holdout_variance_parallel_launch,
    status_h4_holdout_variance_parallel_launch,
)


DEFAULT_PROTOCOL = Path("configs/experiments/phase2-h4-holdout-variance-pilot-v1.json")
DEFAULT_POPULATION = Path("configs/populations/phase2-h4-holdout-population-v1.json")
DEFAULT_ACCEPTANCE = Path("data/gate6/h4-holdout-acceptance-v1.json")
DEFAULT_SNAPSHOT_ROOT = Path("data/gate6/snapshots")
DEFAULT_GATE6_REPORT = Path("data/gate6/gate6-report.json")
DEFAULT_REFERENCE = Path("configs/experiments/preflop-range-strength-reference-v1.json")
DEFAULT_DATABASE = Path("data/phase2/h4-holdout-variance/variance-pilot.sqlite")
DEFAULT_WORK_ROOT = Path("data/phase2/h4-holdout-variance/work")
DEFAULT_REPO_ROOT = Path(".")
DEFAULT_POETRY_LOCK = Path("poetry.lock")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_common_assets(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--execution-contract",
        type=Path,
        default=DEFAULT_H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_PATH,
    )
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--gate6-report", type=Path, default=DEFAULT_GATE6_REPORT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--poetry-lock", type=Path, default=DEFAULT_POETRY_LOCK)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="h4-holdout-variance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    _add_common_assets(plan)
    plan.add_argument("--ram-bytes", required=True, type=_positive_int)
    plan.add_argument("--workers", type=_positive_int, default=4)

    status = subparsers.add_parser("status")
    status.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    status.add_argument(
        "--execution-contract",
        type=Path,
        default=DEFAULT_H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_PATH,
    )
    status.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    status.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)

    readiness = subparsers.add_parser("readiness")
    _add_common_assets(readiness)

    run = subparsers.add_parser("run")
    _add_common_assets(run)
    resume = subparsers.add_parser("resume")
    _add_common_assets(resume)
    return parser


def _launch_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "protocol_path": args.protocol,
        "execution_contract_path": args.execution_contract,
        "population_path": args.population,
        "acceptance_path": args.acceptance,
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
            status = plan_h4_holdout_variance_parallel_launch(
                **_launch_kwargs(args),
                ram_bytes=args.ram_bytes,
                workers=args.workers,
            )
            print(
                "H4 holdout variance pilot planned: "
                f"{status.planned_jobs} jobs, {status.planned_pairs} matched pairs, "
                f"{status.planned_clusters} population clusters, "
                f"{status.planned_engine_hands:,} engine hands; workers=4; "
                f"binding={status.binding_hash}"
            )
            return 0
        if args.command == "status":
            status = status_h4_holdout_variance_parallel_launch(
                protocol_path=args.protocol,
                execution_contract_path=args.execution_contract,
                population_path=args.population,
                work_root=args.work_root,
            )
            print(
                "H4 holdout variance pilot status: "
                f"terminal={status.terminal_jobs}/{status.planned_jobs}, planned={status.planned}, "
                f"running={status.running}, completed={status.completed}, failed={status.failed}; "
                f"binding={status.binding_hash}"
            )
            return 0
        if args.command == "readiness":
            ready = assert_h4_holdout_variance_parallel_launch_ready(**_launch_kwargs(args))
            print(
                "H4 holdout variance pilot READY: "
                f"{ready.planned_jobs} jobs, {ready.planned_pairs} matched pairs, "
                f"{ready.planned_clusters} population clusters, "
                f"{ready.planned_engine_hands:,} engine hands; terminal={ready.terminal_jobs}; "
                f"workers=4; binding={ready.binding_hash}"
            )
            return 0
        if args.command in {"run", "resume"}:
            result = run_h4_holdout_variance_parallel_launch(
                **_launch_kwargs(args),
                resume=args.command == "resume",
            )
            print(
                "H4 holdout variance pilot terminal: "
                f"jobs={result.terminal_jobs}, completed={result.completed_jobs}, "
                f"failed={result.failed_jobs}, pairs={result.pair_count}, "
                f"valid_pairs={result.valid_pairs}, invalid_pairs={result.invalid_pairs}, "
                f"clusters={result.cluster_count}, valid_clusters={result.valid_clusters}, "
                f"invalid_clusters={result.invalid_clusters}, "
                f"scientifically_complete={str(result.scientifically_complete).lower()}"
            )
            return 0 if result.scientifically_complete else 1
        raise AssertionError("unreachable command")
    except H4HoldoutVarianceParallelLaunchError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
