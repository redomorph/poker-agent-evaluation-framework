from __future__ import annotations

from pathlib import Path

import pytest

import pokerbench.experiments.pilot_execution_cli as cli


def test_pilot_execution_cli_parser_freezes_internal_command_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--ram-bytes", "17179869184"])

    assert args.protocol == Path("configs/experiments/phase1-pilot-v1.json")
    assert args.primary_population == Path(
        "configs/populations/baseline-synthetic-v1.json"
    )
    assert args.secondary_population == Path("configs/populations/balanced-v1.json")
    assert args.snapshot_root == Path("data/gate6/snapshots")
    assert args.gate6_report == Path("data/gate6/gate6-report.json")
    assert args.database == Path("data/phase1/pilot/phase1-pilot-v1.sqlite")
    assert args.work_root == Path("data/phase1/pilot/work")
    assert args.report == Path("data/phase1/pilot/phase1-pilot-report.json")
    assert args.poetry_lock == Path("poetry.lock")
    assert args.repo_root == Path(".")
    assert args.workers == 1
    assert args.progress_every == 10


def test_pilot_execution_cli_requires_positive_ram_workers_and_progress() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ram-bytes", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--ram-bytes", "1", "--workers", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--ram-bytes", "1", "--progress-every", "0"])


def test_pilot_execution_cli_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class FakeReport:
        failure_free = True
        report_hash = "a" * 64

    monkeypatch.setattr(cli, "execute_phase1_pilot", lambda **kwargs: FakeReport())
    monkeypatch.setattr(cli, "render_phase1_pilot_execution_report_text", lambda report: "ok\n")
    assert cli.main(["--ram-bytes", "1"]) == 0
    assert "ok" in capsys.readouterr().out

    class FailedReport:
        failure_free = False
        report_hash = "b" * 64

    monkeypatch.setattr(cli, "execute_phase1_pilot", lambda **kwargs: FailedReport())
    assert cli.main(["--ram-bytes", "1"]) == 1

    def raise_error(**kwargs):
        raise cli.PilotExecutionError("boom")

    monkeypatch.setattr(cli, "execute_phase1_pilot", raise_error)
    assert cli.main(["--ram-bytes", "1"]) == 2
    assert "boom" in capsys.readouterr().err


def test_execution_resumes_checkpointed_valid_and_failed_jobs_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import pokerbench.experiments.pilot_execution as execution_module
    from pokerbench.engine.config import BASELINE_GAME_CONFIG
    from pokerbench.experiments.config import ExecutionEnvironment
    from pokerbench.experiments.frozen_protocol import PilotJob
    from pokerbench.experiments.pilot_execution import PilotJobExecution
    from pokerbench.replay.archive import MatchRecord

    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "configs" / "experiments" / "phase1-pilot-v1.json"
    primary_path = root / "configs" / "populations" / "baseline-synthetic-v1.json"
    secondary_path = root / "configs" / "populations" / "balanced-v1.json"
    jobs = (
        PilotJob(
            cell_id="random-vs-nit",
            hero_id="random",
            hero_version="random-baseline-v1",
            opponent_profile_id="nit",
            opponent_profile_version="static-profiles-v2",
            master_seed=6_100_000,
        ),
        PilotJob(
            cell_id="random-vs-nit",
            hero_id="random",
            hero_version="random-baseline-v1",
            opponent_profile_id="nit",
            opponent_profile_version="static-profiles-v2",
            master_seed=6_100_001,
        ),
    )

    acceptance = SimpleNamespace(profile_name="nit", gate_map_hash="e" * 64)
    gate6_report = SimpleNamespace(
        report_hash="c" * 64,
        gate6_result=SimpleNamespace(results=(acceptance,)),
    )
    context = SimpleNamespace(
        gate6_report=gate6_report,
        reference=SimpleNamespace(reference_strength_hash="d" * 64),
    )

    monkeypatch.setattr(execution_module, "iter_pilot_jobs", lambda protocol: jobs)
    monkeypatch.setattr(
        execution_module,
        "load_verified_static_profile_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        execution_module,
        "build_verified_static_profile_agent",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(execution_module, "_clean_git_commit", lambda repo_root: "a" * 40)
    monkeypatch.setattr(
        execution_module,
        "capture_execution_environment",
        lambda **kwargs: ExecutionEnvironment(
            python_version="3.14.0",
            engine_version="git:" + "a" * 40,
            dependency_lockfile="poetry.lock",
            poetry_lock_hash="b" * 64,
            git_commit="a" * 40,
            cpu_model="Test CPU",
            cpu_core_count=8,
            cpu_architecture="AMD64",
            ram_bytes=16 * 1024**3,
            os_identifier="TestOS",
        ),
    )

    calls: list[int] = []

    def fake_worker(request):
        calls.append(request.job.master_seed)
        record = MatchRecord(
            match_id=execution_module.pilot_job_match_id(request.binding, request.job),
            experiment_id=f"phase1-pilot-v1:{request.binding.binding_hash}",
            experiment_config_hash=request.experiment_config_hash,
            master_seed=request.job.master_seed,
            game_config=BASELINE_GAME_CONFIG,
            hands=(),
        )
        if request.job.master_seed == 6_100_000:
            return PilotJobExecution.from_runner_result(
                job=request.job,
                binding=request.binding,
                result_paired_differences=(1.5,),
                num_valid_pairs=1,
                failed_runs=(),
                elapsed_ns=100,
                match_record=record,
            )
        return PilotJobExecution.from_runner_result(
            job=request.job,
            binding=request.binding,
            result_paired_differences=(),
            num_valid_pairs=0,
            failed_runs=({"hand": 0, "config": "A_BTN", "details": "failed"},),
            elapsed_ns=120,
            match_record=record,
        )

    monkeypatch.setattr(execution_module, "_execute_worker_request", fake_worker)
    database = tmp_path / "pilot.sqlite"
    work_root = tmp_path / "work"
    report_path = tmp_path / "report.json"
    lock_path = tmp_path / "poetry.lock"
    lock_path.write_text("lock", encoding="utf-8")

    first = execution_module.execute_phase1_pilot(
        protocol_path=protocol_path,
        primary_population_path=primary_path,
        secondary_population_path=secondary_path,
        snapshot_root=tmp_path / "gate6-snapshots",
        gate6_report_path=tmp_path / "gate6-report.json",
        database=database,
        work_root=work_root,
        report_path=report_path,
        repo_root=root,
        poetry_lock_path=lock_path,
        ram_bytes=16 * 1024**3,
        workers=1,
    )

    assert calls == [6_100_000, 6_100_001]
    assert first.complete
    assert not first.failure_free
    assert first.valid_units == 1
    assert first.failed_units == 1

    calls.clear()

    def must_not_execute(request):
        raise AssertionError("checkpointed pilot seed was executed again")

    monkeypatch.setattr(execution_module, "_execute_worker_request", must_not_execute)
    second = execution_module.execute_phase1_pilot(
        protocol_path=protocol_path,
        primary_population_path=primary_path,
        secondary_population_path=secondary_path,
        snapshot_root=tmp_path / "gate6-snapshots",
        gate6_report_path=tmp_path / "gate6-report.json",
        database=database,
        work_root=work_root,
        report_path=report_path,
        repo_root=root,
        poetry_lock_path=lock_path,
        ram_bytes=16 * 1024**3,
        workers=1,
    )

    assert calls == []
    assert second.report_hash == first.report_hash
