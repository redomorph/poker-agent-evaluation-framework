from __future__ import annotations

from pathlib import Path

import pytest

import pokerbench.experiments.final_execution_cli as cli


def test_final_execution_cli_parser_freezes_internal_defaults() -> None:
    args = cli.build_parser().parse_args(["--ram-bytes", "17179869184"])
    assert args.protocol == Path("configs/experiments/phase1-baseline-experiment-v1.json")
    assert args.pilot_report == Path("data/phase1/pilot/phase1-pilot-report.json")
    assert args.database == Path("data/phase1/final/phase1-baseline-v1.sqlite")
    assert args.work_root == Path("data/phase1/final/work")
    assert args.report == Path("data/phase1/final/phase1-baseline-report.json")
    assert args.workers == 1
    assert args.progress_every == 25


def test_final_execution_cli_rejects_nonpositive_execution_parameters() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ram-bytes", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--ram-bytes", "1", "--workers", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--ram-bytes", "1", "--progress-every", "0"])


def test_final_execution_cli_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class Report:
        failure_free = True
    monkeypatch.setattr(cli, "execute_phase1_final", lambda **kwargs: Report())
    monkeypatch.setattr(cli, "render_phase1_final_execution_report_text", lambda report: "ok\n")
    assert cli.main(["--ram-bytes", "1"]) == 0
    assert "ok" in capsys.readouterr().out

    class Failed:
        failure_free = False
    monkeypatch.setattr(cli, "execute_phase1_final", lambda **kwargs: Failed())
    assert cli.main(["--ram-bytes", "1"]) == 1

    def fail(**kwargs):
        raise cli.FinalExecutionError("boom")
    monkeypatch.setattr(cli, "execute_phase1_final", fail)
    assert cli.main(["--ram-bytes", "1"]) == 2
    assert "boom" in capsys.readouterr().err


def test_final_execution_resumes_valid_and_failed_checkpoints_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import pokerbench.experiments.final_execution as execution_module
    from pokerbench.engine.config import BASELINE_GAME_CONFIG
    from pokerbench.experiments.config import ExecutionEnvironment
    from pokerbench.experiments.final_execution import FinalJobExecution
    from pokerbench.experiments.final_protocol import FinalJob
    from pokerbench.replay.archive import MatchRecord

    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "configs" / "experiments" / "phase1-baseline-experiment-v1.json"
    primary_path = root / "configs" / "populations" / "baseline-synthetic-v1.json"
    secondary_path = root / "configs" / "populations" / "balanced-v1.json"
    jobs = (
        FinalJob(
            cell_id="random-vs-nit", hero_id="random", hero_version="random-baseline-v1",
            opponent_profile_id="nit", opponent_profile_version="static-profiles-v2",
            master_seed=6_200_000,
        ),
        FinalJob(
            cell_id="random-vs-nit", hero_id="random", hero_version="random-baseline-v1",
            opponent_profile_id="nit", opponent_profile_version="static-profiles-v2",
            master_seed=6_200_001,
        ),
    )
    acceptance = SimpleNamespace(profile_name="nit", gate_map_hash="e" * 64)
    context = SimpleNamespace(
        gate6_report=SimpleNamespace(
            report_hash="ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a",
            gate6_result=SimpleNamespace(results=(acceptance,)),
        ),
        reference=SimpleNamespace(reference_strength_hash="d" * 64),
    )

    monkeypatch.setattr(execution_module, "verify_pilot_report_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(execution_module, "iter_final_jobs", lambda protocol: jobs)
    monkeypatch.setattr(execution_module, "load_verified_static_profile_context", lambda **kwargs: context)
    monkeypatch.setattr(execution_module, "build_verified_static_profile_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(execution_module, "_clean_git_commit", lambda repo_root: "a" * 40)
    monkeypatch.setattr(
        execution_module,
        "capture_execution_environment",
        lambda **kwargs: ExecutionEnvironment(
            python_version="3.14.0", engine_version="git:" + "a" * 40,
            dependency_lockfile="poetry.lock", poetry_lock_hash="b" * 64,
            git_commit="a" * 40, cpu_model="Test CPU", cpu_core_count=8,
            cpu_architecture="AMD64", ram_bytes=16 * 1024**3, os_identifier="TestOS",
        ),
    )

    calls: list[int] = []

    def fake_worker(request):
        calls.append(request.job.master_seed)
        record = MatchRecord(
            match_id=execution_module.final_job_match_id(request.binding, request.job),
            experiment_id=f"phase1-baseline-experiment-v1:{request.binding.binding_hash}",
            experiment_config_hash=request.experiment_config_hash,
            master_seed=request.job.master_seed, game_config=BASELINE_GAME_CONFIG, hands=(),
        )
        if request.job.master_seed == 6_200_000:
            return FinalJobExecution.from_runner_result(
                job=request.job, binding=request.binding,
                result_paired_differences=(1.5,), num_valid_pairs=1, failed_runs=(),
                elapsed_ns=100, match_record=record,
            )
        return FinalJobExecution.from_runner_result(
            job=request.job, binding=request.binding,
            result_paired_differences=(), num_valid_pairs=0,
            failed_runs=({"details": "failed"},), elapsed_ns=120, match_record=record,
        )

    monkeypatch.setattr(execution_module, "_execute_worker_request", fake_worker)

    class FakeReport:
        complete = True
        failure_free = False
        report_hash = "9" * 64

    monkeypatch.setattr(execution_module, "build_final_report", lambda **kwargs: FakeReport())
    monkeypatch.setattr(execution_module, "save_phase1_final_execution_report", lambda report, path: Path(path))

    database = tmp_path / "final.sqlite"
    work_root = tmp_path / "work"
    lock_path = tmp_path / "poetry.lock"
    lock_path.write_text("lock", encoding="utf-8")

    first = execution_module.execute_phase1_final(
        protocol_path=protocol_path, primary_population_path=primary_path,
        secondary_population_path=secondary_path, pilot_report_path=tmp_path / "pilot.json",
        snapshot_root=tmp_path / "gate6-snapshots", gate6_report_path=tmp_path / "gate6.json",
        database=database, work_root=work_root, report_path=tmp_path / "report.json",
        repo_root=root, poetry_lock_path=lock_path, ram_bytes=16 * 1024**3, workers=1,
    )
    assert first.failure_free is False
    assert calls == [6_200_000, 6_200_001]

    calls.clear()
    monkeypatch.setattr(
        execution_module, "_execute_worker_request",
        lambda request: (_ for _ in ()).throw(AssertionError("checkpointed final seed reran")),
    )
    second = execution_module.execute_phase1_final(
        protocol_path=protocol_path, primary_population_path=primary_path,
        secondary_population_path=secondary_path, pilot_report_path=tmp_path / "pilot.json",
        snapshot_root=tmp_path / "gate6-snapshots", gate6_report_path=tmp_path / "gate6.json",
        database=database, work_root=work_root, report_path=tmp_path / "report.json",
        repo_root=root, poetry_lock_path=lock_path, ram_bytes=16 * 1024**3, workers=1,
    )
    assert calls == []
    assert second.report_hash == first.report_hash
