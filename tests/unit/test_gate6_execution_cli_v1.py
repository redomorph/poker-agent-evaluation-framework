"""Tests for the synchronous Gate-6 command-line execution wrapper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pokerbench.evaluation import gate6_execution_cli
from pokerbench.evaluation.gate6_execution import (
    Gate6CalibrationReportIntegrityError,
)


def test_main_executes_gate6_with_explicit_paths_and_controls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    report_path = tmp_path / "gate6-report.json"
    report = SimpleNamespace(passed=True)
    calls: list[tuple[Path, Path, int, int]] = []

    def fake_execute(
        snapshot_root_arg: str | Path,
        report_path_arg: str | Path,
        *,
        workers: int,
        checkpoint_every: int,
    ) -> object:
        calls.append(
            (
                Path(snapshot_root_arg),
                Path(report_path_arg),
                workers,
                checkpoint_every,
            )
        )
        return report

    monkeypatch.setattr(
        gate6_execution_cli,
        "execute_gate6_calibration",
        fake_execute,
    )
    monkeypatch.setattr(
        gate6_execution_cli,
        "render_gate6_calibration_report_text",
        lambda value: "Gate 6: PASS\n",
    )

    exit_code = gate6_execution_cli.main(
        [
            "--snapshot-root",
            str(snapshot_root),
            "--report",
            str(report_path),
            "--workers",
            "8",
            "--checkpoint-every",
            "17",
        ]
    )

    assert exit_code == 0
    assert calls == [(snapshot_root, report_path, 8, 17)]
    captured = capsys.readouterr()
    assert "Gate-6 calibration execution" in captured.out
    assert "Gate 6: PASS" in captured.out
    assert captured.err == ""


def test_scientific_gate_failure_returns_exit_code_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        gate6_execution_cli,
        "execute_gate6_calibration",
        lambda *args, **kwargs: SimpleNamespace(passed=False),
    )
    monkeypatch.setattr(
        gate6_execution_cli,
        "render_gate6_calibration_report_text",
        lambda value: "Gate 6: FAIL\n",
    )

    exit_code = gate6_execution_cli.main(
        [
            "--snapshot-root",
            str(tmp_path / "snapshots"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 1


def test_known_integrity_failure_returns_exit_code_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise Gate6CalibrationReportIntegrityError("hash mismatch")

    monkeypatch.setattr(
        gate6_execution_cli,
        "execute_gate6_calibration",
        fail,
    )

    exit_code = gate6_execution_cli.main(
        [
            "--snapshot-root",
            str(tmp_path / "snapshots"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out.startswith("Gate-6 calibration execution")
    assert "Gate-6 execution failed: hash mismatch" in captured.err


def test_workers_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        gate6_execution_cli.main(
            [
                "--snapshot-root",
                str(tmp_path / "snapshots"),
                "--report",
                str(tmp_path / "report.json"),
                "--workers",
                "0",
            ]
        )

    assert exc_info.value.code == 2


def test_checkpoint_every_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        gate6_execution_cli.main(
            [
                "--snapshot-root",
                str(tmp_path / "snapshots"),
                "--report",
                str(tmp_path / "report.json"),
                "--checkpoint-every",
                "0",
            ]
        )

    assert exc_info.value.code == 2


def test_snapshot_root_and_report_are_required() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gate6_execution_cli.main([])

    assert exc_info.value.code == 2


def test_main_starts_and_stops_progress_monitor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, object]] = []

    class FakeMonitor:
        def __init__(
            self,
            snapshot_root: str | Path,
            *,
            interval_seconds: float,
        ) -> None:
            events.append(
                (
                    "init",
                    (Path(snapshot_root), interval_seconds),
                )
            )

        def start(self) -> None:
            events.append(("start", None))

        def stop(self) -> None:
            events.append(("stop", None))

    monkeypatch.setattr(
        gate6_execution_cli,
        "Gate6ProgressMonitor",
        FakeMonitor,
    )
    monkeypatch.setattr(
        gate6_execution_cli,
        "execute_gate6_calibration",
        lambda *args, **kwargs: SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(
        gate6_execution_cli,
        "render_gate6_calibration_report_text",
        lambda value: "Gate 6: PASS\n",
    )

    exit_code = gate6_execution_cli.main(
        [
            "--snapshot-root",
            str(tmp_path / "snapshots"),
            "--report",
            str(tmp_path / "report.json"),
            "--progress-interval",
            "7.5",
        ]
    )

    assert exit_code == 0
    assert events == [
        ("init", (tmp_path / "snapshots", 7.5)),
        ("start", None),
        ("stop", None),
    ]


def test_progress_monitor_is_stopped_when_execution_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class FakeMonitor:
        def __init__(
            self,
            snapshot_root: str | Path,
            *,
            interval_seconds: float,
        ) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    def fail(*args: object, **kwargs: object) -> object:
        raise Gate6CalibrationReportIntegrityError("broken")

    monkeypatch.setattr(
        gate6_execution_cli,
        "Gate6ProgressMonitor",
        FakeMonitor,
    )
    monkeypatch.setattr(
        gate6_execution_cli,
        "execute_gate6_calibration",
        fail,
    )

    exit_code = gate6_execution_cli.main(
        [
            "--snapshot-root",
            str(tmp_path / "snapshots"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 2
    assert events == ["start", "stop"]


def test_progress_interval_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        gate6_execution_cli.main(
            [
                "--snapshot-root",
                str(tmp_path / "snapshots"),
                "--report",
                str(tmp_path / "report.json"),
                "--progress-interval",
                "0",
            ]
        )

    assert exc_info.value.code == 2
