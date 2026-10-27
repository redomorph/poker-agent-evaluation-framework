from __future__ import annotations

from pathlib import Path

from pokerbench.experiments import adaptive_pilot_postrun_cli as cli


def test_postrun_cli_analyze_uses_frozen_parallel_defaults_and_prints_hash(monkeypatch, capsys):  # type: ignore[no-untyped-def]
    class Report:
        report_hash = "a" * 64
        completed_jobs = 50
        valid_pairs = 25
        invalid_pairs = 0
        recorded_engine_hands = 18_000

    captured = {}

    def fake_analyze(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return Report()

    monkeypatch.setattr(cli, "analyze_adaptive_pilot_postrun", fake_analyze)
    assert cli.main(["analyze"]) == 0
    assert captured["database"] == Path("data/phase2/adaptive-pilot-parallel/phase2-adaptive-pilot-v1.sqlite")
    assert captured["work_root"] == Path("data/phase2/adaptive-pilot-parallel/work")
    assert captured["report_path"] == Path("data/phase2/adaptive-pilot-parallel/phase2-adaptive-pilot-report-v1.json")
    output = capsys.readouterr().out
    assert "50 jobs" in output and "25 valid pairs" in output and "18,000 engine hands" in output
    assert "a" * 64 in output
