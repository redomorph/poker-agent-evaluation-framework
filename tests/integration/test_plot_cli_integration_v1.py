from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pokerbench.cli.main import build_parser
from pokerbench.cli.plot import PublicPlotResult


main_module = importlib.import_module("pokerbench.cli.main")


def test_public_parser_exposes_plot_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "plot",
            "--report",
            "first.json",
            "--report",
            "second.json",
            "--output",
            "plot-output",
        ]
    )

    assert args.command == "plot"
    assert args.report_paths == [Path("first.json"), Path("second.json")]
    assert args.output == Path("plot-output")
    assert args.ci_kind == "bootstrap"


def test_public_plot_dispatches_to_plot_service(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}
    expected = PublicPlotResult(
        output=Path("plot-output"),
        svg_path=Path("plot-output/plot.svg"),
        manifest_path=Path("plot-output/manifest.json"),
        report_count=1,
        ci_kind="normal",
        bundle_hash="a" * 64,
    )

    def fake_execute_public_plot(**kwargs: object) -> PublicPlotResult:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(
        main_module,
        "execute_public_plot",
        fake_execute_public_plot,
    )
    monkeypatch.setattr(
        main_module,
        "render_public_plot_text",
        lambda result: "plot complete",
    )

    exit_code = main_module.main(
        [
            "plot",
            "--report",
            "battle.json",
            "--output",
            "plot-output",
            "--ci",
            "normal",
        ]
    )

    assert exit_code == 0
    assert observed == {
        "report_paths": [Path("battle.json")],
        "output": Path("plot-output"),
        "ci_kind": "normal",
    }
    assert "plot complete" in capsys.readouterr().out


def test_public_plot_failure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**kwargs: object) -> PublicPlotResult:
        raise ValueError("bad battle report")

    monkeypatch.setattr(main_module, "execute_public_plot", fail)

    exit_code = main_module.main(
        [
            "plot",
            "--report",
            "battle.json",
            "--output",
            "plot-output",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Plot failed: bad battle report" in captured.err
