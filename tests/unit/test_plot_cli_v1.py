from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pokerbench.cli.plot import (
    PLOT_BUNDLE_VERSION,
    PUBLIC_PLOT_CLI_VERSION,
    execute_public_plot,
    load_battle_plot_point,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _report_payload(
    *,
    bot_a: str = "random",
    bot_b: str = "tag",
    mean: float = 1.25,
    normal_low: float = -0.50,
    normal_high: float = 3.00,
    bootstrap_low: float = -0.25,
    bootstrap_high: float = 2.75,
    complete: bool = True,
    failed_runs: int = 0,
) -> dict[str, object]:
    evaluation_version = "statistical-evaluation-test-v1"
    return {
        "battle_cli_version": "public-battle-cli-v1",
        "random_baseline_version": "random-baseline-v1",
        "static_profile_version": "static-profiles-v2",
        "parametric_policy_version": "parametric-policy-v1",
        "evaluation_version": evaluation_version,
        "bot_a": bot_a,
        "bot_b": bot_b,
        "bot_a_version": "bot-a-v1",
        "bot_b_version": "bot-b-v1",
        "seed_start": 0,
        "seed_count": 2,
        "hands_per_seed": 1,
        "seeds": [0, 1],
        "requested_independent_units": 2,
        "valid_independent_units": 2,
        "requested_engine_hands": 4,
        "analyzed_engine_hands": 4,
        "failed_runs": failed_runs,
        "failure_policy": "exclude_run",
        "match_ids": ["match-0", "match-1"],
        "statistical_unit": "paired_deal",
        "game_config_hash": "a" * 64,
        "gate6_report_hash": None,
        "bootstrap_seed": 0,
        "bootstrap_resamples": 10_000,
        "confidence_level": 0.95,
        "complete": complete,
        "evaluation": {
            "statistical_unit": "paired_deal",
            "num_independent_units": 2,
            "num_analyzed_simulated_hands": 4,
            "num_failed_runs": failed_runs,
            "mean_difference_bb100": mean,
            "std_difference_bb100": 2.0,
            "normal_ci_low_bb100": normal_low,
            "normal_ci_high_bb100": normal_high,
            "bootstrap_ci_low_bb100": bootstrap_low,
            "bootstrap_ci_high_bb100": bootstrap_high,
            "effect_size_dz": 0.625,
            "confidence_level": 0.95,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 0,
            "evaluation_version": evaluation_version,
            "percentile_method": "linear",
        },
    }


def _write_report(
    path: Path,
    **overrides: object,
) -> str:
    report = _report_payload(**overrides)
    report_hash = hashlib.sha256(
        _canonical_json(report).encode("utf-8")
    ).hexdigest()
    outer = {
        "report": report,
        "report_hash": report_hash,
    }
    path.write_text(
        _canonical_json(outer) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report_hash


def test_load_battle_plot_point_uses_bootstrap_ci_by_default(
    tmp_path: Path,
) -> None:
    source = tmp_path / "battle.json"
    report_hash = _write_report(source)

    point = load_battle_plot_point(source)

    assert point.report_hash == report_hash
    assert point.bot_a == "random"
    assert point.bot_b == "tag"
    assert point.mean_difference_bb100 == 1.25
    assert point.ci_low_bb100 == -0.25
    assert point.ci_high_bb100 == 2.75
    assert point.ci_kind == "bootstrap"


def test_load_battle_plot_point_can_select_normal_ci(
    tmp_path: Path,
) -> None:
    source = tmp_path / "battle.json"
    _write_report(source)

    point = load_battle_plot_point(source, ci_kind="normal")

    assert point.ci_low_bb100 == -0.50
    assert point.ci_high_bb100 == 3.00
    assert point.ci_kind == "normal"


def test_tampered_battle_report_hash_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "battle.json"
    _write_report(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["report"]["bot_b"] = "lag"
    source.write_text(
        _canonical_json(payload) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="hash"):
        load_battle_plot_point(source)


def test_incomplete_battle_report_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "battle.json"
    _write_report(source, complete=False, failed_runs=1)

    with pytest.raises(ValueError, match="complete"):
        load_battle_plot_point(source)


def test_invalid_ci_order_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "battle.json"
    _write_report(
        source,
        bootstrap_low=3.0,
        bootstrap_high=-1.0,
    )

    with pytest.raises(ValueError, match="lower"):
        load_battle_plot_point(source)


def test_execute_public_plot_writes_svg_and_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "battle.json"
    report_hash = _write_report(source)
    output = tmp_path / "plot-bundle"

    result = execute_public_plot(
        report_paths=(source,),
        output=output,
    )

    assert result.output == output
    assert result.report_count == 1
    assert result.ci_kind == "bootstrap"
    assert (output / "plot.svg").is_file()
    assert (output / "manifest.json").is_file()

    svg = (output / "plot.svg").read_text(encoding="utf-8")
    assert "random vs tag" in svg
    assert "Paired EV difference (BB/100)" in svg
    assert "95% bootstrap CI" in svg

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["public_plot_cli_version"] == PUBLIC_PLOT_CLI_VERSION
    assert manifest["plot_bundle_version"] == PLOT_BUNDLE_VERSION
    assert manifest["source_report_hashes"] == [report_hash]
    assert manifest["report_count"] == 1
    assert len(manifest["files"]["plot.svg"]) == 64
    assert manifest["bundle_hash"] == result.bundle_hash


def test_plot_bundle_is_byte_identical_independent_of_input_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "z.json"
    second = tmp_path / "a.json"
    _write_report(first, bot_a="tag", bot_b="lag", mean=2.0)
    _write_report(second, bot_a="random", bot_b="tag", mean=-1.0)

    output_a = tmp_path / "plot-a"
    output_b = tmp_path / "plot-b"

    execute_public_plot(
        report_paths=(first, second),
        output=output_a,
    )
    execute_public_plot(
        report_paths=(second, first),
        output=output_b,
    )

    assert (output_a / "plot.svg").read_bytes() == (
        output_b / "plot.svg"
    ).read_bytes()
    assert (output_a / "manifest.json").read_bytes() == (
        output_b / "manifest.json"
    ).read_bytes()


def test_duplicate_source_evidence_is_rejected_without_output(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    duplicate = tmp_path / "duplicate.json"
    _write_report(first)
    duplicate.write_bytes(first.read_bytes())
    output = tmp_path / "plot"

    with pytest.raises(ValueError, match="Duplicate"):
        execute_public_plot(
            report_paths=(first, duplicate),
            output=output,
        )

    assert not output.exists()


def test_existing_output_directory_is_never_overwritten(
    tmp_path: Path,
) -> None:
    source = tmp_path / "battle.json"
    _write_report(source)
    output = tmp_path / "plot"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        execute_public_plot(
            report_paths=(source,),
            output=output,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_empty_report_selection_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        execute_public_plot(
            report_paths=(),
            output=tmp_path / "plot",
        )


def test_manifest_hashes_match_written_plot_and_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "battle.json"
    _write_report(source, bot_a="tag", bot_b="lag")
    output = tmp_path / "plot"

    result = execute_public_plot(
        report_paths=(source,),
        output=output,
        ci_kind="normal",
    )

    plot_bytes = (output / "plot.svg").read_bytes()
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["files"]["plot.svg"] == hashlib.sha256(
        plot_bytes
    ).hexdigest()

    bundle_hash = manifest.pop("bundle_hash")
    assert bundle_hash == hashlib.sha256(
        _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    assert bundle_hash == result.bundle_hash
