from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.agents.calibration_environment import (
    CalibrationPhase,
    CalibrationScenario,
)
from pokerbench.agents.profiles import STATIC_PROFILES
from pokerbench.agents.strength_gates import PolicyStrengthReference
from pokerbench.engine.config import BASELINE_GAME_CONFIG, GameConfig
from pokerbench.evaluation.calibration import (
    BehaviorCounts,
    CalibrationMetric,
    calibrate_profile,
)
from pokerbench.evaluation.calibration_snapshots import (
    CalibrationScenarioStrengthSnapshot,
    CalibrationStrengthSnapshotStore,
    create_empty_calibration_snapshot,
)
from pokerbench.evaluation.gate6_execution import (
    GATE6_CALIBRATION_REPORT_VERSION,
    Gate6CalibrationReport,
    Gate6CalibrationReportIntegrityError,
    execute_gate6_calibration,
    load_gate6_calibration_report,
    render_gate6_calibration_report_text,
    save_gate6_calibration_report,
)
from pokerbench.evaluation.policy_acceptance import (
    PolicyCalibrationAcceptance,
    PolicyValidationStrengths,
    StaticProfileGate6Result,
)


_HASH_C = "c" * 64


def _counts_for_profile(profile) -> BehaviorCounts:
    parameters = profile.parameters
    n = 5_000

    aggression = parameters.aggression
    if aggression is None:
        aggressive = 0
        passive = n
    else:
        aggressive = round(n * aggression / (1.0 + aggression))
        passive = n - aggressive

    return BehaviorCounts(
        vpip_opportunities=n,
        vpip_events=round(n * parameters.vpip),
        pfr_opportunities=n,
        pfr_events=round(n * parameters.pfr),
        three_bet_opportunities=n,
        three_bet_events=round(n * parameters.three_bet_freq),
        cbet_opportunities=n,
        cbet_events=round(n * parameters.cbet_freq),
        postflop_aggressive_actions=aggressive,
        postflop_passive_actions=passive,
        fold_to_cbet_opportunities=n,
        fold_to_cbet_events=round(n * parameters.fold_to_cbet),
    )


def _hashes_for_snapshots(
    snapshots: tuple[CalibrationScenarioStrengthSnapshot, ...],
) -> tuple[str, str]:
    by_identity = {(item.phase, item.scenario): item for item in snapshots}

    def fit(scenario: CalibrationScenario) -> tuple[int, ...]:
        return by_identity[(CalibrationPhase.FIT, scenario)].complete_strengths()

    def validation(scenario: CalibrationScenario) -> tuple[int, ...]:
        return by_identity[(CalibrationPhase.VALIDATION, scenario)].complete_strengths()

    reference = PolicyStrengthReference(
        open_btn=fit(CalibrationScenario.OPEN_BTN),
        three_bet_bb=fit(CalibrationScenario.THREE_BET_BB),
        cbet_btn=fit(CalibrationScenario.CBET_BTN),
        fold_to_cbet_bb=fit(CalibrationScenario.FOLD_TO_CBET_BB),
        postflop_aggression_btn=fit(CalibrationScenario.POSTFLOP_AGGRESSION_BTN),
    )
    validation_strengths = PolicyValidationStrengths(
        open_btn=validation(CalibrationScenario.OPEN_BTN),
        three_bet_bb=validation(CalibrationScenario.THREE_BET_BB),
        cbet_btn=validation(CalibrationScenario.CBET_BTN),
        fold_to_cbet_bb=validation(CalibrationScenario.FOLD_TO_CBET_BB),
        postflop_aggression_btn=validation(
            CalibrationScenario.POSTFLOP_AGGRESSION_BTN
        ),
    )
    return (
        reference.reference_strength_hash,
        validation_strengths.validation_strength_hash,
    )


def _gate6_result(
    *,
    force_failure: bool = False,
    snapshots: tuple[CalibrationScenarioStrengthSnapshot, ...] | None = None,
) -> StaticProfileGate6Result:
    artifact_snapshots = _snapshots() if snapshots is None else snapshots
    reference_hash, validation_hash = _hashes_for_snapshots(artifact_snapshots)
    results: list[PolicyCalibrationAcceptance] = []
    for index, profile in enumerate(STATIC_PROFILES.values()):
        counts = _counts_for_profile(profile)
        if force_failure and index == 0:
            counts = replace(counts, vpip_events=5_000)
        calibration_result = calibrate_profile(profile, counts)
        designated = {metric.value: 5_000 for metric in CalibrationMetric}
        results.append(
            PolicyCalibrationAcceptance(
                profile_name=profile.name.value,
                profile_version=profile.version,
                parameter_schema_version=profile.parameters.schema_version,
                reference_strength_hash=reference_hash,
                gate_map_hash=f"{index + 1:064x}",
                validation_strength_hash=validation_hash,
                behavior_counts=counts,
                designated_opportunities=designated,
                calibration_result=calibration_result,
            )
        )
    return StaticProfileGate6Result(results=tuple(results))


def _complete_snapshot(
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    value: int,
) -> CalibrationScenarioStrengthSnapshot:
    empty = create_empty_calibration_snapshot(phase, scenario)
    return replace(empty, strengths=(value,) * 5_000)


def _snapshots() -> tuple[CalibrationScenarioStrengthSnapshot, ...]:
    return tuple(
        _complete_snapshot(phase, scenario, 400_000 + index)
        for index, (phase, scenario) in enumerate(
            (
                (phase, scenario)
                for phase in (CalibrationPhase.FIT, CalibrationPhase.VALIDATION)
                for scenario in CalibrationScenario
            )
        )
    )


def test_report_is_deterministic_and_captures_all_provenance() -> None:
    result = _gate6_result()
    report = Gate6CalibrationReport.from_artifacts(
        result,
        _snapshots(),
        game_config=BASELINE_GAME_CONFIG,
    )

    assert report.report_version == GATE6_CALIBRATION_REPORT_VERSION
    assert report.passed
    assert report.gate6_hash == result.gate6_hash
    reference_hash, validation_hash = _hashes_for_snapshots(_snapshots())
    assert report.reference_strength_hash == reference_hash
    assert report.validation_strength_hash == validation_hash
    assert len(report.snapshot_hashes) == 10
    assert set(report.profile_acceptance_hashes) == set(result.profile_names)
    assert len(report.report_hash) == 64
    assert report.report_hash == Gate6CalibrationReport.from_artifacts(
        result,
        tuple(reversed(_snapshots())),
        game_config=BASELINE_GAME_CONFIG,
    ).report_hash


def test_report_rejects_missing_or_duplicate_snapshot_identity() -> None:
    result = _gate6_result()
    snapshots = _snapshots()

    with pytest.raises(ValueError, match="exactly all ten"):
        Gate6CalibrationReport.from_artifacts(
            result,
            snapshots[:-1],
            game_config=BASELINE_GAME_CONFIG,
        )

    duplicate = snapshots[:-1] + (snapshots[0],)
    with pytest.raises(ValueError, match="exactly all ten"):
        Gate6CalibrationReport.from_artifacts(
            result,
            duplicate,
            game_config=BASELINE_GAME_CONFIG,
        )


def test_report_rejects_mixed_reference_or_validation_hashes() -> None:
    result = _gate6_result()
    first = result.results[0]
    mixed = StaticProfileGate6Result(
        results=(
            replace(first, reference_strength_hash=_HASH_C),
            *result.results[1:],
        )
    )

    with pytest.raises(ValueError, match="reference strength hash"):
        Gate6CalibrationReport.from_artifacts(
            mixed,
            _snapshots(),
            game_config=BASELINE_GAME_CONFIG,
        )


def test_report_round_trip_is_hash_verified_and_typed(tmp_path: Path) -> None:
    report = Gate6CalibrationReport.from_artifacts(
        _gate6_result(),
        _snapshots(),
        game_config=BASELINE_GAME_CONFIG,
    )
    path = tmp_path / "gate6.json"

    save_gate6_calibration_report(report, path)
    loaded = load_gate6_calibration_report(path)

    assert loaded == report
    assert loaded.report_hash == report.report_hash
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_report_tampering_is_rejected(tmp_path: Path) -> None:
    report = Gate6CalibrationReport.from_artifacts(
        _gate6_result(),
        _snapshots(),
        game_config=BASELINE_GAME_CONFIG,
    )
    path = tmp_path / "gate6.json"
    save_gate6_calibration_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["report"]["passed"] = not payload["report"]["passed"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Gate6CalibrationReportIntegrityError):
        load_gate6_calibration_report(path)


def test_loader_rejects_unknown_report_version_even_with_valid_outer_shape(
    tmp_path: Path,
) -> None:
    report = Gate6CalibrationReport.from_artifacts(
        _gate6_result(),
        _snapshots(),
        game_config=BASELINE_GAME_CONFIG,
    )
    path = tmp_path / "gate6.json"
    save_gate6_calibration_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["report"]["report_version"] = "future-report-v99"
    # Deliberately leave the old hash: integrity failure must occur before
    # accepting an unsupported payload.
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Gate6CalibrationReportIntegrityError):
        load_gate6_calibration_report(path)


def test_text_report_contains_profile_metrics_and_hashes() -> None:
    report = Gate6CalibrationReport.from_artifacts(
        _gate6_result(),
        _snapshots(),
        game_config=BASELINE_GAME_CONFIG,
    )

    rendered = render_gate6_calibration_report_text(report)

    assert "Gate 6: PASS" in rendered
    assert f"Gate-6 hash: {report.gate6_hash}" in rendered
    assert f"Report hash: {report.report_hash}" in rendered
    for profile_name in report.gate6_result.profile_names:
        assert profile_name in rendered
    assert "VPIP" in rendered
    assert "FoldToCBet" in rendered


def test_failed_scientific_result_is_still_a_valid_persisted_report(
    tmp_path: Path,
) -> None:
    report = Gate6CalibrationReport.from_artifacts(
        _gate6_result(force_failure=True),
        _snapshots(),
        game_config=BASELINE_GAME_CONFIG,
    )
    path = tmp_path / "gate6-failed.json"

    save_gate6_calibration_report(report, path)

    assert not report.passed
    assert load_gate6_calibration_report(path) == report
    assert "Gate 6: FAIL" in render_gate6_calibration_report_text(report)


def test_execute_orchestrates_snapshots_artifacts_acceptance_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pokerbench.evaluation import gate6_execution as module

    snapshots = _snapshots()
    result = _gate6_result()
    calls: list[tuple[str, object]] = []

    def fake_generate(store, **kwargs):
        assert isinstance(store, CalibrationStrengthSnapshotStore)
        calls.append(("generate", (kwargs["workers"], kwargs["checkpoint_every"])))
        return snapshots

    sentinel_reference = object()
    sentinel_validation = object()

    def fake_reference(store):
        calls.append(("reference", store.root))
        return sentinel_reference

    def fake_validation(store):
        calls.append(("validation", store.root))
        return sentinel_validation

    def fake_evaluate(reference, validation, **kwargs):
        assert reference is sentinel_reference
        assert validation is sentinel_validation
        calls.append(("evaluate", kwargs["game_config"]))
        return result

    monkeypatch.setattr(module, "generate_gate6_calibration_snapshots", fake_generate)
    monkeypatch.setattr(module, "policy_strength_reference_from_store", fake_reference)
    monkeypatch.setattr(module, "policy_validation_strengths_from_store", fake_validation)
    monkeypatch.setattr(module, "evaluate_static_profile_gate6", fake_evaluate)

    report_path = tmp_path / "result" / "gate6.json"
    report = execute_gate6_calibration(
        tmp_path / "snapshots",
        report_path,
        workers=4,
        checkpoint_every=37,
    )

    assert report.passed
    assert report_path.is_file()
    assert load_gate6_calibration_report(report_path) == report
    assert calls[0] == ("generate", (4, 37))
    assert [name for name, _ in calls] == [
        "generate",
        "reference",
        "validation",
        "evaluate",
    ]


def test_execute_refuses_to_silently_replace_different_existing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pokerbench.evaluation import gate6_execution as module

    snapshots = _snapshots()
    existing = Gate6CalibrationReport.from_artifacts(
        _gate6_result(), snapshots, game_config=BASELINE_GAME_CONFIG
    )
    report_path = tmp_path / "gate6.json"
    save_gate6_calibration_report(existing, report_path)

    monkeypatch.setattr(
        module,
        "generate_gate6_calibration_snapshots",
        lambda store, **kwargs: snapshots,
    )
    monkeypatch.setattr(
        module,
        "policy_strength_reference_from_store",
        lambda store: object(),
    )
    monkeypatch.setattr(
        module,
        "policy_validation_strengths_from_store",
        lambda store: object(),
    )
    monkeypatch.setattr(
        module,
        "evaluate_static_profile_gate6",
        lambda reference, validation, **kwargs: _gate6_result(force_failure=True),
    )

    with pytest.raises(
        Gate6CalibrationReportIntegrityError,
        match="different deterministic result",
    ):
        execute_gate6_calibration(
            tmp_path / "snapshots",
            report_path,
        )

    assert load_gate6_calibration_report(report_path) == existing


def test_execute_accepts_identical_existing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pokerbench.evaluation import gate6_execution as module

    snapshots = _snapshots()
    result = _gate6_result()
    expected = Gate6CalibrationReport.from_artifacts(
        result, snapshots, game_config=BASELINE_GAME_CONFIG
    )
    report_path = tmp_path / "gate6.json"
    save_gate6_calibration_report(expected, report_path)

    monkeypatch.setattr(
        module,
        "generate_gate6_calibration_snapshots",
        lambda store, **kwargs: snapshots,
    )
    monkeypatch.setattr(module, "policy_strength_reference_from_store", lambda store: object())
    monkeypatch.setattr(module, "policy_validation_strengths_from_store", lambda store: object())
    monkeypatch.setattr(
        module,
        "evaluate_static_profile_gate6",
        lambda reference, validation, **kwargs: result,
    )

    actual = execute_gate6_calibration(
        tmp_path / "snapshots",
        report_path,
    )

    assert actual == expected
    assert load_gate6_calibration_report(report_path) == expected


def test_execute_requires_baseline_game_config(tmp_path: Path) -> None:
    nonbaseline = replace(BASELINE_GAME_CONFIG, starting_stack_bb=9_999)

    with pytest.raises(ValueError, match="BASELINE_GAME_CONFIG"):
        execute_gate6_calibration(
            tmp_path / "snapshots",
            tmp_path / "gate6.json",
            game_config=nonbaseline,
        )
