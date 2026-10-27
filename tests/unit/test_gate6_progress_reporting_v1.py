"""Tests for observer-only Gate-6 progress reporting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.agents.calibration_environment import (
    CalibrationPhase,
    CalibrationScenario,
)
from pokerbench.evaluation.calibration_snapshots import (
    CalibrationStrengthSnapshotStore,
    create_empty_calibration_snapshot,
)
from pokerbench.evaluation.gate6_progress import (
    GATE6_TOTAL_OPPORTUNITIES,
    Gate6ProgressTracker,
    read_gate6_progress,
    render_gate6_progress,
)


def _save_completed_count(
    store: CalibrationStrengthSnapshotStore,
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    completed: int,
) -> None:
    snapshot = create_empty_calibration_snapshot(phase, scenario)
    strengths = (
        (500_000,) * completed
        + (None,) * (snapshot.opportunity_count - completed)
    )
    store.save(replace(snapshot, strengths=strengths))


def test_empty_store_reports_first_fit_scenario_and_zero_overall(
    tmp_path: Path,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)

    state = read_gate6_progress(store)

    assert state.phase is CalibrationPhase.FIT
    assert state.scenario is CalibrationScenario.OPEN_BTN
    assert state.scenario_completed == 0
    assert state.scenario_total == 5_000
    assert state.overall_completed == 0
    assert state.overall_total == GATE6_TOTAL_OPPORTUNITIES == 50_000
    assert not state.is_complete


def test_progress_sums_all_persisted_snapshots_and_selects_first_incomplete(
    tmp_path: Path,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    _save_completed_count(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
        5_000,
    )
    _save_completed_count(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.THREE_BET_BB,
        275,
    )

    state = read_gate6_progress(store)

    assert state.phase is CalibrationPhase.FIT
    assert state.scenario is CalibrationScenario.THREE_BET_BB
    assert state.scenario_completed == 275
    assert state.overall_completed == 5_275
    assert state.overall_percent == pytest.approx(10.55)


def test_complete_store_reports_no_active_scenario(tmp_path: Path) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    for phase in (CalibrationPhase.FIT, CalibrationPhase.VALIDATION):
        for scenario in CalibrationScenario:
            _save_completed_count(store, phase, scenario, 5_000)

    state = read_gate6_progress(store)

    assert state.is_complete
    assert state.phase is None
    assert state.scenario is None
    assert state.scenario_completed == 0
    assert state.overall_completed == 50_000
    assert state.overall_percent == 100.0


def test_tracker_uses_only_work_completed_during_current_invocation_for_rate(
    tmp_path: Path,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    _save_completed_count(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
        5_000,
    )
    _save_completed_count(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.THREE_BET_BB,
        275,
    )
    initial = read_gate6_progress(store)
    tracker = Gate6ProgressTracker(initial, started_at=100.0)

    _save_completed_count(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.THREE_BET_BB,
        500,
    )
    current = read_gate6_progress(store)
    metrics = tracker.metrics(current, now=300.0)

    assert metrics.completed_this_run == 225
    assert metrics.elapsed_seconds == pytest.approx(200.0)
    assert metrics.throughput_per_second == pytest.approx(1.125)
    assert metrics.remaining_seconds == pytest.approx(44_500 / 1.125)


def test_tracker_never_uses_preexisting_completed_work_as_current_throughput(
    tmp_path: Path,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    _save_completed_count(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
        5_000,
    )
    state = read_gate6_progress(store)
    tracker = Gate6ProgressTracker(state, started_at=10.0)

    metrics = tracker.metrics(state, now=20.0)

    assert metrics.completed_this_run == 0
    assert metrics.throughput_per_second is None
    assert metrics.remaining_seconds is None


def test_renderer_contains_scenario_overall_percentage_throughput_and_remaining(
    tmp_path: Path,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    _save_completed_count(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
        5_000,
    )
    _save_completed_count(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.THREE_BET_BB,
        275,
    )
    initial = read_gate6_progress(store)
    tracker = Gate6ProgressTracker(initial, started_at=0.0)
    _save_completed_count(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.THREE_BET_BB,
        500,
    )
    current = read_gate6_progress(store)

    text = render_gate6_progress(tracker.metrics(current, now=200.0))

    assert "FIT THREE_BET_BB" in text
    assert "scenario 500/5,000 (10.0%)" in text
    assert "overall 5,500/50,000 (11.0%)" in text
    assert "throughput 1.12 opp/s" in text
    assert "elapsed 00:03:20" in text
    assert "remaining ~10:59:" in text


def test_complete_state_reports_zero_remaining_even_on_noop_resume(
    tmp_path: Path,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    for phase in (CalibrationPhase.FIT, CalibrationPhase.VALIDATION):
        for scenario in CalibrationScenario:
            _save_completed_count(store, phase, scenario, 5_000)
    state = read_gate6_progress(store)
    tracker = Gate6ProgressTracker(state, started_at=100.0)

    metrics = tracker.metrics(state, now=110.0)
    text = render_gate6_progress(metrics)

    assert metrics.remaining_seconds == 0.0
    assert "COMPLETE" in text
    assert "overall 50,000/50,000 (100.0%)" in text
    assert "remaining ~00:00:00" in text
