from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.agents.calibration_environment import (
    FIT_BASE_SEED,
    VALIDATION_BASE_SEED,
    CalibrationPhase,
    CalibrationScenario,
    calibration_probe_observation,
    scenario_deal_seed,
)
from pokerbench.agents.equity import EQUITY_RANGE_VERSION, EQUITY_VERSION
from pokerbench.agents.strength_gates import (
    REFERENCE_OPPORTUNITIES_PER_SCENARIO,
    STRENGTH_REFERENCE_VERSION,
)
from pokerbench.engine.config import BASELINE_GAME_CONFIG, config_hash
from pokerbench.engine.deal_schedule import DealSchedule
from pokerbench.evaluation.calibration_snapshots import (
    CALIBRATION_STRENGTH_SNAPSHOT_VERSION,
    CalibrationScenarioStrengthSnapshot,
    CalibrationStrengthSnapshotIntegrityError,
    CalibrationStrengthSnapshotStore,
    create_empty_calibration_snapshot,
    generate_calibration_scenario_snapshot,
    policy_strength_reference_from_store,
    policy_validation_strengths_from_store,
)


def _complete_strengths(value: int) -> tuple[int, ...]:
    return (value,) * REFERENCE_OPPORTUNITIES_PER_SCENARIO


def _complete_snapshot(
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    value: int,
) -> CalibrationScenarioStrengthSnapshot:
    empty = create_empty_calibration_snapshot(phase, scenario)
    return replace(empty, strengths=_complete_strengths(value))


def test_direct_indexed_deal_matches_schedule() -> None:
    seed = 98_765_432_101
    schedule = DealSchedule(seed, 64)

    for index in (0, 1, 2, 7, 31, 63):
        assert DealSchedule.hand_for_index(seed, index) == schedule.get_hand(index)


def test_calibration_probe_uses_exact_same_indexed_deal() -> None:
    phase = CalibrationPhase.FIT
    scenario = CalibrationScenario.CBET_BTN
    index = 137
    probe = calibration_probe_observation(phase, scenario, index)
    seed = scenario_deal_seed(phase, scenario)
    hand = DealSchedule.hand_for_index(seed, index)

    # Hero is seat 0 in CBET_BTN; only visible cards may be compared here.
    assert probe.observation.hole_cards == hand[0]
    assert probe.observation.board == hand[2][:3]


def test_empty_snapshot_freezes_identity_and_5000_slots() -> None:
    snapshot = create_empty_calibration_snapshot(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
    )

    assert snapshot.snapshot_version == CALIBRATION_STRENGTH_SNAPSHOT_VERSION
    assert snapshot.equity_version == EQUITY_VERSION
    assert snapshot.equity_range_version == EQUITY_RANGE_VERSION
    assert snapshot.base_seed == FIT_BASE_SEED
    assert snapshot.deal_seed == scenario_deal_seed(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
    )
    assert snapshot.game_config_hash == config_hash(BASELINE_GAME_CONFIG)
    assert snapshot.completed_count == 0
    assert not snapshot.is_complete
    assert len(snapshot.strengths) == 5_000
    assert all(value is None for value in snapshot.strengths)


def test_snapshot_round_trip_and_integrity_hash(tmp_path: Path) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    snapshot = _complete_snapshot(
        CalibrationPhase.VALIDATION,
        CalibrationScenario.THREE_BET_BB,
        432_100,
    )

    path = store.save(snapshot)
    loaded = store.load(
        CalibrationPhase.VALIDATION,
        CalibrationScenario.THREE_BET_BB,
    )

    assert loaded == snapshot
    assert loaded.snapshot_hash == snapshot.snapshot_hash
    assert path == store.path_for(
        CalibrationPhase.VALIDATION,
        CalibrationScenario.THREE_BET_BB,
    )
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_snapshot_tampering_is_rejected(tmp_path: Path) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    snapshot = _complete_snapshot(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
        500_000,
    )
    path = store.save(snapshot)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["snapshot"]["strengths"][0] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CalibrationStrengthSnapshotIntegrityError):
        store.load(CalibrationPhase.FIT, CalibrationScenario.OPEN_BTN)


def test_incomplete_snapshot_cannot_be_consumed_as_complete() -> None:
    snapshot = create_empty_calibration_snapshot(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
    )

    with pytest.raises(ValueError):
        snapshot.complete_strengths()


def test_resume_computes_only_missing_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    empty = create_empty_calibration_snapshot(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
    )
    strengths = list(empty.strengths)
    strengths[0] = 100
    strengths[1] = 101
    partial = replace(empty, strengths=tuple(strengths))
    store.save(partial)

    called: list[int] = []

    def fake_compute(task):
        called.append(task.opportunity_index)
        return task.opportunity_index, 200_000 + task.opportunity_index

    monkeypatch.setattr(
        "pokerbench.evaluation.calibration_snapshots._compute_strength_task",
        fake_compute,
    )

    result = generate_calibration_scenario_snapshot(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
        workers=1,
        checkpoint_every=137,
    )

    assert result.is_complete
    assert 0 not in called
    assert 1 not in called
    assert called[0] == 2
    assert called[-1] == 4_999
    assert result.strengths[0] == 100
    assert result.strengths[1] == 101
    assert result.strengths[2] == 200_002


def test_complete_snapshot_is_reused_without_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    snapshot = _complete_snapshot(
        CalibrationPhase.VALIDATION,
        CalibrationScenario.CBET_BTN,
        333_333,
    )
    store.save(snapshot)

    def fail_compute(task):
        raise AssertionError("complete snapshot must not recompute")

    monkeypatch.setattr(
        "pokerbench.evaluation.calibration_snapshots._compute_strength_task",
        fail_compute,
    )

    result = generate_calibration_scenario_snapshot(
        store,
        CalibrationPhase.VALIDATION,
        CalibrationScenario.CBET_BTN,
        workers=4,
    )

    assert result == snapshot


def test_parallel_completion_order_does_not_change_canonical_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)

    # Keep only six missing slots so the test exercises the parallel branch
    # without doing real equity work.
    empty = create_empty_calibration_snapshot(
        CalibrationPhase.FIT,
        CalibrationScenario.THREE_BET_BB,
    )
    strengths = [400_000] * 5_000
    for index in range(6):
        strengths[index] = None
    store.save(replace(empty, strengths=tuple(strengths)))

    def fake_compute(task):
        # Reverse sleep makes completion order intentionally differ from index order.
        time.sleep((5 - task.opportunity_index) * 0.001)
        return task.opportunity_index, 600_000 + task.opportunity_index

    monkeypatch.setattr(
        "pokerbench.evaluation.calibration_snapshots._compute_strength_task",
        fake_compute,
    )
    monkeypatch.setattr(
        "pokerbench.evaluation.calibration_snapshots.ProcessPoolExecutor",
        ThreadPoolExecutor,
    )

    result = generate_calibration_scenario_snapshot(
        store,
        CalibrationPhase.FIT,
        CalibrationScenario.THREE_BET_BB,
        workers=3,
        checkpoint_every=2,
    )

    assert result.strengths[:6] == tuple(600_000 + index for index in range(6))
    assert result.strengths[6] == 400_000


def test_failure_persists_completed_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)

    def fake_compute(task):
        if task.opportunity_index == 3:
            raise RuntimeError("synthetic worker failure")
        return task.opportunity_index, 700_000 + task.opportunity_index

    monkeypatch.setattr(
        "pokerbench.evaluation.calibration_snapshots._compute_strength_task",
        fake_compute,
    )

    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        generate_calibration_scenario_snapshot(
            store,
            CalibrationPhase.FIT,
            CalibrationScenario.CBET_BTN,
            workers=1,
            checkpoint_every=50,
        )

    persisted = store.load(
        CalibrationPhase.FIT,
        CalibrationScenario.CBET_BTN,
    )
    assert persisted.strengths[0] == 700_000
    assert persisted.strengths[1] == 700_001
    assert persisted.strengths[2] == 700_002
    assert persisted.strengths[3] is None
    assert persisted.completed_count == 3


def test_snapshot_at_wrong_identity_is_not_silently_reused(
    tmp_path: Path,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    wrong = _complete_snapshot(
        CalibrationPhase.VALIDATION,
        CalibrationScenario.OPEN_BTN,
        500_000,
    )
    target_path = store.path_for(
        CalibrationPhase.FIT,
        CalibrationScenario.OPEN_BTN,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Save a valid validation snapshot, then move its complete persisted payload
    # under the fit filename. Integrity remains valid; identity must still reject it.
    source = store.save(wrong)
    target_path.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="identity"):
        generate_calibration_scenario_snapshot(
            store,
            CalibrationPhase.FIT,
            CalibrationScenario.OPEN_BTN,
            workers=1,
        )


def test_reference_and_validation_objects_are_built_only_from_complete_snapshots(
    tmp_path: Path,
) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)
    values = {
        CalibrationScenario.OPEN_BTN: 100_000,
        CalibrationScenario.THREE_BET_BB: 200_000,
        CalibrationScenario.CBET_BTN: 300_000,
        CalibrationScenario.FOLD_TO_CBET_BB: 400_000,
        CalibrationScenario.POSTFLOP_AGGRESSION_BTN: 500_000,
    }

    for phase in CalibrationPhase:
        for scenario, value in values.items():
            store.save(_complete_snapshot(phase, scenario, value))

    reference = policy_strength_reference_from_store(store)
    validation = policy_validation_strengths_from_store(store)

    assert reference.version == STRENGTH_REFERENCE_VERSION
    assert reference.open_btn[0] == 100_000
    assert reference.fold_to_cbet_bb[0] == 400_000
    assert validation.open_btn[0] == 100_000
    assert validation.postflop_aggression_btn[0] == 500_000
    assert validation.validation_base_seed == VALIDATION_BASE_SEED


def test_reference_builder_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)

    for scenario in CalibrationScenario:
        snapshot = _complete_snapshot(
            CalibrationPhase.FIT,
            scenario,
            500_000,
        )
        if scenario is CalibrationScenario.CBET_BTN:
            values = list(snapshot.strengths)
            values[-1] = None
            snapshot = replace(snapshot, strengths=tuple(values))
        store.save(snapshot)

    with pytest.raises(ValueError, match="incomplete"):
        policy_strength_reference_from_store(store)


def test_worker_count_and_checkpoint_interval_are_validated(tmp_path: Path) -> None:
    store = CalibrationStrengthSnapshotStore(tmp_path)

    with pytest.raises(ValueError):
        generate_calibration_scenario_snapshot(
            store,
            CalibrationPhase.FIT,
            CalibrationScenario.OPEN_BTN,
            workers=0,
        )

    with pytest.raises(ValueError):
        generate_calibration_scenario_snapshot(
            store,
            CalibrationPhase.FIT,
            CalibrationScenario.OPEN_BTN,
            checkpoint_every=0,
        )


def test_gate6_snapshot_orchestration_visits_all_ten_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pokerbench.evaluation import calibration_snapshots as module

    store = CalibrationStrengthSnapshotStore(tmp_path)
    calls: list[tuple[CalibrationPhase, CalibrationScenario]] = []

    def fake_generate(store_arg, phase, scenario, **kwargs):
        assert store_arg is store
        calls.append((phase, scenario))
        return _complete_snapshot(phase, scenario, 500_000)

    monkeypatch.setattr(
        module,
        "generate_calibration_scenario_snapshot",
        fake_generate,
    )

    result = module.generate_gate6_calibration_snapshots(
        store,
        workers=3,
        checkpoint_every=17,
    )

    assert len(result) == 10
    assert calls == [
        (phase, scenario)
        for phase in (CalibrationPhase.FIT, CalibrationPhase.VALIDATION)
        for scenario in CalibrationScenario
    ]
