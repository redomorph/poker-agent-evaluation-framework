"""Deterministic persisted strength snapshots for Gate-6 calibration."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pokerbench.agents.calibration_environment import (
    CALIBRATION_ENVIRONMENT_VERSION,
    FIT_BASE_SEED,
    VALIDATION_BASE_SEED,
    CalibrationPhase,
    CalibrationScenario,
    calibration_probe_observation,
    scenario_deal_seed,
)
from pokerbench.agents.equity import (
    EQUITY_RANGE_VERSION,
    EQUITY_VERSION,
    estimate_uniform_random_equity,
)
from pokerbench.agents.strength_gates import (
    PPM_SCALE,
    REFERENCE_OPPORTUNITIES_PER_SCENARIO,
    PolicyStrengthReference,
)
from pokerbench.engine.config import (
    BASELINE_GAME_CONFIG,
    GameConfig,
    config_hash,
)

if TYPE_CHECKING:
    from pokerbench.evaluation.policy_acceptance import (
        PolicyValidationStrengths,
    )


CALIBRATION_STRENGTH_SNAPSHOT_VERSION = (
    "calibration-strength-snapshot-v1"
)


class CalibrationStrengthSnapshotError(ValueError):
    """Base error for persisted calibration-strength snapshots."""


class CalibrationStrengthSnapshotIntegrityError(
    CalibrationStrengthSnapshotError
):
    """Raised when persisted snapshot content fails integrity validation."""


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _validate_phase(phase: CalibrationPhase) -> None:
    if not isinstance(phase, CalibrationPhase):
        raise TypeError("phase must be a CalibrationPhase")


def _validate_scenario(scenario: CalibrationScenario) -> None:
    if not isinstance(scenario, CalibrationScenario):
        raise TypeError("scenario must be a CalibrationScenario")


def _validate_game_config(game_config: GameConfig) -> None:
    if not isinstance(game_config, GameConfig):
        raise TypeError("game_config must be a GameConfig")
    if game_config != BASELINE_GAME_CONFIG:
        raise ValueError(
            "calibration-strength-snapshot-v1 requires BASELINE_GAME_CONFIG"
        )


def _base_seed(phase: CalibrationPhase) -> int:
    _validate_phase(phase)
    if phase is CalibrationPhase.FIT:
        return FIT_BASE_SEED
    return VALIDATION_BASE_SEED


def _validate_strength(value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("snapshot strengths must be integers or None")
    if not 0 <= value <= PPM_SCALE:
        raise ValueError("snapshot strengths must be ppm values")


@dataclass(frozen=True)
class CalibrationScenarioStrengthSnapshot:
    """Immutable partial or complete strength snapshot for one scenario."""

    phase: CalibrationPhase
    scenario: CalibrationScenario
    strengths: tuple[int | None, ...]
    base_seed: int
    deal_seed: int
    game_config_hash: str
    snapshot_version: str = CALIBRATION_STRENGTH_SNAPSHOT_VERSION
    calibration_environment_version: str = CALIBRATION_ENVIRONMENT_VERSION
    equity_version: str = EQUITY_VERSION
    equity_range_version: str = EQUITY_RANGE_VERSION
    opportunity_count: int = REFERENCE_OPPORTUNITIES_PER_SCENARIO

    def __post_init__(self) -> None:
        _validate_phase(self.phase)
        _validate_scenario(self.scenario)

        if self.snapshot_version != CALIBRATION_STRENGTH_SNAPSHOT_VERSION:
            raise ValueError("Unsupported calibration-strength snapshot version")
        if self.calibration_environment_version != CALIBRATION_ENVIRONMENT_VERSION:
            raise ValueError("Unsupported calibration-environment version")
        if self.equity_version != EQUITY_VERSION:
            raise ValueError("Unsupported equity version")
        if self.equity_range_version != EQUITY_RANGE_VERSION:
            raise ValueError("Unsupported equity-range version")
        if self.opportunity_count != REFERENCE_OPPORTUNITIES_PER_SCENARIO:
            raise ValueError("Unsupported snapshot opportunity count")
        if self.base_seed != _base_seed(self.phase):
            raise ValueError("Snapshot base seed does not match phase")
        if self.deal_seed != scenario_deal_seed(self.phase, self.scenario):
            raise ValueError("Snapshot deal seed does not match phase/scenario")
        if self.game_config_hash != config_hash(BASELINE_GAME_CONFIG):
            raise ValueError("Snapshot game-config hash is not baseline")
        if not isinstance(self.strengths, tuple):
            raise TypeError("strengths must be a tuple")
        if len(self.strengths) != self.opportunity_count:
            raise ValueError(
                f"strengths must contain exactly {self.opportunity_count} slots"
            )
        for value in self.strengths:
            _validate_strength(value)

    @property
    def completed_count(self) -> int:
        return sum(value is not None for value in self.strengths)

    @property
    def is_complete(self) -> bool:
        return self.completed_count == self.opportunity_count

    def complete_strengths(self) -> tuple[int, ...]:
        if not self.is_complete:
            raise ValueError(
                "Calibration strength snapshot is incomplete: "
                f"phase={self.phase.value}, scenario={self.scenario.value}, "
                f"completed={self.completed_count}/{self.opportunity_count}"
            )
        return tuple(
            value
            for value in self.strengths
            if value is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "calibration_environment_version": (
                self.calibration_environment_version
            ),
            "equity_version": self.equity_version,
            "equity_range_version": self.equity_range_version,
            "phase": self.phase.value,
            "scenario": self.scenario.value,
            "base_seed": self.base_seed,
            "deal_seed": self.deal_seed,
            "game_config_hash": self.game_config_hash,
            "opportunity_count": self.opportunity_count,
            "strengths": list(self.strengths),
        }

    @property
    def snapshot_hash(self) -> str:
        return _sha256_payload(self.to_dict())


def create_empty_calibration_snapshot(
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    *,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> CalibrationScenarioStrengthSnapshot:
    _validate_phase(phase)
    _validate_scenario(scenario)
    _validate_game_config(game_config)

    return CalibrationScenarioStrengthSnapshot(
        phase=phase,
        scenario=scenario,
        strengths=(None,) * REFERENCE_OPPORTUNITIES_PER_SCENARIO,
        base_seed=_base_seed(phase),
        deal_seed=scenario_deal_seed(phase, scenario),
        game_config_hash=config_hash(game_config),
    )


class CalibrationStrengthSnapshotStore:
    """Atomic JSON persistence for scenario-level calibration checkpoints."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(
        self,
        phase: CalibrationPhase,
        scenario: CalibrationScenario,
    ) -> Path:
        _validate_phase(phase)
        _validate_scenario(scenario)
        return self._root / phase.value / f"{scenario.value}.json"

    def exists(
        self,
        phase: CalibrationPhase,
        scenario: CalibrationScenario,
    ) -> bool:
        return self.path_for(phase, scenario).is_file()

    def save(
        self,
        snapshot: CalibrationScenarioStrengthSnapshot,
    ) -> Path:
        if not isinstance(snapshot, CalibrationScenarioStrengthSnapshot):
            raise TypeError(
                "snapshot must be a CalibrationScenarioStrengthSnapshot"
            )

        path = self.path_for(snapshot.phase, snapshot.scenario)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "snapshot": snapshot.to_dict(),
            "snapshot_hash": snapshot.snapshot_hash,
        }
        serialized = _canonical_json(payload) + "\n"

        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

        return path

    def load(
        self,
        phase: CalibrationPhase,
        scenario: CalibrationScenario,
    ) -> CalibrationScenarioStrengthSnapshot:
        path = self.path_for(phase, scenario)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CalibrationStrengthSnapshotIntegrityError(
                f"Invalid calibration snapshot JSON: {path}"
            ) from exc

        if not isinstance(payload, dict):
            raise CalibrationStrengthSnapshotIntegrityError(
                "Calibration snapshot file must contain a JSON object"
            )
        if set(payload) != {"snapshot", "snapshot_hash"}:
            raise CalibrationStrengthSnapshotIntegrityError(
                "Calibration snapshot file has unexpected fields"
            )
        raw_snapshot = payload["snapshot"]
        stored_hash = payload["snapshot_hash"]
        if not isinstance(raw_snapshot, dict) or not isinstance(stored_hash, str):
            raise CalibrationStrengthSnapshotIntegrityError(
                "Calibration snapshot payload has invalid structure"
            )
        actual_hash = _sha256_payload(raw_snapshot)
        if stored_hash != actual_hash:
            raise CalibrationStrengthSnapshotIntegrityError(
                "Calibration snapshot SHA-256 mismatch"
            )

        try:
            snapshot = CalibrationScenarioStrengthSnapshot(
                phase=CalibrationPhase(raw_snapshot["phase"]),
                scenario=CalibrationScenario(raw_snapshot["scenario"]),
                strengths=tuple(raw_snapshot["strengths"]),
                base_seed=raw_snapshot["base_seed"],
                deal_seed=raw_snapshot["deal_seed"],
                game_config_hash=raw_snapshot["game_config_hash"],
                snapshot_version=raw_snapshot["snapshot_version"],
                calibration_environment_version=(
                    raw_snapshot["calibration_environment_version"]
                ),
                equity_version=raw_snapshot["equity_version"],
                equity_range_version=raw_snapshot["equity_range_version"],
                opportunity_count=raw_snapshot["opportunity_count"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationStrengthSnapshotIntegrityError(
                "Calibration snapshot payload is invalid"
            ) from exc

        if snapshot.snapshot_hash != stored_hash:
            raise CalibrationStrengthSnapshotIntegrityError(
                "Calibration snapshot canonical hash mismatch"
            )
        return snapshot


@dataclass(frozen=True)
class _StrengthTask:
    phase: CalibrationPhase
    scenario: CalibrationScenario
    opportunity_index: int
    game_config: GameConfig


def _compute_strength_task(task: _StrengthTask) -> tuple[int, int]:
    probe = calibration_probe_observation(
        task.phase,
        task.scenario,
        task.opportunity_index,
        game_config=task.game_config,
    )
    result = estimate_uniform_random_equity(
        probe.observation,
        master_seed=probe.deal_seed,
        bot_id=probe.hero_seat,
        hand_id=probe.opportunity_index,
        decision_index=probe.decision_index,
    )
    return task.opportunity_index, result.equity_ppm


def _validate_generation_parameters(
    *,
    workers: int,
    checkpoint_every: int,
) -> None:
    if not isinstance(workers, int) or isinstance(workers, bool):
        raise TypeError("workers must be an integer")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if (
        not isinstance(checkpoint_every, int)
        or isinstance(checkpoint_every, bool)
    ):
        raise TypeError("checkpoint_every must be an integer")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")


def _validate_snapshot_identity(
    snapshot: CalibrationScenarioStrengthSnapshot,
    *,
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    game_config: GameConfig,
) -> None:
    expected = create_empty_calibration_snapshot(
        phase,
        scenario,
        game_config=game_config,
    )
    identity_fields = (
        "phase",
        "scenario",
        "base_seed",
        "deal_seed",
        "game_config_hash",
        "snapshot_version",
        "calibration_environment_version",
        "equity_version",
        "equity_range_version",
        "opportunity_count",
    )
    if any(
        getattr(snapshot, field_name) != getattr(expected, field_name)
        for field_name in identity_fields
    ):
        raise ValueError(
            "Persisted calibration snapshot identity does not match "
            "the requested generation job"
        )


def _snapshot_with_values(
    snapshot: CalibrationScenarioStrengthSnapshot,
    values: list[int | None],
) -> CalibrationScenarioStrengthSnapshot:
    return replace(snapshot, strengths=tuple(values))


def generate_calibration_scenario_snapshot(
    store: CalibrationStrengthSnapshotStore,
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
    *,
    workers: int = 1,
    checkpoint_every: int = 25,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> CalibrationScenarioStrengthSnapshot:
    """Generate or resume one deterministic scenario strength snapshot."""
    if not isinstance(store, CalibrationStrengthSnapshotStore):
        raise TypeError("store must be a CalibrationStrengthSnapshotStore")
    _validate_phase(phase)
    _validate_scenario(scenario)
    _validate_game_config(game_config)
    _validate_generation_parameters(
        workers=workers,
        checkpoint_every=checkpoint_every,
    )

    if store.exists(phase, scenario):
        snapshot = store.load(phase, scenario)
        _validate_snapshot_identity(
            snapshot,
            phase=phase,
            scenario=scenario,
            game_config=game_config,
        )
    else:
        snapshot = create_empty_calibration_snapshot(
            phase,
            scenario,
            game_config=game_config,
        )
        store.save(snapshot)

    if snapshot.is_complete:
        return snapshot

    values = list(snapshot.strengths)
    missing = [
        index
        for index, value in enumerate(values)
        if value is None
    ]
    missing_set = set(missing)
    completed_since_checkpoint = 0

    def persist() -> CalibrationScenarioStrengthSnapshot:
        current = _snapshot_with_values(snapshot, values)
        store.save(current)
        return current

    def record(result: tuple[int, int]) -> None:
        nonlocal completed_since_checkpoint
        index, equity_ppm = result
        if index not in missing_set:
            raise RuntimeError(
                f"Worker returned unexpected opportunity index: {index}"
            )
        _validate_strength(equity_ppm)
        if equity_ppm is None:
            raise RuntimeError("Worker returned an empty equity strength")
        if values[index] is not None:
            raise RuntimeError(
                f"Worker returned duplicate opportunity index: {index}"
            )
        values[index] = equity_ppm
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= checkpoint_every:
            persist()
            completed_since_checkpoint = 0

    tasks = (
        _StrengthTask(
            phase=phase,
            scenario=scenario,
            opportunity_index=index,
            game_config=game_config,
        )
        for index in missing
    )

    try:
        if workers == 1:
            for task in tasks:
                record(_compute_strength_task(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_compute_strength_task, task)
                    for task in tasks
                ]
                for future in as_completed(futures):
                    record(future.result())
    except BaseException:
        persist()
        raise

    final_snapshot = persist()
    if not final_snapshot.is_complete:
        raise RuntimeError(
            "Calibration scenario generation ended with an incomplete snapshot"
        )
    return final_snapshot


def generate_gate6_calibration_snapshots(
    store: CalibrationStrengthSnapshotStore,
    *,
    workers: int = 1,
    checkpoint_every: int = 25,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> tuple[CalibrationScenarioStrengthSnapshot, ...]:
    """Generate or resume all five fit and five validation snapshots."""
    snapshots: list[CalibrationScenarioStrengthSnapshot] = []
    for phase in (CalibrationPhase.FIT, CalibrationPhase.VALIDATION):
        for scenario in CalibrationScenario:
            snapshots.append(
                generate_calibration_scenario_snapshot(
                    store,
                    phase,
                    scenario,
                    workers=workers,
                    checkpoint_every=checkpoint_every,
                    game_config=game_config,
                )
            )
    return tuple(snapshots)


def _complete_for_scenario(
    store: CalibrationStrengthSnapshotStore,
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
) -> tuple[int, ...]:
    snapshot = store.load(phase, scenario)
    _validate_snapshot_identity(
        snapshot,
        phase=phase,
        scenario=scenario,
        game_config=BASELINE_GAME_CONFIG,
    )
    try:
        return snapshot.complete_strengths()
    except ValueError as exc:
        raise ValueError(
            "Cannot build calibration artifact from incomplete snapshot: "
            f"phase={phase.value}, scenario={scenario.value}"
        ) from exc


def policy_strength_reference_from_store(
    store: CalibrationStrengthSnapshotStore,
) -> PolicyStrengthReference:
    return PolicyStrengthReference(
        open_btn=_complete_for_scenario(
            store,
            CalibrationPhase.FIT,
            CalibrationScenario.OPEN_BTN,
        ),
        three_bet_bb=_complete_for_scenario(
            store,
            CalibrationPhase.FIT,
            CalibrationScenario.THREE_BET_BB,
        ),
        cbet_btn=_complete_for_scenario(
            store,
            CalibrationPhase.FIT,
            CalibrationScenario.CBET_BTN,
        ),
        fold_to_cbet_bb=_complete_for_scenario(
            store,
            CalibrationPhase.FIT,
            CalibrationScenario.FOLD_TO_CBET_BB,
        ),
        postflop_aggression_btn=_complete_for_scenario(
            store,
            CalibrationPhase.FIT,
            CalibrationScenario.POSTFLOP_AGGRESSION_BTN,
        ),
    )


def policy_validation_strengths_from_store(
    store: CalibrationStrengthSnapshotStore,
) -> "PolicyValidationStrengths":
    from pokerbench.evaluation.policy_acceptance import PolicyValidationStrengths

    return PolicyValidationStrengths(
        open_btn=_complete_for_scenario(
            store,
            CalibrationPhase.VALIDATION,
            CalibrationScenario.OPEN_BTN,
        ),
        three_bet_bb=_complete_for_scenario(
            store,
            CalibrationPhase.VALIDATION,
            CalibrationScenario.THREE_BET_BB,
        ),
        cbet_btn=_complete_for_scenario(
            store,
            CalibrationPhase.VALIDATION,
            CalibrationScenario.CBET_BTN,
        ),
        fold_to_cbet_bb=_complete_for_scenario(
            store,
            CalibrationPhase.VALIDATION,
            CalibrationScenario.FOLD_TO_CBET_BB,
        ),
        postflop_aggression_btn=_complete_for_scenario(
            store,
            CalibrationPhase.VALIDATION,
            CalibrationScenario.POSTFLOP_AGGRESSION_BTN,
        ),
    )
