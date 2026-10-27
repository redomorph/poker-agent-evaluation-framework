"""Observer-only progress reporting for long Gate-6 calibration runs."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from pokerbench.agents.calibration_environment import (
    CalibrationPhase,
    CalibrationScenario,
)
from pokerbench.agents.strength_gates import (
    REFERENCE_OPPORTUNITIES_PER_SCENARIO,
)
from pokerbench.evaluation.calibration_snapshots import (
    CalibrationStrengthSnapshotError,
    CalibrationStrengthSnapshotStore,
)


_GATE6_IDENTITIES = tuple(
    (phase, scenario)
    for phase in (CalibrationPhase.FIT, CalibrationPhase.VALIDATION)
    for scenario in CalibrationScenario
)
GATE6_TOTAL_OPPORTUNITIES = (
    len(_GATE6_IDENTITIES) * REFERENCE_OPPORTUNITIES_PER_SCENARIO
)


@dataclass(frozen=True)
class Gate6ProgressState:
    """Persisted progress observed across all ten Gate-6 snapshots."""

    phase: CalibrationPhase | None
    scenario: CalibrationScenario | None
    scenario_completed: int
    scenario_total: int
    overall_completed: int
    overall_total: int = GATE6_TOTAL_OPPORTUNITIES

    @property
    def is_complete(self) -> bool:
        return self.overall_completed == self.overall_total

    @property
    def scenario_percent(self) -> float:
        if self.phase is None or self.scenario is None:
            return 100.0 if self.is_complete else 0.0
        return 100.0 * self.scenario_completed / self.scenario_total

    @property
    def overall_percent(self) -> float:
        return 100.0 * self.overall_completed / self.overall_total


@dataclass(frozen=True)
class Gate6ProgressMetrics:
    """One progress observation augmented with runtime-derived estimates."""

    state: Gate6ProgressState
    completed_this_run: int
    elapsed_seconds: float
    throughput_per_second: float | None
    remaining_seconds: float | None


class Gate6ProgressTracker:
    """Compute throughput/remaining estimates without counting resumed work."""

    def __init__(
        self,
        initial_state: Gate6ProgressState,
        *,
        started_at: float,
    ) -> None:
        if not isinstance(initial_state, Gate6ProgressState):
            raise TypeError("initial_state must be a Gate6ProgressState")
        if not isinstance(started_at, (int, float)) or isinstance(started_at, bool):
            raise TypeError("started_at must be numeric")
        self._initial_completed = initial_state.overall_completed
        self._started_at = float(started_at)

    def metrics(
        self,
        state: Gate6ProgressState,
        *,
        now: float,
    ) -> Gate6ProgressMetrics:
        if not isinstance(state, Gate6ProgressState):
            raise TypeError("state must be a Gate6ProgressState")
        if not isinstance(now, (int, float)) or isinstance(now, bool):
            raise TypeError("now must be numeric")

        elapsed = max(0.0, float(now) - self._started_at)
        completed_this_run = max(
            0,
            state.overall_completed - self._initial_completed,
        )
        throughput: float | None = None
        remaining: float | None = 0.0 if state.is_complete else None
        if elapsed > 0.0 and completed_this_run > 0:
            throughput = completed_this_run / elapsed
            if not state.is_complete:
                remaining_count = state.overall_total - state.overall_completed
                remaining = remaining_count / throughput

        return Gate6ProgressMetrics(
            state=state,
            completed_this_run=completed_this_run,
            elapsed_seconds=elapsed,
            throughput_per_second=throughput,
            remaining_seconds=remaining,
        )


def read_gate6_progress(
    store: CalibrationStrengthSnapshotStore,
) -> Gate6ProgressState:
    """Read the current aggregate Gate-6 progress from atomic snapshots."""
    if not isinstance(store, CalibrationStrengthSnapshotStore):
        raise TypeError("store must be a CalibrationStrengthSnapshotStore")

    completed_by_identity: dict[
        tuple[CalibrationPhase, CalibrationScenario],
        int,
    ] = {}
    for phase, scenario in _GATE6_IDENTITIES:
        completed = 0
        if store.exists(phase, scenario):
            completed = store.load(phase, scenario).completed_count
        completed_by_identity[(phase, scenario)] = completed

    overall_completed = sum(completed_by_identity.values())
    for phase, scenario in _GATE6_IDENTITIES:
        completed = completed_by_identity[(phase, scenario)]
        if completed < REFERENCE_OPPORTUNITIES_PER_SCENARIO:
            return Gate6ProgressState(
                phase=phase,
                scenario=scenario,
                scenario_completed=completed,
                scenario_total=REFERENCE_OPPORTUNITIES_PER_SCENARIO,
                overall_completed=overall_completed,
            )

    return Gate6ProgressState(
        phase=None,
        scenario=None,
        scenario_completed=0,
        scenario_total=REFERENCE_OPPORTUNITIES_PER_SCENARIO,
        overall_completed=overall_completed,
    )


def _format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_gate6_progress(metrics: Gate6ProgressMetrics) -> str:
    """Render one compact human-readable Gate-6 progress line."""
    if not isinstance(metrics, Gate6ProgressMetrics):
        raise TypeError("metrics must be Gate6ProgressMetrics")

    state = metrics.state
    if state.is_complete:
        prefix = "COMPLETE"
        scenario_text = ""
    else:
        if state.phase is None or state.scenario is None:
            raise ValueError("Incomplete progress state must identify a scenario")
        prefix = f"{state.phase.value.upper()} {state.scenario.value}"
        scenario_text = (
            f" | scenario {state.scenario_completed:,}/{state.scenario_total:,} "
            f"({state.scenario_percent:.1f}%)"
        )

    throughput = (
        f"{metrics.throughput_per_second:.2f} opp/s"
        if metrics.throughput_per_second is not None
        else "--"
    )
    remaining = (
        f"~{_format_duration(metrics.remaining_seconds)}"
        if metrics.remaining_seconds is not None
        else "--"
    )
    return (
        f"[Gate-6 progress] {prefix}{scenario_text}"
        f" | overall {state.overall_completed:,}/{state.overall_total:,} "
        f"({state.overall_percent:.1f}%)"
        f" | throughput {throughput}"
        f" | elapsed {_format_duration(metrics.elapsed_seconds)}"
        f" | remaining {remaining}"
    )


class Gate6ProgressMonitor:
    """Poll persisted snapshots and print progress without touching workers."""

    def __init__(
        self,
        snapshot_root: str | Path,
        *,
        interval_seconds: float = 5.0,
        output: TextIO | None = None,
        error_output: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not isinstance(interval_seconds, (int, float))
            or isinstance(interval_seconds, bool)
        ):
            raise TypeError("interval_seconds must be numeric")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._store = CalibrationStrengthSnapshotStore(snapshot_root)
        self._interval_seconds = float(interval_seconds)
        self._output = output if output is not None else sys.stdout
        self._error_output = error_output if error_output is not None else sys.stderr
        self._clock = clock
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracker: Gate6ProgressTracker | None = None
        self._last_state: Gate6ProgressState | None = None
        self._warning_emitted = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Gate6ProgressMonitor is already started")
        initial = read_gate6_progress(self._store)
        now = self._clock()
        self._tracker = Gate6ProgressTracker(initial, started_at=now)
        self._emit_if_changed(initial, now=now, force=True)
        self._thread = threading.Thread(
            target=self._run,
            name="gate6-progress-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join()
        try:
            self._emit_if_changed(
                read_gate6_progress(self._store),
                now=self._clock(),
            )
        except (CalibrationStrengthSnapshotError, OSError) as exc:
            self._warn(exc)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                state = read_gate6_progress(self._store)
                self._emit_if_changed(state, now=self._clock())
            except (CalibrationStrengthSnapshotError, OSError) as exc:
                self._warn(exc)

    def _emit_if_changed(
        self,
        state: Gate6ProgressState,
        *,
        now: float,
        force: bool = False,
    ) -> None:
        if not force and state == self._last_state:
            return
        tracker = self._tracker
        if tracker is None:
            raise RuntimeError("Gate6ProgressMonitor has not been started")
        print(
            render_gate6_progress(tracker.metrics(state, now=now)),
            file=self._output,
            flush=True,
        )
        self._last_state = state

    def _warn(self, exc: BaseException) -> None:
        if self._warning_emitted:
            return
        print(
            f"[Gate-6 progress] snapshot polling unavailable: {exc}",
            file=self._error_output,
            flush=True,
        )
        self._warning_emitted = True
