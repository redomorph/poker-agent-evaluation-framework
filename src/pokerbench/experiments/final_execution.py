"""Bound, resumable execution of the frozen Phase-1 final baseline experiment."""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pokerbench.agents.base import Agent
from pokerbench.agents.hand_strength_baseline import HandStrengthBaselineAgent
from pokerbench.agents.random_baseline import RandomBaselineAgent
from pokerbench.cli.run import (
    VerifiedStaticProfileContext,
    build_verified_static_profile_agent,
    load_verified_static_profile_context,
)
from pokerbench.engine.config import BASELINE_GAME_CONFIG
from pokerbench.evaluation.population import (
    DEFAULT_POPULATION_BOOTSTRAP_RESAMPLES,
    DiscretePopulationDefinition,
    PopulationEvaluation,
    PopulationProfileEvidenceSummary,
    ProfilePerformanceEvidence,
    evaluate_population,
)
from pokerbench.experiments.config import ExecutionEnvironment, capture_execution_environment
from pokerbench.experiments.final_protocol import (
    PHASE1_FINAL_Z,
    FinalJob,
    Phase1BaselineExperimentProtocol,
    iter_final_jobs,
    load_phase1_baseline_protocol,
    verify_pilot_report_binding,
)
from pokerbench.experiments.frozen_protocol import canonical_file_bytes, load_frozen_population
from pokerbench.experiments.pilot_execution import (
    _profile_parameters,
    _worker_verified_context,
    hardware_fingerprint,
    sha256_canonical,
)
from pokerbench.persistence import SQLiteBenchmarkStore
from pokerbench.persistence.sqlite import MatchPersistenceMetadata
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import IllegalActionPolicy, PairedRunner, StatisticalUnit


PHASE1_FINAL_EXECUTION_BINDING_VERSION = "phase1-final-execution-binding-v1"
PHASE1_FINAL_JOB_SNAPSHOT_VERSION = "phase1-final-job-snapshot-v1"
PHASE1_FINAL_EXPERIMENT_CONFIG_VERSION = "phase1-final-job-config-v1"
PHASE1_FINAL_EXECUTION_REPORT_VERSION = "phase1-final-execution-report-v1"
PHASE1_FINAL_INTERNAL_CLI_VERSION = "phase1-final-execution-cli-v1"
PHASE1_FINAL_PRECISION_DIAGNOSTIC_VERSION = "fixed-profile-normal-mean-ci-v1"


class FinalExecutionError(ValueError):
    """Raised when frozen final execution evidence is inconsistent."""


def _validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FinalExecutionError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FinalExecutionError(f"{field_name} must be a SHA-256 hex digest") from exc
    return value.lower()


def _atomic_write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        payload = canonical_file_bytes(value)
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _clean_git_commit(repo_root: Path) -> str:
    scientific_paths = ["src", "configs", "docs/specs", "pyproject.toml", "poetry.lock"]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *scientific_paths],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise FinalExecutionError(
            "Scientific source/configuration paths must be clean and committed before final binding"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not commit:
        raise FinalExecutionError("Could not resolve git commit")
    return commit


@dataclass(frozen=True)
class FinalExecutionBinding:
    protocol_id: str
    protocol_version: str
    protocol_hash: str
    pilot_report_hash: str
    primary_population_id: str
    primary_population_version: str
    primary_population_hash: str
    secondary_population_id: str
    secondary_population_version: str
    secondary_population_hash: str
    gate6_report_hash: str
    gate6_reference_strength_hash: str
    git_commit: str
    engine_version: str
    python_version: str
    dependency_lockfile: str
    poetry_lock_hash: str
    cpu_model: str
    cpu_core_count: int
    cpu_architecture: str
    ram_bytes: int
    os_identifier: str
    hardware_fingerprint: str
    workers: int
    binding_version: str = PHASE1_FINAL_EXECUTION_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.binding_version != PHASE1_FINAL_EXECUTION_BINDING_VERSION:
            raise FinalExecutionError("Unsupported final execution binding version")
        for name in (
            "protocol_hash", "pilot_report_hash", "primary_population_hash",
            "secondary_population_hash", "gate6_report_hash",
            "gate6_reference_strength_hash", "poetry_lock_hash", "hardware_fingerprint",
        ):
            _validate_sha256(str(getattr(self, name)), name)
        if not self.git_commit or self.engine_version != f"git:{self.git_commit}":
            raise FinalExecutionError("engine_version must be derived from non-empty git_commit")
        for name in ("cpu_core_count", "ram_bytes", "workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FinalExecutionError(f"{name} must be a positive integer")

    def payload(self) -> dict[str, Any]:
        return {
            "binding_version": self.binding_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "pilot_report_hash": self.pilot_report_hash,
            "primary_population": {
                "population_id": self.primary_population_id,
                "version": self.primary_population_version,
                "population_hash": self.primary_population_hash,
            },
            "secondary_population": {
                "population_id": self.secondary_population_id,
                "version": self.secondary_population_version,
                "population_hash": self.secondary_population_hash,
            },
            "gate6_report_hash": self.gate6_report_hash,
            "gate6_reference_strength_hash": self.gate6_reference_strength_hash,
            "git_commit": self.git_commit,
            "engine_version": self.engine_version,
            "python_version": self.python_version,
            "dependency_lockfile": self.dependency_lockfile,
            "poetry_lock_hash": self.poetry_lock_hash,
            "cpu_model": self.cpu_model,
            "cpu_core_count": self.cpu_core_count,
            "cpu_architecture": self.cpu_architecture,
            "ram_bytes": self.ram_bytes,
            "os_identifier": self.os_identifier,
            "hardware_fingerprint": self.hardware_fingerprint,
            "workers": self.workers,
        }

    @property
    def binding_hash(self) -> str:
        return sha256_canonical(self.payload())

    def reference_payload(self) -> dict[str, Any]:
        return {**self.payload(), "binding_hash": self.binding_hash}

    def to_dict(self) -> dict[str, Any]:
        return {"binding": self.payload(), "binding_hash": self.binding_hash}


def build_final_execution_binding(
    *, protocol: Phase1BaselineExperimentProtocol,
    primary_population: DiscretePopulationDefinition,
    secondary_population: DiscretePopulationDefinition,
    gate6_report_hash: str,
    gate6_reference_strength_hash: str,
    environment: ExecutionEnvironment,
    workers: int,
) -> FinalExecutionBinding:
    if protocol.primary_population_id != primary_population.population_id:
        raise FinalExecutionError("Primary population identity does not match final protocol")
    if protocol.primary_population_hash != primary_population.population_hash:
        raise FinalExecutionError("Primary population hash does not match final protocol")
    if secondary_population.population_id not in protocol.secondary_population_ids:
        raise FinalExecutionError("Secondary population identity does not match final protocol")
    if gate6_report_hash != protocol.gate6_report_hash:
        raise FinalExecutionError("Gate-6 report hash does not match final protocol")
    if workers <= 0:
        raise FinalExecutionError("workers must be positive")
    return FinalExecutionBinding(
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_hash=protocol.protocol_hash,
        pilot_report_hash=protocol.pilot_report_hash,
        primary_population_id=primary_population.population_id,
        primary_population_version=primary_population.version,
        primary_population_hash=primary_population.population_hash,
        secondary_population_id=secondary_population.population_id,
        secondary_population_version=secondary_population.version,
        secondary_population_hash=secondary_population.population_hash,
        gate6_report_hash=gate6_report_hash,
        gate6_reference_strength_hash=gate6_reference_strength_hash,
        git_commit=environment.git_commit,
        engine_version=environment.engine_version,
        python_version=environment.python_version,
        dependency_lockfile=environment.dependency_lockfile,
        poetry_lock_hash=environment.poetry_lock_hash,
        cpu_model=environment.cpu_model,
        cpu_core_count=environment.cpu_core_count,
        cpu_architecture=environment.cpu_architecture,
        ram_bytes=environment.ram_bytes,
        os_identifier=environment.os_identifier,
        hardware_fingerprint=hardware_fingerprint(environment),
        workers=workers,
    )


def save_final_execution_binding(binding: FinalExecutionBinding, path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        existing = load_final_execution_binding(destination)
        if existing.binding_hash != binding.binding_hash:
            raise FinalExecutionError("Existing final execution binding does not match current execution")
        return destination
    return _atomic_write_json(destination, binding.to_dict())


def load_final_execution_binding(path: str | Path) -> FinalExecutionBinding:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("binding"), Mapping):
        raise FinalExecutionError("Final binding file is malformed")
    payload = raw["binding"]
    stored_hash = _validate_sha256(str(raw.get("binding_hash")), "binding_hash")
    secondary = payload["secondary_population"]
    primary = payload["primary_population"]
    binding = FinalExecutionBinding(
        protocol_id=str(payload["protocol_id"]), protocol_version=str(payload["protocol_version"]),
        protocol_hash=str(payload["protocol_hash"]), pilot_report_hash=str(payload["pilot_report_hash"]),
        primary_population_id=str(primary["population_id"]), primary_population_version=str(primary["version"]),
        primary_population_hash=str(primary["population_hash"]), secondary_population_id=str(secondary["population_id"]),
        secondary_population_version=str(secondary["version"]), secondary_population_hash=str(secondary["population_hash"]),
        gate6_report_hash=str(payload["gate6_report_hash"]),
        gate6_reference_strength_hash=str(payload["gate6_reference_strength_hash"]),
        git_commit=str(payload["git_commit"]), engine_version=str(payload["engine_version"]),
        python_version=str(payload["python_version"]), dependency_lockfile=str(payload["dependency_lockfile"]),
        poetry_lock_hash=str(payload["poetry_lock_hash"]), cpu_model=str(payload["cpu_model"]),
        cpu_core_count=int(payload["cpu_core_count"]), cpu_architecture=str(payload["cpu_architecture"]),
        ram_bytes=int(payload["ram_bytes"]), os_identifier=str(payload["os_identifier"]),
        hardware_fingerprint=str(payload["hardware_fingerprint"]), workers=int(payload["workers"]),
    )
    if binding.binding_hash != stored_hash:
        raise FinalExecutionError("binding_hash does not match final binding payload")
    return binding


def _load_protocol_payload(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("protocol"), Mapping):
        raise FinalExecutionError("Final protocol file is malformed")
    return dict(raw["protocol"])


def _rng_configuration(protocol: Phase1BaselineExperimentProtocol) -> dict[str, Any]:
    return {
        "rng_contract": "event-indexed-crn-v1",
        "crn_mode": "identical_realisations",
        "deal_namespace": "DEAL_SCHEDULE",
        "bot_a_namespace": "RNG_BOT_A",
        "bot_b_namespace": "RNG_BOT_B",
        "agent_event_key": "(bot_id,hand_id,decision_index,purpose)",
        "seed_schedule_version": protocol.seed_schedule_version,
    }


def build_final_job_experiment_config(
    *, protocol: Phase1BaselineExperimentProtocol, protocol_payload: Mapping[str, Any],
    binding: FinalExecutionBinding, job: FinalJob, opponent_gate_map_hash: str,
    opponent_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_sha256(opponent_gate_map_hash, "opponent_gate_map_hash")
    return {
        "config_version": PHASE1_FINAL_EXPERIMENT_CONFIG_VERSION,
        "protocol": dict(protocol_payload),
        "execution_binding": binding.reference_payload(),
        "job": {
            "cell_id": job.cell_id, "hero_id": job.hero_id, "hero_version": job.hero_version,
            "opponent_profile_id": job.opponent_profile_id,
            "opponent_profile_version": job.opponent_profile_version,
            "opponent_gate_map_hash": opponent_gate_map_hash, "master_seed": job.master_seed,
            "paired_units": job.paired_units, "simulated_engine_hands": job.simulated_engine_hands,
        },
        "run_provenance": {
            "experiment_id": f"{protocol.protocol_id}:{binding.binding_hash}",
            "game_config_hash": protocol.game_config_hash, "bot_version": job.hero_version,
            "opponent_version": f"{job.opponent_profile_version}+parametric-policy-v1",
            "opponent_parameters": dict(opponent_parameters), "population_version": binding.primary_population_version,
            "master_seed": job.master_seed, "rng_configuration": _rng_configuration(protocol),
            "engine_version": binding.engine_version, "python_version": binding.python_version,
            "poetry_lock_hash": binding.poetry_lock_hash, "git_commit": binding.git_commit,
            "cpu_model": binding.cpu_model, "cpu_core_count": binding.cpu_core_count,
            "cpu_architecture": binding.cpu_architecture, "os_identifier": binding.os_identifier,
            "statistical_unit": protocol.statistical_unit, "gate6_report_hash": binding.gate6_report_hash,
            "pilot_report_hash": binding.pilot_report_hash,
        },
        "scientific_use": {
            "final_experiment_data": True, "pilot_data_included": False,
            "confirmatory_inference": False, "optional_stopping": False,
        },
    }


def final_job_match_id(binding: FinalExecutionBinding, job: FinalJob) -> str:
    identity = {
        "protocol_hash": binding.protocol_hash, "binding_hash": binding.binding_hash,
        "cell_id": job.cell_id, "hero_id": job.hero_id,
        "opponent_profile_id": job.opponent_profile_id, "master_seed": job.master_seed,
    }
    return f"phase1-final-{sha256_canonical(identity)[:24]}"


@dataclass(frozen=True)
class FinalJobExecution:
    cell_id: str
    hero_id: str
    hero_version: str
    opponent_profile_id: str
    opponent_profile_version: str
    master_seed: int
    match_id: str
    experiment_config_hash: str
    match_record_hash: str
    status: str
    paired_difference_bb: float | None
    failed_runs: tuple[Mapping[str, Any], ...]
    elapsed_ns: int
    recorded_engine_hands: int
    binding_hash: str
    match_record_payload: Mapping[str, Any] | None = None
    snapshot_version: str = PHASE1_FINAL_JOB_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.snapshot_version != PHASE1_FINAL_JOB_SNAPSHOT_VERSION or self.status not in {"valid", "failed"}:
            raise FinalExecutionError("Invalid final job snapshot version/status")
        if self.elapsed_ns < 0 or self.recorded_engine_hands < 0:
            raise FinalExecutionError("Runtime/hand counts must be non-negative")
        _validate_sha256(self.experiment_config_hash, "experiment_config_hash")
        _validate_sha256(self.match_record_hash, "match_record_hash")
        _validate_sha256(self.binding_hash, "binding_hash")
        if self.status == "valid":
            if self.paired_difference_bb is None or self.failed_runs or not math.isfinite(self.paired_difference_bb):
                raise FinalExecutionError("Valid final job evidence is inconsistent")
        elif self.paired_difference_bb is not None or not self.failed_runs:
            raise FinalExecutionError("Failed final job must retain failure evidence")

    @classmethod
    def from_runner_result(
        cls, *, job: FinalJob, binding: FinalExecutionBinding,
        result_paired_differences: Sequence[float], num_valid_pairs: int,
        failed_runs: Sequence[Mapping[str, Any]], elapsed_ns: int, match_record: MatchRecord,
    ) -> "FinalJobExecution":
        if num_valid_pairs == 1 and len(result_paired_differences) == 1 and not failed_runs:
            status, difference = "valid", float(result_paired_differences[0])
        elif num_valid_pairs == 0 and not result_paired_differences and failed_runs:
            status, difference = "failed", None
        else:
            raise FinalExecutionError("Each frozen final job must contain exactly one paired unit")
        if match_record.match_id != final_job_match_id(binding, job) or match_record.master_seed != job.master_seed:
            raise FinalExecutionError("MatchRecord identity does not match final job")
        if match_record.experiment_config_hash is None:
            raise FinalExecutionError("Final MatchRecord requires experiment_config_hash")
        return cls(
            cell_id=job.cell_id, hero_id=job.hero_id, hero_version=job.hero_version,
            opponent_profile_id=job.opponent_profile_id, opponent_profile_version=job.opponent_profile_version,
            master_seed=job.master_seed, match_id=match_record.match_id,
            experiment_config_hash=match_record.experiment_config_hash, match_record_hash=match_record.record_hash,
            status=status, paired_difference_bb=difference,
            failed_runs=tuple(dict(item) for item in failed_runs), elapsed_ns=elapsed_ns,
            recorded_engine_hands=len(match_record.hands), binding_hash=binding.binding_hash,
            match_record_payload=match_record.to_dict(),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version, "binding_hash": self.binding_hash,
            "cell_id": self.cell_id, "hero_id": self.hero_id, "hero_version": self.hero_version,
            "opponent_profile_id": self.opponent_profile_id, "opponent_profile_version": self.opponent_profile_version,
            "master_seed": self.master_seed, "match_id": self.match_id,
            "experiment_config_hash": self.experiment_config_hash, "match_record_hash": self.match_record_hash,
            "status": self.status, "paired_difference_bb": self.paired_difference_bb,
            "failed_runs": [dict(item) for item in self.failed_runs], "elapsed_ns": self.elapsed_ns,
            "recorded_engine_hands": self.recorded_engine_hands,
            "match_record": dict(self.match_record_payload) if self.match_record_payload is not None else None,
        }

    @property
    def snapshot_hash(self) -> str:
        return sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot": self.payload(), "snapshot_hash": self.snapshot_hash}


def _execution_matches_job(execution: FinalJobExecution, job: FinalJob) -> bool:
    return (
        execution.cell_id == job.cell_id and execution.hero_id == job.hero_id
        and execution.hero_version == job.hero_version
        and execution.opponent_profile_id == job.opponent_profile_id
        and execution.opponent_profile_version == job.opponent_profile_version
        and execution.master_seed == job.master_seed
    )


def save_final_job_snapshot(execution: FinalJobExecution, path: str | Path) -> Path:
    if execution.match_record_payload is None:
        raise FinalExecutionError("Persisted final checkpoint requires MatchRecord payload")
    destination = Path(path)
    if destination.exists():
        existing = load_final_job_snapshot(destination, binding_hash=execution.binding_hash)
        if existing.snapshot_hash != execution.snapshot_hash:
            raise FinalExecutionError("Existing final checkpoint does not match")
        return destination
    return _atomic_write_json(destination, execution.to_dict())


def load_final_job_snapshot(
    path: str | Path, *, binding: FinalExecutionBinding | None = None,
    binding_hash: str | None = None, expected_job: FinalJob | None = None,
) -> FinalJobExecution:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("snapshot"), Mapping):
        raise FinalExecutionError("Final job snapshot file is malformed")
    payload = raw["snapshot"]
    stored_hash = _validate_sha256(str(raw.get("snapshot_hash")), "snapshot_hash")
    if sha256_canonical(payload) != stored_hash:
        raise FinalExecutionError("snapshot_hash does not match final snapshot payload")
    failures = payload.get("failed_runs")
    if not isinstance(failures, list):
        raise FinalExecutionError("failed_runs must be an array")
    execution = FinalJobExecution(
        cell_id=str(payload["cell_id"]), hero_id=str(payload["hero_id"]), hero_version=str(payload["hero_version"]),
        opponent_profile_id=str(payload["opponent_profile_id"]), opponent_profile_version=str(payload["opponent_profile_version"]),
        master_seed=int(payload["master_seed"]), match_id=str(payload["match_id"]),
        experiment_config_hash=str(payload["experiment_config_hash"]), match_record_hash=str(payload["match_record_hash"]),
        status=str(payload["status"]), paired_difference_bb=(None if payload["paired_difference_bb"] is None else float(payload["paired_difference_bb"])),
        failed_runs=tuple(dict(item) for item in failures), elapsed_ns=int(payload["elapsed_ns"]),
        recorded_engine_hands=int(payload["recorded_engine_hands"]), binding_hash=str(payload["binding_hash"]),
        match_record_payload=(None if payload.get("match_record") is None else dict(payload["match_record"])),
    )
    expected_binding_hash = binding.binding_hash if binding is not None else binding_hash
    if expected_binding_hash is not None and execution.binding_hash != expected_binding_hash:
        raise FinalExecutionError("Final checkpoint binding mismatch")
    if expected_job is not None and not _execution_matches_job(execution, expected_job):
        raise FinalExecutionError("Final checkpoint job identity mismatch")
    if execution.match_record_payload is not None:
        record = MatchRecord.from_dict(execution.match_record_payload)
        if record.record_hash != execution.match_record_hash or record.experiment_config_hash != execution.experiment_config_hash:
            raise FinalExecutionError("Embedded MatchRecord integrity mismatch")
    if binding is not None and expected_job is not None and execution.match_id != final_job_match_id(binding, expected_job):
        raise FinalExecutionError("Final checkpoint match_id mismatch")
    return execution


@dataclass(frozen=True)
class FinalPrecisionAssessment:
    estimator_version: str
    confidence_level: float
    target_half_width_bb100: float
    mean_ev_bb100: float
    standard_error_bb100: float
    ci_low_bb100: float
    ci_high_bb100: float
    half_width_bb100: float
    target_met: bool

    def __post_init__(self) -> None:
        if self.estimator_version != PHASE1_FINAL_PRECISION_DIAGNOSTIC_VERSION:
            raise FinalExecutionError("Unsupported final precision diagnostic version")
        numeric = (
            self.confidence_level, self.target_half_width_bb100, self.mean_ev_bb100,
            self.standard_error_bb100, self.ci_low_bb100, self.ci_high_bb100,
            self.half_width_bb100,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise FinalExecutionError("Final precision values must be finite")
        if self.confidence_level != 0.95 or self.target_half_width_bb100 != 50.0:
            raise FinalExecutionError("Final precision disclosure must remain frozen at 95% and ±50 BB/100")
        if self.standard_error_bb100 < 0.0 or self.half_width_bb100 < 0.0:
            raise FinalExecutionError("Final precision widths must be non-negative")
        if not math.isclose(
            self.half_width_bb100, (self.ci_high_bb100 - self.ci_low_bb100) / 2.0,
            rel_tol=0.0, abs_tol=1e-9,
        ):
            raise FinalExecutionError("Final precision CI/half-width are inconsistent")
        if self.target_met != (self.half_width_bb100 <= self.target_half_width_bb100):
            raise FinalExecutionError("Final precision target_met flag is inconsistent")

    def payload(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class FinalHeroResult:
    hero_id: str
    hero_version: str
    requested_units: int
    valid_units: int
    failed_units: int
    recorded_engine_hands: int
    total_runtime_ns: int
    primary_population: PopulationEvaluation | None
    secondary_population: PopulationEvaluation | None
    precision: FinalPrecisionAssessment | None

    @property
    def failure_free(self) -> bool:
        return self.failed_units == 0 and self.valid_units == self.requested_units

    def payload(self) -> dict[str, Any]:
        return {
            "hero_id": self.hero_id, "hero_version": self.hero_version,
            "requested_units": self.requested_units, "valid_units": self.valid_units,
            "failed_units": self.failed_units, "recorded_engine_hands": self.recorded_engine_hands,
            "total_runtime_ns": self.total_runtime_ns, "failure_free": self.failure_free,
            "primary_population": None if self.primary_population is None else self.primary_population.to_dict(),
            "secondary_population": None if self.secondary_population is None else self.secondary_population.to_dict(),
            "precision": None if self.precision is None else self.precision.payload(),
        }


@dataclass(frozen=True)
class Phase1FinalExecutionReport:
    protocol_id: str
    protocol_version: str
    protocol_hash: str
    pilot_report_hash: str
    binding_hash: str
    gate6_report_hash: str
    workers: int
    requested_jobs: int
    completed_jobs: int
    valid_units: int
    failed_units: int
    requested_engine_hands: int
    recorded_engine_hands: int
    heroes: tuple[FinalHeroResult, ...]
    complete: bool
    failure_free: bool
    report_version: str = PHASE1_FINAL_EXECUTION_REPORT_VERSION

    def __post_init__(self) -> None:
        if self.report_version != PHASE1_FINAL_EXECUTION_REPORT_VERSION:
            raise FinalExecutionError("Unsupported final execution report version")
        for name in ("protocol_hash", "pilot_report_hash", "binding_hash", "gate6_report_hash"):
            _validate_sha256(str(getattr(self, name)), name)
        counts = (
            self.workers, self.requested_jobs, self.completed_jobs, self.valid_units,
            self.failed_units, self.requested_engine_hands, self.recorded_engine_hands,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise FinalExecutionError("Final report counts must be integers")
        if self.workers <= 0 or any(value < 0 for value in counts[1:]):
            raise FinalExecutionError("Final report counts are invalid")
        if self.completed_jobs > self.requested_jobs:
            raise FinalExecutionError("Final completed_jobs exceeds requested_jobs")
        if self.valid_units + self.failed_units != self.completed_jobs:
            raise FinalExecutionError("Final valid/failed counts are inconsistent")
        if self.complete != (self.completed_jobs == self.requested_jobs):
            raise FinalExecutionError("Final complete flag is inconsistent")
        if self.failure_free != (self.complete and self.failed_units == 0):
            raise FinalExecutionError("Final failure_free flag is inconsistent")
        if sum(hero.requested_units for hero in self.heroes) != self.requested_jobs:
            raise FinalExecutionError("Final hero requested counts are inconsistent")
        if sum(hero.valid_units for hero in self.heroes) != self.valid_units:
            raise FinalExecutionError("Final hero valid counts are inconsistent")
        if sum(hero.failed_units for hero in self.heroes) != self.failed_units:
            raise FinalExecutionError("Final hero failed counts are inconsistent")
        if sum(hero.recorded_engine_hands for hero in self.heroes) != self.recorded_engine_hands:
            raise FinalExecutionError("Final hero hand counts are inconsistent")
        for hero in self.heroes:
            if hero.failure_free:
                if hero.primary_population is None or hero.secondary_population is None or hero.precision is None:
                    raise FinalExecutionError("Failure-free final hero requires complete scientific results")
                if not hero.primary_population.tail_precision_warning or not hero.secondary_population.tail_precision_warning:
                    raise FinalExecutionError("Five-profile population results must retain tail warning")
            elif any(value is not None for value in (hero.primary_population, hero.secondary_population, hero.precision)):
                raise FinalExecutionError("Failed/incomplete final hero must not expose final population estimands")

    def payload(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version, "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version, "protocol_hash": self.protocol_hash,
            "pilot_report_hash": self.pilot_report_hash, "binding_hash": self.binding_hash,
            "gate6_report_hash": self.gate6_report_hash, "workers": self.workers,
            "requested_jobs": self.requested_jobs, "completed_jobs": self.completed_jobs,
            "valid_units": self.valid_units, "failed_units": self.failed_units,
            "requested_engine_hands": self.requested_engine_hands,
            "recorded_engine_hands": self.recorded_engine_hands,
            "heroes": [hero.payload() for hero in self.heroes], "complete": self.complete,
            "failure_free": self.failure_free,
            "precision_disclosure": {
                "frozen_target_half_width_bb100": 50.0,
                "confidence_level": 0.95,
                "project_documentation_illustrative_half_width_bb100": 1.0,
                "deviation_from_illustrative_example": True,
                "interpretation": "coarse_phase1_measurement_not_equivalent_to_plus_minus_1_bb100_precision",
            },
            "scope_limitations": {
                "static_profile_count": 5, "tail_precision_warning": True,
                "complete_dtest_population_benchmark": False,
                "full_section_11_2_benchmark_matrix": False,
            },
            "scientific_use": {
                "final_experiment_data": True, "pilot_data_included": False,
                "confirmatory_inference": False, "optional_stopping": False,
            },
        }

    @property
    def report_hash(self) -> str:
        return sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"report": self.payload(), "report_hash": self.report_hash}


def _bootstrap_seed(protocol_hash: str, hero_id: str, population_id: str) -> int:
    digest = sha256_canonical({
        "domain": "phase1-final-population-bootstrap-v1",
        "protocol_hash": protocol_hash, "hero_id": hero_id, "population_id": population_id,
    })
    return int(digest[:16], 16)


def _profile_evidence(
    *, hero_id: str, profile_ids: Sequence[str], executions: Sequence[FinalJobExecution]
) -> tuple[ProfilePerformanceEvidence, ...]:
    evidence: list[ProfilePerformanceEvidence] = []
    for profile_id in profile_ids:
        values = tuple(
            50.0 * float(item.paired_difference_bb)
            for item in sorted(
                (e for e in executions if e.hero_id == hero_id and e.opponent_profile_id == profile_id and e.status == "valid"),
                key=lambda e: e.master_seed,
            )
            if item.paired_difference_bb is not None
        )
        if not values:
            raise FinalExecutionError(f"No valid evidence for {hero_id}-vs-{profile_id}")
        evidence.append(ProfilePerformanceEvidence(profile_id=profile_id, run_ev_bb100=values, simulated_hands_per_run=(2,) * len(values)))
    return tuple(evidence)


def _precision_assessment(
    *, protocol: Phase1BaselineExperimentProtocol, population: DiscretePopulationDefinition,
    evidence: Sequence[ProfilePerformanceEvidence],
) -> FinalPrecisionAssessment:
    by_profile = {item.profile_id: item for item in evidence}
    mean = sum(
        weight / 1_000_000 * by_profile[profile_id].mean_ev_bb100
        for profile_id, weight in population.weights_ppm
    )
    variance = 0.0
    for profile_id, weight in population.weights_ppm:
        values = by_profile[profile_id].run_ev_bb100
        if len(values) < 2:
            raise FinalExecutionError("Final precision assessment requires at least two units per profile")
        sample_var = statistics.variance(values)
        variance += (weight / 1_000_000) ** 2 * sample_var / len(values)
    se = math.sqrt(variance)
    half = PHASE1_FINAL_Z * se
    return FinalPrecisionAssessment(
        estimator_version=PHASE1_FINAL_PRECISION_DIAGNOSTIC_VERSION,
        confidence_level=protocol.confidence_level,
        target_half_width_bb100=protocol.target_half_width_bb100,
        mean_ev_bb100=mean, standard_error_bb100=se,
        ci_low_bb100=mean - half, ci_high_bb100=mean + half,
        half_width_bb100=half, target_met=half <= protocol.target_half_width_bb100,
    )


def build_final_report(
    *, protocol: Phase1BaselineExperimentProtocol, binding: FinalExecutionBinding,
    primary_population: DiscretePopulationDefinition, secondary_population: DiscretePopulationDefinition,
    jobs: Sequence[FinalJob], executions: Sequence[FinalJobExecution],
) -> Phase1FinalExecutionReport:
    job_by_key = {(job.cell_id, job.master_seed): job for job in jobs}
    if len(job_by_key) != len(jobs):
        raise FinalExecutionError("Final jobs contain duplicate identities")
    execution_by_key: dict[tuple[str, int], FinalJobExecution] = {}
    for execution in executions:
        key = (execution.cell_id, execution.master_seed)
        expected = job_by_key.get(key)
        if expected is None or key in execution_by_key or not _execution_matches_job(execution, expected):
            raise FinalExecutionError("Final execution evidence contains duplicate/unknown jobs")
        if execution.binding_hash != binding.binding_hash:
            raise FinalExecutionError("Final execution evidence uses a different binding")
        execution_by_key[key] = execution

    hero_results: list[FinalHeroResult] = []
    for plan in protocol.agent_plans:
        hero_jobs = [job for job in jobs if job.hero_id == plan.agent_id]
        hero_execs = [execution_by_key[(job.cell_id, job.master_seed)] for job in hero_jobs if (job.cell_id, job.master_seed) in execution_by_key]
        failed = sum(item.status == "failed" for item in hero_execs)
        valid = sum(item.status == "valid" for item in hero_execs)
        primary_eval: PopulationEvaluation | None = None
        secondary_eval: PopulationEvaluation | None = None
        precision: FinalPrecisionAssessment | None = None
        if len(hero_execs) == len(hero_jobs) and failed == 0:
            evidence = _profile_evidence(hero_id=plan.agent_id, profile_ids=protocol.opponent_profile_ids, executions=hero_execs)
            primary_eval = evaluate_population(
                primary_population, evidence,
                bootstrap_seed=_bootstrap_seed(protocol.protocol_hash, plan.agent_id, primary_population.population_id),
                bootstrap_resamples=DEFAULT_POPULATION_BOOTSTRAP_RESAMPLES,
            )
            secondary_eval = evaluate_population(
                secondary_population, evidence,
                bootstrap_seed=_bootstrap_seed(protocol.protocol_hash, plan.agent_id, secondary_population.population_id),
                bootstrap_resamples=DEFAULT_POPULATION_BOOTSTRAP_RESAMPLES,
            )
            precision = _precision_assessment(protocol=protocol, population=primary_population, evidence=evidence)
            if not math.isclose(precision.mean_ev_bb100, primary_eval.mean_ev_bb100, rel_tol=0.0, abs_tol=1e-12):
                raise FinalExecutionError("Primary precision and population point estimates disagree")
        hero_results.append(FinalHeroResult(
            hero_id=plan.agent_id, hero_version=plan.agent_version, requested_units=len(hero_jobs),
            valid_units=valid, failed_units=failed,
            recorded_engine_hands=sum(item.recorded_engine_hands for item in hero_execs),
            total_runtime_ns=sum(item.elapsed_ns for item in hero_execs),
            primary_population=primary_eval, secondary_population=secondary_eval, precision=precision,
        ))

    completed = len(execution_by_key)
    failed_total = sum(item.status == "failed" for item in execution_by_key.values())
    valid_total = sum(item.status == "valid" for item in execution_by_key.values())
    complete = completed == len(jobs)
    failure_free = complete and failed_total == 0
    return Phase1FinalExecutionReport(
        protocol_id=protocol.protocol_id, protocol_version=protocol.protocol_version,
        protocol_hash=protocol.protocol_hash, pilot_report_hash=protocol.pilot_report_hash,
        binding_hash=binding.binding_hash, gate6_report_hash=binding.gate6_report_hash,
        workers=binding.workers, requested_jobs=len(jobs), completed_jobs=completed,
        valid_units=valid_total, failed_units=failed_total,
        requested_engine_hands=sum(job.simulated_engine_hands for job in jobs),
        recorded_engine_hands=sum(item.recorded_engine_hands for item in execution_by_key.values()),
        heroes=tuple(hero_results), complete=complete, failure_free=failure_free,
    )


def save_phase1_final_execution_report(report: Phase1FinalExecutionReport, path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        existing = load_phase1_final_execution_report(destination)
        if existing.report_hash != report.report_hash:
            raise FinalExecutionError("Existing final report does not match reconstructed evidence")
        return destination
    return _atomic_write_json(destination, report.to_dict())


def _population_from_dict(value: Mapping[str, Any] | None) -> PopulationEvaluation | None:
    if value is None:
        return None
    report = value.get("report")
    stored_hash = _validate_sha256(str(value.get("report_hash")), "population_report_hash")
    if not isinstance(report, Mapping) or sha256_canonical(report) != stored_hash:
        raise FinalExecutionError("Embedded population report hash mismatch")
    profile_evidence = tuple(
        PopulationProfileEvidenceSummary(
            profile_id=str(item["profile_id"]),
            weight_ppm=int(item["weight_ppm"]),
            run_ev_bb100=tuple(float(v) for v in item["run_ev_bb100"]),
            simulated_hands_per_run=tuple(int(v) for v in item["simulated_hands_per_run"]),
            mean_ev_bb100=float(item["mean_ev_bb100"]),
        )
        for item in report["profile_evidence"]
    )
    evaluation = PopulationEvaluation(
        population_id=str(report["population_id"]),
        population_version=str(report["population_version"]),
        population_hash=str(report["population_hash"]),
        evidence_hash=str(report["evidence_hash"]),
        profile_evidence=profile_evidence,
        profile_count=int(report["profile_count"]),
        run_count=int(report["run_count"]),
        total_simulated_hands=int(report["total_simulated_hands"]),
        mean_ev_bb100=float(report["mean_ev_bb100"]),
        mean_ci_low_bb100=float(report["mean_ci_low_bb100"]),
        mean_ci_high_bb100=float(report["mean_ci_high_bb100"]),
        median_ev_bb100=float(report["median_ev_bb100"]),
        r10_bb100=float(report["r10_bb100"]),
        r10_ci_low_bb100=float(report["r10_ci_low_bb100"]),
        r10_ci_high_bb100=float(report["r10_ci_high_bb100"]),
        q25_bb100=float(report["q25_bb100"]),
        stddev_bb100=float(report["stddev_bb100"]),
        empirical_min_bb100=float(report["empirical_min_bb100"]),
        tail_precision_warning=bool(report["tail_precision_warning"]),
        bootstrap_seed=int(report["bootstrap_seed"]),
        bootstrap_resamples=int(report["bootstrap_resamples"]),
        evaluation_version=str(report["evaluation_version"]),
        quantile_version=str(report["quantile_version"]),
        bootstrap_version=str(report["bootstrap_version"]),
        percentile_method=str(report["percentile_method"]),
        confidence_level=float(report["confidence_level"]),
    )
    if evaluation.report_hash != stored_hash:
        raise FinalExecutionError("Embedded population evaluation is not canonical")
    return evaluation


def load_phase1_final_execution_report(path: str | Path) -> Phase1FinalExecutionReport:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("report"), Mapping):
        raise FinalExecutionError("Final execution report file is malformed")
    report_payload = raw["report"]
    stored_hash = _validate_sha256(str(raw.get("report_hash")), "report_hash")
    if sha256_canonical(report_payload) != stored_hash:
        raise FinalExecutionError("report_hash does not match final report payload")
    heroes: list[FinalHeroResult] = []
    for item in report_payload["heroes"]:
        precision_raw = item.get("precision")
        precision = None if precision_raw is None else FinalPrecisionAssessment(**precision_raw)
        heroes.append(FinalHeroResult(
            hero_id=str(item["hero_id"]), hero_version=str(item["hero_version"]),
            requested_units=int(item["requested_units"]), valid_units=int(item["valid_units"]),
            failed_units=int(item["failed_units"]), recorded_engine_hands=int(item["recorded_engine_hands"]),
            total_runtime_ns=int(item["total_runtime_ns"]),
            primary_population=_population_from_dict(item.get("primary_population")),
            secondary_population=_population_from_dict(item.get("secondary_population")),
            precision=precision,
        ))
    loaded = Phase1FinalExecutionReport(
        protocol_id=str(report_payload["protocol_id"]), protocol_version=str(report_payload["protocol_version"]),
        protocol_hash=str(report_payload["protocol_hash"]), pilot_report_hash=str(report_payload["pilot_report_hash"]),
        binding_hash=str(report_payload["binding_hash"]), gate6_report_hash=str(report_payload["gate6_report_hash"]),
        workers=int(report_payload["workers"]), requested_jobs=int(report_payload["requested_jobs"]),
        completed_jobs=int(report_payload["completed_jobs"]), valid_units=int(report_payload["valid_units"]),
        failed_units=int(report_payload["failed_units"]), requested_engine_hands=int(report_payload["requested_engine_hands"]),
        recorded_engine_hands=int(report_payload["recorded_engine_hands"]), heroes=tuple(heroes),
        complete=bool(report_payload["complete"]), failure_free=bool(report_payload["failure_free"]),
    )
    if loaded.report_hash != stored_hash:
        raise FinalExecutionError("Loaded final execution report is not canonical")
    return loaded


def _gate_map_hash(context: VerifiedStaticProfileContext, profile_id: str) -> str:
    acceptance = next(
        (item for item in context.gate6_report.gate6_result.results if item.profile_name == profile_id),
        None,
    )
    if acceptance is None:
        raise FinalExecutionError(f"Gate-6 report is missing profile {profile_id}")
    return acceptance.gate_map_hash


def _build_hero_agent(hero_id: str) -> Agent:
    if hero_id == "random":
        return RandomBaselineAgent()
    if hero_id == "hand_strength":
        return HandStrengthBaselineAgent()
    raise FinalExecutionError(f"Unknown frozen final hero: {hero_id}")


@dataclass(frozen=True)
class _WorkerRequest:
    job: FinalJob
    binding: FinalExecutionBinding
    snapshot_root: str
    gate6_report_path: str
    experiment_config_hash: str


def _execute_worker_request(request: _WorkerRequest) -> FinalJobExecution:
    context = _worker_verified_context(request.snapshot_root, request.gate6_report_path)
    hero = _build_hero_agent(request.job.hero_id)
    opponent = build_verified_static_profile_agent(request.job.opponent_profile_id, context=context)
    runner = PairedRunner(
        agent_a=hero, agent_b=opponent, num_hands=1, seed=request.job.master_seed,
        game_config=BASELINE_GAME_CONFIG, adaptive=False,
        illegal_action_policy=IllegalActionPolicy.EXCLUDE_RUN,
        match_id=final_job_match_id(request.binding, request.job),
        experiment_metadata={
            "experiment_id": f"{request.binding.protocol_id}:{request.binding.binding_hash}",
            "experiment_config_hash": request.experiment_config_hash,
            "phase1_final_protocol_hash": request.binding.protocol_hash,
            "phase1_final_binding_hash": request.binding.binding_hash,
            "phase1_final_cell_id": request.job.cell_id,
            "phase1_pilot_report_hash": request.binding.pilot_report_hash,
        },
    )
    start_ns = time.perf_counter_ns()
    result = runner.run()
    elapsed_ns = time.perf_counter_ns() - start_ns
    if result.statistical_unit is not StatisticalUnit.PAIRED_DEAL or result.match_record is None:
        raise FinalExecutionError("Frozen final worker produced invalid statistical unit/MatchRecord")
    return FinalJobExecution.from_runner_result(
        job=request.job, binding=request.binding,
        result_paired_differences=result.paired_differences, num_valid_pairs=result.num_valid_pairs,
        failed_runs=result.failed_runs, elapsed_ns=elapsed_ns, match_record=result.match_record,
    )


def _snapshot_path(work_root: Path, job: FinalJob) -> Path:
    return work_root / "jobs" / job.cell_id / f"seed-{job.master_seed}.json"


def _match_metadata(
    *, protocol: Phase1BaselineExperimentProtocol, binding: FinalExecutionBinding,
    job: FinalJob, experiment_config: Mapping[str, Any], opponent_gate_map_hash: str,
    opponent_parameters: Mapping[str, Any],
) -> MatchPersistenceMetadata:
    return MatchPersistenceMetadata(
        bot_version=job.hero_version, opponent_hash=opponent_gate_map_hash,
        engine_version=binding.engine_version, python_version=binding.python_version,
        rng_version="event-indexed-crn-v1", git_commit=binding.git_commit,
        population_id=binding.primary_population_id, population_seed=None,
        opponent_parameters=dict(opponent_parameters),
        bot_configuration={
            "agent_id": job.hero_id, "agent_version": job.hero_version,
            "final_binding_hash": binding.binding_hash,
            "pilot_report_hash": binding.pilot_report_hash,
            "gate6_report_hash": binding.gate6_report_hash,
        },
        statistical_method=PHASE1_FINAL_PRECISION_DIAGNOSTIC_VERSION,
        sample_size=1,
        precision_target={
            "version": PHASE1_FINAL_PRECISION_DIAGNOSTIC_VERSION,
            "confidence_level": protocol.confidence_level,
            "target_half_width_bb100": protocol.target_half_width_bb100,
            "estimand": "baseline_synthetic_population_mean_ev_bb100",
        },
        experiment_config=dict(experiment_config), poetry_lock_hash=binding.poetry_lock_hash,
        hardware_fingerprint=binding.hardware_fingerprint, os_identifier=binding.os_identifier,
    )


def _ensure_persisted_match(
    *, store: SQLiteBenchmarkStore, existing_match_ids: set[str],
    execution: FinalJobExecution, expected_metadata: MatchPersistenceMetadata,
) -> None:
    if execution.match_record_payload is None:
        raise FinalExecutionError("Final checkpoint cannot recover SQLite without MatchRecord")
    record = MatchRecord.from_dict(execution.match_record_payload)
    if record.record_hash != execution.match_record_hash:
        raise FinalExecutionError("Final checkpoint MatchRecord hash mismatch")
    if execution.match_id in existing_match_ids:
        if store.load_match(execution.match_id).record_hash != execution.match_record_hash:
            raise FinalExecutionError("Persisted SQLite MatchRecord does not match final checkpoint")
        if store.load_match_metadata(execution.match_id) != expected_metadata:
            raise FinalExecutionError("Persisted SQLite metadata does not match final binding/job")
        return
    store.save_match(record, metadata=expected_metadata)
    existing_match_ids.add(execution.match_id)


def _validated_protocol(
    *, protocol_path: Path, primary_population_path: Path, secondary_population_path: Path,
) -> tuple[Phase1BaselineExperimentProtocol, DiscretePopulationDefinition, DiscretePopulationDefinition]:
    primary = load_frozen_population(primary_population_path)
    secondary = load_frozen_population(secondary_population_path)
    protocol = load_phase1_baseline_protocol(
        protocol_path,
        populations={primary.population_id: primary, secondary.population_id: secondary},
    )
    return protocol, primary, secondary


def execute_phase1_final(
    *, protocol_path: Path, primary_population_path: Path, secondary_population_path: Path,
    pilot_report_path: Path, snapshot_root: Path, gate6_report_path: Path,
    database: Path, work_root: Path, report_path: Path, repo_root: Path,
    poetry_lock_path: Path, ram_bytes: int, workers: int = 1,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> Phase1FinalExecutionReport:
    if isinstance(ram_bytes, bool) or not isinstance(ram_bytes, int) or ram_bytes <= 0:
        raise ValueError("ram_bytes must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")

    protocol, primary, secondary = _validated_protocol(
        protocol_path=Path(protocol_path), primary_population_path=Path(primary_population_path),
        secondary_population_path=Path(secondary_population_path),
    )
    verify_pilot_report_binding(protocol, Path(pilot_report_path))
    protocol_payload = _load_protocol_payload(Path(protocol_path))
    jobs = iter_final_jobs(protocol)

    context = load_verified_static_profile_context(
        snapshot_root=Path(snapshot_root), gate6_report_path=Path(gate6_report_path),
    )
    if context.gate6_report.report_hash != protocol.gate6_report_hash:
        raise FinalExecutionError("Loaded Gate-6 report does not match frozen final protocol")
    for profile_id in protocol.opponent_profile_ids:
        build_verified_static_profile_agent(profile_id, context=context)

    git_commit = _clean_git_commit(Path(repo_root))
    environment = capture_execution_environment(
        engine_version=f"git:{git_commit}", git_commit=git_commit,
        poetry_lock_path=Path(poetry_lock_path), ram_bytes=ram_bytes,
    )
    binding = build_final_execution_binding(
        protocol=protocol, primary_population=primary, secondary_population=secondary,
        gate6_report_hash=context.gate6_report.report_hash,
        gate6_reference_strength_hash=context.reference.reference_strength_hash,
        environment=environment, workers=workers,
    )
    work_root = Path(work_root)
    save_final_execution_binding(binding, work_root / "binding.json")

    store = SQLiteBenchmarkStore(database)
    store.initialize()
    existing_match_ids = set(store.list_match_ids())
    execution_by_key: dict[tuple[str, int], FinalJobExecution] = {}
    missing: list[tuple[FinalJob, Mapping[str, Any], MatchPersistenceMetadata]] = []

    for job in jobs:
        gate_map_hash = _gate_map_hash(context, job.opponent_profile_id)
        parameters = _profile_parameters(job.opponent_profile_id)
        config = build_final_job_experiment_config(
            protocol=protocol, protocol_payload=protocol_payload, binding=binding, job=job,
            opponent_gate_map_hash=gate_map_hash, opponent_parameters=parameters,
        )
        config_hash = sha256_canonical(config)
        metadata = _match_metadata(
            protocol=protocol, binding=binding, job=job, experiment_config=config,
            opponent_gate_map_hash=gate_map_hash, opponent_parameters=parameters,
        )
        snapshot_path = _snapshot_path(work_root, job)
        if snapshot_path.exists():
            execution = load_final_job_snapshot(snapshot_path, binding=binding, expected_job=job)
            if execution.experiment_config_hash != config_hash:
                raise FinalExecutionError("Final checkpoint experiment config hash mismatch")
            _ensure_persisted_match(
                store=store, existing_match_ids=existing_match_ids,
                execution=execution, expected_metadata=metadata,
            )
            execution_by_key[(job.cell_id, job.master_seed)] = execution
            continue
        match_id = final_job_match_id(binding, job)
        if match_id in existing_match_ids:
            raise FinalExecutionError(
                "SQLite contains a final match without its runtime checkpoint; refusing to invent runtime evidence"
            )
        missing.append((job, config, metadata))

    resumed = len(execution_by_key)
    total = len(jobs)
    if progress_callback is not None:
        progress_callback(resumed, total, resumed, sum(item.status == "failed" for item in execution_by_key.values()))

    def persist(job: FinalJob, execution: FinalJobExecution, metadata: MatchPersistenceMetadata) -> None:
        save_final_job_snapshot(execution, _snapshot_path(work_root, job))
        _ensure_persisted_match(
            store=store, existing_match_ids=existing_match_ids,
            execution=execution, expected_metadata=metadata,
        )
        execution_by_key[(job.cell_id, job.master_seed)] = execution
        if progress_callback is not None:
            progress_callback(
                len(execution_by_key), total, resumed,
                sum(item.status == "failed" for item in execution_by_key.values()),
            )

    requests = [
        (job, _WorkerRequest(
            job=job, binding=binding, snapshot_root=str(Path(snapshot_root)),
            gate6_report_path=str(Path(gate6_report_path)), experiment_config_hash=sha256_canonical(config),
        ), metadata)
        for job, config, metadata in missing
    ]
    if workers == 1:
        for job, request, metadata in requests:
            persist(job, _execute_worker_request(request), metadata)
    elif requests:
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = {executor.submit(_execute_worker_request, request): (job, metadata) for job, request, metadata in requests}
        try:
            for future in as_completed(futures):
                job, metadata = futures[future]
                persist(job, future.result(), metadata)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    executions = tuple(execution_by_key[key] for key in sorted(execution_by_key))
    report = build_final_report(
        protocol=protocol, binding=binding, primary_population=primary,
        secondary_population=secondary, jobs=jobs, executions=executions,
    )
    if not report.complete:
        raise FinalExecutionError("Final execution ended without all frozen jobs checkpointed")
    save_phase1_final_execution_report(report, report_path)
    return report


def render_phase1_final_execution_report_text(report: Phase1FinalExecutionReport) -> str:
    lines = [
        "Phase-1 final baseline execution report",
        f"  status: {'COMPLETE' if report.complete and report.failure_free else 'INCOMPLETE_FAILURES' if report.complete else 'INCOMPLETE'}",
        f"  protocol: {report.protocol_id} ({report.protocol_hash})",
        f"  pilot report hash: {report.pilot_report_hash}",
        f"  binding hash: {report.binding_hash}",
        f"  Gate-6 report hash: {report.gate6_report_hash}",
        f"  workers: {report.workers}",
        f"  completed jobs: {report.completed_jobs}/{report.requested_jobs}",
        f"  valid paired units: {report.valid_units}",
        f"  failed paired units: {report.failed_units}",
        f"  recorded engine hands: {report.recorded_engine_hands}/{report.requested_engine_hands}",
        "  precision disclosure: frozen 95% target is ±50 BB/100; project ±1 BB/100 is illustrative only",
    ]
    for hero in report.heroes:
        lines.append(f"  {hero.hero_id}: valid={hero.valid_units}/{hero.requested_units}, failed={hero.failed_units}")
        if hero.primary_population is not None and hero.precision is not None:
            p = hero.primary_population
            q = hero.precision
            lines.extend([
                f"    Baseline Synthetic mean EV: {p.mean_ev_bb100:.6f} BB/100",
                f"    nested-bootstrap 95% mean CI: [{p.mean_ci_low_bb100:.6f}, {p.mean_ci_high_bb100:.6f}] BB/100",
                f"    fixed-profile precision CI: [{q.ci_low_bb100:.6f}, {q.ci_high_bb100:.6f}] BB/100",
                f"    fixed-profile half-width: {q.half_width_bb100:.6f} BB/100 (target {'MET' if q.target_met else 'NOT MET'})",
                f"    R10: {p.r10_bb100:.6f} BB/100 (five-profile tail warning: yes)",
            ])
        else:
            lines.append("    final population estimands withheld because frozen allocation was not failure-free")
    lines.extend([
        "  confirmatory inference: not performed",
        "  pilot observations included in final estimates: no",
        f"  report hash: {report.report_hash}",
    ])
    return "\n".join(lines) + "\n"
