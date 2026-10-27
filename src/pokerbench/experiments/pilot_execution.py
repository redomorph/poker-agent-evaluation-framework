"""Bound, resumable execution of the frozen Phase-1 variance/runtime pilot."""

from __future__ import annotations

import hashlib
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
from pokerbench.agents.profiles import ProfileName, get_static_profile
from pokerbench.agents.random_baseline import RandomBaselineAgent
from pokerbench.cli.run import (
    VerifiedStaticProfileContext,
    build_verified_static_profile_agent,
    load_verified_static_profile_context,
)
from pokerbench.engine.config import BASELINE_GAME_CONFIG
from pokerbench.experiments.config import (
    ExecutionEnvironment,
    capture_execution_environment,
)
from pokerbench.experiments.frozen_protocol import (
    Phase1PilotProtocol,
    PilotJob,
    PilotProtocolError,
    canonical_file_bytes,
    iter_pilot_jobs,
    load_frozen_population,
    load_phase1_pilot_protocol,
)
from pokerbench.persistence import SQLiteBenchmarkStore
from pokerbench.persistence.errors import MatchNotFoundError
from pokerbench.persistence.sqlite import MatchPersistenceMetadata
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import IllegalActionPolicy, PairedRunner, StatisticalUnit


PHASE1_PILOT_EXECUTION_BINDING_VERSION = "phase1-pilot-execution-binding-v1"
PHASE1_PILOT_JOB_SNAPSHOT_VERSION = "phase1-pilot-job-snapshot-v1"
PHASE1_PILOT_EXECUTION_REPORT_VERSION = "phase1-pilot-execution-report-v1"
PHASE1_PILOT_EXPERIMENT_CONFIG_VERSION = "phase1-pilot-job-config-v1"
PHASE1_PILOT_INTERNAL_CLI_VERSION = "phase1-pilot-execution-cli-v1"


class PilotExecutionError(ValueError):
    """Raised when Phase-1 pilot execution evidence is inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_canonical(value: Any) -> str:
    """Return SHA-256 over canonical compact JSON."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PilotExecutionError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PilotExecutionError(f"{field_name} must be a SHA-256 hex digest") from exc
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


def hardware_fingerprint(environment: ExecutionEnvironment) -> str:
    """Hash the platform-independent hardware identity required by the spec."""
    if not isinstance(environment, ExecutionEnvironment):
        raise TypeError("environment must be an ExecutionEnvironment")
    return sha256_canonical(
        {
            "cpu_model": environment.cpu_model,
            "cpu_core_count": environment.cpu_core_count,
            "cpu_architecture": environment.cpu_architecture,
        }
    )


@dataclass(frozen=True)
class PilotExecutionBinding:
    """Immutable execution-environment binding captured before the first hand."""

    protocol_id: str
    protocol_version: str
    protocol_hash: str
    primary_population_id: str
    primary_population_version: str
    primary_population_hash: str
    secondary_population_hashes: tuple[tuple[str, str, str], ...]
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
    binding_version: str = PHASE1_PILOT_EXECUTION_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.binding_version != PHASE1_PILOT_EXECUTION_BINDING_VERSION:
            raise PilotExecutionError("Unsupported pilot execution binding version")
        for field_name in (
            "protocol_hash",
            "primary_population_hash",
            "gate6_report_hash",
            "gate6_reference_strength_hash",
            "poetry_lock_hash",
            "hardware_fingerprint",
        ):
            _validate_sha256(str(getattr(self, field_name)), field_name)
        for population_id, version, population_hash in self.secondary_population_hashes:
            if not population_id or not version:
                raise PilotExecutionError("Secondary population identity must be non-empty")
            _validate_sha256(population_hash, "secondary_population_hash")
        if not self.git_commit:
            raise PilotExecutionError("git_commit must not be empty")
        if self.engine_version != f"git:{self.git_commit}":
            raise PilotExecutionError("engine_version must be derived from git_commit")
        for field_name, value in (
            ("cpu_core_count", self.cpu_core_count),
            ("ram_bytes", self.ram_bytes),
            ("workers", self.workers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PilotExecutionError(f"{field_name} must be a positive integer")

    def payload(self) -> dict[str, Any]:
        return {
            "binding_version": self.binding_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "primary_population": {
                "population_id": self.primary_population_id,
                "version": self.primary_population_version,
                "population_hash": self.primary_population_hash,
            },
            "secondary_populations": [
                {
                    "population_id": population_id,
                    "version": version,
                    "population_hash": population_hash,
                }
                for population_id, version, population_hash in self.secondary_population_hashes
            ],
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


def build_pilot_execution_binding(
    *,
    protocol: Phase1PilotProtocol,
    gate6_report_hash: str,
    gate6_reference_strength_hash: str,
    environment: ExecutionEnvironment,
    workers: int,
) -> PilotExecutionBinding:
    """Build a deterministic binding from validated protocol and environment facts."""
    if not isinstance(protocol, Phase1PilotProtocol):
        raise TypeError("protocol must be a Phase1PilotProtocol")
    if not isinstance(environment, ExecutionEnvironment):
        raise TypeError("environment must be an ExecutionEnvironment")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")

    return PilotExecutionBinding(
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_hash=protocol.protocol_hash,
        primary_population_id=protocol.primary_population.population_id,
        primary_population_version=protocol.primary_population.version,
        primary_population_hash=protocol.primary_population.population_hash,
        secondary_population_hashes=tuple(
            (item.population_id, item.version, item.population_hash)
            for item in protocol.secondary_populations
        ),
        gate6_report_hash=_validate_sha256(gate6_report_hash, "gate6_report_hash"),
        gate6_reference_strength_hash=_validate_sha256(
            gate6_reference_strength_hash, "gate6_reference_strength_hash"
        ),
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


def save_pilot_execution_binding(binding: PilotExecutionBinding, path: str | Path) -> Path:
    if not isinstance(binding, PilotExecutionBinding):
        raise TypeError("binding must be a PilotExecutionBinding")
    destination = Path(path)
    if destination.exists():
        existing = load_pilot_execution_binding(destination)
        if existing != binding:
            raise PilotExecutionError("Existing pilot execution binding does not match")
        return destination
    return _atomic_write_json(destination, binding.to_dict())


def load_pilot_execution_binding(path: str | Path) -> PilotExecutionBinding:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PilotExecutionError("Pilot binding file must be a JSON object")
    payload = raw.get("binding")
    if not isinstance(payload, Mapping):
        raise PilotExecutionError("Pilot binding payload must be a JSON object")
    stored_hash = _validate_sha256(str(raw.get("binding_hash")), "binding_hash")
    actual_hash = sha256_canonical(payload)
    if stored_hash != actual_hash:
        raise PilotExecutionError("binding_hash does not match binding payload")

    primary = payload.get("primary_population")
    if not isinstance(primary, Mapping):
        raise PilotExecutionError("primary_population must be an object")
    secondary_raw = payload.get("secondary_populations")
    if not isinstance(secondary_raw, list):
        raise PilotExecutionError("secondary_populations must be an array")
    secondary: list[tuple[str, str, str]] = []
    for item in secondary_raw:
        if not isinstance(item, Mapping):
            raise PilotExecutionError("secondary population must be an object")
        secondary.append(
            (
                str(item["population_id"]),
                str(item["version"]),
                _validate_sha256(str(item["population_hash"]), "population_hash"),
            )
        )

    binding = PilotExecutionBinding(
        binding_version=str(payload["binding_version"]),
        protocol_id=str(payload["protocol_id"]),
        protocol_version=str(payload["protocol_version"]),
        protocol_hash=_validate_sha256(str(payload["protocol_hash"]), "protocol_hash"),
        primary_population_id=str(primary["population_id"]),
        primary_population_version=str(primary["version"]),
        primary_population_hash=_validate_sha256(
            str(primary["population_hash"]), "primary_population_hash"
        ),
        secondary_population_hashes=tuple(secondary),
        gate6_report_hash=_validate_sha256(
            str(payload["gate6_report_hash"]), "gate6_report_hash"
        ),
        gate6_reference_strength_hash=_validate_sha256(
            str(payload["gate6_reference_strength_hash"]),
            "gate6_reference_strength_hash",
        ),
        git_commit=str(payload["git_commit"]),
        engine_version=str(payload["engine_version"]),
        python_version=str(payload["python_version"]),
        dependency_lockfile=str(payload["dependency_lockfile"]),
        poetry_lock_hash=_validate_sha256(
            str(payload["poetry_lock_hash"]), "poetry_lock_hash"
        ),
        cpu_model=str(payload["cpu_model"]),
        cpu_core_count=int(payload["cpu_core_count"]),
        cpu_architecture=str(payload["cpu_architecture"]),
        ram_bytes=int(payload["ram_bytes"]),
        os_identifier=str(payload["os_identifier"]),
        hardware_fingerprint=_validate_sha256(
            str(payload["hardware_fingerprint"]), "hardware_fingerprint"
        ),
        workers=int(payload["workers"]),
    )
    if binding.binding_hash != stored_hash:
        raise PilotExecutionError("Loaded pilot binding is not canonical")
    return binding


def _profile_parameters(profile_id: str) -> dict[str, Any]:
    profile = get_static_profile(ProfileName(profile_id))
    parameters = profile.parameters
    return {
        "schema_version": parameters.schema_version,
        "vpip": parameters.vpip,
        "pfr": parameters.pfr,
        "three_bet_freq": parameters.three_bet_freq,
        "cbet_freq": parameters.cbet_freq,
        "aggression": parameters.aggression,
        "fold_to_cbet": parameters.fold_to_cbet,
    }


def _opponent_version(protocol: Phase1PilotProtocol) -> str:
    return f"{protocol.static_profile_version}+{protocol.parametric_policy_version}"


def _rng_configuration(protocol: Phase1PilotProtocol) -> dict[str, Any]:
    return {
        "rng_contract": protocol.rng_contract,
        "crn_mode": protocol.crn_mode,
        "deal_namespace": protocol.deal_namespace,
        "bot_a_namespace": protocol.bot_a_namespace,
        "bot_b_namespace": protocol.bot_b_namespace,
        "agent_event_key": protocol.agent_event_key,
        "seed_schedule_version": protocol.seed_schedule_version,
    }


def build_job_experiment_config(
    *,
    protocol: Phase1PilotProtocol,
    protocol_payload: Mapping[str, Any],
    binding: PilotExecutionBinding,
    job: PilotJob,
    opponent_gate_map_hash: str,
    opponent_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete immutable per-job experiment configuration."""
    _validate_sha256(opponent_gate_map_hash, "opponent_gate_map_hash")
    run_provenance: dict[str, Any] = {
        "experiment_id": f"{protocol.protocol_id}:{binding.binding_hash}",
        "game_config_hash": protocol.game_config_hash,
        "bot_version": job.hero_version,
        "opponent_version": _opponent_version(protocol),
        "opponent_parameters": dict(opponent_parameters),
        "population_version": protocol.primary_population.version,
        "master_seed": job.master_seed,
        "rng_configuration": _rng_configuration(protocol),
        "engine_version": binding.engine_version,
        "python_version": binding.python_version,
        "poetry_lock_hash": binding.poetry_lock_hash,
        "git_commit": binding.git_commit,
        "cpu_model": binding.cpu_model,
        "cpu_core_count": binding.cpu_core_count,
        "cpu_architecture": binding.cpu_architecture,
        "os_identifier": binding.os_identifier,
        "statistical_unit": protocol.statistical_unit,
        "gate6_report_hash": binding.gate6_report_hash,
    }
    missing = set(protocol.required_run_provenance) - set(run_provenance)
    if missing:
        raise PilotExecutionError(
            "Per-job configuration is missing required provenance: "
            + ", ".join(sorted(missing))
        )

    return {
        "config_version": PHASE1_PILOT_EXPERIMENT_CONFIG_VERSION,
        "protocol": dict(protocol_payload),
        "execution_binding": binding.reference_payload(),
        "job": {
            "cell_id": job.cell_id,
            "hero_id": job.hero_id,
            "hero_version": job.hero_version,
            "opponent_profile_id": job.opponent_profile_id,
            "opponent_profile_version": job.opponent_profile_version,
            "opponent_gate_map_hash": opponent_gate_map_hash,
            "master_seed": job.master_seed,
            "paired_units": job.paired_units,
            "simulated_engine_hands": job.simulated_engine_hands,
        },
        "run_provenance": run_provenance,
        "scientific_use": {
            "variance_estimation": True,
            "runtime_estimation": True,
            "final_sample_size_planning": True,
            "confirmatory_inference": False,
            "final_benchmark_estimate": False,
        },
    }


def pilot_job_match_id(binding: PilotExecutionBinding, job: PilotJob) -> str:
    identity = {
        "protocol_hash": binding.protocol_hash,
        "binding_hash": binding.binding_hash,
        "cell_id": job.cell_id,
        "hero_id": job.hero_id,
        "opponent_profile_id": job.opponent_profile_id,
        "master_seed": job.master_seed,
    }
    return f"phase1-pilot-{sha256_canonical(identity)[:24]}"


@dataclass(frozen=True)
class PilotJobExecution:
    """Canonical checkpoint for one attempted frozen pilot job."""

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
    snapshot_version: str = PHASE1_PILOT_JOB_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.snapshot_version != PHASE1_PILOT_JOB_SNAPSHOT_VERSION:
            raise PilotExecutionError("Unsupported pilot job snapshot version")
        if self.status not in {"valid", "failed"}:
            raise PilotExecutionError("Pilot job status must be valid or failed")
        for field_name, value in (
            ("master_seed", self.master_seed),
            ("elapsed_ns", self.elapsed_ns),
            ("recorded_engine_hands", self.recorded_engine_hands),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise PilotExecutionError(f"{field_name} must be an integer")
        if self.elapsed_ns < 0 or self.recorded_engine_hands < 0:
            raise PilotExecutionError("Runtime and recorded hand count must be non-negative")
        if self.paired_difference_bb is not None and not math.isfinite(self.paired_difference_bb):
            raise PilotExecutionError("paired_difference_bb must be finite when present")
        _validate_sha256(self.experiment_config_hash, "experiment_config_hash")
        _validate_sha256(self.match_record_hash, "match_record_hash")
        _validate_sha256(self.binding_hash, "binding_hash")
        if self.status == "valid":
            if self.paired_difference_bb is None or self.failed_runs:
                raise PilotExecutionError("Valid pilot job must have one difference and no failures")
        else:
            if self.paired_difference_bb is not None or not self.failed_runs:
                raise PilotExecutionError("Failed pilot job must retain failure evidence")

    @property
    def completed(self) -> bool:
        return True

    @classmethod
    def from_runner_result(
        cls,
        *,
        job: PilotJob,
        binding: PilotExecutionBinding,
        result_paired_differences: Sequence[float],
        num_valid_pairs: int,
        failed_runs: Sequence[Mapping[str, Any]],
        elapsed_ns: int,
        match_record: MatchRecord,
    ) -> "PilotJobExecution":
        if num_valid_pairs == 1:
            if len(result_paired_differences) != 1 or failed_runs:
                raise PilotExecutionError("Valid one-unit pilot result is inconsistent")
            status = "valid"
            difference: float | None = float(result_paired_differences[0])
        elif num_valid_pairs == 0:
            if result_paired_differences or not failed_runs:
                raise PilotExecutionError("Failed one-unit pilot result is inconsistent")
            status = "failed"
            difference = None
        else:
            raise PilotExecutionError("Each frozen pilot job must contain exactly one paired unit")

        if match_record.match_id != pilot_job_match_id(binding, job):
            raise PilotExecutionError("MatchRecord match_id does not match pilot job identity")
        if match_record.master_seed != job.master_seed:
            raise PilotExecutionError("MatchRecord seed does not match pilot job")
        if match_record.experiment_config_hash is None:
            raise PilotExecutionError("Pilot MatchRecord requires experiment_config_hash")

        return cls(
            cell_id=job.cell_id,
            hero_id=job.hero_id,
            hero_version=job.hero_version,
            opponent_profile_id=job.opponent_profile_id,
            opponent_profile_version=job.opponent_profile_version,
            master_seed=job.master_seed,
            match_id=match_record.match_id,
            experiment_config_hash=match_record.experiment_config_hash,
            match_record_hash=match_record.record_hash,
            status=status,
            paired_difference_bb=difference,
            failed_runs=tuple(dict(item) for item in failed_runs),
            elapsed_ns=elapsed_ns,
            recorded_engine_hands=len(match_record.hands),
            binding_hash=binding.binding_hash,
            match_record_payload=match_record.to_dict(),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "binding_hash": self.binding_hash,
            "cell_id": self.cell_id,
            "hero_id": self.hero_id,
            "hero_version": self.hero_version,
            "opponent_profile_id": self.opponent_profile_id,
            "opponent_profile_version": self.opponent_profile_version,
            "master_seed": self.master_seed,
            "match_id": self.match_id,
            "experiment_config_hash": self.experiment_config_hash,
            "match_record_hash": self.match_record_hash,
            "status": self.status,
            "paired_difference_bb": self.paired_difference_bb,
            "failed_runs": [dict(item) for item in self.failed_runs],
            "elapsed_ns": self.elapsed_ns,
            "recorded_engine_hands": self.recorded_engine_hands,
            "match_record": (
                dict(self.match_record_payload)
                if self.match_record_payload is not None
                else None
            ),
        }

    @property
    def snapshot_hash(self) -> str:
        return sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot": self.payload(), "snapshot_hash": self.snapshot_hash}


def _execution_matches_job(execution: PilotJobExecution, job: PilotJob) -> bool:
    return (
        execution.cell_id == job.cell_id
        and execution.hero_id == job.hero_id
        and execution.hero_version == job.hero_version
        and execution.opponent_profile_id == job.opponent_profile_id
        and execution.opponent_profile_version == job.opponent_profile_version
        and execution.master_seed == job.master_seed
    )


def save_pilot_job_snapshot(execution: PilotJobExecution, path: str | Path) -> Path:
    if not isinstance(execution, PilotJobExecution):
        raise TypeError("execution must be a PilotJobExecution")
    if execution.match_record_payload is None:
        raise PilotExecutionError("Persisted pilot job snapshot requires MatchRecord payload")
    destination = Path(path)
    if destination.exists():
        existing = load_pilot_job_snapshot(
            destination,
            binding_hash=execution.binding_hash,
        )
        if existing.snapshot_hash != execution.snapshot_hash:
            raise PilotExecutionError("Existing pilot job snapshot does not match")
        return destination
    return _atomic_write_json(destination, execution.to_dict())


def load_pilot_job_snapshot(
    path: str | Path,
    *,
    binding: PilotExecutionBinding | None = None,
    binding_hash: str | None = None,
    expected_job: PilotJob | None = None,
) -> PilotJobExecution:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("snapshot"), Mapping):
        raise PilotExecutionError("Pilot job snapshot file is malformed")
    payload = raw["snapshot"]
    stored_hash = _validate_sha256(str(raw.get("snapshot_hash")), "snapshot_hash")
    if sha256_canonical(payload) != stored_hash:
        raise PilotExecutionError("snapshot_hash does not match snapshot payload")

    failures_raw = payload.get("failed_runs")
    if not isinstance(failures_raw, list):
        raise PilotExecutionError("failed_runs must be an array")
    failures: list[Mapping[str, Any]] = []
    for item in failures_raw:
        if not isinstance(item, Mapping):
            raise PilotExecutionError("failed_runs entries must be objects")
        failures.append(dict(item))

    record_payload = payload.get("match_record")
    if record_payload is not None and not isinstance(record_payload, Mapping):
        raise PilotExecutionError("match_record must be an object or null")

    execution = PilotJobExecution(
        snapshot_version=str(payload["snapshot_version"]),
        binding_hash=_validate_sha256(str(payload["binding_hash"]), "binding_hash"),
        cell_id=str(payload["cell_id"]),
        hero_id=str(payload["hero_id"]),
        hero_version=str(payload["hero_version"]),
        opponent_profile_id=str(payload["opponent_profile_id"]),
        opponent_profile_version=str(payload["opponent_profile_version"]),
        master_seed=int(payload["master_seed"]),
        match_id=str(payload["match_id"]),
        experiment_config_hash=_validate_sha256(
            str(payload["experiment_config_hash"]), "experiment_config_hash"
        ),
        match_record_hash=_validate_sha256(
            str(payload["match_record_hash"]), "match_record_hash"
        ),
        status=str(payload["status"]),
        paired_difference_bb=(
            None
            if payload.get("paired_difference_bb") is None
            else float(payload["paired_difference_bb"])
        ),
        failed_runs=tuple(failures),
        elapsed_ns=int(payload["elapsed_ns"]),
        recorded_engine_hands=int(payload["recorded_engine_hands"]),
        match_record_payload=(dict(record_payload) if record_payload is not None else None),
    )
    expected_binding_hash = binding.binding_hash if binding is not None else binding_hash
    if expected_binding_hash is not None and execution.binding_hash != expected_binding_hash:
        raise PilotExecutionError("Pilot job snapshot binding does not match campaign")
    if expected_job is not None and not _execution_matches_job(execution, expected_job):
        raise PilotExecutionError("Pilot job snapshot identity does not match expected job")
    if execution.match_record_payload is not None:
        record = MatchRecord.from_dict(execution.match_record_payload)
        if record.record_hash != execution.match_record_hash:
            raise PilotExecutionError("match_record_hash does not match embedded MatchRecord")
        if record.match_id != execution.match_id:
            raise PilotExecutionError("Embedded MatchRecord match_id mismatch")
        if record.experiment_config_hash != execution.experiment_config_hash:
            raise PilotExecutionError("Embedded MatchRecord experiment config mismatch")
        if len(record.hands) != execution.recorded_engine_hands:
            raise PilotExecutionError("Embedded MatchRecord hand count mismatch")
    if binding is not None and expected_job is not None:
        if execution.match_id != pilot_job_match_id(binding, expected_job):
            raise PilotExecutionError("Pilot job snapshot match_id does not match expected identity")
    return execution


@dataclass(frozen=True)
class PilotCellExecutionReport:
    cell_id: str
    hero_id: str
    hero_version: str
    opponent_profile_id: str
    opponent_profile_version: str
    requested_units: int
    completed_units: int
    valid_units: int
    failed_units: int
    requested_engine_hands: int
    recorded_engine_hands: int
    std_difference_bb100: float | None
    variance_estimator: str
    total_runtime_ns: int
    median_runtime_ns: float | None

    def __post_init__(self) -> None:
        counts = (
            self.requested_units,
            self.completed_units,
            self.valid_units,
            self.failed_units,
            self.requested_engine_hands,
            self.recorded_engine_hands,
            self.total_runtime_ns,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise PilotExecutionError("Pilot cell counts/runtime must be non-negative integers")
        if self.completed_units > self.requested_units:
            raise PilotExecutionError("Pilot cell completed units exceed requested units")
        if self.valid_units + self.failed_units != self.completed_units:
            raise PilotExecutionError("Pilot cell valid/failed counts are inconsistent")
        if self.std_difference_bb100 is None:
            if self.valid_units >= 2:
                raise PilotExecutionError("Pilot cell with >=2 valid units requires a variance estimate")
        elif (
            self.valid_units < 2
            or not math.isfinite(self.std_difference_bb100)
            or self.std_difference_bb100 < 0.0
        ):
            raise PilotExecutionError("Pilot cell variance estimate is inconsistent")
        if self.median_runtime_ns is None:
            if self.completed_units != 0:
                raise PilotExecutionError("Completed pilot cell requires median runtime")
        elif not math.isfinite(self.median_runtime_ns) or self.median_runtime_ns < 0.0:
            raise PilotExecutionError("Pilot cell median runtime must be finite and non-negative")

    def payload(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "hero_id": self.hero_id,
            "hero_version": self.hero_version,
            "opponent_profile_id": self.opponent_profile_id,
            "opponent_profile_version": self.opponent_profile_version,
            "requested_units": self.requested_units,
            "completed_units": self.completed_units,
            "valid_units": self.valid_units,
            "failed_units": self.failed_units,
            "requested_engine_hands": self.requested_engine_hands,
            "recorded_engine_hands": self.recorded_engine_hands,
            "std_difference_bb100": self.std_difference_bb100,
            "variance_estimator": self.variance_estimator,
            "total_runtime_ns": self.total_runtime_ns,
            "median_runtime_ns": self.median_runtime_ns,
        }


@dataclass(frozen=True)
class Phase1PilotExecutionReport:
    protocol_id: str
    protocol_version: str
    protocol_hash: str
    binding_hash: str
    gate6_report_hash: str
    workers: int
    requested_jobs: int
    completed_jobs: int
    valid_units: int
    failed_units: int
    requested_engine_hands: int
    recorded_engine_hands: int
    seeds: tuple[int, ...]
    cells: tuple[PilotCellExecutionReport, ...]
    complete: bool
    failure_free: bool
    report_version: str = PHASE1_PILOT_EXECUTION_REPORT_VERSION

    def __post_init__(self) -> None:
        if self.report_version != PHASE1_PILOT_EXECUTION_REPORT_VERSION:
            raise PilotExecutionError("Unsupported Phase-1 pilot execution report version")
        _validate_sha256(self.protocol_hash, "protocol_hash")
        _validate_sha256(self.binding_hash, "binding_hash")
        _validate_sha256(self.gate6_report_hash, "gate6_report_hash")
        counts = (
            self.workers,
            self.requested_jobs,
            self.completed_jobs,
            self.valid_units,
            self.failed_units,
            self.requested_engine_hands,
            self.recorded_engine_hands,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise PilotExecutionError("Pilot report counts must be integers")
        if self.workers <= 0 or any(value < 0 for value in counts[1:]):
            raise PilotExecutionError("Pilot report workers/counts are invalid")
        if self.completed_jobs > self.requested_jobs:
            raise PilotExecutionError("Pilot report completed_jobs exceeds requested_jobs")
        if self.valid_units + self.failed_units != self.completed_jobs:
            raise PilotExecutionError("Pilot report valid/failed counts are inconsistent")
        if self.complete != (self.completed_jobs == self.requested_jobs):
            raise PilotExecutionError("Pilot report complete flag is inconsistent")
        if self.failure_free != (self.complete and self.failed_units == 0):
            raise PilotExecutionError("Pilot report failure_free flag is inconsistent")
        if len(set(self.seeds)) != len(self.seeds):
            raise PilotExecutionError("Pilot report seeds must be unique")
        if sum(cell.requested_units for cell in self.cells) != self.requested_jobs:
            raise PilotExecutionError("Pilot report cell requested counts are inconsistent")
        if sum(cell.completed_units for cell in self.cells) != self.completed_jobs:
            raise PilotExecutionError("Pilot report cell completed counts are inconsistent")
        if sum(cell.valid_units for cell in self.cells) != self.valid_units:
            raise PilotExecutionError("Pilot report cell valid counts are inconsistent")
        if sum(cell.failed_units for cell in self.cells) != self.failed_units:
            raise PilotExecutionError("Pilot report cell failed counts are inconsistent")

    def payload(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "binding_hash": self.binding_hash,
            "gate6_report_hash": self.gate6_report_hash,
            "workers": self.workers,
            "requested_jobs": self.requested_jobs,
            "completed_jobs": self.completed_jobs,
            "valid_units": self.valid_units,
            "failed_units": self.failed_units,
            "requested_engine_hands": self.requested_engine_hands,
            "recorded_engine_hands": self.recorded_engine_hands,
            "seeds": list(self.seeds),
            "cells": [cell.payload() for cell in self.cells],
            "complete": self.complete,
            "failure_free": self.failure_free,
            "scientific_use": {
                "variance_estimation": True,
                "runtime_estimation": True,
                "final_sample_size_planning": True,
                "contains_confirmatory_inference": False,
                "contains_final_benchmark_estimate": False,
                "optional_stopping": False,
            },
        }

    @property
    def report_hash(self) -> str:
        return sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"report": self.payload(), "report_hash": self.report_hash}


def build_pilot_report(
    *,
    protocol: Phase1PilotProtocol,
    binding: PilotExecutionBinding,
    jobs: Sequence[PilotJob],
    executions: Sequence[PilotJobExecution],
) -> Phase1PilotExecutionReport:
    """Summarize only variance/runtime evidence; never compute a pilot EV estimate."""
    job_by_key: dict[tuple[str, int], PilotJob] = {}
    for job in jobs:
        key = (job.cell_id, job.master_seed)
        if key in job_by_key:
            raise PilotExecutionError("Pilot jobs must have unique cell/seed identities")
        job_by_key[key] = job

    execution_by_key: dict[tuple[str, int], PilotJobExecution] = {}
    for execution in executions:
        key = (execution.cell_id, execution.master_seed)
        if key in execution_by_key:
            raise PilotExecutionError("Pilot execution evidence contains duplicate jobs")
        expected = job_by_key.get(key)
        if expected is None or not _execution_matches_job(execution, expected):
            raise PilotExecutionError("Pilot execution evidence contains an unknown job")
        if execution.binding_hash != binding.binding_hash:
            raise PilotExecutionError("Pilot execution evidence uses a different binding")
        execution_by_key[key] = execution

    present_cell_ids = {job.cell_id for job in jobs}
    cell_ids = [
        f"{hero.agent_id}-vs-{profile_id}"
        for hero in protocol.hero_agents
        for profile_id in protocol.opponent_profile_ids
        if f"{hero.agent_id}-vs-{profile_id}" in present_cell_ids
    ]
    cells: list[PilotCellExecutionReport] = []
    for cell_id in cell_ids:
        cell_jobs = sorted(
            (job for job in jobs if job.cell_id == cell_id),
            key=lambda item: item.master_seed,
        )
        cell_executions = [
            execution_by_key[(job.cell_id, job.master_seed)]
            for job in cell_jobs
            if (job.cell_id, job.master_seed) in execution_by_key
        ]
        valid = [item for item in cell_executions if item.status == "valid"]
        failures = [item for item in cell_executions if item.status == "failed"]
        differences = [
            float(item.paired_difference_bb)
            for item in valid
            if item.paired_difference_bb is not None
        ]
        std_bb100 = (
            100.0 * statistics.stdev(differences)
            if len(differences) >= 2
            else None
        )
        runtimes = [item.elapsed_ns for item in cell_executions]
        first_job = cell_jobs[0]
        cells.append(
            PilotCellExecutionReport(
                cell_id=cell_id,
                hero_id=first_job.hero_id,
                hero_version=first_job.hero_version,
                opponent_profile_id=first_job.opponent_profile_id,
                opponent_profile_version=first_job.opponent_profile_version,
                requested_units=len(cell_jobs),
                completed_units=len(cell_executions),
                valid_units=len(valid),
                failed_units=len(failures),
                requested_engine_hands=sum(item.simulated_engine_hands for item in cell_jobs),
                recorded_engine_hands=sum(item.recorded_engine_hands for item in cell_executions),
                std_difference_bb100=std_bb100,
                variance_estimator=protocol.variance_estimator,
                total_runtime_ns=sum(runtimes),
                median_runtime_ns=(float(statistics.median(runtimes)) if runtimes else None),
            )
        )

    requested_jobs = len(job_by_key)
    completed_jobs = len(execution_by_key)
    valid_units = sum(cell.valid_units for cell in cells)
    failed_units = sum(cell.failed_units for cell in cells)
    return Phase1PilotExecutionReport(
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_hash=protocol.protocol_hash,
        binding_hash=binding.binding_hash,
        gate6_report_hash=binding.gate6_report_hash,
        workers=binding.workers,
        requested_jobs=requested_jobs,
        completed_jobs=completed_jobs,
        valid_units=valid_units,
        failed_units=failed_units,
        requested_engine_hands=sum(item.simulated_engine_hands for item in jobs),
        recorded_engine_hands=sum(item.recorded_engine_hands for item in executions),
        seeds=tuple(sorted({job.master_seed for job in jobs})),
        cells=tuple(cells),
        complete=completed_jobs == requested_jobs,
        failure_free=completed_jobs == requested_jobs and failed_units == 0,
    )


def save_phase1_pilot_execution_report(
    report: Phase1PilotExecutionReport,
    path: str | Path,
) -> Path:
    if not isinstance(report, Phase1PilotExecutionReport):
        raise TypeError("report must be a Phase1PilotExecutionReport")
    destination = Path(path)
    if destination.exists():
        existing = load_phase1_pilot_execution_report(destination)
        if existing.report_hash != report.report_hash:
            raise PilotExecutionError("Existing pilot report does not match current evidence")
        return destination
    return _atomic_write_json(destination, report.to_dict())


def load_phase1_pilot_execution_report(path: str | Path) -> Phase1PilotExecutionReport:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("report"), Mapping):
        raise PilotExecutionError("Pilot execution report file is malformed")
    payload = raw["report"]
    stored_hash = _validate_sha256(str(raw.get("report_hash")), "report_hash")
    if sha256_canonical(payload) != stored_hash:
        raise PilotExecutionError("report_hash does not match report payload")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        raise PilotExecutionError("Pilot report cells must be an array")
    cells = tuple(
        PilotCellExecutionReport(
            cell_id=str(item["cell_id"]),
            hero_id=str(item["hero_id"]),
            hero_version=str(item["hero_version"]),
            opponent_profile_id=str(item["opponent_profile_id"]),
            opponent_profile_version=str(item["opponent_profile_version"]),
            requested_units=int(item["requested_units"]),
            completed_units=int(item["completed_units"]),
            valid_units=int(item["valid_units"]),
            failed_units=int(item["failed_units"]),
            requested_engine_hands=int(item["requested_engine_hands"]),
            recorded_engine_hands=int(item["recorded_engine_hands"]),
            std_difference_bb100=(
                None
                if item.get("std_difference_bb100") is None
                else float(item["std_difference_bb100"])
            ),
            variance_estimator=str(item["variance_estimator"]),
            total_runtime_ns=int(item["total_runtime_ns"]),
            median_runtime_ns=(
                None
                if item.get("median_runtime_ns") is None
                else float(item["median_runtime_ns"])
            ),
        )
        for item in raw_cells
    )
    report = Phase1PilotExecutionReport(
        report_version=str(payload["report_version"]),
        protocol_id=str(payload["protocol_id"]),
        protocol_version=str(payload["protocol_version"]),
        protocol_hash=_validate_sha256(str(payload["protocol_hash"]), "protocol_hash"),
        binding_hash=_validate_sha256(str(payload["binding_hash"]), "binding_hash"),
        gate6_report_hash=_validate_sha256(
            str(payload["gate6_report_hash"]), "gate6_report_hash"
        ),
        workers=int(payload["workers"]),
        requested_jobs=int(payload["requested_jobs"]),
        completed_jobs=int(payload["completed_jobs"]),
        valid_units=int(payload["valid_units"]),
        failed_units=int(payload["failed_units"]),
        requested_engine_hands=int(payload["requested_engine_hands"]),
        recorded_engine_hands=int(payload["recorded_engine_hands"]),
        seeds=tuple(int(seed) for seed in payload["seeds"]),
        cells=cells,
        complete=bool(payload["complete"]),
        failure_free=bool(payload["failure_free"]),
    )
    if report.report_hash != stored_hash:
        raise PilotExecutionError("Loaded pilot report is not canonical")
    scientific_use = payload.get("scientific_use")
    if not isinstance(scientific_use, Mapping):
        raise PilotExecutionError("Pilot report scientific_use block is missing")
    if scientific_use.get("contains_confirmatory_inference") is not False:
        raise PilotExecutionError("Pilot report must not contain confirmatory inference")
    if scientific_use.get("contains_final_benchmark_estimate") is not False:
        raise PilotExecutionError("Pilot report must not contain a final benchmark estimate")
    expected_use = {
        "variance_estimation": True,
        "runtime_estimation": True,
        "final_sample_size_planning": True,
        "contains_confirmatory_inference": False,
        "contains_final_benchmark_estimate": False,
        "optional_stopping": False,
    }
    if dict(scientific_use) != expected_use:
        raise PilotExecutionError("Pilot report scientific_use contract mismatch")
    return report


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
        raise PilotExecutionError(
            "Scientific source/configuration paths must be clean and committed before pilot binding"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not commit:
        raise PilotExecutionError("Could not resolve git commit")
    return commit


def _gate_map_hash(context: VerifiedStaticProfileContext, profile_id: str) -> str:
    acceptance = next(
        (
            item
            for item in context.gate6_report.gate6_result.results
            if item.profile_name == profile_id
        ),
        None,
    )
    if acceptance is None:
        raise PilotExecutionError(f"Gate-6 report is missing profile {profile_id}")
    return acceptance.gate_map_hash


def _build_hero_agent(hero_id: str) -> Agent:
    if hero_id == "random":
        return RandomBaselineAgent()
    if hero_id == "hand_strength":
        return HandStrengthBaselineAgent()
    raise PilotExecutionError(f"Unknown frozen pilot hero: {hero_id}")


_WORKER_CONTEXTS: dict[tuple[str, str], VerifiedStaticProfileContext] = {}


def _worker_verified_context(snapshot_root: str, gate6_report_path: str) -> VerifiedStaticProfileContext:
    key = (snapshot_root, gate6_report_path)
    context = _WORKER_CONTEXTS.get(key)
    if context is None:
        context = load_verified_static_profile_context(
            snapshot_root=Path(snapshot_root),
            gate6_report_path=Path(gate6_report_path),
        )
        _WORKER_CONTEXTS[key] = context
    return context


@dataclass(frozen=True)
class _WorkerRequest:
    job: PilotJob
    binding: PilotExecutionBinding
    snapshot_root: str
    gate6_report_path: str
    experiment_config_hash: str


def _execute_worker_request(request: _WorkerRequest) -> PilotJobExecution:
    context = _worker_verified_context(request.snapshot_root, request.gate6_report_path)
    hero = _build_hero_agent(request.job.hero_id)
    opponent = build_verified_static_profile_agent(
        request.job.opponent_profile_id,
        context=context,
    )
    match_id = pilot_job_match_id(request.binding, request.job)
    experiment_id = f"{request.binding.protocol_id}:{request.binding.binding_hash}"
    runner = PairedRunner(
        agent_a=hero,
        agent_b=opponent,
        num_hands=1,
        seed=request.job.master_seed,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
        illegal_action_policy=IllegalActionPolicy.EXCLUDE_RUN,
        match_id=match_id,
        experiment_metadata={
            "experiment_id": experiment_id,
            "experiment_config_hash": request.experiment_config_hash,
            "phase1_pilot_protocol_hash": request.binding.protocol_hash,
            "phase1_pilot_binding_hash": request.binding.binding_hash,
            "pilot_cell_id": request.job.cell_id,
        },
    )
    start_ns = time.perf_counter_ns()
    result = runner.run()
    elapsed_ns = time.perf_counter_ns() - start_ns
    if result.statistical_unit is not StatisticalUnit.PAIRED_DEAL:
        raise PilotExecutionError("Frozen Phase-1 pilot worker produced wrong statistical unit")
    if result.match_record is None:
        raise PilotExecutionError("Frozen Phase-1 pilot worker produced no MatchRecord")
    return PilotJobExecution.from_runner_result(
        job=request.job,
        binding=request.binding,
        result_paired_differences=result.paired_differences,
        num_valid_pairs=result.num_valid_pairs,
        failed_runs=result.failed_runs,
        elapsed_ns=elapsed_ns,
        match_record=result.match_record,
    )


def _snapshot_path(work_root: Path, job: PilotJob) -> Path:
    return work_root / "jobs" / job.cell_id / f"seed-{job.master_seed}.json"


def _match_metadata(
    *,
    protocol: Phase1PilotProtocol,
    binding: PilotExecutionBinding,
    job: PilotJob,
    experiment_config: Mapping[str, Any],
    opponent_gate_map_hash: str,
    opponent_parameters: Mapping[str, Any],
) -> MatchPersistenceMetadata:
    return MatchPersistenceMetadata(
        bot_version=job.hero_version,
        opponent_hash=opponent_gate_map_hash,
        engine_version=binding.engine_version,
        python_version=binding.python_version,
        rng_version=protocol.rng_contract,
        git_commit=binding.git_commit,
        population_id=protocol.primary_population.population_id,
        population_seed=None,
        opponent_parameters=dict(opponent_parameters),
        bot_configuration={
            "agent_id": job.hero_id,
            "agent_version": job.hero_version,
            "pilot_binding_hash": binding.binding_hash,
            "gate6_report_hash": binding.gate6_report_hash,
        },
        statistical_method=protocol.variance_estimator,
        sample_size=1,
        precision_target=None,
        experiment_config=dict(experiment_config),
        poetry_lock_hash=binding.poetry_lock_hash,
        hardware_fingerprint=binding.hardware_fingerprint,
        os_identifier=binding.os_identifier,
    )


def _ensure_persisted_match(
    *,
    store: SQLiteBenchmarkStore,
    existing_match_ids: set[str],
    execution: PilotJobExecution,
    expected_metadata: MatchPersistenceMetadata,
) -> None:
    if execution.match_record_payload is None:
        raise PilotExecutionError("Pilot checkpoint cannot recover SQLite without MatchRecord")
    record = MatchRecord.from_dict(execution.match_record_payload)
    if record.record_hash != execution.match_record_hash:
        raise PilotExecutionError("Pilot checkpoint MatchRecord hash mismatch")
    if execution.match_id in existing_match_ids:
        persisted = store.load_match(execution.match_id)
        if persisted.record_hash != execution.match_record_hash:
            raise PilotExecutionError("Persisted SQLite MatchRecord does not match pilot checkpoint")
        metadata = store.load_match_metadata(execution.match_id)
        if metadata != expected_metadata:
            raise PilotExecutionError("Persisted SQLite metadata does not match pilot binding/job")
        return
    store.save_match(record, metadata=expected_metadata)
    existing_match_ids.add(execution.match_id)


def _load_protocol_payload(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("protocol"), Mapping):
        raise PilotExecutionError("Pilot protocol file is malformed")
    return dict(raw["protocol"])


def _validated_protocol(
    *,
    protocol_path: Path,
    primary_population_path: Path,
    secondary_population_path: Path,
) -> Phase1PilotProtocol:
    primary = load_frozen_population(primary_population_path)
    secondary = load_frozen_population(secondary_population_path)
    populations = {
        primary.population_id: primary,
        secondary.population_id: secondary,
    }
    return load_phase1_pilot_protocol(protocol_path, populations=populations)


def execute_phase1_pilot(
    *,
    protocol_path: Path,
    primary_population_path: Path,
    secondary_population_path: Path,
    snapshot_root: Path,
    gate6_report_path: Path,
    database: Path,
    work_root: Path,
    report_path: Path,
    repo_root: Path,
    poetry_lock_path: Path,
    ram_bytes: int,
    workers: int = 1,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> Phase1PilotExecutionReport:
    """Execute or resume the exact frozen Phase-1 pilot matrix."""
    if isinstance(ram_bytes, bool) or not isinstance(ram_bytes, int) or ram_bytes <= 0:
        raise ValueError("ram_bytes must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")

    protocol_path = Path(protocol_path)
    primary_population_path = Path(primary_population_path)
    secondary_population_path = Path(secondary_population_path)
    protocol = _validated_protocol(
        protocol_path=protocol_path,
        primary_population_path=primary_population_path,
        secondary_population_path=secondary_population_path,
    )
    protocol_payload = _load_protocol_payload(protocol_path)
    jobs = iter_pilot_jobs(protocol)

    context = load_verified_static_profile_context(
        snapshot_root=Path(snapshot_root),
        gate6_report_path=Path(gate6_report_path),
    )
    for profile_id in protocol.opponent_profile_ids:
        build_verified_static_profile_agent(profile_id, context=context)

    git_commit = _clean_git_commit(Path(repo_root))
    environment = capture_execution_environment(
        engine_version=f"git:{git_commit}",
        git_commit=git_commit,
        poetry_lock_path=Path(poetry_lock_path),
        ram_bytes=ram_bytes,
    )
    binding = build_pilot_execution_binding(
        protocol=protocol,
        gate6_report_hash=context.gate6_report.report_hash,
        gate6_reference_strength_hash=context.reference.reference_strength_hash,
        environment=environment,
        workers=workers,
    )

    work_root = Path(work_root)
    binding_path = work_root / "binding.json"
    save_pilot_execution_binding(binding, binding_path)

    store = SQLiteBenchmarkStore(database)
    store.initialize()
    existing_match_ids = set(store.list_match_ids())

    execution_by_key: dict[tuple[str, int], PilotJobExecution] = {}
    missing: list[tuple[PilotJob, Mapping[str, Any], MatchPersistenceMetadata]] = []

    for job in jobs:
        gate_map_hash = _gate_map_hash(context, job.opponent_profile_id)
        parameters = _profile_parameters(job.opponent_profile_id)
        experiment_config = build_job_experiment_config(
            protocol=protocol,
            protocol_payload=protocol_payload,
            binding=binding,
            job=job,
            opponent_gate_map_hash=gate_map_hash,
            opponent_parameters=parameters,
        )
        config_hash = sha256_canonical(experiment_config)
        metadata = _match_metadata(
            protocol=protocol,
            binding=binding,
            job=job,
            experiment_config=experiment_config,
            opponent_gate_map_hash=gate_map_hash,
            opponent_parameters=parameters,
        )
        snapshot_path = _snapshot_path(work_root, job)
        if snapshot_path.exists():
            execution = load_pilot_job_snapshot(
                snapshot_path,
                binding=binding,
                expected_job=job,
            )
            if execution.experiment_config_hash != config_hash:
                raise PilotExecutionError("Pilot checkpoint experiment config hash mismatch")
            _ensure_persisted_match(
                store=store,
                existing_match_ids=existing_match_ids,
                execution=execution,
                expected_metadata=metadata,
            )
            execution_by_key[(job.cell_id, job.master_seed)] = execution
            continue

        match_id = pilot_job_match_id(binding, job)
        if match_id in existing_match_ids:
            raise PilotExecutionError(
                "SQLite contains a pilot match without its runtime checkpoint; "
                "refusing to invent missing runtime evidence"
            )
        missing.append((job, experiment_config, metadata))

    resumed = len(execution_by_key)
    total = len(jobs)
    if progress_callback is not None:
        progress_callback(resumed, total, resumed, sum(item.status == "failed" for item in execution_by_key.values()))

    def persist_execution(
        job: PilotJob,
        execution: PilotJobExecution,
        metadata: MatchPersistenceMetadata,
    ) -> None:
        snapshot_path = _snapshot_path(work_root, job)
        save_pilot_job_snapshot(execution, snapshot_path)
        _ensure_persisted_match(
            store=store,
            existing_match_ids=existing_match_ids,
            execution=execution,
            expected_metadata=metadata,
        )
        execution_by_key[(job.cell_id, job.master_seed)] = execution
        if progress_callback is not None:
            progress_callback(
                len(execution_by_key),
                total,
                resumed,
                sum(item.status == "failed" for item in execution_by_key.values()),
            )

    requests = [
        (
            job,
            _WorkerRequest(
                job=job,
                binding=binding,
                snapshot_root=str(Path(snapshot_root)),
                gate6_report_path=str(Path(gate6_report_path)),
                experiment_config_hash=sha256_canonical(config),
            ),
            metadata,
        )
        for job, config, metadata in missing
    ]

    if workers == 1:
        for job, request, metadata in requests:
            execution = _execute_worker_request(request)
            persist_execution(job, execution, metadata)
    elif requests:
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(_execute_worker_request, request): (job, metadata)
            for job, request, metadata in requests
        }
        try:
            for future in as_completed(futures):
                job, metadata = futures[future]
                execution = future.result()
                persist_execution(job, execution, metadata)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    executions = tuple(execution_by_key[key] for key in sorted(execution_by_key))
    report = build_pilot_report(
        protocol=protocol,
        binding=binding,
        jobs=jobs,
        executions=executions,
    )
    if not report.complete:
        raise PilotExecutionError("Pilot execution ended without all frozen jobs checkpointed")
    save_phase1_pilot_execution_report(report, report_path)
    return report


def render_phase1_pilot_execution_report_text(report: Phase1PilotExecutionReport) -> str:
    lines = [
        "Phase-1 pilot execution report",
        f"  status: {'COMPLETE' if report.complete else 'INCOMPLETE'}",
        f"  failure free: {'yes' if report.failure_free else 'no'}",
        f"  protocol: {report.protocol_id} ({report.protocol_hash})",
        f"  binding hash: {report.binding_hash}",
        f"  Gate-6 report hash: {report.gate6_report_hash}",
        f"  workers: {report.workers}",
        f"  completed jobs: {report.completed_jobs}/{report.requested_jobs}",
        f"  valid paired units: {report.valid_units}",
        f"  failed paired units: {report.failed_units}",
        f"  requested engine hands: {report.requested_engine_hands}",
        f"  recorded engine hands: {report.recorded_engine_hands}",
        "  cell variance estimates (BB/100 SD):",
    ]
    for cell in report.cells:
        variance = (
            "N/A"
            if cell.std_difference_bb100 is None
            else f"{cell.std_difference_bb100:.6f}"
        )
        lines.append(
            f"    {cell.cell_id}: sigma_d={variance}, "
            f"valid={cell.valid_units}/{cell.requested_units}, "
            f"failed={cell.failed_units}, runtime_ns={cell.total_runtime_ns}"
        )
    lines.extend(
        [
            "  confirmatory inference: prohibited",
            "  final benchmark estimate: prohibited",
            f"  report hash: {report.report_hash}",
        ]
    )
    return "\n".join(lines) + "\n"
