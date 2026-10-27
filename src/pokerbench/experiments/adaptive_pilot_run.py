"""Resumable orchestration for the frozen Phase-2 adaptive pilot matrix."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pokerbench.experiments.adaptive_pilot_binding import (
    AdaptivePilotBindingError,
    AdaptivePilotExecutionBinding,
    AdaptivePilotTrajectoryConfig,
    build_adaptive_pilot_trajectory_config,
    load_adaptive_pilot_execution_binding,
    load_adaptive_pilot_trajectory_config,
    save_adaptive_pilot_execution_binding,
    save_adaptive_pilot_trajectory_config,
)
from pokerbench.experiments.adaptive_pilot_execution import (
    AdaptivePilotPairExecution,
    finalize_adaptive_pilot_pair,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotJob,
    AdaptivePilotProtocol,
    iter_adaptive_pilot_jobs,
)
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import PairedResult, StatisticalUnit


ADAPTIVE_PILOT_RUN_MANIFEST_VERSION = "phase2-adaptive-pilot-run-manifest-v1"
ADAPTIVE_PILOT_ARM_SNAPSHOT_VERSION = "phase2-adaptive-pilot-arm-snapshot-v1"
ADAPTIVE_PILOT_PAIR_CHECKPOINT_VERSION = "phase2-adaptive-pilot-pair-checkpoint-v1"

_TERMINAL_STATUSES = frozenset({"completed", "failed"})
_ALL_STATUSES = frozenset({"planned", "running", *_TERMINAL_STATUSES})


class AdaptivePilotRunError(ValueError):
    """Raised when resumable pilot execution evidence is inconsistent."""


@runtime_checkable
class AdaptivePilotRunStore(Protocol):
    """Durable store capabilities required by the resumable pilot runner."""

    def save_match(self, record: MatchRecord) -> None: ...

    def load_match(self, match_id: str) -> MatchRecord: ...

    def list_match_ids(self) -> tuple[str, ...]: ...


class AdaptivePilotJobExecutor(Protocol):
    """Execute one already-bound frozen trajectory job."""

    def __call__(
        self,
        job: AdaptivePilotJob,
        config: AdaptivePilotTrajectoryConfig,
        match_id: str,
        store: AdaptivePilotRunStore,
    ) -> PairedResult: ...


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _canonical_file_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AdaptivePilotRunError(f"{field_name} must be a lowercase SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AdaptivePilotRunError(
            f"{field_name} must be a lowercase SHA-256 hex digest"
        ) from exc
    if value.lower() != value:
        raise AdaptivePilotRunError(f"{field_name} must be lowercase")
    return value


def _atomic_write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_file_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _require_canonical_file(path: Path, parsed: object, what: str) -> None:
    if path.read_bytes() != _canonical_file_bytes(parsed):
        raise AdaptivePilotRunError(f"{what} file is not canonical JSON with one trailing LF")


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdaptivePilotRunError(f"{field_name} must be a positive integer")
    return value


def adaptive_pilot_job_key(job: AdaptivePilotJob) -> str:
    """Return the stable human-readable identity for one frozen arm job."""
    if not isinstance(job, AdaptivePilotJob):
        raise TypeError("job must be an AdaptivePilotJob")
    return f"{job.arm_id}__{job.opponent_profile_id}__{job.master_seed}"


def _pair_key(profile_id: str, master_seed: int) -> str:
    return f"{profile_id}__{master_seed}"


def adaptive_pilot_match_id(
    *,
    binding: AdaptivePilotExecutionBinding,
    trajectory_config_hash: str,
) -> str:
    """Derive the immutable match ID from execution binding and trajectory config."""
    if not isinstance(binding, AdaptivePilotExecutionBinding):
        raise TypeError("binding must be an AdaptivePilotExecutionBinding")
    _validate_sha256(trajectory_config_hash, "trajectory_config_hash")
    material = {
        "binding_hash": binding.binding_hash,
        "trajectory_config_hash": trajectory_config_hash,
    }
    return f"phase2-adaptive-pilot-{_sha256_canonical(material)[:24]}"


@dataclass(frozen=True)
class AdaptivePilotManifestJobState:
    """One immutable plan identity plus mutable-on-disk execution status."""

    job_key: str
    arm_id: str
    opponent_profile_id: str
    master_seed: int
    hands_per_position: int
    simulated_engine_hands: int
    trajectory_config_hash: str
    match_id: str
    status: str = "planned"
    arm_snapshot_hash: str | None = None
    match_record_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.job_key or not self.arm_id or not self.opponent_profile_id:
            raise AdaptivePilotRunError("manifest job identity must be non-empty")
        _positive_int(self.master_seed, "master_seed")
        _positive_int(self.hands_per_position, "hands_per_position")
        _positive_int(self.simulated_engine_hands, "simulated_engine_hands")
        _validate_sha256(self.trajectory_config_hash, "trajectory_config_hash")
        if not self.match_id:
            raise AdaptivePilotRunError("match_id must not be empty")
        if self.status not in _ALL_STATUSES:
            raise AdaptivePilotRunError("manifest job status is unsupported")
        if self.status in _TERMINAL_STATUSES:
            if self.arm_snapshot_hash is None or self.match_record_hash is None:
                raise AdaptivePilotRunError("terminal manifest state requires snapshot and record hashes")
            _validate_sha256(self.arm_snapshot_hash, "arm_snapshot_hash")
            _validate_sha256(self.match_record_hash, "match_record_hash")
        elif self.arm_snapshot_hash is not None or self.match_record_hash is not None:
            raise AdaptivePilotRunError("non-terminal manifest state cannot retain terminal hashes")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "job_key": self.job_key,
            "arm_id": self.arm_id,
            "opponent_profile_id": self.opponent_profile_id,
            "master_seed": self.master_seed,
            "hands_per_position": self.hands_per_position,
            "simulated_engine_hands": self.simulated_engine_hands,
            "trajectory_config_hash": self.trajectory_config_hash,
            "match_id": self.match_id,
        }

    def payload(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "status": self.status,
            "arm_snapshot_hash": self.arm_snapshot_hash,
            "match_record_hash": self.match_record_hash,
        }


@dataclass(frozen=True)
class AdaptivePilotRunManifest:
    """Hash-bound canonical status manifest for all 50 frozen arm jobs."""

    protocol_hash: str
    binding_hash: str
    jobs: tuple[AdaptivePilotManifestJobState, ...]
    manifest_version: str = ADAPTIVE_PILOT_RUN_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.manifest_version != ADAPTIVE_PILOT_RUN_MANIFEST_VERSION:
            raise AdaptivePilotRunError("unsupported adaptive-pilot run manifest version")
        _validate_sha256(self.protocol_hash, "protocol_hash")
        _validate_sha256(self.binding_hash, "binding_hash")
        if len(self.jobs) != 50:
            raise AdaptivePilotRunError("run manifest must contain the canonical 50-job plan")
        if len({item.job_key for item in self.jobs}) != len(self.jobs):
            raise AdaptivePilotRunError("run manifest contains duplicate job keys")
        if len({item.match_id for item in self.jobs}) != len(self.jobs):
            raise AdaptivePilotRunError("run manifest contains duplicate match IDs")

    @property
    def planned_job_count(self) -> int:
        return len(self.jobs)

    @property
    def planned_pair_count(self) -> int:
        return len({(item.opponent_profile_id, item.master_seed) for item in self.jobs})

    @property
    def planned_engine_hands(self) -> int:
        return sum(item.simulated_engine_hands for item in self.jobs)

    @property
    def terminal_jobs(self) -> int:
        return sum(item.status in _TERMINAL_STATUSES for item in self.jobs)

    def payload(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "protocol_hash": self.protocol_hash,
            "binding_hash": self.binding_hash,
            "jobs": [item.payload() for item in self.jobs],
        }

    @property
    def manifest_hash(self) -> str:
        return _sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"manifest": self.payload(), "manifest_hash": self.manifest_hash}


@dataclass(frozen=True)
class AdaptivePilotArmSnapshot:
    """Immutable checkpoint for one attempted trajectory-cluster arm job."""

    binding_hash: str
    job_key: str
    arm_id: str
    opponent_profile_id: str
    master_seed: int
    hands_per_position: int
    trajectory_config_hash: str
    match_id: str
    match_record_hash: str
    status: str
    paired_differences: tuple[float, ...]
    failed_runs: tuple[Mapping[str, Any], ...]
    elapsed_ns: int
    recorded_engine_hands: int
    match_record_payload: Mapping[str, Any]
    snapshot_version: str = ADAPTIVE_PILOT_ARM_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.snapshot_version != ADAPTIVE_PILOT_ARM_SNAPSHOT_VERSION:
            raise AdaptivePilotRunError("unsupported adaptive-pilot arm snapshot version")
        _validate_sha256(self.binding_hash, "binding_hash")
        _validate_sha256(self.trajectory_config_hash, "trajectory_config_hash")
        _validate_sha256(self.match_record_hash, "match_record_hash")
        _positive_int(self.master_seed, "master_seed")
        _positive_int(self.hands_per_position, "hands_per_position")
        if isinstance(self.elapsed_ns, bool) or not isinstance(self.elapsed_ns, int) or self.elapsed_ns < 0:
            raise AdaptivePilotRunError("elapsed_ns must be a non-negative integer")
        if (
            isinstance(self.recorded_engine_hands, bool)
            or not isinstance(self.recorded_engine_hands, int)
            or self.recorded_engine_hands < 0
        ):
            raise AdaptivePilotRunError("recorded_engine_hands must be a non-negative integer")
        if self.status == "completed":
            if len(self.paired_differences) != 1 or self.failed_runs:
                raise AdaptivePilotRunError("completed arm snapshot has inconsistent result shape")
            if self.recorded_engine_hands != 2 * self.hands_per_position:
                raise AdaptivePilotRunError("completed arm snapshot must contain both complete position runs")
        elif self.status == "failed":
            if self.paired_differences or not self.failed_runs:
                raise AdaptivePilotRunError("failed arm snapshot must retain failure evidence only")
            if self.recorded_engine_hands > 2 * self.hands_per_position:
                raise AdaptivePilotRunError("failed arm snapshot contains too many recorded hands")
        else:
            raise AdaptivePilotRunError("arm snapshot status must be completed or failed")

    @property
    def match_record(self) -> MatchRecord:
        record = MatchRecord.from_dict(self.match_record_payload)
        if record.record_hash != self.match_record_hash:
            raise AdaptivePilotRunError("arm snapshot MatchRecord hash mismatch")
        return record

    def payload(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "binding_hash": self.binding_hash,
            "job_key": self.job_key,
            "arm_id": self.arm_id,
            "opponent_profile_id": self.opponent_profile_id,
            "master_seed": self.master_seed,
            "hands_per_position": self.hands_per_position,
            "trajectory_config_hash": self.trajectory_config_hash,
            "match_id": self.match_id,
            "match_record_hash": self.match_record_hash,
            "status": self.status,
            "paired_differences": list(self.paired_differences),
            "failed_runs": [dict(item) for item in self.failed_runs],
            "elapsed_ns": self.elapsed_ns,
            "recorded_engine_hands": self.recorded_engine_hands,
            "match_record": dict(self.match_record_payload),
        }

    @property
    def snapshot_hash(self) -> str:
        return _sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot": self.payload(), "snapshot_hash": self.snapshot_hash}

    def to_paired_result(self) -> PairedResult:
        return PairedResult(
            paired_differences=list(self.paired_differences),
            statistical_unit=StatisticalUnit.TRAJECTORY_CLUSTER,
            num_hands=self.hands_per_position,
            num_valid_pairs=1 if self.status == "completed" else 0,
            failed_runs=[dict(item) for item in self.failed_runs],
            experiment_metadata={"experiment_config_hash": self.trajectory_config_hash},
            seed=self.master_seed,
            match_record=self.match_record,
        )


@dataclass(frozen=True)
class AdaptivePilotPairCheckpoint:
    """Immutable proof that both arm checkpoints were jointly validated."""

    binding_hash: str
    pair_key: str
    opponent_profile_id: str
    master_seed: int
    hands_per_position: int
    treatment_snapshot_hash: str
    control_snapshot_hash: str
    treatment_record_hash: str
    control_record_hash: str
    scientifically_valid: bool
    checkpoint_version: str = ADAPTIVE_PILOT_PAIR_CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        if self.checkpoint_version != ADAPTIVE_PILOT_PAIR_CHECKPOINT_VERSION:
            raise AdaptivePilotRunError("unsupported adaptive-pilot pair checkpoint version")
        for field_name in (
            "binding_hash",
            "treatment_snapshot_hash",
            "control_snapshot_hash",
            "treatment_record_hash",
            "control_record_hash",
        ):
            _validate_sha256(str(getattr(self, field_name)), field_name)
        _positive_int(self.master_seed, "master_seed")
        _positive_int(self.hands_per_position, "hands_per_position")
        if not self.pair_key or not self.opponent_profile_id:
            raise AdaptivePilotRunError("pair checkpoint identity must be non-empty")

    def payload(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "binding_hash": self.binding_hash,
            "pair_key": self.pair_key,
            "opponent_profile_id": self.opponent_profile_id,
            "master_seed": self.master_seed,
            "hands_per_position": self.hands_per_position,
            "treatment_snapshot_hash": self.treatment_snapshot_hash,
            "control_snapshot_hash": self.control_snapshot_hash,
            "treatment_record_hash": self.treatment_record_hash,
            "control_record_hash": self.control_record_hash,
            "scientifically_valid": self.scientifically_valid,
        }

    @property
    def checkpoint_hash(self) -> str:
        return _sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint": self.payload(), "checkpoint_hash": self.checkpoint_hash}


@dataclass(frozen=True)
class AdaptivePilotRunReadiness:
    """Operational readiness summary without executing any pilot hand."""

    binding_hash: str
    planned_jobs: int
    planned_pairs: int
    planned_engine_hands: int
    terminal_jobs: int
    completed_jobs: int
    failed_jobs: int
    ready: bool = True


@dataclass(frozen=True)
class AdaptivePilotRunResult:
    """Deterministic resumable-run status after all frozen arm jobs are terminal."""

    binding_hash: str
    manifest_hash: str
    terminal_jobs: int
    completed_jobs: int
    failed_jobs: int
    pair_checkpoints: tuple[AdaptivePilotPairCheckpoint, ...]

    @property
    def pair_count(self) -> int:
        return len(self.pair_checkpoints)

    @property
    def valid_pairs(self) -> int:
        return sum(item.scientifically_valid for item in self.pair_checkpoints)

    @property
    def invalid_pairs(self) -> int:
        return self.pair_count - self.valid_pairs


def _config_path(work_root: Path, job: AdaptivePilotJob) -> Path:
    return work_root / "trajectory-configs" / f"{adaptive_pilot_job_key(job)}.json"


def _snapshot_path(work_root: Path, job: AdaptivePilotJob) -> Path:
    return work_root / "arm-snapshots" / f"{adaptive_pilot_job_key(job)}.json"


def _pair_checkpoint_path(work_root: Path, profile_id: str, master_seed: int) -> Path:
    return work_root / "pair-checkpoints" / f"{_pair_key(profile_id, master_seed)}.json"


def _planned_states(
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
) -> tuple[AdaptivePilotManifestJobState, ...]:
    states: list[AdaptivePilotManifestJobState] = []
    for job in iter_adaptive_pilot_jobs(protocol):
        config = build_adaptive_pilot_trajectory_config(
            protocol=protocol,
            binding=binding,
            job=job,
        )
        states.append(
            AdaptivePilotManifestJobState(
                job_key=adaptive_pilot_job_key(job),
                arm_id=job.arm_id,
                opponent_profile_id=job.opponent_profile_id,
                master_seed=job.master_seed,
                hands_per_position=job.hands_per_position,
                simulated_engine_hands=job.simulated_engine_hands,
                trajectory_config_hash=config.config_hash,
                match_id=adaptive_pilot_match_id(
                    binding=binding,
                    trajectory_config_hash=config.config_hash,
                ),
            )
        )
    return tuple(states)


def _manifest_from_payload(payload: Mapping[str, Any]) -> AdaptivePilotRunManifest:
    jobs_raw = payload.get("jobs")
    if not isinstance(jobs_raw, list):
        raise AdaptivePilotRunError("run manifest jobs must be an array")
    jobs: list[AdaptivePilotManifestJobState] = []
    for raw in jobs_raw:
        if not isinstance(raw, Mapping):
            raise AdaptivePilotRunError("run manifest job must be an object")
        jobs.append(
            AdaptivePilotManifestJobState(
                job_key=str(raw["job_key"]),
                arm_id=str(raw["arm_id"]),
                opponent_profile_id=str(raw["opponent_profile_id"]),
                master_seed=int(raw["master_seed"]),
                hands_per_position=int(raw["hands_per_position"]),
                simulated_engine_hands=int(raw["simulated_engine_hands"]),
                trajectory_config_hash=_validate_sha256(
                    str(raw["trajectory_config_hash"]), "trajectory_config_hash"
                ),
                match_id=str(raw["match_id"]),
                status=str(raw["status"]),
                arm_snapshot_hash=(
                    None
                    if raw.get("arm_snapshot_hash") is None
                    else _validate_sha256(str(raw["arm_snapshot_hash"]), "arm_snapshot_hash")
                ),
                match_record_hash=(
                    None
                    if raw.get("match_record_hash") is None
                    else _validate_sha256(str(raw["match_record_hash"]), "match_record_hash")
                ),
            )
        )
    return AdaptivePilotRunManifest(
        manifest_version=str(payload["manifest_version"]),
        protocol_hash=_validate_sha256(str(payload["protocol_hash"]), "protocol_hash"),
        binding_hash=_validate_sha256(str(payload["binding_hash"]), "binding_hash"),
        jobs=tuple(jobs),
    )


def save_adaptive_pilot_run_manifest(
    manifest: AdaptivePilotRunManifest,
    path: str | Path,
) -> Path:
    if not isinstance(manifest, AdaptivePilotRunManifest):
        raise TypeError("manifest must be an AdaptivePilotRunManifest")
    return _atomic_write(Path(path), manifest.to_dict())


def load_adaptive_pilot_run_manifest(
    path: str | Path,
    *,
    protocol: AdaptivePilotProtocol,
    expected_binding: AdaptivePilotExecutionBinding,
) -> AdaptivePilotRunManifest:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("manifest"), Mapping):
        raise AdaptivePilotRunError("adaptive-pilot run manifest is malformed")
    _require_canonical_file(source, raw, "adaptive-pilot run manifest")
    payload = raw["manifest"]
    stored_hash = _validate_sha256(str(raw.get("manifest_hash")), "manifest_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise AdaptivePilotRunError("manifest_hash does not match manifest payload")
    manifest = _manifest_from_payload(payload)
    if manifest.manifest_hash != stored_hash:
        raise AdaptivePilotRunError("loaded run manifest is not canonical")
    if manifest.protocol_hash != protocol.payload_hash:
        raise AdaptivePilotRunError("run manifest protocol hash does not match")
    if manifest.binding_hash != expected_binding.binding_hash:
        raise AdaptivePilotRunError("run manifest binding hash does not match")

    expected_states = _planned_states(protocol, expected_binding)
    if len(manifest.jobs) != len(expected_states):
        raise AdaptivePilotRunError("run manifest does not match the canonical 50-job plan")
    for actual, expected in zip(manifest.jobs, expected_states, strict=True):
        if actual.identity_payload() != expected.identity_payload():
            raise AdaptivePilotRunError("run manifest does not match the canonical 50-job plan")
    return manifest


def initialize_adaptive_pilot_run(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    work_root: str | Path,
) -> AdaptivePilotRunManifest:
    """Create or validate the exact on-disk run plan without executing a hand."""
    if not isinstance(protocol, AdaptivePilotProtocol):
        raise TypeError("protocol must be an AdaptivePilotProtocol")
    if not isinstance(binding, AdaptivePilotExecutionBinding):
        raise TypeError("binding must be an AdaptivePilotExecutionBinding")
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("trajectory-configs", "arm-snapshots", "pair-checkpoints"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    try:
        save_adaptive_pilot_execution_binding(binding, root / "binding.json")
    except AdaptivePilotBindingError:
        raise

    jobs = iter_adaptive_pilot_jobs(protocol)
    for job in jobs:
        config = build_adaptive_pilot_trajectory_config(
            protocol=protocol,
            binding=binding,
            job=job,
        )
        save_adaptive_pilot_trajectory_config(config, _config_path(root, job))

    manifest_path = root / "run-manifest.json"
    if manifest_path.exists():
        return load_adaptive_pilot_run_manifest(
            manifest_path,
            protocol=protocol,
            expected_binding=binding,
        )
    manifest = AdaptivePilotRunManifest(
        protocol_hash=protocol.payload_hash,
        binding_hash=binding.binding_hash,
        jobs=_planned_states(protocol, binding),
    )
    if manifest.planned_pair_count != 25 or manifest.planned_engine_hands != 18_000:
        raise AdaptivePilotRunError("run manifest does not reproduce the frozen 25-pair/18,000-hand design")
    save_adaptive_pilot_run_manifest(manifest, manifest_path)
    return manifest


def _arm_snapshot_from_result(
    *,
    job: AdaptivePilotJob,
    config: AdaptivePilotTrajectoryConfig,
    match_id: str,
    result: PairedResult,
    elapsed_ns: int,
) -> AdaptivePilotArmSnapshot:
    if not isinstance(result, PairedResult):
        raise TypeError("job executor must return a PairedResult")
    if result.statistical_unit is not StatisticalUnit.TRAJECTORY_CLUSTER:
        raise AdaptivePilotRunError("pilot arm result has wrong statistical unit")
    if result.seed != job.master_seed or result.num_hands != job.hands_per_position:
        raise AdaptivePilotRunError("pilot arm result does not match frozen job identity")
    record = result.match_record
    if record is None:
        raise AdaptivePilotRunError("pilot arm result must retain a MatchRecord")
    if record.match_id != match_id or record.master_seed != job.master_seed:
        raise AdaptivePilotRunError("pilot MatchRecord identity does not match frozen job")
    if record.experiment_config_hash != config.config_hash:
        raise AdaptivePilotRunError("pilot MatchRecord trajectory config hash does not match")
    if record.experiment_id != config.run_provenance["experiment_id"]:
        raise AdaptivePilotRunError("pilot MatchRecord experiment identity does not match")
    if record.game_config_hash != config.binding.game_config_hash:
        raise AdaptivePilotRunError("pilot MatchRecord game configuration does not match binding")

    valid_shape = result.num_valid_pairs == 1 and len(result.paired_differences) == 1 and not result.failed_runs
    failed_shape = result.num_valid_pairs == 0 and not result.paired_differences and bool(result.failed_runs)
    if not (valid_shape or failed_shape):
        raise AdaptivePilotRunError("pilot arm result has inconsistent validity fields")
    status = "completed" if valid_shape else "failed"
    return AdaptivePilotArmSnapshot(
        binding_hash=config.binding.binding_hash,
        job_key=adaptive_pilot_job_key(job),
        arm_id=job.arm_id,
        opponent_profile_id=job.opponent_profile_id,
        master_seed=job.master_seed,
        hands_per_position=job.hands_per_position,
        trajectory_config_hash=config.config_hash,
        match_id=match_id,
        match_record_hash=record.record_hash,
        status=status,
        paired_differences=tuple(float(value) for value in result.paired_differences),
        failed_runs=tuple(dict(item) for item in result.failed_runs),
        elapsed_ns=elapsed_ns,
        recorded_engine_hands=len(record.hands),
        match_record_payload=record.to_dict(),
    )


def save_adaptive_pilot_arm_snapshot(
    snapshot: AdaptivePilotArmSnapshot,
    path: str | Path,
) -> Path:
    if not isinstance(snapshot, AdaptivePilotArmSnapshot):
        raise TypeError("snapshot must be an AdaptivePilotArmSnapshot")
    destination = Path(path)
    if destination.exists():
        raw = json.loads(destination.read_text(encoding="utf-8"))
        if raw != snapshot.to_dict():
            raise AdaptivePilotRunError("existing arm snapshot does not match immutable evidence")
        return destination
    return _atomic_write(destination, snapshot.to_dict())


def load_adaptive_pilot_arm_snapshot(
    path: str | Path,
    *,
    protocol: AdaptivePilotProtocol,
    expected_binding: AdaptivePilotExecutionBinding,
    expected_job: AdaptivePilotJob,
) -> AdaptivePilotArmSnapshot:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("snapshot"), Mapping):
        raise AdaptivePilotRunError("adaptive-pilot arm snapshot is malformed")
    _require_canonical_file(source, raw, "adaptive-pilot arm snapshot")
    payload = raw["snapshot"]
    stored_hash = _validate_sha256(str(raw.get("snapshot_hash")), "snapshot_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise AdaptivePilotRunError("snapshot_hash does not match arm snapshot payload")

    failures_raw = payload.get("failed_runs")
    differences_raw = payload.get("paired_differences")
    record_raw = payload.get("match_record")
    if not isinstance(failures_raw, list) or not isinstance(differences_raw, list) or not isinstance(record_raw, Mapping):
        raise AdaptivePilotRunError("adaptive-pilot arm snapshot fields are malformed")
    failures: list[Mapping[str, Any]] = []
    for item in failures_raw:
        if not isinstance(item, Mapping):
            raise AdaptivePilotRunError("arm snapshot failure entry must be an object")
        failures.append(dict(item))

    snapshot = AdaptivePilotArmSnapshot(
        snapshot_version=str(payload["snapshot_version"]),
        binding_hash=_validate_sha256(str(payload["binding_hash"]), "binding_hash"),
        job_key=str(payload["job_key"]),
        arm_id=str(payload["arm_id"]),
        opponent_profile_id=str(payload["opponent_profile_id"]),
        master_seed=int(payload["master_seed"]),
        hands_per_position=int(payload["hands_per_position"]),
        trajectory_config_hash=_validate_sha256(
            str(payload["trajectory_config_hash"]), "trajectory_config_hash"
        ),
        match_id=str(payload["match_id"]),
        match_record_hash=_validate_sha256(str(payload["match_record_hash"]), "match_record_hash"),
        status=str(payload["status"]),
        paired_differences=tuple(float(value) for value in differences_raw),
        failed_runs=tuple(failures),
        elapsed_ns=int(payload["elapsed_ns"]),
        recorded_engine_hands=int(payload["recorded_engine_hands"]),
        match_record_payload=dict(record_raw),
    )
    if snapshot.snapshot_hash != stored_hash:
        raise AdaptivePilotRunError("loaded arm snapshot is not canonical")
    expected_config = build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=expected_binding,
        job=expected_job,
    )
    expected_match_id = adaptive_pilot_match_id(
        binding=expected_binding,
        trajectory_config_hash=expected_config.config_hash,
    )
    if snapshot.binding_hash != expected_binding.binding_hash:
        raise AdaptivePilotRunError("arm snapshot binding hash does not match")
    if snapshot.job_key != adaptive_pilot_job_key(expected_job):
        raise AdaptivePilotRunError("arm snapshot job key does not match")
    if snapshot.trajectory_config_hash != expected_config.config_hash:
        raise AdaptivePilotRunError("arm snapshot trajectory config hash does not match")
    if snapshot.match_id != expected_match_id:
        raise AdaptivePilotRunError("arm snapshot match ID does not match")
    if snapshot.arm_id != expected_job.arm_id or snapshot.opponent_profile_id != expected_job.opponent_profile_id:
        raise AdaptivePilotRunError("arm snapshot job identity does not match")
    if snapshot.master_seed != expected_job.master_seed or snapshot.hands_per_position != expected_job.hands_per_position:
        raise AdaptivePilotRunError("arm snapshot job schedule does not match")
    record = snapshot.match_record
    if record.match_id != expected_match_id or record.experiment_config_hash != expected_config.config_hash:
        raise AdaptivePilotRunError("arm snapshot MatchRecord identity does not match")
    return snapshot


def _save_pair_checkpoint(checkpoint: AdaptivePilotPairCheckpoint, path: Path) -> Path:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw != checkpoint.to_dict():
            raise AdaptivePilotRunError("existing pair checkpoint does not match immutable arm evidence")
        return path
    return _atomic_write(path, checkpoint.to_dict())


def _load_pair_checkpoint(
    path: Path,
    *,
    binding: AdaptivePilotExecutionBinding,
    treatment: AdaptivePilotArmSnapshot,
    control: AdaptivePilotArmSnapshot,
) -> AdaptivePilotPairCheckpoint:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("checkpoint"), Mapping):
        raise AdaptivePilotRunError("adaptive-pilot pair checkpoint is malformed")
    _require_canonical_file(path, raw, "adaptive-pilot pair checkpoint")
    payload = raw["checkpoint"]
    stored_hash = _validate_sha256(str(raw.get("checkpoint_hash")), "checkpoint_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise AdaptivePilotRunError("checkpoint_hash does not match pair checkpoint payload")
    checkpoint = AdaptivePilotPairCheckpoint(
        checkpoint_version=str(payload["checkpoint_version"]),
        binding_hash=_validate_sha256(str(payload["binding_hash"]), "binding_hash"),
        pair_key=str(payload["pair_key"]),
        opponent_profile_id=str(payload["opponent_profile_id"]),
        master_seed=int(payload["master_seed"]),
        hands_per_position=int(payload["hands_per_position"]),
        treatment_snapshot_hash=_validate_sha256(
            str(payload["treatment_snapshot_hash"]), "treatment_snapshot_hash"
        ),
        control_snapshot_hash=_validate_sha256(
            str(payload["control_snapshot_hash"]), "control_snapshot_hash"
        ),
        treatment_record_hash=_validate_sha256(
            str(payload["treatment_record_hash"]), "treatment_record_hash"
        ),
        control_record_hash=_validate_sha256(
            str(payload["control_record_hash"]), "control_record_hash"
        ),
        scientifically_valid=bool(payload["scientifically_valid"]),
    )
    if checkpoint.checkpoint_hash != stored_hash:
        raise AdaptivePilotRunError("loaded pair checkpoint is not canonical")
    expected_pair_key = _pair_key(treatment.opponent_profile_id, treatment.master_seed)
    if checkpoint.binding_hash != binding.binding_hash or checkpoint.pair_key != expected_pair_key:
        raise AdaptivePilotRunError("pair checkpoint binding or identity does not match")
    if checkpoint.treatment_snapshot_hash != treatment.snapshot_hash:
        raise AdaptivePilotRunError("pair checkpoint treatment snapshot hash does not match")
    if checkpoint.control_snapshot_hash != control.snapshot_hash:
        raise AdaptivePilotRunError("pair checkpoint control snapshot hash does not match")
    if checkpoint.treatment_record_hash != treatment.match_record_hash:
        raise AdaptivePilotRunError("pair checkpoint treatment record hash does not match")
    if checkpoint.control_record_hash != control.match_record_hash:
        raise AdaptivePilotRunError("pair checkpoint control record hash does not match")
    return checkpoint


def _pair_checkpoint_from_execution(
    *,
    binding: AdaptivePilotExecutionBinding,
    execution: AdaptivePilotPairExecution,
    treatment: AdaptivePilotArmSnapshot,
    control: AdaptivePilotArmSnapshot,
) -> AdaptivePilotPairCheckpoint:
    return AdaptivePilotPairCheckpoint(
        binding_hash=binding.binding_hash,
        pair_key=_pair_key(execution.profile_id, execution.master_seed),
        opponent_profile_id=execution.profile_id,
        master_seed=execution.master_seed,
        hands_per_position=execution.hands_per_position,
        treatment_snapshot_hash=treatment.snapshot_hash,
        control_snapshot_hash=control.snapshot_hash,
        treatment_record_hash=execution.treatment_record_hash,
        control_record_hash=execution.control_record_hash,
        scientifically_valid=execution.scientifically_valid,
    )


def _require_run_store(store: object) -> AdaptivePilotRunStore:
    if not isinstance(store, AdaptivePilotRunStore):
        raise AdaptivePilotRunError("adaptive-pilot run requires durable save/load/list match storage")
    return store


def _exact_store_record(store: AdaptivePilotRunStore, snapshot: AdaptivePilotArmSnapshot) -> None:
    try:
        persisted = store.load_match(snapshot.match_id)
    except Exception as exc:
        raise AdaptivePilotRunError("pilot MatchRecord is not durably persisted") from exc
    if not isinstance(persisted, MatchRecord) or persisted.record_hash != snapshot.match_record_hash:
        raise AdaptivePilotRunError("durably persisted MatchRecord differs from arm snapshot evidence")


def _recover_or_verify_store_record(
    store: AdaptivePilotRunStore,
    snapshot: AdaptivePilotArmSnapshot,
    known_match_ids: set[str],
) -> None:
    if snapshot.match_id not in known_match_ids:
        try:
            store.save_match(snapshot.match_record)
        except Exception as exc:
            raise AdaptivePilotRunError("could not recover durable MatchRecord from arm snapshot") from exc
        known_match_ids.add(snapshot.match_id)
    _exact_store_record(store, snapshot)


def _replace_manifest_job(
    manifest: AdaptivePilotRunManifest,
    index: int,
    state: AdaptivePilotManifestJobState,
) -> AdaptivePilotRunManifest:
    jobs = list(manifest.jobs)
    jobs[index] = state
    return replace(manifest, jobs=tuple(jobs))


def _load_exact_config(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    work_root: Path,
    job: AdaptivePilotJob,
) -> AdaptivePilotTrajectoryConfig:
    path = _config_path(work_root, job)
    if not path.exists():
        raise AdaptivePilotRunError(f"trajectory config is missing for {adaptive_pilot_job_key(job)}")
    try:
        return load_adaptive_pilot_trajectory_config(
            path,
            protocol=protocol,
            expected_binding=binding,
            expected_job=job,
        )
    except (AdaptivePilotBindingError, KeyError, TypeError, ValueError) as exc:
        raise AdaptivePilotRunError(
            f"trajectory config is invalid for {adaptive_pilot_job_key(job)}"
        ) from exc


def _validate_directory_membership(root: Path, expected: set[str], label: str) -> None:
    actual = {path.name for path in root.glob("*.json")}
    unexpected = actual - expected
    if unexpected:
        raise AdaptivePilotRunError(f"unexpected {label} files: {', '.join(sorted(unexpected))}")


def _reconcile_manifest(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    work_root: Path,
    store: AdaptivePilotRunStore,
    manifest: AdaptivePilotRunManifest,
) -> AdaptivePilotRunManifest:
    jobs = iter_adaptive_pilot_jobs(protocol)
    try:
        known_match_ids = set(store.list_match_ids())
    except Exception as exc:
        raise AdaptivePilotRunError("could not enumerate durable adaptive-pilot MatchRecords") from exc

    updated = manifest
    for index, (job, state) in enumerate(zip(jobs, updated.jobs, strict=True)):
        snapshot_path = _snapshot_path(work_root, job)
        if snapshot_path.exists():
            snapshot = load_adaptive_pilot_arm_snapshot(
                snapshot_path,
                protocol=protocol,
                expected_binding=binding,
                expected_job=job,
            )
            _recover_or_verify_store_record(store, snapshot, known_match_ids)
            terminal_state = replace(
                state,
                status=snapshot.status,
                arm_snapshot_hash=snapshot.snapshot_hash,
                match_record_hash=snapshot.match_record_hash,
            )
            if state.status in _TERMINAL_STATUSES and state != terminal_state:
                raise AdaptivePilotRunError("terminal manifest state differs from immutable arm snapshot")
            if state != terminal_state:
                updated = _replace_manifest_job(updated, index, terminal_state)
            continue

        if state.status in _TERMINAL_STATUSES:
            raise AdaptivePilotRunError("terminal manifest state has no arm snapshot")
        if state.match_id in known_match_ids:
            raise AdaptivePilotRunError(
                "orphan durable MatchRecord exists without its arm checkpoint; refusing resume"
            )
        if state.status == "running":
            updated = _replace_manifest_job(updated, index, replace(state, status="planned"))

    if updated != manifest:
        save_adaptive_pilot_run_manifest(updated, work_root / "run-manifest.json")
    return updated


def _arm_snapshots_by_pair(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    work_root: Path,
    manifest: AdaptivePilotRunManifest,
) -> dict[tuple[str, int], dict[str, tuple[AdaptivePilotJob, AdaptivePilotArmSnapshot]]]:
    grouped: dict[tuple[str, int], dict[str, tuple[AdaptivePilotJob, AdaptivePilotArmSnapshot]]] = {}
    for job, state in zip(iter_adaptive_pilot_jobs(protocol), manifest.jobs, strict=True):
        if state.status not in _TERMINAL_STATUSES:
            continue
        snapshot = load_adaptive_pilot_arm_snapshot(
            _snapshot_path(work_root, job),
            protocol=protocol,
            expected_binding=binding,
            expected_job=job,
        )
        grouped.setdefault((job.opponent_profile_id, job.master_seed), {})[job.arm_id] = (job, snapshot)
    return grouped


def _load_or_finalize_pair_checkpoints(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    work_root: Path,
    store: AdaptivePilotRunStore,
    manifest: AdaptivePilotRunManifest,
) -> tuple[AdaptivePilotPairCheckpoint, ...]:
    grouped = _arm_snapshots_by_pair(
        protocol=protocol,
        binding=binding,
        work_root=work_root,
        manifest=manifest,
    )
    checkpoints: list[AdaptivePilotPairCheckpoint] = []
    for profile_id in protocol.opponent_profile_ids:
        for seed in protocol.master_seeds:
            arms = grouped.get((profile_id, seed), {})
            if set(arms) != {"control", "treatment"}:
                continue
            control_job, control = arms["control"]
            treatment_job, treatment = arms["treatment"]
            path = _pair_checkpoint_path(work_root, profile_id, seed)
            if path.exists():
                checkpoint = _load_pair_checkpoint(
                    path,
                    binding=binding,
                    treatment=treatment,
                    control=control,
                )
            else:
                execution = finalize_adaptive_pilot_pair(
                    protocol=protocol,
                    treatment_job=treatment_job,
                    control_job=control_job,
                    treatment_result=treatment.to_paired_result(),
                    control_result=control.to_paired_result(),
                    benchmark_store=store,
                )
                checkpoint = _pair_checkpoint_from_execution(
                    binding=binding,
                    execution=execution,
                    treatment=treatment,
                    control=control,
                )
                _save_pair_checkpoint(checkpoint, path)
            checkpoints.append(checkpoint)
    return tuple(checkpoints)


def assert_adaptive_pilot_run_ready(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    work_root: str | Path,
    benchmark_store: AdaptivePilotRunStore,
    expected_workers: int = 1,
) -> AdaptivePilotRunReadiness:
    """Validate resumable execution state without simulating any new hand."""
    _positive_int(expected_workers, "expected_workers")
    if binding.workers != expected_workers:
        raise AdaptivePilotRunError(
            f"adaptive-pilot run readiness requires provenance workers={expected_workers}"
        )
    store = _require_run_store(benchmark_store)
    root = Path(work_root)
    binding_path = root / "binding.json"
    manifest_path = root / "run-manifest.json"
    if not binding_path.exists() or not manifest_path.exists():
        raise AdaptivePilotRunError("adaptive-pilot run plan is not initialized")
    loaded_binding = load_adaptive_pilot_execution_binding(binding_path)
    if loaded_binding != binding:
        raise AdaptivePilotRunError("persisted adaptive-pilot binding does not match execution binding")

    jobs = iter_adaptive_pilot_jobs(protocol)
    expected_config_names = {f"{adaptive_pilot_job_key(job)}.json" for job in jobs}
    _validate_directory_membership(root / "trajectory-configs", expected_config_names, "trajectory config")
    for job in jobs:
        _load_exact_config(
            protocol=protocol,
            binding=binding,
            work_root=root,
            job=job,
        )

    expected_snapshot_names = {f"{adaptive_pilot_job_key(job)}.json" for job in jobs}
    _validate_directory_membership(root / "arm-snapshots", expected_snapshot_names, "arm snapshot")
    expected_pair_names = {
        f"{_pair_key(profile_id, seed)}.json"
        for profile_id in protocol.opponent_profile_ids
        for seed in protocol.master_seeds
    }
    _validate_directory_membership(root / "pair-checkpoints", expected_pair_names, "pair checkpoint")

    manifest = load_adaptive_pilot_run_manifest(
        manifest_path,
        protocol=protocol,
        expected_binding=binding,
    )
    manifest = _reconcile_manifest(
        protocol=protocol,
        binding=binding,
        work_root=root,
        store=store,
        manifest=manifest,
    )
    _load_or_finalize_pair_checkpoints(
        protocol=protocol,
        binding=binding,
        work_root=root,
        store=store,
        manifest=manifest,
    )
    completed = sum(item.status == "completed" for item in manifest.jobs)
    failed = sum(item.status == "failed" for item in manifest.jobs)
    return AdaptivePilotRunReadiness(
        binding_hash=binding.binding_hash,
        planned_jobs=manifest.planned_job_count,
        planned_pairs=manifest.planned_pair_count,
        planned_engine_hands=manifest.planned_engine_hands,
        terminal_jobs=manifest.terminal_jobs,
        completed_jobs=completed,
        failed_jobs=failed,
    )


def execute_adaptive_pilot_run(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    work_root: str | Path,
    benchmark_store: AdaptivePilotRunStore,
    job_executor: AdaptivePilotJobExecutor,
) -> AdaptivePilotRunResult:
    """Execute or resume all 50 frozen arm jobs using an already-guarded executor."""
    if not callable(job_executor):
        raise TypeError("job_executor must be callable")
    store = _require_run_store(benchmark_store)
    root = Path(work_root)
    initialize_adaptive_pilot_run(protocol=protocol, binding=binding, work_root=root)
    assert_adaptive_pilot_run_ready(
        protocol=protocol,
        binding=binding,
        work_root=root,
        benchmark_store=store,
    )
    manifest = load_adaptive_pilot_run_manifest(
        root / "run-manifest.json",
        protocol=protocol,
        expected_binding=binding,
    )
    manifest = _reconcile_manifest(
        protocol=protocol,
        binding=binding,
        work_root=root,
        store=store,
        manifest=manifest,
    )

    for index, (job, state) in enumerate(zip(iter_adaptive_pilot_jobs(protocol), manifest.jobs, strict=True)):
        if state.status in _TERMINAL_STATUSES:
            continue
        config = _load_exact_config(
            protocol=protocol,
            binding=binding,
            work_root=root,
            job=job,
        )
        running = replace(state, status="running")
        manifest = _replace_manifest_job(manifest, index, running)
        save_adaptive_pilot_run_manifest(manifest, root / "run-manifest.json")

        started_ns = time.perf_counter_ns()
        result = job_executor(job, config, state.match_id, store)
        elapsed_ns = time.perf_counter_ns() - started_ns
        snapshot = _arm_snapshot_from_result(
            job=job,
            config=config,
            match_id=state.match_id,
            result=result,
            elapsed_ns=elapsed_ns,
        )
        _exact_store_record(store, snapshot)
        save_adaptive_pilot_arm_snapshot(snapshot, _snapshot_path(root, job))

        terminal = replace(
            state,
            status=snapshot.status,
            arm_snapshot_hash=snapshot.snapshot_hash,
            match_record_hash=snapshot.match_record_hash,
        )
        manifest = _replace_manifest_job(manifest, index, terminal)
        save_adaptive_pilot_run_manifest(manifest, root / "run-manifest.json")
        _load_or_finalize_pair_checkpoints(
            protocol=protocol,
            binding=binding,
            work_root=root,
            store=store,
            manifest=manifest,
        )

    if manifest.terminal_jobs != 50:
        raise AdaptivePilotRunError("adaptive-pilot run ended without all 50 jobs terminal")
    checkpoints = _load_or_finalize_pair_checkpoints(
        protocol=protocol,
        binding=binding,
        work_root=root,
        store=store,
        manifest=manifest,
    )
    if len(checkpoints) != 25:
        raise AdaptivePilotRunError("adaptive-pilot run ended without all 25 complete pairs")
    return AdaptivePilotRunResult(
        binding_hash=binding.binding_hash,
        manifest_hash=manifest.manifest_hash,
        terminal_jobs=manifest.terminal_jobs,
        completed_jobs=sum(item.status == "completed" for item in manifest.jobs),
        failed_jobs=sum(item.status == "failed" for item in manifest.jobs),
        pair_checkpoints=checkpoints,
    )
