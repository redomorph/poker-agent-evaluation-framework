"""Resumable persistence/orchestration for the frozen Spec-37 variance-pilot matrix."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pokerbench.experiments.h4_holdout_population import H4HoldoutPopulation
from pokerbench.experiments.h4_holdout_variance_analysis import population_contrast_for_seed
from pokerbench.experiments.h4_holdout_variance_binding import (
    H4HoldoutVarianceBindingError,
    H4HoldoutVarianceExecutionBinding,
    H4HoldoutVarianceTrajectoryConfig,
    build_h4_holdout_variance_trajectory_config,
    load_h4_holdout_variance_execution_binding,
    load_h4_holdout_variance_trajectory_config,
    save_h4_holdout_variance_execution_binding,
    save_h4_holdout_variance_trajectory_config,
)
from pokerbench.experiments.h4_holdout_variance_execution import (
    H4HoldoutVariancePairExecution,
    finalize_h4_holdout_variance_pair,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4_HOLDOUT_UNIT_IDS,
    H4_VARIANCE_MASTER_SEEDS,
    H4HoldoutVarianceJob,
    H4HoldoutVarianceProtocol,
    iter_h4_holdout_variance_jobs,
)
from pokerbench.replay.archive import MatchRecord
from pokerbench.runner import PairedResult, StatisticalUnit


H4_HOLDOUT_VARIANCE_RUN_MANIFEST_VERSION = "phase2-h4-holdout-variance-run-manifest-v1"
H4_HOLDOUT_VARIANCE_ARM_SNAPSHOT_VERSION = "phase2-h4-holdout-variance-arm-snapshot-v1"
H4_HOLDOUT_VARIANCE_PAIR_CHECKPOINT_VERSION = "phase2-h4-holdout-variance-pair-checkpoint-v1"
H4_HOLDOUT_VARIANCE_CLUSTER_CHECKPOINT_VERSION = "phase2-h4-holdout-variance-cluster-checkpoint-v1"

_TERMINAL_STATUSES = frozenset({"completed", "failed"})
_ALL_STATUSES = frozenset({"planned", "running", *_TERMINAL_STATUSES})


class H4HoldoutVarianceRunError(ValueError):
    """Raised when resumable Spec-37 execution state is inconsistent."""


@runtime_checkable
class H4HoldoutVarianceRunStore(Protocol):
    def save_match(self, record: MatchRecord) -> None: ...
    def load_match(self, match_id: str) -> MatchRecord: ...
    def list_match_ids(self) -> tuple[str, ...]: ...


class H4HoldoutVarianceJobExecutor(Protocol):
    """Execute one already-bound arm job without owning campaign persistence."""

    def __call__(
        self,
        job: H4HoldoutVarianceJob,
        config: H4HoldoutVarianceTrajectoryConfig,
        match_id: str,
    ) -> PairedResult: ...


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_file_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise H4HoldoutVarianceRunError(f"{field_name} must be a lowercase SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise H4HoldoutVarianceRunError(
            f"{field_name} must be a lowercase SHA-256 hex digest"
        ) from exc
    if value.lower() != value:
        raise H4HoldoutVarianceRunError(f"{field_name} must be lowercase")
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
        raise H4HoldoutVarianceRunError(
            f"{what} file is not canonical JSON with one trailing LF"
        )


def h4_holdout_variance_job_key(job: H4HoldoutVarianceJob) -> str:
    if not isinstance(job, H4HoldoutVarianceJob):
        raise TypeError("job must be H4HoldoutVarianceJob")
    return f"{job.holdout_unit_id}__{job.master_seed}__{job.arm_id}"


def _pair_key(unit_id: str, seed: int) -> str:
    return f"{unit_id}__{seed}"


def _cluster_key(seed: int) -> str:
    return f"population__{seed}"


def h4_holdout_variance_match_id(
    *,
    binding: H4HoldoutVarianceExecutionBinding,
    trajectory_config_hash: str,
) -> str:
    if not isinstance(binding, H4HoldoutVarianceExecutionBinding):
        raise TypeError("binding must be H4HoldoutVarianceExecutionBinding")
    _validate_sha256(trajectory_config_hash, "trajectory_config_hash")
    digest = _sha256_canonical(
        {"binding_hash": binding.binding_hash, "trajectory_config_hash": trajectory_config_hash}
    )
    return f"phase2-h4-variance-{digest[:24]}"


@dataclass(frozen=True)
class H4HoldoutVarianceManifestJobState:
    job_key: str
    arm_id: str
    holdout_unit_id: str
    master_seed: int
    hands_per_position: int
    simulated_engine_hands: int
    trajectory_config_hash: str
    match_id: str
    status: str = "planned"
    arm_snapshot_hash: str | None = None
    match_record_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.job_key or self.arm_id not in {"control", "treatment"}:
            raise H4HoldoutVarianceRunError("manifest job identity is invalid")
        if self.holdout_unit_id not in H4_HOLDOUT_UNIT_IDS or self.master_seed not in H4_VARIANCE_MASTER_SEEDS:
            raise H4HoldoutVarianceRunError("manifest job is outside frozen Spec-37 matrix")
        if self.hands_per_position != 500 or self.simulated_engine_hands != 1000:
            raise H4HoldoutVarianceRunError("manifest job must represent exactly 500+500 hands")
        _validate_sha256(self.trajectory_config_hash, "trajectory_config_hash")
        if not self.match_id or self.status not in _ALL_STATUSES:
            raise H4HoldoutVarianceRunError("manifest job status/identity is invalid")
        if self.status in _TERMINAL_STATUSES:
            if self.arm_snapshot_hash is None or self.match_record_hash is None:
                raise H4HoldoutVarianceRunError("terminal manifest state requires snapshot/record hashes")
            _validate_sha256(self.arm_snapshot_hash, "arm_snapshot_hash")
            _validate_sha256(self.match_record_hash, "match_record_hash")
        elif self.arm_snapshot_hash is not None or self.match_record_hash is not None:
            raise H4HoldoutVarianceRunError("non-terminal manifest state cannot retain terminal hashes")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "job_key": self.job_key,
            "arm_id": self.arm_id,
            "holdout_unit_id": self.holdout_unit_id,
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
class H4HoldoutVarianceRunManifest:
    protocol_hash: str
    binding_hash: str
    jobs: tuple[H4HoldoutVarianceManifestJobState, ...]
    manifest_version: str = H4_HOLDOUT_VARIANCE_RUN_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.manifest_version != H4_HOLDOUT_VARIANCE_RUN_MANIFEST_VERSION:
            raise H4HoldoutVarianceRunError("unsupported variance-pilot run manifest version")
        _validate_sha256(self.protocol_hash, "protocol_hash")
        _validate_sha256(self.binding_hash, "binding_hash")
        if len(self.jobs) != 100:
            raise H4HoldoutVarianceRunError("run manifest must contain the canonical 100-job plan")
        if len({item.job_key for item in self.jobs}) != 100 or len({item.match_id for item in self.jobs}) != 100:
            raise H4HoldoutVarianceRunError("run manifest contains duplicate scientific identities")

    @property
    def planned_job_count(self) -> int:
        return len(self.jobs)

    @property
    def planned_pair_count(self) -> int:
        return len({(item.holdout_unit_id, item.master_seed) for item in self.jobs})

    @property
    def planned_cluster_count(self) -> int:
        return len({item.master_seed for item in self.jobs})

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
class H4HoldoutVarianceArmSnapshot:
    binding_hash: str
    job_key: str
    arm_id: str
    holdout_unit_id: str
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
    snapshot_version: str = H4_HOLDOUT_VARIANCE_ARM_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.snapshot_version != H4_HOLDOUT_VARIANCE_ARM_SNAPSHOT_VERSION:
            raise H4HoldoutVarianceRunError("unsupported variance-pilot arm snapshot version")
        _validate_sha256(self.binding_hash, "binding_hash")
        _validate_sha256(self.trajectory_config_hash, "trajectory_config_hash")
        _validate_sha256(self.match_record_hash, "match_record_hash")
        if self.holdout_unit_id not in H4_HOLDOUT_UNIT_IDS or self.master_seed not in H4_VARIANCE_MASTER_SEEDS:
            raise H4HoldoutVarianceRunError("arm snapshot identity is outside frozen Spec 37")
        if self.hands_per_position != 500 or self.elapsed_ns < 0 or self.recorded_engine_hands < 0:
            raise H4HoldoutVarianceRunError("arm snapshot horizon/timing is invalid")
        if self.status == "completed":
            if len(self.paired_differences) != 1 or self.failed_runs or self.recorded_engine_hands != 1000:
                raise H4HoldoutVarianceRunError("completed arm snapshot has inconsistent evidence")
        elif self.status == "failed":
            if self.paired_differences or not self.failed_runs or self.recorded_engine_hands > 1000:
                raise H4HoldoutVarianceRunError("failed arm snapshot has inconsistent evidence")
        else:
            raise H4HoldoutVarianceRunError("arm snapshot status must be completed or failed")

    @property
    def match_record(self) -> MatchRecord:
        record = MatchRecord.from_dict(self.match_record_payload)
        if record.record_hash != self.match_record_hash:
            raise H4HoldoutVarianceRunError("arm snapshot MatchRecord hash mismatch")
        return record

    def payload(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "binding_hash": self.binding_hash,
            "job_key": self.job_key,
            "arm_id": self.arm_id,
            "holdout_unit_id": self.holdout_unit_id,
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
            num_hands=500,
            num_valid_pairs=1 if self.status == "completed" else 0,
            failed_runs=[dict(item) for item in self.failed_runs],
            experiment_metadata={"experiment_config_hash": self.trajectory_config_hash},
            seed=self.master_seed,
            match_record=self.match_record,
        )


@dataclass(frozen=True)
class H4HoldoutVariancePairCheckpoint:
    binding_hash: str
    pair_key: str
    holdout_unit_id: str
    master_seed: int
    treatment_snapshot_hash: str
    control_snapshot_hash: str
    treatment_record_hash: str
    control_record_hash: str
    scientifically_valid: bool
    unit_contrast_bb100: float | None
    checkpoint_version: str = H4_HOLDOUT_VARIANCE_PAIR_CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        if self.checkpoint_version != H4_HOLDOUT_VARIANCE_PAIR_CHECKPOINT_VERSION:
            raise H4HoldoutVarianceRunError("unsupported variance-pilot pair checkpoint version")
        for name in (
            "binding_hash", "treatment_snapshot_hash", "control_snapshot_hash",
            "treatment_record_hash", "control_record_hash",
        ):
            _validate_sha256(str(getattr(self, name)), name)
        if self.pair_key != _pair_key(self.holdout_unit_id, self.master_seed):
            raise H4HoldoutVarianceRunError("pair checkpoint identity is inconsistent")
        if self.scientifically_valid != (self.unit_contrast_bb100 is not None):
            raise H4HoldoutVarianceRunError("pair checkpoint validity/contrast is inconsistent")

    def payload(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "binding_hash": self.binding_hash,
            "pair_key": self.pair_key,
            "holdout_unit_id": self.holdout_unit_id,
            "master_seed": self.master_seed,
            "treatment_snapshot_hash": self.treatment_snapshot_hash,
            "control_snapshot_hash": self.control_snapshot_hash,
            "treatment_record_hash": self.treatment_record_hash,
            "control_record_hash": self.control_record_hash,
            "scientifically_valid": self.scientifically_valid,
            "unit_contrast_bb100": self.unit_contrast_bb100,
        }

    @property
    def checkpoint_hash(self) -> str:
        return _sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint": self.payload(), "checkpoint_hash": self.checkpoint_hash}


@dataclass(frozen=True)
class H4HoldoutVarianceClusterCheckpoint:
    binding_hash: str
    cluster_key: str
    master_seed: int
    pair_checkpoint_hashes: tuple[tuple[str, str], ...]
    scientifically_valid: bool
    population_contrast_bb100: float | None
    checkpoint_version: str = H4_HOLDOUT_VARIANCE_CLUSTER_CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        if self.checkpoint_version != H4_HOLDOUT_VARIANCE_CLUSTER_CHECKPOINT_VERSION:
            raise H4HoldoutVarianceRunError("unsupported variance-pilot cluster checkpoint version")
        _validate_sha256(self.binding_hash, "binding_hash")
        if self.cluster_key != _cluster_key(self.master_seed) or self.master_seed not in H4_VARIANCE_MASTER_SEEDS:
            raise H4HoldoutVarianceRunError("cluster checkpoint identity is inconsistent")
        if tuple(unit for unit, _ in self.pair_checkpoint_hashes) != H4_HOLDOUT_UNIT_IDS:
            raise H4HoldoutVarianceRunError("cluster checkpoint must bind all five units")
        for _, digest in self.pair_checkpoint_hashes:
            _validate_sha256(digest, "pair_checkpoint_hash")
        if self.scientifically_valid != (self.population_contrast_bb100 is not None):
            raise H4HoldoutVarianceRunError("cluster checkpoint validity/contrast is inconsistent")

    def payload(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "binding_hash": self.binding_hash,
            "cluster_key": self.cluster_key,
            "master_seed": self.master_seed,
            "pair_checkpoint_hashes": [
                {"holdout_unit_id": unit, "checkpoint_hash": digest}
                for unit, digest in self.pair_checkpoint_hashes
            ],
            "scientifically_valid": self.scientifically_valid,
            "population_contrast_bb100": self.population_contrast_bb100,
        }

    @property
    def checkpoint_hash(self) -> str:
        return _sha256_canonical(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint": self.payload(), "checkpoint_hash": self.checkpoint_hash}


@dataclass(frozen=True)
class H4HoldoutVarianceRunReadiness:
    binding_hash: str
    planned_jobs: int
    planned_pairs: int
    planned_clusters: int
    planned_engine_hands: int
    terminal_jobs: int
    completed_jobs: int
    failed_jobs: int
    pair_count: int
    valid_pairs: int
    invalid_pairs: int
    cluster_count: int
    valid_clusters: int
    invalid_clusters: int

    @property
    def ready(self) -> bool:
        return True

    @property
    def scientifically_complete(self) -> bool:
        return (
            self.completed_jobs == 100
            and self.failed_jobs == 0
            and self.valid_pairs == 50
            and self.invalid_pairs == 0
            and self.valid_clusters == 10
            and self.invalid_clusters == 0
        )


@dataclass(frozen=True)
class H4HoldoutVarianceRunResult:
    binding_hash: str
    terminal_jobs: int
    completed_jobs: int
    failed_jobs: int
    pair_count: int
    valid_pairs: int
    invalid_pairs: int
    cluster_count: int
    valid_clusters: int
    invalid_clusters: int
    scientifically_complete: bool


def _config_path(root: Path, job: H4HoldoutVarianceJob) -> Path:
    return root / "trajectory-configs" / f"{h4_holdout_variance_job_key(job)}.json"


def _snapshot_path(root: Path, job: H4HoldoutVarianceJob) -> Path:
    return root / "arm-snapshots" / f"{h4_holdout_variance_job_key(job)}.json"


def _pair_checkpoint_path(root: Path, unit_id: str, seed: int) -> Path:
    return root / "pair-checkpoints" / f"{_pair_key(unit_id, seed)}.json"


def _cluster_checkpoint_path(root: Path, seed: int) -> Path:
    return root / "cluster-checkpoints" / f"{_cluster_key(seed)}.json"


def _planned_states(
    protocol: H4HoldoutVarianceProtocol,
    binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation,
) -> tuple[H4HoldoutVarianceManifestJobState, ...]:
    states: list[H4HoldoutVarianceManifestJobState] = []
    for job in iter_h4_holdout_variance_jobs(protocol):
        config = build_h4_holdout_variance_trajectory_config(
            protocol=protocol, binding=binding, population=population, job=job
        )
        states.append(
            H4HoldoutVarianceManifestJobState(
                job_key=h4_holdout_variance_job_key(job),
                arm_id=job.arm_id,
                holdout_unit_id=job.holdout_unit_id,
                master_seed=job.master_seed,
                hands_per_position=job.hands_per_position,
                simulated_engine_hands=job.simulated_engine_hands,
                trajectory_config_hash=config.config_hash,
                match_id=h4_holdout_variance_match_id(
                    binding=binding, trajectory_config_hash=config.config_hash
                ),
            )
        )
    return tuple(states)


def _manifest_from_payload(payload: Mapping[str, Any]) -> H4HoldoutVarianceRunManifest:
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise H4HoldoutVarianceRunError("run manifest jobs must be an array")
    jobs: list[H4HoldoutVarianceManifestJobState] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise H4HoldoutVarianceRunError("run manifest job must be an object")
        jobs.append(
            H4HoldoutVarianceManifestJobState(
                job_key=str(raw["job_key"]), arm_id=str(raw["arm_id"]),
                holdout_unit_id=str(raw["holdout_unit_id"]), master_seed=int(raw["master_seed"]),
                hands_per_position=int(raw["hands_per_position"]), simulated_engine_hands=int(raw["simulated_engine_hands"]),
                trajectory_config_hash=_validate_sha256(str(raw["trajectory_config_hash"]), "trajectory_config_hash"),
                match_id=str(raw["match_id"]), status=str(raw["status"]),
                arm_snapshot_hash=None if raw.get("arm_snapshot_hash") is None else _validate_sha256(str(raw["arm_snapshot_hash"]), "arm_snapshot_hash"),
                match_record_hash=None if raw.get("match_record_hash") is None else _validate_sha256(str(raw["match_record_hash"]), "match_record_hash"),
            )
        )
    return H4HoldoutVarianceRunManifest(
        manifest_version=str(payload["manifest_version"]),
        protocol_hash=_validate_sha256(str(payload["protocol_hash"]), "protocol_hash"),
        binding_hash=_validate_sha256(str(payload["binding_hash"]), "binding_hash"),
        jobs=tuple(jobs),
    )


def save_h4_holdout_variance_run_manifest(manifest: H4HoldoutVarianceRunManifest, path: str | Path) -> Path:
    return _atomic_write(Path(path), manifest.to_dict())


def load_h4_holdout_variance_run_manifest(
    path: str | Path,
    *,
    protocol: H4HoldoutVarianceProtocol,
    expected_binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation,
) -> H4HoldoutVarianceRunManifest:
    source = Path(path)
    raw = json.loads(source.read_bytes())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("manifest"), Mapping):
        raise H4HoldoutVarianceRunError("variance-pilot run manifest is malformed")
    _require_canonical_file(source, raw, "variance-pilot run manifest")
    payload = raw["manifest"]
    stored_hash = _validate_sha256(str(raw.get("manifest_hash")), "manifest_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise H4HoldoutVarianceRunError("manifest_hash does not match manifest payload")
    manifest = _manifest_from_payload(payload)
    if manifest.manifest_hash != stored_hash:
        raise H4HoldoutVarianceRunError("loaded run manifest is not canonical")
    if manifest.protocol_hash != protocol.payload_hash or manifest.binding_hash != expected_binding.binding_hash:
        raise H4HoldoutVarianceRunError("run manifest protocol/binding does not match")
    expected = _planned_states(protocol, expected_binding, population)
    if len(manifest.jobs) != len(expected):
        raise H4HoldoutVarianceRunError("run manifest does not match the canonical 100-job plan")
    for actual, planned in zip(manifest.jobs, expected, strict=True):
        if actual.identity_payload() != planned.identity_payload():
            raise H4HoldoutVarianceRunError("run manifest does not match the canonical 100-job plan")
    return manifest


def initialize_h4_holdout_variance_run(
    *, protocol: H4HoldoutVarianceProtocol, binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation, work_root: str | Path,
) -> H4HoldoutVarianceRunManifest:
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("trajectory-configs", "arm-snapshots", "pair-checkpoints", "cluster-checkpoints"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    save_h4_holdout_variance_execution_binding(binding, root / "binding.json")
    for job in iter_h4_holdout_variance_jobs(protocol):
        config = build_h4_holdout_variance_trajectory_config(
            protocol=protocol, binding=binding, population=population, job=job
        )
        save_h4_holdout_variance_trajectory_config(config, _config_path(root, job))
    manifest_path = root / "run-manifest.json"
    if manifest_path.exists():
        return load_h4_holdout_variance_run_manifest(
            manifest_path, protocol=protocol, expected_binding=binding, population=population
        )
    manifest = H4HoldoutVarianceRunManifest(
        protocol_hash=protocol.payload_hash,
        binding_hash=binding.binding_hash,
        jobs=_planned_states(protocol, binding, population),
    )
    if (
        manifest.planned_pair_count != 50
        or manifest.planned_cluster_count != 10
        or manifest.planned_engine_hands != 100_000
    ):
        raise H4HoldoutVarianceRunError("run manifest does not reproduce frozen 50-pair/10-cluster/100000-hand design")
    save_h4_holdout_variance_run_manifest(manifest, manifest_path)
    return manifest


def _arm_snapshot_from_result(
    *, job: H4HoldoutVarianceJob, config: H4HoldoutVarianceTrajectoryConfig,
    match_id: str, result: PairedResult, elapsed_ns: int,
) -> H4HoldoutVarianceArmSnapshot:
    if not isinstance(result, PairedResult):
        raise TypeError("job executor must return PairedResult")
    if result.statistical_unit is not StatisticalUnit.TRAJECTORY_CLUSTER:
        raise H4HoldoutVarianceRunError("variance-pilot arm result has wrong statistical unit")
    if result.seed != job.master_seed or result.num_hands != 500:
        raise H4HoldoutVarianceRunError("variance-pilot arm result does not match frozen job")
    record = result.match_record
    if record is None:
        raise H4HoldoutVarianceRunError("variance-pilot arm result must retain a MatchRecord")
    if record.match_id != match_id or record.master_seed != job.master_seed:
        raise H4HoldoutVarianceRunError("variance-pilot MatchRecord identity does not match")
    if record.experiment_config_hash != config.config_hash:
        raise H4HoldoutVarianceRunError("variance-pilot MatchRecord config hash does not match")
    if record.experiment_id != config.run_provenance["experiment_id"]:
        raise H4HoldoutVarianceRunError("variance-pilot MatchRecord experiment identity does not match")
    if record.game_config_hash != config.binding.game_config_hash:
        raise H4HoldoutVarianceRunError("variance-pilot MatchRecord game hash does not match")
    valid_shape = result.num_valid_pairs == 1 and len(result.paired_differences) == 1 and not result.failed_runs
    failed_shape = result.num_valid_pairs == 0 and not result.paired_differences and bool(result.failed_runs)
    if not (valid_shape or failed_shape):
        raise H4HoldoutVarianceRunError("variance-pilot arm result has inconsistent validity fields")
    status = "completed" if valid_shape else "failed"
    return H4HoldoutVarianceArmSnapshot(
        binding_hash=config.binding.binding_hash,
        job_key=h4_holdout_variance_job_key(job), arm_id=job.arm_id,
        holdout_unit_id=job.holdout_unit_id, master_seed=job.master_seed,
        hands_per_position=500, trajectory_config_hash=config.config_hash,
        match_id=match_id, match_record_hash=record.record_hash, status=status,
        paired_differences=tuple(float(value) for value in result.paired_differences),
        failed_runs=tuple(dict(item) for item in result.failed_runs), elapsed_ns=elapsed_ns,
        recorded_engine_hands=len(record.hands), match_record_payload=record.to_dict(),
    )


def save_h4_holdout_variance_arm_snapshot(snapshot: H4HoldoutVarianceArmSnapshot, path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        raw = json.loads(destination.read_bytes())
        if raw != snapshot.to_dict() or destination.read_bytes() != _canonical_file_bytes(raw):
            raise H4HoldoutVarianceRunError("existing arm snapshot does not match immutable evidence")
        return destination
    return _atomic_write(destination, snapshot.to_dict())


def load_h4_holdout_variance_arm_snapshot(
    path: str | Path, *, protocol: H4HoldoutVarianceProtocol,
    expected_binding: H4HoldoutVarianceExecutionBinding, population: H4HoldoutPopulation,
    expected_job: H4HoldoutVarianceJob,
) -> H4HoldoutVarianceArmSnapshot:
    source = Path(path)
    raw = json.loads(source.read_bytes())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("snapshot"), Mapping):
        raise H4HoldoutVarianceRunError("variance-pilot arm snapshot is malformed")
    _require_canonical_file(source, raw, "variance-pilot arm snapshot")
    payload = raw["snapshot"]
    stored_hash = _validate_sha256(str(raw.get("snapshot_hash")), "snapshot_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise H4HoldoutVarianceRunError("snapshot_hash does not match arm snapshot payload")
    failures_raw, differences_raw, record_raw = payload.get("failed_runs"), payload.get("paired_differences"), payload.get("match_record")
    if not isinstance(failures_raw, list) or not isinstance(differences_raw, list) or not isinstance(record_raw, Mapping):
        raise H4HoldoutVarianceRunError("arm snapshot fields are malformed")
    snapshot = H4HoldoutVarianceArmSnapshot(
        snapshot_version=str(payload["snapshot_version"]), binding_hash=_validate_sha256(str(payload["binding_hash"]), "binding_hash"),
        job_key=str(payload["job_key"]), arm_id=str(payload["arm_id"]), holdout_unit_id=str(payload["holdout_unit_id"]),
        master_seed=int(payload["master_seed"]), hands_per_position=int(payload["hands_per_position"]),
        trajectory_config_hash=_validate_sha256(str(payload["trajectory_config_hash"]), "trajectory_config_hash"),
        match_id=str(payload["match_id"]), match_record_hash=_validate_sha256(str(payload["match_record_hash"]), "match_record_hash"),
        status=str(payload["status"]), paired_differences=tuple(float(value) for value in differences_raw),
        failed_runs=tuple(dict(item) for item in failures_raw), elapsed_ns=int(payload["elapsed_ns"]),
        recorded_engine_hands=int(payload["recorded_engine_hands"]), match_record_payload=dict(record_raw),
    )
    if snapshot.snapshot_hash != stored_hash:
        raise H4HoldoutVarianceRunError("loaded arm snapshot is not canonical")
    expected_config = build_h4_holdout_variance_trajectory_config(
        protocol=protocol, binding=expected_binding, population=population, job=expected_job
    )
    expected_match_id = h4_holdout_variance_match_id(
        binding=expected_binding, trajectory_config_hash=expected_config.config_hash
    )
    if (
        snapshot.binding_hash != expected_binding.binding_hash
        or snapshot.job_key != h4_holdout_variance_job_key(expected_job)
        or snapshot.trajectory_config_hash != expected_config.config_hash
        or snapshot.match_id != expected_match_id
        or snapshot.arm_id != expected_job.arm_id
        or snapshot.holdout_unit_id != expected_job.holdout_unit_id
        or snapshot.master_seed != expected_job.master_seed
    ):
        raise H4HoldoutVarianceRunError("arm snapshot identity does not match frozen job")
    record = snapshot.match_record
    if record.match_id != expected_match_id or record.experiment_config_hash != expected_config.config_hash:
        raise H4HoldoutVarianceRunError("arm snapshot MatchRecord identity does not match")
    return snapshot


def _save_checkpoint(checkpoint: Any, path: Path) -> Path:
    if path.exists():
        raw = json.loads(path.read_bytes())
        if raw != checkpoint.to_dict() or path.read_bytes() != _canonical_file_bytes(raw):
            raise H4HoldoutVarianceRunError("existing checkpoint differs from immutable evidence")
        return path
    return _atomic_write(path, checkpoint.to_dict())


def _pair_checkpoint_from_execution(
    *, binding: H4HoldoutVarianceExecutionBinding, execution: H4HoldoutVariancePairExecution,
    treatment: H4HoldoutVarianceArmSnapshot, control: H4HoldoutVarianceArmSnapshot,
) -> H4HoldoutVariancePairCheckpoint:
    return H4HoldoutVariancePairCheckpoint(
        binding_hash=binding.binding_hash,
        pair_key=_pair_key(execution.holdout_unit_id, execution.master_seed),
        holdout_unit_id=execution.holdout_unit_id, master_seed=execution.master_seed,
        treatment_snapshot_hash=treatment.snapshot_hash, control_snapshot_hash=control.snapshot_hash,
        treatment_record_hash=execution.treatment_record_hash, control_record_hash=execution.control_record_hash,
        scientifically_valid=execution.scientifically_valid,
        unit_contrast_bb100=None if execution.unit_contrast_bb100 is None else float(execution.unit_contrast_bb100),
    )


def _load_pair_checkpoint(
    path: Path, *, binding: H4HoldoutVarianceExecutionBinding,
    treatment: H4HoldoutVarianceArmSnapshot, control: H4HoldoutVarianceArmSnapshot,
) -> H4HoldoutVariancePairCheckpoint:
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("checkpoint"), Mapping):
        raise H4HoldoutVarianceRunError("pair checkpoint is malformed")
    _require_canonical_file(path, raw, "variance-pilot pair checkpoint")
    payload = raw["checkpoint"]
    stored_hash = _validate_sha256(str(raw.get("checkpoint_hash")), "checkpoint_hash")
    checkpoint = H4HoldoutVariancePairCheckpoint(
        checkpoint_version=str(payload["checkpoint_version"]), binding_hash=_validate_sha256(str(payload["binding_hash"]), "binding_hash"),
        pair_key=str(payload["pair_key"]), holdout_unit_id=str(payload["holdout_unit_id"]), master_seed=int(payload["master_seed"]),
        treatment_snapshot_hash=_validate_sha256(str(payload["treatment_snapshot_hash"]), "treatment_snapshot_hash"),
        control_snapshot_hash=_validate_sha256(str(payload["control_snapshot_hash"]), "control_snapshot_hash"),
        treatment_record_hash=_validate_sha256(str(payload["treatment_record_hash"]), "treatment_record_hash"),
        control_record_hash=_validate_sha256(str(payload["control_record_hash"]), "control_record_hash"),
        scientifically_valid=bool(payload["scientifically_valid"]),
        unit_contrast_bb100=None if payload.get("unit_contrast_bb100") is None else float(payload["unit_contrast_bb100"]),
    )
    if _sha256_canonical(payload) != stored_hash or checkpoint.checkpoint_hash != stored_hash:
        raise H4HoldoutVarianceRunError("pair checkpoint hash is invalid")
    if checkpoint.binding_hash != binding.binding_hash or checkpoint.pair_key != _pair_key(treatment.holdout_unit_id, treatment.master_seed):
        raise H4HoldoutVarianceRunError("pair checkpoint binding/identity does not match")
    if (
        checkpoint.treatment_snapshot_hash != treatment.snapshot_hash
        or checkpoint.control_snapshot_hash != control.snapshot_hash
        or checkpoint.treatment_record_hash != treatment.match_record_hash
        or checkpoint.control_record_hash != control.match_record_hash
    ):
        raise H4HoldoutVarianceRunError("pair checkpoint hashes do not match arm evidence")
    return checkpoint


def _cluster_checkpoint_from_pairs(
    *, binding: H4HoldoutVarianceExecutionBinding, seed: int,
    pairs: tuple[H4HoldoutVariancePairCheckpoint, ...],
) -> H4HoldoutVarianceClusterCheckpoint:
    if tuple(pair.holdout_unit_id for pair in pairs) != H4_HOLDOUT_UNIT_IDS:
        raise H4HoldoutVarianceRunError("cluster checkpoint requires all five units in canonical order")
    scientifically_valid = all(pair.scientifically_valid for pair in pairs)
    population_contrast: float | None = None
    if scientifically_valid:
        contrasts = {
            pair.holdout_unit_id: float(pair.unit_contrast_bb100)
            for pair in pairs
            if pair.unit_contrast_bb100 is not None
        }
        population_contrast = population_contrast_for_seed(contrasts)
    return H4HoldoutVarianceClusterCheckpoint(
        binding_hash=binding.binding_hash, cluster_key=_cluster_key(seed), master_seed=seed,
        pair_checkpoint_hashes=tuple((pair.holdout_unit_id, pair.checkpoint_hash) for pair in pairs),
        scientifically_valid=scientifically_valid, population_contrast_bb100=population_contrast,
    )


def _load_cluster_checkpoint(
    path: Path, *, binding: H4HoldoutVarianceExecutionBinding,
    pairs: tuple[H4HoldoutVariancePairCheckpoint, ...], seed: int,
) -> H4HoldoutVarianceClusterCheckpoint:
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("checkpoint"), Mapping):
        raise H4HoldoutVarianceRunError("cluster checkpoint is malformed")
    _require_canonical_file(path, raw, "variance-pilot cluster checkpoint")
    payload = raw["checkpoint"]
    stored_hash = _validate_sha256(str(raw.get("checkpoint_hash")), "checkpoint_hash")
    pair_hashes_raw = payload.get("pair_checkpoint_hashes")
    if not isinstance(pair_hashes_raw, list):
        raise H4HoldoutVarianceRunError("cluster pair hashes must be an array")
    pair_hashes = tuple(
        (str(item["holdout_unit_id"]), _validate_sha256(str(item["checkpoint_hash"]), "pair_checkpoint_hash"))
        for item in pair_hashes_raw if isinstance(item, Mapping)
    )
    checkpoint = H4HoldoutVarianceClusterCheckpoint(
        checkpoint_version=str(payload["checkpoint_version"]), binding_hash=_validate_sha256(str(payload["binding_hash"]), "binding_hash"),
        cluster_key=str(payload["cluster_key"]), master_seed=int(payload["master_seed"]), pair_checkpoint_hashes=pair_hashes,
        scientifically_valid=bool(payload["scientifically_valid"]),
        population_contrast_bb100=None if payload.get("population_contrast_bb100") is None else float(payload["population_contrast_bb100"]),
    )
    expected = _cluster_checkpoint_from_pairs(binding=binding, seed=seed, pairs=pairs)
    if _sha256_canonical(payload) != stored_hash or checkpoint.checkpoint_hash != stored_hash or checkpoint != expected:
        raise H4HoldoutVarianceRunError("cluster checkpoint differs from pair evidence")
    return checkpoint


def _require_run_store(store: object) -> H4HoldoutVarianceRunStore:
    if not isinstance(store, H4HoldoutVarianceRunStore):
        raise H4HoldoutVarianceRunError("variance-pilot run requires durable save/load/list match storage")
    return store


def _recover_or_verify_store_record(
    store: H4HoldoutVarianceRunStore,
    snapshot: H4HoldoutVarianceArmSnapshot,
    known_ids: set[str],
) -> None:
    if snapshot.match_id not in known_ids:
        store.save_match(snapshot.match_record)
        known_ids.add(snapshot.match_id)
    try:
        persisted = store.load_match(snapshot.match_id)
    except Exception as exc:
        raise H4HoldoutVarianceRunError("variance-pilot MatchRecord is not durably persisted") from exc
    if not isinstance(persisted, MatchRecord) or persisted.record_hash != snapshot.match_record_hash:
        raise H4HoldoutVarianceRunError("durable MatchRecord differs from arm snapshot")


def _load_exact_config(
    *, protocol: H4HoldoutVarianceProtocol, binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation, root: Path, job: H4HoldoutVarianceJob,
) -> H4HoldoutVarianceTrajectoryConfig:
    path = _config_path(root, job)
    if not path.exists():
        raise H4HoldoutVarianceRunError(f"trajectory config is missing for {h4_holdout_variance_job_key(job)}")
    try:
        return load_h4_holdout_variance_trajectory_config(
            path, protocol=protocol, expected_binding=binding, population=population, expected_job=job
        )
    except (H4HoldoutVarianceBindingError, KeyError, TypeError, ValueError) as exc:
        raise H4HoldoutVarianceRunError(
            f"trajectory config is invalid for {h4_holdout_variance_job_key(job)}"
        ) from exc


def _validate_directory_membership(root: Path, expected: set[str], label: str) -> None:
    actual = {path.name for path in root.glob("*.json")}
    unexpected = actual - expected
    if unexpected:
        raise H4HoldoutVarianceRunError(f"unexpected {label} files: {', '.join(sorted(unexpected))}")


def _replace_manifest_job(
    manifest: H4HoldoutVarianceRunManifest, index: int, state: H4HoldoutVarianceManifestJobState,
) -> H4HoldoutVarianceRunManifest:
    jobs = list(manifest.jobs)
    jobs[index] = state
    return replace(manifest, jobs=tuple(jobs))


def _reconcile_manifest(
    *, protocol: H4HoldoutVarianceProtocol, binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation, root: Path, store: H4HoldoutVarianceRunStore,
    manifest: H4HoldoutVarianceRunManifest,
) -> H4HoldoutVarianceRunManifest:
    try:
        known_ids = set(store.list_match_ids())
    except Exception as exc:
        raise H4HoldoutVarianceRunError("could not enumerate durable variance-pilot MatchRecords") from exc
    updated = manifest
    for index, (job, state) in enumerate(zip(iter_h4_holdout_variance_jobs(protocol), updated.jobs, strict=True)):
        path = _snapshot_path(root, job)
        if path.exists():
            snapshot = load_h4_holdout_variance_arm_snapshot(
                path, protocol=protocol, expected_binding=binding, population=population, expected_job=job
            )
            _recover_or_verify_store_record(store, snapshot, known_ids)
            terminal = replace(
                state, status=snapshot.status, arm_snapshot_hash=snapshot.snapshot_hash,
                match_record_hash=snapshot.match_record_hash,
            )
            if state.status in _TERMINAL_STATUSES and state != terminal:
                raise H4HoldoutVarianceRunError("terminal manifest state differs from immutable arm snapshot")
            if state != terminal:
                updated = _replace_manifest_job(updated, index, terminal)
            continue
        if state.status in _TERMINAL_STATUSES:
            raise H4HoldoutVarianceRunError("terminal manifest state has no arm snapshot")
        if state.match_id in known_ids:
            raise H4HoldoutVarianceRunError("orphan durable MatchRecord exists without arm snapshot; refusing resume")
        if state.status == "running":
            updated = _replace_manifest_job(updated, index, replace(state, status="planned"))
    if updated != manifest:
        save_h4_holdout_variance_run_manifest(updated, root / "run-manifest.json")
    return updated


def _snapshots_by_pair(
    *, protocol: H4HoldoutVarianceProtocol, binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation, root: Path, manifest: H4HoldoutVarianceRunManifest,
) -> dict[tuple[str, int], dict[str, tuple[H4HoldoutVarianceJob, H4HoldoutVarianceArmSnapshot]]]:
    grouped: dict[tuple[str, int], dict[str, tuple[H4HoldoutVarianceJob, H4HoldoutVarianceArmSnapshot]]] = {}
    for job, state in zip(iter_h4_holdout_variance_jobs(protocol), manifest.jobs, strict=True):
        if state.status not in _TERMINAL_STATUSES:
            continue
        snapshot = load_h4_holdout_variance_arm_snapshot(
            _snapshot_path(root, job), protocol=protocol, expected_binding=binding,
            population=population, expected_job=job,
        )
        grouped.setdefault((job.holdout_unit_id, job.master_seed), {})[job.arm_id] = (job, snapshot)
    return grouped


def _load_or_finalize_pair_checkpoints(
    *, protocol: H4HoldoutVarianceProtocol, binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation, root: Path, store: H4HoldoutVarianceRunStore,
    manifest: H4HoldoutVarianceRunManifest,
) -> tuple[H4HoldoutVariancePairCheckpoint, ...]:
    grouped = _snapshots_by_pair(
        protocol=protocol, binding=binding, population=population, root=root, manifest=manifest
    )
    checkpoints: list[H4HoldoutVariancePairCheckpoint] = []
    for unit_id in H4_HOLDOUT_UNIT_IDS:
        for seed in H4_VARIANCE_MASTER_SEEDS:
            arms = grouped.get((unit_id, seed), {})
            if set(arms) != {"control", "treatment"}:
                continue
            control_job, control = arms["control"]
            treatment_job, treatment = arms["treatment"]
            path = _pair_checkpoint_path(root, unit_id, seed)
            if path.exists():
                checkpoint = _load_pair_checkpoint(
                    path, binding=binding, treatment=treatment, control=control
                )
            else:
                execution = finalize_h4_holdout_variance_pair(
                    protocol=protocol, treatment_job=treatment_job, control_job=control_job,
                    treatment_result=treatment.to_paired_result(), control_result=control.to_paired_result(),
                    benchmark_store=store,
                )
                checkpoint = _pair_checkpoint_from_execution(
                    binding=binding, execution=execution, treatment=treatment, control=control
                )
                _save_checkpoint(checkpoint, path)
            checkpoints.append(checkpoint)
    return tuple(checkpoints)


def _load_or_finalize_cluster_checkpoints(
    *, binding: H4HoldoutVarianceExecutionBinding, root: Path,
    pair_checkpoints: tuple[H4HoldoutVariancePairCheckpoint, ...],
) -> tuple[H4HoldoutVarianceClusterCheckpoint, ...]:
    by_seed: dict[int, dict[str, H4HoldoutVariancePairCheckpoint]] = {}
    for pair in pair_checkpoints:
        by_seed.setdefault(pair.master_seed, {})[pair.holdout_unit_id] = pair
    clusters: list[H4HoldoutVarianceClusterCheckpoint] = []
    for seed in H4_VARIANCE_MASTER_SEEDS:
        units = by_seed.get(seed, {})
        if set(units) != set(H4_HOLDOUT_UNIT_IDS):
            continue
        pairs = tuple(units[unit] for unit in H4_HOLDOUT_UNIT_IDS)
        path = _cluster_checkpoint_path(root, seed)
        if path.exists():
            checkpoint = _load_cluster_checkpoint(
                path, binding=binding, pairs=pairs, seed=seed
            )
        else:
            checkpoint = _cluster_checkpoint_from_pairs(binding=binding, seed=seed, pairs=pairs)
            _save_checkpoint(checkpoint, path)
        clusters.append(checkpoint)
    return tuple(clusters)


def assert_h4_holdout_variance_run_ready(
    *, protocol: H4HoldoutVarianceProtocol, binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation, work_root: str | Path,
    benchmark_store: H4HoldoutVarianceRunStore,
) -> H4HoldoutVarianceRunReadiness:
    store = _require_run_store(benchmark_store)
    root = Path(work_root)
    binding_path, manifest_path = root / "binding.json", root / "run-manifest.json"
    if not binding_path.exists() or not manifest_path.exists():
        raise H4HoldoutVarianceRunError("variance-pilot run plan is not initialized")
    if load_h4_holdout_variance_execution_binding(binding_path) != binding:
        raise H4HoldoutVarianceRunError("persisted variance-pilot binding does not match")
    jobs = iter_h4_holdout_variance_jobs(protocol)
    expected_configs = {f"{h4_holdout_variance_job_key(job)}.json" for job in jobs}
    _validate_directory_membership(root / "trajectory-configs", expected_configs, "trajectory config")
    for job in jobs:
        _load_exact_config(protocol=protocol, binding=binding, population=population, root=root, job=job)
    _validate_directory_membership(root / "arm-snapshots", expected_configs, "arm snapshot")
    expected_pairs = {f"{_pair_key(unit, seed)}.json" for unit in H4_HOLDOUT_UNIT_IDS for seed in H4_VARIANCE_MASTER_SEEDS}
    _validate_directory_membership(root / "pair-checkpoints", expected_pairs, "pair checkpoint")
    expected_clusters = {f"{_cluster_key(seed)}.json" for seed in H4_VARIANCE_MASTER_SEEDS}
    _validate_directory_membership(root / "cluster-checkpoints", expected_clusters, "cluster checkpoint")
    manifest = load_h4_holdout_variance_run_manifest(
        manifest_path, protocol=protocol, expected_binding=binding, population=population
    )
    manifest = _reconcile_manifest(
        protocol=protocol, binding=binding, population=population, root=root, store=store, manifest=manifest
    )
    pairs = _load_or_finalize_pair_checkpoints(
        protocol=protocol, binding=binding, population=population, root=root, store=store, manifest=manifest
    )
    clusters = _load_or_finalize_cluster_checkpoints(binding=binding, root=root, pair_checkpoints=pairs)
    completed = sum(item.status == "completed" for item in manifest.jobs)
    failed = sum(item.status == "failed" for item in manifest.jobs)
    valid_pairs = sum(item.scientifically_valid for item in pairs)
    valid_clusters = sum(item.scientifically_valid for item in clusters)
    return H4HoldoutVarianceRunReadiness(
        binding_hash=binding.binding_hash, planned_jobs=100, planned_pairs=50, planned_clusters=10,
        planned_engine_hands=100_000, terminal_jobs=manifest.terminal_jobs,
        completed_jobs=completed, failed_jobs=failed, pair_count=len(pairs), valid_pairs=valid_pairs,
        invalid_pairs=len(pairs) - valid_pairs, cluster_count=len(clusters), valid_clusters=valid_clusters,
        invalid_clusters=len(clusters) - valid_clusters,
    )


def execute_h4_holdout_variance_run(
    *, protocol: H4HoldoutVarianceProtocol, binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation, work_root: str | Path,
    benchmark_store: H4HoldoutVarianceRunStore, job_executor: H4HoldoutVarianceJobExecutor,
) -> H4HoldoutVarianceRunResult:
    """Execute/resume canonical jobs; this infrastructure does not itself define authority semantics."""
    root = Path(work_root)
    initialize_h4_holdout_variance_run(
        protocol=protocol, binding=binding, population=population, work_root=root
    )
    store = _require_run_store(benchmark_store)
    readiness = assert_h4_holdout_variance_run_ready(
        protocol=protocol, binding=binding, population=population, work_root=root, benchmark_store=store
    )
    del readiness
    manifest = load_h4_holdout_variance_run_manifest(
        root / "run-manifest.json", protocol=protocol, expected_binding=binding, population=population
    )
    manifest = _reconcile_manifest(
        protocol=protocol, binding=binding, population=population, root=root, store=store, manifest=manifest
    )
    jobs = iter_h4_holdout_variance_jobs(protocol)
    for index, (job, state) in enumerate(zip(jobs, manifest.jobs, strict=True)):
        if state.status in _TERMINAL_STATUSES:
            continue
        config = _load_exact_config(
            protocol=protocol, binding=binding, population=population, root=root, job=job
        )
        running = replace(state, status="running")
        manifest = _replace_manifest_job(manifest, index, running)
        save_h4_holdout_variance_run_manifest(manifest, root / "run-manifest.json")
        started = time.perf_counter_ns()
        result = job_executor(job, config, state.match_id)
        elapsed = time.perf_counter_ns() - started
        snapshot = _arm_snapshot_from_result(
            job=job, config=config, match_id=state.match_id, result=result, elapsed_ns=elapsed
        )
        # Immutable snapshot first. If central persistence fails, readiness can recover
        # the exact MatchRecord from this snapshot without repeating scientific work.
        save_h4_holdout_variance_arm_snapshot(snapshot, _snapshot_path(root, job))
        store.save_match(snapshot.match_record)
        persisted = store.load_match(snapshot.match_id)
        if persisted.record_hash != snapshot.match_record_hash:
            raise H4HoldoutVarianceRunError("durable MatchRecord differs after persistence")
        terminal = replace(
            running, status=snapshot.status, arm_snapshot_hash=snapshot.snapshot_hash,
            match_record_hash=snapshot.match_record_hash,
        )
        manifest = _replace_manifest_job(manifest, index, terminal)
        save_h4_holdout_variance_run_manifest(manifest, root / "run-manifest.json")

    ready = assert_h4_holdout_variance_run_ready(
        protocol=protocol, binding=binding, population=population, work_root=root, benchmark_store=store
    )
    return H4HoldoutVarianceRunResult(
        binding_hash=binding.binding_hash, terminal_jobs=ready.terminal_jobs,
        completed_jobs=ready.completed_jobs, failed_jobs=ready.failed_jobs,
        pair_count=ready.pair_count, valid_pairs=ready.valid_pairs, invalid_pairs=ready.invalid_pairs,
        cluster_count=ready.cluster_count, valid_clusters=ready.valid_clusters,
        invalid_clusters=ready.invalid_clusters, scientifically_complete=ready.scientifically_complete,
    )
