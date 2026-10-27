"""Hash-bound execution and trajectory provenance for the frozen Spec-37 variance pilot."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokerbench.agents.profiles import PARAMETER_SCHEMA_VERSION
from pokerbench.agents.strength_gates import PARAMETRIC_POLICY_VERSION
from pokerbench.engine.public_action import PUBLIC_ACTION_TRACE_VERSION
from pokerbench.evaluation.h4_holdout_acceptance import (
    H4_HOLDOUT_ACCEPTANCE_HASH,
    H4HoldoutAcceptance,
    require_h4_holdout_acceptance_closed,
)
from pokerbench.experiments.config import ExecutionEnvironment, capture_execution_environment
from pokerbench.experiments.h4_holdout_population import (
    H4_HOLDOUT_POPULATION_HASH,
    H4HoldoutPopulation,
)
from pokerbench.experiments.h4_holdout_variance_execution import (
    H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4_HOLDOUT_UNIT_IDS,
    H4_VARIANCE_MASTER_SEEDS,
    H4HoldoutVarianceJob,
    H4HoldoutVarianceProtocol,
    H4_HOLDOUT_VARIANCE_PROTOCOL_HASH,
    expected_h4_holdout_variance_payload,
    iter_h4_holdout_variance_jobs,
)
from pokerbench.replay.archive import MATCH_RECORD_SCHEMA_VERSION


H4_HOLDOUT_VARIANCE_EXECUTION_BINDING_VERSION = (
    "phase2-h4-holdout-variance-execution-binding-v1"
)
H4_HOLDOUT_VARIANCE_TRAJECTORY_CONFIG_VERSION = (
    "phase2-h4-holdout-variance-trajectory-config-v1"
)
H4_HOLDOUT_VARIANCE_PERSISTENCE_LAYOUT_VERSION = (
    "phase2-h4-holdout-variance-persistence-layout-v1"
)
H4_HOLDOUT_VARIANCE_PERSISTENCE_KEYS = (
    "arm_snapshots",
    "cluster_checkpoints",
    "database",
    "pair_checkpoints",
    "trajectory_configs",
    "work_root",
)


class H4HoldoutVarianceBindingError(ValueError):
    """Raised when Spec-37 execution/provenance bindings are inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_file_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise H4HoldoutVarianceBindingError(
            f"{field_name} must be a lowercase SHA-256 hex digest"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise H4HoldoutVarianceBindingError(
            f"{field_name} must be a lowercase SHA-256 hex digest"
        ) from exc
    if value.lower() != value:
        raise H4HoldoutVarianceBindingError(f"{field_name} must be lowercase")
    return value


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise H4HoldoutVarianceBindingError(f"{field_name} must be a positive integer")
    return value


def _hardware_fingerprint(environment: ExecutionEnvironment) -> str:
    return _sha256_canonical(
        {
            "cpu_model": environment.cpu_model,
            "cpu_core_count": environment.cpu_core_count,
            "cpu_architecture": environment.cpu_architecture,
        }
    )


def _atomic_write(path: Path, payload: Any) -> Path:
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
        raise H4HoldoutVarianceBindingError(
            f"{what} file is not canonical JSON with one trailing LF"
        )


def _clean_scientific_git_commit(repo_root: Path) -> str:
    scientific_paths = ["src", "configs", "docs/specs", "pyproject.toml", "poetry.lock"]
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *scientific_paths],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise H4HoldoutVarianceBindingError(
                "scientific source/configuration paths must be clean and committed before variance-pilot binding"
            )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise H4HoldoutVarianceBindingError(
            "could not resolve clean scientific Git state"
        ) from exc
    if not commit:
        raise H4HoldoutVarianceBindingError("could not resolve Git commit")
    return commit


def _frozen_dependency_hash(name: str) -> str:
    dependencies = expected_h4_holdout_variance_payload()["bindings"]["adaptive_dependencies"]
    dependency = dependencies[name]
    if not isinstance(dependency, Mapping):
        raise H4HoldoutVarianceBindingError(f"frozen dependency {name} is not hash-bound")
    return _validate_sha256(str(dependency["payload_hash"]), f"{name}_hash")


def _expected_rng_configuration() -> tuple[tuple[str, Any], ...]:
    rng = expected_h4_holdout_variance_payload()["rng"]
    if not isinstance(rng, Mapping):
        raise H4HoldoutVarianceBindingError("frozen RNG configuration is malformed")
    return tuple(sorted((str(key), value) for key, value in rng.items()))


@dataclass(frozen=True)
class H4HoldoutVarianceExecutionAuthorityReference:
    """Versioned execution-authority identity consumed by the generic binding layer."""

    authority_id: str
    authority_version: str
    payload_hash: str
    authorized_protocol_id: str
    authorized_protocol_version: str
    authorized_protocol_hash: str

    def __post_init__(self) -> None:
        if not self.authority_id or not self.authority_version:
            raise H4HoldoutVarianceBindingError("execution authority identity must be non-empty")
        _validate_sha256(self.payload_hash, "execution_authority_hash")
        _validate_sha256(self.authorized_protocol_hash, "authorized_protocol_hash")
        if not self.authorized_protocol_id or not self.authorized_protocol_version:
            raise H4HoldoutVarianceBindingError(
                "execution authority protocol identity must be non-empty"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "authority_id": self.authority_id,
            "authority_version": self.authority_version,
            "payload_hash": self.payload_hash,
            "authorized_protocol_id": self.authorized_protocol_id,
            "authorized_protocol_version": self.authorized_protocol_version,
            "authorized_protocol_hash": self.authorized_protocol_hash,
        }


@dataclass(frozen=True)
class H4HoldoutVarianceExecutionBinding:
    """Run-wide immutable Spec-37 provenance captured before any execution."""

    protocol_id: str
    protocol_version: str
    protocol_hash: str
    planning_spec_id: str
    population_id: str
    population_version: str
    population_role: str
    population_hash: str
    holdout_acceptance_hash: str
    unit_acceptance_hashes: tuple[tuple[str, str], ...]
    game_config_hash: str
    master_seeds: tuple[int, ...]
    holdout_unit_ids: tuple[str, ...]
    rng_configuration: tuple[tuple[str, Any], ...]
    adaptive_learning_protocol_hash: str
    posterior_to_range_protocol_hash: str
    adaptive_policy_protocol_hash: str
    adaptive_lifecycle_protocol_hash: str
    opponent_model_version: str
    opportunity_definition_version: str
    public_action_trace_version: str
    parameter_schema_version: str
    parametric_policy_version: str
    match_record_schema_version: str
    execution_enforcement_version: str
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
    process_model: str
    persistence_paths: tuple[tuple[str, str], ...]
    execution_authority: H4HoldoutVarianceExecutionAuthorityReference
    persistence_layout_version: str = H4_HOLDOUT_VARIANCE_PERSISTENCE_LAYOUT_VERSION
    binding_version: str = H4_HOLDOUT_VARIANCE_EXECUTION_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.binding_version != H4_HOLDOUT_VARIANCE_EXECUTION_BINDING_VERSION:
            raise H4HoldoutVarianceBindingError("unsupported variance-pilot binding version")
        if self.persistence_layout_version != H4_HOLDOUT_VARIANCE_PERSISTENCE_LAYOUT_VERSION:
            raise H4HoldoutVarianceBindingError("unsupported variance-pilot persistence layout")
        for field_name in (
            "protocol_hash",
            "population_hash",
            "holdout_acceptance_hash",
            "game_config_hash",
            "adaptive_learning_protocol_hash",
            "posterior_to_range_protocol_hash",
            "adaptive_policy_protocol_hash",
            "adaptive_lifecycle_protocol_hash",
            "poetry_lock_hash",
            "hardware_fingerprint",
        ):
            _validate_sha256(str(getattr(self, field_name)), field_name)
        if not self.protocol_id or not self.protocol_version or not self.planning_spec_id:
            raise H4HoldoutVarianceBindingError("scientific protocol identity must be non-empty")
        if not self.population_id or not self.population_version or not self.population_role:
            raise H4HoldoutVarianceBindingError("holdout population identity must be non-empty")
        if self.master_seeds != H4_VARIANCE_MASTER_SEEDS:
            raise H4HoldoutVarianceBindingError("binding master seeds differ from frozen Spec 37")
        if self.holdout_unit_ids != H4_HOLDOUT_UNIT_IDS:
            raise H4HoldoutVarianceBindingError("binding holdout units differ from frozen Spec 37")
        if tuple(unit_id for unit_id, _ in self.unit_acceptance_hashes) != H4_HOLDOUT_UNIT_IDS:
            raise H4HoldoutVarianceBindingError("binding unit-acceptance identities are incomplete")
        for _, digest in self.unit_acceptance_hashes:
            _validate_sha256(digest, "unit_acceptance_hash")
        if not self.git_commit or self.engine_version != f"git:{self.git_commit}":
            raise H4HoldoutVarianceBindingError("engine_version must be derived from git_commit")
        if not self.python_version or not self.dependency_lockfile:
            raise H4HoldoutVarianceBindingError("software provenance must be non-empty")
        if not self.cpu_model or not self.cpu_architecture or not self.os_identifier:
            raise H4HoldoutVarianceBindingError("hardware/OS provenance must be non-empty")
        _positive_int(self.cpu_core_count, "cpu_core_count")
        _positive_int(self.ram_bytes, "ram_bytes")
        _positive_int(self.workers, "workers")
        if not self.process_model:
            raise H4HoldoutVarianceBindingError("process_model must not be empty")
        if tuple(key for key, _ in self.persistence_paths) != H4_HOLDOUT_VARIANCE_PERSISTENCE_KEYS:
            raise H4HoldoutVarianceBindingError(
                "persistence paths must contain the exact frozen infrastructure keys"
            )
        if any(not value for _, value in self.persistence_paths):
            raise H4HoldoutVarianceBindingError("persistence paths must not be empty")

    def payload(self) -> dict[str, Any]:
        return {
            "binding_version": self.binding_version,
            "persistence_layout_version": self.persistence_layout_version,
            "execution_authority": self.execution_authority.payload(),
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "planning_spec_id": self.planning_spec_id,
            "population_id": self.population_id,
            "population_version": self.population_version,
            "population_role": self.population_role,
            "population_hash": self.population_hash,
            "holdout_acceptance_hash": self.holdout_acceptance_hash,
            "unit_acceptance_hashes": [
                {"unit_id": unit_id, "acceptance_hash": digest}
                for unit_id, digest in self.unit_acceptance_hashes
            ],
            "game_config_hash": self.game_config_hash,
            "master_seeds": list(self.master_seeds),
            "holdout_unit_ids": list(self.holdout_unit_ids),
            "rng_configuration": dict(self.rng_configuration),
            "adaptive_dependencies": {
                "adaptive_learning_protocol_hash": self.adaptive_learning_protocol_hash,
                "posterior_to_range_protocol_hash": self.posterior_to_range_protocol_hash,
                "adaptive_policy_protocol_hash": self.adaptive_policy_protocol_hash,
                "adaptive_lifecycle_protocol_hash": self.adaptive_lifecycle_protocol_hash,
                "opponent_model_version": self.opponent_model_version,
                "opportunity_definition_version": self.opportunity_definition_version,
                "public_action_trace_version": self.public_action_trace_version,
            },
            "opponent_policy": {
                "parameter_schema_version": self.parameter_schema_version,
                "parametric_policy_version": self.parametric_policy_version,
            },
            "replay": {
                "match_record_schema_version": self.match_record_schema_version,
                "execution_enforcement_version": self.execution_enforcement_version,
            },
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
            "process_model": self.process_model,
            "persistence_paths": dict(self.persistence_paths),
        }

    @property
    def binding_hash(self) -> str:
        return _sha256_canonical(self.payload())

    def reference_payload(self) -> dict[str, Any]:
        return {**self.payload(), "binding_hash": self.binding_hash}

    def to_dict(self) -> dict[str, Any]:
        return {"binding": self.payload(), "binding_hash": self.binding_hash}


def build_h4_holdout_variance_execution_binding(
    *,
    protocol: H4HoldoutVarianceProtocol,
    population: H4HoldoutPopulation,
    acceptance: H4HoldoutAcceptance,
    environment: ExecutionEnvironment,
    workers: int,
    process_model: str,
    persistence_paths: Mapping[str, str],
    execution_authority: H4HoldoutVarianceExecutionAuthorityReference,
) -> H4HoldoutVarianceExecutionBinding:
    """Bind Spec-37 science, holdout seal, runtime, persistence, and execution authority."""
    if not isinstance(protocol, H4HoldoutVarianceProtocol):
        raise TypeError("protocol must be H4HoldoutVarianceProtocol")
    if not isinstance(population, H4HoldoutPopulation):
        raise TypeError("population must be H4HoldoutPopulation")
    if not isinstance(acceptance, H4HoldoutAcceptance):
        raise TypeError("acceptance must be H4HoldoutAcceptance")
    if not isinstance(environment, ExecutionEnvironment):
        raise TypeError("environment must be ExecutionEnvironment")
    if not isinstance(execution_authority, H4HoldoutVarianceExecutionAuthorityReference):
        raise TypeError("execution_authority must be H4HoldoutVarianceExecutionAuthorityReference")
    _positive_int(workers, "workers")
    if not process_model:
        raise H4HoldoutVarianceBindingError("process_model must not be empty")
    require_h4_holdout_acceptance_closed(acceptance)

    frozen = expected_h4_holdout_variance_payload()
    bindings = frozen["bindings"]
    holdout = bindings["holdout_population"]
    game = bindings["game"]
    opponent = bindings["opponent"]
    dependencies = bindings["adaptive_dependencies"]
    if protocol.payload_hash != H4_HOLDOUT_VARIANCE_PROTOCOL_HASH or protocol.protocol_id != frozen["protocol_id"]:
        raise H4HoldoutVarianceBindingError("protocol differs from frozen Spec 37")
    if population.population_hash != H4_HOLDOUT_POPULATION_HASH or population.population_hash != holdout["population_hash"]:
        raise H4HoldoutVarianceBindingError("holdout population hash does not match Spec 37")
    if acceptance.population_hash != population.population_hash:
        raise H4HoldoutVarianceBindingError("holdout acceptance population hash does not match")
    if acceptance.acceptance_hash != H4_HOLDOUT_ACCEPTANCE_HASH or acceptance.acceptance_hash != holdout["holdout_acceptance_hash"]:
        raise H4HoldoutVarianceBindingError("holdout acceptance hash does not match Spec 37")
    if environment.engine_version != f"git:{environment.git_commit}":
        raise H4HoldoutVarianceBindingError("engine_version must be derived from git_commit")
    if execution_authority.authorized_protocol_id != protocol.protocol_id:
        raise H4HoldoutVarianceBindingError("execution authority authorized protocol id does not match")
    if execution_authority.authorized_protocol_version != protocol.protocol_version:
        raise H4HoldoutVarianceBindingError("execution authority authorized protocol version does not match")
    if execution_authority.authorized_protocol_hash != protocol.payload_hash:
        raise H4HoldoutVarianceBindingError("execution authority authorized protocol hash does not match")

    normalized_paths = tuple(sorted((str(key), str(value)) for key, value in persistence_paths.items()))
    if tuple(key for key, _ in normalized_paths) != H4_HOLDOUT_VARIANCE_PERSISTENCE_KEYS:
        raise H4HoldoutVarianceBindingError(
            "persistence paths must contain exactly work/database/config/snapshot/pair/cluster paths"
        )

    return H4HoldoutVarianceExecutionBinding(
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_hash=protocol.payload_hash,
        planning_spec_id=str(bindings["planning_spec"]),
        population_id=population.population_id,
        population_version=population.population_version,
        population_role=population.population_role,
        population_hash=population.population_hash,
        holdout_acceptance_hash=acceptance.acceptance_hash,
        unit_acceptance_hashes=tuple(acceptance.unit_acceptance_hashes.items()),
        game_config_hash=str(game["game_config_hash"]),
        master_seeds=tuple(protocol.master_seeds),
        holdout_unit_ids=tuple(protocol.holdout_unit_ids),
        rng_configuration=_expected_rng_configuration(),
        adaptive_learning_protocol_hash=_frozen_dependency_hash("adaptive_learning_protocol"),
        posterior_to_range_protocol_hash=_frozen_dependency_hash("posterior_to_range"),
        adaptive_policy_protocol_hash=_frozen_dependency_hash("adaptive_policy"),
        adaptive_lifecycle_protocol_hash=_frozen_dependency_hash("adaptive_lifecycle"),
        opponent_model_version=str(dependencies["opponent_model"]),
        opportunity_definition_version=str(dependencies["opportunity_definition"]),
        public_action_trace_version=str(dependencies["public_action_trace"]),
        parameter_schema_version=str(opponent["parameter_schema_version"]),
        parametric_policy_version=str(opponent["parametric_policy_version"]),
        match_record_schema_version=MATCH_RECORD_SCHEMA_VERSION,
        execution_enforcement_version=H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION,
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
        hardware_fingerprint=_hardware_fingerprint(environment),
        workers=workers,
        process_model=process_model,
        persistence_paths=normalized_paths,
        execution_authority=execution_authority,
    )


def capture_h4_holdout_variance_execution_binding(
    *,
    protocol: H4HoldoutVarianceProtocol,
    population: H4HoldoutPopulation,
    acceptance: H4HoldoutAcceptance,
    repo_root: str | Path,
    poetry_lock_path: str | Path,
    ram_bytes: int,
    workers: int,
    process_model: str,
    persistence_paths: Mapping[str, str],
    execution_authority: H4HoldoutVarianceExecutionAuthorityReference,
) -> H4HoldoutVarianceExecutionBinding:
    """Capture clean Git/software/hardware provenance for a future authorized run."""
    _positive_int(ram_bytes, "ram_bytes")
    commit = _clean_scientific_git_commit(Path(repo_root))
    environment = capture_execution_environment(
        engine_version=f"git:{commit}",
        git_commit=commit,
        poetry_lock_path=Path(poetry_lock_path),
        ram_bytes=ram_bytes,
    )
    return build_h4_holdout_variance_execution_binding(
        protocol=protocol,
        population=population,
        acceptance=acceptance,
        environment=environment,
        workers=workers,
        process_model=process_model,
        persistence_paths=persistence_paths,
        execution_authority=execution_authority,
    )


def _authority_from_payload(raw: Mapping[str, Any]) -> H4HoldoutVarianceExecutionAuthorityReference:
    return H4HoldoutVarianceExecutionAuthorityReference(
        authority_id=str(raw["authority_id"]),
        authority_version=str(raw["authority_version"]),
        payload_hash=_validate_sha256(str(raw["payload_hash"]), "execution_authority_hash"),
        authorized_protocol_id=str(raw["authorized_protocol_id"]),
        authorized_protocol_version=str(raw["authorized_protocol_version"]),
        authorized_protocol_hash=_validate_sha256(
            str(raw["authorized_protocol_hash"]), "authorized_protocol_hash"
        ),
    )


def _binding_from_payload(payload: Mapping[str, Any]) -> H4HoldoutVarianceExecutionBinding:
    authority_raw = payload.get("execution_authority")
    dependencies = payload.get("adaptive_dependencies")
    opponent = payload.get("opponent_policy")
    replay = payload.get("replay")
    rng = payload.get("rng_configuration")
    paths = payload.get("persistence_paths")
    unit_hashes = payload.get("unit_acceptance_hashes")
    if not all(isinstance(item, Mapping) for item in (authority_raw, dependencies, opponent, replay, rng, paths)):
        raise H4HoldoutVarianceBindingError("variance-pilot binding nested payload is malformed")
    if not isinstance(unit_hashes, list):
        raise H4HoldoutVarianceBindingError("unit_acceptance_hashes must be an array")
    parsed_unit_hashes: list[tuple[str, str]] = []
    for item in unit_hashes:
        if not isinstance(item, Mapping):
            raise H4HoldoutVarianceBindingError("unit acceptance entry must be an object")
        parsed_unit_hashes.append(
            (str(item["unit_id"]), _validate_sha256(str(item["acceptance_hash"]), "unit_acceptance_hash"))
        )
    return H4HoldoutVarianceExecutionBinding(
        binding_version=str(payload["binding_version"]),
        persistence_layout_version=str(payload["persistence_layout_version"]),
        execution_authority=_authority_from_payload(authority_raw),
        protocol_id=str(payload["protocol_id"]),
        protocol_version=str(payload["protocol_version"]),
        protocol_hash=_validate_sha256(str(payload["protocol_hash"]), "protocol_hash"),
        planning_spec_id=str(payload["planning_spec_id"]),
        population_id=str(payload["population_id"]),
        population_version=str(payload["population_version"]),
        population_role=str(payload["population_role"]),
        population_hash=_validate_sha256(str(payload["population_hash"]), "population_hash"),
        holdout_acceptance_hash=_validate_sha256(
            str(payload["holdout_acceptance_hash"]), "holdout_acceptance_hash"
        ),
        unit_acceptance_hashes=tuple(parsed_unit_hashes),
        game_config_hash=_validate_sha256(str(payload["game_config_hash"]), "game_config_hash"),
        master_seeds=tuple(int(seed) for seed in payload["master_seeds"]),
        holdout_unit_ids=tuple(str(unit) for unit in payload["holdout_unit_ids"]),
        rng_configuration=tuple(sorted((str(key), value) for key, value in rng.items())),
        adaptive_learning_protocol_hash=_validate_sha256(
            str(dependencies["adaptive_learning_protocol_hash"]), "adaptive_learning_protocol_hash"
        ),
        posterior_to_range_protocol_hash=_validate_sha256(
            str(dependencies["posterior_to_range_protocol_hash"]), "posterior_to_range_protocol_hash"
        ),
        adaptive_policy_protocol_hash=_validate_sha256(
            str(dependencies["adaptive_policy_protocol_hash"]), "adaptive_policy_protocol_hash"
        ),
        adaptive_lifecycle_protocol_hash=_validate_sha256(
            str(dependencies["adaptive_lifecycle_protocol_hash"]), "adaptive_lifecycle_protocol_hash"
        ),
        opponent_model_version=str(dependencies["opponent_model_version"]),
        opportunity_definition_version=str(dependencies["opportunity_definition_version"]),
        public_action_trace_version=str(dependencies["public_action_trace_version"]),
        parameter_schema_version=str(opponent["parameter_schema_version"]),
        parametric_policy_version=str(opponent["parametric_policy_version"]),
        match_record_schema_version=str(replay["match_record_schema_version"]),
        execution_enforcement_version=str(replay["execution_enforcement_version"]),
        git_commit=str(payload["git_commit"]),
        engine_version=str(payload["engine_version"]),
        python_version=str(payload["python_version"]),
        dependency_lockfile=str(payload["dependency_lockfile"]),
        poetry_lock_hash=_validate_sha256(str(payload["poetry_lock_hash"]), "poetry_lock_hash"),
        cpu_model=str(payload["cpu_model"]),
        cpu_core_count=int(payload["cpu_core_count"]),
        cpu_architecture=str(payload["cpu_architecture"]),
        ram_bytes=int(payload["ram_bytes"]),
        os_identifier=str(payload["os_identifier"]),
        hardware_fingerprint=_validate_sha256(
            str(payload["hardware_fingerprint"]), "hardware_fingerprint"
        ),
        workers=int(payload["workers"]),
        process_model=str(payload["process_model"]),
        persistence_paths=tuple(sorted((str(key), str(value)) for key, value in paths.items())),
    )


def save_h4_holdout_variance_execution_binding(
    binding: H4HoldoutVarianceExecutionBinding,
    path: str | Path,
) -> Path:
    if not isinstance(binding, H4HoldoutVarianceExecutionBinding):
        raise TypeError("binding must be H4HoldoutVarianceExecutionBinding")
    destination = Path(path)
    if destination.exists():
        existing = load_h4_holdout_variance_execution_binding(destination)
        if existing != binding:
            raise H4HoldoutVarianceBindingError("existing variance-pilot binding does not match")
        return destination
    return _atomic_write(destination, binding.to_dict())


def load_h4_holdout_variance_execution_binding(
    path: str | Path,
) -> H4HoldoutVarianceExecutionBinding:
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise H4HoldoutVarianceBindingError("invalid variance-pilot binding JSON") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("binding"), Mapping):
        raise H4HoldoutVarianceBindingError("variance-pilot binding file is malformed")
    _require_canonical_file(source, raw, "variance-pilot binding")
    payload = raw["binding"]
    stored_hash = _validate_sha256(str(raw.get("binding_hash")), "binding_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise H4HoldoutVarianceBindingError("binding_hash does not match binding payload")
    binding = _binding_from_payload(payload)
    if binding.binding_hash != stored_hash:
        raise H4HoldoutVarianceBindingError("loaded variance-pilot binding is not canonical")
    return binding


def _validate_binding_against_protocol(
    protocol: H4HoldoutVarianceProtocol,
    binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation,
) -> None:
    frozen = expected_h4_holdout_variance_payload()
    bindings = frozen["bindings"]
    if binding.protocol_id != protocol.protocol_id or binding.protocol_version != protocol.protocol_version:
        raise H4HoldoutVarianceBindingError("binding protocol identity does not match")
    if binding.protocol_hash != protocol.payload_hash:
        raise H4HoldoutVarianceBindingError("binding protocol hash does not match")
    if binding.population_hash != population.population_hash or binding.population_hash != H4_HOLDOUT_POPULATION_HASH:
        raise H4HoldoutVarianceBindingError("binding population hash does not match")
    if binding.holdout_acceptance_hash != H4_HOLDOUT_ACCEPTANCE_HASH:
        raise H4HoldoutVarianceBindingError("binding holdout acceptance hash does not match")
    if binding.game_config_hash != bindings["game"]["game_config_hash"]:
        raise H4HoldoutVarianceBindingError("binding game config hash does not match")
    if binding.master_seeds != H4_VARIANCE_MASTER_SEEDS or binding.holdout_unit_ids != H4_HOLDOUT_UNIT_IDS:
        raise H4HoldoutVarianceBindingError("binding matrix identity does not match")
    if binding.rng_configuration != _expected_rng_configuration():
        raise H4HoldoutVarianceBindingError("binding RNG configuration does not match")
    if binding.execution_authority.authorized_protocol_hash != protocol.payload_hash:
        raise H4HoldoutVarianceBindingError("binding execution authority protocol hash does not match")


def _unit_parameters(population: H4HoldoutPopulation, unit_id: str) -> tuple[tuple[str, Any], ...]:
    unit = next((item for item in population.units if item.unit_id == unit_id), None)
    if unit is None:
        raise H4HoldoutVarianceBindingError("trajectory holdout unit is not in frozen population")
    parameters = unit.parameters
    if parameters.schema_version != PARAMETER_SCHEMA_VERSION:
        raise H4HoldoutVarianceBindingError("holdout parameter schema is not frozen v1")
    return (
        ("schema_version", parameters.schema_version),
        ("vpip", parameters.vpip),
        ("pfr", parameters.pfr),
        ("three_bet_freq", parameters.three_bet_freq),
        ("cbet_freq", parameters.cbet_freq),
        ("aggression", parameters.aggression),
        ("fold_to_cbet", parameters.fold_to_cbet),
    )


@dataclass(frozen=True)
class H4HoldoutVarianceTrajectoryConfig:
    """Complete immutable controller-side provenance for one Spec-37 arm job."""

    binding: H4HoldoutVarianceExecutionBinding
    arm_id: str
    agent_version: str
    adaptive: bool
    holdout_unit_id: str
    opponent_parameters: tuple[tuple[str, Any], ...]
    unit_acceptance_hash: str
    master_seed: int
    hands_per_position: int
    position_run_count: int
    config_version: str = H4_HOLDOUT_VARIANCE_TRAJECTORY_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.config_version != H4_HOLDOUT_VARIANCE_TRAJECTORY_CONFIG_VERSION:
            raise H4HoldoutVarianceBindingError("unsupported variance-pilot trajectory config version")
        if self.arm_id not in {"control", "treatment"}:
            raise H4HoldoutVarianceBindingError("hero arm must be control or treatment")
        if not self.agent_version or self.holdout_unit_id not in H4_HOLDOUT_UNIT_IDS:
            raise H4HoldoutVarianceBindingError("trajectory identities must be frozen Spec-37 values")
        _validate_sha256(self.unit_acceptance_hash, "unit_acceptance_hash")
        if self.master_seed not in H4_VARIANCE_MASTER_SEEDS:
            raise H4HoldoutVarianceBindingError("trajectory master seed is outside frozen Spec 37")
        if self.hands_per_position != 500 or self.position_run_count != 2:
            raise H4HoldoutVarianceBindingError("trajectory must contain exactly two 500-hand position runs")

    @property
    def job_simulated_engine_hands(self) -> int:
        return self.hands_per_position * self.position_run_count

    @property
    def run_provenance(self) -> dict[str, Any]:
        return {
            "experiment_id": f"{self.binding.protocol_id}:{self.binding.binding_hash}",
            "protocol_id": self.binding.protocol_id,
            "protocol_version": self.binding.protocol_version,
            "protocol_hash": self.binding.protocol_hash,
            "execution_authority_id": self.binding.execution_authority.authority_id,
            "execution_authority_version": self.binding.execution_authority.authority_version,
            "execution_authority_hash": self.binding.execution_authority.payload_hash,
            "execution_binding_hash": self.binding.binding_hash,
            "game_config_hash": self.binding.game_config_hash,
            "holdout_population_id": self.binding.population_id,
            "holdout_population_version": self.binding.population_version,
            "holdout_population_hash": self.binding.population_hash,
            "holdout_acceptance_hash": self.binding.holdout_acceptance_hash,
            "holdout_unit_id": self.holdout_unit_id,
            "holdout_unit_acceptance_hash": self.unit_acceptance_hash,
            "true_opponent_parameters": dict(self.opponent_parameters),
            "hero_arm": self.arm_id,
            "bot_version": self.agent_version,
            "opponent_version": PARAMETRIC_POLICY_VERSION,
            "master_seed": self.master_seed,
            "planned_trajectory_horizon": self.hands_per_position,
            "position_run_count": self.position_run_count,
            "statistical_unit": "matched_population_trajectory_cluster",
            "execution_mode": "full-match",
            "rng_configuration": dict(self.binding.rng_configuration),
            "adaptive_learning_protocol_hash": self.binding.adaptive_learning_protocol_hash,
            "posterior_to_range_protocol_hash": self.binding.posterior_to_range_protocol_hash,
            "adaptive_policy_protocol_hash": self.binding.adaptive_policy_protocol_hash,
            "adaptive_lifecycle_protocol_hash": self.binding.adaptive_lifecycle_protocol_hash,
            "public_action_trace_version": PUBLIC_ACTION_TRACE_VERSION,
            "match_record_schema_version": self.binding.match_record_schema_version,
            "execution_enforcement_version": self.binding.execution_enforcement_version,
            "engine_version": self.binding.engine_version,
            "python_version": self.binding.python_version,
            "poetry_lock_hash": self.binding.poetry_lock_hash,
            "git_commit": self.binding.git_commit,
            "cpu_model": self.binding.cpu_model,
            "cpu_core_count": self.binding.cpu_core_count,
            "cpu_architecture": self.binding.cpu_architecture,
            "os_identifier": self.binding.os_identifier,
            "workers": self.binding.workers,
            "process_model": self.binding.process_model,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "execution_binding": self.binding.reference_payload(),
            "job": {
                "arm_id": self.arm_id,
                "agent_version": self.agent_version,
                "adaptive": self.adaptive,
                "holdout_unit_id": self.holdout_unit_id,
                "master_seed": self.master_seed,
                "hands_per_position": self.hands_per_position,
                "position_run_count": self.position_run_count,
                "simulated_engine_hands": self.job_simulated_engine_hands,
            },
            "controller_side_opponent": {
                "parameters": dict(self.opponent_parameters),
                "unit_acceptance_hash": self.unit_acceptance_hash,
            },
            "run_provenance": self.run_provenance,
            "scientific_use": {
                "variance_planning_only": True,
                "confirmatory_h4_inference_permitted": False,
                "variance_pilot_mean_design_use": "prohibited",
                "pilot_final_pooling": False,
                "variance_pilot_final_pooling": False,
            },
        }

    @property
    def config_hash(self) -> str:
        return _sha256_canonical(self.payload())

    @property
    def runner_metadata(self) -> dict[str, Any]:
        return {
            **self.run_provenance,
            "experiment_config_hash": self.config_hash,
            "trajectory_config_version": self.config_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.payload(), "config_hash": self.config_hash}


def _require_canonical_job(
    protocol: H4HoldoutVarianceProtocol,
    job: H4HoldoutVarianceJob,
) -> None:
    if not isinstance(job, H4HoldoutVarianceJob):
        raise TypeError("job must be H4HoldoutVarianceJob")
    if job not in iter_h4_holdout_variance_jobs(protocol):
        raise H4HoldoutVarianceBindingError(
            "job must be an exact canonical frozen Spec-37 job; replacement seeds are forbidden"
        )


def build_h4_holdout_variance_trajectory_config(
    *,
    protocol: H4HoldoutVarianceProtocol,
    binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation,
    job: H4HoldoutVarianceJob,
) -> H4HoldoutVarianceTrajectoryConfig:
    if not isinstance(binding, H4HoldoutVarianceExecutionBinding):
        raise TypeError("binding must be H4HoldoutVarianceExecutionBinding")
    if not isinstance(population, H4HoldoutPopulation):
        raise TypeError("population must be H4HoldoutPopulation")
    _validate_binding_against_protocol(protocol, binding, population)
    _require_canonical_job(protocol, job)
    unit_hashes = dict(binding.unit_acceptance_hashes)
    config = H4HoldoutVarianceTrajectoryConfig(
        binding=binding,
        arm_id=job.arm_id,
        agent_version=job.agent_version,
        adaptive=job.adaptive,
        holdout_unit_id=job.holdout_unit_id,
        opponent_parameters=_unit_parameters(population, job.holdout_unit_id),
        unit_acceptance_hash=unit_hashes[job.holdout_unit_id],
        master_seed=job.master_seed,
        hands_per_position=job.hands_per_position,
        position_run_count=job.position_run_count,
    )
    return config


def save_h4_holdout_variance_trajectory_config(
    config: H4HoldoutVarianceTrajectoryConfig,
    path: str | Path,
) -> Path:
    if not isinstance(config, H4HoldoutVarianceTrajectoryConfig):
        raise TypeError("config must be H4HoldoutVarianceTrajectoryConfig")
    destination = Path(path)
    if destination.exists():
        try:
            existing = json.loads(destination.read_bytes())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise H4HoldoutVarianceBindingError("existing trajectory config is invalid") from exc
        if existing != config.to_dict() or destination.read_bytes() != _canonical_file_bytes(existing):
            raise H4HoldoutVarianceBindingError("existing trajectory config does not match")
        return destination
    return _atomic_write(destination, config.to_dict())


def load_h4_holdout_variance_trajectory_config(
    path: str | Path,
    *,
    protocol: H4HoldoutVarianceProtocol,
    expected_binding: H4HoldoutVarianceExecutionBinding,
    population: H4HoldoutPopulation,
    expected_job: H4HoldoutVarianceJob,
) -> H4HoldoutVarianceTrajectoryConfig:
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise H4HoldoutVarianceBindingError("invalid variance-pilot trajectory config JSON") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("config"), Mapping):
        raise H4HoldoutVarianceBindingError("variance-pilot trajectory config is malformed")
    _require_canonical_file(source, raw, "variance-pilot trajectory config")
    payload = raw["config"]
    stored_hash = _validate_sha256(str(raw.get("config_hash")), "config_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise H4HoldoutVarianceBindingError("config_hash does not match trajectory config payload")
    expected = build_h4_holdout_variance_trajectory_config(
        protocol=protocol,
        binding=expected_binding,
        population=population,
        job=expected_job,
    )
    if payload != expected.payload():
        raise H4HoldoutVarianceBindingError("trajectory config has semantic drift")
    if expected.config_hash != stored_hash:
        raise H4HoldoutVarianceBindingError("loaded trajectory config is not canonical")
    return expected
