"""Hash-bound execution and trajectory provenance for the frozen Phase-2 adaptive pilot."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokerbench.agents.profiles import (
    PARAMETER_SCHEMA_VERSION,
    STATIC_PROFILE_VERSION,
    ProfileName,
    get_static_profile,
)
from pokerbench.agents.strength_gates import PARAMETRIC_POLICY_VERSION
from pokerbench.evaluation.gate6_execution import (
    Gate6CalibrationReport,
    load_gate6_calibration_report,
)
from pokerbench.evaluation.population import DiscretePopulationDefinition
from pokerbench.experiments.adaptive_pilot_execution_contract import (
    ADAPTIVE_PILOT_EXECUTION_CONTRACT_HASH,
    ADAPTIVE_PILOT_EXECUTION_CONTRACT_ID,
    ADAPTIVE_PILOT_EXECUTION_CONTRACT_VERSION,
    AdaptivePilotExecutionContract,
    frozen_adaptive_pilot_execution_contract,
)
from pokerbench.experiments.adaptive_pilot_parallel_execution_contract import (
    AdaptivePilotParallelExecutionContract,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    AdaptivePilotJob,
    AdaptivePilotProtocol,
    expected_adaptive_pilot_payload,
    iter_adaptive_pilot_jobs,
)
from pokerbench.experiments.config import (
    ExecutionEnvironment,
    capture_execution_environment,
)
from pokerbench.experiments.frozen_protocol import load_frozen_population


ADAPTIVE_PILOT_EXECUTION_BINDING_VERSION = "phase2-adaptive-pilot-execution-binding-v1"
ADAPTIVE_PILOT_TRAJECTORY_CONFIG_VERSION = "phase2-adaptive-pilot-trajectory-config-v1"


class AdaptivePilotBindingError(ValueError):
    """Raised when Spec-32 execution/provenance bindings are inconsistent."""


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
        raise AdaptivePilotBindingError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AdaptivePilotBindingError(
            f"{field_name} must be a SHA-256 hex digest"
        ) from exc
    return value.lower()


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdaptivePilotBindingError(f"{field_name} must be a positive integer")
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
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def _expected_population_versions() -> tuple[tuple[str, str, str], ...]:
    populations = expected_adaptive_pilot_payload()["populations"]
    primary = populations["primary_descriptive"]
    secondary = populations["secondary_descriptive"]
    return (
        (
            str(primary["population_id"]),
            str(primary["version"]),
            str(primary["population_hash"]),
        ),
        *tuple(
            (
                str(item["population_id"]),
                str(item["version"]),
                str(item["population_hash"]),
            )
            for item in secondary
        ),
    )


def _expected_rng_configuration(protocol: AdaptivePilotProtocol) -> tuple[tuple[str, Any], ...]:
    rng = dict(expected_adaptive_pilot_payload()["rng"])
    rng["seed_schedule_version"] = protocol.seed_schedule_version
    return tuple(sorted(rng.items()))


def _dependency_hash(name: str) -> str:
    dependency = expected_adaptive_pilot_payload()["dependencies"][name]
    if not isinstance(dependency, Mapping):
        raise AdaptivePilotBindingError(f"frozen dependency {name} is not hash-bound")
    return _validate_sha256(str(dependency["payload_hash"]), f"{name}_hash")


@dataclass(frozen=True)
class AdaptivePilotExecutionBinding:
    """Run-wide immutable execution facts captured before the first pilot hand."""

    pilot_protocol_id: str
    pilot_protocol_version: str
    pilot_protocol_hash: str
    game_config_hash: str
    gate6_report_hash: str
    gate6_reference_strength_hash: str
    population_versions: tuple[tuple[str, str, str], ...]
    rng_configuration: tuple[tuple[str, Any], ...]
    adaptive_learning_protocol_hash: str
    posterior_to_range_protocol_hash: str
    adaptive_policy_protocol_hash: str
    adaptive_lifecycle_protocol_hash: str
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
    execution_contract_id: str = ADAPTIVE_PILOT_EXECUTION_CONTRACT_ID
    execution_contract_version: str = ADAPTIVE_PILOT_EXECUTION_CONTRACT_VERSION
    execution_contract_hash: str = ADAPTIVE_PILOT_EXECUTION_CONTRACT_HASH
    binding_version: str = ADAPTIVE_PILOT_EXECUTION_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.binding_version != ADAPTIVE_PILOT_EXECUTION_BINDING_VERSION:
            raise AdaptivePilotBindingError("unsupported adaptive-pilot binding version")
        if not self.execution_contract_id or not self.execution_contract_version:
            raise AdaptivePilotBindingError("execution contract identity must be non-empty")
        for field_name in (
            "pilot_protocol_hash",
            "game_config_hash",
            "gate6_report_hash",
            "gate6_reference_strength_hash",
            "adaptive_learning_protocol_hash",
            "posterior_to_range_protocol_hash",
            "adaptive_policy_protocol_hash",
            "adaptive_lifecycle_protocol_hash",
            "poetry_lock_hash",
            "hardware_fingerprint",
            "execution_contract_hash",
        ):
            _validate_sha256(str(getattr(self, field_name)), field_name)
        if not self.pilot_protocol_id or not self.pilot_protocol_version:
            raise AdaptivePilotBindingError("pilot protocol identity must be non-empty")
        if not self.git_commit:
            raise AdaptivePilotBindingError("git_commit must not be empty")
        if self.engine_version != f"git:{self.git_commit}":
            raise AdaptivePilotBindingError("engine_version must be derived from git_commit")
        if not self.python_version or not self.dependency_lockfile:
            raise AdaptivePilotBindingError("software provenance must be non-empty")
        if not self.cpu_model or not self.cpu_architecture or not self.os_identifier:
            raise AdaptivePilotBindingError("hardware/OS provenance must be non-empty")
        _positive_int(self.cpu_core_count, "cpu_core_count")
        _positive_int(self.ram_bytes, "ram_bytes")
        _positive_int(self.workers, "workers")
        if not self.population_versions:
            raise AdaptivePilotBindingError("population_versions must not be empty")
        for population_id, version, population_hash in self.population_versions:
            if not population_id or not version:
                raise AdaptivePilotBindingError("population identity must be non-empty")
            _validate_sha256(population_hash, "population_hash")

    def payload(self) -> dict[str, Any]:
        return {
            "binding_version": self.binding_version,
            "execution_contract_id": self.execution_contract_id,
            "execution_contract_version": self.execution_contract_version,
            "execution_contract_hash": self.execution_contract_hash,
            "pilot_protocol_id": self.pilot_protocol_id,
            "pilot_protocol_version": self.pilot_protocol_version,
            "pilot_protocol_hash": self.pilot_protocol_hash,
            "game_config_hash": self.game_config_hash,
            "gate6_report_hash": self.gate6_report_hash,
            "gate6_reference_strength_hash": self.gate6_reference_strength_hash,
            "population_versions": [
                {
                    "population_id": population_id,
                    "version": version,
                    "population_hash": population_hash,
                }
                for population_id, version, population_hash in self.population_versions
            ],
            "rng_configuration": dict(self.rng_configuration),
            "adaptive_learning_protocol_hash": self.adaptive_learning_protocol_hash,
            "posterior_to_range_protocol_hash": self.posterior_to_range_protocol_hash,
            "adaptive_policy_protocol_hash": self.adaptive_policy_protocol_hash,
            "adaptive_lifecycle_protocol_hash": self.adaptive_lifecycle_protocol_hash,
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
        return _sha256_canonical(self.payload())

    def reference_payload(self) -> dict[str, Any]:
        return {**self.payload(), "binding_hash": self.binding_hash}

    def to_dict(self) -> dict[str, Any]:
        return {"binding": self.payload(), "binding_hash": self.binding_hash}


def _validate_population(
    actual: DiscretePopulationDefinition,
    expected: tuple[str, str, str],
    field_name: str,
) -> None:
    if not isinstance(actual, DiscretePopulationDefinition):
        raise TypeError(f"{field_name} must be a DiscretePopulationDefinition")
    identity = (actual.population_id, actual.version, actual.population_hash)
    if identity != expected:
        raise AdaptivePilotBindingError(f"{field_name} does not match frozen Spec-32 identity")


def build_adaptive_pilot_execution_binding(
    *,
    protocol: AdaptivePilotProtocol,
    execution_contract: AdaptivePilotExecutionContract | AdaptivePilotParallelExecutionContract | None = None,
    gate6_report: Gate6CalibrationReport,
    primary_population: DiscretePopulationDefinition,
    secondary_populations: Sequence[DiscretePopulationDefinition],
    environment: ExecutionEnvironment,
    workers: int,
) -> AdaptivePilotExecutionBinding:
    """Bind validated frozen scientific dependencies to one execution environment."""
    if not isinstance(protocol, AdaptivePilotProtocol):
        raise TypeError("protocol must be an AdaptivePilotProtocol")
    contract = execution_contract or frozen_adaptive_pilot_execution_contract()
    if not isinstance(
        contract,
        (AdaptivePilotExecutionContract, AdaptivePilotParallelExecutionContract),
    ):
        raise TypeError("execution_contract must be a frozen adaptive-pilot execution contract")
    if contract.authorized_protocol_id != protocol.protocol_id:
        raise AdaptivePilotBindingError("execution contract protocol id does not match")
    if contract.authorized_protocol_version != protocol.protocol_version:
        raise AdaptivePilotBindingError("execution contract protocol version does not match")
    if contract.authorized_protocol_hash != protocol.payload_hash:
        raise AdaptivePilotBindingError("execution contract authorized protocol hash does not match")
    # Generic construction helpers retain their pre-Spec-33 ability to model an
    # arbitrary worker count. Real pilot capture/launch always supplies the loaded
    # execution contract explicitly, which freezes workers=1 and is checked here.
    if execution_contract is not None and contract.workers != workers:
        raise AdaptivePilotBindingError("execution contract worker count does not match")
    if not isinstance(gate6_report, Gate6CalibrationReport):
        raise TypeError("gate6_report must be a Gate6CalibrationReport")
    if not isinstance(environment, ExecutionEnvironment):
        raise TypeError("environment must be an ExecutionEnvironment")
    _positive_int(workers, "workers")

    frozen = expected_adaptive_pilot_payload()
    if not gate6_report.passed:
        raise AdaptivePilotBindingError("Gate-6 report must be scientifically passing")
    expected_game_hash = str(frozen["game"]["game_config_hash"])
    if gate6_report.game_config_hash != expected_game_hash:
        raise AdaptivePilotBindingError("Gate-6 report game configuration differs from Spec 32")
    if environment.engine_version != f"git:{environment.git_commit}":
        raise AdaptivePilotBindingError("engine_version must be derived from git_commit")

    expected_populations = _expected_population_versions()
    _validate_population(primary_population, expected_populations[0], "primary population")
    secondary = tuple(secondary_populations)
    if len(secondary) != len(expected_populations) - 1:
        raise AdaptivePilotBindingError("secondary populations do not match frozen Spec 32")
    for index, (actual, expected) in enumerate(zip(secondary, expected_populations[1:], strict=True)):
        _validate_population(actual, expected, f"secondary population {index}")

    return AdaptivePilotExecutionBinding(
        pilot_protocol_id=protocol.protocol_id,
        pilot_protocol_version=protocol.protocol_version,
        pilot_protocol_hash=protocol.payload_hash,
        game_config_hash=expected_game_hash,
        gate6_report_hash=gate6_report.report_hash,
        gate6_reference_strength_hash=gate6_report.reference_strength_hash,
        population_versions=expected_populations,
        rng_configuration=_expected_rng_configuration(protocol),
        adaptive_learning_protocol_hash=_dependency_hash("adaptive_learning_protocol"),
        posterior_to_range_protocol_hash=_dependency_hash("posterior_to_range"),
        adaptive_policy_protocol_hash=_dependency_hash("adaptive_policy"),
        adaptive_lifecycle_protocol_hash=_dependency_hash("adaptive_lifecycle"),
        execution_contract_id=contract.execution_contract_id,
        execution_contract_version=contract.execution_contract_version,
        execution_contract_hash=contract.payload_hash,
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
            raise AdaptivePilotBindingError(
                "scientific source/configuration paths must be clean and committed before adaptive-pilot binding"
            )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise AdaptivePilotBindingError("could not resolve clean scientific Git state") from exc
    if not commit:
        raise AdaptivePilotBindingError("could not resolve Git commit")
    return commit


def capture_adaptive_pilot_execution_binding(
    *,
    protocol: AdaptivePilotProtocol,
    execution_contract: AdaptivePilotExecutionContract | AdaptivePilotParallelExecutionContract | None = None,
    gate6_report_path: str | Path,
    primary_population_path: str | Path,
    secondary_population_paths: Sequence[str | Path],
    repo_root: str | Path,
    poetry_lock_path: str | Path,
    ram_bytes: int,
    workers: int,
) -> AdaptivePilotExecutionBinding:
    """Capture the actual clean Git/software/hardware binding used for pilot execution."""
    _positive_int(ram_bytes, "ram_bytes")
    _positive_int(workers, "workers")
    commit = _clean_scientific_git_commit(Path(repo_root))
    environment = capture_execution_environment(
        engine_version=f"git:{commit}",
        git_commit=commit,
        poetry_lock_path=Path(poetry_lock_path),
        ram_bytes=ram_bytes,
    )
    return build_adaptive_pilot_execution_binding(
        protocol=protocol,
        execution_contract=execution_contract,
        gate6_report=load_gate6_calibration_report(Path(gate6_report_path)),
        primary_population=load_frozen_population(Path(primary_population_path)),
        secondary_populations=tuple(
            load_frozen_population(Path(path)) for path in secondary_population_paths
        ),
        environment=environment,
        workers=workers,
    )


def _binding_from_payload(payload: Mapping[str, Any]) -> AdaptivePilotExecutionBinding:
    populations_raw = payload.get("population_versions")
    rng_raw = payload.get("rng_configuration")
    if not isinstance(populations_raw, list) or not isinstance(rng_raw, Mapping):
        raise AdaptivePilotBindingError("binding populations/RNG payload is malformed")
    populations: list[tuple[str, str, str]] = []
    for item in populations_raw:
        if not isinstance(item, Mapping):
            raise AdaptivePilotBindingError("population_versions entries must be objects")
        populations.append(
            (
                str(item["population_id"]),
                str(item["version"]),
                _validate_sha256(str(item["population_hash"]), "population_hash"),
            )
        )
    return AdaptivePilotExecutionBinding(
        binding_version=str(payload["binding_version"]),
        execution_contract_id=str(payload["execution_contract_id"]),
        execution_contract_version=str(payload["execution_contract_version"]),
        execution_contract_hash=_validate_sha256(
            str(payload["execution_contract_hash"]), "execution_contract_hash"
        ),
        pilot_protocol_id=str(payload["pilot_protocol_id"]),
        pilot_protocol_version=str(payload["pilot_protocol_version"]),
        pilot_protocol_hash=_validate_sha256(str(payload["pilot_protocol_hash"]), "pilot_protocol_hash"),
        game_config_hash=_validate_sha256(str(payload["game_config_hash"]), "game_config_hash"),
        gate6_report_hash=_validate_sha256(str(payload["gate6_report_hash"]), "gate6_report_hash"),
        gate6_reference_strength_hash=_validate_sha256(
            str(payload["gate6_reference_strength_hash"]), "gate6_reference_strength_hash"
        ),
        population_versions=tuple(populations),
        rng_configuration=tuple(sorted((str(key), value) for key, value in rng_raw.items())),
        adaptive_learning_protocol_hash=_validate_sha256(
            str(payload["adaptive_learning_protocol_hash"]), "adaptive_learning_protocol_hash"
        ),
        posterior_to_range_protocol_hash=_validate_sha256(
            str(payload["posterior_to_range_protocol_hash"]), "posterior_to_range_protocol_hash"
        ),
        adaptive_policy_protocol_hash=_validate_sha256(
            str(payload["adaptive_policy_protocol_hash"]), "adaptive_policy_protocol_hash"
        ),
        adaptive_lifecycle_protocol_hash=_validate_sha256(
            str(payload["adaptive_lifecycle_protocol_hash"]), "adaptive_lifecycle_protocol_hash"
        ),
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
    )


def save_adaptive_pilot_execution_binding(
    binding: AdaptivePilotExecutionBinding,
    path: str | Path,
) -> Path:
    if not isinstance(binding, AdaptivePilotExecutionBinding):
        raise TypeError("binding must be an AdaptivePilotExecutionBinding")
    destination = Path(path)
    if destination.exists():
        existing = load_adaptive_pilot_execution_binding(destination)
        if existing != binding:
            raise AdaptivePilotBindingError("existing adaptive-pilot binding does not match")
        return destination
    return _atomic_write(destination, binding.to_dict())


def load_adaptive_pilot_execution_binding(path: str | Path) -> AdaptivePilotExecutionBinding:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("binding"), Mapping):
        raise AdaptivePilotBindingError("adaptive-pilot binding file is malformed")
    payload = raw["binding"]
    stored_hash = _validate_sha256(str(raw.get("binding_hash")), "binding_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise AdaptivePilotBindingError("binding_hash does not match binding payload")
    binding = _binding_from_payload(payload)
    if binding.binding_hash != stored_hash:
        raise AdaptivePilotBindingError("loaded binding is not canonical")
    return binding


def _validate_binding_against_protocol(
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
) -> None:
    frozen = expected_adaptive_pilot_payload()
    if binding.pilot_protocol_id != protocol.protocol_id:
        raise AdaptivePilotBindingError("binding pilot protocol id does not match")
    if binding.pilot_protocol_version != protocol.protocol_version:
        raise AdaptivePilotBindingError("binding pilot protocol version does not match")
    if binding.pilot_protocol_hash != protocol.payload_hash:
        raise AdaptivePilotBindingError("binding pilot protocol hash does not match")
    if binding.game_config_hash != str(frozen["game"]["game_config_hash"]):
        raise AdaptivePilotBindingError("binding game config hash does not match")
    if binding.population_versions != _expected_population_versions():
        raise AdaptivePilotBindingError("binding population versions do not match")
    if binding.rng_configuration != _expected_rng_configuration(protocol):
        raise AdaptivePilotBindingError("binding RNG configuration does not match")
    expected_dependencies = (
        ("adaptive_learning_protocol_hash", _dependency_hash("adaptive_learning_protocol")),
        ("posterior_to_range_protocol_hash", _dependency_hash("posterior_to_range")),
        ("adaptive_policy_protocol_hash", _dependency_hash("adaptive_policy")),
        ("adaptive_lifecycle_protocol_hash", _dependency_hash("adaptive_lifecycle")),
    )
    for field_name, expected in expected_dependencies:
        if getattr(binding, field_name) != expected:
            raise AdaptivePilotBindingError(f"binding {field_name} does not match")


def _profile_parameters(profile_id: str) -> tuple[tuple[str, Any], ...]:
    profile = get_static_profile(ProfileName(profile_id))
    parameters = profile.parameters
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
class AdaptivePilotTrajectoryConfig:
    """Complete immutable per-trajectory configuration and Spec-32 provenance."""

    binding: AdaptivePilotExecutionBinding
    arm_id: str
    agent_version: str
    adaptive: bool
    opponent_profile_id: str
    opponent_parameters: tuple[tuple[str, Any], ...]
    master_seed: int
    hands_per_position: int
    position_run_count: int
    statistical_unit: str
    execution_mode: str
    config_version: str = ADAPTIVE_PILOT_TRAJECTORY_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.config_version != ADAPTIVE_PILOT_TRAJECTORY_CONFIG_VERSION:
            raise AdaptivePilotBindingError("unsupported adaptive-pilot trajectory config version")
        if self.arm_id not in {"control", "treatment"}:
            raise AdaptivePilotBindingError("hero arm must be control or treatment")
        if not self.agent_version or not self.opponent_profile_id:
            raise AdaptivePilotBindingError("trajectory identities must be non-empty")
        _positive_int(self.master_seed, "master_seed")
        _positive_int(self.hands_per_position, "hands_per_position")
        if self.position_run_count != 2:
            raise AdaptivePilotBindingError("trajectory config must contain two position runs")
        if self.statistical_unit != "trajectory_cluster" or self.execution_mode != "full-match":
            raise AdaptivePilotBindingError("trajectory config must use trajectory_cluster/full-match")

    @property
    def job_simulated_engine_hands(self) -> int:
        return self.hands_per_position * self.position_run_count

    @property
    def run_provenance(self) -> dict[str, Any]:
        return {
            "experiment_id": f"{self.binding.pilot_protocol_id}:{self.binding.binding_hash}",
            "pilot_protocol_id": self.binding.pilot_protocol_id,
            "pilot_protocol_hash": self.binding.pilot_protocol_hash,
            "execution_contract_id": self.binding.execution_contract_id,
            "execution_contract_version": self.binding.execution_contract_version,
            "execution_contract_hash": self.binding.execution_contract_hash,
            "game_config_hash": self.binding.game_config_hash,
            "hero_arm": self.arm_id,
            "bot_version": self.agent_version,
            "opponent_version": f"{STATIC_PROFILE_VERSION}+{PARAMETRIC_POLICY_VERSION}",
            "opponent_profile_id": self.opponent_profile_id,
            "opponent_parameters": dict(self.opponent_parameters),
            "population_versions": [
                {
                    "population_id": population_id,
                    "version": version,
                    "population_hash": population_hash,
                }
                for population_id, version, population_hash in self.binding.population_versions
            ],
            "master_seed": self.master_seed,
            "planned_trajectory_horizon": self.hands_per_position,
            "rng_configuration": dict(self.binding.rng_configuration),
            "engine_version": self.binding.engine_version,
            "python_version": self.binding.python_version,
            "poetry_lock_hash": self.binding.poetry_lock_hash,
            "git_commit": self.binding.git_commit,
            "cpu_model": self.binding.cpu_model,
            "cpu_core_count": self.binding.cpu_core_count,
            "cpu_architecture": self.binding.cpu_architecture,
            "os_identifier": self.binding.os_identifier,
            "statistical_unit": self.statistical_unit,
            "gate6_report_hash": self.binding.gate6_report_hash,
            "adaptive_learning_protocol_hash": self.binding.adaptive_learning_protocol_hash,
            "posterior_to_range_protocol_hash": self.binding.posterior_to_range_protocol_hash,
            "adaptive_policy_protocol_hash": self.binding.adaptive_policy_protocol_hash,
            "adaptive_lifecycle_protocol_hash": self.binding.adaptive_lifecycle_protocol_hash,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "execution_binding": self.binding.reference_payload(),
            "job": {
                "arm_id": self.arm_id,
                "agent_version": self.agent_version,
                "adaptive": self.adaptive,
                "opponent_profile_id": self.opponent_profile_id,
                "master_seed": self.master_seed,
                "hands_per_position": self.hands_per_position,
                "position_run_count": self.position_run_count,
                "statistical_unit": self.statistical_unit,
                "execution_mode": self.execution_mode,
                "simulated_engine_hands": self.job_simulated_engine_hands,
            },
            "run_provenance": self.run_provenance,
            "scientific_use": {
                "final_experiment_data": False,
                "eligible_for_confirmatory_analysis": False,
                "pilot_final_pooling": False,
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
            "execution_binding_hash": self.binding.binding_hash,
            "trajectory_config_version": self.config_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.payload(), "config_hash": self.config_hash}


def _require_canonical_job(protocol: AdaptivePilotProtocol, job: AdaptivePilotJob) -> None:
    if not isinstance(job, AdaptivePilotJob):
        raise TypeError("job must be an AdaptivePilotJob")
    if job not in iter_adaptive_pilot_jobs(protocol):
        raise AdaptivePilotBindingError("job must be an exact canonical frozen pilot job")


def build_adaptive_pilot_trajectory_config(
    *,
    protocol: AdaptivePilotProtocol,
    binding: AdaptivePilotExecutionBinding,
    job: AdaptivePilotJob,
) -> AdaptivePilotTrajectoryConfig:
    """Build the exact hash-bound configuration persisted for one frozen pilot trajectory."""
    if not isinstance(protocol, AdaptivePilotProtocol):
        raise TypeError("protocol must be an AdaptivePilotProtocol")
    if not isinstance(binding, AdaptivePilotExecutionBinding):
        raise TypeError("binding must be an AdaptivePilotExecutionBinding")
    _validate_binding_against_protocol(protocol, binding)
    _require_canonical_job(protocol, job)

    parameters = _profile_parameters(job.opponent_profile_id)
    if dict(parameters).get("schema_version") != PARAMETER_SCHEMA_VERSION:
        raise AdaptivePilotBindingError("opponent parameter schema is not frozen v1")

    config = AdaptivePilotTrajectoryConfig(
        binding=binding,
        arm_id=job.arm_id,
        agent_version=job.agent_version,
        adaptive=job.adaptive,
        opponent_profile_id=job.opponent_profile_id,
        opponent_parameters=parameters,
        master_seed=job.master_seed,
        hands_per_position=job.hands_per_position,
        position_run_count=job.position_run_count,
        statistical_unit=job.statistical_unit,
        execution_mode=job.execution_mode,
    )
    missing = set(protocol.required_run_provenance) - set(config.run_provenance)
    if missing:
        raise AdaptivePilotBindingError(
            "trajectory config is missing required provenance: " + ", ".join(sorted(missing))
        )
    return config


def validate_adaptive_pilot_trajectory_config(
    *,
    protocol: AdaptivePilotProtocol,
    config: AdaptivePilotTrajectoryConfig,
    expected_job: AdaptivePilotJob,
) -> None:
    if not isinstance(config, AdaptivePilotTrajectoryConfig):
        raise TypeError("config must be an AdaptivePilotTrajectoryConfig")
    expected = build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=config.binding,
        job=expected_job,
    )
    if config != expected:
        raise AdaptivePilotBindingError("trajectory config does not match the frozen pilot job")


def save_adaptive_pilot_trajectory_config(
    config: AdaptivePilotTrajectoryConfig,
    path: str | Path,
) -> Path:
    if not isinstance(config, AdaptivePilotTrajectoryConfig):
        raise TypeError("config must be an AdaptivePilotTrajectoryConfig")
    destination = Path(path)
    if destination.exists():
        existing_raw = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(existing_raw, Mapping) or existing_raw != config.to_dict():
            raise AdaptivePilotBindingError("existing trajectory config does not match")
        return destination
    return _atomic_write(destination, config.to_dict())


def load_adaptive_pilot_trajectory_config(
    path: str | Path,
    *,
    protocol: AdaptivePilotProtocol,
    expected_binding: AdaptivePilotExecutionBinding,
    expected_job: AdaptivePilotJob,
) -> AdaptivePilotTrajectoryConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("config"), Mapping):
        raise AdaptivePilotBindingError("trajectory config file is malformed")
    payload = raw["config"]
    stored_hash = _validate_sha256(str(raw.get("config_hash")), "config_hash")
    if _sha256_canonical(payload) != stored_hash:
        raise AdaptivePilotBindingError("config_hash does not match trajectory config payload")

    binding_payload = payload.get("execution_binding")
    if not isinstance(binding_payload, Mapping):
        raise AdaptivePilotBindingError("trajectory execution binding is malformed")
    embedded_hash = _validate_sha256(str(binding_payload.get("binding_hash")), "binding_hash")
    binding_without_hash = {key: value for key, value in binding_payload.items() if key != "binding_hash"}
    if _sha256_canonical(binding_without_hash) != embedded_hash:
        raise AdaptivePilotBindingError("embedded binding_hash does not match")
    embedded_binding = _binding_from_payload(binding_without_hash)
    if embedded_binding != expected_binding:
        raise AdaptivePilotBindingError("trajectory config execution binding does not match")

    expected = build_adaptive_pilot_trajectory_config(
        protocol=protocol,
        binding=embedded_binding,
        job=expected_job,
    )
    if payload != expected.payload():
        raise AdaptivePilotBindingError("trajectory config has semantic drift")
    if expected.config_hash != stored_hash:
        raise AdaptivePilotBindingError("loaded trajectory config is not canonical")
    return expected
