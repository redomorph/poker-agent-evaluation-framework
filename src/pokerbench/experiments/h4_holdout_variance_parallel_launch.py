"""Non-executing plan/status/readiness gate for the frozen Spec-38 variance pilot."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokerbench.evaluation.calibration_snapshots import (
    CalibrationStrengthSnapshotStore,
    policy_strength_reference_from_store,
    policy_validation_strengths_from_store,
)
from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.evaluation.h4_holdout_acceptance import (
    H4HoldoutAcceptance,
    load_h4_holdout_acceptance,
)
from pokerbench.experiments.h4_holdout_population import (
    H4HoldoutPopulation,
    load_h4_holdout_population,
)
from pokerbench.experiments.h4_holdout_variance_binding import (
    H4_HOLDOUT_VARIANCE_EXECUTION_BINDING_VERSION,
    H4_HOLDOUT_VARIANCE_PERSISTENCE_LAYOUT_VERSION,
    H4_HOLDOUT_VARIANCE_TRAJECTORY_CONFIG_VERSION,
    H4HoldoutVarianceBindingError,
    H4HoldoutVarianceExecutionAuthorityReference,
    H4HoldoutVarianceExecutionBinding,
    capture_h4_holdout_variance_execution_binding,
    load_h4_holdout_variance_execution_binding,
)
from pokerbench.experiments.h4_holdout_variance_execution import (
    H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION,
    H4HoldoutVarianceExecutionError,
)
from pokerbench.experiments.h4_holdout_variance_execution_contract import (
    H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH,
    H4_HOLDOUT_VARIANCE_MAX_IN_FLIGHT_JOBS,
    H4_HOLDOUT_VARIANCE_PROCESS_MODEL,
    H4_HOLDOUT_VARIANCE_WORKERS,
    H4HoldoutVarianceExecutionContract,
    H4HoldoutVarianceExecutionContractError,
    load_h4_holdout_variance_execution_contract,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4_HOLDOUT_VARIANCE_PROTOCOL_HASH,
    H4HoldoutVarianceProtocol,
    load_h4_holdout_variance_protocol,
)
from pokerbench.experiments.h4_holdout_variance_run import (
    H4_HOLDOUT_VARIANCE_ARM_SNAPSHOT_VERSION,
    H4_HOLDOUT_VARIANCE_CLUSTER_CHECKPOINT_VERSION,
    H4_HOLDOUT_VARIANCE_PAIR_CHECKPOINT_VERSION,
    H4_HOLDOUT_VARIANCE_RUN_MANIFEST_VERSION,
    H4HoldoutVarianceRunError,
    H4HoldoutVarianceRunReadiness,
    H4HoldoutVarianceRunResult,
    assert_h4_holdout_variance_run_ready,
    h4_holdout_variance_job_key,
    initialize_h4_holdout_variance_run,
    load_h4_holdout_variance_run_manifest,
)
from pokerbench.experiments.h4_holdout_variance_parallel_run import (
    H4_HOLDOUT_VARIANCE_PARALLEL_RUN_VERSION,
    H4HoldoutVarianceParallelRunError,
    H4HoldoutVarianceParallelWorkerTask,
    execute_h4_holdout_variance_parallel_run,
)
from pokerbench.persistence.sqlite import SQLiteBenchmarkStore


H4_HOLDOUT_VARIANCE_PARALLEL_LAUNCH_VERSION = "phase2-h4-holdout-variance-parallel-launch-v1"
DEFAULT_H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_PATH = Path(
    "configs/experiments/phase2-h4-holdout-variance-execution-v1.json"
)


class H4HoldoutVarianceParallelLaunchError(ValueError):
    """Raised when the Spec-38 non-executing launch gate fails closed."""


@dataclass(frozen=True)
class H4HoldoutVarianceLaunchAssets:
    protocol: H4HoldoutVarianceProtocol
    contract: H4HoldoutVarianceExecutionContract
    population: H4HoldoutPopulation
    acceptance: H4HoldoutAcceptance


@dataclass(frozen=True)
class H4HoldoutVarianceLaunchStatus:
    binding_hash: str
    planned_jobs: int
    planned_pairs: int
    planned_clusters: int
    planned_engine_hands: int
    terminal_jobs: int
    planned: int
    running: int
    completed: int
    failed: int

    @property
    def virgin(self) -> bool:
        return self.terminal_jobs == 0 and self.running == 0 and self.planned == self.planned_jobs


def _require_parallel_host() -> None:
    count = os.cpu_count()
    if count is None or count < H4_HOLDOUT_VARIANCE_WORKERS:
        raise H4HoldoutVarianceParallelLaunchError(
            "Spec-38 variance pilot requires at least four logical CPU cores"
        )


def _load_contract(path: str | Path) -> H4HoldoutVarianceExecutionContract:
    try:
        contract = load_h4_holdout_variance_execution_contract(path)
    except (OSError, ValueError, H4HoldoutVarianceExecutionContractError) as exc:
        raise H4HoldoutVarianceParallelLaunchError(
            f"could not load frozen Spec-38 execution contract: {exc}"
        ) from exc
    if (
        contract.payload_hash != H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH
        or contract.workers != H4_HOLDOUT_VARIANCE_WORKERS
        or contract.max_in_flight_jobs != H4_HOLDOUT_VARIANCE_MAX_IN_FLIGHT_JOBS
        or contract.process_model != H4_HOLDOUT_VARIANCE_PROCESS_MODEL
    ):
        raise H4HoldoutVarianceParallelLaunchError("Spec-38 execution contract runtime fields drifted")
    return contract


def _validate_implemented_component_versions(contract: H4HoldoutVarianceExecutionContract) -> None:
    versions = dict(contract.component_versions)
    expected = {
        "execution_enforcement_version": H4_HOLDOUT_VARIANCE_EXECUTION_ENFORCEMENT_VERSION,
        "execution_binding_version": H4_HOLDOUT_VARIANCE_EXECUTION_BINDING_VERSION,
        "trajectory_config_version": H4_HOLDOUT_VARIANCE_TRAJECTORY_CONFIG_VERSION,
        "persistence_layout_version": H4_HOLDOUT_VARIANCE_PERSISTENCE_LAYOUT_VERSION,
        "run_manifest_version": H4_HOLDOUT_VARIANCE_RUN_MANIFEST_VERSION,
        "arm_snapshot_version": H4_HOLDOUT_VARIANCE_ARM_SNAPSHOT_VERSION,
        "pair_checkpoint_version": H4_HOLDOUT_VARIANCE_PAIR_CHECKPOINT_VERSION,
        "cluster_checkpoint_version": H4_HOLDOUT_VARIANCE_CLUSTER_CHECKPOINT_VERSION,
        "parallel_run_version": H4_HOLDOUT_VARIANCE_PARALLEL_RUN_VERSION,
        "parallel_launch_version": H4_HOLDOUT_VARIANCE_PARALLEL_LAUNCH_VERSION,
    }
    for key, value in expected.items():
        if versions.get(key) != value:
            raise H4HoldoutVarianceParallelLaunchError(
                f"Spec-38 execution component {key} differs from active implementation"
            )


def load_h4_holdout_variance_launch_assets(
    *,
    protocol_path: str | Path,
    execution_contract_path: str | Path,
    population_path: str | Path,
    acceptance_path: str | Path,
    snapshot_root: str | Path,
    gate6_report_path: str | Path,
    reference_path: str | Path,
) -> H4HoldoutVarianceLaunchAssets:
    try:
        protocol = load_h4_holdout_variance_protocol(protocol_path)
        contract = _load_contract(execution_contract_path)
        population = load_h4_holdout_population(population_path)
        report = load_gate6_calibration_report(gate6_report_path)
        snapshots = CalibrationStrengthSnapshotStore(Path(snapshot_root))
        acceptance = load_h4_holdout_acceptance(
            acceptance_path,
            population=population,
            reference=policy_strength_reference_from_store(snapshots),
            validation_strengths=policy_validation_strengths_from_store(snapshots),
            gate6_report_hash=report.report_hash,
        )
    except H4HoldoutVarianceParallelLaunchError:
        raise
    except (OSError, ValueError) as exc:
        raise H4HoldoutVarianceParallelLaunchError(f"could not verify frozen launch assets: {exc}") from exc
    if not Path(reference_path).is_file():
        raise H4HoldoutVarianceParallelLaunchError("frozen reference asset is missing")
    if protocol.payload_hash != H4_HOLDOUT_VARIANCE_PROTOCOL_HASH:
        raise H4HoldoutVarianceParallelLaunchError("Spec-37 protocol hash drifted")
    if (
        contract.authorized_protocol_id != protocol.protocol_id
        or contract.authorized_protocol_version != protocol.protocol_version
        or contract.authorized_protocol_hash != protocol.payload_hash
    ):
        raise H4HoldoutVarianceParallelLaunchError("Spec-38 contract does not authorize exact Spec-37 protocol")
    _validate_implemented_component_versions(contract)
    return H4HoldoutVarianceLaunchAssets(
        protocol=protocol,
        contract=contract,
        population=population,
        acceptance=acceptance,
    )


def _authority(contract: H4HoldoutVarianceExecutionContract) -> H4HoldoutVarianceExecutionAuthorityReference:
    return H4HoldoutVarianceExecutionAuthorityReference(
        authority_id=contract.execution_contract_id,
        authority_version=contract.execution_contract_version,
        payload_hash=contract.payload_hash,
        authorized_protocol_id=contract.authorized_protocol_id,
        authorized_protocol_version=contract.authorized_protocol_version,
        authorized_protocol_hash=contract.authorized_protocol_hash,
    )


def _repo_relative(path: str | Path, repo_root: str | Path, field_name: str) -> str:
    root = Path(repo_root).resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise H4HoldoutVarianceParallelLaunchError(
            f"{field_name} must remain inside the repository root"
        ) from exc


def _persistence_paths(
    *, database: str | Path, work_root: str | Path, repo_root: str | Path,
) -> dict[str, str]:
    work = _repo_relative(work_root, repo_root, "work_root")
    db = _repo_relative(database, repo_root, "database")
    return {
        "work_root": work,
        "database": db,
        "trajectory_configs": f"{work}/trajectory-configs",
        "arm_snapshots": f"{work}/arm-snapshots",
        "pair_checkpoints": f"{work}/pair-checkpoints",
        "cluster_checkpoints": f"{work}/cluster-checkpoints",
    }


def _require_frozen_persistence_paths(
    contract: H4HoldoutVarianceExecutionContract,
    actual: dict[str, str],
) -> None:
    expected = dict(contract.persistence_paths)
    if actual != expected:
        raise H4HoldoutVarianceParallelLaunchError(
            "campaign persistence paths differ from frozen Spec-38 roots"
        )


def _load_binding(
    *, work_root: str | Path, contract: H4HoldoutVarianceExecutionContract,
) -> H4HoldoutVarianceExecutionBinding:
    path = Path(work_root) / "binding.json"
    if not path.exists():
        raise H4HoldoutVarianceParallelLaunchError("variance pilot is not planned; binding.json is missing")
    try:
        binding = load_h4_holdout_variance_execution_binding(path)
    except (OSError, ValueError, H4HoldoutVarianceBindingError) as exc:
        raise H4HoldoutVarianceParallelLaunchError(f"could not load persisted Spec-38 binding: {exc}") from exc
    authority = binding.execution_authority
    if (
        authority.authority_id != contract.execution_contract_id
        or authority.authority_version != contract.execution_contract_version
        or authority.payload_hash != contract.payload_hash
        or authority.authorized_protocol_hash != contract.authorized_protocol_hash
        or binding.workers != contract.workers
        or binding.process_model != contract.process_model
        or dict(binding.persistence_paths) != dict(contract.persistence_paths)
    ):
        raise H4HoldoutVarianceParallelLaunchError("persisted binding differs from frozen Spec-38 authority")
    return binding


def _status(
    *,
    protocol: H4HoldoutVarianceProtocol,
    population: H4HoldoutPopulation,
    binding: H4HoldoutVarianceExecutionBinding,
    work_root: str | Path,
) -> H4HoldoutVarianceLaunchStatus:
    path = Path(work_root) / "run-manifest.json"
    if not path.exists():
        raise H4HoldoutVarianceParallelLaunchError("variance pilot is not planned; run-manifest.json is missing")
    manifest = load_h4_holdout_variance_run_manifest(
        path,
        protocol=protocol,
        expected_binding=binding,
        population=population,
    )
    counts = {
        name: sum(item.status == name for item in manifest.jobs)
        for name in ("planned", "running", "completed", "failed")
    }
    return H4HoldoutVarianceLaunchStatus(
        binding_hash=binding.binding_hash,
        planned_jobs=manifest.planned_job_count,
        planned_pairs=manifest.planned_pair_count,
        planned_clusters=manifest.planned_cluster_count,
        planned_engine_hands=manifest.planned_engine_hands,
        terminal_jobs=manifest.terminal_jobs,
        planned=counts["planned"],
        running=counts["running"],
        completed=counts["completed"],
        failed=counts["failed"],
    )


def status_h4_holdout_variance_parallel_launch(
    *,
    protocol_path: str | Path,
    population_path: str | Path,
    work_root: str | Path,
    execution_contract_path: str | Path = DEFAULT_H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_PATH,
) -> H4HoldoutVarianceLaunchStatus:
    try:
        protocol = load_h4_holdout_variance_protocol(protocol_path)
        population = load_h4_holdout_population(population_path)
        contract = _load_contract(execution_contract_path)
        binding = _load_binding(work_root=work_root, contract=contract)
        return _status(protocol=protocol, population=population, binding=binding, work_root=work_root)
    except H4HoldoutVarianceParallelLaunchError:
        raise
    except (OSError, ValueError) as exc:
        raise H4HoldoutVarianceParallelLaunchError(f"could not read variance-pilot status: {exc}") from exc


def _require_virgin_destination(database: Path, work_root: Path) -> None:
    if database.exists():
        raise H4HoldoutVarianceParallelLaunchError("variance-pilot database already exists; refusing re-plan")
    if work_root.exists() and any(work_root.iterdir()):
        raise H4HoldoutVarianceParallelLaunchError("variance-pilot work root is not empty; refusing re-plan")


def plan_h4_holdout_variance_parallel_launch(
    *,
    protocol_path: str | Path,
    execution_contract_path: str | Path,
    population_path: str | Path,
    acceptance_path: str | Path,
    snapshot_root: str | Path,
    gate6_report_path: str | Path,
    reference_path: str | Path,
    database: str | Path,
    work_root: str | Path,
    repo_root: str | Path,
    poetry_lock_path: str | Path,
    ram_bytes: int,
    workers: int = H4_HOLDOUT_VARIANCE_WORKERS,
) -> H4HoldoutVarianceLaunchStatus:
    if workers != H4_HOLDOUT_VARIANCE_WORKERS:
        raise H4HoldoutVarianceParallelLaunchError("Spec-38 variance pilot requires workers=4")
    _require_parallel_host()
    assets = load_h4_holdout_variance_launch_assets(
        protocol_path=protocol_path,
        execution_contract_path=execution_contract_path,
        population_path=population_path,
        acceptance_path=acceptance_path,
        snapshot_root=snapshot_root,
        gate6_report_path=gate6_report_path,
        reference_path=reference_path,
    )
    persistence = _persistence_paths(database=database, work_root=work_root, repo_root=repo_root)
    _require_frozen_persistence_paths(assets.contract, persistence)
    database_path = Path(database)
    work_root_path = Path(work_root)
    _require_virgin_destination(database_path, work_root_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        binding = capture_h4_holdout_variance_execution_binding(
            protocol=assets.protocol,
            population=assets.population,
            acceptance=assets.acceptance,
            repo_root=repo_root,
            poetry_lock_path=poetry_lock_path,
            ram_bytes=ram_bytes,
            workers=workers,
            process_model=assets.contract.process_model,
            persistence_paths=persistence,
            execution_authority=_authority(assets.contract),
        )
        store = SQLiteBenchmarkStore(database_path)
        store.initialize()
        initialize_h4_holdout_variance_run(
            protocol=assets.protocol,
            binding=binding,
            population=assets.population,
            work_root=work_root_path,
        )
        _require_zero_preflight_evidence(work_root=work_root_path, store=store)
    except H4HoldoutVarianceParallelLaunchError:
        raise
    except (OSError, ValueError, H4HoldoutVarianceBindingError, H4HoldoutVarianceRunError) as exc:
        raise H4HoldoutVarianceParallelLaunchError(f"could not plan Spec-38 variance pilot: {exc}") from exc
    status = _status(
        protocol=assets.protocol,
        population=assets.population,
        binding=binding,
        work_root=work_root_path,
    )
    if (
        status.planned_jobs != assets.contract.planned_jobs
        or status.planned_pairs != assets.contract.planned_pairs
        or status.planned_clusters != assets.contract.planned_clusters
        or status.planned_engine_hands != assets.contract.planned_engine_hands
        or not status.virgin
    ):
        raise H4HoldoutVarianceParallelLaunchError("planned campaign differs from frozen Spec-38 matrix")
    return status


def _tree_digest(work_root: Path, database: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    if work_root.exists():
        paths.extend(sorted(path for path in work_root.rglob("*") if path.is_file()))
    for candidate in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm")):
        if candidate.exists():
            paths.append(candidate)
    for path in sorted(paths, key=lambda item: item.as_posix()):
        try:
            label = path.relative_to(work_root.parent).as_posix()
        except ValueError:
            label = path.as_posix()
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_zero_preflight_evidence(
    *, work_root: Path, store: SQLiteBenchmarkStore,
) -> None:
    if store.list_match_ids():
        raise H4HoldoutVarianceParallelLaunchError("preflight requires zero MatchRecords")
    for directory, label in (
        ("arm-snapshots", "arm snapshots"),
        ("pair-checkpoints", "pair checkpoints"),
        ("cluster-checkpoints", "cluster checkpoints"),
    ):
        root = work_root / directory
        if root.exists() and any(root.iterdir()):
            raise H4HoldoutVarianceParallelLaunchError(f"preflight requires zero {label}")


def assert_h4_holdout_variance_parallel_launch_ready(
    *,
    protocol_path: str | Path,
    execution_contract_path: str | Path,
    population_path: str | Path,
    acceptance_path: str | Path,
    snapshot_root: str | Path,
    gate6_report_path: str | Path,
    reference_path: str | Path,
    database: str | Path,
    work_root: str | Path,
    repo_root: str | Path,
    poetry_lock_path: str | Path,
) -> H4HoldoutVarianceRunReadiness:
    """Validate the mandatory first-hand preflight without mutating campaign bytes."""
    _require_parallel_host()
    assets = load_h4_holdout_variance_launch_assets(
        protocol_path=protocol_path,
        execution_contract_path=execution_contract_path,
        population_path=population_path,
        acceptance_path=acceptance_path,
        snapshot_root=snapshot_root,
        gate6_report_path=gate6_report_path,
        reference_path=reference_path,
    )
    persistence = _persistence_paths(database=database, work_root=work_root, repo_root=repo_root)
    _require_frozen_persistence_paths(assets.contract, persistence)
    persisted = _load_binding(work_root=work_root, contract=assets.contract)
    try:
        current = capture_h4_holdout_variance_execution_binding(
            protocol=assets.protocol,
            population=assets.population,
            acceptance=assets.acceptance,
            repo_root=repo_root,
            poetry_lock_path=poetry_lock_path,
            ram_bytes=persisted.ram_bytes,
            workers=assets.contract.workers,
            process_model=assets.contract.process_model,
            persistence_paths=persistence,
            execution_authority=_authority(assets.contract),
        )
    except (OSError, ValueError, H4HoldoutVarianceBindingError) as exc:
        raise H4HoldoutVarianceParallelLaunchError(f"could not recapture current Spec-38 binding: {exc}") from exc
    if current != persisted:
        raise H4HoldoutVarianceParallelLaunchError("current Spec-38 execution binding differs from persisted binding")

    status = _status(
        protocol=assets.protocol,
        population=assets.population,
        binding=persisted,
        work_root=work_root,
    )
    if not status.virgin:
        raise H4HoldoutVarianceParallelLaunchError(
            "first-hand readiness requires a virgin plan; non-virgin reconciliation belongs to resume"
        )
    if (
        status.planned_jobs != assets.contract.planned_jobs
        or status.planned_pairs != assets.contract.planned_pairs
        or status.planned_clusters != assets.contract.planned_clusters
        or status.planned_engine_hands != assets.contract.planned_engine_hands
    ):
        raise H4HoldoutVarianceParallelLaunchError("readiness matrix differs from frozen Spec-38 contract")

    database_path = Path(database)
    work_root_path = Path(work_root)
    if not database_path.exists():
        raise H4HoldoutVarianceParallelLaunchError("planned variance-pilot database is missing")
    store = SQLiteBenchmarkStore(database_path)
    _require_zero_preflight_evidence(work_root=work_root_path, store=store)
    before = _tree_digest(work_root_path, database_path)
    try:
        readiness = assert_h4_holdout_variance_run_ready(
            protocol=assets.protocol,
            binding=persisted,
            population=assets.population,
            work_root=work_root_path,
            benchmark_store=store,
        )
    except (OSError, ValueError, H4HoldoutVarianceRunError) as exc:
        raise H4HoldoutVarianceParallelLaunchError(f"Spec-38 readiness failed: {exc}") from exc
    after = _tree_digest(work_root_path, database_path)
    if before != after:
        raise H4HoldoutVarianceParallelLaunchError("readiness mutated campaign bytes")
    if (
        readiness.binding_hash != persisted.binding_hash
        or readiness.planned_jobs != assets.contract.planned_jobs
        or readiness.planned_pairs != assets.contract.planned_pairs
        or readiness.planned_clusters != assets.contract.planned_clusters
        or readiness.planned_engine_hands != assets.contract.planned_engine_hands
        or readiness.terminal_jobs != 0
        or readiness.pair_count != 0
        or readiness.cluster_count != 0
    ):
        raise H4HoldoutVarianceParallelLaunchError("readiness evidence differs from virgin Spec-38 plan")
    return readiness


def _recapture_same_bound_execution(
    *,
    assets: H4HoldoutVarianceLaunchAssets,
    persisted: H4HoldoutVarianceExecutionBinding,
    persistence: dict[str, str],
    repo_root: str | Path,
    poetry_lock_path: str | Path,
) -> None:
    try:
        current = capture_h4_holdout_variance_execution_binding(
            protocol=assets.protocol,
            population=assets.population,
            acceptance=assets.acceptance,
            repo_root=repo_root,
            poetry_lock_path=poetry_lock_path,
            ram_bytes=persisted.ram_bytes,
            workers=assets.contract.workers,
            process_model=assets.contract.process_model,
            persistence_paths=persistence,
            execution_authority=_authority(assets.contract),
        )
    except (OSError, ValueError, H4HoldoutVarianceBindingError) as exc:
        raise H4HoldoutVarianceParallelLaunchError(
            f"could not recapture current Spec-38 binding: {exc}"
        ) from exc
    if current != persisted:
        raise H4HoldoutVarianceParallelLaunchError(
            "current Spec-38 execution binding differs from persisted binding"
        )


def run_h4_holdout_variance_parallel_launch(
    *,
    protocol_path: str | Path,
    execution_contract_path: str | Path,
    population_path: str | Path,
    acceptance_path: str | Path,
    snapshot_root: str | Path,
    gate6_report_path: str | Path,
    reference_path: str | Path,
    database: str | Path,
    work_root: str | Path,
    repo_root: str | Path,
    poetry_lock_path: str | Path,
    resume: bool,
) -> H4HoldoutVarianceRunResult:
    """Execute a virgin Spec-38 plan or resume the same exact bound campaign."""
    _require_parallel_host()
    assets = load_h4_holdout_variance_launch_assets(
        protocol_path=protocol_path,
        execution_contract_path=execution_contract_path,
        population_path=population_path,
        acceptance_path=acceptance_path,
        snapshot_root=snapshot_root,
        gate6_report_path=gate6_report_path,
        reference_path=reference_path,
    )
    persistence = _persistence_paths(database=database, work_root=work_root, repo_root=repo_root)
    _require_frozen_persistence_paths(assets.contract, persistence)
    persisted = _load_binding(work_root=work_root, contract=assets.contract)
    _recapture_same_bound_execution(
        assets=assets,
        persisted=persisted,
        persistence=persistence,
        repo_root=repo_root,
        poetry_lock_path=poetry_lock_path,
    )
    status = _status(
        protocol=assets.protocol,
        population=assets.population,
        binding=persisted,
        work_root=work_root,
    )
    if resume:
        if status.virgin:
            raise H4HoldoutVarianceParallelLaunchError(
                "variance pilot is still a virgin plan; use run rather than resume"
            )
    else:
        if not status.virgin:
            raise H4HoldoutVarianceParallelLaunchError(
                "variance pilot already contains execution state; use resume"
            )
        ready = assert_h4_holdout_variance_parallel_launch_ready(
            protocol_path=protocol_path,
            execution_contract_path=execution_contract_path,
            population_path=population_path,
            acceptance_path=acceptance_path,
            snapshot_root=snapshot_root,
            gate6_report_path=gate6_report_path,
            reference_path=reference_path,
            database=database,
            work_root=work_root,
            repo_root=repo_root,
            poetry_lock_path=poetry_lock_path,
        )
        if ready.binding_hash != status.binding_hash:
            raise H4HoldoutVarianceParallelLaunchError(
                "first-hand readiness/status binding mismatch"
            )

    database_path = Path(database)
    if not database_path.exists():
        raise H4HoldoutVarianceParallelLaunchError(
            "planned variance-pilot database is missing"
        )
    root = Path(work_root).resolve()

    def task_builder(index, job, config, match_id):  # type: ignore[no-untyped-def]
        return H4HoldoutVarianceParallelWorkerTask(
            protocol_path=str(Path(protocol_path).resolve()),
            execution_contract_path=str(Path(execution_contract_path).resolve()),
            population_path=str(Path(population_path).resolve()),
            acceptance_path=str(Path(acceptance_path).resolve()),
            snapshot_root=str(Path(snapshot_root).resolve()),
            gate6_report_path=str(Path(gate6_report_path).resolve()),
            reference_path=str(Path(reference_path).resolve()),
            job_index=index,
            trajectory_config_path=str(
                root / "trajectory-configs" / f"{h4_holdout_variance_job_key(job)}.json"
            ),
            match_id=match_id,
            binding_path=str(root / "binding.json"),
        )

    store = SQLiteBenchmarkStore(database_path)
    try:
        return execute_h4_holdout_variance_parallel_run(
            protocol=assets.protocol,
            binding=persisted,
            population=assets.population,
            work_root=work_root,
            benchmark_store=store,
            task_builder=task_builder,
        )
    except (
        OSError,
        ValueError,
        H4HoldoutVarianceRunError,
        H4HoldoutVarianceParallelRunError,
        H4HoldoutVarianceExecutionError,
    ) as exc:
        raise H4HoldoutVarianceParallelLaunchError(
            f"Spec-38 parallel variance-pilot execution failed: {exc}"
        ) from exc
