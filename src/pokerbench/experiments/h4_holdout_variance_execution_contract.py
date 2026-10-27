"""Frozen operational execution authority for the Spec-37 holdout variance pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_ID = "phase2-h4-holdout-variance-execution-v1"
H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_VERSION = (
    "phase2-h4-holdout-variance-execution-contract-v1"
)
H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_FILE_VERSION = (
    "frozen-phase2-h4-holdout-variance-execution-contract-file-v1"
)
AUTHORIZED_H4_HOLDOUT_VARIANCE_PROTOCOL_HASH = (
    "d2ad5569a82308a61390ccdc2212ac3b2790fa369e96506a418bac4a7045aac0"
)
H4_HOLDOUT_VARIANCE_WORKERS = 4
H4_HOLDOUT_VARIANCE_MAX_IN_FLIGHT_JOBS = 4
H4_HOLDOUT_VARIANCE_PROCESS_MODEL = "spawn-compatible-process-pool-v1"
H4_HOLDOUT_VARIANCE_WORKER_MODEL = "process_pool_spawn_compatible_top_level_task"
H4_HOLDOUT_VARIANCE_PARENT_COMMIT_MODEL = "single_parent_serial_commit"


class H4HoldoutVarianceExecutionContractError(ValueError):
    """Raised when the frozen Spec-38 execution authority is invalid."""


@dataclass(frozen=True)
class H4HoldoutVarianceExecutionContract:
    execution_contract_id: str
    execution_contract_version: str
    status: str
    purpose: str
    payload_hash: str
    authorized_protocol_id: str
    authorized_protocol_version: str
    authorized_protocol_hash: str
    component_versions: tuple[tuple[str, str], ...]
    commands: tuple[str, ...]
    workers: int
    max_in_flight_jobs: int
    process_model: str
    worker_model: str
    parent_commit_model: str
    worker_campaign_persistence: bool
    planned_jobs: int
    planned_pairs: int
    planned_clusters: int
    planned_engine_hands: int
    persistence_paths: tuple[tuple[str, str], ...]

    def authority_payload(self) -> dict[str, str]:
        return {
            "authority_id": self.execution_contract_id,
            "authority_version": self.execution_contract_version,
            "payload_hash": self.payload_hash,
            "authorized_protocol_id": self.authorized_protocol_id,
            "authorized_protocol_version": self.authorized_protocol_version,
            "authorized_protocol_hash": self.authorized_protocol_hash,
        }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_h4_holdout_variance_execution_contract_file_bytes(value: object) -> bytes:
    return _canonical(value) + b"\n"


def expected_h4_holdout_variance_execution_contract_payload() -> dict[str, Any]:
    return {
        "execution_contract_id": H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_ID,
        "execution_contract_version": H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_VERSION,
        "status": "frozen",
        "purpose": "operational_execution_authorization_and_reproducibility_only",
        "authorized_protocol": {
            "protocol_id": "phase2-h4-holdout-variance-pilot-v1",
            "protocol_version": "phase2-h4-holdout-variance-pilot-protocol-v1",
            "payload_hash": AUTHORIZED_H4_HOLDOUT_VARIANCE_PROTOCOL_HASH,
        },
        "execution_components": {
            "execution_enforcement_version": "phase2-h4-holdout-variance-execution-enforcement-v1",
            "execution_binding_version": "phase2-h4-holdout-variance-execution-binding-v1",
            "trajectory_config_version": "phase2-h4-holdout-variance-trajectory-config-v1",
            "persistence_layout_version": "phase2-h4-holdout-variance-persistence-layout-v1",
            "run_manifest_version": "phase2-h4-holdout-variance-run-manifest-v1",
            "arm_snapshot_version": "phase2-h4-holdout-variance-arm-snapshot-v1",
            "pair_checkpoint_version": "phase2-h4-holdout-variance-pair-checkpoint-v1",
            "cluster_checkpoint_version": "phase2-h4-holdout-variance-cluster-checkpoint-v1",
            "parallel_run_version": "phase2-h4-holdout-variance-parallel-run-v1",
            "parallel_launch_version": "phase2-h4-holdout-variance-parallel-launch-v1",
        },
        "cli": {
            "module": "pokerbench.experiments.h4_holdout_variance_parallel_cli",
            "commands": ["plan", "status", "readiness", "run", "resume"],
            "defaults": {
                "protocol": "configs/experiments/phase2-h4-holdout-variance-pilot-v1.json",
                "execution_contract": "configs/experiments/phase2-h4-holdout-variance-execution-v1.json",
                "population": "configs/populations/phase2-h4-holdout-population-v1.json",
                "acceptance": "data/gate6/h4-holdout-acceptance-v1.json",
                "snapshot_root": "data/gate6/snapshots",
                "gate6_report": "data/gate6/gate6-report.json",
                "reference": "configs/experiments/preflop-range-strength-reference-v1.json",
                "database": "data/phase2/h4-holdout-variance/variance-pilot.sqlite",
                "work_root": "data/phase2/h4-holdout-variance/work",
                "repo_root": ".",
                "poetry_lock": "poetry.lock",
            },
        },
        "workers": H4_HOLDOUT_VARIANCE_WORKERS,
        "max_in_flight_jobs": H4_HOLDOUT_VARIANCE_MAX_IN_FLIGHT_JOBS,
        "process_model": H4_HOLDOUT_VARIANCE_PROCESS_MODEL,
        "worker_model": H4_HOLDOUT_VARIANCE_WORKER_MODEL,
        "parent_commit_model": H4_HOLDOUT_VARIANCE_PARENT_COMMIT_MODEL,
        "worker_campaign_persistence": False,
        "matrix": {
            "planned_jobs": 100,
            "planned_pairs": 50,
            "planned_clusters": 10,
            "planned_engine_hands": 100_000,
        },
        "persistence": {
            "paths": {
                "work_root": "data/phase2/h4-holdout-variance/work",
                "database": "data/phase2/h4-holdout-variance/variance-pilot.sqlite",
                "trajectory_configs": "data/phase2/h4-holdout-variance/work/trajectory-configs",
                "arm_snapshots": "data/phase2/h4-holdout-variance/work/arm-snapshots",
                "pair_checkpoints": "data/phase2/h4-holdout-variance/work/pair-checkpoints",
                "cluster_checkpoints": "data/phase2/h4-holdout-variance/work/cluster-checkpoints",
            },
            "parent_commit_order": [
                "arm_snapshot",
                "campaign_sqlite_match_record",
                "manifest_terminal_state",
                "pair_checkpoint",
                "cluster_checkpoint",
            ],
            "snapshot_before_database": True,
            "replacement_seed_policy": "forbidden",
            "orphan_match_record_policy": "fail_closed",
            "stale_running_without_durable_evidence": "reset_to_planned_on_resume_only",
        },
        "preflight": {
            "plan_executes_hands": False,
            "readiness_executes_hands": False,
            "readiness_byte_mutation": "forbidden",
            "required_initial_match_records": 0,
            "required_initial_arm_snapshots": 0,
            "required_initial_pair_checkpoints": 0,
            "required_initial_cluster_checkpoints": 0,
            "scientific_tree_must_be_clean_and_committed": True,
            "minimum_logical_cpu_cores": 4,
        },
        "authorization_boundary": {
            "variance_pilot_execution_authorized_after_complete_spec38_gate": True,
            "confirmatory_h4_inference": False,
            "final_h4_data_collection": False,
            "variance_pilot_mean_as_design_input": False,
            "variance_pilot_final_pooling": False,
            "reduced_sample_sigma_500": False,
            "replacement_seeds": False,
        },
    }


# Frozen below after canonical payload construction.
H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH = "8d68c39a1248c1432658b77f857b8d971fa5e171322ccc529e48861e94824265"


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise H4HoldoutVarianceExecutionContractError(f"{field_name} must be a JSON object")
    return value


def validate_h4_holdout_variance_execution_contract_document(
    document: object,
) -> H4HoldoutVarianceExecutionContract:
    root = _mapping(document, "Spec-38 execution contract document")
    if set(root) != {"file_version", "payload", "payload_hash"}:
        raise H4HoldoutVarianceExecutionContractError("invalid Spec-38 execution contract document keys")
    if root["file_version"] != H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_FILE_VERSION:
        raise H4HoldoutVarianceExecutionContractError("invalid Spec-38 execution contract file version")
    payload = _mapping(root["payload"], "payload")
    payload_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    if root["payload_hash"] != payload_hash:
        raise H4HoldoutVarianceExecutionContractError("Spec-38 execution contract payload hash mismatch")
    if dict(payload) != expected_h4_holdout_variance_execution_contract_payload():
        raise H4HoldoutVarianceExecutionContractError(
            "Spec-38 execution contract semantics differ from frozen v1"
        )
    if payload_hash != H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH:
        raise H4HoldoutVarianceExecutionContractError(
            "Spec-38 execution contract hash differs from frozen v1"
        )

    authorized = _mapping(payload["authorized_protocol"], "authorized_protocol")
    components = _mapping(payload["execution_components"], "execution_components")
    cli = _mapping(payload["cli"], "cli")
    matrix = _mapping(payload["matrix"], "matrix")
    persistence = _mapping(payload["persistence"], "persistence")
    paths = _mapping(persistence["paths"], "persistence.paths")
    commands = cli["commands"]
    if not isinstance(commands, list):
        raise H4HoldoutVarianceExecutionContractError("cli.commands must be an array")

    return H4HoldoutVarianceExecutionContract(
        execution_contract_id=str(payload["execution_contract_id"]),
        execution_contract_version=str(payload["execution_contract_version"]),
        status=str(payload["status"]),
        purpose=str(payload["purpose"]),
        payload_hash=payload_hash,
        authorized_protocol_id=str(authorized["protocol_id"]),
        authorized_protocol_version=str(authorized["protocol_version"]),
        authorized_protocol_hash=str(authorized["payload_hash"]),
        component_versions=tuple(sorted((str(key), str(value)) for key, value in components.items())),
        commands=tuple(str(item) for item in commands),
        workers=int(payload["workers"]),
        max_in_flight_jobs=int(payload["max_in_flight_jobs"]),
        process_model=str(payload["process_model"]),
        worker_model=str(payload["worker_model"]),
        parent_commit_model=str(payload["parent_commit_model"]),
        worker_campaign_persistence=bool(payload["worker_campaign_persistence"]),
        planned_jobs=int(matrix["planned_jobs"]),
        planned_pairs=int(matrix["planned_pairs"]),
        planned_clusters=int(matrix["planned_clusters"]),
        planned_engine_hands=int(matrix["planned_engine_hands"]),
        persistence_paths=tuple(sorted((str(key), str(value)) for key, value in paths.items())),
    )


def load_h4_holdout_variance_execution_contract(
    path: str | Path,
) -> H4HoldoutVarianceExecutionContract:
    source = Path(path)
    try:
        raw = source.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise H4HoldoutVarianceExecutionContractError("invalid Spec-38 execution contract JSON") from exc
    if raw != canonical_h4_holdout_variance_execution_contract_file_bytes(document):
        raise H4HoldoutVarianceExecutionContractError(
            "Spec-38 execution contract file is not canonical JSON with one trailing LF"
        )
    return validate_h4_holdout_variance_execution_contract_document(document)


def frozen_h4_holdout_variance_execution_contract() -> H4HoldoutVarianceExecutionContract:
    payload = expected_h4_holdout_variance_execution_contract_payload()
    return validate_h4_holdout_variance_execution_contract_document(
        {
            "file_version": H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_FILE_VERSION,
            "payload": payload,
            "payload_hash": hashlib.sha256(_canonical(payload)).hexdigest(),
        }
    )
