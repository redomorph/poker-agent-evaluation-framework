"""Frozen parallel execution authorization for the Phase-2 adaptive pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_ID = "phase2-adaptive-pilot-parallel-execution-v1"
ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_VERSION = (
    "phase2-adaptive-pilot-parallel-execution-contract-v1"
)
ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_FILE_VERSION = (
    "frozen-phase2-adaptive-pilot-parallel-execution-contract-file-v1"
)
AUTHORIZED_ADAPTIVE_PILOT_PROTOCOL_HASH = (
    "ef86d3da94369a86ac1c496ddf981b39d30919e086b35e026039bac3afc8ccb4"
)


class AdaptivePilotParallelExecutionContractError(ValueError):
    """Raised when the frozen Spec-34 parallel execution contract is invalid."""


@dataclass(frozen=True)
class AdaptivePilotParallelExecutionContract:
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
    worker_model: str
    parent_commit_model: str
    worker_campaign_persistence: bool
    planned_jobs: int
    planned_pairs: int
    planned_engine_hands: int


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_parallel_execution_contract_file_bytes(value: object) -> bytes:
    return _canonical(value) + b"\n"


def expected_adaptive_pilot_parallel_execution_contract_payload() -> dict[str, Any]:
    return {
        "execution_contract_id": ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_ID,
        "execution_contract_version": ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_VERSION,
        "status": "frozen",
        "purpose": "parallel_operational_execution_authorization_only",
        "authorized_protocol": {
            "protocol_id": "phase2-adaptive-pilot-v1",
            "protocol_version": "phase2-adaptive-pilot-protocol-v1",
            "payload_hash": AUTHORIZED_ADAPTIVE_PILOT_PROTOCOL_HASH,
        },
        "execution_components": {
            "execution_enforcement_version": "phase2-adaptive-pilot-execution-enforcement-v1",
            "execution_binding_version": "phase2-adaptive-pilot-execution-binding-v1",
            "trajectory_config_version": "phase2-adaptive-pilot-trajectory-config-v1",
            "run_manifest_version": "phase2-adaptive-pilot-run-manifest-v1",
            "arm_snapshot_version": "phase2-adaptive-pilot-arm-snapshot-v1",
            "pair_checkpoint_version": "phase2-adaptive-pilot-pair-checkpoint-v1",
            "parallel_run_version": "phase2-adaptive-pilot-parallel-run-v1",
            "parallel_launch_version": "phase2-adaptive-pilot-parallel-launch-v1",
        },
        "cli": {
            "module": "pokerbench.experiments.adaptive_pilot_parallel_cli",
            "commands": ["plan", "status", "readiness", "run", "resume"],
            "execution_contract_default": (
                "configs/experiments/phase2-adaptive-pilot-parallel-execution-v1.json"
            ),
        },
        "workers": 4,
        "max_in_flight_jobs": 4,
        "worker_model": "process_pool_spawn_compatible_top_level_task",
        "parent_commit_model": "single_parent_serial_commit",
        "worker_campaign_persistence": False,
        "matrix": {
            "planned_jobs": 50,
            "planned_pairs": 25,
            "planned_engine_hands": 18_000,
        },
        "persistence": {
            "worker_campaign_writes": False,
            "parent_commit_order": [
                "arm_snapshot",
                "campaign_sqlite_match_record",
                "manifest_terminal_state",
                "pair_checkpoint",
            ],
            "replacement_seed_policy": "forbidden",
            "orphan_match_record_policy": "fail_closed",
        },
        "authorization_boundary": {
            "pilot_execution_authorized_after_gate": True,
            "confirmatory_h4_inference": False,
            "final_h4_data_collection": False,
            "pilot_final_pooling": False,
        },
    }


ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_HASH = (
    "1959966b1c5116dbdd94fe695d76fe648d25b439d2282da52947a20749a4ba00"
)


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdaptivePilotParallelExecutionContractError(f"{field_name} must be a JSON object")
    return value


def validate_adaptive_pilot_parallel_execution_contract_document(
    document: object,
) -> AdaptivePilotParallelExecutionContract:
    root = _mapping(document, "parallel execution contract document")
    if set(root) != {"file_version", "payload", "payload_hash"}:
        raise AdaptivePilotParallelExecutionContractError("invalid parallel execution contract document keys")
    if root["file_version"] != ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_FILE_VERSION:
        raise AdaptivePilotParallelExecutionContractError("invalid parallel execution contract file version")
    payload = _mapping(root["payload"], "payload")
    payload_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    if root["payload_hash"] != payload_hash:
        raise AdaptivePilotParallelExecutionContractError("parallel execution contract payload hash mismatch")
    if dict(payload) != expected_adaptive_pilot_parallel_execution_contract_payload():
        raise AdaptivePilotParallelExecutionContractError(
            "adaptive pilot parallel execution contract semantics differ from frozen v1"
        )
    if payload_hash != ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_HASH:
        raise AdaptivePilotParallelExecutionContractError(
            "adaptive pilot parallel execution contract hash differs from frozen v1"
        )
    authorized = _mapping(payload["authorized_protocol"], "authorized_protocol")
    components = _mapping(payload["execution_components"], "execution_components")
    cli = _mapping(payload["cli"], "cli")
    matrix = _mapping(payload["matrix"], "matrix")
    commands = cli["commands"]
    if not isinstance(commands, list):
        raise AdaptivePilotParallelExecutionContractError("cli.commands must be an array")
    return AdaptivePilotParallelExecutionContract(
        execution_contract_id=str(payload["execution_contract_id"]),
        execution_contract_version=str(payload["execution_contract_version"]),
        status=str(payload["status"]),
        purpose=str(payload["purpose"]),
        payload_hash=payload_hash,
        authorized_protocol_id=str(authorized["protocol_id"]),
        authorized_protocol_version=str(authorized["protocol_version"]),
        authorized_protocol_hash=str(authorized["payload_hash"]),
        component_versions=tuple(sorted((str(k), str(v)) for k, v in components.items())),
        commands=tuple(str(item) for item in commands),
        workers=int(payload["workers"]),
        max_in_flight_jobs=int(payload["max_in_flight_jobs"]),
        worker_model=str(payload["worker_model"]),
        parent_commit_model=str(payload["parent_commit_model"]),
        worker_campaign_persistence=bool(payload["worker_campaign_persistence"]),
        planned_jobs=int(matrix["planned_jobs"]),
        planned_pairs=int(matrix["planned_pairs"]),
        planned_engine_hands=int(matrix["planned_engine_hands"]),
    )


def load_adaptive_pilot_parallel_execution_contract(
    path: str | Path,
) -> AdaptivePilotParallelExecutionContract:
    source = Path(path)
    raw = source.read_bytes()
    document = json.loads(raw)
    if raw != canonical_parallel_execution_contract_file_bytes(document):
        raise AdaptivePilotParallelExecutionContractError(
            "adaptive pilot parallel execution contract file is not canonical JSON/LF"
        )
    return validate_adaptive_pilot_parallel_execution_contract_document(document)


def frozen_adaptive_pilot_parallel_execution_contract() -> AdaptivePilotParallelExecutionContract:
    payload = expected_adaptive_pilot_parallel_execution_contract_payload()
    return validate_adaptive_pilot_parallel_execution_contract_document(
        {
            "file_version": ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_FILE_VERSION,
            "payload": payload,
            "payload_hash": hashlib.sha256(_canonical(payload)).hexdigest(),
        }
    )
