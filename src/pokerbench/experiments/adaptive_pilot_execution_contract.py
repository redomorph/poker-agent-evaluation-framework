"""Frozen machine-readable execution authorization for the Phase-2 adaptive pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTIVE_PILOT_EXECUTION_CONTRACT_ID = "phase2-adaptive-pilot-execution-v1"
ADAPTIVE_PILOT_EXECUTION_CONTRACT_VERSION = "phase2-adaptive-pilot-execution-contract-v1"
ADAPTIVE_PILOT_EXECUTION_CONTRACT_FILE_VERSION = (
    "frozen-phase2-adaptive-pilot-execution-contract-file-v1"
)
ADAPTIVE_PILOT_EXECUTION_CONTRACT_HASH = (
    "c7bfd26a28b54f750a70492f31c4895c8eaacf8a05f381c5f7e85525ae82f398"
)
AUTHORIZED_ADAPTIVE_PILOT_PROTOCOL_HASH = (
    "ef86d3da94369a86ac1c496ddf981b39d30919e086b35e026039bac3afc8ccb4"
)


class AdaptivePilotExecutionContractError(ValueError):
    """Raised when the frozen Spec-33 execution authorization is invalid."""


@dataclass(frozen=True)
class AdaptivePilotExecutionContract:
    """Validated immutable execution authorization used by launch/binding code."""

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


def canonical_execution_contract_file_bytes(value: object) -> bytes:
    """Return canonical UTF-8 JSON with exactly one trailing LF."""
    return _canonical(value) + b"\n"


def expected_adaptive_pilot_execution_contract_payload() -> dict[str, Any]:
    """Return the exact machine-readable semantics frozen by Spec 33."""
    return {
        "execution_contract_id": ADAPTIVE_PILOT_EXECUTION_CONTRACT_ID,
        "execution_contract_version": ADAPTIVE_PILOT_EXECUTION_CONTRACT_VERSION,
        "status": "frozen",
        "purpose": "operational_execution_authorization_and_reproducibility_only",
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
            "launch_version": "phase2-adaptive-pilot-launch-v1",
        },
        "cli": {
            "module": "pokerbench.experiments.adaptive_pilot_cli",
            "commands": ["plan", "status", "readiness", "run", "resume"],
            "execution_contract_default": (
                "configs/experiments/phase2-adaptive-pilot-execution-v1.json"
            ),
        },
        "workers": 1,
        "matrix": {
            "planned_jobs": 50,
            "planned_pairs": 25,
            "planned_engine_hands": 18_000,
        },
        "persistence": {
            "atomic_canonical_json": True,
            "replacement_seed_policy": "forbidden",
            "orphan_match_record_policy": "fail_closed",
            "snapshot_store_recovery": (
                "hash_verified_snapshot_may_restore_missing_store_copy"
            ),
            "pair_checkpoint_policy": "both_arms_terminal_and_complete_pair_validation",
        },
        "authorization_boundary": {
            "pilot_execution_authorized_after_gate": True,
            "confirmatory_h4_inference": False,
            "final_h4_data_collection": False,
            "pilot_final_pooling": False,
        },
    }


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdaptivePilotExecutionContractError(f"{field_name} must be a JSON object")
    return value


def validate_adaptive_pilot_execution_contract_document(
    document: object,
) -> AdaptivePilotExecutionContract:
    """Validate a decoded machine-readable Spec-33 execution contract."""
    root = _mapping(document, "adaptive pilot execution contract document")
    if set(root) != {"file_version", "payload", "payload_hash"}:
        raise AdaptivePilotExecutionContractError("invalid execution contract document keys")
    if root["file_version"] != ADAPTIVE_PILOT_EXECUTION_CONTRACT_FILE_VERSION:
        raise AdaptivePilotExecutionContractError("invalid execution contract file version")
    payload = _mapping(root["payload"], "payload")
    payload_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    if root["payload_hash"] != payload_hash:
        raise AdaptivePilotExecutionContractError("execution contract payload hash mismatch")
    if dict(payload) != expected_adaptive_pilot_execution_contract_payload():
        raise AdaptivePilotExecutionContractError(
            "adaptive pilot execution contract semantics differ from frozen v1"
        )
    if payload_hash != ADAPTIVE_PILOT_EXECUTION_CONTRACT_HASH:
        raise AdaptivePilotExecutionContractError(
            "adaptive pilot execution contract hash differs from frozen v1"
        )

    authorized = _mapping(payload["authorized_protocol"], "authorized_protocol")
    components = _mapping(payload["execution_components"], "execution_components")
    cli = _mapping(payload["cli"], "cli")
    matrix = _mapping(payload["matrix"], "matrix")
    commands = cli["commands"]
    if not isinstance(commands, list):
        raise AdaptivePilotExecutionContractError("cli.commands must be an array")

    return AdaptivePilotExecutionContract(
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
        planned_jobs=int(matrix["planned_jobs"]),
        planned_pairs=int(matrix["planned_pairs"]),
        planned_engine_hands=int(matrix["planned_engine_hands"]),
    )


def load_adaptive_pilot_execution_contract(
    path: str | Path,
) -> AdaptivePilotExecutionContract:
    """Load canonical Spec-33 execution authorization from disk."""
    source = Path(path)
    raw = source.read_bytes()
    document = json.loads(raw)
    if raw != canonical_execution_contract_file_bytes(document):
        raise AdaptivePilotExecutionContractError(
            "adaptive pilot execution contract file is not canonical JSON/LF"
        )
    return validate_adaptive_pilot_execution_contract_document(document)


def frozen_adaptive_pilot_execution_contract() -> AdaptivePilotExecutionContract:
    """Return the built-in exact Spec-33 contract for non-I/O construction helpers."""
    payload = expected_adaptive_pilot_execution_contract_payload()
    return validate_adaptive_pilot_execution_contract_document(
        {
            "file_version": ADAPTIVE_PILOT_EXECUTION_CONTRACT_FILE_VERSION,
            "payload": payload,
            "payload_hash": hashlib.sha256(_canonical(payload)).hexdigest(),
        }
    )
