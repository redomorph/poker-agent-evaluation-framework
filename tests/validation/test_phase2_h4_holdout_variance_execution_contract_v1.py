from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pokerbench.experiments.h4_holdout_variance_execution_contract import (
    AUTHORIZED_H4_HOLDOUT_VARIANCE_PROTOCOL_HASH,
    H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH,
    H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_ID,
    H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_VERSION,
    H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_FILE_VERSION,
    H4_HOLDOUT_VARIANCE_MAX_IN_FLIGHT_JOBS,
    H4_HOLDOUT_VARIANCE_PROCESS_MODEL,
    H4_HOLDOUT_VARIANCE_WORKERS,
    H4HoldoutVarianceExecutionContractError,
    canonical_h4_holdout_variance_execution_contract_file_bytes,
    expected_h4_holdout_variance_execution_contract_payload,
    frozen_h4_holdout_variance_execution_contract,
    load_h4_holdout_variance_execution_contract,
    validate_h4_holdout_variance_execution_contract_document,
)
from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4_HOLDOUT_VARIANCE_PROTOCOL_HASH,
    load_h4_holdout_variance_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs" / "experiments" / "phase2-h4-holdout-variance-execution-v1.json"
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-h4-holdout-variance-pilot-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def test_frozen_spec38_execution_contract_is_canonical_and_hash_bound() -> None:
    raw = CONTRACT_PATH.read_bytes()
    document = json.loads(raw)
    contract = load_h4_holdout_variance_execution_contract(CONTRACT_PATH)

    assert raw == canonical_h4_holdout_variance_execution_contract_file_bytes(document)
    assert raw.endswith(b"\n") and not raw.endswith(b"\r\n")
    assert document["file_version"] == H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_FILE_VERSION
    assert document["payload_hash"] == H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH
    assert hashlib.sha256(_canonical(document["payload"])).hexdigest() == H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH
    assert contract.payload_hash == H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH


def test_spec38_authorizes_exact_spec37_protocol_only() -> None:
    protocol = load_h4_holdout_variance_protocol(PROTOCOL_PATH)
    contract = load_h4_holdout_variance_execution_contract(CONTRACT_PATH)

    assert protocol.payload_hash == H4_HOLDOUT_VARIANCE_PROTOCOL_HASH
    assert AUTHORIZED_H4_HOLDOUT_VARIANCE_PROTOCOL_HASH == H4_HOLDOUT_VARIANCE_PROTOCOL_HASH
    assert contract.authorized_protocol_id == protocol.protocol_id
    assert contract.authorized_protocol_version == protocol.protocol_version
    assert contract.authorized_protocol_hash == protocol.payload_hash


def test_spec38_freezes_four_worker_matrix_components_and_persistence_roots() -> None:
    contract = load_h4_holdout_variance_execution_contract(CONTRACT_PATH)
    components = dict(contract.component_versions)
    paths = dict(contract.persistence_paths)

    assert contract.execution_contract_id == H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_ID
    assert contract.execution_contract_version == H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_VERSION
    assert contract.workers == H4_HOLDOUT_VARIANCE_WORKERS == 4
    assert contract.max_in_flight_jobs == H4_HOLDOUT_VARIANCE_MAX_IN_FLIGHT_JOBS == 4
    assert contract.process_model == H4_HOLDOUT_VARIANCE_PROCESS_MODEL
    assert contract.worker_campaign_persistence is False
    assert (contract.planned_jobs, contract.planned_pairs, contract.planned_clusters, contract.planned_engine_hands) == (
        100,
        50,
        10,
        100_000,
    )
    assert components["parallel_launch_version"] == "phase2-h4-holdout-variance-parallel-launch-v1"
    assert components["parallel_run_version"] == "phase2-h4-holdout-variance-parallel-run-v1"
    assert paths == {
        "work_root": "data/phase2/h4-holdout-variance/work",
        "database": "data/phase2/h4-holdout-variance/variance-pilot.sqlite",
        "trajectory_configs": "data/phase2/h4-holdout-variance/work/trajectory-configs",
        "arm_snapshots": "data/phase2/h4-holdout-variance/work/arm-snapshots",
        "pair_checkpoints": "data/phase2/h4-holdout-variance/work/pair-checkpoints",
        "cluster_checkpoints": "data/phase2/h4-holdout-variance/work/cluster-checkpoints",
    }


def test_spec38_semantic_tamper_is_rejected_even_after_rehash() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    document["payload"]["workers"] = 3
    document["payload_hash"] = hashlib.sha256(_canonical(document["payload"])).hexdigest()

    with pytest.raises(H4HoldoutVarianceExecutionContractError, match="semantics differ"):
        validate_h4_holdout_variance_execution_contract_document(document)


def test_spec38_loader_rejects_noncanonical_crlf_bytes(tmp_path: Path) -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "contract.json"
    path.write_bytes(_canonical(document) + b"\r\n")

    with pytest.raises(H4HoldoutVarianceExecutionContractError, match="canonical JSON"):
        load_h4_holdout_variance_execution_contract(path)


def test_spec38_expected_payload_and_frozen_object_are_exact() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    frozen = frozen_h4_holdout_variance_execution_contract()

    assert document["payload"] == expected_h4_holdout_variance_execution_contract_payload()
    assert frozen == load_h4_holdout_variance_execution_contract(CONTRACT_PATH)
    assert frozen.commands == ("plan", "status", "readiness", "run", "resume")
