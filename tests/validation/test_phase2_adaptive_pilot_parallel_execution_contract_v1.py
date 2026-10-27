from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pokerbench.experiments.adaptive_pilot_execution_contract import (
    ADAPTIVE_PILOT_EXECUTION_CONTRACT_HASH,
    load_adaptive_pilot_execution_contract,
)
from pokerbench.experiments.adaptive_pilot_parallel_execution_contract import (
    ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_HASH,
    ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_ID,
    AdaptivePilotParallelExecutionContractError,
    canonical_parallel_execution_contract_file_bytes,
    load_adaptive_pilot_parallel_execution_contract,
)
from pokerbench.experiments.adaptive_pilot_protocol import ADAPTIVE_PILOT_PROTOCOL_HASH

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-parallel-execution-v1.json"
SERIAL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-execution-v1.json"


def test_parallel_execution_contract_is_canonical_hash_bound_and_semantically_frozen(tmp_path: Path) -> None:
    contract = load_adaptive_pilot_parallel_execution_contract(PATH)
    raw = PATH.read_bytes()
    document = json.loads(raw)
    assert raw == canonical_parallel_execution_contract_file_bytes(document)
    assert contract.execution_contract_id == ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_ID
    assert contract.payload_hash == ADAPTIVE_PILOT_PARALLEL_EXECUTION_CONTRACT_HASH
    assert contract.payload_hash == "1959966b1c5116dbdd94fe695d76fe648d25b439d2282da52947a20749a4ba00"
    assert contract.authorized_protocol_hash == ADAPTIVE_PILOT_PROTOCOL_HASH
    assert contract.workers == 4
    assert contract.max_in_flight_jobs == 4

    tampered = json.loads(raw)
    tampered["payload"]["workers"] = 3
    encoded = json.dumps(tampered["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    tampered["payload_hash"] = hashlib.sha256(encoded).hexdigest()
    target = tmp_path / "tampered.json"
    target.write_bytes(canonical_parallel_execution_contract_file_bytes(tampered))
    with pytest.raises(AdaptivePilotParallelExecutionContractError, match="semantics differ"):
        load_adaptive_pilot_parallel_execution_contract(target)


def test_parallel_contract_preserves_exact_matrix_and_parent_only_persistence() -> None:
    contract = load_adaptive_pilot_parallel_execution_contract(PATH)
    assert (contract.planned_jobs, contract.planned_pairs, contract.planned_engine_hands) == (50, 25, 18_000)
    assert contract.worker_model == "process_pool_spawn_compatible_top_level_task"
    assert contract.parent_commit_model == "single_parent_serial_commit"
    assert contract.worker_campaign_persistence is False


def test_serial_spec33_contract_remains_unchanged() -> None:
    serial = load_adaptive_pilot_execution_contract(SERIAL_PATH)
    assert serial.payload_hash == ADAPTIVE_PILOT_EXECUTION_CONTRACT_HASH
    assert serial.workers == 1
