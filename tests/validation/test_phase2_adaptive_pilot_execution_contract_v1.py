from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.experiments.adaptive_pilot_binding import (
    AdaptivePilotBindingError,
    build_adaptive_pilot_execution_binding,
    build_adaptive_pilot_trajectory_config,
)
from pokerbench.experiments.adaptive_pilot_execution_contract import (
    ADAPTIVE_PILOT_EXECUTION_CONTRACT_HASH,
    ADAPTIVE_PILOT_EXECUTION_CONTRACT_ID,
    ADAPTIVE_PILOT_EXECUTION_CONTRACT_VERSION,
    AdaptivePilotExecutionContractError,
    canonical_execution_contract_file_bytes,
    load_adaptive_pilot_execution_contract,
)
from pokerbench.experiments.adaptive_pilot_launch import (
    AdaptivePilotLaunchError,
    plan_adaptive_pilot_launch,
    run_adaptive_pilot_launch,
)
from pokerbench.experiments.adaptive_pilot_protocol import (
    ADAPTIVE_PILOT_PROTOCOL_HASH,
    iter_adaptive_pilot_jobs,
    load_adaptive_pilot_protocol,
)
from pokerbench.experiments.adaptive_pilot_cli import build_parser
from pokerbench.experiments.config import ExecutionEnvironment
from pokerbench.experiments.frozen_protocol import load_frozen_population


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-v1.json"
EXECUTION_CONTRACT_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-execution-v1.json"
PRIMARY_PATH = ROOT / "configs" / "populations" / "baseline-synthetic-v1.json"
SECONDARY_PATH = ROOT / "configs" / "populations" / "balanced-v1.json"
GATE6_PATH = ROOT / "data" / "gate6" / "gate6-report.json"


def _environment() -> ExecutionEnvironment:
    commit = "a" * 40
    return ExecutionEnvironment(
        python_version="3.14.0",
        engine_version=f"git:{commit}",
        dependency_lockfile="poetry.lock",
        poetry_lock_hash="b" * 64,
        git_commit=commit,
        cpu_model="Test CPU",
        cpu_core_count=8,
        cpu_architecture="AMD64",
        ram_bytes=16 * 1024**3,
        os_identifier="TestOS",
    )


def _binding(contract=None):
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    return build_adaptive_pilot_execution_binding(
        protocol=protocol,
        execution_contract=contract,
        gate6_report=load_gate6_calibration_report(GATE6_PATH),
        primary_population=load_frozen_population(PRIMARY_PATH),
        secondary_populations=(load_frozen_population(SECONDARY_PATH),),
        environment=_environment(),
        workers=1,
    )


def test_execution_contract_file_is_canonical_hash_bound_and_semantically_frozen(tmp_path: Path) -> None:
    contract = load_adaptive_pilot_execution_contract(EXECUTION_CONTRACT_PATH)
    raw = EXECUTION_CONTRACT_PATH.read_bytes()
    document = json.loads(raw)

    assert raw == canonical_execution_contract_file_bytes(document)
    assert contract.execution_contract_id == ADAPTIVE_PILOT_EXECUTION_CONTRACT_ID
    assert contract.execution_contract_version == ADAPTIVE_PILOT_EXECUTION_CONTRACT_VERSION
    assert contract.payload_hash == ADAPTIVE_PILOT_EXECUTION_CONTRACT_HASH
    assert contract.status == "frozen"

    tampered = json.loads(raw)
    tampered["payload"]["workers"] = 2
    canonical_payload = json.dumps(
        tampered["payload"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    tampered["payload_hash"] = hashlib.sha256(canonical_payload).hexdigest()
    path = tmp_path / "rehashed-tamper.json"
    path.write_bytes(canonical_execution_contract_file_bytes(tampered))

    with pytest.raises(AdaptivePilotExecutionContractError, match="semantics differ"):
        load_adaptive_pilot_execution_contract(path)


def test_execution_contract_authorizes_only_exact_spec32_and_frozen_v1_components() -> None:
    contract = load_adaptive_pilot_execution_contract(EXECUTION_CONTRACT_PATH)

    assert contract.authorized_protocol_id == "phase2-adaptive-pilot-v1"
    assert contract.authorized_protocol_version == "phase2-adaptive-pilot-protocol-v1"
    assert contract.authorized_protocol_hash == ADAPTIVE_PILOT_PROTOCOL_HASH
    assert contract.workers == 1
    assert dict(contract.component_versions) == {
        "arm_snapshot_version": "phase2-adaptive-pilot-arm-snapshot-v1",
        "execution_binding_version": "phase2-adaptive-pilot-execution-binding-v1",
        "execution_enforcement_version": "phase2-adaptive-pilot-execution-enforcement-v1",
        "launch_version": "phase2-adaptive-pilot-launch-v1",
        "pair_checkpoint_version": "phase2-adaptive-pilot-pair-checkpoint-v1",
        "run_manifest_version": "phase2-adaptive-pilot-run-manifest-v1",
        "trajectory_config_version": "phase2-adaptive-pilot-trajectory-config-v1",
    }
    assert contract.commands == ("plan", "status", "readiness", "run", "resume")
    assert (contract.planned_jobs, contract.planned_pairs, contract.planned_engine_hands) == (50, 25, 18_000)


def test_binding_and_trajectory_config_bind_exact_execution_contract() -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    contract = load_adaptive_pilot_execution_contract(EXECUTION_CONTRACT_PATH)
    binding = _binding(contract)

    assert binding.execution_contract_id == contract.execution_contract_id
    assert binding.execution_contract_version == contract.execution_contract_version
    assert binding.execution_contract_hash == contract.payload_hash

    job = iter_adaptive_pilot_jobs(protocol)[0]
    config = build_adaptive_pilot_trajectory_config(protocol=protocol, binding=binding, job=job)
    embedded = config.payload()["execution_binding"]
    assert embedded["execution_contract_id"] == contract.execution_contract_id
    assert embedded["execution_contract_version"] == contract.execution_contract_version
    assert embedded["execution_contract_hash"] == contract.payload_hash
    assert config.run_provenance["execution_contract_id"] == contract.execution_contract_id
    assert config.run_provenance["execution_contract_version"] == contract.execution_contract_version
    assert config.run_provenance["execution_contract_hash"] == contract.payload_hash


def test_binding_rejects_execution_contract_that_authorizes_different_protocol() -> None:
    contract = load_adaptive_pilot_execution_contract(EXECUTION_CONTRACT_PATH)
    drifted = replace(contract, authorized_protocol_hash="f" * 64)

    with pytest.raises(AdaptivePilotBindingError, match="execution contract.*protocol hash"):
        _binding(drifted)


def test_plan_fails_closed_when_execution_contract_is_missing(tmp_path: Path) -> None:
    with pytest.raises(AdaptivePilotLaunchError, match="execution contract"):
        plan_adaptive_pilot_launch(
            protocol_path=PROTOCOL_PATH,
            execution_contract_path=tmp_path / "missing-execution-contract.json",
            primary_population_path=PRIMARY_PATH,
            secondary_population_path=SECONDARY_PATH,
            snapshot_root=ROOT / "data" / "gate6" / "snapshots",
            gate6_report_path=GATE6_PATH,
            reference_path=ROOT / "configs" / "experiments" / "preflop-range-strength-reference-v1.json",
            database=tmp_path / "pilot.sqlite",
            work_root=tmp_path / "work",
            repo_root=ROOT,
            poetry_lock_path=ROOT / "poetry.lock",
            ram_bytes=16 * 1024**3,
            workers=1,
        )
    assert not (tmp_path / "work" / "binding.json").exists()


def test_cli_binds_execution_contract_path_for_all_five_commands() -> None:
    parser = build_parser()
    expected = Path("configs/experiments/phase2-adaptive-pilot-execution-v1.json")
    for command in ("plan", "status", "readiness", "run", "resume"):
        argv = [command]
        if command == "plan":
            argv += ["--ram-bytes", "1024"]
        args = parser.parse_args(argv)
        assert args.execution_contract == expected


def test_all_50_trajectory_configs_bind_the_exact_execution_contract() -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    contract = load_adaptive_pilot_execution_contract(EXECUTION_CONTRACT_PATH)
    binding = _binding(contract)

    configs = [
        build_adaptive_pilot_trajectory_config(protocol=protocol, binding=binding, job=job)
        for job in iter_adaptive_pilot_jobs(protocol)
    ]

    assert len(configs) == contract.planned_jobs == 50
    assert {config.binding.execution_contract_id for config in configs} == {contract.execution_contract_id}
    assert {config.binding.execution_contract_version for config in configs} == {contract.execution_contract_version}
    assert {config.binding.execution_contract_hash for config in configs} == {contract.payload_hash}


@pytest.mark.parametrize("resume", [False, True])
def test_run_and_resume_cannot_bypass_execution_contract_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    called = False

    def forbidden_execute(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("execution must not be reached")

    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_launch.execute_adaptive_pilot_run",
        forbidden_execute,
    )

    with pytest.raises(AdaptivePilotLaunchError, match="execution contract"):
        run_adaptive_pilot_launch(
            protocol_path=PROTOCOL_PATH,
            execution_contract_path=tmp_path / "missing-execution-contract.json",
            primary_population_path=PRIMARY_PATH,
            secondary_population_path=SECONDARY_PATH,
            snapshot_root=ROOT / "data" / "gate6" / "snapshots",
            gate6_report_path=GATE6_PATH,
            reference_path=ROOT / "configs" / "experiments" / "preflop-range-strength-reference-v1.json",
            database=tmp_path / "pilot.sqlite",
            work_root=tmp_path / "work",
            repo_root=ROOT,
            poetry_lock_path=ROOT / "poetry.lock",
            resume=resume,
        )

    assert called is False
