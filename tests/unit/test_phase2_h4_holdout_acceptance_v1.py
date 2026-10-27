from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.evaluation.calibration import CalibrationMetric
from pokerbench.evaluation.calibration_snapshots import (
    CalibrationStrengthSnapshotStore,
    policy_strength_reference_from_store,
    policy_validation_strengths_from_store,
)
from pokerbench.evaluation.gate6_execution import load_gate6_calibration_report
from pokerbench.evaluation.h4_holdout_acceptance import (
    H4_HOLDOUT_ACCEPTANCE_FILE_VERSION,
    H4_HOLDOUT_ACCEPTANCE_VERSION,
    H4HoldoutAcceptanceError,
    H4HoldoutAcceptanceNotClosedError,
    evaluate_h4_holdout_acceptance,
    load_h4_holdout_acceptance,
    require_h4_holdout_acceptance_closed,
    validate_h4_holdout_acceptance_document,
)
from pokerbench.evaluation.policy_acceptance import evaluate_parameter_acceptance
from pokerbench.experiments.h4_holdout_population import (
    H4_HOLDOUT_POPULATION_HASH,
    load_h4_holdout_population,
)

ROOT = Path(__file__).resolve().parents[2]
POPULATION_PATH = ROOT / "configs" / "populations" / "phase2-h4-holdout-population-v1.json"
ACCEPTANCE_PATH = ROOT / "data" / "gate6" / "h4-holdout-acceptance-v1.json"
GATE6_REPORT_PATH = ROOT / "data" / "gate6" / "gate6-report.json"
SNAPSHOT_ROOT = ROOT / "data" / "gate6" / "snapshots"

EXPECTED_UNIT_ACCEPTANCE_HASHES = {
    "h4_holdout_000": "148e8c22c9d1ad55ca99d81cfb7fe2d521b5acb375640f64289b430322a80a48",
    "h4_holdout_001": "d9d08ed10acea4c4e18275ccecef0fbf4943f6ac24e126f2e260aa138ab06fa6",
    "h4_holdout_002": "cf7de52475eb5a7f84eb9292630a307e97d42d9df16e2ed94cb7ec1973a34ee4",
    "h4_holdout_003": "222526f9ec05be5d4c50cf00ff6307dd2cbfe88c0e09e243bc090aaddc71f2c3",
    "h4_holdout_004": "d6618728e6d467b4893ae9797b0d00fb35d9bfaa7794e67185215dfee7e3f1f9",
}


def _dependencies():
    population = load_h4_holdout_population(POPULATION_PATH)
    report = load_gate6_calibration_report(GATE6_REPORT_PATH)
    store = CalibrationStrengthSnapshotStore(SNAPSHOT_ROOT)
    reference = policy_strength_reference_from_store(store)
    validation = policy_validation_strengths_from_store(store)
    return population, report, reference, validation


@pytest.fixture(scope="module")
def official_acceptance():
    population, report, reference, validation = _dependencies()
    return (
        population,
        report,
        reference,
        validation,
        evaluate_h4_holdout_acceptance(
            population,
            reference,
            validation,
            gate6_report_hash=report.report_hash,
        ),
    )


def _rehash_acceptance(document: dict[str, object]) -> dict[str, object]:
    acceptance = document["acceptance"]
    canonical = json.dumps(
        acceptance,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    result = copy.deepcopy(document)
    result["acceptance_hash"] = hashlib.sha256(canonical).hexdigest()
    return result


def test_generic_parameter_acceptance_supports_holdout_identity_and_all_six_metrics() -> None:
    population, _, reference, validation = _dependencies()
    unit = population.units[0]

    result = evaluate_parameter_acceptance(
        profile_name=unit.unit_id,
        profile_version=population.population_version,
        parameters=unit.parameters,
        reference=reference,
        validation_strengths=validation,
    )

    assert result.profile_name == unit.unit_id
    assert result.profile_version == population.population_version
    assert result.is_calibrated
    assert all(metric.required for metric in result.calibration_result.metrics)
    assert {metric.metric for metric in result.calibration_result.metrics} == set(CalibrationMetric)


def test_official_holdout_acceptance_passes_all_five_units_with_frozen_hashes(official_acceptance) -> None:
    population, report, reference, validation, acceptance = official_acceptance

    assert acceptance.acceptance_version == H4_HOLDOUT_ACCEPTANCE_VERSION
    assert acceptance.population_hash == H4_HOLDOUT_POPULATION_HASH
    assert acceptance.complete
    assert acceptance.passed
    assert len(acceptance.results) == 5
    assert acceptance.unit_acceptance_hashes == EXPECTED_UNIT_ACCEPTANCE_HASHES
    assert all(result.is_calibrated for result in acceptance.results)
    assert all(
        all(metric.required and metric.passed is True for metric in result.calibration_result.metrics)
        for result in acceptance.results
    )


def test_acceptance_binds_exact_gate6_reference_validation_and_calibration_contracts(official_acceptance) -> None:
    population, report, reference, validation, acceptance = official_acceptance

    assert acceptance.gate6_report_hash == "ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a"
    assert acceptance.reference_strength_hash == "d2552b7ffa6aca5e7712b2ddef0dce8805f911cad5eb4c0e6b39fb96a06a6143"
    assert acceptance.validation_strength_hash == "d7ed741ec44627e49d034d87de36f636a1463aa7a6804656c594189bf25c54e2"
    assert acceptance.calibration_version == "policy-calibration-v1"
    assert acceptance.calibration_environment_version == "policy-calibration-environment-v1"
    assert acceptance.opportunity_definition_version == "calibration-opportunities-v2"
    assert acceptance.tolerance_version == "calibration-tolerances-v1"
    assert acceptance.required_metric_ids == (
        "vpip",
        "pfr",
        "three_bet_freq",
        "cbet_freq",
        "aggression",
        "fold_to_cbet",
    )


def test_frozen_acceptance_artifact_is_canonical_complete_and_semantically_recomputed() -> None:
    population, report, reference, validation = _dependencies()
    acceptance = load_h4_holdout_acceptance(
        ACCEPTANCE_PATH,
        population=population,
        reference=reference,
        validation_strengths=validation,
        gate6_report_hash=report.report_hash,
    )

    raw = ACCEPTANCE_PATH.read_bytes()
    envelope = json.loads(raw)
    assert envelope["file_version"] == H4_HOLDOUT_ACCEPTANCE_FILE_VERSION
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    assert acceptance.complete and acceptance.passed
    assert envelope["acceptance_hash"] == acceptance.acceptance_hash
    assert envelope["acceptance"]["unit_acceptance_hashes"] == [
        {"unit_id": unit_id, "acceptance_hash": acceptance_hash}
        for unit_id, acceptance_hash in EXPECTED_UNIT_ACCEPTANCE_HASHES.items()
    ]


def test_acceptance_loader_rejects_semantic_tamper_even_when_outer_hash_is_recomputed() -> None:
    population, report, reference, validation = _dependencies()
    raw = json.loads(ACCEPTANCE_PATH.read_bytes())
    acceptance_payload = raw["acceptance"]
    assert isinstance(acceptance_payload, dict)
    results = acceptance_payload["results"]
    assert isinstance(results, list)
    first = results[0]
    assert isinstance(first, dict)
    first["gate_map_hash"] = "0" * 64

    with pytest.raises(H4HoldoutAcceptanceError, match="unit acceptance hash"):
        validate_h4_holdout_acceptance_document(
            _rehash_acceptance(raw),
            population=population,
            reference=reference,
            validation_strengths=validation,
            gate6_report_hash=report.report_hash,
        )


def test_any_failed_unit_keeps_all_five_units_and_fails_closed_without_replacement(official_acceptance) -> None:
    population, report, reference, validation, acceptance = official_acceptance
    results = list(acceptance.results)
    failed = results[2]
    metrics = list(failed.calibration_result.metrics)
    metrics[0] = replace(metrics[0], passed=False)
    failed_calibration = replace(
        failed.calibration_result,
        metrics=tuple(metrics),
        is_calibrated=False,
    )
    results[2] = replace(failed, calibration_result=failed_calibration)
    failed_acceptance = replace(acceptance, results=tuple(results))

    assert len(failed_acceptance.results) == 5
    assert tuple(result.profile_name for result in failed_acceptance.results) == tuple(
        unit.unit_id for unit in population.units
    )
    assert not failed_acceptance.passed
    with pytest.raises(H4HoldoutAcceptanceNotClosedError, match="all five"):
        require_h4_holdout_acceptance_closed(failed_acceptance)
