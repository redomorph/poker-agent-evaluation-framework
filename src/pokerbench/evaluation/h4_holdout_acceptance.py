"""Official all-six-metric acceptance gate for the Spec-36 H4 holdout."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokerbench.agents.calibration_environment import CALIBRATION_ENVIRONMENT_VERSION
from pokerbench.agents.strength_gates import PolicyStrengthReference
from pokerbench.evaluation.calibration import (
    CALIBRATION_VERSION,
    DEFAULT_CALIBRATION_TOLERANCES,
    OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
    CalibrationMetric,
)
from pokerbench.evaluation.gate6_execution import (
    Gate6CalibrationReportIntegrityError,
    parse_policy_calibration_acceptance_payload,
)
from pokerbench.evaluation.policy_acceptance import (
    PolicyCalibrationAcceptance,
    PolicyValidationStrengths,
    evaluate_parameter_acceptance,
)
from pokerbench.experiments.h4_holdout_population import (
    H4_HOLDOUT_GATE6_REPORT_HASH,
    H4_HOLDOUT_POPULATION_HASH,
    H4_HOLDOUT_POPULATION_ID,
    H4_HOLDOUT_POPULATION_ROLE,
    H4_HOLDOUT_POPULATION_VERSION,
    H4_HOLDOUT_REFERENCE_STRENGTH_HASH,
    H4_HOLDOUT_VALIDATION_STRENGTH_HASH,
    H4HoldoutPopulation,
    H4HoldoutUnit,
)


H4_HOLDOUT_ACCEPTANCE_FILE_VERSION = "phase2-h4-holdout-acceptance-file-v1"
H4_HOLDOUT_ACCEPTANCE_VERSION = "phase2-h4-holdout-acceptance-v1"
H4_HOLDOUT_ACCEPTANCE_HASH = "9ba347fb1d615cec1d6458944df3c293dcac630564e8847a407e987bbb98af84"


class H4HoldoutAcceptanceError(ValueError):
    """Raised when the holdout acceptance artifact is invalid."""


class H4HoldoutAcceptanceNotClosedError(H4HoldoutAcceptanceError):
    """Raised when not all five frozen holdout units pass acceptance."""


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


def _expected_target(unit: H4HoldoutUnit, metric: CalibrationMetric) -> float:
    mapping = {
        CalibrationMetric.VPIP: unit.parameters.vpip,
        CalibrationMetric.PFR: unit.parameters.pfr,
        CalibrationMetric.THREE_BET: unit.parameters.three_bet_freq,
        CalibrationMetric.CBET: unit.parameters.cbet_freq,
        CalibrationMetric.AGGRESSION: unit.parameters.aggression,
        CalibrationMetric.FOLD_TO_CBET: unit.parameters.fold_to_cbet,
    }
    value = mapping[metric]
    if value is None:
        raise H4HoldoutAcceptanceError(
            f"Spec-36 holdout metric {metric.value} must not be don't-care"
        )
    return float(value)


@dataclass(frozen=True)
class H4HoldoutAcceptance:
    population_hash: str
    gate6_report_hash: str
    reference_strength_hash: str
    validation_strength_hash: str
    results: tuple[PolicyCalibrationAcceptance, ...]
    acceptance_version: str = H4_HOLDOUT_ACCEPTANCE_VERSION
    population_id: str = H4_HOLDOUT_POPULATION_ID
    population_version: str = H4_HOLDOUT_POPULATION_VERSION
    population_role: str = H4_HOLDOUT_POPULATION_ROLE
    calibration_version: str = CALIBRATION_VERSION
    calibration_environment_version: str = CALIBRATION_ENVIRONMENT_VERSION
    opportunity_definition_version: str = OPPORTUNITY_DEFINITION_CONTRACT_VERSION
    tolerance_version: str = DEFAULT_CALIBRATION_TOLERANCES.version

    def __post_init__(self) -> None:
        if self.acceptance_version != H4_HOLDOUT_ACCEPTANCE_VERSION:
            raise H4HoldoutAcceptanceError("Unsupported holdout acceptance version")
        if self.population_id != H4_HOLDOUT_POPULATION_ID:
            raise H4HoldoutAcceptanceError("Unexpected holdout population_id")
        if self.population_version != H4_HOLDOUT_POPULATION_VERSION:
            raise H4HoldoutAcceptanceError("Unexpected holdout population_version")
        if self.population_role != H4_HOLDOUT_POPULATION_ROLE:
            raise H4HoldoutAcceptanceError("Unexpected holdout population_role")
        if self.population_hash != H4_HOLDOUT_POPULATION_HASH:
            raise H4HoldoutAcceptanceError("Unexpected holdout population_hash")
        if self.gate6_report_hash != H4_HOLDOUT_GATE6_REPORT_HASH:
            raise H4HoldoutAcceptanceError("Unexpected Gate-6 report hash")
        if self.reference_strength_hash != H4_HOLDOUT_REFERENCE_STRENGTH_HASH:
            raise H4HoldoutAcceptanceError("Unexpected reference-strength hash")
        if self.validation_strength_hash != H4_HOLDOUT_VALIDATION_STRENGTH_HASH:
            raise H4HoldoutAcceptanceError("Unexpected validation-strength hash")
        if self.calibration_version != CALIBRATION_VERSION:
            raise H4HoldoutAcceptanceError("Unexpected calibration version")
        if self.calibration_environment_version != CALIBRATION_ENVIRONMENT_VERSION:
            raise H4HoldoutAcceptanceError("Unexpected calibration-environment version")
        if self.opportunity_definition_version != OPPORTUNITY_DEFINITION_CONTRACT_VERSION:
            raise H4HoldoutAcceptanceError("Unexpected opportunity-definition version")
        if self.tolerance_version != DEFAULT_CALIBRATION_TOLERANCES.version:
            raise H4HoldoutAcceptanceError("Unexpected calibration tolerance version")
        if not isinstance(self.results, tuple):
            raise TypeError("results must be a tuple")

        expected_ids = tuple(f"h4_holdout_{index:03d}" for index in range(5))
        actual_ids = tuple(result.profile_name for result in self.results)
        if actual_ids != expected_ids:
            raise H4HoldoutAcceptanceError(
                "Holdout acceptance must contain exactly all five frozen units in order"
            )

    @property
    def required_metric_ids(self) -> tuple[str, ...]:
        return tuple(metric.value for metric in CalibrationMetric)

    @property
    def complete(self) -> bool:
        return len(self.results) == 5

    @property
    def passed(self) -> bool:
        return self.complete and all(result.is_calibrated for result in self.results)

    @property
    def unit_acceptance_hashes(self) -> dict[str, str]:
        return {
            result.profile_name: result.acceptance_hash
            for result in self.results
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_version": self.acceptance_version,
            "population_id": self.population_id,
            "population_version": self.population_version,
            "population_role": self.population_role,
            "population_hash": self.population_hash,
            "gate6_report_hash": self.gate6_report_hash,
            "reference_strength_hash": self.reference_strength_hash,
            "validation_strength_hash": self.validation_strength_hash,
            "calibration_version": self.calibration_version,
            "calibration_environment_version": self.calibration_environment_version,
            "opportunity_definition_version": self.opportunity_definition_version,
            "tolerance_version": self.tolerance_version,
            "required_metric_ids": list(self.required_metric_ids),
            "complete": self.complete,
            "passed": self.passed,
            "unit_acceptance_hashes": [
                {
                    "unit_id": result.profile_name,
                    "acceptance_hash": result.acceptance_hash,
                }
                for result in self.results
            ],
            "results": [result.to_dict() for result in self.results],
        }

    @property
    def acceptance_hash(self) -> str:
        return _sha256_canonical(self.to_dict())


def _validate_result_against_unit(
    result: PolicyCalibrationAcceptance,
    unit: H4HoldoutUnit,
    *,
    population_version: str,
) -> None:
    if result.profile_name != unit.unit_id:
        raise H4HoldoutAcceptanceError("Holdout acceptance unit identity mismatch")
    if result.profile_version != population_version:
        raise H4HoldoutAcceptanceError("Holdout acceptance unit version mismatch")
    if result.reference_strength_hash != H4_HOLDOUT_REFERENCE_STRENGTH_HASH:
        raise H4HoldoutAcceptanceError("Holdout acceptance reference hash mismatch")
    if result.validation_strength_hash != H4_HOLDOUT_VALIDATION_STRENGTH_HASH:
        raise H4HoldoutAcceptanceError("Holdout acceptance validation hash mismatch")

    metrics = result.calibration_result.metrics
    if tuple(metric.metric for metric in metrics) != tuple(CalibrationMetric):
        raise H4HoldoutAcceptanceError("Holdout acceptance metric order mismatch")
    for metric in metrics:
        if not metric.required:
            raise H4HoldoutAcceptanceError(
                "All six Spec-36 holdout calibration metrics must be required"
            )
        if metric.target != _expected_target(unit, metric.metric):
            raise H4HoldoutAcceptanceError(
                f"Holdout target mismatch for {unit.unit_id}/{metric.metric.value}"
            )


def evaluate_h4_holdout_acceptance(
    population: H4HoldoutPopulation,
    reference: PolicyStrengthReference,
    validation_strengths: PolicyValidationStrengths,
    *,
    gate6_report_hash: str,
) -> H4HoldoutAcceptance:
    """Evaluate every frozen holdout unit; never drop or replace a failure."""
    if not isinstance(population, H4HoldoutPopulation):
        raise TypeError("population must be H4HoldoutPopulation")
    if population.population_hash != H4_HOLDOUT_POPULATION_HASH:
        raise H4HoldoutAcceptanceError("Holdout population hash mismatch")
    if gate6_report_hash != H4_HOLDOUT_GATE6_REPORT_HASH:
        raise H4HoldoutAcceptanceError("Gate-6 report hash mismatch")
    if reference.reference_strength_hash != H4_HOLDOUT_REFERENCE_STRENGTH_HASH:
        raise H4HoldoutAcceptanceError("Reference-strength hash mismatch")
    if validation_strengths.validation_strength_hash != H4_HOLDOUT_VALIDATION_STRENGTH_HASH:
        raise H4HoldoutAcceptanceError("Validation-strength hash mismatch")

    results = tuple(
        evaluate_parameter_acceptance(
            profile_name=unit.unit_id,
            profile_version=population.population_version,
            parameters=unit.parameters,
            reference=reference,
            validation_strengths=validation_strengths,
        )
        for unit in population.units
    )
    for result, unit in zip(results, population.units, strict=True):
        _validate_result_against_unit(
            result,
            unit,
            population_version=population.population_version,
        )

    return H4HoldoutAcceptance(
        population_hash=population.population_hash,
        gate6_report_hash=gate6_report_hash,
        reference_strength_hash=reference.reference_strength_hash,
        validation_strength_hash=validation_strengths.validation_strength_hash,
        results=results,
    )


def require_h4_holdout_acceptance_closed(
    acceptance: H4HoldoutAcceptance,
) -> None:
    if not isinstance(acceptance, H4HoldoutAcceptance):
        raise TypeError("acceptance must be H4HoldoutAcceptance")
    if not acceptance.complete or not acceptance.passed:
        raise H4HoldoutAcceptanceNotClosedError(
            "H4 holdout acceptance requires all five frozen units to pass"
        )


def validate_h4_holdout_acceptance_document(
    document: Any,
    *,
    population: H4HoldoutPopulation,
    reference: PolicyStrengthReference,
    validation_strengths: PolicyValidationStrengths,
    gate6_report_hash: str,
) -> H4HoldoutAcceptance:
    if not isinstance(document, dict):
        raise H4HoldoutAcceptanceError("Holdout acceptance file must be an object")
    if set(document) != {"file_version", "acceptance", "acceptance_hash"}:
        raise H4HoldoutAcceptanceError("Holdout acceptance file has unexpected fields")
    if document["file_version"] != H4_HOLDOUT_ACCEPTANCE_FILE_VERSION:
        raise H4HoldoutAcceptanceError("Unsupported holdout acceptance file_version")
    raw = document["acceptance"]
    stored_hash = document["acceptance_hash"]
    if not isinstance(raw, dict) or not isinstance(stored_hash, str):
        raise H4HoldoutAcceptanceError("Holdout acceptance file structure is invalid")

    expected_keys = {
        "acceptance_version",
        "population_id",
        "population_version",
        "population_role",
        "population_hash",
        "gate6_report_hash",
        "reference_strength_hash",
        "validation_strength_hash",
        "calibration_version",
        "calibration_environment_version",
        "opportunity_definition_version",
        "tolerance_version",
        "required_metric_ids",
        "complete",
        "passed",
        "unit_acceptance_hashes",
        "results",
    }
    if set(raw) != expected_keys:
        raise H4HoldoutAcceptanceError("Holdout acceptance payload has unexpected fields")
    if population.population_hash != H4_HOLDOUT_POPULATION_HASH:
        raise H4HoldoutAcceptanceError("Holdout population hash mismatch")
    if gate6_report_hash != H4_HOLDOUT_GATE6_REPORT_HASH:
        raise H4HoldoutAcceptanceError("Gate-6 report hash mismatch")
    if reference.reference_strength_hash != H4_HOLDOUT_REFERENCE_STRENGTH_HASH:
        raise H4HoldoutAcceptanceError("Reference-strength hash mismatch")
    if validation_strengths.validation_strength_hash != H4_HOLDOUT_VALIDATION_STRENGTH_HASH:
        raise H4HoldoutAcceptanceError("Validation-strength hash mismatch")

    raw_results = raw["results"]
    if not isinstance(raw_results, list):
        raise H4HoldoutAcceptanceError("Holdout acceptance results must be an array")
    try:
        results = tuple(
            parse_policy_calibration_acceptance_payload(item)
            for item in raw_results
        )
    except Gate6CalibrationReportIntegrityError as exc:
        raise H4HoldoutAcceptanceError(
            "Holdout unit acceptance payload is invalid"
        ) from exc

    acceptance = H4HoldoutAcceptance(
        population_hash=raw["population_hash"],
        gate6_report_hash=raw["gate6_report_hash"],
        reference_strength_hash=raw["reference_strength_hash"],
        validation_strength_hash=raw["validation_strength_hash"],
        results=results,
        acceptance_version=raw["acceptance_version"],
        population_id=raw["population_id"],
        population_version=raw["population_version"],
        population_role=raw["population_role"],
        calibration_version=raw["calibration_version"],
        calibration_environment_version=raw["calibration_environment_version"],
        opportunity_definition_version=raw["opportunity_definition_version"],
        tolerance_version=raw["tolerance_version"],
    )
    for result, unit in zip(acceptance.results, population.units, strict=True):
        _validate_result_against_unit(
            result,
            unit,
            population_version=population.population_version,
        )

    expected_unit_hashes = [
        {"unit_id": unit_id, "acceptance_hash": acceptance_hash}
        for unit_id, acceptance_hash in acceptance.unit_acceptance_hashes.items()
    ]
    if raw["unit_acceptance_hashes"] != expected_unit_hashes:
        raise H4HoldoutAcceptanceError("Holdout unit acceptance hash mismatch")
    if raw != acceptance.to_dict():
        raise H4HoldoutAcceptanceError(
            "Holdout acceptance payload is not semantically consistent"
        )
    if _sha256_canonical(raw) != stored_hash:
        raise H4HoldoutAcceptanceError("Holdout acceptance SHA-256 mismatch")
    if stored_hash != H4_HOLDOUT_ACCEPTANCE_HASH:
        raise H4HoldoutAcceptanceError("Unexpected frozen holdout acceptance hash")
    require_h4_holdout_acceptance_closed(acceptance)
    return acceptance


def save_h4_holdout_acceptance(
    acceptance: H4HoldoutAcceptance,
    path: str | Path,
) -> Path:
    require_h4_holdout_acceptance_closed(acceptance)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    document = {
        "file_version": H4_HOLDOUT_ACCEPTANCE_FILE_VERSION,
        "acceptance": acceptance.to_dict(),
        "acceptance_hash": acceptance.acceptance_hash,
    }
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_file_bytes(document))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_h4_holdout_acceptance(
    path: str | Path,
    *,
    population: H4HoldoutPopulation,
    reference: PolicyStrengthReference,
    validation_strengths: PolicyValidationStrengths,
    gate6_report_hash: str,
) -> H4HoldoutAcceptance:
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        document = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise H4HoldoutAcceptanceError("Invalid H4 holdout acceptance JSON") from exc
    if raw_bytes != _canonical_file_bytes(document):
        raise H4HoldoutAcceptanceError("H4 holdout acceptance file is not canonical JSON")
    return validate_h4_holdout_acceptance_document(
        document,
        population=population,
        reference=reference,
        validation_strengths=validation_strengths,
        gate6_report_hash=gate6_report_hash,
    )
