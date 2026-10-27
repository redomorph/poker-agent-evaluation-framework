"""Frozen Phase-2 H4 holdout population from Spec 36."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Mapping

from pokerbench.agents.profiles import OpponentParameters, PARAMETER_SCHEMA_VERSION
from pokerbench.agents.strength_gates import PARAMETRIC_POLICY_VERSION
from pokerbench.agents.calibration_environment import CALIBRATION_ENVIRONMENT_VERSION
from pokerbench.evaluation.calibration import (
    CALIBRATION_VERSION,
    DEFAULT_CALIBRATION_TOLERANCES,
    OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
)


H4_HOLDOUT_POPULATION_FILE_VERSION = "phase2-h4-holdout-population-file-v1"
H4_HOLDOUT_POPULATION_ID = "phase2-h4-holdout-population-v1"
H4_HOLDOUT_POPULATION_VERSION = "phase2-h4-holdout-parametric-population-v1"
H4_HOLDOUT_POPULATION_ROLE = "holdout_test"
H4_HOLDOUT_POPULATION_KIND = "fixed-finite-equal-weight-parametric-v1"
H4_HOLDOUT_GENERATION_ALGORITHM = "sha256-latin-hypercube-fixedpoint-v1"
H4_HOLDOUT_POPULATION_HASH = "b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3"
H4_HOLDOUT_GATE6_REPORT_HASH = "ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a"
H4_HOLDOUT_REFERENCE_STRENGTH_HASH = "d2552b7ffa6aca5e7712b2ddef0dce8805f911cad5eb4c0e6b39fb96a06a6143"
H4_HOLDOUT_VALIDATION_STRENGTH_HASH = "d7ed741ec44627e49d034d87de36f636a1463aa7a6804656c594189bf25c54e2"
H4_HOLDOUT_NUM_UNITS = 5
H4_HOLDOUT_UNIT_WEIGHT = 0.2
H4_HOLDOUT_QUANTIZATION_DECIMALS = 6
_H4_QUANTUM = Decimal("0.000001")
_H4_SEED_SUFFIX = ":sealed-generation-v1"

H4_HOLDOUT_SUPPORT: Mapping[str, tuple[Decimal, Decimal]] = {
    "vpip": (Decimal("0.08"), Decimal("0.60")),
    "pfr_ratio": (Decimal("0.05"), Decimal("0.90")),
    "three_bet_freq": (Decimal("0.01"), Decimal("0.20")),
    "cbet_freq": (Decimal("0.30"), Decimal("0.85")),
    "aggression": (Decimal("0.20"), Decimal("1.80")),
    "fold_to_cbet": (Decimal("0.10"), Decimal("0.60")),
}

H4_HOLDOUT_HERO_FORBIDDEN_INPUTS = (
    "holdout_unit_id",
    "population_id",
    "population_weight",
    "true_opponent_parameter_vector",
    "generation_seed_material",
    "latent_pfr_ratio",
    "calibration_targets",
    "calibration_acceptance_result",
    "controller_side_classification_labels",
)


class H4HoldoutPopulationError(ValueError):
    """Raised when a holdout population artifact violates frozen Spec 36."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_file_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON bytes with one trailing LF."""
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_uint256(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def derive_h4_holdout_seed_material() -> str:
    return hashlib.sha256(
        f"{H4_HOLDOUT_POPULATION_ID}{_H4_SEED_SUFFIX}".encode("utf-8")
    ).hexdigest()


H4_HOLDOUT_SEED_MATERIAL = derive_h4_holdout_seed_material()


def _dimension_permutation(dimension: str) -> tuple[int, ...]:
    if dimension not in H4_HOLDOUT_SUPPORT:
        raise ValueError(f"Unsupported H4 holdout dimension: {dimension}")
    return tuple(
        sorted(
            range(H4_HOLDOUT_NUM_UNITS),
            key=lambda stratum: (
                _sha256_uint256(
                    f"{H4_HOLDOUT_POPULATION_ID}:{H4_HOLDOUT_SEED_MATERIAL}:"
                    f"{dimension}:perm:{stratum}"
                ),
                stratum,
            ),
        )
    )


def h4_holdout_dimension_permutations() -> dict[str, tuple[int, ...]]:
    return {
        dimension: _dimension_permutation(dimension)
        for dimension in H4_HOLDOUT_SUPPORT
    }


def _jitter(dimension: str, unit_index: int) -> Decimal:
    numerator = _sha256_uint256(
        f"{H4_HOLDOUT_POPULATION_ID}:{H4_HOLDOUT_SEED_MATERIAL}:"
        f"{dimension}:jitter:{unit_index}"
    )
    with localcontext() as context:
        context.prec = 100
        return Decimal(numerator) / Decimal(2**256)


def _generated_latent(dimension: str, unit_index: int) -> Decimal:
    lower, upper = H4_HOLDOUT_SUPPORT[dimension]
    stratum = _dimension_permutation(dimension)[unit_index]
    with localcontext() as context:
        context.prec = 100
        unit_interval = (
            Decimal(stratum) + _jitter(dimension, unit_index)
        ) / Decimal(H4_HOLDOUT_NUM_UNITS)
        value = lower + (upper - lower) * unit_interval
        return value.quantize(_H4_QUANTUM, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class H4HoldoutUnit:
    unit_id: str
    weight: float
    parameters: OpponentParameters

    @property
    def parameters_tuple(self) -> tuple[float, float, float, float, float, float]:
        assert self.parameters.three_bet_freq is not None
        assert self.parameters.cbet_freq is not None
        assert self.parameters.aggression is not None
        assert self.parameters.fold_to_cbet is not None
        return (
            self.parameters.vpip,
            self.parameters.pfr,
            self.parameters.three_bet_freq,
            self.parameters.cbet_freq,
            self.parameters.aggression,
            self.parameters.fold_to_cbet,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "parameters": {
                "aggression": self.parameters.aggression,
                "cbet_freq": self.parameters.cbet_freq,
                "fold_to_cbet": self.parameters.fold_to_cbet,
                "pfr": self.parameters.pfr,
                "three_bet_freq": self.parameters.three_bet_freq,
                "vpip": self.parameters.vpip,
            },
            "unit_id": self.unit_id,
            "weight": self.weight,
        }


def generate_h4_holdout_units() -> tuple[H4HoldoutUnit, ...]:
    units: list[H4HoldoutUnit] = []
    for unit_index in range(H4_HOLDOUT_NUM_UNITS):
        vpip = _generated_latent("vpip", unit_index)
        pfr_ratio = _generated_latent("pfr_ratio", unit_index)
        pfr = (vpip * pfr_ratio).quantize(
            _H4_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
        parameters = OpponentParameters(
            vpip=float(vpip),
            pfr=float(pfr),
            three_bet_freq=float(
                _generated_latent("three_bet_freq", unit_index)
            ),
            cbet_freq=float(_generated_latent("cbet_freq", unit_index)),
            aggression=float(_generated_latent("aggression", unit_index)),
            fold_to_cbet=float(
                _generated_latent("fold_to_cbet", unit_index)
            ),
        )
        units.append(
            H4HoldoutUnit(
                unit_id=f"h4_holdout_{unit_index:03d}",
                weight=H4_HOLDOUT_UNIT_WEIGHT,
                parameters=parameters,
            )
        )
    return tuple(units)


def _support_payload() -> dict[str, list[float]]:
    return {
        dimension: [float(lower), float(upper)]
        for dimension, (lower, upper) in H4_HOLDOUT_SUPPORT.items()
    }


def expected_h4_holdout_population_payload() -> dict[str, Any]:
    return {
        "calibration_environment_version": CALIBRATION_ENVIRONMENT_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "gate6_report_hash": H4_HOLDOUT_GATE6_REPORT_HASH,
        "generation": {
            "algorithm": H4_HOLDOUT_GENERATION_ALGORITHM,
            "num_units": H4_HOLDOUT_NUM_UNITS,
            "pfr_construction": "pfr=round_half_even(vpip*pfr_ratio,6)",
            "quantization_decimals": H4_HOLDOUT_QUANTIZATION_DECIMALS,
            "seed_material": H4_HOLDOUT_SEED_MATERIAL,
            "support": _support_payload(),
        },
        "opportunity_definition_version": OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
        "parameter_schema_version": PARAMETER_SCHEMA_VERSION,
        "parametric_policy_version": PARAMETRIC_POLICY_VERSION,
        "population_id": H4_HOLDOUT_POPULATION_ID,
        "population_kind": H4_HOLDOUT_POPULATION_KIND,
        "population_role": H4_HOLDOUT_POPULATION_ROLE,
        "population_version": H4_HOLDOUT_POPULATION_VERSION,
        "reference_strength_hash": H4_HOLDOUT_REFERENCE_STRENGTH_HASH,
        "tolerance_version": DEFAULT_CALIBRATION_TOLERANCES.version,
        "units": [unit.payload() for unit in generate_h4_holdout_units()],
        "validation_strength_hash": H4_HOLDOUT_VALIDATION_STRENGTH_HASH,
    }


@dataclass(frozen=True)
class H4HoldoutPopulation:
    population_id: str
    population_version: str
    population_role: str
    population_kind: str
    population_hash: str
    units: tuple[H4HoldoutUnit, ...]

    def __post_init__(self) -> None:
        if self.population_id != H4_HOLDOUT_POPULATION_ID:
            raise H4HoldoutPopulationError("Unexpected holdout population_id")
        if self.population_version != H4_HOLDOUT_POPULATION_VERSION:
            raise H4HoldoutPopulationError("Unexpected holdout population_version")
        if self.population_role != H4_HOLDOUT_POPULATION_ROLE:
            raise H4HoldoutPopulationError("Unexpected holdout population_role")
        if self.population_kind != H4_HOLDOUT_POPULATION_KIND:
            raise H4HoldoutPopulationError("Unexpected holdout population_kind")
        if self.population_hash != H4_HOLDOUT_POPULATION_HASH:
            raise H4HoldoutPopulationError("Unexpected holdout population_hash")
        if self.units != generate_h4_holdout_units():
            raise H4HoldoutPopulationError("Holdout units differ from frozen generation")


def validate_h4_holdout_population_document(
    document: Any,
) -> H4HoldoutPopulation:
    if not isinstance(document, dict):
        raise H4HoldoutPopulationError("Holdout population file must be an object")
    if set(document) != {"file_version", "population", "population_hash"}:
        raise H4HoldoutPopulationError("Holdout population file has unexpected fields")
    if document["file_version"] != H4_HOLDOUT_POPULATION_FILE_VERSION:
        raise H4HoldoutPopulationError("Unsupported holdout population file_version")
    payload = document["population"]
    if not isinstance(payload, dict):
        raise H4HoldoutPopulationError("population must be an object")
    stored_hash = document["population_hash"]
    if not isinstance(stored_hash, str):
        raise H4HoldoutPopulationError("population_hash must be a string")
    if _sha256_canonical(payload) != stored_hash:
        raise H4HoldoutPopulationError("population_hash does not match payload")

    expected = expected_h4_holdout_population_payload()
    if payload != expected:
        raise H4HoldoutPopulationError(
            "population payload differs from frozen Spec-36 population"
        )
    if stored_hash != H4_HOLDOUT_POPULATION_HASH:
        raise H4HoldoutPopulationError("population_hash differs from frozen Spec-36 hash")

    return H4HoldoutPopulation(
        population_id=H4_HOLDOUT_POPULATION_ID,
        population_version=H4_HOLDOUT_POPULATION_VERSION,
        population_role=H4_HOLDOUT_POPULATION_ROLE,
        population_kind=H4_HOLDOUT_POPULATION_KIND,
        population_hash=H4_HOLDOUT_POPULATION_HASH,
        units=generate_h4_holdout_units(),
    )


def load_h4_holdout_population(path: str | Path) -> H4HoldoutPopulation:
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        document = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise H4HoldoutPopulationError("Invalid H4 holdout population JSON") from exc
    if raw_bytes != canonical_file_bytes(document):
        raise H4HoldoutPopulationError("H4 holdout population file is not canonical JSON")
    return validate_h4_holdout_population_document(document)
