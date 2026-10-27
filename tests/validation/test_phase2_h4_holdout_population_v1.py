from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pytest

from pokerbench.agents.observation import AgentObservation
from pokerbench.experiments.h4_holdout_population import (
    H4_HOLDOUT_GENERATION_ALGORITHM,
    H4_HOLDOUT_HERO_FORBIDDEN_INPUTS,
    H4_HOLDOUT_POPULATION_FILE_VERSION,
    H4_HOLDOUT_POPULATION_HASH,
    H4_HOLDOUT_POPULATION_ID,
    H4_HOLDOUT_POPULATION_KIND,
    H4_HOLDOUT_POPULATION_ROLE,
    H4_HOLDOUT_POPULATION_VERSION,
    H4_HOLDOUT_SEED_MATERIAL,
    H4HoldoutPopulationError,
    canonical_file_bytes,
    derive_h4_holdout_seed_material,
    generate_h4_holdout_units,
    h4_holdout_dimension_permutations,
    load_h4_holdout_population,
    validate_h4_holdout_population_document,
)

ROOT = Path(__file__).resolve().parents[2]
POPULATION_PATH = ROOT / "configs" / "populations" / "phase2-h4-holdout-population-v1.json"


def _raw() -> dict[str, object]:
    return json.loads(POPULATION_PATH.read_bytes())


def _rehash(document: dict[str, object]) -> dict[str, object]:
    payload = document["population"]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    result = copy.deepcopy(document)
    result["population_hash"] = hashlib.sha256(canonical).hexdigest()
    return result


def test_machine_readable_population_has_exact_identity_hash_and_canonical_bytes() -> None:
    population = load_h4_holdout_population(POPULATION_PATH)

    assert population.population_id == H4_HOLDOUT_POPULATION_ID
    assert population.population_version == H4_HOLDOUT_POPULATION_VERSION
    assert population.population_role == H4_HOLDOUT_POPULATION_ROLE == "holdout_test"
    assert population.population_kind == H4_HOLDOUT_POPULATION_KIND
    assert population.population_hash == H4_HOLDOUT_POPULATION_HASH
    assert population.population_hash == "b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3"

    raw = POPULATION_PATH.read_bytes()
    assert b"\r" not in raw
    assert raw == canonical_file_bytes(json.loads(raw))
    assert _raw()["file_version"] == H4_HOLDOUT_POPULATION_FILE_VERSION


def test_generation_seed_and_latin_hypercube_permutations_are_exact() -> None:
    assert derive_h4_holdout_seed_material() == H4_HOLDOUT_SEED_MATERIAL
    assert H4_HOLDOUT_SEED_MATERIAL == "a1f17ebdb8504b9c5b59e70439fbe4eaf4a6de559b6436e996f8a0c1c9647d75"
    assert H4_HOLDOUT_GENERATION_ALGORITHM == "sha256-latin-hypercube-fixedpoint-v1"
    assert h4_holdout_dimension_permutations() == {
        "vpip": (2, 1, 4, 3, 0),
        "pfr_ratio": (4, 3, 2, 1, 0),
        "three_bet_freq": (1, 0, 2, 4, 3),
        "cbet_freq": (4, 3, 1, 0, 2),
        "aggression": (4, 3, 1, 2, 0),
        "fold_to_cbet": (3, 2, 4, 0, 1),
    }


def test_generated_units_match_exact_frozen_vectors_and_equal_weights() -> None:
    units = generate_h4_holdout_units()

    assert tuple(unit.unit_id for unit in units) == tuple(
        f"h4_holdout_{index:03d}" for index in range(5)
    )
    assert tuple(unit.weight for unit in units) == (0.2,) * 5
    assert sum(unit.weight for unit in units) == pytest.approx(1.0)
    assert tuple(unit.parameters_tuple for unit in units) == (
        (0.363630, 0.291240, 0.051497, 0.746223, 1.515065, 0.406611),
        (0.184105, 0.113241, 0.018270, 0.690724, 1.426503, 0.383963),
        (0.583082, 0.227593, 0.117397, 0.447809, 0.670847, 0.592238),
        (0.460487, 0.111152, 0.171613, 0.343563, 0.885767, 0.131947),
        (0.128226, 0.007052, 0.159610, 0.531827, 0.417032, 0.200850),
    )


def test_generation_support_and_pfr_construction_are_semantically_enforced() -> None:
    raw = _raw()
    payload = raw["population"]
    assert isinstance(payload, dict)
    generation = payload["generation"]
    assert isinstance(generation, dict)

    assert generation["support"] == {
        "aggression": [0.2, 1.8],
        "cbet_freq": [0.3, 0.85],
        "fold_to_cbet": [0.1, 0.6],
        "pfr_ratio": [0.05, 0.9],
        "three_bet_freq": [0.01, 0.2],
        "vpip": [0.08, 0.6],
    }
    assert generation["quantization_decimals"] == 6
    assert generation["pfr_construction"] == "pfr=round_half_even(vpip*pfr_ratio,6)"

    generated = generate_h4_holdout_units()
    for unit in generated:
        assert 0.0 <= unit.parameters.pfr <= unit.parameters.vpip <= 1.0
        assert unit.parameters.aggression is not None


def test_population_loader_rejects_semantic_tamper_even_if_outer_hash_is_recomputed() -> None:
    raw = _raw()
    payload = raw["population"]
    assert isinstance(payload, dict)
    units = payload["units"]
    assert isinstance(units, list)
    first = units[0]
    assert isinstance(first, dict)
    parameters = first["parameters"]
    assert isinstance(parameters, dict)
    parameters["vpip"] = 0.363631

    tampered = _rehash(raw)
    with pytest.raises(H4HoldoutPopulationError, match="frozen Spec-36 population"):
        validate_h4_holdout_population_document(tampered)


def test_population_loader_rejects_generation_rule_or_dependency_drift() -> None:
    for key, value in (
        ("parametric_policy_version", "parametric-policy-v999"),
        ("gate6_report_hash", "0" * 64),
    ):
        raw = _raw()
        payload = raw["population"]
        assert isinstance(payload, dict)
        payload[key] = value
        with pytest.raises(H4HoldoutPopulationError):
            validate_h4_holdout_population_document(_rehash(raw))


def test_holdout_information_is_absent_from_hero_observation_contract() -> None:
    observation_fields = {field.name for field in fields(AgentObservation)}

    assert set(H4_HOLDOUT_HERO_FORBIDDEN_INPUTS).isdisjoint(observation_fields)
    assert H4_HOLDOUT_HERO_FORBIDDEN_INPUTS == (
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
