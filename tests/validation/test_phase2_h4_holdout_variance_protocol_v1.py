from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokerbench.experiments.h4_holdout_variance_protocol import (
    H4_HOLDOUT_ACCEPTANCE_HASH, H4_HOLDOUT_POPULATION_HASH, H4_HOLDOUT_VARIANCE_PROTOCOL_HASH,
    H4_VARIANCE_MASTER_SEEDS, canonical_file_bytes, iter_h4_holdout_variance_jobs,
    iter_h4_holdout_variance_pairs, load_h4_holdout_variance_protocol, validate_h4_holdout_variance_document,
    H4HoldoutVarianceProtocolError,
)

CONFIG = Path("configs/experiments/phase2-h4-holdout-variance-pilot-v1.json")

def test_frozen_protocol_is_canonical_and_hash_bound() -> None:
    protocol = load_h4_holdout_variance_protocol(CONFIG)
    raw = CONFIG.read_bytes(); document = json.loads(raw)
    assert raw == canonical_file_bytes(document)
    assert protocol.payload_hash == H4_HOLDOUT_VARIANCE_PROTOCOL_HASH
    assert protocol.status == "design_frozen_execution_not_yet_authorized"

def test_protocol_binds_exact_holdout_and_acceptance() -> None:
    document = json.loads(CONFIG.read_bytes())
    holdout = document["payload"]["bindings"]["holdout_population"]
    assert holdout["population_hash"] == H4_HOLDOUT_POPULATION_HASH
    assert holdout["holdout_acceptance_hash"] == H4_HOLDOUT_ACCEPTANCE_HASH
    assert [item["unit_id"] for item in document["payload"]["matrix"]["holdout_units"]] == [f"h4_holdout_{i:03d}" for i in range(5)]
    assert [item["weight"] for item in document["payload"]["matrix"]["holdout_units"]] == [0.2] * 5

def test_matrix_and_canonical_expansion_are_exact() -> None:
    protocol = load_h4_holdout_variance_protocol(CONFIG)
    jobs = iter_h4_holdout_variance_jobs(protocol); pairs = iter_h4_holdout_variance_pairs(protocol)
    assert protocol.master_seeds == H4_VARIANCE_MASTER_SEEDS == tuple(range(6_400_000, 6_400_010))
    assert len(jobs) == 100 and len(pairs) == 50
    assert sum(job.simulated_engine_hands for job in jobs) == 100_000
    assert [(j.holdout_unit_id, j.master_seed, j.arm_id) for j in jobs[:4]] == [("h4_holdout_000",6400000,"control"),("h4_holdout_000",6400000,"treatment"),("h4_holdout_000",6400001,"control"),("h4_holdout_000",6400001,"treatment")]

def test_protocol_freezes_trajectory_rng_and_reset_semantics() -> None:
    payload = json.loads(CONFIG.read_bytes())["payload"]
    assert payload["trajectory"]["hands_per_position"] == 500
    assert payload["trajectory"]["position_run_count"] == 2
    assert payload["trajectory"]["adaptive_reset_before_each_position_run"] is True
    assert payload["rng"]["deal_relation"] == "identical_defined_card_realisations"
    assert payload["rng"]["full_action_path_identity"] == "not_claimed"

def test_protocol_is_variance_only_and_all_or_nothing() -> None:
    payload = json.loads(CONFIG.read_bytes())["payload"]
    assert payload["scientific_use"]["confirmatory_h4_inference_permitted"] is False
    assert payload["scientific_use"]["variance_pilot_mean_design_use"] == "prohibited"
    assert payload["analysis"]["required_valid_population_clusters"] == 10
    assert payload["analysis"]["reduced_complete_case_planning"] is False
    assert payload["matrix"]["replacement_seeds"] == "forbidden"

def test_seed_domains_are_disjoint_from_prior_experiments() -> None:
    seeds = set(H4_VARIANCE_MASTER_SEEDS)
    assert seeds.isdisjoint(range(6_100_000, 6_100_100))
    assert seeds.isdisjoint(range(6_200_000, 6_202_769))
    assert seeds.isdisjoint(range(6_300_000, 6_300_005))

def test_semantic_tampering_fails_even_with_recomputed_outer_hash() -> None:
    document = json.loads(CONFIG.read_bytes())
    document["payload"]["matrix"]["planned_engine_hands"] = 99_999
    import hashlib
    encoded = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    document["payload_hash"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(H4HoldoutVarianceProtocolError):
        validate_h4_holdout_variance_document(document)
