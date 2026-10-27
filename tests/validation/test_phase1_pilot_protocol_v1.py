from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokerbench.agents.hand_strength_baseline import HAND_STRENGTH_BASELINE_VERSION
from pokerbench.agents.profiles import STATIC_PROFILE_VERSION
from pokerbench.agents.random_baseline import RANDOM_BASELINE_VERSION
from pokerbench.agents.strength_gates import PARAMETRIC_POLICY_VERSION
from pokerbench.engine.config import BASELINE_GAME_CONFIG, config_hash
from pokerbench.evaluation.population import (
    BALANCED_STATIC_POPULATION,
    BASELINE_SYNTHETIC_STATIC_POPULATION,
)
from pokerbench.experiments.frozen_protocol import (
    PHASE1_PILOT_PROTOCOL_VERSION,
    PilotProtocolError,
    canonical_file_bytes,
    iter_pilot_jobs,
    load_frozen_population,
    load_phase1_pilot_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = ROOT / "configs" / "experiments" / "phase1-pilot-v1.json"
BASELINE_POPULATION_PATH = (
    ROOT / "configs" / "populations" / "baseline-synthetic-v1.json"
)
BALANCED_POPULATION_PATH = ROOT / "configs" / "populations" / "balanced-v1.json"


def _load_protocol() -> object:
    populations = {
        "baseline-synthetic-static-profiles": load_frozen_population(
            BASELINE_POPULATION_PATH
        ),
        "balanced-static-profiles": load_frozen_population(BALANCED_POPULATION_PATH),
    }
    return load_phase1_pilot_protocol(PILOT_PATH, populations=populations)


def test_frozen_population_files_match_code_definitions() -> None:
    baseline = load_frozen_population(BASELINE_POPULATION_PATH)
    balanced = load_frozen_population(BALANCED_POPULATION_PATH)

    assert baseline == BASELINE_SYNTHETIC_STATIC_POPULATION
    assert balanced == BALANCED_STATIC_POPULATION
    assert baseline.population_hash == BASELINE_SYNTHETIC_STATIC_POPULATION.population_hash
    assert balanced.population_hash == BALANCED_STATIC_POPULATION.population_hash


def test_pilot_protocol_freezes_current_scientific_versions() -> None:
    protocol = _load_protocol()

    assert protocol.protocol_version == PHASE1_PILOT_PROTOCOL_VERSION
    assert protocol.game_config_hash == config_hash(BASELINE_GAME_CONFIG)
    assert protocol.game_abstraction_version == BASELINE_GAME_CONFIG.abstraction_version
    assert tuple(agent.agent_version for agent in protocol.hero_agents) == (
        RANDOM_BASELINE_VERSION,
        HAND_STRENGTH_BASELINE_VERSION,
    )
    assert protocol.static_profile_version == STATIC_PROFILE_VERSION
    assert protocol.parametric_policy_version == PARAMETRIC_POLICY_VERSION


def test_pilot_protocol_is_variance_only_not_final_data() -> None:
    protocol = _load_protocol()

    assert protocol.status == "frozen"
    assert protocol.purpose == "variance_and_runtime_estimation_only"
    assert not protocol.final_experiment_data
    assert not protocol.eligible_for_confirmatory_analysis
    assert protocol.minimum_relevant_difference_bb100 is None
    assert protocol.precision_target_half_width_bb100 is None
    assert protocol.final_independent_units is None
    assert protocol.replacement_seed_policy == "forbidden"


def test_pilot_matrix_is_1000_pairs_and_2000_engine_hands() -> None:
    protocol = _load_protocol()

    assert protocol.paired_units_per_cell == 100
    assert protocol.cell_count == 10
    assert protocol.planned_independent_units == 1000
    assert protocol.engine_hands_per_paired_unit == 2
    assert protocol.planned_simulated_engine_hands == 2000
    assert 1000 <= protocol.planned_simulated_engine_hands <= 5000


def test_seed_schedule_is_explicit_unique_and_shared_across_cells() -> None:
    protocol = _load_protocol()

    assert protocol.seed_schedule_version == "phase1-pilot-seeds-v1"
    assert protocol.master_seeds == tuple(range(6_100_000, 6_100_100))
    assert len(set(protocol.master_seeds)) == 100
    assert protocol.reuse_same_seed_schedule_across_cells


def test_primary_and_secondary_population_references_are_hash_bound() -> None:
    protocol = _load_protocol()

    assert protocol.primary_population.population_id == (
        BASELINE_SYNTHETIC_STATIC_POPULATION.population_id
    )
    assert protocol.primary_population.population_hash == (
        BASELINE_SYNTHETIC_STATIC_POPULATION.population_hash
    )
    assert tuple(ref.population_id for ref in protocol.secondary_populations) == (
        BALANCED_STATIC_POPULATION.population_id,
    )
    assert protocol.secondary_populations[0].population_hash == (
        BALANCED_STATIC_POPULATION.population_hash
    )


def test_pilot_jobs_are_canonical_and_reuse_each_seed_for_every_cell() -> None:
    protocol = _load_protocol()
    jobs = tuple(iter_pilot_jobs(protocol))

    assert len(jobs) == 1000
    assert jobs[0].hero_id == "random"
    assert jobs[0].opponent_profile_id == "nit"
    assert jobs[0].master_seed == 6_100_000
    assert jobs[-1].hero_id == "hand_strength"
    assert jobs[-1].opponent_profile_id == "maniac"
    assert jobs[-1].master_seed == 6_100_099

    first_seed_jobs = [job for job in jobs if job.master_seed == 6_100_000]
    assert len(first_seed_jobs) == 10
    assert {job.cell_id for job in first_seed_jobs} == {
        "random-vs-nit",
        "random-vs-tag",
        "random-vs-lag",
        "random-vs-calling_station",
        "random-vs-maniac",
        "hand_strength-vs-nit",
        "hand_strength-vs-tag",
        "hand_strength-vs-lag",
        "hand_strength-vs-calling_station",
        "hand_strength-vs-maniac",
    }


def test_protocol_file_is_already_canonical() -> None:
    raw = PILOT_PATH.read_bytes()
    assert raw == canonical_file_bytes(json.loads(raw))


def test_tampered_protocol_hash_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    payload["protocol"]["matrix"]["paired_units_per_cell"] = 101
    destination = tmp_path / "tampered.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    populations = {
        "baseline-synthetic-static-profiles": load_frozen_population(
            BASELINE_POPULATION_PATH
        ),
        "balanced-static-profiles": load_frozen_population(BALANCED_POPULATION_PATH),
    }
    with pytest.raises(PilotProtocolError, match="protocol_hash"):
        load_phase1_pilot_protocol(destination, populations=populations)


def test_population_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(BASELINE_POPULATION_PATH.read_text(encoding="utf-8"))
    payload["population"]["weights_ppm"][0]["weight_ppm"] += 1
    destination = tmp_path / "population.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PilotProtocolError, match="population_hash"):
        load_frozen_population(destination)


def test_protocol_rejects_final_design_values_even_with_recomputed_hash(
    tmp_path: Path,
) -> None:
    payload = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    protocol_payload = payload["protocol"]
    protocol_payload["final_design"]["final_independent_units"] = 50_000

    import hashlib

    canonical_protocol = json.dumps(
        protocol_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    payload["protocol_hash"] = hashlib.sha256(
        canonical_protocol.encode("utf-8")
    ).hexdigest()
    destination = tmp_path / "invalid-final.json"
    destination.write_bytes(canonical_file_bytes(payload))

    populations = {
        "baseline-synthetic-static-profiles": load_frozen_population(
            BASELINE_POPULATION_PATH
        ),
        "balanced-static-profiles": load_frozen_population(BALANCED_POPULATION_PATH),
    }
    with pytest.raises(PilotProtocolError, match="final design fields"):
        load_phase1_pilot_protocol(destination, populations=populations)
