from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pokerbench.experiments.adaptive_pilot_protocol import (
    ADAPTIVE_PILOT_FILE_VERSION,
    ADAPTIVE_PILOT_PROTOCOL_HASH,
    ADAPTIVE_PILOT_PROTOCOL_ID,
    ADAPTIVE_PILOT_PROTOCOL_VERSION,
    AdaptivePilotProtocolError,
    canonical_file_bytes,
    iter_adaptive_pilot_jobs,
    load_adaptive_pilot_protocol,
    validate_adaptive_pilot_document,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-v1.json"


def _load_raw() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_bytes())


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "file_version": ADAPTIVE_PILOT_FILE_VERSION,
        "payload": payload,
        "payload_hash": hashlib.sha256(canonical).hexdigest(),
    }


def test_protocol_is_canonical_hash_bound_and_has_exact_identity() -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)

    assert protocol.protocol_id == ADAPTIVE_PILOT_PROTOCOL_ID
    assert protocol.protocol_version == ADAPTIVE_PILOT_PROTOCOL_VERSION
    assert protocol.status == "design_frozen_execution_not_yet_authorized"
    assert protocol.purpose == "adaptive_learning_curve_variance_runtime_and_h4_planning_only"
    assert protocol.payload_hash == ADAPTIVE_PILOT_PROTOCOL_HASH
    assert protocol.payload_hash == "ef86d3da94369a86ac1c496ddf981b39d30919e086b35e026039bac3afc8ccb4"
    raw = PROTOCOL_PATH.read_bytes()
    assert b"\r" not in raw
    assert raw == canonical_file_bytes(json.loads(raw))


def test_treatment_control_and_frozen_dependency_versions_are_exact() -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)

    assert protocol.control_agent_version == "hand-strength-baseline-v1"
    assert protocol.treatment_agent_version == "adaptive-hand-strength-agent-v1"
    assert protocol.dependency_versions == (
        ("adaptive_learning_protocol", "adaptive-learning-protocol-v1"),
        ("posterior_to_range", "posterior-to-range-v1"),
        ("adaptive_policy", "adaptive-hand-strength-policy-v1"),
        ("adaptive_lifecycle", "adaptive-lifecycle-v1"),
        ("opponent_model", "bayesian-opponent-model-v1"),
        ("opportunity_definition", "calibration-opportunities-v2"),
        ("public_action_trace", "public-action-trace-v1"),
    )

    raw = _load_raw()
    payload = raw["payload"]
    assert isinstance(payload, dict)
    dependencies = payload["dependencies"]
    assert isinstance(dependencies, dict)
    assert dependencies["adaptive_learning_protocol"]["payload_hash"] == "03f8ab5b2b35c86910d46df1d5187c306814e427b3f514eb85748a9a424adf5c"
    assert dependencies["posterior_to_range"]["payload_hash"] == "af19b040f9049ecd3a3a12615dcbe049e1d1f859546681487176238e9d2ed668"
    assert dependencies["adaptive_policy"]["payload_hash"] == "76b6c561c2cb3715cc6352520dfbd3b0b01ce219a1b37d1a0acb62bb6ce10e4a"
    assert dependencies["adaptive_lifecycle"]["payload_hash"] == "bc7f26c66861a37dec8c544863a9e97fbcf33e07176fa2733e99f5426bf10edc"


def test_development_opponents_populations_and_gate6_binding_are_exact() -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)

    assert protocol.opponent_profile_ids == (
        "nit",
        "tag",
        "lag",
        "calling_station",
        "maniac",
    )
    assert protocol.gate6_pass_required is True
    assert protocol.gate6_report_hash_binding == "bind_before_first_pilot_hand"
    assert protocol.primary_population_id == "baseline-synthetic-static-profiles"
    assert protocol.secondary_population_ids == ("balanced-static-profiles",)


def test_two_tier_seed_schedule_horizons_and_segments_are_exact() -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)

    assert protocol.seed_schedule_version == "phase2-adaptive-pilot-seeds-v1"
    assert protocol.master_seeds == (6_300_000, 6_300_001, 6_300_002, 6_300_003, 6_300_004)
    assert protocol.long_horizon_master_seed == 6_300_000
    assert protocol.short_horizon_hands_per_position == 100
    assert protocol.long_horizon_hands_per_position == 500
    assert protocol.cumulative_horizons == (10, 20, 50, 100, 500)
    assert protocol.segments == (
        ("early_adaptation", 1, 20),
        ("main_adaptation", 21, 100),
        ("steady_state", 101, 500),
    )


def test_job_expansion_is_deterministic_and_totals_18000_engine_hands() -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    jobs = iter_adaptive_pilot_jobs(protocol)

    assert len(jobs) == 50
    assert sum(job.simulated_engine_hands for job in jobs) == 18_000
    assert jobs[0].arm_id == "control"
    assert jobs[0].opponent_profile_id == "nit"
    assert jobs[0].master_seed == 6_300_000
    assert jobs[0].hands_per_position == 500
    assert jobs[-1].arm_id == "treatment"
    assert jobs[-1].opponent_profile_id == "maniac"
    assert jobs[-1].master_seed == 6_300_004
    assert jobs[-1].hands_per_position == 100

    long_jobs = tuple(job for job in jobs if job.master_seed == 6_300_000)
    assert len(long_jobs) == 10
    assert all(job.simulated_engine_hands == 1_000 for job in long_jobs)
    assert all(job.statistical_unit == "trajectory_cluster" for job in jobs)
    assert all(job.execution_mode == "full-match" for job in jobs)
    assert all(job.position_run_count == 2 for job in jobs)


def test_trajectory_execution_contract_distinguishes_adaptive_state_only() -> None:
    raw = _load_raw()
    payload = raw["payload"]
    assert isinstance(payload, dict)
    trajectory = payload["trajectory"]
    arms = payload["arms"]
    assert isinstance(trajectory, dict)
    assert isinstance(arms, dict)

    assert trajectory["statistical_unit"] == "trajectory_cluster"
    assert trajectory["execution_mode"] == "full-match"
    assert trajectory["position_runs"] == [
        "hero-BTN_vs_opponent-BB",
        "opponent-BTN_vs_hero-BB",
    ]
    treatment = arms["treatment"]
    control = arms["control"]
    assert isinstance(treatment, dict)
    assert isinstance(control, dict)
    assert treatment["adaptive"] is True
    assert treatment["reset_before_each_position_run"] is True
    assert treatment["share_state_between_position_runs"] is False
    assert control["adaptive"] is False
    assert control["bayesian_learning_state"] == "none"


def test_information_boundary_and_crn_claim_are_exact() -> None:
    raw = _load_raw()
    payload = raw["payload"]
    assert isinstance(payload, dict)
    information = payload["information_boundary"]
    rng = payload["rng"]
    assert isinstance(information, dict)
    assert isinstance(rng, dict)

    assert information["hero_forbidden_inputs"] == [
        "profile_id",
        "true_opponent_parameter_vector",
        "population_weights",
        "hidden_opponent_cards",
        "hidden_deck_state",
        "opponent_rng_state",
    ]
    assert information["adaptive_learning_source"] == "persisted-public-action-trace-v1"
    assert rng["deal_relation"] == "identical_defined_card_realisations"
    assert rng["agent_rng_relation"] == "event_indexed_common_identity_where_events_correspond"
    assert rng["full_action_path_identity"] == "not_claimed"


def test_failure_analysis_and_500_hand_inference_boundaries_are_frozen() -> None:
    raw = _load_raw()
    payload = raw["payload"]
    assert isinstance(payload, dict)
    failure = payload["failure_handling"]
    analysis = payload["pilot_analysis"]
    assert isinstance(failure, dict)
    assert isinstance(analysis, dict)

    assert failure == {
        "adaptive_timeout_policy": "invalidate_trajectory_cluster",
        "illegal_action_policy": "exclude_run",
        "matched_contrast_policy": "complete_pair_only",
        "replacement_seed_policy": "forbidden",
    }
    assert analysis["variance_horizons"] == [10, 20, 50, 100]
    assert analysis["horizon_500_usage"] == "descriptive_only_no_variance_ci_or_inference"
    assert analysis["population_aggregation_order"] == "within_seed_weighting_then_across_seed_variance"
    assert "confirmatory_h4_inference" in analysis["forbidden_scientific_use"]
    assert "pilot_plus_final_data_pooling" in analysis["forbidden_scientific_use"]


def test_final_h4_design_fields_remain_deliberately_unset() -> None:
    raw = _load_raw()
    payload = raw["payload"]
    assert isinstance(payload, dict)
    final_h4 = payload["final_h4_design"]
    assert isinstance(final_h4, dict)

    assert final_h4 == {
        "confirmatory_alpha": None,
        "confirmatory_beta": None,
        "final_h4_population": None,
        "final_h4_seed_schedule": None,
        "final_independent_trajectory_clusters": None,
        "minimum_relevant_difference_bb100": None,
        "multiplicity_binding": None,
        "precision_target_half_width_bb100": None,
        "primary_h4_horizon": None,
    }


def test_required_provenance_binds_protocol_and_all_adaptive_dependencies() -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)

    assert protocol.required_run_provenance == (
        "experiment_id",
        "pilot_protocol_id",
        "pilot_protocol_hash",
        "game_config_hash",
        "hero_arm",
        "bot_version",
        "opponent_version",
        "opponent_profile_id",
        "opponent_parameters",
        "population_versions",
        "master_seed",
        "planned_trajectory_horizon",
        "rng_configuration",
        "engine_version",
        "python_version",
        "poetry_lock_hash",
        "git_commit",
        "cpu_model",
        "cpu_core_count",
        "cpu_architecture",
        "os_identifier",
        "statistical_unit",
        "gate6_report_hash",
        "adaptive_learning_protocol_hash",
        "posterior_to_range_protocol_hash",
        "adaptive_policy_protocol_hash",
        "adaptive_lifecycle_protocol_hash",
    )


def test_semantic_tampering_is_rejected_even_after_outer_rehash() -> None:
    raw = _load_raw()
    payload = copy.deepcopy(raw["payload"])
    assert isinstance(payload, dict)
    matrix = payload["matrix"]
    assert isinstance(matrix, dict)
    matrix["long_horizon_hands_per_position"] = 400

    with pytest.raises(AdaptivePilotProtocolError, match="semantics"):
        validate_adaptive_pilot_document(_rehash(payload))
