"""Frozen machine-readable design for the first Phase-2 adaptive pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTIVE_PILOT_PROTOCOL_ID = "phase2-adaptive-pilot-v1"
ADAPTIVE_PILOT_PROTOCOL_VERSION = "phase2-adaptive-pilot-protocol-v1"
ADAPTIVE_PILOT_FILE_VERSION = "frozen-phase2-adaptive-pilot-file-v1"
ADAPTIVE_PILOT_PROTOCOL_HASH = "ef86d3da94369a86ac1c496ddf981b39d30919e086b35e026039bac3afc8ccb4"


class AdaptivePilotProtocolError(ValueError):
    """Raised when the frozen adaptive-pilot design is malformed or inconsistent."""


@dataclass(frozen=True)
class AdaptivePilotProtocol:
    """Validated immutable subset of the adaptive-pilot design used by planners."""

    protocol_id: str
    protocol_version: str
    status: str
    purpose: str
    payload_hash: str
    control_agent_version: str
    treatment_agent_version: str
    dependency_versions: tuple[tuple[str, str], ...]
    opponent_profile_ids: tuple[str, ...]
    gate6_pass_required: bool
    gate6_report_hash_binding: str
    primary_population_id: str
    secondary_population_ids: tuple[str, ...]
    seed_schedule_version: str
    master_seeds: tuple[int, ...]
    long_horizon_master_seed: int
    short_horizon_hands_per_position: int
    long_horizon_hands_per_position: int
    cumulative_horizons: tuple[int, ...]
    segments: tuple[tuple[str, int, int], ...]
    required_run_provenance: tuple[str, ...]


@dataclass(frozen=True)
class AdaptivePilotJob:
    """One deterministic arm/profile/seed trajectory-cluster plan."""

    arm_id: str
    agent_version: str
    adaptive: bool
    opponent_profile_id: str
    master_seed: int
    hands_per_position: int
    position_run_count: int = 2
    statistical_unit: str = "trajectory_cluster"
    execution_mode: str = "full-match"

    @property
    def simulated_engine_hands(self) -> int:
        return self.hands_per_position * self.position_run_count


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_file_bytes(value: object) -> bytes:
    """Return canonical UTF-8 JSON file bytes with exactly one trailing LF."""
    return _canonical(value) + b"\n"


def expected_adaptive_pilot_payload() -> dict[str, Any]:
    """Return the exact semantic payload frozen by Spec 32."""
    return {
        "protocol_id": ADAPTIVE_PILOT_PROTOCOL_ID,
        "protocol_version": ADAPTIVE_PILOT_PROTOCOL_VERSION,
        "status": "design_frozen_execution_not_yet_authorized",
        "purpose": "adaptive_learning_curve_variance_runtime_and_h4_planning_only",
        "scientific_use": {
            "final_experiment_data": False,
            "eligible_for_confirmatory_analysis": False,
            "pilot_final_pooling": False,
        },
        "dependencies": {
            "adaptive_learning_protocol": {
                "version": "adaptive-learning-protocol-v1",
                "payload_hash": "03f8ab5b2b35c86910d46df1d5187c306814e427b3f514eb85748a9a424adf5c",
            },
            "posterior_to_range": {
                "version": "posterior-to-range-v1",
                "payload_hash": "af19b040f9049ecd3a3a12615dcbe049e1d1f859546681487176238e9d2ed668",
            },
            "adaptive_policy": {
                "version": "adaptive-hand-strength-policy-v1",
                "payload_hash": "76b6c561c2cb3715cc6352520dfbd3b0b01ce219a1b37d1a0acb62bb6ce10e4a",
            },
            "adaptive_lifecycle": {
                "version": "adaptive-lifecycle-v1",
                "payload_hash": "bc7f26c66861a37dec8c544863a9e97fbcf33e07176fa2733e99f5426bf10edc",
            },
            "opponent_model": "bayesian-opponent-model-v1",
            "opportunity_definition": "calibration-opportunities-v2",
            "public_action_trace": "public-action-trace-v1",
        },
        "game": {
            "abstraction_version": "baseline-v1",
            "game_config_hash": "b1916b8c91e234bb0b4d4863aab601db8ccfd8b73e336a924310e13644bae625",
        },
        "arms": {
            "control": {
                "agent_version": "hand-strength-baseline-v1",
                "adaptive": False,
                "opponent_range": "uniform-random-range-v1",
                "bayesian_learning_state": "none",
            },
            "treatment": {
                "agent_version": "adaptive-hand-strength-agent-v1",
                "adaptive": True,
                "opponent_range": "posterior-to-range-v1",
                "lifecycle": "adaptive-lifecycle-v1",
                "reset_before_each_position_run": True,
                "share_state_between_position_runs": False,
            },
        },
        "information_boundary": {
            "hero_forbidden_inputs": [
                "profile_id",
                "true_opponent_parameter_vector",
                "population_weights",
                "hidden_opponent_cards",
                "hidden_deck_state",
                "opponent_rng_state",
            ],
            "adaptive_learning_source": "persisted-public-action-trace-v1",
        },
        "opponents": {
            "profile_ids": ["nit", "tag", "lag", "calling_station", "maniac"],
            "static_profile_version": "static-profiles-v2",
            "parametric_policy_version": "parametric-policy-v1",
            "gate6_pass_required": True,
            "gate6_report_hash_binding": "bind_before_first_pilot_hand",
        },
        "populations": {
            "primary_descriptive": {
                "population_id": "baseline-synthetic-static-profiles",
                "version": "baseline-synthetic-static-profiles-v1",
                "population_hash": "ae332873e33e409be48118116c75ecfa44ed612785630c4799f6948902665888",
            },
            "secondary_descriptive": [
                {
                    "population_id": "balanced-static-profiles",
                    "version": "balanced-static-profiles-v1",
                    "population_hash": "776d1ffb3a2558143d061129df968949e26621cce8595f48e0d1739726d5dc7f",
                }
            ],
            "final_h4_population": None,
        },
        "trajectory": {
            "statistical_unit": "trajectory_cluster",
            "execution_mode": "full-match",
            "position_runs": [
                "hero-BTN_vs_opponent-BB",
                "opponent-BTN_vs_hero-BB",
            ],
            "position_run_count": 2,
            "hand_level_independence": False,
            "same_defined_card_realisations_between_positions": True,
        },
        "matrix": {
            "seed_schedule_version": "phase2-adaptive-pilot-seeds-v1",
            "master_seeds": [6_300_000, 6_300_001, 6_300_002, 6_300_003, 6_300_004],
            "long_horizon_master_seed": 6_300_000,
            "short_horizon_hands_per_position": 100,
            "long_horizon_hands_per_position": 500,
            "arm_count": 2,
            "profile_count": 5,
            "planned_trajectory_cluster_jobs": 50,
            "planned_engine_hands": 18_000,
            "sequential_stopping": False,
        },
        "rng": {
            "deal_relation": "identical_defined_card_realisations",
            "agent_rng_relation": "event_indexed_common_identity_where_events_correspond",
            "full_action_path_identity": "not_claimed",
            "reuse_same_seed_schedule_across_profiles": True,
            "reuse_same_seed_schedule_across_arms": True,
            "reuse_same_seed_schedule_across_positions": True,
        },
        "horizons_and_segments": {
            "cumulative_horizons": [10, 20, 50, 100, 500],
            "segments": [
                {"segment_id": "early_adaptation", "start_hand": 1, "end_hand": 20},
                {"segment_id": "main_adaptation", "start_hand": 21, "end_hand": 100},
                {"segment_id": "steady_state", "start_hand": 101, "end_hand": 500},
            ],
            "steady_state_descriptive_only": True,
        },
        "pilot_analysis": {
            "arm_estimator": "position-compensated-cumulative-bb100-v1",
            "matched_contrast": "adaptive-minus-control-trajectory-bb100-v1",
            "variance_estimator": "sample-sd-matched-trajectory-contrast-bb100-ddof1-v1",
            "variance_horizons": [10, 20, 50, 100],
            "horizon_500_usage": "descriptive_only_no_variance_ci_or_inference",
            "population_aggregation_order": "within_seed_weighting_then_across_seed_variance",
            "allowed_scientific_use": [
                "end_to_end_adaptive_validation",
                "learning_curve_description",
                "posterior_state_diagnostics",
                "variance_estimation_at_10_20_50_100",
                "runtime_and_storage_estimation",
                "final_h4_sample_size_planning_input",
                "final_h4_horizon_planning_from_non_outcome_diagnostics",
            ],
            "forbidden_scientific_use": [
                "confirmatory_h4_inference",
                "final_h4_effect_estimate",
                "pilot_plus_final_data_pooling",
                "optional_stopping",
                "seed_replacement_after_failure",
                "adaptive_policy_tuning_from_pilot_ev",
                "opponent_profile_tuning_from_pilot_ev",
                "primary_h4_horizon_selection_from_observed_delta_direction_or_p_value",
                "holdout_population_inspection",
                "mcts_evaluation",
            ],
        },
        "learning_diagnostics": {
            "required_fields": [
                "trajectory_hand_count",
                "public_evidence_counts_all_six_metrics",
                "posterior_alpha_beta_all_six_metrics",
                "posterior_state_hash",
                "valid_learning_transition_count",
            ],
            "replay_required": True,
            "replay_source": "public-evidence-and-frozen-state-hashes",
        },
        "failure_handling": {
            "illegal_action_policy": "exclude_run",
            "replacement_seed_policy": "forbidden",
            "matched_contrast_policy": "complete_pair_only",
            "adaptive_timeout_policy": "invalidate_trajectory_cluster",
        },
        "final_h4_design": {
            "primary_h4_horizon": None,
            "final_h4_population": None,
            "minimum_relevant_difference_bb100": None,
            "precision_target_half_width_bb100": None,
            "final_independent_trajectory_clusters": None,
            "confirmatory_alpha": None,
            "confirmatory_beta": None,
            "multiplicity_binding": None,
            "final_h4_seed_schedule": None,
        },
        "required_run_provenance": [
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
        ],
    }


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdaptivePilotProtocolError(f"{field_name} must be a JSON object")
    return value


def validate_adaptive_pilot_document(document: object) -> AdaptivePilotProtocol:
    """Validate a decoded machine-readable Spec-32 document."""
    root = _require_mapping(document, "adaptive pilot document")
    if set(root) != {"file_version", "payload", "payload_hash"}:
        raise AdaptivePilotProtocolError("invalid adaptive pilot document keys")
    if root["file_version"] != ADAPTIVE_PILOT_FILE_VERSION:
        raise AdaptivePilotProtocolError("invalid adaptive pilot file version")
    payload = _require_mapping(root["payload"], "payload")
    payload_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    if root["payload_hash"] != payload_hash:
        raise AdaptivePilotProtocolError("adaptive pilot payload hash mismatch")
    expected = expected_adaptive_pilot_payload()
    if dict(payload) != expected:
        raise AdaptivePilotProtocolError("adaptive pilot protocol semantics differ from frozen v1")
    if payload_hash != ADAPTIVE_PILOT_PROTOCOL_HASH:
        raise AdaptivePilotProtocolError("adaptive pilot protocol hash differs from frozen v1")

    dependencies = _require_mapping(payload["dependencies"], "dependencies")
    opponents = _require_mapping(payload["opponents"], "opponents")
    populations = _require_mapping(payload["populations"], "populations")
    matrix = _require_mapping(payload["matrix"], "matrix")
    horizons = _require_mapping(payload["horizons_and_segments"], "horizons_and_segments")
    arms = _require_mapping(payload["arms"], "arms")
    control = _require_mapping(arms["control"], "arms.control")
    treatment = _require_mapping(arms["treatment"], "arms.treatment")
    primary = _require_mapping(populations["primary_descriptive"], "populations.primary_descriptive")
    secondary = populations["secondary_descriptive"]
    if not isinstance(secondary, list):
        raise AdaptivePilotProtocolError("populations.secondary_descriptive must be an array")
    secondary_ids = tuple(
        str(_require_mapping(item, "secondary population")["population_id"])
        for item in secondary
    )
    segments_raw = horizons["segments"]
    if not isinstance(segments_raw, list):
        raise AdaptivePilotProtocolError("horizons_and_segments.segments must be an array")
    segments = tuple(
        (
            str(_require_mapping(item, "segment")["segment_id"]),
            int(_require_mapping(item, "segment")["start_hand"]),
            int(_require_mapping(item, "segment")["end_hand"]),
        )
        for item in segments_raw
    )
    version_keys = (
        "adaptive_learning_protocol",
        "posterior_to_range",
        "adaptive_policy",
        "adaptive_lifecycle",
    )
    dependency_versions = tuple(
        (key, str(_require_mapping(dependencies[key], key)["version"]))
        for key in version_keys
    ) + (
        ("opponent_model", str(dependencies["opponent_model"])),
        ("opportunity_definition", str(dependencies["opportunity_definition"])),
        ("public_action_trace", str(dependencies["public_action_trace"])),
    )
    provenance = payload["required_run_provenance"]
    if not isinstance(provenance, list):
        raise AdaptivePilotProtocolError("required_run_provenance must be an array")

    return AdaptivePilotProtocol(
        protocol_id=str(payload["protocol_id"]),
        protocol_version=str(payload["protocol_version"]),
        status=str(payload["status"]),
        purpose=str(payload["purpose"]),
        payload_hash=payload_hash,
        control_agent_version=str(control["agent_version"]),
        treatment_agent_version=str(treatment["agent_version"]),
        dependency_versions=dependency_versions,
        opponent_profile_ids=tuple(str(item) for item in opponents["profile_ids"]),
        gate6_pass_required=bool(opponents["gate6_pass_required"]),
        gate6_report_hash_binding=str(opponents["gate6_report_hash_binding"]),
        primary_population_id=str(primary["population_id"]),
        secondary_population_ids=secondary_ids,
        seed_schedule_version=str(matrix["seed_schedule_version"]),
        master_seeds=tuple(int(item) for item in matrix["master_seeds"]),
        long_horizon_master_seed=int(matrix["long_horizon_master_seed"]),
        short_horizon_hands_per_position=int(matrix["short_horizon_hands_per_position"]),
        long_horizon_hands_per_position=int(matrix["long_horizon_hands_per_position"]),
        cumulative_horizons=tuple(int(item) for item in horizons["cumulative_horizons"]),
        segments=segments,
        required_run_provenance=tuple(str(item) for item in provenance),
    )


def load_adaptive_pilot_protocol(path: str | Path) -> AdaptivePilotProtocol:
    """Load and validate the canonical machine-readable Spec-32 protocol."""
    source = Path(path)
    raw = source.read_bytes()
    document = json.loads(raw)
    if raw != canonical_file_bytes(document):
        raise AdaptivePilotProtocolError("adaptive pilot protocol file is not canonical JSON/LF")
    return validate_adaptive_pilot_document(document)


def iter_adaptive_pilot_jobs(protocol: AdaptivePilotProtocol) -> tuple[AdaptivePilotJob, ...]:
    """Expand Spec 32 into canonical arm/profile/seed trajectory-cluster jobs."""
    if not isinstance(protocol, AdaptivePilotProtocol):
        raise TypeError("protocol must be an AdaptivePilotProtocol")

    arm_definitions = (
        ("control", protocol.control_agent_version, False),
        ("treatment", protocol.treatment_agent_version, True),
    )
    jobs = tuple(
        AdaptivePilotJob(
            arm_id=arm_id,
            agent_version=agent_version,
            adaptive=adaptive,
            opponent_profile_id=profile_id,
            master_seed=seed,
            hands_per_position=(
                protocol.long_horizon_hands_per_position
                if seed == protocol.long_horizon_master_seed
                else protocol.short_horizon_hands_per_position
            ),
        )
        for arm_id, agent_version, adaptive in arm_definitions
        for profile_id in protocol.opponent_profile_ids
        for seed in protocol.master_seeds
    )
    if len(jobs) != 50:
        raise AdaptivePilotProtocolError("adaptive pilot job expansion must contain 50 jobs")
    if sum(job.simulated_engine_hands for job in jobs) != 18_000:
        raise AdaptivePilotProtocolError("adaptive pilot job expansion must contain 18,000 engine hands")
    return jobs
