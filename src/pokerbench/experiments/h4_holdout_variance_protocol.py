"""Frozen machine-readable Spec-37 H4 holdout variance-pilot design."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

H4_HOLDOUT_VARIANCE_PROTOCOL_ID = "phase2-h4-holdout-variance-pilot-v1"
H4_HOLDOUT_VARIANCE_PROTOCOL_VERSION = "phase2-h4-holdout-variance-pilot-protocol-v1"
H4_HOLDOUT_VARIANCE_FILE_VERSION = "frozen-phase2-h4-holdout-variance-pilot-file-v1"
H4_HOLDOUT_VARIANCE_PROTOCOL_HASH = "d2ad5569a82308a61390ccdc2212ac3b2790fa369e96506a418bac4a7045aac0"
H4_HOLDOUT_POPULATION_HASH = "b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3"
H4_HOLDOUT_ACCEPTANCE_HASH = "9ba347fb1d615cec1d6458944df3c293dcac630564e8847a407e987bbb98af84"
H4_VARIANCE_MASTER_SEEDS = tuple(range(6_400_000, 6_400_010))
H4_HOLDOUT_UNIT_IDS = tuple(f"h4_holdout_{index:03d}" for index in range(5))
H4_HOLDOUT_UNIT_WEIGHT = 0.2

class H4HoldoutVarianceProtocolError(ValueError):
    """Raised when machine-readable Spec 37 differs from the frozen design."""

@dataclass(frozen=True)
class H4HoldoutVarianceProtocol:
    protocol_id: str
    protocol_version: str
    status: str
    purpose: str
    payload_hash: str
    master_seeds: tuple[int, ...]
    holdout_unit_ids: tuple[str, ...]
    hands_per_position: int
    position_run_count: int
    planned_population_clusters: int
    planned_unit_pairs: int
    planned_arm_jobs: int
    planned_engine_hands: int

@dataclass(frozen=True)
class H4HoldoutVarianceJob:
    holdout_unit_id: str
    master_seed: int
    arm_id: str
    agent_version: str
    adaptive: bool
    hands_per_position: int = 500
    position_run_count: int = 2

    @property
    def simulated_engine_hands(self) -> int:
        return self.hands_per_position * self.position_run_count

@dataclass(frozen=True)
class H4HoldoutVariancePair:
    holdout_unit_id: str
    master_seed: int
    control_job: H4HoldoutVarianceJob
    treatment_job: H4HoldoutVarianceJob

def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")

def canonical_file_bytes(value: object) -> bytes:
    return _canonical(value) + b"\n"

def expected_h4_holdout_variance_payload() -> dict[str, Any]:
    return {'admission': {'campaign_rule': '10-of-10-valid-population-clusters-required-for-sigma_500',
                   'invalid_campaign_n_final': None,
                   'invalid_campaign_sigma_500': None,
                   'population_cluster_rule': 'all-five-unit-pairs-valid',
                   'replacement_seed_policy': 'forbidden',
                   'replay_required': True},
     'analysis': {'arm_estimator': 'position-compensated-cumulative-bb100-v1',
                  'matched_unit_contrast': 'adaptive-minus-control-trajectory-bb100-v1',
                  'population_aggregation': 'within-seed-equal-weight-before-across-seed-variance-v1',
                  'reduced_complete_case_planning': False,
                  'required_valid_population_clusters': 10,
                  'variance_ddof': 1,
                  'variance_estimator': 'sample-sd-matched-population-trajectory-contrast-bb100-ddof1-v1'},
     'arms': {'control': {'adaptive': False, 'agent_version': 'hand-strength-baseline-v1'},
              'treatment': {'adaptive': True,
                            'agent_version': 'adaptive-hand-strength-agent-v1',
                            'reset_before_each_position_run': True,
                            'share_state_between_position_runs': False}},
     'bindings': {'adaptive_dependencies': {'adaptive_learning_protocol': {'payload_hash': '03f8ab5b2b35c86910d46df1d5187c306814e427b3f514eb85748a9a424adf5c',
                                                                           'version': 'adaptive-learning-protocol-v1'},
                                            'adaptive_lifecycle': {'payload_hash': 'bc7f26c66861a37dec8c544863a9e97fbcf33e07176fa2733e99f5426bf10edc',
                                                                   'version': 'adaptive-lifecycle-v1'},
                                            'adaptive_policy': {'payload_hash': '76b6c561c2cb3715cc6352520dfbd3b0b01ce219a1b37d1a0acb62bb6ce10e4a',
                                                                'version': 'adaptive-hand-strength-policy-v1'},
                                            'opponent_model': 'bayesian-opponent-model-v1',
                                            'opportunity_definition': 'calibration-opportunities-v2',
                                            'posterior_to_range': {'payload_hash': 'af19b040f9049ecd3a3a12615dcbe049e1d1f859546681487176238e9d2ed668',
                                                                   'version': 'posterior-to-range-v1'},
                                            'public_action_trace': 'public-action-trace-v1'},
                  'game': {'abstraction_version': 'baseline-v1',
                           'game_config_hash': 'b1916b8c91e234bb0b4d4863aab601db8ccfd8b73e336a924310e13644bae625'},
                  'holdout_population': {'holdout_acceptance_hash': '9ba347fb1d615cec1d6458944df3c293dcac630564e8847a407e987bbb98af84',
                                         'population_hash': 'b54582201f6a88aa367cc210a6a65e545f2f95bac576b8c90d0bdec02283c9b3',
                                         'population_id': 'phase2-h4-holdout-population-v1',
                                         'population_role': 'holdout_test',
                                         'population_version': 'phase2-h4-holdout-parametric-population-v1'},
                  'opponent': {'parameter_schema_version': 'opponent-parameters-v1',
                               'parametric_policy_version': 'parametric-policy-v1'},
                  'planning_spec': 'phase2-h4-confirmatory-planning-v1'},
     'information_boundary': {'adaptive_learning_source': 'persisted-public-action-trace-v1',
                              'hero_forbidden_inputs': ['holdout_unit_id',
                                                        'population_id',
                                                        'population_role',
                                                        'population_weight',
                                                        'true_opponent_parameter_vector',
                                                        'generation_seed_material',
                                                        'latent_pfr_ratio',
                                                        'calibration_target',
                                                        'calibration_acceptance_result',
                                                        'controller_side_classification']},
     'matrix': {'arm_order': ['control', 'treatment'],
                'canonical_order': 'holdout_unit_then_master_seed_then_arm',
                'holdout_units': [{'unit_id': 'h4_holdout_000', 'weight': 0.2},
                                  {'unit_id': 'h4_holdout_001', 'weight': 0.2},
                                  {'unit_id': 'h4_holdout_002', 'weight': 0.2},
                                  {'unit_id': 'h4_holdout_003', 'weight': 0.2},
                                  {'unit_id': 'h4_holdout_004', 'weight': 0.2}],
                'master_seeds': [6400000,
                                 6400001,
                                 6400002,
                                 6400003,
                                 6400004,
                                 6400005,
                                 6400006,
                                 6400007,
                                 6400008,
                                 6400009],
                'optional_stopping': False,
                'planned_arm_jobs': 100,
                'planned_engine_hands': 100000,
                'planned_population_clusters': 10,
                'planned_unit_pairs': 50,
                'replacement_seeds': 'forbidden',
                'seed_schedule_version': 'phase2-h4-holdout-variance-pilot-seeds-v1'},
     'prohibited_uses': ['confirmatory_h4_inference',
                         'final_h4_effect_estimation',
                         'policy_or_model_tuning',
                         'holdout_population_tuning',
                         'primary_horizon_selection',
                         'mrd_alpha_beta_or_precision_changes',
                         'optional_stopping',
                         'pilot_plus_final_pooling',
                         'variance_pilot_plus_final_pooling',
                         'seed_replacement'],
     'protocol_id': 'phase2-h4-holdout-variance-pilot-v1',
     'protocol_version': 'phase2-h4-holdout-variance-pilot-protocol-v1',
     'purpose': 'holdout_long_horizon_variance_runtime_and_final_h4_sample_size_planning_only',
     'rng': {'agent_rng_relation': 'event_indexed_common_identity_where_events_correspond',
             'deal_relation': 'identical_defined_card_realisations',
             'full_action_path_identity': 'not_claimed',
             'reuse_same_seed_schedule_across_arms': True,
             'reuse_same_seed_schedule_across_positions': True,
             'reuse_same_seed_schedule_across_units': True},
     'sample_size_planning': {'final_formula': 'max(minimum_final_clusters,ceil(n_power_raw),ceil(n_precision_raw))',
                              'minimum_final_clusters': 10,
                              'minimum_relevant_difference_bb100': 50.0,
                              'planning_half_width_formula': 'z_0_975*sigma_500/sqrt(n_final)',
                              'power_formula': '((z_0_975+z_0_80)^2*sigma_500^2)/(minimum_relevant_difference_bb100^2)',
                              'precision_formula': '((z_0_975*sigma_500)/precision_target_half_width_bb100)^2',
                              'precision_target_half_width_bb100': 50.0,
                              'z_0_80': 0.8416212335729143,
                              'z_0_975': 1.9599639845400536},
     'scientific_use': {'confirmatory_h4_inference_permitted': False,
                        'final_experiment_data': False,
                        'pilot_final_pooling': False,
                        'variance_pilot_mean_design_use': 'prohibited'},
     'seed_disjointness': {'adaptive_development_pilot': [6300000, 6300004],
                           'final_h4': 'future_separately_frozen_disjoint_seed_domain',
                           'phase1_final': [6200000, 6202768],
                           'phase1_pilot': [6100000, 6100099]},
     'status': 'design_frozen_execution_not_yet_authorized',
     'trajectory': {'adaptive_reset_before_each_position_run': True,
                    'execution_mode': 'full-match',
                    'hand_level_independence': False,
                    'hands_per_position': 500,
                    'position_run_count': 2,
                    'position_runs': ['hero-BTN_vs_opponent-BB', 'opponent-BTN_vs_hero-BB'],
                    'statistical_unit': 'matched_population_trajectory_cluster'}}

def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise H4HoldoutVarianceProtocolError(f"{name} must be a JSON object")
    return value

def validate_h4_holdout_variance_document(document: object) -> H4HoldoutVarianceProtocol:
    root = _mapping(document, "variance-pilot document")
    if set(root) != {"file_version", "payload", "payload_hash"}:
        raise H4HoldoutVarianceProtocolError("invalid variance-pilot document keys")
    if root["file_version"] != H4_HOLDOUT_VARIANCE_FILE_VERSION:
        raise H4HoldoutVarianceProtocolError("invalid variance-pilot file version")
    payload = _mapping(root["payload"], "payload")
    payload_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    if root["payload_hash"] != payload_hash:
        raise H4HoldoutVarianceProtocolError("variance-pilot payload hash mismatch")
    if dict(payload) != expected_h4_holdout_variance_payload():
        raise H4HoldoutVarianceProtocolError("variance-pilot semantics differ from frozen Spec 37")
    if payload_hash != H4_HOLDOUT_VARIANCE_PROTOCOL_HASH:
        raise H4HoldoutVarianceProtocolError("unexpected frozen Spec-37 payload hash")
    matrix = _mapping(payload["matrix"], "matrix")
    trajectory = _mapping(payload["trajectory"], "trajectory")
    units = matrix["holdout_units"]
    if not isinstance(units, list):
        raise H4HoldoutVarianceProtocolError("holdout_units must be an array")
    return H4HoldoutVarianceProtocol(
        protocol_id=str(payload["protocol_id"]), protocol_version=str(payload["protocol_version"]),
        status=str(payload["status"]), purpose=str(payload["purpose"]), payload_hash=payload_hash,
        master_seeds=tuple(int(seed) for seed in matrix["master_seeds"]),
        holdout_unit_ids=tuple(str(item["unit_id"]) for item in units),
        hands_per_position=int(trajectory["hands_per_position"]), position_run_count=int(trajectory["position_run_count"]),
        planned_population_clusters=int(matrix["planned_population_clusters"]), planned_unit_pairs=int(matrix["planned_unit_pairs"]),
        planned_arm_jobs=int(matrix["planned_arm_jobs"]), planned_engine_hands=int(matrix["planned_engine_hands"]),
    )

def load_h4_holdout_variance_protocol(path: str | Path) -> H4HoldoutVarianceProtocol:
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if raw != canonical_file_bytes(document):
        raise H4HoldoutVarianceProtocolError("variance-pilot protocol file is not canonical JSON/LF")
    return validate_h4_holdout_variance_document(document)

def iter_h4_holdout_variance_jobs(protocol: H4HoldoutVarianceProtocol) -> tuple[H4HoldoutVarianceJob, ...]:
    if not isinstance(protocol, H4HoldoutVarianceProtocol):
        raise TypeError("protocol must be H4HoldoutVarianceProtocol")
    arms = (("control", "hand-strength-baseline-v1", False), ("treatment", "adaptive-hand-strength-agent-v1", True))
    jobs = tuple(H4HoldoutVarianceJob(unit, seed, arm, agent, adaptive, protocol.hands_per_position, protocol.position_run_count)
                 for unit in protocol.holdout_unit_ids for seed in protocol.master_seeds for arm, agent, adaptive in arms)
    if len(jobs) != 100 or sum(job.simulated_engine_hands for job in jobs) != 100_000:
        raise H4HoldoutVarianceProtocolError("Spec-37 job expansion mismatch")
    return jobs

def iter_h4_holdout_variance_pairs(protocol: H4HoldoutVarianceProtocol) -> tuple[H4HoldoutVariancePair, ...]:
    jobs = iter_h4_holdout_variance_jobs(protocol)
    pairs = []
    for offset in range(0, len(jobs), 2):
        control, treatment = jobs[offset], jobs[offset + 1]
        if control.arm_id != "control" or treatment.arm_id != "treatment" or (control.holdout_unit_id, control.master_seed) != (treatment.holdout_unit_id, treatment.master_seed):
            raise H4HoldoutVarianceProtocolError("Spec-37 pair expansion mismatch")
        pairs.append(H4HoldoutVariancePair(control.holdout_unit_id, control.master_seed, control, treatment))
    if len(pairs) != 50:
        raise H4HoldoutVarianceProtocolError("Spec-37 pair expansion must contain 50 pairs")
    return tuple(pairs)
