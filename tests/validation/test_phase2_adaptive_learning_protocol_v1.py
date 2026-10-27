from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pokerbench.experiments.adaptive_learning_protocol import (
    ADAPTIVE_LEARNING_PROTOCOL_ID,
    ADAPTIVE_LEARNING_PROTOCOL_VERSION,
    AdaptiveLearningProtocolError,
    canonical_file_bytes,
    load_adaptive_learning_protocol,
    validate_adaptive_learning_payload,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-learning-protocol-v1.json"


def _load_raw() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_bytes())


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    from hashlib import sha256

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "file_version": "frozen-adaptive-learning-protocol-file-v1",
        "payload": payload,
        "payload_hash": sha256(canonical).hexdigest(),
    }


def test_frozen_protocol_loads_and_has_expected_identity() -> None:
    protocol = load_adaptive_learning_protocol(PROTOCOL_PATH)

    assert protocol.protocol_id == ADAPTIVE_LEARNING_PROTOCOL_ID
    assert protocol.protocol_version == ADAPTIVE_LEARNING_PROTOCOL_VERSION
    assert protocol.status == "frozen"
    assert protocol.scope == "observation-and-learning-only"


def test_all_six_priors_are_explicit_beta_1_1() -> None:
    protocol = load_adaptive_learning_protocol(PROTOCOL_PATH)

    assert set(protocol.priors) == {
        "vpip",
        "pfr",
        "three_bet_freq",
        "cbet_freq",
        "postflop_aggressive_share",
        "fold_to_cbet",
    }
    assert all(prior == (1, 1) for prior in protocol.priors.values())
    assert protocol.prior_selection_basis == "independent-uniform-beta-1-1-v1"


def test_updates_are_once_per_completed_hand_and_never_affect_same_hand() -> None:
    protocol = load_adaptive_learning_protocol(PROTOCOL_PATH)

    assert protocol.update_timing == "after-terminal-hand-v1"
    assert protocol.evidence_batches_per_hand == 1
    assert not protocol.same_hand_adaptation
    assert protocol.first_effective_hand_offset == 1


def test_evidence_contract_is_public_only_and_reuses_calibration_v2() -> None:
    protocol = load_adaptive_learning_protocol(PROTOCOL_PATH)

    assert protocol.opportunity_definition_version == "calibration-opportunities-v2"
    assert protocol.public_action_trace_version == "public-action-trace-v1"
    assert protocol.agent_observation_version == "agent-observation-v2"
    assert protocol.evidence_source == "public-action-trace-only"
    assert protocol.hidden_information_inputs == ()
    assert protocol.showdown_hole_cards_used is False
    assert protocol.framework_timeout_resolutions_are_behavior is False


def test_model_persists_within_run_but_resets_between_trajectory_runs() -> None:
    protocol = load_adaptive_learning_protocol(PROTOCOL_PATH)

    assert protocol.persist_between_hands_within_run
    assert protocol.reset_at_independent_run_start
    assert protocol.reset_at_opponent_identity_change
    assert not protocol.share_state_between_position_runs


def test_position_pairing_uses_separate_symmetric_adaptation_histories() -> None:
    protocol = load_adaptive_learning_protocol(PROTOCOL_PATH)

    assert protocol.position_pairing_version == "adaptive-position-pairing-v1"
    assert protocol.position_runs == ("A-BTN_vs_B-BB", "B-BTN_vs_A-BB")
    assert protocol.position_pairing_chance_relation == "identical-defined-card-realisations"
    assert protocol.position_run_histories == "separately-initialized"


def test_statistical_unit_and_adaptation_horizons_are_frozen() -> None:
    protocol = load_adaptive_learning_protocol(PROTOCOL_PATH)

    assert protocol.statistical_unit == "trajectory_cluster"
    assert protocol.bootstrap_unit == "trajectory_cluster"
    assert protocol.hand_level_independence is False
    assert protocol.adaptation_horizons == (10, 20, 50, 100, 500)
    assert protocol.segments == ((1, 20), (21, 100), (101, 500))
    assert protocol.primary_h4_horizon is None


def test_learning_and_evidence_ingestion_consume_no_rng() -> None:
    protocol = load_adaptive_learning_protocol(PROTOCOL_PATH)

    assert protocol.posterior_update_rng_draws == 0
    assert protocol.evidence_classifier_rng_draws == 0
    assert protocol.rng_policy == "no-learning-rng-v1"


def test_replay_contract_requires_before_evidence_after_hash_chain() -> None:
    protocol = load_adaptive_learning_protocol(PROTOCOL_PATH)

    assert protocol.replay_version == "adaptive-learning-replay-v1"
    assert protocol.replay_fields == (
        "model_state_before_hash",
        "public_evidence_counts",
        "model_state_after_hash",
    )


def test_semantic_tampering_is_rejected_even_after_outer_rehash() -> None:
    raw = _load_raw()
    payload = copy.deepcopy(raw["payload"])
    assert isinstance(payload, dict)
    priors = payload["priors"]
    assert isinstance(priors, dict)
    vpip = priors["vpip"]
    assert isinstance(vpip, dict)
    vpip["alpha"] = 2

    rehashed = _rehash(payload)

    with pytest.raises(AdaptiveLearningProtocolError, match="priors"):
        validate_adaptive_learning_payload(rehashed)


def test_protocol_file_is_already_canonical_lf_bytes() -> None:
    raw = PROTOCOL_PATH.read_bytes()

    assert b"\r" not in raw
    assert raw == canonical_file_bytes(json.loads(raw))
