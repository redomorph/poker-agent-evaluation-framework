"""Frozen lifecycle contract for the first runner-managed adaptive agent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTIVE_LIFECYCLE_PROTOCOL_ID = "phase2-adaptive-lifecycle-v1"
ADAPTIVE_LIFECYCLE_PROTOCOL_VERSION = "adaptive-lifecycle-v1"
ADAPTIVE_LIFECYCLE_FILE_VERSION = "frozen-adaptive-lifecycle-file-v1"
ADAPTIVE_LIFECYCLE_PROTOCOL_HASH = "bc7f26c66861a37dec8c544863a9e97fbcf33e07176fa2733e99f5426bf10edc"


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_file(payload: object) -> bytes:
    return _canonical(payload) + b"\n"


def _expected_payload() -> dict[str, Any]:
    return {
        "protocol_id": ADAPTIVE_LIFECYCLE_PROTOCOL_ID,
        "protocol_version": ADAPTIVE_LIFECYCLE_PROTOCOL_VERSION,
        "status": "frozen",
        "scope": "runner-managed-adaptive-learning-lifecycle",
        "dependencies": {
            "adaptive_learning_protocol": "adaptive-learning-protocol-v1",
            "adaptive_learning_protocol_hash": "03f8ab5b2b35c86910d46df1d5187c306814e427b3f514eb85748a9a424adf5c",
            "posterior_to_range": "posterior-to-range-v1",
            "posterior_to_range_hash": "af19b040f9049ecd3a3a12615dcbe049e1d1f859546681487176238e9d2ed668",
            "adaptive_policy": "adaptive-hand-strength-policy-v1",
            "adaptive_policy_hash": "76b6c561c2cb3715cc6352520dfbd3b0b01ce219a1b37d1a0acb62bb6ce10e4a",
            "opponent_model": "bayesian-opponent-model-v1",
            "opportunity_definition": "calibration-opportunities-v2",
        },
        "agent": {
            "version": "adaptive-hand-strength-agent-v1",
            "initial_state": "frozen-beta-1-1-prior",
            "decision_mechanism": "adaptive-hand-strength-policy-v1",
            "action_rng_draws": 0,
        },
        "runner": {
            "adaptive_statistical_unit": "trajectory_cluster",
            "reset_before_first_position_run": True,
            "reset_before_second_position_run": True,
            "share_state_between_position_runs": False,
            "update_timing": "after-normal-terminal-hand-v1",
            "same_hand_adaptation": False,
            "stable_bot_identity": True,
        },
        "evidence": {
            "classifier": "calibration-opportunities-v2",
            "source": "persisted-public-action-trace-v1",
            "normal_timeout_free_hand_required": True,
            "timeout_altered_hand_policy": "invalidate-adaptive-trajectory-v1",
            "timeout_resolution_is_behavior": False,
            "failed_hand_update": False,
            "replacement_seed": False,
        },
        "persistence": {
            "transition_version": "adaptive-learning-transition-v1",
            "per_hand_fields": [
                "learner_bot_id",
                "opponent_bot_id",
                "trajectory_hand_index",
                "model_state_before_hash",
                "public_evidence_counts",
                "model_state_after_hash",
                "transition_hash",
            ],
            "recorded_hand_field": "adaptive_learning_events",
            "replay_reclassifies_public_evidence": True,
        },
        "non_goals": [
            "mid-hand-learning",
            "timeout-action-learning",
            "postflop-behavior-policy-adjustment",
            "h4-experiment-design",
            "mcts-semantics",
        ],
    }


@dataclass(frozen=True)
class AdaptiveLifecycleProtocol:
    protocol_id: str
    protocol_version: str
    payload_hash: str


def load_adaptive_lifecycle_protocol(path: Path) -> AdaptiveLifecycleProtocol:
    raw = path.read_bytes()
    document = json.loads(raw)
    if raw != _canonical_file(document):
        raise ValueError("adaptive lifecycle protocol file is not canonical JSON/LF")
    if set(document) != {"file_version", "payload", "payload_hash"}:
        raise ValueError("invalid adaptive lifecycle protocol document keys")
    if document["file_version"] != ADAPTIVE_LIFECYCLE_FILE_VERSION:
        raise ValueError("invalid adaptive lifecycle protocol file version")
    payload = document["payload"]
    payload_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    if document["payload_hash"] != payload_hash:
        raise ValueError("adaptive lifecycle protocol payload hash mismatch")
    if payload_hash != ADAPTIVE_LIFECYCLE_PROTOCOL_HASH:
        raise ValueError("adaptive lifecycle protocol hash differs from frozen v1")
    if payload != _expected_payload():
        raise ValueError("adaptive lifecycle protocol semantics differ from frozen v1")
    return AdaptiveLifecycleProtocol(
        protocol_id=payload["protocol_id"],
        protocol_version=payload["protocol_version"],
        payload_hash=payload_hash,
    )
