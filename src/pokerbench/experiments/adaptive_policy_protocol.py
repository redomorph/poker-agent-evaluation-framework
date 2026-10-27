"""Loader/validator for the frozen Phase-2 adaptive hand-strength policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTIVE_POLICY_PROTOCOL_ID = "phase2-adaptive-hand-strength-policy-v1"
ADAPTIVE_POLICY_PROTOCOL_VERSION = "adaptive-hand-strength-policy-v1"
ADAPTIVE_POLICY_PROTOCOL_FILE_VERSION = "frozen-adaptive-hand-strength-policy-file-v1"


@dataclass(frozen=True)
class AdaptivePolicyProtocol:
    protocol_id: str
    protocol_version: str
    payload_hash: str


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_file(payload: object) -> bytes:
    return _canonical(payload) + b"\n"


def expected_adaptive_policy_payload() -> dict[str, Any]:
    return {
        "protocol_id": ADAPTIVE_POLICY_PROTOCOL_ID,
        "protocol_version": ADAPTIVE_POLICY_PROTOCOL_VERSION,
        "status": "frozen",
        "scope": "range-conditioned-equity-and-passive-action-only",
        "dependencies": {
            "adaptive_learning_protocol": "adaptive-learning-protocol-v1",
            "adaptive_learning_protocol_hash": "03f8ab5b2b35c86910d46df1d5187c306814e427b3f514eb85748a9a424adf5c",
            "bayesian_opponent_model": "bayesian-opponent-model-v1",
            "posterior_to_range": "posterior-to-range-v1",
            "posterior_to_range_hash": "af19b040f9049ecd3a3a12615dcbe049e1d1f859546681487176238e9d2ed668",
            "preflop_strength_reference_hash": "cb110768ad74a9747064ba25e88af635055b1491a58deae8f09112aae801fc1e",
            "control_policy": "hand-strength-baseline-v1",
            "pot_odds": "hand-strength-pot-odds-v1",
        },
        "range_conditioned_equity": {
            "version": "posterior-range-equity-v1",
            "range_version": "posterior-to-range-v1",
            "opponent_combo_distribution": "uniform-over-surviving-selected-combinations",
            "runout_distribution": "uniform-without-replacement",
            "exact_joint_outcome_limit": 500000,
            "monte_carlo_samples": 10000,
            "equity_ppm_scale": 1000000,
            "equity_ppm_rounding": "nearest-half-up-v1",
            "tie_value": "one-half",
        },
        "rng": {
            "version": "posterior-range-equity-rng-v1",
            "domain": "range-equity",
            "identity_fields": [
                "master_seed",
                "bot_id",
                "hand_id",
                "decision_index",
            ],
            "range_hash_in_rng_identity": False,
            "action_rng_draws": 0,
            "posterior_sampling": False,
        },
        "action_policy": {
            "no_wager": "check",
            "facing_wager": "call_if_equity_ppm_gte_pot_odds_ppm_else_fold",
            "aggressive_actions": False,
            "legal_fallback": "no-aggressive-substitution",
            "pot_odds_formula": "call_cost/(pot+call_cost)",
            "pot_odds_ppm_rounding": "nearest-half-up-v1",
            "control_difference": "opponent-range-only",
        },
        "behavior_metrics_action_inactive": [
            "cbet_freq",
            "postflop_aggressive_share",
            "fold_to_cbet",
        ],
        "replay": {
            "persist_model_state_hash": True,
            "persist_range_hash": True,
            "persist_range_signal": True,
            "persist_range_width_ppm": True,
            "persist_legal_combo_count": True,
            "persist_equity_result": True,
            "persist_pot_odds_ppm": True,
            "persist_decision_hash": True,
        },
        "non_goals": [
            "terminal-hand-model-update-integration",
            "postflop-action-conditioned-range",
            "behavioral-threshold-adjustment",
            "aggressive-adaptive-actions",
            "h4-experiment-design",
            "mcts-semantics",
        ],
    }


def load_adaptive_policy_protocol(path: Path) -> AdaptivePolicyProtocol:
    raw = path.read_bytes()
    doc = json.loads(raw)
    if raw != _canonical_file(doc):
        raise ValueError("adaptive policy protocol is not canonical JSON/LF")
    if set(doc) != {"file_version", "payload", "payload_hash"}:
        raise ValueError("invalid adaptive policy protocol document keys")
    if doc["file_version"] != ADAPTIVE_POLICY_PROTOCOL_FILE_VERSION:
        raise ValueError("invalid adaptive policy protocol file version")
    payload = doc["payload"]
    payload_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    if doc["payload_hash"] != payload_hash:
        raise ValueError("adaptive policy protocol payload hash mismatch")
    if payload != expected_adaptive_policy_payload():
        raise ValueError("adaptive policy protocol semantics differ from v1")
    return AdaptivePolicyProtocol(
        protocol_id=payload["protocol_id"],
        protocol_version=payload["protocol_version"],
        payload_hash=payload_hash,
    )
