"""Frozen Phase-2 adaptive observation and learning protocol.

This module validates the machine-readable protocol that sits between the
Bayesian opponent model and any later adaptive action policy.  It intentionally
contains no posterior-to-range or posterior-to-action mapping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTIVE_LEARNING_PROTOCOL_ID = "phase2-adaptive-learning-protocol-v1"
ADAPTIVE_LEARNING_PROTOCOL_VERSION = "adaptive-learning-protocol-v1"
ADAPTIVE_LEARNING_PROTOCOL_FILE_VERSION = (
    "frozen-adaptive-learning-protocol-file-v1"
)

_EXPECTED_PRIOR_METRICS = (
    "vpip",
    "pfr",
    "three_bet_freq",
    "cbet_freq",
    "postflop_aggressive_share",
    "fold_to_cbet",
)


class AdaptiveLearningProtocolError(ValueError):
    """Raised when the frozen adaptive-learning protocol is invalid."""


@dataclass(frozen=True)
class AdaptiveLearningProtocol:
    """Validated immutable view of the v1 adaptive-learning freeze."""

    protocol_id: str
    protocol_version: str
    status: str
    scope: str
    priors: dict[str, tuple[int, int]]
    prior_selection_basis: str
    opportunity_definition_version: str
    public_action_trace_version: str
    agent_observation_version: str
    evidence_source: str
    update_timing: str
    evidence_batches_per_hand: int
    same_hand_adaptation: bool
    first_effective_hand_offset: int
    hidden_information_inputs: tuple[str, ...]
    showdown_hole_cards_used: bool
    framework_timeout_resolutions_are_behavior: bool
    persist_between_hands_within_run: bool
    reset_at_independent_run_start: bool
    reset_at_opponent_identity_change: bool
    share_state_between_position_runs: bool
    position_pairing_version: str
    position_runs: tuple[str, str]
    position_pairing_chance_relation: str
    position_run_histories: str
    statistical_unit: str
    bootstrap_unit: str
    hand_level_independence: bool
    adaptation_horizons: tuple[int, ...]
    segments: tuple[tuple[int, int], ...]
    primary_h4_horizon: int | None
    posterior_update_rng_draws: int
    evidence_classifier_rng_draws: int
    rng_policy: str
    replay_version: str
    replay_fields: tuple[str, ...]
    payload_hash: str


def canonical_json_bytes(payload: object) -> bytes:
    """Return canonical JSON bytes without a trailing newline."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_file_bytes(payload: object) -> bytes:
    """Return the canonical on-disk JSON representation with LF newline."""

    return canonical_json_bytes(payload) + b"\n"


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdaptiveLearningProtocolError(f"{context} must be an object")
    return value


def _require_exact_keys(
    payload: dict[str, Any], expected: set[str], context: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise AdaptiveLearningProtocolError(
            f"{context} keys differ: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise AdaptiveLearningProtocolError(
            f"{context} must equal {expected!r}; got {actual!r}"
        )


def validate_adaptive_learning_payload(
    document: dict[str, Any],
) -> AdaptiveLearningProtocol:
    """Validate the full frozen protocol document and return its typed view."""

    document = _require_object(document, "protocol document")
    _require_exact_keys(
        document,
        {"file_version", "payload", "payload_hash"},
        "protocol document",
    )
    _require_equal(
        document["file_version"],
        ADAPTIVE_LEARNING_PROTOCOL_FILE_VERSION,
        "file_version",
    )

    payload = _require_object(document["payload"], "payload")
    expected_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    _require_equal(document["payload_hash"], expected_hash, "payload_hash")

    _require_exact_keys(
        payload,
        {
            "protocol_id",
            "protocol_version",
            "status",
            "scope",
            "versions",
            "priors",
            "prior_selection_basis",
            "evidence",
            "state",
            "position_pairing",
            "statistics",
            "rng",
            "replay",
            "non_goals",
        },
        "payload",
    )
    _require_equal(
        payload["protocol_id"], ADAPTIVE_LEARNING_PROTOCOL_ID, "protocol_id"
    )
    _require_equal(
        payload["protocol_version"],
        ADAPTIVE_LEARNING_PROTOCOL_VERSION,
        "protocol_version",
    )
    _require_equal(payload["status"], "frozen", "status")
    _require_equal(
        payload["scope"], "observation-and-learning-only", "scope"
    )

    versions = _require_object(payload["versions"], "versions")
    expected_versions = {
        "opponent_model": "bayesian-opponent-model-v1",
        "opponent_prior": "bayesian-opponent-prior-v1",
        "opportunity_definition": "calibration-opportunities-v2",
        "public_action_trace": "public-action-trace-v1",
        "agent_observation": "agent-observation-v2",
    }
    _require_equal(versions, expected_versions, "versions")

    priors_payload = _require_object(payload["priors"], "priors")
    if set(priors_payload) != set(_EXPECTED_PRIOR_METRICS):
        raise AdaptiveLearningProtocolError(
            "priors must contain exactly the six frozen metrics"
        )
    priors: dict[str, tuple[int, int]] = {}
    for metric in _EXPECTED_PRIOR_METRICS:
        prior = _require_object(priors_payload[metric], f"priors.{metric}")
        _require_exact_keys(prior, {"alpha", "beta"}, f"priors.{metric}")
        _require_equal(prior["alpha"], 1, f"priors.{metric}.alpha")
        _require_equal(prior["beta"], 1, f"priors.{metric}.beta")
        priors[metric] = (1, 1)
    _require_equal(
        payload["prior_selection_basis"],
        "independent-uniform-beta-1-1-v1",
        "prior_selection_basis",
    )

    evidence = _require_object(payload["evidence"], "evidence")
    expected_evidence = {
        "source": "public-action-trace-only",
        "classification_contract": "calibration-opportunities-v2",
        "update_timing": "after-terminal-hand-v1",
        "batches_per_hand": 1,
        "same_hand_adaptation": False,
        "first_effective_hand_offset": 1,
        "hidden_information_inputs": [],
        "showdown_hole_cards_used": False,
        "framework_timeout_resolutions_are_behavior": False,
    }
    _require_equal(evidence, expected_evidence, "evidence")

    state = _require_object(payload["state"], "state")
    expected_state = {
        "persist_between_hands_within_run": True,
        "reset_at_independent_run_start": True,
        "reset_at_opponent_identity_change": True,
        "share_state_between_position_runs": False,
        "opponent_identity_within_run": "fixed",
    }
    _require_equal(state, expected_state, "state")

    pairing = _require_object(payload["position_pairing"], "position_pairing")
    expected_pairing = {
        "version": "adaptive-position-pairing-v1",
        "runs": ["A-BTN_vs_B-BB", "B-BTN_vs_A-BB"],
        "chance_relation": "identical-defined-card-realisations",
        "adaptation_histories": "separately-initialized",
    }
    _require_equal(pairing, expected_pairing, "position_pairing")

    statistics = _require_object(payload["statistics"], "statistics")
    expected_statistics = {
        "statistical_unit": "trajectory_cluster",
        "bootstrap_unit": "trajectory_cluster",
        "hand_level_independence": False,
        "adaptation_horizons": [10, 20, 50, 100, 500],
        "segments": [[1, 20], [21, 100], [101, 500]],
        "primary_h4_horizon": None,
    }
    _require_equal(statistics, expected_statistics, "statistics")

    rng = _require_object(payload["rng"], "rng")
    expected_rng = {
        "policy": "no-learning-rng-v1",
        "posterior_update_draws": 0,
        "evidence_classifier_draws": 0,
    }
    _require_equal(rng, expected_rng, "rng")

    replay = _require_object(payload["replay"], "replay")
    expected_replay = {
        "version": "adaptive-learning-replay-v1",
        "per_hand_fields": [
            "model_state_before_hash",
            "public_evidence_counts",
            "model_state_after_hash",
        ],
    }
    _require_equal(replay, expected_replay, "replay")

    expected_non_goals = [
        "posterior-to-range-mapping",
        "posterior-to-action-mapping",
        "dynamic-preflop-range",
        "dynamic-postflop-policy",
        "mcts-semantics",
        "h4-primary-horizon",
        "h4-power-or-precision-plan",
    ]
    _require_equal(payload["non_goals"], expected_non_goals, "non_goals")

    return AdaptiveLearningProtocol(
        protocol_id=payload["protocol_id"],
        protocol_version=payload["protocol_version"],
        status=payload["status"],
        scope=payload["scope"],
        priors=priors,
        prior_selection_basis=payload["prior_selection_basis"],
        opportunity_definition_version=versions["opportunity_definition"],
        public_action_trace_version=versions["public_action_trace"],
        agent_observation_version=versions["agent_observation"],
        evidence_source=evidence["source"],
        update_timing=evidence["update_timing"],
        evidence_batches_per_hand=evidence["batches_per_hand"],
        same_hand_adaptation=evidence["same_hand_adaptation"],
        first_effective_hand_offset=evidence["first_effective_hand_offset"],
        hidden_information_inputs=tuple(evidence["hidden_information_inputs"]),
        showdown_hole_cards_used=evidence["showdown_hole_cards_used"],
        framework_timeout_resolutions_are_behavior=evidence[
            "framework_timeout_resolutions_are_behavior"
        ],
        persist_between_hands_within_run=state[
            "persist_between_hands_within_run"
        ],
        reset_at_independent_run_start=state["reset_at_independent_run_start"],
        reset_at_opponent_identity_change=state[
            "reset_at_opponent_identity_change"
        ],
        share_state_between_position_runs=state[
            "share_state_between_position_runs"
        ],
        position_pairing_version=pairing["version"],
        position_runs=tuple(pairing["runs"]),
        position_pairing_chance_relation=pairing["chance_relation"],
        position_run_histories=pairing["adaptation_histories"],
        statistical_unit=statistics["statistical_unit"],
        bootstrap_unit=statistics["bootstrap_unit"],
        hand_level_independence=statistics["hand_level_independence"],
        adaptation_horizons=tuple(statistics["adaptation_horizons"]),
        segments=tuple(tuple(segment) for segment in statistics["segments"]),
        primary_h4_horizon=statistics["primary_h4_horizon"],
        posterior_update_rng_draws=rng["posterior_update_draws"],
        evidence_classifier_rng_draws=rng["evidence_classifier_draws"],
        rng_policy=rng["policy"],
        replay_version=replay["version"],
        replay_fields=tuple(replay["per_hand_fields"]),
        payload_hash=document["payload_hash"],
    )


def load_adaptive_learning_protocol(path: Path) -> AdaptiveLearningProtocol:
    """Load, canonical-byte validate, and semantically validate a protocol."""

    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdaptiveLearningProtocolError("protocol is not valid JSON") from exc
    if raw != canonical_file_bytes(document):
        raise AdaptiveLearningProtocolError(
            "protocol file is not canonical UTF-8 JSON with LF newline"
        )
    return validate_adaptive_learning_payload(document)
