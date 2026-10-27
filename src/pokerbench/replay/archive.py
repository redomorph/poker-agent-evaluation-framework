"""
Durable deterministic match-replay persistence.

This module preserves the established replay API while providing:

- deterministic MatchRecord serialization;
- SHA-256 integrity checking;
- deterministic replay through HandReplay;
- exceptional-hand metadata;
- timeout metadata;
- compatibility module-level load/save functions;
- directory-oriented ReplayArchive access.

A missing replay file is a lookup failure, not an integrity failure.
FileNotFoundError is therefore deliberately preserved.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, TypeVar, overload

from pokerbench.engine.config import (
    Action,
    GameConfig,
    GameType,
    canonical_config_dict,
    config_hash,
)
from pokerbench.engine.constants import CHIP_UNIT
from pokerbench.engine.state import GameState
from pokerbench.replay.replay import HandReplay, ReplayResult


MATCH_RECORD_SCHEMA_VERSION = "match-record-v1"
MATCH_RECORD_FILE_VERSION = "match-record-file-v1"

T = TypeVar("T")


class ReplayIntegrityError(ValueError):
    """Raised when persisted replay data fails integrity validation."""


class HandTermination(StrEnum):
    """Classification of an attempted hand's termination."""

    NORMAL = "normal"
    TIMEOUT_FORFEIT = "timeout_forfeit"
    AGENT_ERROR = "agent_error"
    ILLEGAL_ACTION = "illegal_action"
    ENGINE_ERROR = "engine_error"


@dataclass(frozen=True, init=False)
class RecordedTimeoutEvent:
    """
    Persisted timeout information for one agent decision.

    Seat identity and stable bot identity are distinct:

        player
            Seat/player index in this concrete hand (0 or 1).

        bot_id
            Stable agent identity used by the RNG namespace (0 or 1).

    They may differ in mirrored position-compensation conditions such as
    B_BTN, where the stable bots swap seats.

    Likewise, action_index and decision_index are stored independently.
    Older callers that provide only one of each pair remain supported through
    deterministic fallback values.
    """

    player: int
    bot_id: int

    action_index: int | None
    decision_index: int | None

    elapsed_ns: int | None
    budget_ms: float | None

    timeout_policy: str
    resolution: str | None
    applied_action: Action | None

    timeout_policy_version: str

    def __init__(
        self,
        *,
        player: int | None = None,
        bot_id: int | None = None,
        action_index: int | None = None,
        decision_index: int | None = None,
        elapsed_ns: int | None = None,
        elapsed_ms: float | None = None,
        budget_ms: float | None = None,
        time_budget_ms: float | None = None,
        timeout_policy: str | None = None,
        policy: str | None = None,
        resolution: str | None = None,
        applied_action: Action | None = None,
        action: Action | None = None,
        timeout_policy_version: str | None = None,
        policy_version: str | None = None,
    ) -> None:
        if player is None and bot_id is None:
            raise TypeError(
                "RecordedTimeoutEvent requires at least "
                "'player' or 'bot_id'"
            )

        resolved_player = (
            player
            if player is not None
            else bot_id
        )
        resolved_bot_id = (
            bot_id
            if bot_id is not None
            else player
        )

        assert resolved_player is not None
        assert resolved_bot_id is not None

        _validate_binary_identity(
            resolved_player,
            "player",
        )
        _validate_binary_identity(
            resolved_bot_id,
            "bot_id",
        )

        if (
            action_index is None
            and decision_index is not None
        ):
            resolved_action_index = (
                decision_index
            )
        else:
            resolved_action_index = (
                action_index
            )

        if (
            decision_index is None
            and action_index is not None
        ):
            resolved_decision_index = (
                action_index
            )
        else:
            resolved_decision_index = (
                decision_index
            )

        _validate_optional_non_negative_int(
            resolved_action_index,
            "action_index",
        )
        _validate_optional_non_negative_int(
            resolved_decision_index,
            "decision_index",
        )

        resolved_elapsed_ns = (
            _resolve_elapsed_ns(
                elapsed_ns=elapsed_ns,
                elapsed_ms=elapsed_ms,
            )
        )

        resolved_budget_ms = (
            _resolve_alias(
                primary=budget_ms,
                alias=time_budget_ms,
                primary_name="budget_ms",
                alias_name="time_budget_ms",
            )
        )

        if resolved_budget_ms is not None:
            _require_finite_positive(
                resolved_budget_ms,
                "budget_ms",
            )

        resolved_policy = (
            _resolve_alias(
                primary=timeout_policy,
                alias=policy,
                primary_name="timeout_policy",
                alias_name="policy",
            )
        )

        if resolved_policy is None:
            raise TypeError(
                "RecordedTimeoutEvent requires "
                "'timeout_policy' "
                "(or compatibility alias 'policy')"
            )

        if not isinstance(
            resolved_policy,
            str,
        ):
            raise TypeError(
                "timeout_policy must be a string"
            )

        if not resolved_policy.strip():
            raise ValueError(
                "timeout_policy must not be empty"
            )

        if resolution is not None:
            if not isinstance(
                resolution,
                str,
            ):
                raise TypeError(
                    "resolution must be a string or None"
                )

            if not resolution.strip():
                raise ValueError(
                    "resolution must not be empty "
                    "when provided"
                )

        resolved_action = (
            _resolve_alias(
                primary=applied_action,
                alias=action,
                primary_name="applied_action",
                alias_name="action",
            )
        )

        resolved_policy_version = (
            _resolve_alias(
                primary=timeout_policy_version,
                alias=policy_version,
                primary_name="timeout_policy_version",
                alias_name="policy_version",
            )
        )

        if resolved_policy_version is None:
            resolved_policy_version = (
                "timeout-policy-v1"
            )

        if not isinstance(
            resolved_policy_version,
            str,
        ):
            raise TypeError(
                "timeout_policy_version "
                "must be a string"
            )

        if not (
            resolved_policy_version
            .strip()
        ):
            raise ValueError(
                "timeout_policy_version "
                "must not be empty"
            )

        object.__setattr__(
            self,
            "player",
            resolved_player,
        )
        object.__setattr__(
            self,
            "bot_id",
            resolved_bot_id,
        )
        object.__setattr__(
            self,
            "action_index",
            resolved_action_index,
        )
        object.__setattr__(
            self,
            "decision_index",
            resolved_decision_index,
        )
        object.__setattr__(
            self,
            "elapsed_ns",
            resolved_elapsed_ns,
        )
        object.__setattr__(
            self,
            "budget_ms",
            (
                float(
                    resolved_budget_ms
                )
                if resolved_budget_ms
                is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "timeout_policy",
            resolved_policy,
        )
        object.__setattr__(
            self,
            "resolution",
            resolution,
        )
        object.__setattr__(
            self,
            "applied_action",
            resolved_action,
        )
        object.__setattr__(
            self,
            "timeout_policy_version",
            resolved_policy_version,
        )

    @property
    def elapsed_ms(
        self,
    ) -> float | None:
        """Derived compatibility view of elapsed_ns."""
        if self.elapsed_ns is None:
            return None

        return (
            self.elapsed_ns
            / 1_000_000.0
        )

    @property
    def time_budget_ms(
        self,
    ) -> float | None:
        """Compatibility alias for budget_ms."""
        return self.budget_ms

    @property
    def policy(
        self,
    ) -> str:
        """Compatibility alias for timeout_policy."""
        return self.timeout_policy

    @property
    def action(
        self,
    ) -> Action | None:
        """Compatibility alias for applied_action."""
        return self.applied_action

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return deterministic JSON-compatible timeout metadata."""
        return {
            "player": (
                self.player
            ),
            "bot_id": (
                self.bot_id
            ),
            "action_index": (
                self.action_index
            ),
            "decision_index": (
                self.decision_index
            ),
            "elapsed_ns": (
                self.elapsed_ns
            ),
            "budget_ms": (
                self.budget_ms
            ),
            "timeout_policy": (
                self.timeout_policy
            ),
            "resolution": (
                self.resolution
            ),
            "timeout_policy_version": (
                self.timeout_policy_version
            ),
            "applied_action": (
                self.applied_action.value
                if self.applied_action
                is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "RecordedTimeoutEvent":
        """
        Reconstruct timeout metadata.

        Older records without bot_id/decision_index remain readable by
        falling back to player/action_index respectively.
        """
        raw_player = (
            data.get(
                "player"
            )
        )
        raw_bot_id = (
            data.get(
                "bot_id"
            )
        )

        if (
            raw_player is None
            and raw_bot_id is None
        ):
            raise ValueError(
                "Timeout record has neither "
                "player nor bot_id"
            )

        raw_action_index = (
            data.get(
                "action_index"
            )
        )
        raw_decision_index = (
            data.get(
                "decision_index"
            )
        )

        raw_elapsed_ns = (
            data.get(
                "elapsed_ns"
            )
        )
        raw_elapsed_ms = (
            data.get(
                "elapsed_ms"
            )
        )

        raw_budget_ms = (
            data.get(
                "budget_ms",
                data.get(
                    "time_budget_ms"
                ),
            )
        )

        raw_policy = (
            data.get(
                "timeout_policy",
                data.get(
                    "policy"
                ),
            )
        )

        if raw_policy is None:
            raise ValueError(
                "Timeout record has no "
                "timeout_policy"
            )

        raw_action = (
            data.get(
                "applied_action",
                data.get(
                    "action"
                ),
            )
        )

        return cls(
            player=(
                int(raw_player)
                if raw_player
                is not None
                else None
            ),
            bot_id=(
                int(raw_bot_id)
                if raw_bot_id
                is not None
                else None
            ),
            action_index=(
                int(
                    raw_action_index
                )
                if raw_action_index
                is not None
                else None
            ),
            decision_index=(
                int(
                    raw_decision_index
                )
                if raw_decision_index
                is not None
                else None
            ),
            elapsed_ns=(
                int(
                    raw_elapsed_ns
                )
                if raw_elapsed_ns
                is not None
                else None
            ),
            elapsed_ms=(
                float(
                    raw_elapsed_ms
                )
                if raw_elapsed_ms
                is not None
                else None
            ),
            budget_ms=(
                float(
                    raw_budget_ms
                )
                if raw_budget_ms
                is not None
                else None
            ),
            timeout_policy=str(
                raw_policy
            ),
            resolution=(
                str(
                    data[
                        "resolution"
                    ]
                )
                if data.get(
                    "resolution"
                )
                is not None
                else None
            ),
            applied_action=(
                Action(
                    raw_action
                )
                if raw_action
                is not None
                else None
            ),
            timeout_policy_version=str(
                data.get(
                    "timeout_policy_version",
                    data.get(
                        "policy_version",
                        "timeout-policy-v1",
                    ),
                )
            ),
        )

ADAPTIVE_LEARNING_TRANSITION_VERSION = "adaptive-learning-transition-v1"


def _adaptive_transition_canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


_ADAPTIVE_EVIDENCE_COUNT_FIELDS = (
    "vpip_opportunities",
    "vpip_events",
    "pfr_opportunities",
    "pfr_events",
    "three_bet_opportunities",
    "three_bet_events",
    "cbet_opportunities",
    "cbet_events",
    "postflop_aggressive_actions",
    "postflop_passive_actions",
    "fold_to_cbet_opportunities",
    "fold_to_cbet_events",
)


def _validate_adaptive_evidence_mapping(payload: Mapping[str, Any]) -> None:
    expected = set(_ADAPTIVE_EVIDENCE_COUNT_FIELDS) | {
        "opportunity_definition_version"
    }
    if set(payload) != expected:
        raise ValueError("invalid adaptive public evidence keys")
    if payload["opportunity_definition_version"] != "calibration-opportunities-v2":
        raise ValueError("unsupported adaptive opportunity-definition version")
    for field_name in _ADAPTIVE_EVIDENCE_COUNT_FIELDS:
        value = payload[field_name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"adaptive evidence field {field_name} must be a non-negative integer"
            )
    for prefix in (
        "vpip",
        "pfr",
        "three_bet",
        "cbet",
        "fold_to_cbet",
    ):
        if payload[f"{prefix}_events"] > payload[f"{prefix}_opportunities"]:
            raise ValueError(
                f"adaptive evidence {prefix} events cannot exceed opportunities"
            )


def _adaptive_evidence_payload(evidence: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in _ADAPTIVE_EVIDENCE_COUNT_FIELDS + (
        "opportunity_definition_version",
    ):
        if not hasattr(evidence, field_name):
            raise TypeError(f"evidence is missing {field_name}")
        payload[field_name] = getattr(evidence, field_name)
    _validate_adaptive_evidence_mapping(payload)
    return payload


@dataclass(frozen=True)
class RecordedAdaptiveLearningEvent:
    """One replay-verifiable post-terminal Bayesian learning transition."""

    learner_bot_id: int
    opponent_bot_id: int
    trajectory_hand_index: int
    model_state_before_hash: str
    public_evidence_counts: Mapping[str, Any]
    model_state_after_hash: str
    transition_hash: str
    version: str = ADAPTIVE_LEARNING_TRANSITION_VERSION

    def __post_init__(self) -> None:
        if self.version != ADAPTIVE_LEARNING_TRANSITION_VERSION:
            raise ValueError("unsupported adaptive learning transition version")
        if self.learner_bot_id not in (0, 1) or self.opponent_bot_id not in (0, 1):
            raise ValueError("adaptive learning bot IDs must be 0 or 1")
        if self.learner_bot_id == self.opponent_bot_id:
            raise ValueError("learner and opponent bot IDs must be different")
        if (
            not isinstance(self.trajectory_hand_index, int)
            or isinstance(self.trajectory_hand_index, bool)
            or self.trajectory_hand_index < 0
        ):
            raise ValueError("trajectory_hand_index must be a non-negative integer")
        for name, value in (
            ("model_state_before_hash", self.model_state_before_hash),
            ("model_state_after_hash", self.model_state_after_hash),
            ("transition_hash", self.transition_hash),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a 64-character SHA-256 hex string")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must be hexadecimal") from exc
        if not isinstance(self.public_evidence_counts, Mapping):
            raise TypeError("public_evidence_counts must be a mapping")
        _validate_adaptive_evidence_mapping(self.public_evidence_counts)

    @classmethod
    def from_transition(
        cls,
        *,
        learner_bot_id: int,
        opponent_bot_id: int,
        trajectory_hand_index: int,
        before: Any,
        evidence: Any,
        after: Any,
    ) -> "RecordedAdaptiveLearningEvent":
        before_hash = getattr(before, "state_hash", None)
        after_hash = getattr(after, "state_hash", None)
        if not isinstance(before_hash, str) or not isinstance(after_hash, str):
            raise TypeError("before and after states must expose state_hash")
        evidence_payload = _adaptive_evidence_payload(evidence)
        core = {
            "version": ADAPTIVE_LEARNING_TRANSITION_VERSION,
            "learner_bot_id": learner_bot_id,
            "opponent_bot_id": opponent_bot_id,
            "trajectory_hand_index": trajectory_hand_index,
            "model_state_before_hash": before_hash,
            "public_evidence_counts": evidence_payload,
            "model_state_after_hash": after_hash,
        }
        transition_hash = hashlib.sha256(
            _adaptive_transition_canonical(core)
        ).hexdigest()
        return cls(
            learner_bot_id=learner_bot_id,
            opponent_bot_id=opponent_bot_id,
            trajectory_hand_index=trajectory_hand_index,
            model_state_before_hash=before_hash,
            public_evidence_counts=evidence_payload,
            model_state_after_hash=after_hash,
            transition_hash=transition_hash,
        )

    def _core_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "learner_bot_id": self.learner_bot_id,
            "opponent_bot_id": self.opponent_bot_id,
            "trajectory_hand_index": self.trajectory_hand_index,
            "model_state_before_hash": self.model_state_before_hash,
            "public_evidence_counts": dict(self.public_evidence_counts),
            "model_state_after_hash": self.model_state_after_hash,
        }

    def validate_hash(self) -> None:
        expected = hashlib.sha256(
            _adaptive_transition_canonical(self._core_dict())
        ).hexdigest()
        if self.transition_hash != expected:
            raise ValueError("adaptive learning transition hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        self.validate_hash()
        return {**self._core_dict(), "transition_hash": self.transition_hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecordedAdaptiveLearningEvent":
        expected = {
            "version",
            "learner_bot_id",
            "opponent_bot_id",
            "trajectory_hand_index",
            "model_state_before_hash",
            "public_evidence_counts",
            "model_state_after_hash",
            "transition_hash",
        }
        if set(data) != expected:
            raise ValueError("invalid adaptive learning transition keys")
        evidence = data["public_evidence_counts"]
        if not isinstance(evidence, Mapping):
            raise ValueError("public_evidence_counts must be an object")
        event = cls(
            version=str(data["version"]),
            learner_bot_id=int(data["learner_bot_id"]),
            opponent_bot_id=int(data["opponent_bot_id"]),
            trajectory_hand_index=int(data["trajectory_hand_index"]),
            model_state_before_hash=str(data["model_state_before_hash"]),
            public_evidence_counts=dict(evidence),
            model_state_after_hash=str(data["model_state_after_hash"]),
            transition_hash=str(data["transition_hash"]),
        )
        event.validate_hash()
        return event


@dataclass(frozen=True)
class RecordedReplayResult:
    """Result of replaying one persisted hand."""

    record: "RecordedHand"
    engine_replay: ReplayResult

    @property
    def actions(self) -> tuple[Action, ...]:
        """Return the exact replayed action sequence."""
        return self.engine_replay.actions

    @property
    def final_profit_bb(self) -> tuple[float, float] | None:
        """Return the recorded final profit in BB."""
        return self.record.terminal_profit_bb

    @property
    def initial_state(self) -> GameState:
        """Return the reconstructed initial engine state."""
        return self.engine_replay.initial_state

    @property
    def final_state(self) -> GameState:
        """Return the reconstructed final engine state."""
        return self.engine_replay.final_state

    @property
    def states(self) -> tuple[GameState, ...]:
        """Return the complete reconstructed state sequence."""
        return self.engine_replay.states


@dataclass(frozen=True)
class RecordedHand:
    """
    Durable representation of one attempted hand.

    actions contains only actions successfully submitted to the engine.

    For exceptional termination, acting_player_at_termination identifies the
    player whose decision caused or encountered the exceptional event.
    """

    config_id: str
    master_seed: int
    hand_index: int
    dealer: int
    seat0_bot_id: int
    seat1_bot_id: int
    actions: tuple[Action, ...]
    termination: HandTermination
    terminal_profit_bb: tuple[float, float] | None
    timeout_events: tuple[RecordedTimeoutEvent, ...] = ()
    adaptive_learning_events: tuple[RecordedAdaptiveLearningEvent, ...] = ()
    requested_action: Action | None = None
    legal_actions_at_termination: tuple[Action, ...] = ()
    acting_player_at_termination: int | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config_id, str):
            raise TypeError("config_id must be a string")
        if not self.config_id.strip():
            raise ValueError("config_id must not be empty")

        if (
            not isinstance(self.master_seed, int)
            or isinstance(self.master_seed, bool)
        ):
            raise TypeError("master_seed must be an integer")

        if (
            not isinstance(self.hand_index, int)
            or isinstance(self.hand_index, bool)
        ):
            raise TypeError("hand_index must be an integer")
        if self.hand_index < 0:
            raise ValueError("hand_index must be non-negative")

        if self.dealer not in (0, 1):
            raise ValueError("dealer must be 0 or 1")

        if self.seat0_bot_id not in (0, 1):
            raise ValueError("seat0_bot_id must be 0 or 1")
        if self.seat1_bot_id not in (0, 1):
            raise ValueError("seat1_bot_id must be 0 or 1")
        if self.seat0_bot_id == self.seat1_bot_id:
            raise ValueError("seat bot IDs must be different")

        if (
            self.acting_player_at_termination is not None
            and self.acting_player_at_termination not in (0, 1)
        ):
            raise ValueError(
                "acting_player_at_termination must be 0, 1, or None"
            )

        if self.terminal_profit_bb is not None:
            if len(self.terminal_profit_bb) != 2:
                raise ValueError(
                    "terminal_profit_bb must contain exactly two values"
                )

            for value in self.terminal_profit_bb:
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                ):
                    raise TypeError(
                        "terminal_profit_bb values must be numeric"
                    )
                if not math.isfinite(float(value)):
                    raise ValueError(
                        "terminal_profit_bb values must be finite"
                    )

            total_profit = (
                float(self.terminal_profit_bb[0])
                + float(self.terminal_profit_bb[1])
            )
            if not math.isclose(
                total_profit,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "terminal_profit_bb must be zero-sum"
                )

        if (
            self.termination
            in (
                HandTermination.NORMAL,
                HandTermination.TIMEOUT_FORFEIT,
            )
            and self.terminal_profit_bb is None
        ):
            raise ValueError(
                "Completed hands must record terminal_profit_bb"
            )

        if self.adaptive_learning_events:
            if self.termination is not HandTermination.NORMAL or self.timeout_events:
                raise ValueError(
                    "timeout-altered or exceptional hands cannot carry adaptive learning events"
                )
            for event in self.adaptive_learning_events:
                if not isinstance(event, RecordedAdaptiveLearningEvent):
                    raise TypeError(
                        "adaptive_learning_events must contain RecordedAdaptiveLearningEvent values"
                    )
                if event.trajectory_hand_index != self.hand_index:
                    raise ValueError(
                        "adaptive learning trajectory_hand_index must match hand_index"
                    )
                if {event.learner_bot_id, event.opponent_bot_id} != {
                    self.seat0_bot_id,
                    self.seat1_bot_id,
                }:
                    raise ValueError(
                        "adaptive learning bot identities must match the recorded hand"
                    )
                event.validate_hash()

    def replay(
        self,
        config: GameConfig,
    ) -> RecordedReplayResult:
        """
        Deterministically reconstruct this recorded hand.

        Agent-returned actions are stored in ``self.actions``. Actions applied
        by the framework because of a timeout policy are stored separately in
        ``timeout_events``. Replay reconstructs the complete engine action
        sequence before delegating to ``HandReplay``.
        """
        replay_actions = (
            self._engine_actions_for_replay()
        )

        engine_replay = HandReplay(
            config=config,
            master_seed=self.master_seed,
            hand_index=self.hand_index,
            dealer=self.dealer,
            actions=replay_actions,
        ).replay()

        if (
            engine_replay.actions
            != replay_actions
        ):
            raise ReplayIntegrityError(
                "Replayed actions do not match "
                "the reconstructed engine action sequence"
            )

        if self.termination in (
            HandTermination.NORMAL,
            HandTermination.TIMEOUT_FORFEIT,
        ):
            if not (
                engine_replay
                .final_state
                .is_terminal
            ):
                raise ReplayIntegrityError(
                    "Completed recorded hand did not replay "
                    "to a terminal state"
                )

            terminal_profit = (
                engine_replay
                .final_state
                .terminal_profit
            )

            if terminal_profit is None:
                raise ReplayIntegrityError(
                    "Replayed terminal hand has "
                    "no terminal profit"
                )

            expected_profit_bb = (
                terminal_profit[0]
                / CHIP_UNIT,
                terminal_profit[1]
                / CHIP_UNIT,
            )

            if self.terminal_profit_bb is None:
                raise ReplayIntegrityError(
                    "Completed recorded hand has "
                    "no terminal_profit_bb"
                )

            if not (
                math.isclose(
                    expected_profit_bb[0],
                    self.terminal_profit_bb[0],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    expected_profit_bb[1],
                    self.terminal_profit_bb[1],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ReplayIntegrityError(
                    "Replayed terminal profit does not "
                    "match recorded profit"
                )

        return RecordedReplayResult(
            record=self,
            engine_replay=engine_replay,
        )

    def _engine_actions_for_replay(
        self,
    ) -> tuple[Action, ...]:
        """
        Reconstruct every action submitted to the engine.

        Timeout-policy actions are inserted at their recorded action indexes.
        Older records containing exactly one unindexed timeout resolution are
        supported by appending that resolution after the agent actions.
        """
        timeout_actions: list[
            tuple[int | None, Action]
        ] = []

        for event in self.timeout_events:
            resolved_action = (
                _timeout_resolution_action(
                    event
                )
            )

            if resolved_action is None:
                continue

            timeout_actions.append(
                (
                    event.action_index,
                    resolved_action,
                )
            )

        if not timeout_actions:
            return self.actions

        indexed: dict[
            int,
            Action,
        ] = {}

        unindexed: list[
            Action
        ] = []

        for (
            action_index,
            action,
        ) in timeout_actions:
            if action_index is None:
                unindexed.append(
                    action
                )
                continue

            if action_index in indexed:
                raise ReplayIntegrityError(
                    "Multiple timeout resolutions "
                    "share the same action_index"
                )

            indexed[
                action_index
            ] = action

        if unindexed:
            if (
                len(unindexed) != 1
                or indexed
            ):
                raise ReplayIntegrityError(
                    "Cannot deterministically place "
                    "unindexed timeout resolutions"
                )

            return (
                self.actions
                + (
                    unindexed[0],
                )
            )

        total_engine_actions = (
            len(self.actions)
            + len(indexed)
        )

        if any(
            index < 0
            or index
            >= total_engine_actions
            for index
            in indexed
        ):
            raise ReplayIntegrityError(
                "Timeout action_index is outside "
                "the reconstructed action sequence"
            )

        agent_actions = iter(
            self.actions
        )

        reconstructed: list[
            Action
        ] = []

        for action_index in range(
            total_engine_actions
        ):
            timeout_action = (
                indexed.get(
                    action_index
                )
            )

            if timeout_action is not None:
                reconstructed.append(
                    timeout_action
                )
                continue

            try:
                agent_action = next(
                    agent_actions
                )

            except StopIteration as exc:
                raise ReplayIntegrityError(
                    "Timeout action indexes do not "
                    "match the recorded action count"
                ) from exc

            reconstructed.append(
                agent_action
            )

        try:
            next(
                agent_actions
            )

        except StopIteration:
            pass

        else:
            raise ReplayIntegrityError(
                "Recorded agent actions remain after "
                "timeout reconstruction"
            )

        return tuple(
            reconstructed
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible hand data."""
        return {
            "config_id": self.config_id,
            "master_seed": self.master_seed,
            "hand_index": self.hand_index,
            "dealer": self.dealer,
            "seat0_bot_id": self.seat0_bot_id,
            "seat1_bot_id": self.seat1_bot_id,
            "actions": [
                action.value
                for action in self.actions
            ],
            "termination": self.termination.value,
            "terminal_profit_bb": (
                list(self.terminal_profit_bb)
                if self.terminal_profit_bb is not None
                else None
            ),
            "timeout_events": [
                event.to_dict()
                for event in self.timeout_events
            ],
            **(
                {
                    "adaptive_learning_events": [
                        event.to_dict()
                        for event in self.adaptive_learning_events
                    ]
                }
                if self.adaptive_learning_events
                else {}
            ),
            "requested_action": (
                self.requested_action.value
                if self.requested_action is not None
                else None
            ),
            "legal_actions_at_termination": [
                action.value
                for action in self.legal_actions_at_termination
            ],
            "acting_player_at_termination": (
                self.acting_player_at_termination
            ),
            "failure_message": self.failure_message,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "RecordedHand":
        """Reconstruct a recorded hand from persisted data."""
        raw_profit = data.get("terminal_profit_bb")

        terminal_profit: tuple[float, float] | None
        if raw_profit is None:
            terminal_profit = None
        else:
            if (
                not isinstance(raw_profit, list)
                or len(raw_profit) != 2
            ):
                raise ValueError(
                    "terminal_profit_bb must be "
                    "a two-element list or null"
                )
            terminal_profit = (
                float(raw_profit[0]),
                float(raw_profit[1]),
            )

        raw_actions = data["actions"]
        if not isinstance(raw_actions, list):
            raise ValueError("actions must be a list")

        raw_timeouts = data.get(
            "timeout_events",
            [],
        )
        if not isinstance(raw_timeouts, list):
            raise ValueError(
                "timeout_events must be a list"
            )

        raw_learning_events = data.get(
            "adaptive_learning_events",
            [],
        )
        if not isinstance(raw_learning_events, list):
            raise ValueError(
                "adaptive_learning_events must be a list"
            )

        raw_legal_actions = data.get(
            "legal_actions_at_termination",
            [],
        )
        if not isinstance(raw_legal_actions, list):
            raise ValueError(
                "legal_actions_at_termination must be a list"
            )

        raw_requested_action = data.get(
            "requested_action"
        )
        raw_acting_player = data.get(
            "acting_player_at_termination"
        )

        return cls(
            config_id=str(data["config_id"]),
            master_seed=int(data["master_seed"]),
            hand_index=int(data["hand_index"]),
            dealer=int(data["dealer"]),
            seat0_bot_id=int(data["seat0_bot_id"]),
            seat1_bot_id=int(data["seat1_bot_id"]),
            actions=tuple(
                Action(value)
                for value in raw_actions
            ),
            termination=HandTermination(
                data["termination"]
            ),
            terminal_profit_bb=terminal_profit,
            timeout_events=tuple(
                RecordedTimeoutEvent.from_dict(timeout)
                for timeout in raw_timeouts
            ),
            adaptive_learning_events=tuple(
                RecordedAdaptiveLearningEvent.from_dict(event)
                for event in raw_learning_events
            ),
            requested_action=(
                Action(raw_requested_action)
                if raw_requested_action is not None
                else None
            ),
            legal_actions_at_termination=tuple(
                Action(value)
                for value in raw_legal_actions
            ),
            acting_player_at_termination=(
                int(raw_acting_player)
                if raw_acting_player is not None
                else None
            ),
            failure_message=(
                str(data["failure_message"])
                if data.get("failure_message") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class MatchRecord:
    """
    Durable deterministic replay record for a complete match.

    experiment_id identifies the scientific experiment.

    experiment_config_hash identifies the exact frozen experiment
    configuration used for this match.
    """

    match_id: str
    master_seed: int
    game_config: GameConfig
    hands: tuple[RecordedHand, ...]
    experiment_id: str | None = None
    experiment_config_hash: str | None = None
    schema_version: str = MATCH_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.match_id, str):
            raise TypeError("match_id must be a string")
        if not self.match_id.strip():
            raise ValueError("match_id must not be empty")

        if (
            not isinstance(self.master_seed, int)
            or isinstance(self.master_seed, bool)
        ):
            raise TypeError("master_seed must be an integer")

        if self.experiment_id is not None:
            if not isinstance(
                self.experiment_id,
                str,
            ):
                raise TypeError(
                    "experiment_id must be a string or None"
                )
            if not self.experiment_id.strip():
                raise ValueError(
                    "experiment_id must not be empty when provided"
                )

        if self.experiment_config_hash is not None:
            if not isinstance(
                self.experiment_config_hash,
                str,
            ):
                raise TypeError(
                    "experiment_config_hash must be a string or None"
                )
            if not self.experiment_config_hash.strip():
                raise ValueError(
                    "experiment_config_hash must not be empty "
                    "when provided"
                )

        if self.schema_version != MATCH_RECORD_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported match-record schema version: "
                f"{self.schema_version}"
            )

        for hand in self.hands:
            if hand.master_seed != self.master_seed:
                raise ValueError(
                    "RecordedHand master_seed does not "
                    "match MatchRecord"
                )

    @property
    def game_config_hash(self) -> str:
        """Return the canonical frozen game-config hash."""
        return config_hash(self.game_config)

    @property
    def record_hash(self) -> str:
        """Return the canonical SHA-256 record hash."""
        return _hash_json_value(
            self.to_dict()
        )

    def replay_all(
        self,
    ) -> tuple[RecordedReplayResult, ...]:
        """Deterministically replay every recorded hand."""
        return tuple(
            hand.replay(
                self.game_config
            )
            for hand in self.hands
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible match data."""
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "experiment_id": self.experiment_id,
            "experiment_config_hash": (
                self.experiment_config_hash
            ),
            "master_seed": self.master_seed,
            "game_config": (
                _game_config_to_dict(
                    self.game_config
                )
            ),
            "game_config_hash": (
                self.game_config_hash
            ),
            "hands": [
                hand.to_dict()
                for hand in self.hands
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "MatchRecord":
        """Construct a MatchRecord from verified persisted data."""
        schema_version = str(
            data["schema_version"]
        )
        if schema_version != MATCH_RECORD_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported match-record schema version: "
                f"{schema_version}"
            )

        raw_config = data[
            "game_config"
        ]
        if not isinstance(
            raw_config,
            Mapping,
        ):
            raise ValueError(
                "game_config must be an object"
            )

        game_config = (
            _game_config_from_dict(
                raw_config
            )
        )

        stored_config_hash = str(
            data["game_config_hash"]
        )
        calculated_config_hash = (
            config_hash(game_config)
        )
        if (
            stored_config_hash
            != calculated_config_hash
        ):
            raise ReplayIntegrityError(
                "Game configuration hash mismatch"
            )

        raw_hands = data["hands"]
        if not isinstance(
            raw_hands,
            list,
        ):
            raise ValueError(
                "hands must be a list"
            )

        raw_experiment_id = data.get(
            "experiment_id"
        )
        raw_experiment_config_hash = data.get(
            "experiment_config_hash"
        )

        return cls(
            match_id=str(
                data["match_id"]
            ),
            master_seed=int(
                data["master_seed"]
            ),
            game_config=game_config,
            hands=tuple(
                RecordedHand.from_dict(hand)
                for hand in raw_hands
            ),
            experiment_id=(
                str(raw_experiment_id)
                if raw_experiment_id is not None
                else None
            ),
            experiment_config_hash=(
                str(raw_experiment_config_hash)
                if raw_experiment_config_hash is not None
                else None
            ),
            schema_version=schema_version,
        )


class ReplayArchive:
    """
    Directory-backed durable MatchRecord archive.

    Files are stored as:

        <root>/<match_id>.json
    """

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """Return the archive root."""
        return self._root

    def path_for(
        self,
        match_id: str,
    ) -> Path:
        """Return the archive path for one match."""
        validated = (
            _validate_match_id(
                match_id
            )
        )
        return (
            self._root
            / f"{validated}.json"
        )

    def save(
        self,
        record: MatchRecord,
    ) -> Path:
        """Persist a MatchRecord."""
        return save_match_record(
            record,
            self.path_for(
                record.match_id
            ),
        )

    def load(
        self,
        match_id: str,
    ) -> MatchRecord:
        """
        Load one persisted match.

        FileNotFoundError deliberately propagates for an unknown match ID.
        """
        record = load_match_record(
            self.path_for(
                match_id
            )
        )

        if record.match_id != match_id:
            raise ReplayIntegrityError(
                "Stored match ID does not match "
                "requested match ID"
            )

        return record

    def replay(
        self,
        match_id: str,
    ) -> tuple[RecordedReplayResult, ...]:
        """
        Load and deterministically replay every hand in one persisted match.

        This is a convenience wrapper only. Replay semantics remain defined by
        MatchRecord.replay_all() and RecordedHand.replay().
        """
        return self.load(
            match_id
        ).replay_all()

    def exists(
        self,
        match_id: str,
    ) -> bool:
        """Return whether the archive contains the match."""
        return (
            self.path_for(
                match_id
            )
            .is_file()
        )


@overload
def save_match_record(
    record: MatchRecord,
    path: str | Path,
) -> Path:
    ...


@overload
def save_match_record(
    record: str | Path,
    path: MatchRecord,
) -> Path:
    ...


def save_match_record(
    record: MatchRecord | str | Path,
    path: MatchRecord | str | Path,
) -> Path:
    """
    Persist a MatchRecord to an explicit path.

    Both argument orders are accepted for compatibility:

        save_match_record(record, path)
        save_match_record(path, record)
    """
    match_record: MatchRecord
    destination: Path

    if (
        isinstance(
            record,
            MatchRecord,
        )
        and isinstance(
            path,
            (str, Path),
        )
    ):
        match_record = record
        destination = Path(path)

    elif (
        isinstance(
            record,
            (str, Path),
        )
        and isinstance(
            path,
            MatchRecord,
        )
    ):
        match_record = path
        destination = Path(record)

    else:
        raise TypeError(
            "save_match_record requires exactly "
            "one MatchRecord and one path"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record_payload = (
        match_record.to_dict()
    )

    file_payload = {
        "file_version": (
            MATCH_RECORD_FILE_VERSION
        ),
        "record_hash": (
            _hash_json_value(
                record_payload
            )
        ),
        "record": record_payload,
    }

    encoded = json.dumps(
        file_payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    )

    temporary = (
        destination.with_suffix(
            destination.suffix
            + ".tmp"
        )
    )

    temporary.write_text(
        encoded,
        encoding="utf-8",
    )
    temporary.replace(
        destination
    )

    return destination


def load_match_record(
    path: str | Path,
) -> MatchRecord:
    """
    Load and integrity-check one persisted MatchRecord.

    Error semantics:

        missing file
            -> FileNotFoundError

        existing but unreadable file
            -> ReplayIntegrityError

        malformed JSON
            -> ReplayIntegrityError

        tampered / invalid payload
            -> ReplayIntegrityError
    """
    source = Path(path)

    try:
        raw_text = source.read_text(
            encoding="utf-8"
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ReplayIntegrityError(
            "Could not read match-record file"
        ) from exc

    try:
        file_payload = json.loads(
            raw_text
        )
    except json.JSONDecodeError as exc:
        raise ReplayIntegrityError(
            "Could not read valid match-record JSON"
        ) from exc

    if not isinstance(
        file_payload,
        dict,
    ):
        raise ReplayIntegrityError(
            "Match-record file root must be an object"
        )

    try:
        file_version = file_payload[
            "file_version"
        ]
        stored_record_hash = file_payload[
            "record_hash"
        ]
        raw_record = file_payload[
            "record"
        ]
    except KeyError as exc:
        raise ReplayIntegrityError(
            "Match-record file is missing required fields"
        ) from exc

    if file_version != MATCH_RECORD_FILE_VERSION:
        raise ReplayIntegrityError(
            "Unsupported match-record file version: "
            f"{file_version}"
        )

    if not isinstance(
        stored_record_hash,
        str,
    ):
        raise ReplayIntegrityError(
            "record_hash must be a string"
        )

    if not isinstance(
        raw_record,
        dict,
    ):
        raise ReplayIntegrityError(
            "record must be an object"
        )

    calculated_record_hash = (
        _hash_json_value(
            raw_record
        )
    )
    if (
        calculated_record_hash
        != stored_record_hash
    ):
        raise ReplayIntegrityError(
            "Match-record integrity hash mismatch"
        )

    try:
        return MatchRecord.from_dict(
            raw_record
        )
    except ReplayIntegrityError:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ReplayIntegrityError(
            "Invalid match-record structure"
        ) from exc


def _game_config_to_dict(
    config: GameConfig,
) -> dict[str, Any]:
    """Serialize GameConfig using its canonical representation."""
    return canonical_config_dict(
        config
    )


def _game_config_from_dict(
    data: Mapping[str, Any],
) -> GameConfig:
    """Reconstruct GameConfig from canonical persisted data."""
    raw_bet_abstraction = data[
        "bet_abstraction"
    ]
    raw_allowed_actions = data[
        "allowed_actions"
    ]

    if not isinstance(
        raw_bet_abstraction,
        list,
    ):
        raise ValueError(
            "bet_abstraction must be a list"
        )

    if not isinstance(
        raw_allowed_actions,
        list,
    ):
        raise ValueError(
            "allowed_actions must be a list"
        )

    return GameConfig(
        game_type=GameType(
            data["game_type"]
        ),
        player_count=int(
            data["player_count"]
        ),
        starting_stack_bb=Decimal(
            str(
                data[
                    "starting_stack_bb"
                ]
            )
        ),
        small_blind_bb=Decimal(
            str(
                data[
                    "small_blind_bb"
                ]
            )
        ),
        big_blind_bb=Decimal(
            str(
                data[
                    "big_blind_bb"
                ]
            )
        ),
        ante_bb=Decimal(
            str(
                data[
                    "ante_bb"
                ]
            )
        ),
        rake=Decimal(
            str(
                data[
                    "rake"
                ]
            )
        ),
        stack_reset_bb=Decimal(
            str(
                data[
                    "stack_reset_bb"
                ]
            )
        ),
        bet_abstraction=tuple(
            Decimal(
                str(value)
            )
            for value in raw_bet_abstraction
        ),
        allowed_actions=tuple(
            Action(value)
            for value in raw_allowed_actions
        ),
        abstraction_version=str(
            data[
                "abstraction_version"
            ]
        ),
    )


def _hash_json_value(
    value: Mapping[str, Any],
) -> str:
    """Return SHA-256 over deterministic canonical JSON."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def _validate_match_id(
    match_id: str,
) -> str:
    """Validate a filesystem-safe match identifier."""
    if not isinstance(
        match_id,
        str,
    ):
        raise TypeError(
            "match_id must be a string"
        )

    normalized = (
        match_id.strip()
    )

    if not normalized:
        raise ValueError(
            "match_id must not be empty"
        )

    if normalized in (
        ".",
        "..",
    ):
        raise ValueError(
            "Invalid match_id"
        )

    if (
        "/"
        in normalized
        or "\\"
        in normalized
    ):
        raise ValueError(
            "match_id must not contain path separators"
        )

    return normalized


def _resolve_alias(
    *,
    primary: T | None,
    alias: T | None,
    primary_name: str,
    alias_name: str,
) -> T | None:
    """
    Resolve two names for the same value.

    If both are provided, they must agree.
    """
    if (
        primary is not None
        and alias is not None
        and primary != alias
    ):
        raise ValueError(
            f"{primary_name} and {alias_name} disagree"
        )

    if primary is not None:
        return primary

    return alias


def _timeout_resolution_action(
    event: RecordedTimeoutEvent,
) -> Action | None:
    """
    Return the engine action required to reconstruct a timeout resolution.

    Persisted timeout metadata separates policy from outcome:

    - ``timeout_policy`` describes what the framework is configured to do;
    - ``resolution`` describes the resulting timeout outcome;
    - ``applied_action`` records the exact engine action when available.

    The frozen default policy is ``fold``. A runner record may therefore
    contain ``timeout_policy="fold"`` together with
    ``resolution="forfeit"`` and no explicit applied_action. In that case
    deterministic replay must submit ``Action.FOLD``.

    For non-default policies, replay requires either an explicit
    ``applied_action`` or a resolution that is itself an Action value. This
    avoids inventing behaviour for policies whose state-dependent mapping is
    not encoded in the replay record.
    """
    if event.applied_action is not None:
        return event.applied_action

    if (
        event.timeout_policy == "fold"
        and event.resolution == "forfeit"
    ):
        return Action.FOLD

    if event.resolution is None:
        return None

    try:
        return Action(
            event.resolution
        )

    except ValueError as exc:
        raise ReplayIntegrityError(
            "Timeout record does not contain a replayable engine action: "
            f"timeout_policy={event.timeout_policy!r}, "
            f"resolution={event.resolution!r}"
        ) from exc


def _validate_binary_identity(
    value: int,
    field_name: str,
) -> None:
    """Validate a seat or stable bot identity."""
    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise TypeError(
            f"{field_name} must be an integer"
        )

    if value not in (
        0,
        1,
    ):
        raise ValueError(
            f"{field_name} must be 0 or 1"
        )


def _validate_optional_non_negative_int(
    value: int | None,
    field_name: str,
) -> None:
    """Validate an optional non-negative integer index."""
    if value is None:
        return

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise TypeError(
            f"{field_name} must be "
            "an integer or None"
        )

    if value < 0:
        raise ValueError(
            f"{field_name} must be "
            "non-negative"
        )


def _resolve_elapsed_ns(
    *,
    elapsed_ns: int | None,
    elapsed_ms: float | None,
) -> int | None:
    """
    Resolve elapsed time to the runner's canonical nanosecond representation.

    If both representations are supplied, they must agree within one
    nanosecond after conversion.
    """
    if elapsed_ns is not None:
        if (
            not isinstance(elapsed_ns, int)
            or isinstance(elapsed_ns, bool)
        ):
            raise TypeError("elapsed_ns must be an integer or None")
        if elapsed_ns < 0:
            raise ValueError("elapsed_ns must be non-negative")

    converted_ms_ns: int | None = None
    if elapsed_ms is not None:
        _require_finite_non_negative(
            elapsed_ms,
            "elapsed_ms",
        )
        converted_ms_ns = round(
            float(elapsed_ms)
            * 1_000_000.0
        )

    if (
        elapsed_ns is not None
        and converted_ms_ns is not None
        and abs(elapsed_ns - converted_ms_ns) > 1
    ):
        raise ValueError(
            "elapsed_ns and elapsed_ms disagree"
        )

    if elapsed_ns is not None:
        return elapsed_ns

    return converted_ms_ns


def _require_finite_non_negative(
    value: float,
    field_name: str,
) -> None:
    """Validate a finite non-negative number."""
    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise TypeError(
            f"{field_name} must be numeric"
        )

    numeric = float(value)

    if not math.isfinite(
        numeric
    ):
        raise ValueError(
            f"{field_name} must be finite"
        )

    if numeric < 0.0:
        raise ValueError(
            f"{field_name} must be non-negative"
        )


def _require_finite_positive(
    value: float,
    field_name: str,
) -> None:
    """Validate a finite positive number."""
    _require_finite_non_negative(
        value,
        field_name,
    )

    if float(value) <= 0.0:
        raise ValueError(
            f"{field_name} must be positive"
        )