"""Frozen passive adaptive policy using Bayesian posterior-derived ranges."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pokerbench.agents.hand_strength_baseline import (
    HAND_STRENGTH_BASELINE_VERSION,
    HAND_STRENGTH_POT_ODDS_VERSION,
)
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.opponent_model import (
    BAYESIAN_OPPONENT_MODEL_VERSION,
    BetaPrior,
    BayesianOpponentModel,
    OpponentModelMetric,
)
from pokerbench.agents.opponent_range import (
    PREFLOP_RANGE_STRENGTH_REFERENCE_VERSION,
    POSTERIOR_TO_RANGE_PROTOCOL_VERSION,
    OpponentRangeSnapshot,
    PreflopRangeSignal,
    PreflopStrengthReference,
    classify_preflop_range_signal,
    construct_opponent_range,
)
from pokerbench.agents.range_equity import (
    RANGE_CONDITIONED_EQUITY_VERSION,
    RangeConditionedEquityResult,
    estimate_range_conditioned_equity,
)
from pokerbench.engine.config import Action
from pokerbench.engine.state import Street
from pokerbench.rng import AgentRNG


ADAPTIVE_HAND_STRENGTH_POLICY_VERSION = "adaptive-hand-strength-policy-v1"
ADAPTIVE_HAND_STRENGTH_DECISION_VERSION = "adaptive-hand-strength-decision-v1"
_FROZEN_PREFLOP_REFERENCE_HASH = "cb110768ad74a9747064ba25e88af635055b1491a58deae8f09112aae801fc1e"
_PPM_SCALE = 1_000_000


class AdaptiveHandStrengthDecisionError(ValueError):
    """Raised when the frozen adaptive passive policy cannot decide legally."""


@dataclass(frozen=True)
class AdaptiveHandStrengthDecision:
    """Replay-ready provenance for one frozen adaptive-policy decision."""

    action: Action
    signal: PreflopRangeSignal
    model_state_hash: str
    range_snapshot: OpponentRangeSnapshot
    legal_combo_count: int
    pot_odds_ppm: int | None
    equity: RangeConditionedEquityResult | None
    decision_hash: str
    version: str = ADAPTIVE_HAND_STRENGTH_DECISION_VERSION

    def to_dict(self) -> dict[str, Any]:
        equity_payload: dict[str, Any] | None
        if self.equity is None:
            equity_payload = None
        else:
            equity_payload = {
                "method": self.equity.method.value,
                "equity_ppm": self.equity.equity_ppm,
                "wins": self.equity.wins,
                "ties": self.equity.ties,
                "losses": self.equity.losses,
                "joint_outcomes": self.equity.joint_outcomes,
                "sample_count": self.equity.sample_count,
                "legal_combo_count": self.equity.legal_combo_count,
                "range_hash": self.equity.range_hash,
                "equity_version": self.equity.equity_version,
                "range_version": self.equity.range_version,
            }
        return {
            "version": self.version,
            "action": self.action.value,
            "signal": self.signal.value,
            "model_state_hash": self.model_state_hash,
            "range_hash": self.range_snapshot.range_hash,
            "range_width_ppm": self.range_snapshot.width_ppm,
            "selected_combo_count": self.range_snapshot.selected_combo_count,
            "legal_combo_count": self.legal_combo_count,
            "pot_odds_ppm": self.pot_odds_ppm,
            "equity": equity_payload,
            "decision_hash": self.decision_hash,
        }


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _validate_observation(observation: AgentObservation) -> None:
    if not isinstance(observation, AgentObservation):
        raise TypeError("observation must be an AgentObservation")
    if observation.player_id not in (0, 1):
        raise AdaptiveHandStrengthDecisionError("player_id must be 0 or 1")
    if observation.current_player != observation.player_id:
        raise AdaptiveHandStrengthDecisionError(
            "adaptive-hand-strength-policy-v1 may act only for current player"
        )
    if observation.street not in (
        Street.PREFLOP,
        Street.FLOP,
        Street.TURN,
        Street.RIVER,
    ):
        raise AdaptiveHandStrengthDecisionError("unsupported decision street")


def _validate_legal_actions(legal_actions: Sequence[Action]) -> tuple[Action, ...]:
    if isinstance(legal_actions, (str, bytes)):
        raise TypeError("legal_actions must be a sequence of Action values")
    actions = tuple(legal_actions)
    if not actions:
        raise AdaptiveHandStrengthDecisionError("legal_actions must not be empty")
    if not all(isinstance(action, Action) for action in actions):
        raise TypeError("legal_actions must contain only Action values")
    return actions


def _validate_adaptive_model(model: BayesianOpponentModel) -> None:
    if not isinstance(model, BayesianOpponentModel):
        raise TypeError("model must be a BayesianOpponentModel")
    if model.model_version != BAYESIAN_OPPONENT_MODEL_VERSION:
        raise ValueError("unsupported Bayesian opponent model version")
    expected = BetaPrior(1, 1)
    for metric in OpponentModelMetric:
        if model.prior.for_metric(metric) != expected:
            raise ValueError(
                "adaptive-hand-strength-policy-v1 requires the frozen "
                "Beta(1,1) adaptive-learning priors"
            )


def _validate_reference(reference: PreflopStrengthReference) -> None:
    if not isinstance(reference, PreflopStrengthReference):
        raise TypeError("reference must be a PreflopStrengthReference")
    if reference.reference_version != PREFLOP_RANGE_STRENGTH_REFERENCE_VERSION:
        raise ValueError("unsupported preflop strength reference version")
    if reference.payload_hash != _FROZEN_PREFLOP_REFERENCE_HASH:
        raise ValueError("preflop strength reference hash does not match the frozen v1 reference")


def _faces_wager(observation: AgentObservation) -> bool:
    hero = observation.player_id
    opponent = 1 - hero
    return observation.current_bets[opponent] > observation.current_bets[hero]


def _pot_odds_ppm(observation: AgentObservation) -> int:
    hero = observation.player_id
    opponent = 1 - hero
    call_cost = min(
        observation.current_bets[opponent] - observation.current_bets[hero],
        observation.stacks[hero],
    )
    if call_cost <= 0:
        raise AdaptiveHandStrengthDecisionError("pot odds require positive call cost")
    denominator = observation.pot + call_cost
    if denominator <= 0:
        raise AdaptiveHandStrengthDecisionError("pot plus call cost must be positive")
    return (call_cost * _PPM_SCALE + denominator // 2) // denominator


def _decision_hash_payload(
    *,
    action: Action,
    signal: PreflopRangeSignal,
    model_state_hash: str,
    snapshot: OpponentRangeSnapshot,
    legal_combo_count: int,
    pot_odds_ppm: int | None,
    equity: RangeConditionedEquityResult | None,
) -> dict[str, Any]:
    return {
        "version": ADAPTIVE_HAND_STRENGTH_DECISION_VERSION,
        "policy_version": ADAPTIVE_HAND_STRENGTH_POLICY_VERSION,
        "control_policy_version": HAND_STRENGTH_BASELINE_VERSION,
        "pot_odds_version": HAND_STRENGTH_POT_ODDS_VERSION,
        "posterior_to_range_version": POSTERIOR_TO_RANGE_PROTOCOL_VERSION,
        "range_equity_version": RANGE_CONDITIONED_EQUITY_VERSION,
        "action": action.value,
        "signal": signal.value,
        "model_state_hash": model_state_hash,
        "range_hash": snapshot.range_hash,
        "range_width_ppm": snapshot.width_ppm,
        "selected_combo_count": snapshot.selected_combo_count,
        "legal_combo_count": legal_combo_count,
        "pot_odds_ppm": pot_odds_ppm,
        "equity": None
        if equity is None
        else {
            "method": equity.method.value,
            "equity_ppm": equity.equity_ppm,
            "wins": equity.wins,
            "ties": equity.ties,
            "losses": equity.losses,
            "joint_outcomes": equity.joint_outcomes,
            "sample_count": equity.sample_count,
            "legal_combo_count": equity.legal_combo_count,
            "range_hash": equity.range_hash,
            "equity_version": equity.equity_version,
            "range_version": equity.range_version,
        },
    }


def decide_adaptive_hand_strength_action(
    observation: AgentObservation,
    legal_actions: Sequence[Action],
    rng: AgentRNG,
    *,
    model: BayesianOpponentModel,
    reference: PreflopStrengthReference,
) -> AdaptiveHandStrengthDecision:
    """Choose the v1 passive action from current Bayesian model state."""

    _validate_observation(observation)
    actions = _validate_legal_actions(legal_actions)
    if not isinstance(rng, AgentRNG):
        raise TypeError("rng must be an AgentRNG")
    _validate_adaptive_model(model)
    _validate_reference(reference)

    opponent_id = 1 - observation.player_id
    signal = classify_preflop_range_signal(
        observation.public_action_history,
        opponent_id=opponent_id,
    )
    snapshot = construct_opponent_range(model, signal, reference)
    known_cards = tuple(observation.hole_cards) + tuple(observation.board)
    legal_combo_count = len(snapshot.legal_combinations(known_cards))

    equity: RangeConditionedEquityResult | None = None
    pot_odds_ppm: int | None = None

    if not _faces_wager(observation):
        if Action.CHECK not in actions:
            raise AdaptiveHandStrengthDecisionError(
                "CHECK must be legal when no wager is faced; "
                "v1 does not invent aggression"
            )
        action = Action.CHECK
    else:
        pot_odds_ppm = _pot_odds_ppm(observation)
        equity = estimate_range_conditioned_equity(
            observation,
            snapshot,
            master_seed=rng.master_seed,
            bot_id=rng.bot_id,
            hand_id=rng.hand_id,
            decision_index=rng.decision_index,
        )
        if equity.equity_ppm >= pot_odds_ppm and Action.CALL in actions:
            action = Action.CALL
        elif Action.FOLD in actions:
            action = Action.FOLD
        else:
            raise AdaptiveHandStrengthDecisionError(
                "frozen passive action is unavailable; v1 does not substitute aggression"
            )

    payload = _decision_hash_payload(
        action=action,
        signal=signal,
        model_state_hash=model.state_hash,
        snapshot=snapshot,
        legal_combo_count=legal_combo_count,
        pot_odds_ppm=pot_odds_ppm,
        equity=equity,
    )
    decision_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    return AdaptiveHandStrengthDecision(
        action=action,
        signal=signal,
        model_state_hash=model.state_hash,
        range_snapshot=snapshot,
        legal_combo_count=legal_combo_count,
        pot_odds_ppm=pot_odds_ppm,
        equity=equity,
        decision_hash=decision_hash,
    )
