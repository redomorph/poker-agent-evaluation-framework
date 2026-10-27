"""Runner-managed lifecycle for adaptive-hand-strength-agent-v1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pokerbench.agents.adaptive_hand_strength import (
    AdaptiveHandStrengthDecision,
    decide_adaptive_hand_strength_action,
)
from pokerbench.agents.base import Agent
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.opponent_model import (
    BetaPrior,
    BayesianOpponentModel,
    OpponentModelPrior,
)
from pokerbench.agents.opponent_range import PreflopStrengthReference
from pokerbench.engine.config import Action
from pokerbench.evaluation.calibration import BehaviorCounts
from pokerbench.evaluation.opportunities import measure_behavior_counts
from pokerbench.replay.archive import (
    HandTermination,
    MatchRecord,
    RecordedAdaptiveLearningEvent,
)
from pokerbench.rng import AgentRNG


ADAPTIVE_HAND_STRENGTH_AGENT_VERSION = "adaptive-hand-strength-agent-v1"
ADAPTIVE_LIFECYCLE_PROTOCOL_VERSION = "adaptive-lifecycle-v1"


def frozen_adaptive_prior() -> OpponentModelPrior:
    """Return the explicit six-metric Beta(1,1) prior frozen by Spec 28."""
    prior = BetaPrior(1, 1)
    return OpponentModelPrior(
        vpip=prior,
        pfr=prior,
        three_bet=prior,
        cbet=prior,
        aggressive_share=prior,
        fold_to_cbet=prior,
    )


@runtime_checkable
class AdaptiveLearningLifecycleAgent(Protocol):
    """Structural runner hook for post-terminal public-evidence updates."""

    @property
    def is_adaptive(self) -> bool: ...

    def reset(self) -> None: ...

    def observe_opponent_hand(
        self,
        counts: BehaviorCounts,
        *,
        learner_bot_id: int,
        opponent_bot_id: int,
        trajectory_hand_index: int,
    ) -> RecordedAdaptiveLearningEvent: ...


class AdaptiveHandStrengthAgent(Agent):
    """Stateful wrapper around the frozen pure Spec-30 decision mechanism."""

    version = ADAPTIVE_HAND_STRENGTH_AGENT_VERSION

    def __init__(self, reference: PreflopStrengthReference) -> None:
        if not isinstance(reference, PreflopStrengthReference):
            raise TypeError("reference must be a PreflopStrengthReference")
        self._reference = reference
        self._model = self.initial_model()
        self._last_decision: AdaptiveHandStrengthDecision | None = None

    @staticmethod
    def initial_model() -> BayesianOpponentModel:
        return BayesianOpponentModel.from_prior(frozen_adaptive_prior())

    @property
    def model(self) -> BayesianOpponentModel:
        return self._model

    @property
    def last_decision(self) -> AdaptiveHandStrengthDecision | None:
        return self._last_decision

    @property
    def is_adaptive(self) -> bool:
        return True

    def reset(self) -> None:
        self._model = self.initial_model()
        self._last_decision = None

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        decision = decide_adaptive_hand_strength_action(
            observation,
            legal_actions,
            rng,
            model=self._model,
            reference=self._reference,
        )
        self._last_decision = decision
        return decision.action

    def observe_opponent_hand(
        self,
        counts: BehaviorCounts,
        *,
        learner_bot_id: int,
        opponent_bot_id: int,
        trajectory_hand_index: int,
    ) -> RecordedAdaptiveLearningEvent:
        if learner_bot_id not in (0, 1) or opponent_bot_id not in (0, 1):
            raise ValueError("bot IDs must be 0 or 1")
        if learner_bot_id == opponent_bot_id:
            raise ValueError("learner and opponent bot IDs must be different")
        if counts.vpip_opportunities != 1:
            raise ValueError(
                "each learnable terminal hand must contain exactly one VPIP opportunity"
            )
        if trajectory_hand_index != self._model.hands_observed:
            raise ValueError(
                "trajectory_hand_index must equal current learned-hand count"
            )

        before = self._model
        after = before.updated(
            counts,
            first_hand_index=before.hands_observed,
        )
        event = RecordedAdaptiveLearningEvent.from_transition(
            learner_bot_id=learner_bot_id,
            opponent_bot_id=opponent_bot_id,
            trajectory_hand_index=trajectory_hand_index,
            before=before,
            evidence=counts,
            after=after,
        )
        self._model = after
        return event


def _counts_dict(counts: BehaviorCounts) -> dict[str, int | str]:
    return {
        "vpip_opportunities": counts.vpip_opportunities,
        "vpip_events": counts.vpip_events,
        "pfr_opportunities": counts.pfr_opportunities,
        "pfr_events": counts.pfr_events,
        "three_bet_opportunities": counts.three_bet_opportunities,
        "three_bet_events": counts.three_bet_events,
        "cbet_opportunities": counts.cbet_opportunities,
        "cbet_events": counts.cbet_events,
        "postflop_aggressive_actions": counts.postflop_aggressive_actions,
        "postflop_passive_actions": counts.postflop_passive_actions,
        "fold_to_cbet_opportunities": counts.fold_to_cbet_opportunities,
        "fold_to_cbet_events": counts.fold_to_cbet_events,
        "opportunity_definition_version": counts.opportunity_definition_version,
    }


def _counts_from_mapping(payload: Mapping[str, object]) -> BehaviorCounts:
    return BehaviorCounts(
        vpip_opportunities=int(payload["vpip_opportunities"]),
        vpip_events=int(payload["vpip_events"]),
        pfr_opportunities=int(payload["pfr_opportunities"]),
        pfr_events=int(payload["pfr_events"]),
        three_bet_opportunities=int(payload["three_bet_opportunities"]),
        three_bet_events=int(payload["three_bet_events"]),
        cbet_opportunities=int(payload["cbet_opportunities"]),
        cbet_events=int(payload["cbet_events"]),
        postflop_aggressive_actions=int(payload["postflop_aggressive_actions"]),
        postflop_passive_actions=int(payload["postflop_passive_actions"]),
        fold_to_cbet_opportunities=int(payload["fold_to_cbet_opportunities"]),
        fold_to_cbet_events=int(payload["fold_to_cbet_events"]),
        opportunity_definition_version=str(payload["opportunity_definition_version"]),
    )


def replay_adaptive_learning_for_bot(
    match_record: MatchRecord,
    *,
    bot_id: int,
) -> dict[str, BayesianOpponentModel]:
    """Reclassify public evidence and verify the complete learning hash chain."""
    if bot_id not in (0, 1):
        raise ValueError("bot_id must be 0 or 1")

    results: dict[str, BayesianOpponentModel] = {}
    current_config: str | None = None
    model = AdaptiveHandStrengthAgent.initial_model()

    for hand in match_record.hands:
        if hand.config_id != current_config:
            current_config = hand.config_id
            model = AdaptiveHandStrengthAgent.initial_model()

        events = tuple(
            event
            for event in hand.adaptive_learning_events
            if event.learner_bot_id == bot_id
        )
        if not events:
            continue
        if len(events) != 1:
            raise ValueError("each learner may have exactly one transition per hand")
        event = events[0]
        if hand.termination is not HandTermination.NORMAL or hand.timeout_events:
            raise ValueError("timeout-altered or exceptional hand is not learnable")
        if event.trajectory_hand_index != model.hands_observed:
            raise ValueError("learning transition chronology mismatch")
        if event.model_state_before_hash != model.state_hash:
            raise ValueError("learning transition before-state hash mismatch")

        recomputed = measure_behavior_counts(
            (hand,),
            match_record.game_config,
            bot_id=event.opponent_bot_id,
        )
        if _counts_dict(recomputed) != dict(event.public_evidence_counts):
            raise ValueError("persisted public evidence differs from replay classification")

        persisted_counts = _counts_from_mapping(event.public_evidence_counts)
        after = model.updated(
            persisted_counts,
            first_hand_index=model.hands_observed,
        )
        if event.model_state_after_hash != after.state_hash:
            raise ValueError("learning transition after-state hash mismatch")
        event.validate_hash()
        model = after
        results[hand.config_id] = model

    return results
