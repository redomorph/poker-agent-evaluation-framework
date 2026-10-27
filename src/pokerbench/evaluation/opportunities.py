"""
Versioned calibration-opportunity measurement from deterministic replays.

The definitions in this module implement ``calibration-opportunities-v2``.
They measure observable behaviour only; they do not define or infer the
parametric policy mechanism phi.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import (
    Action,
    GameConfig,
)
from pokerbench.engine.state import (
    GameState,
    Street,
)
from pokerbench.evaluation.calibration import (
    OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
    BehaviorCounts,
)
from pokerbench.replay import (
    HandTermination,
    MatchRecord,
    RecordedHand,
)


class CalibrationTraceError(ValueError):
    """Raised when a replay trace is not valid calibration evidence."""


class ActionEffect(StrEnum):
    """Betting effect used by the v2 calibration definitions."""

    FOLD = "fold"
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"


@dataclass
class _MutableCounts:
    vpip_opportunities: int = 0
    vpip_events: int = 0

    pfr_opportunities: int = 0
    pfr_events: int = 0

    three_bet_opportunities: int = 0
    three_bet_events: int = 0

    cbet_opportunities: int = 0
    cbet_events: int = 0

    postflop_aggressive_actions: int = 0
    postflop_passive_actions: int = 0

    fold_to_cbet_opportunities: int = 0
    fold_to_cbet_events: int = 0

    def freeze(self) -> BehaviorCounts:
        return BehaviorCounts(
            vpip_opportunities=(
                self.vpip_opportunities
            ),
            vpip_events=(
                self.vpip_events
            ),
            pfr_opportunities=(
                self.pfr_opportunities
            ),
            pfr_events=(
                self.pfr_events
            ),
            three_bet_opportunities=(
                self.three_bet_opportunities
            ),
            three_bet_events=(
                self.three_bet_events
            ),
            cbet_opportunities=(
                self.cbet_opportunities
            ),
            cbet_events=(
                self.cbet_events
            ),
            postflop_aggressive_actions=(
                self.postflop_aggressive_actions
            ),
            postflop_passive_actions=(
                self.postflop_passive_actions
            ),
            fold_to_cbet_opportunities=(
                self.fold_to_cbet_opportunities
            ),
            fold_to_cbet_events=(
                self.fold_to_cbet_events
            ),
            opportunity_definition_version=(
                OPPORTUNITY_DEFINITION_CONTRACT_VERSION
            ),
        )


@dataclass(frozen=True)
class _Decision:
    action_index: int
    state: GameState
    action: Action
    actor_seat: int
    actor_bot_id: int
    effect: ActionEffect


def _classify_action_effect_values(
    *,
    action: Action,
    current_player: int,
    current_bets: tuple[int, int],
    stacks: tuple[int, int],
) -> ActionEffect:
    if action == Action.FOLD:
        return ActionEffect.FOLD

    if action in (
        Action.CHECK,
        Action.CALL,
    ):
        return ActionEffect.PASSIVE

    if action in (
        Action.BET_33,
        Action.BET_66,
        Action.BET_100,
        Action.BET_200,
    ):
        return ActionEffect.AGGRESSIVE

    if action == Action.ALL_IN:
        opponent = 1 - current_player
        all_in_target = (
            current_bets[current_player]
            + stacks[current_player]
        )

        if all_in_target > current_bets[opponent]:
            return ActionEffect.AGGRESSIVE

        return ActionEffect.PASSIVE

    raise ValueError(
        f"Unsupported action: {action}"
    )


def classify_action_effect(
    state: GameState,
    action: Action,
) -> ActionEffect:
    """Classify one action by betting effect in a true pre-action state."""
    return _classify_action_effect_values(
        action=action,
        current_player=state.current_player,
        current_bets=state.current_bets,
        stacks=state.stacks,
    )


def classify_observation_action_effect(
    observation: AgentObservation,
    action: Action,
) -> ActionEffect:
    """Classify an action using only the agent-visible pre-action state."""
    if not isinstance(observation, AgentObservation):
        raise TypeError("observation must be an AgentObservation")

    return _classify_action_effect_values(
        action=action,
        current_player=observation.current_player,
        current_bets=observation.current_bets,
        stacks=observation.stacks,
    )


def is_voluntary_preflop_contribution(
    action: Action,
) -> bool:
    """Return whether an action is a voluntary preflop chip contribution."""
    return _is_voluntary_preflop_contribution(action)


def measure_behavior_counts(
    hands: Iterable[RecordedHand],
    game_config: GameConfig,
    *,
    bot_id: int,
) -> BehaviorCounts:
    """
    Measure v2 calibration counts for one stable bot identity.

    Every supplied hand must be a normal, timeout-free recorded hand.
    Exceptional or timeout-altered hands are rejected instead of silently
    entering calibration denominators.
    """
    _validate_bot_id(
        bot_id
    )

    counts = _MutableCounts()

    for hand in hands:
        _measure_hand(
            hand,
            game_config,
            bot_id=bot_id,
            counts=counts,
        )

    return counts.freeze()


def measure_match_behavior(
    match_record: MatchRecord,
    *,
    bot_id: int,
) -> BehaviorCounts:
    """Measure v2 calibration counts from one persisted match."""
    return measure_behavior_counts(
        match_record.hands,
        match_record.game_config,
        bot_id=bot_id,
    )


def _measure_hand(
    hand: RecordedHand,
    game_config: GameConfig,
    *,
    bot_id: int,
    counts: _MutableCounts,
) -> None:
    _validate_calibration_hand(
        hand
    )

    target_seat = _seat_for_bot(
        hand,
        bot_id,
    )

    replayed = hand.replay(
        game_config
    )

    decisions = tuple(
        _decision_from_replay(
            hand,
            replayed.states[index],
            action,
            action_index=index,
        )
        for index, action in enumerate(
            replayed.actions
        )
    )

    _measure_preflop(
        decisions,
        target_seat=target_seat,
        counts=counts,
    )

    _measure_postflop_aggression(
        decisions,
        target_seat=target_seat,
        counts=counts,
    )

    _measure_cbet_and_fold_to_cbet(
        decisions,
        target_seat=target_seat,
        counts=counts,
    )


def _measure_preflop(
    decisions: tuple[_Decision, ...],
    *,
    target_seat: int,
    counts: _MutableCounts,
) -> None:
    # VPIP opportunity is one dealt hand, regardless of whether the player
    # eventually receives a decision before the hand terminates.
    counts.vpip_opportunities += 1

    vpip_event = False
    pfr_opportunity = False
    pfr_event = False

    prior_raise_count = 0
    prior_raiser: int | None = None

    for decision in decisions:
        if decision.state.street != Street.PREFLOP:
            break

        is_target = (
            decision.actor_seat
            == target_seat
        )

        if is_target:
            if _has_legal_wager_increase(
                decision.state
            ):
                pfr_opportunity = True

            if _is_voluntary_preflop_contribution(
                decision.action
            ):
                vpip_event = True

            if (
                decision.effect
                == ActionEffect.AGGRESSIVE
            ):
                pfr_event = True

            if (
                prior_raise_count == 1
                and prior_raiser
                == 1 - target_seat
                and _faces_wager(
                    decision.state
                )
                and _has_legal_wager_increase(
                    decision.state
                )
            ):
                counts.three_bet_opportunities += 1

                if (
                    decision.effect
                    == ActionEffect.AGGRESSIVE
                ):
                    counts.three_bet_events += 1

        if (
            decision.effect
            == ActionEffect.AGGRESSIVE
        ):
            prior_raise_count += 1
            prior_raiser = (
                decision.actor_seat
            )

    if vpip_event:
        counts.vpip_events += 1

    if pfr_opportunity:
        counts.pfr_opportunities += 1

    if pfr_event:
        counts.pfr_events += 1


def _measure_postflop_aggression(
    decisions: tuple[_Decision, ...],
    *,
    target_seat: int,
    counts: _MutableCounts,
) -> None:
    for decision in decisions:
        if (
            decision.state.street
            == Street.PREFLOP
            or decision.actor_seat
            != target_seat
        ):
            continue

        if (
            decision.effect
            == ActionEffect.AGGRESSIVE
        ):
            counts.postflop_aggressive_actions += 1
            continue

        if (
            decision.effect
            == ActionEffect.PASSIVE
        ):
            counts.postflop_passive_actions += 1


def _measure_cbet_and_fold_to_cbet(
    decisions: tuple[_Decision, ...],
    *,
    target_seat: int,
    counts: _MutableCounts,
) -> None:
    final_preflop_aggressor = (
        _final_preflop_aggressor(
            decisions
        )
    )

    if final_preflop_aggressor is None:
        return

    first_aggressor_flop_decision_seen = False
    opponent_wager_increase_before = False
    qualifying_cbet_index: int | None = None

    for decision in decisions:
        if decision.state.street != Street.FLOP:
            continue

        if first_aggressor_flop_decision_seen:
            break

        if (
            decision.actor_seat
            != final_preflop_aggressor
        ):
            if (
                decision.effect
                == ActionEffect.AGGRESSIVE
            ):
                opponent_wager_increase_before = True

            continue

        first_aggressor_flop_decision_seen = True

        qualifies = (
            not opponent_wager_increase_before
            and _has_legal_wager_increase(
                decision.state
            )
        )

        if not qualifies:
            break

        if (
            final_preflop_aggressor
            == target_seat
        ):
            counts.cbet_opportunities += 1

            if (
                decision.effect
                == ActionEffect.AGGRESSIVE
            ):
                counts.cbet_events += 1

        if (
            decision.effect
            == ActionEffect.AGGRESSIVE
        ):
            qualifying_cbet_index = (
                decision.action_index
            )

        break

    if (
        qualifying_cbet_index is None
        or final_preflop_aggressor
        == target_seat
    ):
        return

    next_index = (
        qualifying_cbet_index
        + 1
    )

    if next_index >= len(decisions):
        return

    response = decisions[
        next_index
    ]

    if (
        response.state.street
        != Street.FLOP
        or response.actor_seat
        != target_seat
        or not _faces_wager(
            response.state
        )
    ):
        return

    counts.fold_to_cbet_opportunities += 1

    if response.action == Action.FOLD:
        counts.fold_to_cbet_events += 1


def _final_preflop_aggressor(
    decisions: tuple[_Decision, ...],
) -> int | None:
    final_aggressor: int | None = None

    for decision in decisions:
        if decision.state.street != Street.PREFLOP:
            break

        if (
            decision.effect
            == ActionEffect.AGGRESSIVE
        ):
            final_aggressor = (
                decision.actor_seat
            )

    return final_aggressor


def _decision_from_replay(
    hand: RecordedHand,
    state: GameState,
    action: Action,
    *,
    action_index: int,
) -> _Decision:
    actor_seat = state.current_player

    actor_bot_id = (
        hand.seat0_bot_id
        if actor_seat == 0
        else hand.seat1_bot_id
    )

    return _Decision(
        action_index=action_index,
        state=state,
        action=action,
        actor_seat=actor_seat,
        actor_bot_id=actor_bot_id,
        effect=classify_action_effect(
            state,
            action,
        ),
    )


def _validate_calibration_hand(
    hand: RecordedHand,
) -> None:
    if (
        hand.termination
        != HandTermination.NORMAL
    ):
        raise CalibrationTraceError(
            "Calibration requires normal hand termination; "
            f"got {hand.termination.value} for "
            f"{hand.config_id}/hand-{hand.hand_index}"
        )

    if hand.timeout_events:
        raise CalibrationTraceError(
            "Calibration rejects hands altered by a timeout policy"
        )


def _validate_bot_id(
    bot_id: int,
) -> None:
    if (
        not isinstance(
            bot_id,
            int,
        )
        or isinstance(
            bot_id,
            bool,
        )
    ):
        raise TypeError(
            "bot_id must be an integer"
        )

    if bot_id not in (
        0,
        1,
    ):
        raise ValueError(
            "bot_id must be 0 or 1"
        )


def _seat_for_bot(
    hand: RecordedHand,
    bot_id: int,
) -> int:
    if hand.seat0_bot_id == bot_id:
        return 0

    if hand.seat1_bot_id == bot_id:
        return 1

    raise CalibrationTraceError(
        f"Bot {bot_id} is not present in recorded hand"
    )


def _has_legal_wager_increase(
    state: GameState,
) -> bool:
    return any(
        classify_action_effect(
            state,
            action,
        )
        == ActionEffect.AGGRESSIVE
        for action in state.legal_actions()
    )


def _faces_wager(
    state: GameState,
) -> bool:
    player = state.current_player
    opponent = 1 - player

    return (
        state.current_bets[opponent]
        > state.current_bets[player]
    )


def _is_voluntary_preflop_contribution(
    action: Action,
) -> bool:
    return action in (
        Action.CALL,
        Action.ALL_IN,
        Action.BET_33,
        Action.BET_66,
        Action.BET_100,
        Action.BET_200,
    )
