from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from pokerbench.engine.config import (
    Action,
    GameConfig,
)
from pokerbench.engine.deal_schedule import (
    DealSchedule,
    HandDeal,
)
from pokerbench.engine.state import GameState


@dataclass(frozen=True)
class ReplayResult:
    """
    Deterministic result of replaying one poker hand.

    replayed_actions stores the exact sequence supplied to HandReplay.

    It is intentionally not reconstructed from final_state.action_history.
    Some terminal engine transitions, such as fold termination, do not
    necessarily retain the terminal action in GameState.action_history.
    """

    initial_state: GameState
    states: Tuple[GameState, ...]
    final_state: GameState

    replayed_actions: Tuple[Action, ...] = ()

    @property
    def actions(self) -> Tuple[Action, ...]:
        """
        Return the complete sequence of actions actually replayed.

        The replay object is the authoritative source for this sequence.
        """
        return self.replayed_actions

    @property
    def deal(self) -> HandDeal:
        """Return the complete deal used for the hand."""
        state = self.initial_state

        return (
            state.hole_cards[0],
            state.hole_cards[1],
            state.board,
        )


@dataclass(frozen=True)
class HandReplay:
    """
    Immutable specification for replaying one deterministic hand.

    The deal is derived from the master seed and hand index. The supplied
    action sequence is then applied to the resulting initial game state.
    """

    config: GameConfig
    master_seed: int
    hand_index: int
    dealer: int
    actions: Tuple[Action, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.master_seed,
            int,
        ):
            raise TypeError(
                "master_seed must be an integer"
            )

        if not isinstance(
            self.hand_index,
            int,
        ):
            raise TypeError(
                "hand_index must be an integer"
            )

        if self.hand_index < 0:
            raise ValueError(
                "hand_index must be non-negative"
            )

        if self.dealer not in (
            0,
            1,
        ):
            raise ValueError(
                "dealer must be 0 or 1"
            )

        for action in self.actions:
            if not isinstance(
                action,
                Action,
            ):
                raise TypeError(
                    "actions must contain Action values"
                )

    @property
    def deal(self) -> HandDeal:
        """Return the deterministic deal for this replay."""
        schedule = DealSchedule(
            master_seed=self.master_seed,
            num_hands=self.hand_index + 1,
        )

        return schedule.get_hand(
            self.hand_index
        )

    def initial_state(self) -> GameState:
        """Create the deterministic initial state for this hand."""
        (
            hole_a,
            hole_b,
            board,
        ) = self.deal

        return GameState.new_hand(
            config=self.config,
            hole_cards=(
                hole_a,
                hole_b,
            ),
            board=board,
            dealer=self.dealer,
        )

    def replay(self) -> ReplayResult:
        """
        Replay the complete action sequence deterministically.

        The returned state sequence contains:

            initial state
            state after action 1
            state after action 2
            ...

        replayed_actions separately preserves the exact action sequence
        supplied to this replay. It does not depend on how individual
        terminal engine transitions encode action history.
        """
        initial = self.initial_state()

        states: list[GameState] = [
            initial
        ]

        current = initial

        for action in self.actions:
            current = current.apply_action(
                action
            )

            states.append(
                current
            )

        return ReplayResult(
            initial_state=initial,
            states=tuple(
                states
            ),
            final_state=current,
            replayed_actions=(
                self.actions
            ),
        )