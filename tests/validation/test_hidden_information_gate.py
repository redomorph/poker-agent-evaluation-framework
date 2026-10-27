# tests/validation/test_hidden_information_gate.py
"""
Phase-0 hidden-information safety gate.

The true game state contains information that must never be exposed to an
agent, including:

- opponent hole cards;
- unrevealed future board cards;
- hidden deck state;
- hidden opponent RNG state.

The mandatory Phase-0 test constructs true states that differ in hidden
information while producing identical legal observations. A deterministic
policy must produce exactly the same action from those observations.

This gate intentionally tests both:

1. preflop, where the entire five-card board is hidden;
2. flop, where opponent hole cards and turn/river cards remain hidden.
"""

from __future__ import annotations

from dataclasses import fields

from pokerbench.agents.base import Agent
from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.engine.state import (
    GameState,
    Street,
)
from pokerbench.rng import AgentRNG


class ObservationOnlyDeterministicAgent(Agent):
    """
    Deterministic policy depending only on AgentObservation.

    The externally supplied legal_actions and RNG are deliberately ignored
    for decision logic. The chosen action is derived entirely from the
    observable information embedded in AgentObservation.

    Priority:

        CHECK
        CALL
        FOLD

    The test states below always contain one of these actions.
    """

    def __init__(
        self,
        name: str = "ObservationOnlyDeterministic",
    ) -> None:
        self.name = name

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        del legal_actions
        del rng

        observable_actions = (
            observation.legal_actions
        )

        if Action.CHECK in observable_actions:
            return Action.CHECK

        if Action.CALL in observable_actions:
            return Action.CALL

        if Action.FOLD in observable_actions:
            return Action.FOLD

        raise ValueError(
            "ObservationOnlyDeterministicAgent received "
            "no supported observable action"
        )


def _preflop_hidden_variants(
) -> tuple[
    GameState,
    GameState,
]:
    """
    Return two preflop states with the same legal observation for player 0.

    Visible information is identical.

    Hidden information differs in:

    - player 1's hole cards;
    - every future board card.
    """
    first = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=(
            (0, 1),   # player 0: visible and fixed
            (2, 3),   # player 1: hidden
        ),
        board=(
            4,
            5,
            6,
            7,
            8,
        ),
        dealer=0,
    )

    second = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=(
            (0, 1),   # player 0: identical
            (9, 10),  # different hidden opponent cards
        ),
        board=(
            11,
            12,
            13,
            14,
            15,
        ),
        dealer=0,
    )

    return (
        first,
        second,
    )


def _flop_hidden_variants(
) -> tuple[
    GameState,
    GameState,
]:
    """
    Return two flop states with the same legal observation for player 1.

    Visible information is identical:

    - player 1 hole cards;
    - flop;
    - stacks;
    - pot;
    - betting history;
    - legal actions;
    - acting player.

    Hidden information differs in:

    - player 0's hole cards;
    - turn card;
    - river card.
    """
    first = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=(
            (0, 1),   # player 0: hidden from player 1
            (2, 3),   # player 1: visible and fixed
        ),
        board=(
            4,
            5,
            6,        # public flop
            7,        # hidden turn
            8,        # hidden river
        ),
        dealer=0,
    )

    second = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=(
            (9, 10),  # different hidden opponent cards
            (2, 3),   # player 1: identical
        ),
        board=(
            4,
            5,
            6,        # identical public flop
            11,       # different hidden turn
            12,       # different hidden river
        ),
        dealer=0,
    )

    # Heads-up preflop:
    #
    # player 0 = BTN / SB
    # player 0 calls
    # player 1 = BB checks
    #
    # Both states then enter the flop with player 1 acting first.
    first = first.apply_action(
        Action.CALL
    )
    second = second.apply_action(
        Action.CALL
    )

    first = first.apply_action(
        Action.CHECK
    )
    second = second.apply_action(
        Action.CHECK
    )

    return (
        first,
        second,
    )


def test_preflop_hidden_information_changes_true_state_not_observation(
) -> None:
    """
    The complete future board and opponent cards are hidden preflop.
    """
    first, second = (
        _preflop_hidden_variants()
    )

    assert (
        first.hole_cards[1]
        != second.hole_cards[1]
    )

    assert (
        first.board
        != second.board
    )

    assert (
        first.current_player
        == second.current_player
        == 0
    )

    assert (
        first.street
        == second.street
        == Street.PREFLOP
    )

    observation_first = (
        first.observation(0)
    )

    observation_second = (
        second.observation(0)
    )

    # The entire board remains invisible preflop.
    assert (
        observation_first.board
        == ()
    )

    assert (
        observation_second.board
        == ()
    )

    # Critical information-set assertion:
    #
    # changing only hidden information must not alter the legal
    # observation supplied to the policy.
    assert (
        observation_first
        == observation_second
    )


def test_flop_hidden_information_changes_true_state_not_observation(
) -> None:
    """
    Opponent cards plus turn/river remain hidden on the flop.
    """
    first, second = (
        _flop_hidden_variants()
    )

    assert (
        first.street
        == second.street
        == Street.FLOP
    )

    # Heads-up postflop action starts with the BB.
    assert (
        first.current_player
        == second.current_player
        == 1
    )

    # True hidden information differs.
    assert (
        first.hole_cards[0]
        != second.hole_cards[0]
    )

    assert (
        first.board[3:]
        != second.board[3:]
    )

    observation_first = (
        first.observation(1)
    )

    observation_second = (
        second.observation(1)
    )

    # Only the public flop may be exposed.
    assert (
        observation_first.board
        == (
            4,
            5,
            6,
        )
    )

    assert (
        observation_second.board
        == (
            4,
            5,
            6,
        )
    )

    # Player 1 sees only its own cards.
    assert (
        observation_first.hole_cards
        == (
            2,
            3,
        )
    )

    assert (
        observation_second.hole_cards
        == (
            2,
            3,
        )
    )

    # Critical information-set assertion.
    assert (
        observation_first
        == observation_second
    )


def test_deterministic_policy_returns_same_action_for_hidden_variants(
) -> None:
    """
    Mandatory Section-9.8 acceptance test.

    Two true states differ in information unavailable to the player, but
    their legal observations are identical. A deterministic observation-only
    policy must therefore make exactly the same decision.

    Different RNG objects are deliberately supplied as an additional guard:
    the deterministic decision must not depend on any hidden RNG state.
    """
    first, second = (
        _flop_hidden_variants()
    )

    observation_first = (
        first.observation(1)
    )

    observation_second = (
        second.observation(1)
    )

    assert (
        observation_first
        == observation_second
    )

    agent = (
        ObservationOnlyDeterministicAgent()
    )

    action_first = agent.act(
        observation=observation_first,
        legal_actions=(
            first.legal_actions()
        ),
        rng=AgentRNG(
            master_seed=111,
            bot_id=1,
            hand_id=0,
            purpose="decision",
        ),
    )

    action_second = agent.act(
        observation=observation_second,
        legal_actions=(
            second.legal_actions()
        ),
        rng=AgentRNG(
            master_seed=999,
            bot_id=1,
            hand_id=0,
            purpose="decision",
        ),
    )

    assert (
        action_first
        == action_second
        == Action.CHECK
    )


def test_agent_observation_schema_contains_no_hidden_state_fields(
) -> None:
    """
    Guard the public observation structure against obvious hidden-state leaks.

    The true GameState may contain full cards and engine internals, while
    AgentObservation must remain an information-set representation.
    """
    observation_field_names = {
        field.name
        for field in fields(
            AgentObservation
        )
    }

    forbidden_fields = {
        "opponent_hole_cards",
        "all_hole_cards",
        "deck",
        "deck_state",
        "undealt_cards",
        "future_board",
        "full_board",
        "opponent_rng",
        "opponent_rng_state",
        "rng_state",
    }

    assert (
        observation_field_names
        .isdisjoint(
            forbidden_fields
        )
    )


def test_observation_does_not_embed_true_game_state(
) -> None:
    """
    AgentObservation must not provide a back-reference to GameState.

    Such a reference would defeat the information boundary even if the
    observation's direct fields appeared safe.
    """
    first, _ = (
        _flop_hidden_variants()
    )

    observation = (
        first.observation(1)
    )

    for field in fields(
        observation
    ):
        value = getattr(
            observation,
            field.name,
        )

        assert not isinstance(
            value,
            GameState,
        )