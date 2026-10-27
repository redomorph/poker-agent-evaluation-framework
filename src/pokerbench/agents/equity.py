from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.hand_evaluator import HandEvaluator
from pokerbench.engine.state import Street


EQUITY_VERSION = "uniform-random-equity-v1"
EQUITY_RANGE_VERSION = "uniform-random-range-v1"
EXACT_OUTCOME_LIMIT = 500_000
EQUITY_SAMPLES = 10_000
_EQUITY_PPM_SCALE = 1_000_000
_FULL_DECK = tuple(range(52))


class EquityMethod(StrEnum):
    EXACT = "exact"
    MONTE_CARLO = "monte_carlo"


@dataclass(frozen=True)
class EquityResult:
    method: EquityMethod
    equity_ppm: int
    wins: int
    ties: int
    losses: int
    joint_outcomes: int
    sample_count: int
    equity_version: str = EQUITY_VERSION
    range_version: str = EQUITY_RANGE_VERSION

    @property
    def total_outcomes(self) -> int:
        return self.wins + self.ties + self.losses


def _round_ratio_half_up(
    numerator: int,
    denominator: int,
    *,
    scale: int,
) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")

    return (
        numerator * scale
        + denominator // 2
    ) // denominator


def _expected_board_length(street: Street) -> int:
    lengths = {
        Street.PREFLOP: 0,
        Street.FLOP: 3,
        Street.TURN: 4,
        Street.RIVER: 5,
    }

    try:
        return lengths[street]
    except KeyError as exc:
        raise ValueError(
            "Unsupported policy decision street: "
            f"{street!r}"
        ) from exc


def _validate_observation_cards(
    observation: AgentObservation,
) -> tuple[tuple[int, ...], int]:
    expected_board_length = _expected_board_length(
        observation.street
    )

    if len(observation.hole_cards) != 2:
        raise ValueError(
            "Hero must have exactly two hole cards"
        )

    if len(observation.board) != expected_board_length:
        raise ValueError(
            "Visible board length does not match street: "
            f"street={observation.street.name}, "
            f"expected={expected_board_length}, "
            f"actual={len(observation.board)}"
        )

    visible_cards = (
        tuple(observation.hole_cards)
        + tuple(observation.board)
    )

    for card in visible_cards:
        if not isinstance(card, int):
            raise TypeError("Cards must be integers")

        if not 0 <= card < 52:
            raise ValueError(f"Invalid card: {card}")

    if len(set(visible_cards)) != len(visible_cards):
        raise ValueError(
            "Hero-visible cards contain duplicate cards"
        )

    future_board_cards = 5 - len(observation.board)

    return visible_cards, future_board_cards


def uniform_random_candidate_deck(
    observation: AgentObservation,
) -> tuple[int, ...]:
    visible_cards, _ = _validate_observation_cards(
        observation
    )
    blocked = set(visible_cards)

    return tuple(
        card
        for card in _FULL_DECK
        if card not in blocked
    )


def joint_outcome_count(
    observation: AgentObservation,
) -> int:
    candidate_deck = uniform_random_candidate_deck(
        observation
    )
    _, future_board_cards = _validate_observation_cards(
        observation
    )

    remaining = len(candidate_deck)

    return (
        math.comb(remaining, 2)
        * math.comb(
            remaining - 2,
            future_board_cards,
        )
    )


def select_equity_method(
    observation: AgentObservation,
) -> EquityMethod:
    if joint_outcome_count(observation) <= EXACT_OUTCOME_LIMIT:
        return EquityMethod.EXACT

    return EquityMethod.MONTE_CARLO


def _hand_value(
    hole_cards: tuple[int, int],
    board: tuple[int, ...],
) -> tuple[int, ...]:
    cards = hole_cards + board

    if len(cards) != 7:
        raise ValueError(
            "Showdown evaluation requires seven cards"
        )

    return max(
        HandEvaluator.evaluate(combo)
        for combo in combinations(cards, 5)
    )


def _compare_showdown(
    hero_hole: tuple[int, int],
    opponent_hole: tuple[int, int],
    board: tuple[int, ...],
) -> int:
    hero_value = _hand_value(
        hero_hole,
        board,
    )
    opponent_value = _hand_value(
        opponent_hole,
        board,
    )

    if hero_value > opponent_value:
        return 1

    if hero_value < opponent_value:
        return -1

    return 0


def _equity_ppm(
    *,
    wins: int,
    ties: int,
    total: int,
) -> int:
    if wins < 0 or ties < 0 or total <= 0:
        raise ValueError("Invalid equity outcome counts")

    if wins + ties > total:
        raise ValueError("Invalid equity outcome counts")

    doubled_numerator = 2 * wins + ties
    doubled_denominator = 2 * total

    return _round_ratio_half_up(
        doubled_numerator,
        doubled_denominator,
        scale=_EQUITY_PPM_SCALE,
    )


def _equity_seed(
    *,
    master_seed: int,
    bot_id: int,
    hand_id: int,
    decision_index: int,
) -> int:
    values = {
        "master_seed": master_seed,
        "bot_id": bot_id,
        "hand_id": hand_id,
        "decision_index": decision_index,
    }

    for name, value in values.items():
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

    if hand_id < 0:
        raise ValueError("hand_id must be non-negative")

    if decision_index < 0:
        raise ValueError(
            "decision_index must be non-negative"
        )

    material = (
        "pokerbench-equity:"
        f"{EQUITY_VERSION}:"
        "equity:"
        f"{master_seed}:"
        f"{bot_id}:"
        f"{hand_id}:"
        f"{decision_index}"
    ).encode("utf-8")

    digest = hashlib.sha256(material).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )


def _exact_equity(
    observation: AgentObservation,
    candidate_deck: tuple[int, ...],
    future_board_cards: int,
    joint_outcomes: int,
) -> EquityResult:
    wins = 0
    ties = 0
    losses = 0
    hero_hole = (
        observation.hole_cards[0],
        observation.hole_cards[1],
    )
    visible_board = tuple(observation.board)

    for opponent_hole_raw in combinations(
        candidate_deck,
        2,
    ):
        opponent_hole = (
            opponent_hole_raw[0],
            opponent_hole_raw[1],
        )
        opponent_cards = set(opponent_hole)
        runout_deck = tuple(
            card
            for card in candidate_deck
            if card not in opponent_cards
        )

        for runout in combinations(
            runout_deck,
            future_board_cards,
        ):
            board = visible_board + tuple(runout)
            comparison = _compare_showdown(
                hero_hole,
                opponent_hole,
                board,
            )

            if comparison > 0:
                wins += 1
            elif comparison < 0:
                losses += 1
            else:
                ties += 1

    total = wins + ties + losses

    if total != joint_outcomes:
        raise RuntimeError(
            "Exact equity enumeration count mismatch: "
            f"expected={joint_outcomes}, actual={total}"
        )

    return EquityResult(
        method=EquityMethod.EXACT,
        equity_ppm=_equity_ppm(
            wins=wins,
            ties=ties,
            total=total,
        ),
        wins=wins,
        ties=ties,
        losses=losses,
        joint_outcomes=joint_outcomes,
        sample_count=total,
    )


def _monte_carlo_equity(
    observation: AgentObservation,
    candidate_deck: tuple[int, ...],
    future_board_cards: int,
    joint_outcomes: int,
    *,
    master_seed: int,
    bot_id: int,
    hand_id: int,
    decision_index: int,
) -> EquityResult:
    rng = random.Random(
        _equity_seed(
            master_seed=master_seed,
            bot_id=bot_id,
            hand_id=hand_id,
            decision_index=decision_index,
        )
    )

    wins = 0
    ties = 0
    losses = 0
    hero_hole = (
        observation.hole_cards[0],
        observation.hole_cards[1],
    )
    visible_board = tuple(observation.board)
    sample_width = 2 + future_board_cards

    for _ in range(EQUITY_SAMPLES):
        sample = rng.sample(
            candidate_deck,
            sample_width,
        )
        opponent_hole = (
            sample[0],
            sample[1],
        )
        board = (
            visible_board
            + tuple(sample[2:])
        )

        comparison = _compare_showdown(
            hero_hole,
            opponent_hole,
            board,
        )

        if comparison > 0:
            wins += 1
        elif comparison < 0:
            losses += 1
        else:
            ties += 1

    return EquityResult(
        method=EquityMethod.MONTE_CARLO,
        equity_ppm=_equity_ppm(
            wins=wins,
            ties=ties,
            total=EQUITY_SAMPLES,
        ),
        wins=wins,
        ties=ties,
        losses=losses,
        joint_outcomes=joint_outcomes,
        sample_count=EQUITY_SAMPLES,
    )


def estimate_uniform_random_equity(
    observation: AgentObservation,
    *,
    master_seed: int,
    bot_id: int,
    hand_id: int,
    decision_index: int,
) -> EquityResult:
    candidate_deck = uniform_random_candidate_deck(
        observation
    )
    _, future_board_cards = _validate_observation_cards(
        observation
    )
    joint_outcomes = joint_outcome_count(
        observation
    )
    method = (
        EquityMethod.EXACT
        if joint_outcomes <= EXACT_OUTCOME_LIMIT
        else EquityMethod.MONTE_CARLO
    )

    if method is EquityMethod.EXACT:
        # Validate seed inputs even though exact enumeration does not use them.
        # The public API is versioned uniformly across both computation modes.
        _equity_seed(
            master_seed=master_seed,
            bot_id=bot_id,
            hand_id=hand_id,
            decision_index=decision_index,
        )

        return _exact_equity(
            observation,
            candidate_deck,
            future_board_cards,
            joint_outcomes,
        )

    return _monte_carlo_equity(
        observation,
        candidate_deck,
        future_board_cards,
        joint_outcomes,
        master_seed=master_seed,
        bot_id=bot_id,
        hand_id=hand_id,
        decision_index=decision_index,
    )
