"""Range-conditioned equity for the first adaptive hand-strength policy."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from itertools import combinations

from pokerbench.agents.equity import EquityMethod
from pokerbench.agents.observation import AgentObservation
from pokerbench.agents.opponent_range import (
    POSTERIOR_TO_RANGE_PROTOCOL_VERSION,
    OpponentRangeSnapshot,
)
from pokerbench.engine.hand_evaluator import HandEvaluator
from pokerbench.engine.state import Street


RANGE_CONDITIONED_EQUITY_VERSION = "posterior-range-equity-v1"
RANGE_CONDITIONED_RANGE_VERSION = POSTERIOR_TO_RANGE_PROTOCOL_VERSION
RANGE_EQUITY_RNG_VERSION = "posterior-range-equity-rng-v1"
RANGE_EQUITY_EXACT_OUTCOME_LIMIT = 500_000
RANGE_EQUITY_SAMPLES = 10_000
_PPM_SCALE = 1_000_000
_FULL_DECK = tuple(range(52))
_FROZEN_PREFLOP_REFERENCE_HASH = "cb110768ad74a9747064ba25e88af635055b1491a58deae8f09112aae801fc1e"


@dataclass(frozen=True)
class RangeConditionedEquityResult:
    """One exact or Monte Carlo equity result against a frozen range snapshot."""

    method: EquityMethod
    equity_ppm: int
    wins: int
    ties: int
    losses: int
    joint_outcomes: int
    sample_count: int
    legal_combo_count: int
    range_hash: str
    equity_version: str = RANGE_CONDITIONED_EQUITY_VERSION
    range_version: str = RANGE_CONDITIONED_RANGE_VERSION

    @property
    def total_outcomes(self) -> int:
        return self.wins + self.ties + self.losses


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
        raise ValueError(f"unsupported equity street: {street!r}") from exc


def _visible_cards(observation: AgentObservation) -> tuple[tuple[int, ...], int]:
    if not isinstance(observation, AgentObservation):
        raise TypeError("observation must be an AgentObservation")
    if len(observation.hole_cards) != 2:
        raise ValueError("hero must have exactly two hole cards")
    expected = _expected_board_length(observation.street)
    if len(observation.board) != expected:
        raise ValueError(
            "visible board length does not match street: "
            f"expected={expected}, actual={len(observation.board)}"
        )
    cards = tuple(observation.hole_cards) + tuple(observation.board)
    for card in cards:
        if type(card) is not int or not 0 <= card < 52:
            raise ValueError(f"invalid visible card: {card!r}")
    if len(cards) != len(set(cards)):
        raise ValueError("hero-visible cards contain duplicates")
    return cards, 5 - expected


def _canonical_range_core(snapshot: OpponentRangeSnapshot) -> bytes:
    import json

    core = {
        "version": snapshot.version,
        "signal": snapshot.signal.value,
        "width_ppm": snapshot.width_ppm,
        "selected_combo_count": snapshot.selected_combo_count,
        "selected_combinations": [list(combo) for combo in snapshot.selected_combinations],
        "model_state_hash": snapshot.model_state_hash,
        "reference_hash": snapshot.reference_hash,
    }
    return json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _validate_snapshot(snapshot: OpponentRangeSnapshot) -> None:
    if not isinstance(snapshot, OpponentRangeSnapshot):
        raise TypeError("range_snapshot must be an OpponentRangeSnapshot")
    if snapshot.version != RANGE_CONDITIONED_RANGE_VERSION:
        raise ValueError("unsupported opponent range version")
    if snapshot.reference_hash != _FROZEN_PREFLOP_REFERENCE_HASH:
        raise ValueError("opponent range does not use the frozen preflop strength reference")
    if snapshot.selected_combo_count != len(snapshot.selected_combinations):
        raise ValueError("selected_combo_count does not match range payload")
    if snapshot.selected_combo_count <= 0:
        raise ValueError("opponent range must contain at least one combination")
    if len(set(snapshot.selected_combinations)) != len(snapshot.selected_combinations):
        raise ValueError("opponent range combinations must be unique")
    for combo in snapshot.selected_combinations:
        if (
            not isinstance(combo, tuple)
            or len(combo) != 2
            or type(combo[0]) is not int
            or type(combo[1]) is not int
            or not 0 <= combo[0] < 52
            or not 0 <= combo[1] < 52
            or combo[0] >= combo[1]
        ):
            raise ValueError(f"invalid canonical opponent combination: {combo!r}")
    expected_hash = hashlib.sha256(_canonical_range_core(snapshot)).hexdigest()
    if snapshot.range_hash != expected_hash:
        raise ValueError("opponent range snapshot hash mismatch")


def _round_equity_ppm(*, wins: int, ties: int, total: int) -> int:
    if wins < 0 or ties < 0 or total <= 0 or wins + ties > total:
        raise ValueError("invalid equity outcome counts")
    numerator = 2 * wins + ties
    denominator = 2 * total
    return (numerator * _PPM_SCALE + denominator // 2) // denominator


def _hand_value(
    hole_cards: tuple[int, int],
    board: tuple[int, ...],
) -> tuple[int, ...]:
    cards = hole_cards + board
    if len(cards) != 7:
        raise ValueError("showdown evaluation requires seven cards")
    return max(HandEvaluator.evaluate(combo) for combo in combinations(cards, 5))


def _compare_showdown(
    hero_hole: tuple[int, int],
    opponent_hole: tuple[int, int],
    board: tuple[int, ...],
) -> int:
    hero = _hand_value(hero_hole, board)
    opponent = _hand_value(opponent_hole, board)
    if hero > opponent:
        return 1
    if hero < opponent:
        return -1
    return 0


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
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    if hand_id < 0 or decision_index < 0:
        raise ValueError("hand_id and decision_index must be non-negative")
    material = (
        "pokerbench-range-equity:"
        f"{RANGE_EQUITY_RNG_VERSION}:"
        f"{master_seed}:{bot_id}:{hand_id}:{decision_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big", signed=False)


def range_joint_outcome_count(
    observation: AgentObservation,
    range_snapshot: OpponentRangeSnapshot,
) -> int:
    _validate_snapshot(range_snapshot)
    visible, future_board_cards = _visible_cards(observation)
    legal = range_snapshot.legal_combinations(visible)
    remaining_after_opponent = 52 - len(visible) - 2
    return len(legal) * math.comb(remaining_after_opponent, future_board_cards)


def estimate_range_conditioned_equity(
    observation: AgentObservation,
    range_snapshot: OpponentRangeSnapshot,
    *,
    master_seed: int,
    bot_id: int,
    hand_id: int,
    decision_index: int,
) -> RangeConditionedEquityResult:
    """Estimate hero equity against a uniform distribution over the live range."""

    _validate_snapshot(range_snapshot)
    visible, future_board_cards = _visible_cards(observation)
    legal_combos = range_snapshot.legal_combinations(visible)
    visible_set = set(visible)
    candidate_deck = tuple(card for card in _FULL_DECK if card not in visible_set)
    runout_count_per_combo = math.comb(
        len(candidate_deck) - 2,
        future_board_cards,
    )
    joint_outcomes = len(legal_combos) * runout_count_per_combo
    if joint_outcomes <= 0:
        raise ValueError("range-conditioned equity has no legal outcomes")

    # Validate the versioned RNG identity in both exact and Monte Carlo modes.
    seed = _equity_seed(
        master_seed=master_seed,
        bot_id=bot_id,
        hand_id=hand_id,
        decision_index=decision_index,
    )
    hero_hole = (observation.hole_cards[0], observation.hole_cards[1])
    visible_board = tuple(observation.board)
    wins = ties = losses = 0

    if joint_outcomes <= RANGE_EQUITY_EXACT_OUTCOME_LIMIT:
        method = EquityMethod.EXACT
        for opponent_hole in legal_combos:
            opponent_cards = set(opponent_hole)
            runout_deck = tuple(
                card for card in candidate_deck if card not in opponent_cards
            )
            for runout in combinations(runout_deck, future_board_cards):
                result = _compare_showdown(
                    hero_hole,
                    opponent_hole,
                    visible_board + tuple(runout),
                )
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    ties += 1
        sample_count = wins + ties + losses
        if sample_count != joint_outcomes:
            raise RuntimeError("exact range equity enumeration count mismatch")
    else:
        method = EquityMethod.MONTE_CARLO
        rng = random.Random(seed)
        for _ in range(RANGE_EQUITY_SAMPLES):
            opponent_hole = legal_combos[rng.randrange(len(legal_combos))]
            opponent_cards = set(opponent_hole)
            runout_deck = tuple(
                card for card in candidate_deck if card not in opponent_cards
            )
            runout = tuple(rng.sample(runout_deck, future_board_cards))
            result = _compare_showdown(
                hero_hole,
                opponent_hole,
                visible_board + runout,
            )
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                ties += 1
        sample_count = RANGE_EQUITY_SAMPLES

    return RangeConditionedEquityResult(
        method=method,
        equity_ppm=_round_equity_ppm(
            wins=wins,
            ties=ties,
            total=sample_count,
        ),
        wins=wins,
        ties=ties,
        losses=losses,
        joint_outcomes=joint_outcomes,
        sample_count=sample_count,
        legal_combo_count=len(legal_combos),
        range_hash=range_snapshot.range_hash,
    )
