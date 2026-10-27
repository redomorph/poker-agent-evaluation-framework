# `src/pokerbench/engine/hand_evaluator.py`

from collections import Counter
from enum import IntEnum
from itertools import combinations
from typing import Iterable


class HandRank(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


def rank_of(card: int) -> int:
    if not 0 <= card < 52:
        raise ValueError(f"Invalid card: {card}")

    return card % 13


def suit_of(card: int) -> int:
    if not 0 <= card < 52:
        raise ValueError(f"Invalid card: {card}")

    return card // 13


def card_to_string(card: int) -> str:
    ranks = "23456789TJQKA"
    suits = "CDHS"

    return f"{ranks[rank_of(card)]}{suits[suit_of(card)]}"


def _validate_cards(cards: Iterable[int]) -> tuple[int, ...]:
    cards = tuple(cards)

    if len(cards) not in (5, 6, 7):
        raise ValueError(
            "Texas Hold'em hand evaluation requires 5, 6, or 7 cards"
        )

    for card in cards:
        if not 0 <= card < 52:
            raise ValueError(f"Invalid card: {card}")

    if len(set(cards)) != len(cards):
        raise ValueError("A hand cannot contain duplicate cards")

    return cards


def _straight_high(ranks: Iterable[int]) -> int | None:
    unique = set(ranks)

    # Ace can be used as the low card in A-2-3-4-5.
    # Rank 12 represents Ace, while ranks 0..3 represent 2..5.
    if 12 in unique:
        unique.add(-1)

    for high in range(12, 3, -1):
        if all(rank in unique for rank in range(high - 4, high + 1)):
            return high

    # Wheel straight: A-2-3-4-5.
    if {-1, 0, 1, 2, 3}.issubset(unique):
        return 3

    return None


class HandEvaluator:
    @staticmethod
    def evaluate(cards: Iterable[int]) -> tuple[int, ...]:
        cards = _validate_cards(cards)

        ranks = [rank_of(card) for card in cards]
        suits = [suit_of(card) for card in cards]

        rank_counts = Counter(ranks)

        flush = len(set(suits)) == 1
        straight_high = _straight_high(ranks)

        if flush and straight_high is not None:
            if straight_high == 12:
                return (HandRank.ROYAL_FLUSH,)

            return (
                HandRank.STRAIGHT_FLUSH,
                straight_high,
            )

        quads = sorted(
            (
                rank
                for rank, count in rank_counts.items()
                if count == 4
            ),
            reverse=True,
        )

        if quads:
            quad_rank = quads[0]
            kicker = max(
                rank
                for rank in ranks
                if rank != quad_rank
            )

            return (
                HandRank.FOUR_OF_A_KIND,
                quad_rank,
                kicker,
            )

        trips = sorted(
            (
                rank
                for rank, count in rank_counts.items()
                if count == 3
            ),
            reverse=True,
        )

        pairs = sorted(
            (
                rank
                for rank, count in rank_counts.items()
                if count >= 2
            ),
            reverse=True,
        )

        if trips:
            trip_rank = trips[0]
            remaining_pairs = [
                rank
                for rank in pairs
                if rank != trip_rank
            ]

            if remaining_pairs:
                return (
                    HandRank.FULL_HOUSE,
                    trip_rank,
                    remaining_pairs[0],
                )

        if flush:
            return (
                HandRank.FLUSH,
                *sorted(ranks, reverse=True),
            )

        if straight_high is not None:
            return (
                HandRank.STRAIGHT,
                straight_high,
            )

        if trips:
            trip_rank = trips[0]
            kickers = sorted(
                (
                    rank
                    for rank in ranks
                    if rank != trip_rank
                ),
                reverse=True,
            )

            return (
                HandRank.THREE_OF_A_KIND,
                trip_rank,
                *kickers[:2],
            )

        pair_ranks = sorted(
            (
                rank
                for rank, count in rank_counts.items()
                if count == 2
            ),
            reverse=True,
        )

        if len(pair_ranks) >= 2:
            high_pair, low_pair = pair_ranks[:2]
            kicker = max(
                rank
                for rank in ranks
                if rank not in (high_pair, low_pair)
            )

            return (
                HandRank.TWO_PAIR,
                high_pair,
                low_pair,
                kicker,
            )

        if len(pair_ranks) == 1:
            pair_rank = pair_ranks[0]
            kickers = sorted(
                (
                    rank
                    for rank in ranks
                    if rank != pair_rank
                ),
                reverse=True,
            )

            return (
                HandRank.PAIR,
                pair_rank,
                *kickers[:3],
            )

        return (
            HandRank.HIGH_CARD,
            *sorted(ranks, reverse=True),
        )

    @staticmethod
    def best_hand(cards: Iterable[int]) -> tuple[int, ...]:
        cards = _validate_cards(cards)

        best = max(
            combinations(cards, 5),
            key=HandEvaluator.evaluate,
        )

        return tuple(best)


def compare_hands(
    hand_a: Iterable[int],
    hand_b: Iterable[int],
) -> int:
    value_a = HandEvaluator.evaluate(hand_a)
    value_b = HandEvaluator.evaluate(hand_b)

    if value_a > value_b:
        return 1

    if value_a < value_b:
        return -1

    return 0
