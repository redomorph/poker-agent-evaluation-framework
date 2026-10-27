import pytest

from pokerbench.engine.hand_evaluator import (
    HandEvaluator,
    HandRank,
    card_to_string,
    compare_hands,
    rank_of,
    suit_of,
)


def make_hand(ranks, suits):
    return tuple(
        rank + suit * 13
        for rank, suit in zip(ranks, suits)
    )


def test_card_conversion():
    assert card_to_string(0) == "2C"
    assert card_to_string(12) == "AC"
    assert card_to_string(13) == "2D"
    assert card_to_string(26) == "2H"
    assert card_to_string(51) == "AS"


def test_rank_and_suit():
    assert rank_of(0) == 0
    assert rank_of(12) == 12
    assert rank_of(13) == 0
    assert rank_of(51) == 12

    assert suit_of(0) == 0
    assert suit_of(13) == 1
    assert suit_of(26) == 2
    assert suit_of(51) == 3


@pytest.mark.parametrize(
    ("ranks", "expected_rank"),
    [
        ([0, 1, 2, 3, 5], HandRank.HIGH_CARD),
        ([0, 0, 1, 2, 3], HandRank.PAIR),
        ([0, 0, 1, 1, 3], HandRank.TWO_PAIR),
        ([0, 0, 0, 1, 2], HandRank.THREE_OF_A_KIND),
        ([0, 1, 2, 3, 4], HandRank.STRAIGHT),
        ([0, 1, 2, 3, 5], HandRank.FLUSH),
        ([0, 0, 0, 1, 1], HandRank.FULL_HOUSE),
        ([0, 0, 0, 0, 1], HandRank.FOUR_OF_A_KIND),
        ([0, 1, 2, 3, 4], HandRank.STRAIGHT_FLUSH),
        ([8, 9, 10, 11, 12], HandRank.ROYAL_FLUSH),
    ],
)
def test_five_card_hand_ranking(ranks, expected_rank):
    suits = [0, 1, 2, 3, 0]

    if expected_rank in (
        HandRank.FLUSH,
        HandRank.STRAIGHT_FLUSH,
        HandRank.ROYAL_FLUSH,
    ):
        suits = [0, 0, 0, 0, 0]

    value = HandEvaluator.evaluate(
        make_hand(ranks, suits)
    )

    assert value[0] == expected_rank


def test_ace_low_straight():
    hand = make_hand(
        [12, 0, 1, 2, 3],
        [0, 1, 0, 1, 0],
    )

    value = HandEvaluator.evaluate(hand)

    assert value == (
        HandRank.STRAIGHT,
        3,
    )


def test_ace_low_straight_flush():
    hand = make_hand(
        [12, 0, 1, 2, 3],
        [0, 0, 0, 0, 0],
    )

    value = HandEvaluator.evaluate(hand)

    assert value == (
        HandRank.STRAIGHT_FLUSH,
        3,
    )


def test_pair_kickers_are_compared():
    first = make_hand(
        [5, 5, 12, 8, 3],
        [0, 1, 0, 1, 2],
    )

    second = make_hand(
        [5, 5, 11, 8, 3],
        [0, 1, 0, 1, 2],
    )

    assert compare_hands(first, second) == 1


def test_two_pair_kicker_is_compared():
    first = make_hand(
        [5, 5, 3, 3, 12],
        [0, 1, 0, 1, 2],
    )

    second = make_hand(
        [5, 5, 3, 3, 11],
        [0, 1, 0, 1, 2],
    )

    assert compare_hands(first, second) == 1


def test_full_house_pair_is_compared():
    first = make_hand(
        [5, 5, 5, 3, 3],
        [0, 1, 2, 0, 1],
    )

    second = make_hand(
        [5, 5, 5, 2, 2],
        [0, 1, 2, 0, 1],
    )

    assert compare_hands(first, second) == 1


def test_four_of_a_kind_kicker_is_compared():
    first = make_hand(
        [5, 5, 5, 5, 12],
        [0, 1, 2, 3, 0],
    )

    second = make_hand(
        [5, 5, 5, 5, 11],
        [0, 1, 2, 3, 0],
    )

    assert compare_hands(first, second) == 1


def test_seven_card_best_hand():
    hand = make_hand(
        [0, 0, 1, 2, 3, 4, 5],
        [0, 1, 0, 1, 2, 3, 0],
    )

    best = HandEvaluator.best_hand(hand)

    assert len(best) == 5
    assert HandEvaluator.evaluate(best)[0] == HandRank.STRAIGHT


def test_duplicate_cards_are_rejected():
    hand = (
        0,
        0,
        1,
        2,
        3,
    )

    with pytest.raises(ValueError):
        HandEvaluator.evaluate(hand)


def test_invalid_cards_are_rejected():
    with pytest.raises(ValueError):
        HandEvaluator.evaluate(
            (0, 1, 2, 3, 52)
        )