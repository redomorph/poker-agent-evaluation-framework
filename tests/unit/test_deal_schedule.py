import random

import pytest

from pokerbench.engine.deal_schedule import DealSchedule


def test_constructor_creates_correct_number_of_hands():
    schedule = DealSchedule(master_seed=12345, num_hands=10)

    assert len(schedule) == 10
    assert len(schedule.get_all_hands()) == 10


def test_hand_has_correct_card_counts():
    schedule = DealSchedule(master_seed=42, num_hands=5)

    for hole_a, hole_b, board in schedule:
        assert len(hole_a) == 2
        assert len(hole_b) == 2
        assert len(board) == 5


def test_no_duplicate_cards_in_hand():
    schedule = DealSchedule(master_seed=99, num_hands=10)

    for hole_a, hole_b, board in schedule:
        cards = (
            list(hole_a)
            + list(hole_b)
            + list(board)
        )

        assert len(cards) == 9
        assert len(set(cards)) == 9
        assert all(0 <= card < 52 for card in cards)


def test_deterministic_with_same_seed():
    first = DealSchedule(master_seed=12345, num_hands=10)
    second = DealSchedule(master_seed=12345, num_hands=10)

    assert first.get_all_hands() == second.get_all_hands()


def test_different_seeds_produce_different_schedules():
    first = DealSchedule(master_seed=1, num_hands=10)
    second = DealSchedule(master_seed=2, num_hands=10)

    assert first.get_all_hands() != second.get_all_hands()


def test_get_hand_returns_correct_entry():
    schedule = DealSchedule(master_seed=999, num_hands=5)

    assert schedule.get_hand(0) == schedule.get_all_hands()[0]
    assert schedule.get_hand(2) == schedule.get_all_hands()[2]


def test_get_hand_rejects_invalid_index():
    schedule = DealSchedule(master_seed=111, num_hands=2)

    with pytest.raises(IndexError):
        schedule.get_hand(-1)

    with pytest.raises(IndexError):
        schedule.get_hand(2)


def test_schedule_is_immutable_from_caller_perspective():
    schedule = DealSchedule(master_seed=123, num_hands=2)

    hands = schedule.get_all_hands()

    assert isinstance(hands, tuple)

    with pytest.raises(TypeError):
        hands[0] = hands[1]  # type: ignore


def test_external_rng_does_not_affect_schedule():
    master_seed = 777

    first = DealSchedule(master_seed, 5)

    external_rng = random.Random(123456)

    for _ in range(1000):
        external_rng.random()

    second = DealSchedule(master_seed, 5)

    assert first.get_all_hands() == second.get_all_hands()