import hashlib
import random
from typing import Iterator

Card = int
HoleCards = tuple[Card, Card]
Board = tuple[Card, Card, Card, Card, Card]
HandDeal = tuple[HoleCards, HoleCards, Board]


class DealSchedule:
    """
    Deterministic, immutable deal schedule.

    The schedule is generated once from the master seed and is completely
    independent of agent RNG streams.
    """

    def __init__(self, master_seed: int, num_hands: int) -> None:
        self._validate_master_seed(master_seed)

        if not isinstance(num_hands, int):
            raise TypeError("num_hands must be an integer")

        if num_hands < 0:
            raise ValueError("num_hands must be non-negative")

        self._master_seed = master_seed
        self._hands = tuple(
            self.hand_for_index(master_seed, hand_index)
            for hand_index in range(num_hands)
        )

    def __len__(self) -> int:
        return len(self._hands)

    def __iter__(self) -> Iterator[HandDeal]:
        return iter(self._hands)

    def get_all_hands(self) -> tuple[HandDeal, ...]:
        return self._hands

    def get_hand(self, hand_index: int) -> HandDeal:
        if hand_index < 0 or hand_index >= len(self._hands):
            raise IndexError("hand_index out of range")

        return self._hands[hand_index]

    @staticmethod
    def _validate_master_seed(master_seed: int) -> None:
        if not isinstance(master_seed, int):
            raise TypeError("master_seed must be an integer")

    @staticmethod
    def _validate_hand_index(hand_index: int) -> None:
        if not isinstance(hand_index, int):
            raise TypeError("hand_index must be an integer")
        if hand_index < 0:
            raise ValueError("hand_index must be non-negative")

    @staticmethod
    def _seed_for_index(master_seed: int, hand_index: int) -> int:
        DealSchedule._validate_master_seed(master_seed)
        DealSchedule._validate_hand_index(hand_index)
        material = (
            f"pokerbench-deal-schedule:"
            f"{master_seed}:"
            f"{hand_index}"
        ).encode("utf-8")

        digest = hashlib.sha256(material).digest()

        return int.from_bytes(digest, byteorder="big")

    @classmethod
    def hand_for_index(
        cls,
        master_seed: int,
        hand_index: int,
    ) -> HandDeal:
        """Return one deterministic deal without materializing prior deals."""
        rng = random.Random(
            cls._seed_for_index(master_seed, hand_index)
        )

        deck = list(range(52))
        rng.shuffle(deck)

        hole_a = (deck[0], deck[1])
        hole_b = (deck[2], deck[3])
        board = (
            deck[4],
            deck[5],
            deck[6],
            deck[7],
            deck[8],
        )

        return hole_a, hole_b, board

    def _hand_seed(self, hand_index: int) -> int:
        return self._seed_for_index(
            self._master_seed,
            hand_index,
        )

    def _generate_hand(self, hand_index: int) -> HandDeal:
        return self.hand_for_index(
            self._master_seed,
            hand_index,
        )
