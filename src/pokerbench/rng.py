# pokerbench/rng.py
"""Event-indexed RNG for agents."""

from __future__ import annotations

import hashlib
import random
from typing import Sequence, TypeVar


T = TypeVar("T")


class AgentRNG:
    """
    Deterministic, event-indexed random number generator for an agent.

    The RNG namespace is defined by:

        (master_seed, bot_id, hand_id, purpose)

    The table position / experimental configuration is deliberately NOT part
    of the namespace. This is required for paired position-compensation runs:
    the same agent must receive the same random stream when moved between
    Button/SB and Big Blind.

    Each random call consumes one deterministic decision event.
    """

    def __init__(
        self,
        master_seed: int,
        bot_id: int,
        hand_id: int,
        purpose: str = "decision",
    ) -> None:
        if not isinstance(master_seed, int):
            raise TypeError("master_seed must be an integer")

        if not isinstance(bot_id, int):
            raise TypeError("bot_id must be an integer")

        if not isinstance(hand_id, int):
            raise TypeError("hand_id must be an integer")

        if hand_id < 0:
            raise ValueError("hand_id must be non-negative")

        if not isinstance(purpose, str):
            raise TypeError("purpose must be a string")

        self.master_seed = master_seed
        self.bot_id = bot_id
        self.hand_id = hand_id
        self.purpose = purpose
        self._counter = 0

    def _event_rng(self) -> random.Random:
        """
        Create a deterministic RNG for the next random event.

        The event counter is part of the seed, so every decision event has
        its own independent deterministic random source.
        """
        material = (
            f"agent-rng:"
            f"{self.master_seed}:"
            f"{self.bot_id}:"
            f"{self.hand_id}:"
            f"{self._counter}:"
            f"{self.purpose}"
        ).encode("utf-8")

        digest = hashlib.sha256(material).digest()
        seed = int.from_bytes(digest, byteorder="big")

        return random.Random(seed)

    def random(self) -> float:
        """Return a deterministic random float in [0.0, 1.0)."""
        rng = self._event_rng()
        value = rng.random()
        self._counter += 1
        return value

    def randint(self, a: int, b: int) -> int:
        """Return a deterministic random integer in [a, b]."""
        if a > b:
            raise ValueError("a must be <= b")

        rng = self._event_rng()
        value = rng.randint(a, b)
        self._counter += 1
        return value

    def choice(self, seq: Sequence[T]) -> T:
        """Return a deterministic random element from a non-empty sequence."""
        if not seq:
            raise IndexError("cannot choose from an empty sequence")

        rng = self._event_rng()
        value = rng.choice(seq)
        self._counter += 1
        return value

    @property
    def decision_index(self) -> int:
        """Return the index of the next random decision event."""
        return self._counter