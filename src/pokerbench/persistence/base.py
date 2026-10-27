"""Persistence boundary used by benchmark execution."""

from __future__ import annotations

from typing import Protocol

from pokerbench.replay import MatchRecord


class MatchStore(Protocol):
    """
    Minimal durable match-store contract required by PairedRunner.

    Concrete stores may expose richer querying and metadata APIs, but the
    runner only needs immutable MatchRecord persistence.
    """

    def save_match(
        self,
        record: MatchRecord,
    ) -> None:
        """Persist one completed match or raise on failure."""
        ...