from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pokerbench.engine.config import Action

if TYPE_CHECKING:
    from pokerbench.engine.state import Street


PUBLIC_ACTION_TRACE_VERSION = "public-action-trace-v1"


@dataclass(frozen=True)
class PublicActionEvent:
    """One already-executed action that is public to both players."""

    actor: int
    street: Street
    action: Action
    wager_increase: bool

    def __post_init__(self) -> None:
        from pokerbench.engine.state import Street

        if type(self.actor) is not int:
            raise TypeError(
                "actor must be an integer"
            )

        if self.actor not in (0, 1):
            raise ValueError(
                "actor must be 0 or 1"
            )

        if not isinstance(
            self.street,
            Street,
        ):
            raise TypeError(
                "street must be a Street"
            )

        if not isinstance(
            self.action,
            Action,
        ):
            raise TypeError(
                "action must be an Action"
            )

        if type(self.wager_increase) is not bool:
            raise TypeError(
                "wager_increase must be a boolean"
            )
