# src/pokerbench/agents/budget.py
"""
Frozen scientific agent execution-budget definitions.

The scientific search budget and the optional wall-clock safety limit are
different quantities.

simulation_budget
    Primary scientific budget for search agents, expressed as simulations
    per decision.

time_budget_ms
    Optional wall-clock safety limit.

The generic runner does not attempt to introspect an agent's private search
implementation to count simulations. Search agents are responsible for
executing exactly their declared scientific simulation budget.

The runner does enforce the declared wall-clock overrun policy after
agent.act() returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


TIMEOUT_POLICY_VERSION = "timeout-policy-v1"


class TimeoutPolicy(StrEnum):
    """
    Frozen timeout behaviour.

    FOLD
        The timed-out player immediately forfeits the hand.

    CHECK_IF_LEGAL_ELSE_FOLD
        Check if CHECK is legal. Otherwise immediately forfeit the hand.
    """

    FOLD = "fold"
    CHECK_IF_LEGAL_ELSE_FOLD = (
        "check_if_legal_else_fold"
    )


@dataclass(frozen=True)
class AgentExecutionBudget:
    """
    Frozen execution budget for one experimental agent.

    simulation_budget is the scientific simulations-per-decision budget.

    time_budget_ms is only an optional technical wall-clock bound and must
    never silently replace the scientific simulation budget.
    """

    simulation_budget: int | None = None
    time_budget_ms: int | None = None

    timeout_policy: TimeoutPolicy = (
        TimeoutPolicy.FOLD
    )

    timeout_policy_version: str = (
        TIMEOUT_POLICY_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        if (
            self.simulation_budget
            is not None
        ):
            if (
                not isinstance(
                    self.simulation_budget,
                    int,
                )
                or isinstance(
                    self.simulation_budget,
                    bool,
                )
            ):
                raise TypeError(
                    "simulation_budget must be "
                    "an integer or None"
                )

            if (
                self.simulation_budget
                <= 0
            ):
                raise ValueError(
                    "simulation_budget must be "
                    "positive when declared"
                )

        if (
            self.time_budget_ms
            is not None
        ):
            if (
                not isinstance(
                    self.time_budget_ms,
                    int,
                )
                or isinstance(
                    self.time_budget_ms,
                    bool,
                )
            ):
                raise TypeError(
                    "time_budget_ms must be "
                    "an integer or None"
                )

            if (
                self.time_budget_ms
                <= 0
            ):
                raise ValueError(
                    "time_budget_ms must be "
                    "positive when declared"
                )

        if (
            self.timeout_policy_version
            != TIMEOUT_POLICY_VERSION
        ):
            raise ValueError(
                "Unsupported timeout policy version: "
                f"{self.timeout_policy_version}"
            )

    def to_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return deterministic metadata for experiment persistence.
        """
        return {
            "simulation_budget": (
                self.simulation_budget
            ),
            "time_budget_ms": (
                self.time_budget_ms
            ),
            "timeout_policy": (
                self.timeout_policy.value
            ),
            "timeout_policy_version": (
                self.timeout_policy_version
            ),
        }