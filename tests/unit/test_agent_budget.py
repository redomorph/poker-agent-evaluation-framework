# tests/unit/test_agent_budget.py
"""Unit tests for frozen agent execution budgets."""

from __future__ import annotations

import dataclasses

import pytest

from pokerbench.agents.budget import (
    TIMEOUT_POLICY_VERSION,
    AgentExecutionBudget,
    TimeoutPolicy,
)


def test_default_budget_has_no_scientific_or_time_limit() -> None:
    budget = AgentExecutionBudget()

    assert budget.simulation_budget is None
    assert budget.time_budget_ms is None

    assert (
        budget.timeout_policy
        == TimeoutPolicy.FOLD
    )

    assert (
        budget.timeout_policy_version
        == TIMEOUT_POLICY_VERSION
    )


def test_budget_is_frozen() -> None:
    budget = AgentExecutionBudget(
        simulation_budget=100,
        time_budget_ms=250,
    )

    with pytest.raises(
        dataclasses.FrozenInstanceError
    ):
        budget.time_budget_ms = 500  # type: ignore[misc]


def test_simulation_budget_must_be_positive_when_declared() -> None:
    with pytest.raises(
        ValueError,
        match="simulation_budget",
    ):
        AgentExecutionBudget(
            simulation_budget=0,
        )

    with pytest.raises(
        ValueError,
        match="simulation_budget",
    ):
        AgentExecutionBudget(
            simulation_budget=-1,
        )


def test_time_budget_must_be_positive_when_declared() -> None:
    with pytest.raises(
        ValueError,
        match="time_budget_ms",
    ):
        AgentExecutionBudget(
            time_budget_ms=0,
        )

    with pytest.raises(
        ValueError,
        match="time_budget_ms",
    ):
        AgentExecutionBudget(
            time_budget_ms=-1,
        )


def test_timeout_policies_are_frozen_and_versioned() -> None:
    assert (
        TimeoutPolicy.FOLD.value
        == "fold"
    )

    assert (
        TimeoutPolicy.CHECK_IF_LEGAL_ELSE_FOLD.value
        == "check_if_legal_else_fold"
    )

    assert (
        TIMEOUT_POLICY_VERSION
        == "timeout-policy-v1"
    )


def test_budget_metadata_is_explicit() -> None:
    budget = AgentExecutionBudget(
        simulation_budget=100,
        time_budget_ms=250,
        timeout_policy=(
            TimeoutPolicy.CHECK_IF_LEGAL_ELSE_FOLD
        ),
    )

    assert budget.to_metadata() == {
        "simulation_budget": 100,
        "time_budget_ms": 250,
        "timeout_policy": (
            "check_if_legal_else_fold"
        ),
        "timeout_policy_version": (
            "timeout-policy-v1"
        ),
    }