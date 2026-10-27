# tests/validation/test_agent_budget_gates.py
"""
Phase-0 validation gates for agent execution budgets and timeout policy.

The tests verify:

1. an over-budget action is discarded;
2. the default timeout policy forfeits the pot;
3. timeout takes precedence over an illegal late action;
4. check_if_legal_else_fold is resolved correctly;
5. budget and timeout configuration are persisted in runner metadata.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import pokerbench.runner as runner_module
from pokerbench.agents.base import Agent
from pokerbench.agents.budget import (
    TIMEOUT_POLICY_VERSION,
    AgentExecutionBudget,
    TimeoutPolicy,
)
from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.engine.deal_schedule import DealSchedule
from pokerbench.engine.state import GameState
from pokerbench.rng import AgentRNG
from pokerbench.runner import (
    IllegalActionPolicy,
    PairedRunner,
)


class CallCheckAgent(Agent):
    """Call when facing a wager and otherwise check."""

    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        if Action.CALL in legal_actions:
            return Action.CALL

        if Action.CHECK in legal_actions:
            return Action.CHECK

        return Action.FOLD


class IllegalAgent(Agent):
    """Agent deliberately returning an illegal action."""

    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        return Action.BET_33


def _two_ms_clock() -> Iterator[int]:
    """
    Produce timestamps for deterministic 2 ms decision durations.

    Calls occur in start/end pairs:

        0 ns
        2,000,000 ns
        4,000,000 ns
        6,000,000 ns
        ...

    Every measured action therefore takes exactly 2 ms.
    """
    value = 0

    while True:
        yield value
        value += 2_000_000


def test_timeout_fold_forfeits_hand_and_remains_valid_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _two_ms_clock()

    monkeypatch.setattr(
        runner_module.time,
        "perf_counter_ns",
        lambda: next(clock),
    )

    budget = AgentExecutionBudget(
        time_budget_ms=1,
        timeout_policy=(
            TimeoutPolicy.FOLD
        ),
    )

    runner = PairedRunner(
        agent_a=CallCheckAgent("A"),
        agent_b=CallCheckAgent("B"),
        num_hands=1,
        seed=42,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
        illegal_action_policy=(
            IllegalActionPolicy.EXCLUDE_RUN
        ),
        agent_a_budget=budget,
        agent_b_budget=budget,
    )

    result = runner.run()

    # Both positional conditions time out on the first decision.
    #
    # A BTN forfeits condition A.
    # B BTN forfeits condition B.
    #
    # The identical timeout policy therefore remains position symmetric.
    assert result.num_valid_pairs == 1
    assert result.failed_runs == []

    assert (
        result.paired_differences
        == pytest.approx([0.0])
    )

    assert len(
        result.timeout_events
    ) == 2

    for event in result.timeout_events:
        assert event.budget_ms == 1
        assert event.elapsed_ns == 2_000_000

        assert (
            event.timeout_policy
            == TimeoutPolicy.FOLD
        )

        assert (
            event.timeout_policy_version
            == TIMEOUT_POLICY_VERSION
        )

        assert (
            event.resolution
            == "forfeit"
        )


def test_timeout_precedes_validation_of_late_illegal_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _two_ms_clock()

    monkeypatch.setattr(
        runner_module.time,
        "perf_counter_ns",
        lambda: next(clock),
    )

    budget = AgentExecutionBudget(
        time_budget_ms=1,
        timeout_policy=(
            TimeoutPolicy.FOLD
        ),
    )

    runner = PairedRunner(
        agent_a=IllegalAgent("A"),
        agent_b=IllegalAgent("B"),
        num_hands=1,
        seed=123,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
        illegal_action_policy=(
            IllegalActionPolicy.RAISE_EXCEPTION
        ),
        agent_a_budget=budget,
        agent_b_budget=budget,
    )

    # The late returned action is ignored.
    #
    # Therefore the illegal-action policy must never see BET_33.
    result = runner.run()

    assert result.num_valid_pairs == 1
    assert result.failed_runs == []

    assert len(
        result.timeout_events
    ) == 2


def test_check_if_legal_else_fold_checks_when_check_is_legal(
) -> None:
    schedule = DealSchedule(
        master_seed=42,
        num_hands=1,
    )

    hole_a, hole_b, board = (
        schedule.get_hand(0)
    )

    state = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=(
            hole_a,
            hole_b,
        ),
        board=board,
        dealer=0,
    )

    # BTN/SB calls.
    state = state.apply_action(
        Action.CALL
    )

    # BB checks.
    #
    # This advances to the flop, where BB acts first.
    state = state.apply_action(
        Action.CHECK
    )

    assert Action.CHECK in (
        state.legal_actions()
    )

    runner = PairedRunner(
        agent_a=CallCheckAgent("A"),
        agent_b=CallCheckAgent("B"),
        num_hands=1,
        seed=42,
        game_config=BASELINE_GAME_CONFIG,
    )

    action, profit, resolution = (
        runner._resolve_timeout(
            state=state,
            legal_actions=(
                state.legal_actions()
            ),
            timeout_policy=(
                TimeoutPolicy.CHECK_IF_LEGAL_ELSE_FOLD
            ),
        )
    )

    assert action == Action.CHECK
    assert profit is None
    assert resolution == "check"


def test_check_if_legal_else_fold_forfeits_when_check_is_not_legal(
) -> None:
    schedule = DealSchedule(
        master_seed=42,
        num_hands=1,
    )

    hole_a, hole_b, board = (
        schedule.get_hand(0)
    )

    state = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=(
            hole_a,
            hole_b,
        ),
        board=board,
        dealer=0,
    )

    legal_actions = (
        state.legal_actions()
    )

    assert Action.CHECK not in legal_actions
    assert Action.FOLD in legal_actions

    runner = PairedRunner(
        agent_a=CallCheckAgent("A"),
        agent_b=CallCheckAgent("B"),
        num_hands=1,
        seed=42,
        game_config=BASELINE_GAME_CONFIG,
    )

    action, profit, resolution = (
        runner._resolve_timeout(
            state=state,
            legal_actions=legal_actions,
            timeout_policy=(
                TimeoutPolicy.CHECK_IF_LEGAL_ELSE_FOLD
            ),
        )
    )

    assert action is None

    assert profit == pytest.approx(
        (
            -0.5,
            0.5,
        )
    )

    assert resolution == "forfeit"


def test_runner_persists_agent_budget_configuration(
) -> None:
    budget_a = AgentExecutionBudget(
        simulation_budget=100,
        time_budget_ms=250,
        timeout_policy=(
            TimeoutPolicy.FOLD
        ),
    )

    budget_b = AgentExecutionBudget(
        simulation_budget=1_000,
        time_budget_ms=500,
        timeout_policy=(
            TimeoutPolicy.CHECK_IF_LEGAL_ELSE_FOLD
        ),
    )

    runner = PairedRunner(
        agent_a=CallCheckAgent("A"),
        agent_b=CallCheckAgent("B"),
        num_hands=1,
        seed=999,
        game_config=BASELINE_GAME_CONFIG,
        agent_a_budget=budget_a,
        agent_b_budget=budget_b,
    )

    result = runner.run()

    metadata = (
        result.experiment_metadata
    )

    assert (
        metadata["agent_a_budget"]
        == budget_a.to_metadata()
    )

    assert (
        metadata["agent_b_budget"]
        == budget_b.to_metadata()
    )