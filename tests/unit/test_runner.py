# tests/unit/test_runner.py
"""Unit tests for the paired runner."""

from __future__ import annotations

from typing import ClassVar

import pytest

import pokerbench.runner as runner_module
from pokerbench.agents.base import Agent
from pokerbench.agents.observation import (
    AgentObservation,
)
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.engine.deal_schedule import (
    DealSchedule,
)
from pokerbench.rng import AgentRNG
from pokerbench.runner import (
    IllegalActionPolicy,
    PairedRunner,
    StatisticalUnit,
)


class DeterministicAgent(Agent):
    """
    Agent that returns a predetermined action.

    If the predetermined action is unavailable, BET_33 is deliberately
    returned so tests can exercise illegal-action handling.
    """

    def __init__(
        self,
        name: str,
        action: Action = Action.FOLD,
        is_adaptive: bool = False,
    ) -> None:
        self.name = name
        self.action = action
        self.is_adaptive_flag = (
            is_adaptive
        )
        self.act_count = 0

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        self.act_count += 1

        if self.action not in legal_actions:
            return Action.BET_33

        return self.action

    @property
    def is_adaptive(self) -> bool:
        return self.is_adaptive_flag

    def reset(self) -> None:
        self.act_count = 0


class CallAgent(DeterministicAgent):
    """Agent that calls when possible and otherwise checks."""

    def __init__(
        self,
        name: str,
        is_adaptive: bool = False,
    ) -> None:
        super().__init__(
            name,
            Action.CALL,
            is_adaptive,
        )

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        self.act_count += 1

        if Action.CALL in legal_actions:
            return Action.CALL

        if Action.CHECK in legal_actions:
            return Action.CHECK

        return Action.FOLD


class FoldAgent(DeterministicAgent):
    def __init__(
        self,
        name: str,
        is_adaptive: bool = False,
    ) -> None:
        super().__init__(
            name,
            Action.FOLD,
            is_adaptive,
        )


class RaiseAgent(DeterministicAgent):
    def __init__(
        self,
        name: str,
        is_adaptive: bool = False,
    ) -> None:
        super().__init__(
            name,
            Action.BET_200,
            is_adaptive,
        )


class CountingDealSchedule(DealSchedule):
    """
    DealSchedule test double that counts constructor calls.

    This verifies schedule reuse structurally without relying on
    environment-dependent wall-clock timing.
    """

    construction_count: ClassVar[int] = 0

    def __init__(
        self,
        master_seed: int,
        num_hands: int,
    ) -> None:
        type(self).construction_count += 1

        super().__init__(
            master_seed=master_seed,
            num_hands=num_hands,
        )


def test_static_paired_runner() -> None:
    agent_a = FoldAgent("A")
    agent_b = FoldAgent("B")

    runner = PairedRunner(
        agent_a=agent_a,
        agent_b=agent_b,
        num_hands=1,
        seed=42,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
    )

    result = runner.run()

    assert (
        result.statistical_unit
        == StatisticalUnit.PAIRED_DEAL
    )
    assert result.num_valid_pairs == 1
    assert (
        len(result.paired_differences)
        == 1
    )

    # Identical policies must have zero position-controlled
    # A-vs-B performance difference.
    assert (
        abs(
            result.paired_differences[0]
        )
        < 1e-12
    )

    assert result.mean_d is not None
    assert (
        abs(result.mean_d)
        < 1e-12
    )

    assert result.std_d is None


def test_position_controlled_difference_compares_agents_in_both_positions(
) -> None:
    agent_a = FoldAgent("A")
    agent_b = FoldAgent("B")

    runner = PairedRunner(
        agent_a=agent_a,
        agent_b=agent_b,
        num_hands=1,
        seed=42,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
    )

    # Artificial example:
    #
    # Condition 1:
    #   A BTN = +3.0
    #   B BB  = -3.0
    #
    # Condition 2:
    #   B BTN = +1.5
    #   A BB  = -1.5
    #
    # BTN:
    #   A_BTN - B_BTN
    #   3.0 - 1.5
    #   = 1.5
    #
    # BB:
    #   A_BB - B_BB
    #   -1.5 - (-3.0)
    #   = 1.5
    #
    # Average:
    #   (1.5 + 1.5) / 2
    #   = 1.5
    difference = (
        runner._position_controlled_difference(
            profit_a_btn=3.0,
            profit_b_bb=-3.0,
            profit_b_btn=1.5,
            profit_a_bb=-1.5,
        )
    )

    assert difference == pytest.approx(
        1.5
    )


def test_adaptive_paired_runner() -> None:
    agent_a = CallAgent(
        "A",
        is_adaptive=True,
    )
    agent_b = CallAgent(
        "B",
        is_adaptive=True,
    )

    runner = PairedRunner(
        agent_a=agent_a,
        agent_b=agent_b,
        num_hands=2,
        seed=123,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
    )

    result = runner.run()

    assert (
        result.statistical_unit
        == StatisticalUnit.TRAJECTORY_CLUSTER
    )
    assert result.num_valid_pairs == 1
    assert (
        len(result.paired_differences)
        == 1
    )
    assert isinstance(
        result.paired_differences[0],
        float,
    )

    assert agent_a.act_count > 0
    assert agent_b.act_count > 0


def test_illegal_action_exclusion() -> None:
    agent_a = DeterministicAgent(
        "A",
        action=Action.BET_33,
    )
    agent_b = FoldAgent("B")

    runner = PairedRunner(
        agent_a=agent_a,
        agent_b=agent_b,
        num_hands=1,
        seed=42,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
        illegal_action_policy=(
            IllegalActionPolicy.EXCLUDE_RUN
        ),
    )

    result = runner.run()

    assert result.num_valid_pairs == 0
    assert len(result.failed_runs) == 1
    assert (
        result.paired_differences
        == []
    )


def test_illegal_action_raise_exception() -> None:
    agent_a = DeterministicAgent(
        "A",
        action=Action.BET_33,
    )
    agent_b = FoldAgent("B")

    runner = PairedRunner(
        agent_a=agent_a,
        agent_b=agent_b,
        num_hands=1,
        seed=42,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
        illegal_action_policy=(
            IllegalActionPolicy.RAISE_EXCEPTION
        ),
    )

    with pytest.raises(ValueError):
        runner.run()


def test_metadata_and_hash() -> None:
    agent_a = FoldAgent("A")
    agent_b = FoldAgent("B")

    runner = PairedRunner(
        agent_a=agent_a,
        agent_b=agent_b,
        num_hands=1,
        seed=123,
        game_config=BASELINE_GAME_CONFIG,
        experiment_metadata={
            "extra": "value",
        },
        adaptive=False,
    )

    result = runner.run()
    metadata = (
        result.experiment_metadata
    )

    assert metadata["seed"] == 123
    assert (
        metadata["extra"]
        == "value"
    )
    assert (
        "game_config_hash"
        in metadata
    )

    from pokerbench.engine.config import (
        config_hash,
    )

    expected_hash = config_hash(
        BASELINE_GAME_CONFIG
    )

    assert (
        metadata["game_config_hash"]
        == expected_hash
    )


def test_paired_conditions_keep_agent_rng_identity_stable(
) -> None:
    """
    The same experimental agent must receive the same RNG namespace
    in both paired positional conditions.

    Position / configuration must not alter the agent RNG identity.
    """
    rng_a_condition_1 = AgentRNG(
        master_seed=999,
        bot_id=0,
        hand_id=7,
        purpose="decision",
    )

    rng_a_condition_2 = AgentRNG(
        master_seed=999,
        bot_id=0,
        hand_id=7,
        purpose="decision",
    )

    assert (
        rng_a_condition_1.random()
        == rng_a_condition_2.random()
    )
    assert (
        rng_a_condition_1.random()
        == rng_a_condition_2.random()
    )
    assert (
        rng_a_condition_1.random()
        == rng_a_condition_2.random()
    )


def test_static_runner_constructs_deal_schedule_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Static execution must generate one complete schedule, not one
    increasingly large schedule prefix per hand.

    This protects the runner from the former O(N^2) schedule-generation
    behaviour.
    """
    CountingDealSchedule.construction_count = 0

    monkeypatch.setattr(
        runner_module,
        "DealSchedule",
        CountingDealSchedule,
    )

    runner = PairedRunner(
        agent_a=FoldAgent("A"),
        agent_b=FoldAgent("B"),
        num_hands=20,
        seed=42,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
    )

    result = runner.run()

    assert (
        result.num_valid_pairs
        == 20
    )
    assert (
        CountingDealSchedule.construction_count
        == 1
    )


def test_adaptive_runner_constructs_deal_schedule_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Both adaptive positional trajectories must reuse one immutable
    pre-generated deal schedule.

    This simultaneously protects performance and the paired CRN design.
    """
    CountingDealSchedule.construction_count = 0

    monkeypatch.setattr(
        runner_module,
        "DealSchedule",
        CountingDealSchedule,
    )

    runner = PairedRunner(
        agent_a=FoldAgent(
            "A",
            is_adaptive=True,
        ),
        agent_b=FoldAgent(
            "B",
            is_adaptive=True,
        ),
        num_hands=20,
        seed=123,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
    )

    result = runner.run()

    assert (
        result.statistical_unit
        == StatisticalUnit.TRAJECTORY_CLUSTER
    )
    assert result.num_valid_pairs == 1

    assert (
        CountingDealSchedule.construction_count
        == 1
    )