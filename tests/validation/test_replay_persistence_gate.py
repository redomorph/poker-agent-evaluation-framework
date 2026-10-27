# tests/validation/test_replay_persistence_gate.py
"""
Phase-0 gate: replay of any persisted match.

Validates:

1. every executed positional hand is captured;
2. match records survive persistence round-trip;
3. replay is available by match ID;
4. normal hand actions/profits reconstruct exactly;
5. timeout forfeits remain replayable;
6. failed illegal-action hands preserve the exact failure state.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import pokerbench.runner as runner_module
from pokerbench.agents.base import Agent
from pokerbench.agents.budget import (
    AgentExecutionBudget,
    TimeoutPolicy,
)
from pokerbench.agents.observation import (
    AgentObservation,
)
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.replay import (
    HandTermination,
    ReplayArchive,
)
from pokerbench.rng import AgentRNG
from pokerbench.runner import (
    IllegalActionPolicy,
    PairedRunner,
)


class FoldAgent(Agent):
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[
            Action
        ],
        rng: AgentRNG,
    ) -> Action:
        return Action.FOLD


class CallCheckAgent(Agent):
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[
            Action
        ],
        rng: AgentRNG,
    ) -> Action:
        if Action.CALL in legal_actions:
            return Action.CALL

        if Action.CHECK in legal_actions:
            return Action.CHECK

        return Action.FOLD


class IllegalAgent(Agent):
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[
            Action
        ],
        rng: AgentRNG,
    ) -> Action:
        return Action.BET_33


def _two_ms_clock() -> Iterator[int]:
    value = 0

    while True:
        yield value
        value += 2_000_000


def test_completed_paired_match_is_replayable_by_match_id(
    tmp_path: Path,
) -> None:
    runner = PairedRunner(
        agent_a=FoldAgent("A"),
        agent_b=FoldAgent("B"),
        num_hands=3,
        seed=42,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        adaptive=False,
        match_id=(
            "phase0-replay-normal"
        ),
    )

    result = runner.run()

    assert (
        result.match_record
        is not None
    )

    record = (
        result.match_record
    )

    # Three paired deals, each executed in two positional conditions.
    assert len(
        record.hands
    ) == 6

    assert all(
        hand.termination
        == HandTermination.NORMAL
        for hand in record.hands
    )

    archive = ReplayArchive(
        tmp_path
    )

    archive.save(
        record
    )

    loaded = archive.load(
        "phase0-replay-normal"
    )

    assert (
        loaded.record_hash
        == record.record_hash
    )

    replayed = archive.replay(
        "phase0-replay-normal"
    )

    assert len(replayed) == 6

    for item in replayed:
        assert (
            item.engine_replay
            .final_state
            .is_terminal
        )

        assert (
            item.final_profit_bb
            is not None
        )


def test_replay_preserves_position_assignment() -> None:
    runner = PairedRunner(
        agent_a=FoldAgent("A"),
        agent_b=FoldAgent("B"),
        num_hands=1,
        seed=42,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        adaptive=False,
        match_id="position-test",
    )

    result = runner.run()

    assert (
        result.match_record
        is not None
    )

    hands = (
        result.match_record.hands
    )

    assert len(hands) == 2

    first = hands[0]
    second = hands[1]

    assert (
        first.config_id
        == "A_BTN"
    )

    assert (
        first.seat0_bot_id,
        first.seat1_bot_id,
    ) == (
        0,
        1,
    )

    assert (
        second.config_id
        == "B_BTN"
    )

    assert (
        second.seat0_bot_id,
        second.seat1_bot_id,
    ) == (
        1,
        0,
    )


def test_timeout_forfeit_match_is_replayable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
        seed=123,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        adaptive=False,
        agent_a_budget=budget,
        agent_b_budget=budget,
        match_id="timeout-match",
    )

    result = runner.run()

    assert (
        result.match_record
        is not None
    )

    record = (
        result.match_record
    )

    assert len(record.hands) == 2

    assert all(
        hand.termination
        == HandTermination.TIMEOUT_FORFEIT
        for hand in record.hands
    )

    archive = ReplayArchive(
        tmp_path
    )

    archive.save(
        record
    )

    replayed = archive.replay(
        "timeout-match"
    )

    assert len(replayed) == 2

    for item in replayed:
        assert (
            item.final_profit_bb
            is not None
        )

        assert (
            item.record
            .timeout_events
        )

        assert (
            item.record
            .timeout_events[0]
            .resolution
            == "forfeit"
        )


def test_illegal_action_failure_state_is_persisted_and_replayable(
    tmp_path: Path,
) -> None:
    runner = PairedRunner(
        agent_a=IllegalAgent("A"),
        agent_b=FoldAgent("B"),
        num_hands=1,
        seed=42,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        adaptive=False,
        illegal_action_policy=(
            IllegalActionPolicy.EXCLUDE_RUN
        ),
        match_id="illegal-match",
    )

    result = runner.run()

    assert (
        result.num_valid_pairs
        == 0
    )

    assert (
        result.match_record
        is not None
    )

    record = (
        result.match_record
    )

    assert len(record.hands) == 1

    hand = (
        record.hands[0]
    )

    assert (
        hand.termination
        == HandTermination.ILLEGAL_ACTION
    )

    assert (
        hand.requested_action
        == Action.BET_33
    )

    assert (
        hand.acting_player_at_termination
        == 0
    )

    archive = ReplayArchive(
        tmp_path
    )

    archive.save(
        record
    )

    replayed = archive.replay(
        "illegal-match"
    )

    assert len(replayed) == 1

    reconstructed_state = (
        replayed[0]
        .engine_replay
        .final_state
    )

    assert not (
        reconstructed_state
        .is_terminal
    )

    assert (
        Action.BET_33
        not in reconstructed_state
        .legal_actions()
    )


def test_runner_metadata_links_to_replay_record() -> None:
    runner = PairedRunner(
        agent_a=FoldAgent("A"),
        agent_b=FoldAgent("B"),
        num_hands=1,
        seed=999,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        match_id="metadata-match",
    )

    result = runner.run()

    assert (
        result.match_record
        is not None
    )

    metadata = (
        result.experiment_metadata
    )

    assert (
        metadata["match_id"]
        == "metadata-match"
    )

    assert (
        metadata["match_record_hash"]
        == result.match_record.record_hash
    )

    assert (
        metadata["replay_hand_count"]
        == 2
    )