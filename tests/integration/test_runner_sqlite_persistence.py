"""Integration tests for PairedRunner -> SQLite persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokerbench.agents.base import Agent
from pokerbench.agents.observation import (
    AgentObservation,
)
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.persistence import (
    DatabaseNotInitializedError,
    MatchAlreadyExistsError,
    SQLiteBenchmarkStore,
)
from pokerbench.rng import AgentRNG
from pokerbench.runner import PairedRunner


MATCH_ID = "runner-sqlite-match"


class FoldAgent(Agent):
    """Deterministic legal fold agent for persistence integration tests."""

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
        del observation
        del rng

        if Action.FOLD not in legal_actions:
            raise AssertionError(
                "FoldAgent expected fold to be legal"
            )

        return Action.FOLD


def _runner(
    *,
    benchmark_store: SQLiteBenchmarkStore | None = None,
    match_id: str = MATCH_ID,
) -> PairedRunner:
    return PairedRunner(
        agent_a=FoldAgent("A"),
        agent_b=FoldAgent("B"),
        num_hands=2,
        seed=123,
        game_config=(
            BASELINE_GAME_CONFIG
        ),
        adaptive=False,
        match_id=match_id,
        benchmark_store=(
            benchmark_store
        ),
    )


def test_runner_without_store_preserves_existing_behavior() -> None:
    result = _runner().run()

    assert result.match_record is not None
    assert (
        result.match_record.match_id
        == MATCH_ID
    )


def test_runner_persists_exact_match_record(
    tmp_path: Path,
) -> None:
    store = SQLiteBenchmarkStore(
        tmp_path
        / "benchmark.sqlite3"
    )
    store.initialize()

    result = _runner(
        benchmark_store=store
    ).run()

    assert result.match_record is not None

    assert store.match_exists(
        MATCH_ID
    )

    persisted = store.load_match(
        MATCH_ID
    )

    assert (
        persisted
        == result.match_record
    )

    assert (
        persisted.record_hash
        == result.match_record.record_hash
    )


def test_runner_persisted_match_is_replayable(
    tmp_path: Path,
) -> None:
    store = SQLiteBenchmarkStore(
        tmp_path
        / "benchmark.sqlite3"
    )
    store.initialize()

    result = _runner(
        benchmark_store=store
    ).run()

    assert result.match_record is not None

    persisted = store.load_match(
        MATCH_ID
    )

    replayed = (
        persisted.replay_all()
    )

    assert replayed
    assert (
        len(replayed)
        == len(
            result.match_record.hands
        )
    )


def test_runner_does_not_initialize_database_implicitly(
    tmp_path: Path,
) -> None:
    database = (
        tmp_path
        / "benchmark.sqlite3"
    )

    store = SQLiteBenchmarkStore(
        database
    )

    with pytest.raises(
        DatabaseNotInitializedError,
    ):
        _runner(
            benchmark_store=store
        ).run()

    assert not database.exists()


def test_runner_propagates_duplicate_match_id_failure(
    tmp_path: Path,
) -> None:
    store = SQLiteBenchmarkStore(
        tmp_path
        / "benchmark.sqlite3"
    )
    store.initialize()

    _runner(
        benchmark_store=store
    ).run()

    with pytest.raises(
        MatchAlreadyExistsError,
        match=MATCH_ID,
    ):
        _runner(
            benchmark_store=store
        ).run()


def test_enabling_persistence_does_not_change_match_identity(
    tmp_path: Path,
) -> None:
    without_store = (
        _runner().run()
    )

    store = SQLiteBenchmarkStore(
        tmp_path
        / "benchmark.sqlite3"
    )
    store.initialize()

    with_store = (
        _runner(
            benchmark_store=store,
        ).run()
    )

    assert (
        without_store.match_record
        is not None
    )
    assert (
        with_store.match_record
        is not None
    )

    assert (
        without_store.match_record
        .match_id
        == with_store.match_record
        .match_id
    )

    assert (
        without_store.match_record
        .record_hash
        == with_store.match_record
        .record_hash
    )