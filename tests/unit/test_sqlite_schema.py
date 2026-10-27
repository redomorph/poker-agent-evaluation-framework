"""Unit tests for explicit SQLite schema-v1 management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pokerbench.persistence import (
    DatabaseNotInitializedError,
    SQLITE_SCHEMA_MIGRATION_ID,
    SQLITE_SCHEMA_VERSION,
    SQLiteBenchmarkStore,
    SchemaStateError,
    UnsupportedSchemaVersionError,
)


def test_initialize_creates_schema_v1(
    tmp_path: Path,
) -> None:
    database = (
        tmp_path
        / "benchmark.sqlite3"
    )

    store = SQLiteBenchmarkStore(
        database
    )

    assert not database.exists()

    store.initialize()

    assert database.is_file()

    with sqlite3.connect(
        database
    ) as connection:
        row = connection.execute(
            """
            SELECT
                version,
                migration_id
            FROM schema_version
            WHERE singleton = 1
            """
        ).fetchone()

        assert row == (
            SQLITE_SCHEMA_VERSION,
            SQLITE_SCHEMA_MIGRATION_ID,
        )

        tables = {
            row[0]
            for row
            in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

    assert "matches" in tables
    assert "hands" in tables
    assert "schema_version" in tables
    assert "alembic_version" in tables

    with sqlite3.connect(
        database
    ) as connection:
        alembic_row = connection.execute(
            """
            SELECT version_num
            FROM alembic_version
            """
        ).fetchone()

    assert alembic_row == (
        SQLITE_SCHEMA_MIGRATION_ID,
    )


def test_initialize_is_idempotent_for_valid_v1(
    tmp_path: Path,
) -> None:
    store = SQLiteBenchmarkStore(
        tmp_path
        / "benchmark.sqlite3"
    )

    store.initialize()
    store.initialize()


def test_normal_operations_do_not_initialize_database_implicitly(
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
        store.match_exists(
            "missing"
        )

    assert not database.exists()


def test_initialize_rejects_nonempty_unversioned_database(
    tmp_path: Path,
) -> None:
    database = (
        tmp_path
        / "benchmark.sqlite3"
    )

    with sqlite3.connect(
        database
    ) as connection:
        connection.execute(
            "CREATE TABLE legacy (id INTEGER PRIMARY KEY)"
        )

    store = SQLiteBenchmarkStore(
        database
    )

    with pytest.raises(
        SchemaStateError,
        match="no schema_version",
    ):
        store.initialize()


def test_unknown_schema_version_is_rejected_without_migration(
    tmp_path: Path,
) -> None:
    database = (
        tmp_path
        / "benchmark.sqlite3"
    )

    store = SQLiteBenchmarkStore(
        database
    )
    store.initialize()

    with sqlite3.connect(
        database
    ) as connection:
        connection.execute(
            """
            UPDATE schema_version
            SET version = 999
            WHERE singleton = 1
            """
        )
        connection.commit()

    with pytest.raises(
        UnsupportedSchemaVersionError,
        match="999",
    ):
        store.match_exists(
            "missing"
        )
