"""SQLite schema v1 for durable poker benchmark persistence."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


SQLITE_SCHEMA_VERSION = 1
SQLITE_SCHEMA_MIGRATION_ID = "0001_sqlite_schema_v1"


_CREATE_SCHEMA_SQL = """
CREATE TABLE schema_version (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL,
    migration_id TEXT NOT NULL,
    applied_at_utc TEXT NOT NULL
);

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL PRIMARY KEY
);

CREATE TABLE matches (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    seed INTEGER NOT NULL,

    experiment_id TEXT,
    experiment_config_hash TEXT,
    experiment_config_json TEXT,

    game_config_hash TEXT NOT NULL,
    abstraction_version TEXT NOT NULL,

    bot_version TEXT,
    opponent_hash TEXT,

    hands_played INTEGER NOT NULL
        CHECK (hands_played >= 0),

    ev_bb_100 REAL,
    ci_low REAL,
    ci_high REAL,

    engine_version TEXT,
    python_version TEXT,
    rng_version TEXT,
    git_commit TEXT,

    starting_stack REAL NOT NULL,
    small_blind REAL NOT NULL,
    big_blind REAL NOT NULL,
    rake REAL NOT NULL,

    population_id TEXT,
    population_seed INTEGER,
    opponent_parameters_json TEXT,
    bot_configuration_json TEXT,

    statistical_method TEXT,
    sample_size INTEGER
        CHECK (sample_size IS NULL OR sample_size >= 0),
    precision_target_json TEXT,

    poetry_lock_hash TEXT,
    hardware_fingerprint TEXT,
    os_identifier TEXT,

    record_json TEXT NOT NULL,
    record_hash TEXT NOT NULL,

    created_at_utc TEXT NOT NULL
);

CREATE TABLE hands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    hand_ordinal INTEGER NOT NULL
        CHECK (hand_ordinal >= 0),

    hand_index INTEGER NOT NULL
        CHECK (hand_index >= 0),
    config_id TEXT NOT NULL,

    dealer INTEGER NOT NULL
        CHECK (dealer IN (0, 1)),
    seat0_bot_id INTEGER NOT NULL
        CHECK (seat0_bot_id IN (0, 1)),
    seat1_bot_id INTEGER NOT NULL
        CHECK (seat1_bot_id IN (0, 1)),

    street TEXT NOT NULL,

    hero_cards TEXT NOT NULL,
    villain_cards TEXT NOT NULL,
    board TEXT NOT NULL,

    actions_json TEXT NOT NULL,

    pot REAL NOT NULL
        CHECK (pot >= 0),
    ev_change REAL,

    termination TEXT NOT NULL,
    timeout_events_json TEXT NOT NULL,

    hand_record_json TEXT NOT NULL,
    hand_record_hash TEXT NOT NULL,

    FOREIGN KEY (match_id)
        REFERENCES matches(id)
        ON DELETE CASCADE,

    UNIQUE (match_id, hand_ordinal)
);

CREATE INDEX idx_hands_match_id
    ON hands(match_id);

CREATE INDEX idx_hands_match_street
    ON hands(match_id, street);

CREATE INDEX idx_hands_match_termination
    ON hands(match_id, termination);
"""


def utc_now_iso() -> str:
    """Return an explicit UTC timestamp for persistence metadata."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
    )


def create_schema_v1(
    connection: sqlite3.Connection,
) -> None:
    """
    Create schema v1 in an empty SQLite database.

    This function is deliberately not called implicitly by normal store
    operations. Schema creation is an explicit operation.
    """
    connection.executescript(
        _CREATE_SCHEMA_SQL
    )

    connection.execute(
        """
        INSERT INTO schema_version (
            singleton,
            version,
            migration_id,
            applied_at_utc
        )
        VALUES (1, ?, ?, ?)
        """,
        (
            SQLITE_SCHEMA_VERSION,
            SQLITE_SCHEMA_MIGRATION_ID,
            utc_now_iso(),
        ),
    )

    # Runtime bootstrap and Alembic must agree about the current revision.
    # This makes a directly initialized v1 database safe to migrate later.
    connection.execute(
        """
        INSERT INTO alembic_version (
            version_num
        )
        VALUES (?)
        """,
        (
            SQLITE_SCHEMA_MIGRATION_ID,
        ),
    )
