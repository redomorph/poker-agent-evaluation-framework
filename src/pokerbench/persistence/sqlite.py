"""
SQLite-backed durable benchmark store.

Schema v1 deliberately stores both:

1. normalized MATCHES/HANDS columns for querying and reporting; and
2. canonical replay JSON plus SHA-256 hashes for exact MatchRecord recovery.

The normalized rows are validated against the canonical replay payload on
load. A MatchRecord is never reconstructed from lossy summary columns.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from pokerbench.engine.constants import CHIP_UNIT
from pokerbench.replay import (
    MatchRecord,
    ReplayIntegrityError,
)

from pokerbench.persistence.errors import (
    DatabaseNotInitializedError,
    MatchAlreadyExistsError,
    MatchNotFoundError,
    PersistenceIntegrityError,
    SchemaStateError,
    UnsupportedSchemaVersionError,
)
from pokerbench.persistence.schema import (
    SQLITE_SCHEMA_MIGRATION_ID,
    SQLITE_SCHEMA_VERSION,
    create_schema_v1,
    utc_now_iso,
)


@dataclass(frozen=True)
class MatchPersistenceMetadata:
    """
    Optional scientific/reporting metadata stored with one match.

    The exact replay payload remains MatchRecord. These fields provide the
    normalized experiment metadata required by the benchmark data model.
    """

    bot_version: str | None = None
    opponent_hash: str | None = None

    ev_bb_100: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    engine_version: str | None = None
    python_version: str | None = None
    rng_version: str | None = None
    git_commit: str | None = None

    population_id: str | None = None
    population_seed: int | None = None

    opponent_parameters: Mapping[str, object] | None = None
    bot_configuration: Mapping[str, object] | None = None

    statistical_method: str | None = None
    sample_size: int | None = None
    precision_target: Mapping[str, object] | None = None

    experiment_config: Mapping[str, object] | None = None

    poetry_lock_hash: str | None = None
    hardware_fingerprint: str | None = None
    os_identifier: str | None = None

    def __post_init__(self) -> None:
        if (
            self.sample_size is not None
            and self.sample_size < 0
        ):
            raise ValueError(
                "sample_size must be non-negative"
            )


class SQLiteBenchmarkStore:
    """
    Explicitly initialized SQLite benchmark store.

    Constructing the object never creates or migrates a database.

    Use initialize() exactly when schema creation is intended.
    Normal reads/writes validate schema v1 and never mutate an unknown schema.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Return the SQLite database path."""
        return self._path

    def initialize(self) -> None:
        """
        Explicitly initialize an empty database as schema v1.

        Calling initialize() on an already valid v1 database is idempotent.
        A non-empty unversioned database is rejected rather than modified.
        """
        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connect(
            create_if_missing=True,
        ) as connection:
            tables = _user_tables(
                connection
            )

            if not tables:
                create_schema_v1(
                    connection
                )
                connection.commit()
                return

            if (
                "schema_version"
                not in tables
            ):
                raise SchemaStateError(
                    "SQLite database contains tables "
                    "but has no schema_version table"
                )

            self._validate_schema(
                connection
            )

    def save_match(
        self,
        record: MatchRecord,
        *,
        metadata: MatchPersistenceMetadata | None = None,
    ) -> None:
        """
        Persist one immutable MatchRecord transactionally.

        Existing match IDs are never overwritten.
        """
        normalized_metadata = (
            metadata
            if metadata is not None
            else MatchPersistenceMetadata()
        )

        with self._connect() as connection:
            self._validate_schema(
                connection
            )

            if self._match_exists(
                connection,
                record.match_id,
            ):
                raise MatchAlreadyExistsError(
                    "Match already exists: "
                    f"{record.match_id}"
                )

            record_payload = (
                record.to_dict()
            )
            record_json = (
                _canonical_json(
                    record_payload
                )
            )
            record_hash = (
                _sha256_text(
                    record_json
                )
            )

            if (
                record_hash
                != record.record_hash
            ):
                raise PersistenceIntegrityError(
                    "MatchRecord record_hash does not "
                    "match its canonical payload"
                )

            experiment_config_json = (
                _canonical_json(
                    normalized_metadata
                    .experiment_config
                )
                if normalized_metadata
                .experiment_config
                is not None
                else None
            )

            experiment_config_hash = (
                _sha256_text(
                    experiment_config_json
                )
                if experiment_config_json
                is not None
                else record.experiment_config_hash
            )

            if (
                record.experiment_config_hash
                is not None
                and experiment_config_hash
                is not None
                and record.experiment_config_hash
                != experiment_config_hash
            ):
                raise PersistenceIntegrityError(
                    "Experiment configuration hash "
                    "does not match MatchRecord"
                )

            try:
                connection.execute(
                    """
                    INSERT INTO matches (
                        id,
                        schema_version,
                        seed,
                        experiment_id,
                        experiment_config_hash,
                        experiment_config_json,
                        game_config_hash,
                        abstraction_version,
                        bot_version,
                        opponent_hash,
                        hands_played,
                        ev_bb_100,
                        ci_low,
                        ci_high,
                        engine_version,
                        python_version,
                        rng_version,
                        git_commit,
                        starting_stack,
                        small_blind,
                        big_blind,
                        rake,
                        population_id,
                        population_seed,
                        opponent_parameters_json,
                        bot_configuration_json,
                        statistical_method,
                        sample_size,
                        precision_target_json,
                        poetry_lock_hash,
                        hardware_fingerprint,
                        os_identifier,
                        record_json,
                        record_hash,
                        created_at_utc
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        record.match_id,
                        SQLITE_SCHEMA_VERSION,
                        record.master_seed,
                        record.experiment_id,
                        experiment_config_hash,
                        experiment_config_json,
                        record.game_config_hash,
                        record.game_config.abstraction_version,
                        normalized_metadata.bot_version,
                        normalized_metadata.opponent_hash,
                        len(record.hands),
                        normalized_metadata.ev_bb_100,
                        normalized_metadata.ci_low,
                        normalized_metadata.ci_high,
                        normalized_metadata.engine_version,
                        normalized_metadata.python_version,
                        normalized_metadata.rng_version,
                        normalized_metadata.git_commit,
                        float(
                            record.game_config
                            .starting_stack_bb
                        ),
                        float(
                            record.game_config
                            .small_blind_bb
                        ),
                        float(
                            record.game_config
                            .big_blind_bb
                        ),
                        float(
                            record.game_config
                            .rake
                        ),
                        normalized_metadata.population_id,
                        normalized_metadata.population_seed,
                        _optional_canonical_json(
                            normalized_metadata
                            .opponent_parameters
                        ),
                        _optional_canonical_json(
                            normalized_metadata
                            .bot_configuration
                        ),
                        normalized_metadata.statistical_method,
                        normalized_metadata.sample_size,
                        _optional_canonical_json(
                            normalized_metadata
                            .precision_target
                        ),
                        normalized_metadata.poetry_lock_hash,
                        normalized_metadata.hardware_fingerprint,
                        normalized_metadata.os_identifier,
                        record_json,
                        record_hash,
                        utc_now_iso(),
                    ),
                )

                for hand_ordinal, hand in enumerate(
                    record.hands
                ):
                    replay = hand.replay(
                        record.game_config
                    )

                    initial_state = (
                        replay.initial_state
                    )
                    final_state = (
                        replay.final_state
                    )

                    hero_seat = (
                        0
                        if hand.seat0_bot_id
                        == 0
                        else 1
                    )
                    villain_seat = (
                        1 - hero_seat
                    )

                    hero_profit = (
                        hand.terminal_profit_bb[
                            hero_seat
                        ]
                        if hand.terminal_profit_bb
                        is not None
                        else None
                    )

                    hand_payload = (
                        hand.to_dict()
                    )
                    hand_record_json = (
                        _canonical_json(
                            hand_payload
                        )
                    )
                    hand_record_hash = (
                        _sha256_text(
                            hand_record_json
                        )
                    )

                    connection.execute(
                        """
                        INSERT INTO hands (
                            match_id,
                            hand_ordinal,
                            hand_index,
                            config_id,
                            dealer,
                            seat0_bot_id,
                            seat1_bot_id,
                            street,
                            hero_cards,
                            villain_cards,
                            board,
                            actions_json,
                            pot,
                            ev_change,
                            termination,
                            timeout_events_json,
                            hand_record_json,
                            hand_record_hash
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            record.match_id,
                            hand_ordinal,
                            hand.hand_index,
                            hand.config_id,
                            hand.dealer,
                            hand.seat0_bot_id,
                            hand.seat1_bot_id,
                            final_state.street.value,
                            _canonical_json(
                                list(
                                    initial_state
                                    .hole_cards[
                                        hero_seat
                                    ]
                                )
                            ),
                            _canonical_json(
                                list(
                                    initial_state
                                    .hole_cards[
                                        villain_seat
                                    ]
                                )
                            ),
                            _canonical_json(
                                list(
                                    initial_state.board
                                )
                            ),
                            _canonical_json(
                                [
                                    action.value
                                    for action
                                    in replay.actions
                                ]
                            ),
                            (
                                final_state.pot
                                / CHIP_UNIT
                            ),
                            hero_profit,
                            hand.termination.value,
                            _canonical_json(
                                [
                                    event.to_dict()
                                    for event
                                    in hand.timeout_events
                                ]
                            ),
                            hand_record_json,
                            hand_record_hash,
                        ),
                    )

                connection.commit()

            except Exception:
                connection.rollback()
                raise

    def list_match_ids(
        self,
    ) -> tuple[str, ...]:
        """Return all persisted match IDs in canonical lexical order."""
        with self._connect() as connection:
            self._validate_schema(
                connection
            )

            rows = connection.execute(
                """
                SELECT id
                FROM matches
                ORDER BY id ASC
                """
            ).fetchall()

            return tuple(
                str(row["id"])
                for row in rows
            )

    def load_match_metadata(
        self,
        match_id: str,
    ) -> MatchPersistenceMetadata:
        """Load normalized optional scientific metadata for one match."""
        with self._connect() as connection:
            self._validate_schema(
                connection
            )

            row = connection.execute(
                """
                SELECT
                    bot_version,
                    opponent_hash,
                    ev_bb_100,
                    ci_low,
                    ci_high,
                    engine_version,
                    python_version,
                    rng_version,
                    git_commit,
                    population_id,
                    population_seed,
                    opponent_parameters_json,
                    bot_configuration_json,
                    statistical_method,
                    sample_size,
                    precision_target_json,
                    experiment_config_json,
                    experiment_config_hash,
                    poetry_lock_hash,
                    hardware_fingerprint,
                    os_identifier
                FROM matches
                WHERE id = ?
                """,
                (
                    match_id,
                ),
            ).fetchone()

            if row is None:
                raise MatchNotFoundError(
                    f"Match not found: {match_id}"
                )

            opponent_parameters = (
                _optional_mapping_from_json(
                    row["opponent_parameters_json"],
                    field_name="opponent_parameters_json",
                )
            )
            bot_configuration = (
                _optional_mapping_from_json(
                    row["bot_configuration_json"],
                    field_name="bot_configuration_json",
                )
            )
            precision_target = (
                _optional_mapping_from_json(
                    row["precision_target_json"],
                    field_name="precision_target_json",
                )
            )
            experiment_config = (
                _optional_mapping_from_json(
                    row["experiment_config_json"],
                    field_name="experiment_config_json",
                )
            )

            if experiment_config is not None:
                canonical_config = _canonical_json(
                    experiment_config
                )
                persisted_config_json = str(
                    row["experiment_config_json"]
                )
                if canonical_config != persisted_config_json:
                    raise PersistenceIntegrityError(
                        "Persisted experiment configuration is not canonical"
                    )

                stored_config_hash = row["experiment_config_hash"]
                if stored_config_hash is None:
                    raise PersistenceIntegrityError(
                        "Persisted experiment configuration has no hash"
                    )
                if (
                    _sha256_text(canonical_config)
                    != str(stored_config_hash)
                ):
                    raise PersistenceIntegrityError(
                        "Persisted experiment configuration hash mismatch"
                    )

            return MatchPersistenceMetadata(
                bot_version=_optional_text(row["bot_version"]),
                opponent_hash=_optional_text(row["opponent_hash"]),
                ev_bb_100=_optional_float(row["ev_bb_100"]),
                ci_low=_optional_float(row["ci_low"]),
                ci_high=_optional_float(row["ci_high"]),
                engine_version=_optional_text(row["engine_version"]),
                python_version=_optional_text(row["python_version"]),
                rng_version=_optional_text(row["rng_version"]),
                git_commit=_optional_text(row["git_commit"]),
                population_id=_optional_text(row["population_id"]),
                population_seed=_optional_int(row["population_seed"]),
                opponent_parameters=opponent_parameters,
                bot_configuration=bot_configuration,
                statistical_method=_optional_text(row["statistical_method"]),
                sample_size=_optional_int(row["sample_size"]),
                precision_target=precision_target,
                experiment_config=experiment_config,
                poetry_lock_hash=_optional_text(row["poetry_lock_hash"]),
                hardware_fingerprint=_optional_text(row["hardware_fingerprint"]),
                os_identifier=_optional_text(row["os_identifier"]),
            )

    def load_match(
        self,
        match_id: str,
    ) -> MatchRecord:
        """
        Load and integrity-check one persisted MatchRecord.

        Both the canonical match payload and normalized hand rows are checked.
        """
        with self._connect() as connection:
            self._validate_schema(
                connection
            )

            row = connection.execute(
                """
                SELECT
                    record_json,
                    record_hash,
                    hands_played
                FROM matches
                WHERE id = ?
                """,
                (
                    match_id,
                ),
            ).fetchone()

            if row is None:
                raise MatchNotFoundError(
                    f"Match not found: {match_id}"
                )

            record_json = str(
                row["record_json"]
            )
            stored_record_hash = str(
                row["record_hash"]
            )
            hands_played = int(
                row["hands_played"]
            )

            calculated_hash = (
                _sha256_text(
                    record_json
                )
            )

            if (
                calculated_hash
                != stored_record_hash
            ):
                raise PersistenceIntegrityError(
                    "Persisted match record_hash mismatch"
                )

            try:
                raw_record = json.loads(
                    record_json
                )

            except json.JSONDecodeError as exc:
                raise PersistenceIntegrityError(
                    "Persisted match JSON is invalid"
                ) from exc

            if not isinstance(
                raw_record,
                dict,
            ):
                raise PersistenceIntegrityError(
                    "Persisted match JSON root "
                    "must be an object"
                )

            try:
                record = MatchRecord.from_dict(
                    raw_record
                )

            except ReplayIntegrityError as exc:
                raise PersistenceIntegrityError(
                    "Persisted MatchRecord failed "
                    "replay integrity validation"
                ) from exc

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise PersistenceIntegrityError(
                    "Persisted MatchRecord is invalid"
                ) from exc

            if (
                record.match_id
                != match_id
            ):
                raise PersistenceIntegrityError(
                    "Persisted MatchRecord ID does not "
                    "match database key"
                )

            if (
                record.record_hash
                != stored_record_hash
            ):
                raise PersistenceIntegrityError(
                    "Loaded MatchRecord hash does not "
                    "match database hash"
                )

            if (
                len(record.hands)
                != hands_played
            ):
                raise PersistenceIntegrityError(
                    "hands_played does not match "
                    "the MatchRecord hand count"
                )

            self._validate_hand_rows(
                connection,
                record,
            )

            return record

    def match_exists(
        self,
        match_id: str,
    ) -> bool:
        """Return whether a match exists in a valid schema-v1 database."""
        with self._connect() as connection:
            self._validate_schema(
                connection
            )
            return self._match_exists(
                connection,
                match_id,
            )

    def _validate_hand_rows(
        self,
        connection: sqlite3.Connection,
        record: MatchRecord,
    ) -> None:
        rows = connection.execute(
            """
            SELECT
                hand_ordinal,
                hand_record_json,
                hand_record_hash
            FROM hands
            WHERE match_id = ?
            ORDER BY hand_ordinal
            """,
            (
                record.match_id,
            ),
        ).fetchall()

        if (
            len(rows)
            != len(record.hands)
        ):
            raise PersistenceIntegrityError(
                "Normalized hand-row count does not "
                "match MatchRecord"
            )

        for expected_ordinal, (
            row,
            hand,
        ) in enumerate(
            zip(
                rows,
                record.hands,
                strict=True,
            )
        ):
            hand_ordinal = int(
                row["hand_ordinal"]
            )

            if (
                hand_ordinal
                != expected_ordinal
            ):
                raise PersistenceIntegrityError(
                    "Persisted hand ordinals are not "
                    "contiguous and ordered"
                )

            hand_json = str(
                row["hand_record_json"]
            )
            stored_hash = str(
                row["hand_record_hash"]
            )

            calculated_hash = (
                _sha256_text(
                    hand_json
                )
            )

            if (
                calculated_hash
                != stored_hash
            ):
                raise PersistenceIntegrityError(
                    "Persisted hand record hash mismatch"
                )

            expected_json = (
                _canonical_json(
                    hand.to_dict()
                )
            )

            if (
                hand_json
                != expected_json
            ):
                raise PersistenceIntegrityError(
                    "Normalized hand payload does not "
                    "match MatchRecord"
                )

    def _validate_schema(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        tables = _user_tables(
            connection
        )

        if (
            "schema_version"
            not in tables
        ):
            raise DatabaseNotInitializedError(
                "SQLite benchmark database is not initialized"
            )

        rows = connection.execute(
            """
            SELECT
                version,
                migration_id
            FROM schema_version
            WHERE singleton = 1
            """
        ).fetchall()

        if len(rows) != 1:
            raise SchemaStateError(
                "schema_version must contain exactly "
                "one singleton row"
            )

        version = int(
            rows[0]["version"]
        )
        migration_id = str(
            rows[0]["migration_id"]
        )

        if (
            version
            != SQLITE_SCHEMA_VERSION
        ):
            raise UnsupportedSchemaVersionError(
                "Unsupported SQLite schema version: "
                f"{version}; expected "
                f"{SQLITE_SCHEMA_VERSION}"
            )

        if (
            migration_id
            != SQLITE_SCHEMA_MIGRATION_ID
        ):
            raise SchemaStateError(
                "Schema migration ID does not match "
                "schema version"
            )

        required_tables = {
            "schema_version",
            "alembic_version",
            "matches",
            "hands",
        }

        missing = (
            required_tables
            - tables
        )

        if missing:
            raise SchemaStateError(
                "SQLite schema is missing required tables: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        alembic_rows = connection.execute(
            """
            SELECT version_num
            FROM alembic_version
            """
        ).fetchall()

        if len(alembic_rows) != 1:
            raise SchemaStateError(
                "alembic_version must contain exactly "
                "one revision row"
            )

        alembic_revision = str(
            alembic_rows[0][
                "version_num"
            ]
        )

        if (
            alembic_revision
            != SQLITE_SCHEMA_MIGRATION_ID
        ):
            raise SchemaStateError(
                "Alembic revision does not match "
                "the schema migration ID"
            )

    def _match_exists(
        self,
        connection: sqlite3.Connection,
        match_id: str,
    ) -> bool:
        row = connection.execute(
            """
            SELECT 1
            FROM matches
            WHERE id = ?
            """,
            (
                match_id,
            ),
        ).fetchone()

        return (
            row
            is not None
        )

    def _connect(
        self,
        *,
        create_if_missing: bool = False,
    ) -> sqlite3.Connection:
        if (
            not create_if_missing
            and not self._path.is_file()
        ):
            raise DatabaseNotInitializedError(
                "SQLite benchmark database "
                "does not exist"
            )

        connection = sqlite3.connect(
            self._path
        )
        connection.row_factory = (
            sqlite3.Row
        )
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )
        return connection


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_mapping_from_json(
    raw: object,
    *,
    field_name: str,
) -> Mapping[str, object] | None:
    if raw is None:
        return None

    text = str(raw)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PersistenceIntegrityError(
            f"Persisted {field_name} is invalid JSON"
        ) from exc

    if not isinstance(value, dict):
        raise PersistenceIntegrityError(
            f"Persisted {field_name} must be a JSON object"
        )

    canonical = _canonical_json(value)
    if canonical != text:
        raise PersistenceIntegrityError(
            f"Persisted {field_name} is not canonical JSON"
        )

    return value


def _user_tables(
    connection: sqlite3.Connection,
) -> set[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE
            type = 'table'
            AND name NOT LIKE 'sqlite_%'
        """
    ).fetchall()

    return {
        str(
            row["name"]
        )
        for row
        in rows
    }


def _canonical_json(
    value: object,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    )


def _optional_canonical_json(
    value: Mapping[str, object] | None,
) -> str | None:
    if value is None:
        return None

    return _canonical_json(
        value
    )


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()
