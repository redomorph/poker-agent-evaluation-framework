"""Durable benchmark persistence interfaces."""

from pokerbench.persistence.base import (
    MatchStore,
)

from pokerbench.persistence.errors import (
    DatabaseNotInitializedError,
    MatchAlreadyExistsError,
    MatchNotFoundError,
    PersistenceError,
    PersistenceIntegrityError,
    SchemaStateError,
    UnsupportedSchemaVersionError,
)
from pokerbench.persistence.schema import (
    SQLITE_SCHEMA_MIGRATION_ID,
    SQLITE_SCHEMA_VERSION,
)
from pokerbench.persistence.sqlite import (
    MatchPersistenceMetadata,
    SQLiteBenchmarkStore,
)


__all__ = [
    "DatabaseNotInitializedError",
    "MatchAlreadyExistsError",
    "MatchNotFoundError",
    "MatchPersistenceMetadata",
    "MatchStore",
    "PersistenceError",
    "PersistenceIntegrityError",
    "SQLITE_SCHEMA_MIGRATION_ID",
    "SQLITE_SCHEMA_VERSION",
    "SQLiteBenchmarkStore",
    "SchemaStateError",
    "UnsupportedSchemaVersionError",
]