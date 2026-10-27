"""Persistence-specific exception hierarchy."""

from __future__ import annotations


class PersistenceError(RuntimeError):
    """Base class for benchmark persistence failures."""


class DatabaseNotInitializedError(PersistenceError):
    """Raised when the SQLite database has no recognized schema."""


class UnsupportedSchemaVersionError(PersistenceError):
    """Raised when the database schema version is not supported."""


class SchemaStateError(PersistenceError):
    """Raised when the database schema is internally inconsistent."""


class MatchNotFoundError(PersistenceError):
    """Raised when a requested persisted match does not exist."""


class MatchAlreadyExistsError(PersistenceError):
    """Raised when an immutable match ID is already present."""


class PersistenceIntegrityError(PersistenceError):
    """Raised when persisted content fails deterministic integrity checks."""
