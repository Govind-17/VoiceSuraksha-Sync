"""Database engine, durability helpers, and schema migrations for VoiceSuraksha-Sync."""

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DB_URL = "sqlite:///./voicesuraksha.db"
CURRENT_SCHEMA_VERSION = 2


def get_database_url() -> str:
    """Return configured database URL (Render DATABASE_URL, VOICESURAKSHA_DB_URL, or local SQLite)."""
    url = (os.getenv("DATABASE_URL") or os.getenv("VOICESURAKSHA_DB_URL") or DEFAULT_DB_URL).strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url.removeprefix("postgres://")
    return url


def get_storage_backend(database_url: str | None = None) -> str:
    """Return human-readable storage backend identifier ('postgresql' or 'sqlite_ephemeral')."""
    url = (database_url or get_database_url()).lower()
    if url.startswith("postgresql"):
        return "postgresql"
    return "sqlite_ephemeral"


def is_storage_durable(database_url: str | None = None) -> bool:
    """Return True when backed by managed PostgreSQL rather than ephemeral local SQLite."""
    return get_storage_backend(database_url) == "postgresql"


def _ensure_sqlite_parent_dir(url: str) -> None:
    """Create parent directory for file-backed SQLite URLs (e.g., Render persistent disks)."""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        raw_path = url.removeprefix("sqlite:///")
        if raw_path:
            Path(raw_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy models."""


def create_db_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine configured for SQLite with WAL and busy timeout."""
    url = database_url or get_database_url()
    is_sqlite = url.startswith("sqlite")
    if is_sqlite:
        _ensure_sqlite_parent_dir(url)
    connect_args = {"check_same_thread": False, "timeout": 15} if is_sqlite else {}
    eng = create_engine(url, connect_args=connect_args, future=True)

    if is_sqlite:
        @event.listens_for(eng, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()

    return eng


engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def run_schema_migrations(target_engine=None) -> None:
    """Apply lightweight, idempotent schema migrations and record schema version."""
    eng = target_engine or engine
    inspector = inspect(eng)
    is_postgres = eng.dialect.name == "postgresql"
    bool_default = "FALSE" if is_postgres else "0"

    with eng.begin() as conn:
        for table_name in ("enrollments", "user_accounts"):
            if inspector.has_table(table_name):
                cols = {c["name"] for c in inspector.get_columns(table_name)}
                if "privileged_role" not in cols:
                    conn.execute(
                        text(
                            f"ALTER TABLE {table_name} ADD COLUMN privileged_role BOOLEAN NOT NULL DEFAULT {bool_default}"
                        )
                    )

        existing = conn.execute(
            text("SELECT version FROM schema_version WHERE version = :ver"),
            {"ver": CURRENT_SCHEMA_VERSION},
        ).scalar_one_or_none()
        if existing is None:
            conn.execute(
                text(
                    "INSERT INTO schema_version (version, description, applied_at) "
                    "VALUES (:ver, :desc, CURRENT_TIMESTAMP)"
                ),
                {
                    "ver": CURRENT_SCHEMA_VERSION,
                    "desc": "Add privileged_role subject attributes and schema_version tracking",
                },
            )


def init_db(target_engine=None) -> None:
    """Create all database tables and run lightweight migrations."""
    from app import models  # noqa: F401

    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)
    run_schema_migrations(eng)


def get_db() -> Generator[Session, None, None]:
    """Yield a scoped SQLAlchemy session for FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
