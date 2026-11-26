"""Database engine, durability helpers, and schema migrations for VoiceSuraksha-Sync."""

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DB_URL = "sqlite:///./voicesuraksha.db"
CURRENT_SCHEMA_VERSION = 5


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
    ts_type = "TIMESTAMP WITH TIME ZONE" if is_postgres else "DATETIME"

    migrations: dict[str, list[tuple[str, str]]] = {
        "enrollments": [
            ("privileged_role", f"BOOLEAN NOT NULL DEFAULT {bool_default}"),
            ("consent_text", "TEXT NOT NULL DEFAULT ''"),
            ("re_enrolled_at", f"{ts_type}"),
            ("enrollment_status", "VARCHAR(32) NOT NULL DEFAULT 'ACTIVE'"),
            ("duplicate_of_subject_ref", "VARCHAR(64)"),
            ("enrollment_reason_codes_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("enroll_deepfake_score", "FLOAT"),
            ("enroll_deepfake_threshold", "FLOAT"),
            ("enroll_deepfake_segments_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("enroll_deepfake_request_id", "VARCHAR(128)"),
            ("reviewed_by", "VARCHAR(64)"),
            ("review_reason", "TEXT"),
            ("reviewed_at", f"{ts_type}"),
        ],
        "user_accounts": [
            ("privileged_role", f"BOOLEAN NOT NULL DEFAULT {bool_default}"),
        ],
        "verification_sessions": [
            ("parent_session_id", "VARCHAR(64)"),
            ("challenge_attempt", "INTEGER NOT NULL DEFAULT 1"),
            ("sim_age_days", "INTEGER"),
            ("device_changed", f"BOOLEAN NOT NULL DEFAULT {bool_default}"),
            ("call_velocity_1h", "INTEGER NOT NULL DEFAULT 1"),
        ],
        "verification_telemetry": [
            ("perceptual_hash", "VARCHAR(32)"),
        ],
        "audit_events": [
            ("prev_hash", "VARCHAR(64)"),
            ("entry_hash", "VARCHAR(64)"),
        ],
    }

    with eng.begin() as conn:
        for table_name, col_specs in migrations.items():
            if inspector.has_table(table_name):
                existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
                for col_name, col_ddl in col_specs:
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_ddl}"))

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
                    "desc": "Add perceptual hash, session lineage, enrollment cooldown, and audit chain columns",
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
