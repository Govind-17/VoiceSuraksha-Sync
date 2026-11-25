"""Database engine and session configuration for VoiceSuraksha-Sync."""

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DB_URL = "sqlite:///./voicesuraksha.db"


def get_database_url() -> str:
    """Return the configured SQLite URL."""
    return os.getenv("VOICESURAKSHA_DB_URL", DEFAULT_DB_URL)


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


def init_db(target_engine=None) -> None:
    """Create all database tables if they do not exist."""
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=target_engine or engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a scoped SQLAlchemy session for FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
