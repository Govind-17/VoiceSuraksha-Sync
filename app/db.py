"""Database engine and session configuration for VoiceSuraksha-Sync."""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DB_URL = "sqlite:///./voicesuraksha.db"


def get_database_url() -> str:
    """Return the configured SQLite URL."""
    return os.getenv("VOICESURAKSHA_DB_URL", DEFAULT_DB_URL)


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy models."""


def create_db_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine configured for SQLite."""
    url = database_url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, future=True)


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
