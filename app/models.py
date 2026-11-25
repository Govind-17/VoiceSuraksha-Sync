"""SQLite ORM models for VoiceSuraksha-Sync.

Raw audio bytes are never persisted in any table.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utcnow() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Enrollment(Base):
    """Pseudonymous speaker enrollment metadata (no raw audio stored)."""

    __tablename__ = "enrollments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_ref: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    provider_speaker_id: Mapped[str] = mapped_column(String(128), nullable=False)
    consent_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    consent_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    audio_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    speech_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    privileged_role: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    re_enrolled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class VerificationSession(Base):
    """Short-lived, single-use action-bound verification session."""

    __tablename__ = "verification_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sector: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    subject_ref: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    challenge_text: Mapped[str] = mapped_column(Text, nullable=False)
    challenge_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    after_hours: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    privileged_role: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    amount_inr: Mapped[float | None] = mapped_column(Float, nullable=True)
    sim_age_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    device_changed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    call_velocity_1h: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    parent_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    challenge_attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class VoiceResult(Base):
    """Acoustic verification and deepfake detection evidence returned by DetectifAI."""

    __tablename__ = "voice_results"

    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("verification_sessions.session_id"),
        primary_key=True,
    )
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    operation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    speaker_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    speaker_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    speaker_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    deepfake_verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    deepfake_is_synthetic: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    deepfake_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    deepfake_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    segments_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    phrase_match: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class Decision(Base):
    """Deterministic authorization decision and one-time action-bound token hash."""

    __tablename__ = "decisions"

    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("verification_sessions.session_id"),
        primary_key=True,
    )
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_codes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    next_step_script: Mapped[str] = mapped_column(Text, nullable=False)
    token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RiskEvent(Base):
    """Cross-domain risk signal (e.g., simulated SIM_SWAP_ATTEMPT_BLOCKED)."""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_sector: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_ref: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    source_case_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    risk_state: Mapped[str] = mapped_column(String(32), default="ACTIVE", nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    demo_flag: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class AuditEvent(Base):
    """Immutable tamper-evident audit trail entry linked via SHA-256 hash chain."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    sector: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_codes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    prev_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entry_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class UserAccount(Base):
    """Registered end-user account bound to a pseudonymous subject_ref."""

    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    subject_ref: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    privileged_role: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class VerificationTelemetry(Base):
    """Additive telemetry log for audio SHA-256/perceptual hashes and provider credits (zero raw audio)."""

    __tablename__ = "verification_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    event_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    audio_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    perceptual_hash: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    credits_balance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RiskWebhookDispatch(Base):
    """Cryptographically HMAC-SHA256 signed webhook payloads attached to RiskEvents."""

    __tablename__ = "risk_webhook_dispatches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    risk_event_id: Mapped[int] = mapped_column(Integer, unique=True, index=True, nullable=False)
    hmac_signature: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SchemaVersion(Base):
    """Schema migration tracking table for zero-downtime redeploys."""

    __tablename__ = "schema_version"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    description: Mapped[str] = mapped_column(String(256), nullable=False)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RateLimitCounter(Base):
    """DB-backed rate-limit and lockout persistence."""

    __tablename__ = "rate_limit_counters"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
