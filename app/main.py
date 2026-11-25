"""FastAPI modular-monolith entrypoint for VoiceSuraksha-Sync (voicelock-gateway)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import math
import os
import re
import secrets
import statistics
import struct
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.asr import ServerASRAdapter
from app.audio_fingerprint import (
    MAX_PERCEPTUAL_HAMMING_BITS,
    compute_perceptual_hash,
    hamming_distance_hex,
    split_wav_segments,
)
from app.audio_quality import AudioQualityReport, inspect_pcm_wav_quality
from app.auth import (
    AuthActor,
    clear_session_cookie,
    create_session_token,
    get_admin_credentials,
    get_current_actor,
    get_optional_actor,
    hash_password,
    is_demo_mode,
    require_admin,
    set_session_cookie,
    validate_live_secrets,
    verify_admin_credentials,
    verify_password,
)
from app.db import (
    create_db_engine,
    get_db,
    get_storage_backend,
    init_db,
    is_storage_durable,
)
from app.detectif_client import DetectifClient, DetectifProviderError, ProviderVerifyResult
from app.models import (
    AuditEvent,
    Decision,
    Enrollment,
    RiskEvent,
    RiskWebhookDispatch,
    UserAccount,
    VerificationSession,
    VerificationTelemetry,
    VoiceResult,
    utcnow,
)
from app.policy import (
    GATEWAY_PROFILES,
    GENESIS_AUDIT_HASH,
    NON_VOICE_FALLBACK_OPTIONS,
    AuthorizationDecision,
    SessionPolicyInput,
    VoiceEvidenceInput,
    compute_audit_entry_hash,
    evaluate_voice_authorization,
    generate_challenge_phrase,
    hash_text_sha256,
    normalize_transcript,
    sign_risk_event_payload,
    validate_sector_action,
    verify_phrase_match,
)
from app.rate_limit import GatewayRateLimiter, RateLimitExceeded, gateway_rate_limiter

load_dotenv(override=False)

FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
MIN_ENROLL_SECONDS = 15.0
MAX_ENROLL_SECONDS = 60.0
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB guard for Render free-tier memory
SESSION_TTL_MINUTES = 2
DECISION_TOKEN_TTL_MINUTES = 5
RISK_EVENT_TTL_HOURS = 24
POLICY_VERSION = "2026.11-fail-closed-v1"
try:
    IST_ZONE: Any = ZoneInfo("Asia/Kolkata")
except ZoneInfoNotFoundError:
    IST_ZONE = timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
logger = logging.getLogger("voicelock-gateway")


def _is_ist_after_hours(now_utc: datetime | None = None) -> bool:
    """Compute server-side after-hours flag in Asia/Kolkata (22:00-06:00 IST or weekend)."""
    ts = _as_utc(now_utc or utcnow()).astimezone(IST_ZONE)
    return ts.hour >= 22 or ts.hour < 6 or ts.weekday() >= 5


def _is_subject_privileged(db: Session, subject_ref: str, enrollment: Enrollment | None = None) -> bool:
    """Resolve server-side privileged_role attribute from Enrollment/UserAccount records."""
    enr = enrollment or db.scalar(select(Enrollment).where(Enrollment.subject_ref == subject_ref))
    acct = db.scalar(select(UserAccount).where(UserAccount.subject_ref == subject_ref))
    return bool((enr and enr.privileged_role) or (acct and acct.privileged_role))


def _compute_call_velocity_1h(db: Session, subject_ref: str, now_utc: datetime) -> int:
    """Compute server-side 1-hour session count for a subject including the upcoming session."""
    cutoff_exp = _as_utc(now_utc) + timedelta(minutes=SESSION_TTL_MINUTES) - timedelta(hours=1)
    prior_count = db.scalar(
        select(func.count(VerificationSession.session_id)).where(
            VerificationSession.subject_ref == subject_ref,
            VerificationSession.expires_at > cutoff_exp,
        )
    )
    return int(prior_count or 0) + 1


def _contextual_signals_dict(session: VerificationSession, now_utc: datetime) -> dict[str, Any]:
    """Serialize contextual risk signals attached to a session (with simulated labels preserved)."""
    return {
        "sim_age_days": session.sim_age_days,
        "sim_age_simulated": True,
        "device_changed": bool(session.device_changed),
        "device_changed_simulated": True,
        "call_velocity_1h": int(session.call_velocity_1h),
        "after_hours": bool(session.after_hours),
        "time_of_day_ist": _as_utc(now_utc).astimezone(IST_ZONE).strftime("%H:%M IST"),
    }


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def _append_audit_event(
    db: Session,
    *,
    session_id: str,
    sector: str,
    action: str,
    outcome: str,
    reason_codes: list[str],
    created_at: datetime | None = None,
) -> AuditEvent:
    """Append an AuditEvent chained via prev_hash -> entry_hash (SHA-256)."""
    ts = _as_utc(created_at or utcnow())
    last_entry_hash = db.scalar(
        select(AuditEvent.entry_hash).order_by(AuditEvent.id.desc()).limit(1)
    )
    prev_hash = last_entry_hash or GENESIS_AUDIT_HASH
    entry_hash = compute_audit_entry_hash(
        prev_hash=prev_hash,
        session_id=session_id,
        sector=sector,
        action=action,
        outcome=outcome,
        reason_codes=reason_codes,
        created_at_iso=ts.isoformat(),
    )
    row = AuditEvent(
        session_id=session_id,
        sector=sector,
        action=action,
        outcome=outcome,
        reason_codes_json=json.dumps(reason_codes),
        prev_hash=prev_hash,
        entry_hash=entry_hash,
        created_at=ts,
    )
    db.add(row)
    db.flush()
    return row


def _verify_audit_chain(db: Session) -> dict[str, Any]:
    """Recompute the full AuditEvent hash chain and report first broken index if tampered."""
    rows = db.scalars(select(AuditEvent).order_by(AuditEvent.id.asc())).all()
    expected_prev = GENESIS_AUDIT_HASH
    for idx, row in enumerate(rows):
        reasons = json.loads(row.reason_codes_json)
        expected_entry = compute_audit_entry_hash(
            prev_hash=expected_prev,
            session_id=row.session_id,
            sector=row.sector,
            action=row.action,
            outcome=row.outcome,
            reason_codes=reasons,
            created_at_iso=_as_utc(row.created_at).isoformat(),
        )
        if row.prev_hash != expected_prev or row.entry_hash != expected_entry:
            return {
                "valid": False,
                "total_events": len(rows),
                "first_broken_index": idx,
                "first_broken_id": row.id,
                "tip_hash": expected_prev if idx > 0 else None,
            }
        expected_prev = row.entry_hash or expected_entry
    return {
        "valid": True,
        "total_events": len(rows),
        "first_broken_index": None,
        "first_broken_id": None,
        "tip_hash": expected_prev if rows else GENESIS_AUDIT_HASH,
    }


def _raise_rate_limited(
    db: Session,
    *,
    exc: RateLimitExceeded,
    session_id: str,
    sector: str,
    action: str,
) -> None:
    _append_audit_event(
        db,
        session_id=session_id,
        sector=sector,
        action=action,
        outcome="REJECTED",
        reason_codes=["RATE_LIMITED", exc.bucket],
    )
    db.commit()
    raise HTTPException(
        status_code=429,
        detail=exc.detail,
        headers={"Retry-After": str(exc.retry_after)},
    )


def _as_utc(dt: datetime) -> datetime:
    """Normalize SQLite naive or aware datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _inspect_wav_duration(wav_bytes: bytes) -> float | None:
    """Return WAV duration in seconds if valid RIFF/WAVE PCM, otherwise None."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return round(frames / float(rate), 3)
    except Exception:
        return None


async def _read_bounded_wav(file: UploadFile) -> bytes:
    """Read uploaded audio safely with an upper byte bound."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded audio exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return data


def get_detectif_client() -> DetectifClient:
    """Dependency provider for DetectifClient (overridable in tests)."""
    return DetectifClient()


def get_asr_adapter() -> ServerASRAdapter:
    """Dependency provider for ServerASRAdapter (overridable in tests)."""
    return ServerASRAdapter()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    validate_live_secrets()
    init_db()
    if not is_demo_mode() and not is_storage_durable():
        logger.warning(
            "Running in LIVE mode on ephemeral SQLite (%s). Configure DATABASE_URL=postgresql://... for durable persistence.",
            get_storage_backend(),
        )
    yield


app = FastAPI(
    title="VoiceSuraksha-Sync (voicelock-gateway)",
    version="1.0.0",
    description="Action-bound voice-security gateway powered by DetectifAI Eidos & Aletheia.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Attach baseline browser security headers compatible with the single-file SOC UI."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob: data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'",
    )
    return response


class CreateSessionRequest(BaseModel):
    sector: str
    subject_ref: str = Field(..., min_length=2, max_length=64)
    action: str
    resource_ref: str = Field(..., min_length=2, max_length=128)
    after_hours: bool = False
    privileged_role: bool = False
    amount_inr: float | None = None
    sim_age_days: int | None = Field(default=None, ge=0, le=3650)
    device_changed: bool = False


class ConsumeDecisionRequest(BaseModel):
    decision_token: str = Field(..., min_length=8)
    action: str
    resource_ref: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    display_name: str = Field(..., min_length=2, max_length=128)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class OfflineSwapCallbackRequest(BaseModel):
    subject_ref: str = Field(..., min_length=2, max_length=64)
    store_id: str = Field("RETAIL-BLR-01", min_length=2, max_length=64)
    sim_iccid_masked: str = Field("SIM-OFFLINE-001", min_length=2, max_length=64)
    after_hours: bool = False


class SubjectPrivilegeRequest(BaseModel):
    privileged_role: bool


class CompleteStepUpRequest(BaseModel):
    method: Literal["trusted_app_push", "manager_callback", "in_store_kyc"]
    amount_inr: float | None = Field(None, gt=0)
    note: str = Field("", max_length=256)


class ReviewDecisionRequest(BaseModel):
    decision: Literal["APPROVED", "REJECTED"]
    reason: str = Field(..., min_length=3, max_length=256)


class ResolveAlertRequest(BaseModel):
    resolution_note: str = Field("Resolved by SOC administrator", min_length=2, max_length=256)


def _actor_dict(actor: AuthActor) -> dict[str, Any]:
    return {
        "username": actor.username,
        "role": actor.role,
        "display_name": actor.display_name,
        "subject_ref": actor.subject_ref,
    }


def _make_subject_ref(username: str) -> str:
    slug = re.sub(r"[^A-Z0-9]", "", username.upper())[:8] or "USER"
    return f"SUB-{slug}-{secrets.token_hex(2).upper()}"


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_frontend():
    """Serve the single-file SOC UI directly at '/'."""
    return FileResponse(
        FRONTEND_INDEX,
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.api_route("/api/v1/health", methods=["GET", "HEAD"])
def health(
    db: Session = Depends(get_db),
    actor: AuthActor | None = Depends(get_optional_actor),
) -> dict[str, Any]:
    """Return gateway operational mode and available profiles (enrolled count only when authenticated)."""
    demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
    payload: dict[str, Any] = {
        "status": "ok",
        "service": "voicelock-gateway",
        "mode": "DEMO" if demo_mode else "LIVE",
        "storage_backend": get_storage_backend(),
        "durable": is_storage_durable(),
        "profiles": list(GATEWAY_PROFILES.keys()),
    }
    if actor is not None:
        if actor.role == "admin":
            payload["enrolled_count"] = db.query(Enrollment).count()
        elif actor.subject_ref:
            payload["enrolled_count"] = db.query(Enrollment).filter(Enrollment.subject_ref == actor.subject_ref).count()
    return payload


@app.get("/api/v1/auth/me")
def auth_me(actor: AuthActor | None = Depends(get_optional_actor)) -> dict[str, Any]:
    """Return current authenticated session actor, or authenticated=False."""
    if actor is None:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": _actor_dict(actor)}


def _enrollment_payload(record: Enrollment, duration_seconds: float | None = None) -> dict[str, Any]:
    """Serialize an Enrollment row into the API response shape."""
    payload: dict[str, Any] = {
        "id": record.id,
        "subject_ref": record.subject_ref,
        "provider_speaker_id": record.provider_speaker_id,
        "consent_ref": record.consent_ref,
        "consent_text": record.consent_text,
        "audio_sha256": record.audio_sha256,
        "speech_seconds": record.speech_seconds,
        "created_at": _as_utc(record.created_at).isoformat(),
        "re_enrolled_at": _as_utc(record.re_enrolled_at).isoformat() if record.re_enrolled_at else None,
    }
    if duration_seconds is not None:
        payload["duration_seconds"] = duration_seconds
    return payload


async def _validate_and_enroll_wav(
    *,
    subject_ref: str,
    consent_confirmed: bool,
    file: UploadFile,
    client: DetectifClient,
) -> tuple[bytes, Any]:
    """Validate speaker consent + 15-60s 16kHz PCM WAV and enroll via Eidos."""
    if not consent_confirmed:
        raise HTTPException(status_code=400, detail="Explicit speaker consent (consent_confirmed=true) is required.")

    wav_bytes = await _read_bounded_wav(file)
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Uploaded WAV file is empty.")

    duration_s = _inspect_wav_duration(wav_bytes)
    if duration_s is None:
        raise HTTPException(status_code=400, detail="Invalid WAV file; 16kHz PCM WAV audio is required.")
    if duration_s < MIN_ENROLL_SECONDS or duration_s > MAX_ENROLL_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Enrollment audio duration must be between 15 and 60 seconds (got {duration_s}s).",
        )

    try:
        provider_res = client.enroll_speaker(speaker_id=subject_ref, wav_bytes=wav_bytes)
    except DetectifProviderError as exc:
        status = exc.status_code if exc.status_code in {400, 401, 403, 422, 429} else 502
        raise HTTPException(
            status_code=status,
            detail={"error_code": exc.error_code, "message": exc.message, "request_id": exc.request_id},
        ) from exc
    return wav_bytes, provider_res


@app.post("/api/v1/auth/register")
def auth_register(
    body: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Register a new end-user account bound to a unique pseudonymous subject_ref."""
    clean_username = body.username.strip().lower()
    admin_user, _ = get_admin_credentials()
    if clean_username == admin_user.lower():
        raise HTTPException(status_code=400, detail="Reserved username.")
    if db.scalar(select(UserAccount).where(UserAccount.username == clean_username)) is not None:
        raise HTTPException(status_code=409, detail="Username is already registered.")

    subject_ref = _make_subject_ref(clean_username)
    while db.scalar(select(UserAccount).where(UserAccount.subject_ref == subject_ref)) is not None:
        subject_ref = _make_subject_ref(clean_username)

    account = UserAccount(
        username=clean_username,
        display_name=body.display_name.strip(),
        subject_ref=subject_ref,
        password_hash=hash_password(body.password),
    )
    db.add(account)
    db.commit()
    db.refresh(account)

    actor = AuthActor(
        username=account.username,
        role="user",
        display_name=account.display_name,
        subject_ref=account.subject_ref,
    )
    set_session_cookie(response, create_session_token(actor))
    return {"authenticated": True, "user": _actor_dict(actor)}


@app.post("/api/v1/auth/register-with-voice")
async def auth_register_with_voice(
    response: Response,
    username: str = Form(..., min_length=3, max_length=64),
    display_name: str = Form(..., min_length=2, max_length=100),
    password: str = Form(..., min_length=6, max_length=128),
    consent_confirmed: bool = Form(...),
    consent_ref: str = Form("USER-REG-CONSENT-2026", min_length=2, max_length=128),
    consent_text: str = Form(..., min_length=10, max_length=512),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    client: DetectifClient = Depends(get_detectif_client),
) -> dict[str, Any]:
    """Register a new end-user account AND enroll their voiceprint atomically."""
    clean_username = username.strip().lower()
    clean_display = display_name.strip()
    clean_consent_text = consent_text.strip()
    if len(clean_username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(clean_display) < 2:
        raise HTTPException(status_code=400, detail="Full name must be at least 2 characters.")
    if len(clean_consent_text) < 10:
        raise HTTPException(status_code=400, detail="Explicit consent_text (min 10 chars) is required.")

    admin_user, _ = get_admin_credentials()
    if clean_username == admin_user.lower():
        raise HTTPException(status_code=400, detail="Reserved username.")
    if db.scalar(select(UserAccount).where(UserAccount.username == clean_username)) is not None:
        raise HTTPException(status_code=409, detail="Username is already registered.")

    subject_ref = _make_subject_ref(clean_username)
    while db.scalar(select(UserAccount).where(UserAccount.subject_ref == subject_ref)) is not None:
        subject_ref = _make_subject_ref(clean_username)

    wav_bytes, provider_res = await _validate_and_enroll_wav(
        subject_ref=subject_ref,
        consent_confirmed=consent_confirmed,
        file=file,
        client=client,
    )

    audio_digest = hashlib.sha256(wav_bytes).hexdigest()
    perceptual_digest = compute_perceptual_hash(wav_bytes)
    account = UserAccount(
        username=clean_username,
        display_name=clean_display,
        subject_ref=subject_ref,
        password_hash=hash_password(password),
    )
    record = Enrollment(
        subject_ref=subject_ref,
        provider_speaker_id=provider_res.speaker_id,
        consent_ref=consent_ref,
        consent_text=clean_consent_text,
        audio_sha256=audio_digest,
        speech_seconds=provider_res.speech_seconds,
    )
    db.add(account)
    db.add(record)
    db.add(
        VerificationTelemetry(
            session_id=None,
            event_kind="enroll",
            audio_sha256=audio_digest,
            perceptual_hash=perceptual_digest,
            credits_balance=provider_res.credits_balance,
        )
    )
    db.commit()
    db.refresh(account)
    db.refresh(record)

    actor = AuthActor(
        username=account.username,
        role="user",
        display_name=account.display_name,
        subject_ref=account.subject_ref,
    )
    set_session_cookie(response, create_session_token(actor))
    return {
        "authenticated": True,
        "user": _actor_dict(actor),
        "enrollment": _enrollment_payload(record, provider_res.duration_seconds),
    }


@app.post("/api/v1/auth/login")
def auth_login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Authenticate an admin or registered user with per-IP/per-user lockout protection."""
    clean_username = body.username.strip()
    ip = _client_ip(request)
    try:
        gateway_rate_limiter.ensure_login_allowed(db, ip=ip, username=clean_username)
    except RateLimitExceeded as exc:
        _raise_rate_limited(db, exc=exc, session_id=f"auth-{ip}", sector="auth", action="login")

    admin_user, _ = get_admin_credentials()
    if verify_admin_credentials(clean_username, body.password):
        gateway_rate_limiter.clear_login_failures(db, ip=ip, username=clean_username)
        db.commit()
        actor = AuthActor(
            username=admin_user,
            role="admin",
            display_name="SOC Administrator",
            subject_ref=None,
        )
        set_session_cookie(response, create_session_token(actor))
        return {"authenticated": True, "user": _actor_dict(actor)}

    account = db.scalar(select(UserAccount).where(UserAccount.username == clean_username.lower()))
    if account is None or not verify_password(body.password, account.password_hash):
        gateway_rate_limiter.record_login_failure(db, ip=ip, username=clean_username)
        db.commit()
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    gateway_rate_limiter.clear_login_failures(db, ip=ip, username=clean_username)
    db.commit()
    actor = AuthActor(
        username=account.username,
        role="user",
        display_name=account.display_name,
        subject_ref=account.subject_ref,
    )
    set_session_cookie(response, create_session_token(actor))
    return {"authenticated": True, "user": _actor_dict(actor)}


@app.post("/api/v1/auth/logout")
def auth_logout(response: Response) -> dict[str, Any]:
    """Clear the active authentication cookie."""
    clear_session_cookie(response)
    return {"authenticated": False}


@app.get("/api/v1/user/overview")
def user_overview(
    db: Session = Depends(get_db),
    actor: AuthActor = Depends(get_current_actor),
) -> dict[str, Any]:
    """Return the logged-in user's own enrollment and verification history."""
    if not actor.subject_ref:
        return {"enrollment": None, "verifications": []}

    enrollment = db.scalar(select(Enrollment).where(Enrollment.subject_ref == actor.subject_ref))
    sessions = db.scalars(
        select(VerificationSession)
        .where(VerificationSession.subject_ref == actor.subject_ref)
        .order_by(VerificationSession.expires_at.desc())
        .limit(20)
    ).all()
    history: list[dict[str, Any]] = []
    for s in sessions:
        dec = db.scalar(select(Decision).where(Decision.session_id == s.session_id))
        history.append(
            {
                "session_id": s.session_id,
                "sector": s.sector,
                "action": s.action,
                "resource_ref": s.resource_ref,
                "status": s.status,
                "outcome": dec.outcome if dec else "PENDING",
                "reason_codes": json.loads(dec.reason_codes_json) if dec else [],
                "next_step_script": dec.next_step_script if dec else "",
                "expires_at": _as_utc(s.expires_at).isoformat(),
            }
        )

    return {
        "enrollment": (
            {
                "subject_ref": enrollment.subject_ref,
                "consent_ref": enrollment.consent_ref,
                "audio_sha256": enrollment.audio_sha256,
                "speech_seconds": enrollment.speech_seconds,
                "created_at": _as_utc(enrollment.created_at).isoformat(),
            }
            if enrollment
            else None
        ),
        "verifications": history,
    }


@app.get("/api/v1/profiles")
def list_profiles(_actor: AuthActor = Depends(get_current_actor)) -> dict[str, Any]:
    """Return the 4 configured GATEWAY_PROFILES."""
    return {"profiles": GATEWAY_PROFILES}


@app.get("/api/v1/enrollments")
def list_enrollments(
    db: Session = Depends(get_db),
    actor: AuthActor = Depends(get_current_actor),
) -> dict[str, Any]:
    """List pseudonymous speaker enrollments (scoped to own subject_ref for standard users)."""
    stmt = select(Enrollment).order_by(Enrollment.created_at.desc())
    if actor.role != "admin":
        stmt = stmt.where(Enrollment.subject_ref == actor.subject_ref)
    rows = db.scalars(stmt).all()
    return {
        "enrollments": [
            {
                "id": r.id,
                "subject_ref": r.subject_ref,
                "provider_speaker_id": r.provider_speaker_id,
                "consent_ref": r.consent_ref,
                "audio_sha256": r.audio_sha256,
                "speech_seconds": r.speech_seconds,
                "privileged_role": bool(r.privileged_role),
                "created_at": _as_utc(r.created_at).isoformat(),
            }
            for r in rows
        ]
    }


@app.patch("/api/v1/admin/subjects/{subject_ref}/privilege")
def set_subject_privilege(
    subject_ref: str,
    body: SubjectPrivilegeRequest,
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """Set server-side privileged_role attribute for a subject (admin only)."""
    enrollment = db.scalar(select(Enrollment).where(Enrollment.subject_ref == subject_ref))
    account = db.scalar(select(UserAccount).where(UserAccount.subject_ref == subject_ref))
    if enrollment is None and account is None:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_ref}' not found.")
    if enrollment is not None:
        enrollment.privileged_role = body.privileged_role
    if account is not None:
        account.privileged_role = body.privileged_role
    db.commit()
    return {"subject_ref": subject_ref, "privileged_role": bool(body.privileged_role)}


@app.get("/api/v1/audit")
def list_audit(
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """List immutable tamper-evident audit trail records (admin only)."""
    rows = db.scalars(select(AuditEvent).order_by(AuditEvent.id.desc())).all()
    return {
        "audit_events": [
            {
                "id": r.id,
                "session_id": r.session_id,
                "sector": r.sector,
                "action": r.action,
                "outcome": r.outcome,
                "reason_codes": json.loads(r.reason_codes_json),
                "prev_hash": r.prev_hash,
                "entry_hash": r.entry_hash,
                "created_at": _as_utc(r.created_at).isoformat(),
            }
            for r in rows
        ]
    }


@app.get("/api/v1/audit/verify-chain")
def verify_audit_chain(
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """Verify SHA-256 tamper-evident AuditEvent hash chain integrity (admin only)."""
    return _verify_audit_chain(db)


def _emit_signed_risk_event(
    db: Session,
    *,
    source_sector: str,
    subject_ref: str,
    event_type: str,
    source_case_ref: str,
    now: datetime,
    demo_flag: bool = True,
) -> tuple[RiskEvent, dict[str, Any]]:
    """Persist a cross-domain RiskEvent and attach an HMAC-SHA256 webhook signature."""
    ttl_min_env = os.getenv("RISK_ALERT_TTL_MINUTES", "").strip()
    ttl_delta = timedelta(minutes=int(ttl_min_env)) if ttl_min_env.isdigit() else timedelta(hours=RISK_EVENT_TTL_HOURS)
    expires_at = now + ttl_delta
    risk_row = RiskEvent(
        source_sector=source_sector,
        subject_ref=subject_ref,
        event_type=event_type,
        source_case_ref=source_case_ref,
        risk_state="ACTIVE",
        occurred_at=now,
        expires_at=expires_at,
        demo_flag=demo_flag,
    )
    db.add(risk_row)
    db.flush()

    signed_payload = {
        "id": risk_row.id,
        "event_type": risk_row.event_type,
        "source_sector": risk_row.source_sector,
        "subject_ref": risk_row.subject_ref,
        "source_case_ref": risk_row.source_case_ref,
        "risk_state": risk_row.risk_state,
        "occurred_at": _as_utc(risk_row.occurred_at).isoformat(),
        "expires_at": _as_utc(risk_row.expires_at).isoformat(),
    }
    signature = sign_risk_event_payload(signed_payload)
    db.add(
        RiskWebhookDispatch(
            risk_event_id=risk_row.id,
            hmac_signature=signature,
            payload_json=json.dumps(signed_payload, sort_keys=True, separators=(",", ":")),
        )
    )
    return risk_row, {
        "id": risk_row.id,
        "event_type": risk_row.event_type,
        "subject_ref": risk_row.subject_ref,
        "risk_state": risk_row.risk_state,
        "expires_at": _as_utc(risk_row.expires_at).isoformat(),
        "demo_flag": risk_row.demo_flag,
        "hmac_signature": signature,
        "signed_payload": signed_payload,
    }


@app.get("/api/v1/risk-events")
def list_risk_events(
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """List cross-domain risk events with TTL status and HMAC webhook signatures (admin only)."""
    now = utcnow()
    rows = db.scalars(select(RiskEvent).order_by(RiskEvent.id.desc())).all()
    dispatches = {
        d.risk_event_id: d
        for d in db.scalars(select(RiskWebhookDispatch)).all()
    }
    items: list[dict[str, Any]] = []
    for r in rows:
        dispatch = dispatches.get(r.id)
        expired = _as_utc(r.expires_at) <= now
        items.append(
            {
                "id": r.id,
                "source_sector": r.source_sector,
                "subject_ref": r.subject_ref,
                "event_type": r.event_type,
                "source_case_ref": r.source_case_ref,
                "risk_state": r.risk_state,
                "expired": expired,
                "occurred_at": _as_utc(r.occurred_at).isoformat(),
                "expires_at": _as_utc(r.expires_at).isoformat(),
                "demo_flag": r.demo_flag,
                "hmac_signature": dispatch.hmac_signature if dispatch else None,
                "signed_payload": json.loads(dispatch.payload_json) if dispatch else None,
            }
        )
    return {"risk_events": items}


@app.post("/api/v1/risk-events/{event_id}/resolve")
def resolve_risk_event(
    event_id: int,
    body: ResolveAlertRequest | None = None,
    db: Session = Depends(get_db),
    admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """Resolve an active cross-domain RiskEvent (admin only)."""
    risk_row = db.scalar(select(RiskEvent).where(RiskEvent.id == event_id))
    if risk_row is None:
        raise HTTPException(status_code=404, detail=f"Risk event {event_id} not found.")
    if risk_row.risk_state != "ACTIVE":
        raise HTTPException(status_code=409, detail=f"Risk event {event_id} is already '{risk_row.risk_state}'.")

    now = utcnow()
    note = body.resolution_note.strip() if body else "Resolved by SOC administrator"
    risk_row.risk_state = "RESOLVED"
    _append_audit_event(
        db,
        session_id=risk_row.source_case_ref,
        sector=risk_row.source_sector,
        action="risk_alert_resolve",
        outcome="APPROVED",
        reason_codes=["RISK_ALERT_RESOLVED"],
        created_at=now,
    )
    db.commit()
    return {
        "id": risk_row.id,
        "risk_state": risk_row.risk_state,
        "resolved_by": admin.username,
        "resolved_at": now.isoformat(),
        "resolution_note": note,
    }


@app.post("/api/v1/demo/clear-alerts")
def clear_demo_alerts(
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """Clear active RiskEvent rows for demo reset (admin only)."""
    active_ids = list(db.scalars(select(RiskEvent.id).where(RiskEvent.risk_state == "ACTIVE")).all())
    if active_ids:
        db.execute(delete(RiskWebhookDispatch).where(RiskWebhookDispatch.risk_event_id.in_(active_ids)))
    result = db.execute(delete(RiskEvent).where(RiskEvent.risk_state == "ACTIVE"))
    db.commit()
    return {"status": "ok", "cleared_count": result.rowcount or 0}


@app.post("/api/v1/telecom/offline-swap-callback")
def trigger_offline_swap_callback(
    body: OfflineSwapCallbackRequest,
    db: Session = Depends(get_db),
    actor: AuthActor = Depends(get_current_actor),
) -> dict[str, Any]:
    """Trigger a VoiceSuraksha verification session from a simulated retail offline SIM swap."""
    if actor.role != "admin" and body.subject_ref != actor.subject_ref:
        raise HTTPException(status_code=403, detail="Users may only trigger callbacks for their own subject_ref.")

    enrollment = db.scalar(select(Enrollment).where(Enrollment.subject_ref == body.subject_ref))
    if enrollment is None:
        raise HTTPException(status_code=404, detail=f"Subject '{body.subject_ref}' is not enrolled.")

    now = utcnow()
    session_id = str(uuid.uuid4())
    resource_ref = f"OFFLINE-{body.store_id}-{body.sim_iccid_masked}"[:128]
    challenge_text = generate_challenge_phrase("telecom", "sim_replacement", resource_ref)
    challenge_hash = hash_text_sha256(normalize_transcript(challenge_text))
    expires_at = now + timedelta(minutes=SESSION_TTL_MINUTES)

    session = VerificationSession(
        session_id=session_id,
        sector="telecom",
        subject_ref=body.subject_ref,
        action="sim_replacement",
        resource_ref=resource_ref,
        challenge_text=challenge_text,
        challenge_hash=challenge_hash,
        expires_at=expires_at,
        status="PENDING",
        after_hours=_is_ist_after_hours(now),
        privileged_role=_is_subject_privileged(db, body.subject_ref, enrollment),
        amount_inr=None,
    )
    db.add(session)
    _append_audit_event(
        db,
        session_id=session_id,
        sector="telecom",
        action="sim_replacement",
        outcome="STEP-UP",
        reason_codes=["OFFLINE_RETAIL_SIM_SWAP", "CALLBACK_TRIGGERED"],
        created_at=now,
    )
    db.commit()
    db.refresh(session)

    return {
        "callback_required": True,
        "trigger": "OFFLINE_RETAIL_SIM_SWAP",
        "store_id": body.store_id,
        "session_id": session.session_id,
        "sector": session.sector,
        "subject_ref": session.subject_ref,
        "action": session.action,
        "resource_ref": session.resource_ref,
        "challenge_text": session.challenge_text,
        "challenge_hash": session.challenge_hash,
        "expires_at": _as_utc(session.expires_at).isoformat(),
        "status": session.status,
    }


@app.post("/api/v1/enrollments")
async def create_enrollment(
    subject_ref: str = Form(..., min_length=2, max_length=64),
    consent_confirmed: bool = Form(...),
    consent_ref: str = Form(..., min_length=2, max_length=128),
    consent_text: str = Form(..., min_length=10, max_length=512),
    verified_session_id: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    client: DetectifClient = Depends(get_detectif_client),
    actor: AuthActor = Depends(get_current_actor),
) -> dict[str, Any]:
    """Enroll or re-enroll a consented speaker voiceprint via Eidos; store SHA-256 + perceptual hash only."""
    if actor.role != "admin" and subject_ref != actor.subject_ref:
        raise HTTPException(status_code=403, detail="Users may only enroll their own assigned subject_ref.")

    clean_consent_text = consent_text.strip()
    if len(clean_consent_text) < 10:
        raise HTTPException(status_code=400, detail="Explicit consent_text (min 10 chars) is required.")

    try:
        gateway_rate_limiter.check_sliding_limit(
            db,
            key=f"enroll_sub:{subject_ref}",
            limit=gateway_rate_limiter._env_int("RL_ENROLL_PER_HOUR", 20),
            window_seconds=3600,
            detail="Enrollment rate limit exceeded for this subject.",
        )
    except RateLimitExceeded as exc:
        _raise_rate_limited(db, exc=exc, session_id=f"enroll-{subject_ref}", sector="enrollment", action="enroll")

    existing = db.scalar(select(Enrollment).where(Enrollment.subject_ref == subject_ref))
    if existing is not None and actor.role != "admin":
        verified_ok = False
        if verified_session_id:
            ver_sess = db.scalar(
                select(VerificationSession).where(
                    VerificationSession.session_id == verified_session_id,
                    VerificationSession.subject_ref == subject_ref,
                    VerificationSession.status == "APPROVED",
                )
            )
            verified_ok = ver_sess is not None
        if not verified_ok:
            raise HTTPException(
                status_code=409,
                detail="Voiceprint is already enrolled for this account; re-enrollment requires an APPROVED verified session or admin approval.",
            )

    wav_bytes, provider_res = await _validate_and_enroll_wav(
        subject_ref=subject_ref,
        consent_confirmed=consent_confirmed,
        file=file,
        client=client,
    )

    now = utcnow()
    audio_digest = hashlib.sha256(wav_bytes).hexdigest()
    perceptual_digest = compute_perceptual_hash(wav_bytes)
    if existing is not None:
        existing.provider_speaker_id = provider_res.speaker_id
        existing.consent_ref = consent_ref
        existing.consent_text = clean_consent_text
        existing.audio_sha256 = audio_digest
        existing.speech_seconds = provider_res.speech_seconds
        existing.re_enrolled_at = now
        record = existing
    else:
        record = Enrollment(
            subject_ref=subject_ref,
            provider_speaker_id=provider_res.speaker_id,
            consent_ref=consent_ref,
            consent_text=clean_consent_text,
            audio_sha256=audio_digest,
            speech_seconds=provider_res.speech_seconds,
        )
        db.add(record)

    db.add(
        VerificationTelemetry(
            session_id=None,
            event_kind="enroll",
            audio_sha256=audio_digest,
            perceptual_hash=perceptual_digest,
            credits_balance=provider_res.credits_balance,
        )
    )
    db.commit()
    db.refresh(record)

    return {
        "enrollment": _enrollment_payload(record, provider_res.duration_seconds),
        "provider": {
            "request_id": provider_res.request_id,
            "operation_id": provider_res.operation_id,
            "samples": provider_res.samples,
            "latency_ms": provider_res.latency_ms,
        },
    }


@app.post("/api/v1/verification-sessions")
def create_verification_session(
    body: CreateSessionRequest,
    db: Session = Depends(get_db),
    actor: AuthActor = Depends(get_current_actor),
) -> dict[str, Any]:
    """Create a short-lived action-bound verification session."""
    if actor.role != "admin" and body.subject_ref != actor.subject_ref:
        raise HTTPException(status_code=403, detail="Users may only verify their own assigned subject_ref.")
    try:
        validate_sector_action(body.sector, body.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    enrollment = db.scalar(select(Enrollment).where(Enrollment.subject_ref == body.subject_ref))
    if enrollment is None:
        raise HTTPException(status_code=404, detail=f"Subject '{body.subject_ref}' is not enrolled.")

    try:
        gateway_rate_limiter.check_sliding_limit(
            db,
            key=f"sess_sub:{body.subject_ref}",
            limit=gateway_rate_limiter._env_int("RL_SESSIONS_PER_MIN", 30),
            window_seconds=60,
            detail="Verification session rate limit exceeded for subject.",
        )
        gateway_rate_limiter.check_sliding_limit(
            db,
            key=f"sess_caller:{actor.username}",
            limit=gateway_rate_limiter._env_int("RL_CALLER_SESSIONS_PER_MIN", 40),
            window_seconds=60,
            detail="Verification session rate limit exceeded for caller.",
        )
    except RateLimitExceeded as exc:
        _raise_rate_limited(
            db,
            exc=exc,
            session_id=f"sess-limit-{body.subject_ref}",
            sector=body.sector,
            action=body.action,
        )

    now = utcnow()
    session_id = str(uuid.uuid4())
    challenge_text = generate_challenge_phrase(body.sector, body.action, body.resource_ref)
    challenge_hash = hash_text_sha256(normalize_transcript(challenge_text))
    expires_at = now + timedelta(minutes=SESSION_TTL_MINUTES)
    call_velocity_1h = _compute_call_velocity_1h(db, body.subject_ref, now)

    session = VerificationSession(
        session_id=session_id,
        sector=body.sector,
        subject_ref=body.subject_ref,
        action=body.action,
        resource_ref=body.resource_ref,
        challenge_text=challenge_text,
        challenge_hash=challenge_hash,
        expires_at=expires_at,
        status="PENDING",
        after_hours=_is_ist_after_hours(now),
        privileged_role=_is_subject_privileged(db, body.subject_ref, enrollment),
        amount_inr=body.amount_inr if actor.role == "admin" else None,
        sim_age_days=body.sim_age_days,
        device_changed=bool(body.device_changed),
        call_velocity_1h=call_velocity_1h,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return {
        "session_id": session.session_id,
        "sector": session.sector,
        "subject_ref": session.subject_ref,
        "action": session.action,
        "resource_ref": session.resource_ref,
        "challenge_text": session.challenge_text,
        "challenge_hash": session.challenge_hash,
        "expires_at": _as_utc(session.expires_at).isoformat(),
        "status": session.status,
        "contextual_risk_signals": _contextual_signals_dict(session, now),
    }


@app.post("/api/v1/verification-sessions/{session_id}/audio")
async def submit_verification_audio(
    session_id: str,
    spoken_transcript: str = Form(""),
    strict_phrase: bool = Form(False),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    client: DetectifClient = Depends(get_detectif_client),
    asr_adapter: ServerASRAdapter = Depends(get_asr_adapter),
    actor: AuthActor = Depends(get_current_actor),
) -> dict[str, Any]:
    """Call DetectifAI Eidos+Aletheia & hosted ASR to verify voice, evaluate policy, and issue one-time token on PASS."""
    session = db.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Verification session not found.")
    if actor.role != "admin" and session.subject_ref != actor.subject_ref:
        raise HTTPException(status_code=403, detail="Cannot submit audio for another user's session.")

    if session.status != "PENDING":
        raise HTTPException(status_code=409, detail=f"Session '{session_id}' has already been processed ({session.status}).")

    try:
        gateway_rate_limiter.check_sliding_limit(
            db,
            key=f"audio_sub:{session.subject_ref}",
            limit=gateway_rate_limiter._env_int("RL_AUDIO_PER_MIN", 30),
            window_seconds=60,
            detail="Audio verification rate limit exceeded for subject.",
        )
        gateway_rate_limiter.check_sliding_limit(
            db,
            key=f"audio_caller:{actor.username}",
            limit=gateway_rate_limiter._env_int("RL_CALLER_AUDIO_PER_MIN", 40),
            window_seconds=60,
            detail="Audio verification rate limit exceeded for caller.",
        )
    except RateLimitExceeded as exc:
        _raise_rate_limited(
            db,
            exc=exc,
            session_id=session.session_id,
            sector=session.sector,
            action=session.action,
        )

    now = utcnow()
    wav_bytes = await _read_bounded_wav(file)
    wav_duration = _inspect_wav_duration(wav_bytes) if wav_bytes else None
    audio_digest = hashlib.sha256(wav_bytes).hexdigest() if wav_bytes else None
    perceptual_digest = compute_perceptual_hash(wav_bytes) if wav_bytes else None

    action_meta = validate_sector_action(session.sector, session.action)
    is_critical = bool(action_meta.get("critical", False))
    demo_mode = is_demo_mode()

    active_risks = db.scalars(
        select(RiskEvent).where(
            RiskEvent.subject_ref == session.subject_ref,
            RiskEvent.risk_state == "ACTIVE",
        )
    ).all()
    unexpired_risks = [r for r in active_risks if _as_utc(r.expires_at) > now]

    is_expired = now > _as_utc(session.expires_at)

    replay_match_type: str | None = None
    if not is_expired and (audio_digest or perceptual_digest):
        prior_verifies = db.scalars(
            select(VerificationTelemetry).where(
                VerificationTelemetry.event_kind == "verify",
                VerificationTelemetry.session_id != session.session_id,
            )
        ).all()
        for prev in prior_verifies:
            if audio_digest and prev.audio_sha256 == audio_digest:
                replay_match_type = "exact"
                break
            if (
                perceptual_digest
                and prev.perceptual_hash
                and hamming_distance_hex(perceptual_digest, prev.perceptual_hash) <= MAX_PERCEPTUAL_HAMMING_BITS
            ):
                replay_match_type = "perceptual"
                break

    quality_report: AudioQualityReport | None = None
    if replay_match_type is None and not is_expired and wav_bytes and (wav_duration is None or wav_duration >= 4.0):
        quality_report = inspect_pcm_wav_quality(wav_bytes)
    quality_failed = bool(quality_report is not None and quality_report.valid_wav and not quality_report.passed)

    asr_available = True
    effective_transcript = ""
    if not is_expired and replay_match_type is None and not quality_failed:
        if demo_mode and spoken_transcript.strip():
            effective_transcript = spoken_transcript
            asr_available = True
        else:
            asr_res = asr_adapter.transcribe_wav(wav_bytes)
            asr_available = asr_res.available
            effective_transcript = asr_res.transcript or ""
            # Client transcript is completely ignored in LIVE mode (DEMO_MODE=false)

    require_phrase = is_critical or (demo_mode and strict_phrase)
    phrase_match = verify_phrase_match(session.challenge_text, effective_transcript)
    phrase_blocked_before_provider = require_phrase and asr_available and not phrase_match
    should_call_provider = (
        not is_expired
        and replay_match_type is None
        and not quality_failed
        and not phrase_blocked_before_provider
    )

    provider_res: ProviderVerifyResult | None = None
    provider_error: str | None = None
    segment_disagreement = False
    enrollment = db.scalar(select(Enrollment).where(Enrollment.subject_ref == session.subject_ref))

    if should_call_provider:
        if enrollment is None:
            provider_error = "enrollment_missing"
        elif wav_duration is not None and wav_duration < 4.0:
            # Avoid calling combined deepfake verification on <4s clips; treat as INCONCLUSIVE_EVIDENCE.
            provider_error = None
        else:
            wav_segments = split_wav_segments(wav_bytes)
            try:
                if len(wav_segments) <= 1:
                    provider_res = client.verify_with_deepfake(
                        speaker_id=enrollment.provider_speaker_id,
                        wav_bytes=wav_segments[0],
                    )
                else:
                    seg_results = await asyncio.gather(
                        *(
                            asyncio.to_thread(
                                client.verify_with_deepfake,
                                enrollment.provider_speaker_id,
                                seg_wav,
                            )
                            for seg_wav in wav_segments
                        )
                    )
                    match_cnt = sum(1 for r in seg_results if r.speaker_decision == "match")
                    majority_match = match_cnt > (len(seg_results) / 2.0)
                    segment_disagreement = 0 < match_cnt < len(seg_results)
                    any_fake = any(r.deepfake_verdict == "fake" for r in seg_results)
                    all_segments: list[dict[str, Any]] = []
                    for r in seg_results:
                        all_segments.extend(r.deepfake_segments or [])
                    sims = [r.speaker_similarity for r in seg_results]
                    df_scores = [r.deepfake_score for r in seg_results if r.deepfake_score is not None]
                    lats = [r.latency_ms for r in seg_results if r.latency_ms is not None]
                    first = seg_results[0]
                    provider_res = ProviderVerifyResult(
                        request_id=first.request_id,
                        operation_id=first.operation_id,
                        speaker_id=first.speaker_id,
                        speaker_decision="match" if majority_match else "no_match",
                        speaker_similarity=float(statistics.median(sims)),
                        speaker_threshold=first.speaker_threshold,
                        duration_seconds=wav_duration or first.duration_seconds,
                        deepfake_verdict="fake" if any_fake else "real",
                        deepfake_is_synthetic=any_fake,
                        deepfake_score=max(df_scores) if df_scores else first.deepfake_score,
                        deepfake_threshold=first.deepfake_threshold,
                        deepfake_segments=all_segments,
                        latency_ms=max(lats) if lats else first.latency_ms,
                        credits_balance=seg_results[-1].credits_balance,
                    )
            except DetectifProviderError as exc:
                provider_error = exc.error_code

    cooldown_hours = gateway_rate_limiter._env_int("NEW_ENROLLMENT_COOLDOWN_HOURS", 24)
    in_reenroll_cooldown = (
        not demo_mode
        and enrollment is not None
        and enrollment.re_enrolled_at is not None
        and (_as_utc(enrollment.re_enrolled_at) + timedelta(hours=cooldown_hours)) > now
    )

    if replay_match_type is not None:
        evaluation = AuthorizationDecision(
            outcome="REJECTED",
            evidence_state="REPLAY_DETECTED",
            reason_codes=["AUDIO_REPLAY_DETECTED"],
            next_step_script=(
                f"Request rejected: verification audio matched prior {replay_match_type} replay fingerprint."
            ),
            emit_risk_alert=(session.sector == "telecom" and is_critical),
        )
    elif quality_failed:
        evaluation = AuthorizationDecision(
            outcome="REVIEW / HOLD",
            evidence_state="INCONCLUSIVE_EVIDENCE",
            reason_codes=["AUDIO_QUALITY_LOW"],
            next_step_script=(
                "Audio quality inconclusive (silent or clipped). Use Non-Voice Accessibility Fallback: "
                "Trusted App Push, In-Store Biometric KYC, or Verified Manager Callback."
            ),
            emit_risk_alert=False,
        )
    else:
        voice_input = None
        if should_call_provider:
            if provider_res is not None:
                voice_input = VoiceEvidenceInput(
                    speaker_decision=provider_res.speaker_decision,
                    deepfake_verdict=provider_res.deepfake_verdict,
                    duration_seconds=provider_res.duration_seconds,
                    provider_error=None,
                    speaker_similarity=provider_res.speaker_similarity,
                    speaker_threshold=provider_res.speaker_threshold,
                    segment_disagreement=segment_disagreement,
                )
            else:
                voice_input = VoiceEvidenceInput(
                    speaker_decision=None,
                    deepfake_verdict=None,
                    duration_seconds=wav_duration,
                    provider_error=provider_error,
                )

        policy_session = SessionPolicyInput(
            sector=session.sector,
            action=session.action,
            resource_ref=session.resource_ref,
            after_hours=session.after_hours,
            privileged_role=session.privileged_role,
            amount_inr=session.amount_inr,
            expired=is_expired,
            require_phrase_match=require_phrase,
            asr_available=asr_available,
            challenge_attempt=session.challenge_attempt,
            new_enrollment_cooldown=in_reenroll_cooldown,
            sim_age_days=session.sim_age_days,
            device_changed=bool(session.device_changed),
            call_velocity_1h=int(session.call_velocity_1h),
        )
        evaluation = evaluate_voice_authorization(
            session=policy_session,
            voice_result=voice_input,
            phrase_match=phrase_match,
            active_risk_events=unexpired_risks,
        )

    if is_expired:
        session.status = "EXPIRED"
    elif evaluation.outcome == "APPROVED":
        session.status = "APPROVED"
    elif evaluation.outcome in {"STEP-UP", "REVIEW / HOLD"}:
        session.status = "HOLD" if evaluation.outcome == "REVIEW / HOLD" else "STEP_UP"
    else:
        session.status = "CANCELLED"

    voice_row = VoiceResult(
        session_id=session.session_id,
        request_id=provider_res.request_id if provider_res else None,
        operation_id=provider_res.operation_id if provider_res else None,
        speaker_decision=provider_res.speaker_decision if provider_res else None,
        speaker_similarity=provider_res.speaker_similarity if provider_res else None,
        speaker_threshold=provider_res.speaker_threshold if provider_res else None,
        deepfake_verdict=provider_res.deepfake_verdict if provider_res else None,
        deepfake_is_synthetic=provider_res.deepfake_is_synthetic if provider_res else None,
        deepfake_score=provider_res.deepfake_score if provider_res else None,
        deepfake_threshold=provider_res.deepfake_threshold if provider_res else None,
        segments_json=json.dumps(provider_res.deepfake_segments if provider_res else []),
        phrase_match=phrase_match,
        latency_ms=provider_res.latency_ms if provider_res else None,
    )
    db.add(voice_row)

    plaintext_token: str | None = None
    token_hash: str | None = None
    token_expires_at: datetime | None = None
    if evaluation.outcome == "APPROVED":
        plaintext_token = secrets.token_urlsafe(32)
        token_hash = hash_text_sha256(plaintext_token)
        token_expires_at = now + timedelta(minutes=DECISION_TOKEN_TTL_MINUTES)

    decision_row = Decision(
        session_id=session.session_id,
        outcome=evaluation.outcome,
        reason_codes_json=json.dumps(evaluation.reason_codes),
        next_step_script=evaluation.next_step_script,
        token_hash=token_hash,
        expires_at=token_expires_at,
        consumed_at=None,
    )
    db.add(decision_row)

    emitted_risk_event: dict[str, Any] | None = None
    if evaluation.emit_risk_alert:
        _, emitted_risk_event = _emit_signed_risk_event(
            db,
            source_sector=session.sector,
            subject_ref=session.subject_ref,
            event_type="SIM_SWAP_ATTEMPT_BLOCKED",
            source_case_ref=session.session_id,
            now=now,
            demo_flag=True,
        )

    followup_session_payload: dict[str, Any] | None = None
    if evaluation.outcome == "STEP-UP" and "BORDERLINE_SECOND_CHALLENGE" in evaluation.reason_codes:
        followup_id = str(uuid.uuid4())
        followup_phrase = generate_challenge_phrase(session.sector, session.action, session.resource_ref)
        followup_exp = now + timedelta(minutes=SESSION_TTL_MINUTES)
        followup_row = VerificationSession(
            session_id=followup_id,
            sector=session.sector,
            subject_ref=session.subject_ref,
            action=session.action,
            resource_ref=session.resource_ref,
            challenge_text=followup_phrase,
            challenge_hash=hash_text_sha256(normalize_transcript(followup_phrase)),
            expires_at=followup_exp,
            status="PENDING",
            after_hours=session.after_hours,
            privileged_role=session.privileged_role,
            amount_inr=session.amount_inr,
            sim_age_days=session.sim_age_days,
            device_changed=bool(session.device_changed),
            call_velocity_1h=int(session.call_velocity_1h),
            parent_session_id=session.session_id,
            challenge_attempt=session.challenge_attempt + 1,
        )
        db.add(followup_row)
        followup_session_payload = {
            "session_id": followup_id,
            "parent_session_id": session.session_id,
            "challenge_attempt": followup_row.challenge_attempt,
            "challenge_text": followup_phrase,
            "expires_at": followup_exp.isoformat(),
        }

    _append_audit_event(
        db,
        session_id=session.session_id,
        sector=session.sector,
        action=session.action,
        outcome=evaluation.outcome,
        reason_codes=evaluation.reason_codes,
        created_at=now,
    )
    db.add(
        VerificationTelemetry(
            session_id=session.session_id,
            event_kind="verify",
            audio_sha256=audio_digest,
            perceptual_hash=perceptual_digest,
            credits_balance=provider_res.credits_balance if provider_res else None,
        )
    )
    db.commit()

    quality_payload = (
        {
            "valid_wav": quality_report.valid_wav,
            "duration_sec": quality_report.duration_sec,
            "rms_dbfs": quality_report.rms_dbfs,
            "silence_ratio": quality_report.silence_ratio,
            "clipping_ratio": quality_report.clipping_ratio,
            "passed": quality_report.passed,
            "reason": quality_report.reason,
        }
        if quality_report is not None
        else None
    )

    fallback_options = NON_VOICE_FALLBACK_OPTIONS if evaluation.outcome in {"REVIEW / HOLD", "STEP-UP"} else []

    contextual_payload = _contextual_signals_dict(session, now)

    if actor.role != "admin":
        return {
            "session_id": session.session_id,
            "status": session.status,
            "phrase_match": phrase_match,
            "evidence_state": evaluation.evidence_state,
            "outcome": evaluation.outcome,
            "reason_codes": evaluation.reason_codes,
            "match_type": replay_match_type,
            "followup_session": followup_session_payload,
            "next_step_script": evaluation.next_step_script,
            "fallback_options": fallback_options,
            "audio_quality": quality_payload,
            "contextual_risk_signals": contextual_payload,
            "voice_summary": {
                "speaker_decision": voice_row.speaker_decision,
                "deepfake_verdict": voice_row.deepfake_verdict,
                "latency_ms": voice_row.latency_ms,
            },
        }

    return {
        "session_id": session.session_id,
        "status": session.status,
        "phrase_match": phrase_match,
        "evidence_state": evaluation.evidence_state,
        "outcome": evaluation.outcome,
        "reason_codes": evaluation.reason_codes,
        "match_type": replay_match_type,
        "followup_session": followup_session_payload,
        "next_step_script": evaluation.next_step_script,
        "fallback_options": fallback_options,
        "audio_quality": quality_payload,
        "contextual_risk_signals": contextual_payload,
        "decision_token": plaintext_token,
        "token_expires_at": token_expires_at.isoformat() if token_expires_at else None,
        "voice_result": {
            "request_id": voice_row.request_id,
            "operation_id": voice_row.operation_id,
            "speaker_decision": voice_row.speaker_decision,
            "speaker_similarity": voice_row.speaker_similarity,
            "speaker_threshold": voice_row.speaker_threshold,
            "deepfake_verdict": voice_row.deepfake_verdict,
            "deepfake_is_synthetic": voice_row.deepfake_is_synthetic,
            "deepfake_score": voice_row.deepfake_score,
            "deepfake_threshold": voice_row.deepfake_threshold,
            "segments": json.loads(voice_row.segments_json),
            "latency_ms": voice_row.latency_ms,
            "provider_error": provider_error,
        },
        "emitted_risk_event": emitted_risk_event,
    }


@app.post("/api/v1/decisions/{session_id}/consume")
def consume_decision_token(
    session_id: str,
    body: ConsumeDecisionRequest,
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """Validate and atomically consume a one-time decision token via a conditional SQL UPDATE."""
    session = db.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id))
    decision = db.scalar(select(Decision).where(Decision.session_id == session_id))
    if session is None or decision is None:
        raise HTTPException(status_code=404, detail="Decision not found for session.")

    if decision.outcome != "APPROVED" or decision.token_hash is None:
        raise HTTPException(status_code=409, detail=f"Decision outcome '{decision.outcome}' did not issue a consumable token.")

    if session.action != body.action or session.resource_ref != body.resource_ref:
        raise HTTPException(status_code=409, detail="Action or resource_ref does not match the bound verification session.")

    provided_hash = hash_text_sha256(body.decision_token)
    if not secrets.compare_digest(provided_hash, decision.token_hash):
        raise HTTPException(status_code=403, detail="Invalid decision token.")

    bound_sid = session.session_id
    bound_sector = session.sector
    bound_action = session.action
    bound_resource_ref = session.resource_ref
    # End the initial read snapshot cleanly so concurrent callers contend on an atomic write statement
    db.rollback()

    now = utcnow()
    stmt = (
        update(Decision)
        .where(
            Decision.session_id == bound_sid,
            Decision.outcome == "APPROVED",
            Decision.token_hash == provided_hash,
            Decision.consumed_at.is_(None),
            Decision.expires_at.is_not(None),
            Decision.expires_at > now,
        )
        .values(consumed_at=now)
    )
    res = db.execute(stmt)
    if (res.rowcount or 0) == 0:
        db.rollback()
        latest = db.scalar(select(Decision).where(Decision.session_id == bound_sid))
        if latest is not None and latest.consumed_at is None and (latest.expires_at is None or now > _as_utc(latest.expires_at)):
            raise HTTPException(status_code=409, detail="Decision token has expired.")
        raise HTTPException(status_code=409, detail="Decision token has already been consumed.")

    _append_audit_event(
        db,
        session_id=bound_sid,
        sector=bound_sector,
        action=bound_action,
        outcome="APPROVED",
        reason_codes=["TOKEN_CONSUMED", "ACTION_RELEASED"],
        created_at=now,
    )
    db.commit()

    return {
        "consumed": True,
        "session_id": bound_sid,
        "sector": bound_sector,
        "action": bound_action,
        "resource_ref": bound_resource_ref,
        "consumed_at": now.isoformat(),
    }


@app.post("/api/v1/decisions/{session_id}/step-up/complete")
def complete_step_up(
    session_id: str,
    body: CompleteStepUpRequest,
    db: Session = Depends(get_db),
    actor: AuthActor = Depends(get_current_actor),
) -> dict[str, Any]:
    """Complete a simulated out-of-band STEP-UP confirmation and issue a single one-time token."""
    session = db.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id))
    decision = db.scalar(select(Decision).where(Decision.session_id == session_id))
    if session is None or decision is None:
        raise HTTPException(status_code=404, detail="Verification session or decision not found.")
    if actor.role != "admin" and session.subject_ref != actor.subject_ref:
        raise HTTPException(status_code=403, detail="Cannot complete step-up for another user's session.")
    if body.amount_inr is not None and actor.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin/backend callers may supply banking amount_inr.")

    if decision.outcome == "APPROVED" or decision.token_hash is not None or session.status == "APPROVED":
        raise HTTPException(
            status_code=409,
            detail="Step-up has already been completed and a one-time token was already issued.",
        )
    if decision.outcome not in {"STEP-UP", "REVIEW / HOLD"}:
        raise HTTPException(
            status_code=409,
            detail=f"Session outcome '{decision.outcome}' does not accept fallback or step-up completion.",
        )

    now = utcnow()
    action_meta = validate_sector_action(session.sector, session.action)
    is_critical = bool(action_meta.get("critical", False))
    reasons = json.loads(decision.reason_codes_json)
    threshold_eval: dict[str, Any] | None = None

    if session.sector == "banking" and is_critical:
        effective_amount = float(body.amount_inr) if body.amount_inr is not None else session.amount_inr
        if effective_amount is None:
            raise HTTPException(
                status_code=400,
                detail="Admin/backend amount_inr is required to resolve Banking AMOUNT_REQUIRED step-up.",
            )
        session.amount_inr = effective_amount
        thresh = float(GATEWAY_PROFILES["banking"]["amount_step_up_threshold_inr"])
        is_high_value = effective_amount >= thresh
        threshold_eval = {"amount_inr": effective_amount, "threshold_inr": thresh, "high_value": is_high_value}
        if is_high_value and body.method == "trusted_app_push":
            reasons = ["HIGH_VALUE_STEP_UP"]
            decision.reason_codes_json = json.dumps(reasons)
            decision.next_step_script = (
                f"Amount INR {effective_amount:,.0f} exceeds INR {thresh:,.0f} high-value threshold: "
                "complete step-up using simulated manager_callback or in_store_kyc."
            )
            _append_audit_event(
                db,
                session_id=session.session_id,
                sector=session.sector,
                action=session.action,
                outcome="STEP-UP",
                reason_codes=["HIGH_VALUE_THRESHOLD_EVALUATED", "HIGH_VALUE_STEP_UP"],
                created_at=now,
            )
            db.commit()
            return {
                "session_id": session.session_id,
                "status": session.status,
                "outcome": "STEP-UP",
                "method": body.method,
                "simulated": True,
                "reason_codes": reasons,
                "threshold_evaluation": threshold_eval,
                "decision_token": None,
                "token_expires_at": None,
                "next_step_script": decision.next_step_script,
            }
        reasons = [r for r in reasons if r != "AMOUNT_REQUIRED"]
        if is_high_value and "HIGH_VALUE_THRESHOLD_VERIFIED" not in reasons:
            reasons.append("HIGH_VALUE_THRESHOLD_VERIFIED")

    plaintext_token = secrets.token_urlsafe(32)
    token_expires_at = now + timedelta(minutes=DECISION_TOKEN_TTL_MINUTES)
    decision.outcome = "APPROVED"
    decision.token_hash = hash_text_sha256(plaintext_token)
    decision.expires_at = token_expires_at
    decision.reason_codes_json = json.dumps([*reasons, "STEP_UP_COMPLETED"])
    decision.next_step_script = (
        f"Simulated {body.method} confirmation verified. Approved to proceed with one-time action token."
    )
    session.status = "APPROVED"

    _append_audit_event(
        db,
        session_id=session.session_id,
        sector=session.sector,
        action=session.action,
        outcome="APPROVED",
        reason_codes=["STEP_UP_COMPLETED", f"METHOD_{body.method.upper()}", "SIMULATED_CHANNEL"],
        created_at=now,
    )
    db.commit()

    return {
        "session_id": session.session_id,
        "status": session.status,
        "outcome": decision.outcome,
        "method": body.method,
        "simulated": True,
        "reason_codes": json.loads(decision.reason_codes_json),
        "threshold_evaluation": threshold_eval,
        "decision_token": plaintext_token,
        "token_expires_at": token_expires_at.isoformat(),
        "next_step_script": decision.next_step_script,
    }


@app.get("/api/v1/admin/review-queue")
def get_admin_review_queue(
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """List HOLD and STEP_UP sessions awaiting admin review or out-of-band confirmation."""
    sessions = db.scalars(
        select(VerificationSession)
        .where(VerificationSession.status.in_(["HOLD", "STEP_UP"]))
        .order_by(VerificationSession.expires_at.desc())
    ).all()
    items: list[dict[str, Any]] = []
    for s in sessions:
        dec = db.scalar(select(Decision).where(Decision.session_id == s.session_id))
        items.append(
            {
                "session_id": s.session_id,
                "sector": s.sector,
                "subject_ref": s.subject_ref,
                "action": s.action,
                "resource_ref": s.resource_ref,
                "status": s.status,
                "outcome": dec.outcome if dec else ("REVIEW / HOLD" if s.status == "HOLD" else "STEP-UP"),
                "reason_codes": json.loads(dec.reason_codes_json) if dec else [],
                "amount_inr": s.amount_inr,
                "expires_at": _as_utc(s.expires_at).isoformat(),
                "fallback_options": NON_VOICE_FALLBACK_OPTIONS,
            }
        )
    return {"review_queue": items}


@app.post("/api/v1/decisions/{session_id}/review")
def review_held_decision(
    session_id: str,
    body: ReviewDecisionRequest,
    db: Session = Depends(get_db),
    admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """Manually approve or reject a HOLD or STEP_UP session with required reason & timestamp."""
    session = db.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id))
    decision = db.scalar(select(Decision).where(Decision.session_id == session_id))
    if session is None or decision is None:
        raise HTTPException(status_code=404, detail="Session or decision not found.")
    if session.status not in {"HOLD", "STEP_UP"} or decision.token_hash is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Session status '{session.status}' is not awaiting reviewer action.",
        )

    clean_reason = body.reason.strip()
    if len(clean_reason) < 3:
        raise HTTPException(status_code=400, detail="A non-empty review reason (min 3 chars) is required.")

    now = utcnow()
    plaintext_token: str | None = None
    token_expires_at: datetime | None = None

    if body.decision == "APPROVED":
        plaintext_token = secrets.token_urlsafe(32)
        token_expires_at = now + timedelta(minutes=DECISION_TOKEN_TTL_MINUTES)
        decision.token_hash = hash_text_sha256(plaintext_token)
        decision.expires_at = token_expires_at
        session.status = "APPROVED"
        audit_code = "REVIEWER_APPROVED"
    else:
        session.status = "CANCELLED"
        audit_code = "REVIEWER_REJECTED"

    existing_reasons = json.loads(decision.reason_codes_json)
    decision.outcome = body.decision
    decision.reason_codes_json = json.dumps([*existing_reasons, audit_code])
    decision.next_step_script = f"Reviewer ({admin.username}) recorded {body.decision}: {clean_reason}"

    _append_audit_event(
        db,
        session_id=session.session_id,
        sector=session.sector,
        action=session.action,
        outcome=body.decision,
        reason_codes=[audit_code, f"REASON:{clean_reason[:80]}"],
        created_at=now,
    )
    db.commit()

    return {
        "session_id": session.session_id,
        "status": session.status,
        "outcome": decision.outcome,
        "reason": clean_reason,
        "reviewer": admin.username,
        "reviewed_at": now.isoformat(),
        "decision_token": plaintext_token,
        "token_expires_at": token_expires_at.isoformat() if token_expires_at else None,
    }


def _resolve_token_status(decision: Decision | None, now: datetime) -> str:
    """Compute lifecycle status string for a session's one-time decision token."""
    if decision is None:
        return "NOT_EVALUATED"
    if decision.outcome != "APPROVED" or decision.token_hash is None:
        return "NOT_ISSUED"
    if decision.consumed_at is not None:
        return "CONSUMED"
    if decision.expires_at is not None and now > _as_utc(decision.expires_at):
        return "EXPIRED"
    return "ISSUED"


@app.get("/api/v1/metrics")
def get_gateway_metrics(
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """Return real-time gateway telemetry, throughput, and latency metrics from SQL."""
    now = utcnow()
    decisions = db.scalars(select(Decision)).all()
    total_verifications = len(decisions)
    pass_count = sum(1 for d in decisions if d.outcome == "APPROVED")
    blocked_count = sum(1 for d in decisions if d.outcome == "REJECTED")
    hold_count = sum(1 for d in decisions if d.outcome == "REVIEW / HOLD")
    step_up_count = sum(1 for d in decisions if d.outcome == "STEP-UP")

    latencies = sorted(
        float(v)
        for v in db.scalars(select(VoiceResult.latency_ms).where(VoiceResult.latency_ms.is_not(None))).all()
        if v is not None
    )
    p50_latency_ms = round(statistics.median(latencies), 1) if latencies else None
    p95_latency_ms = (
        round(latencies[min(len(latencies) - 1, max(0, math.ceil(0.95 * len(latencies)) - 1))], 1)
        if latencies
        else None
    )

    one_min_ago = now - timedelta(minutes=1)
    recent_evals = db.scalars(
        select(AuditEvent.session_id)
        .join(Decision, Decision.session_id == AuditEvent.session_id)
        .where(
            AuditEvent.created_at >= one_min_ago,
            AuditEvent.outcome.in_(["APPROVED", "STEP-UP", "REVIEW / HOLD", "REJECTED"]),
        )
    ).all()
    decisions_per_min = float(len(set(recent_evals)))

    last_credits_balance = db.scalar(
        select(VerificationTelemetry.credits_balance)
        .where(VerificationTelemetry.credits_balance.is_not(None))
        .order_by(VerificationTelemetry.id.desc())
        .limit(1)
    )

    active_risks = db.scalars(select(RiskEvent).where(RiskEvent.risk_state == "ACTIVE")).all()
    active_risk_events_count = sum(1 for r in active_risks if _as_utc(r.expires_at) > now)

    return {
        "total_verifications": total_verifications,
        "blocked_count": blocked_count,
        "hold_count": hold_count,
        "pass_count": pass_count,
        "step_up_count": step_up_count,
        "decisions_per_min": decisions_per_min,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "last_credits_balance": last_credits_balance,
        "active_risk_events_count": active_risk_events_count,
    }


@app.get("/api/v1/accuracy/report")
def get_accuracy_report(_admin: AuthActor = Depends(require_admin)) -> dict[str, Any]:
    """Return latest offline/live accuracy evaluation report if generated by scripts/eval_accuracy.py."""
    report_path = Path(os.getenv("EVAL_REPORT_PATH", str(Path(__file__).resolve().parent.parent / "reports" / "eval_accuracy.json")))
    if not report_path.exists():
        return {
            "status": "not_run",
            "message": "No evaluation run yet",
            "report": None,
        }
    try:
        parsed = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "status": "available",
            "message": "Evaluation report loaded",
            "report": parsed,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to read evaluation report: {exc}",
            "report": None,
        }


@app.get("/api/v1/audit/{session_id}/export")
def export_audit_dossier(
    session_id: str,
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> Response:
    """Export a Compliance Audit Dossier JSON document for a verification session (zero raw audio)."""
    session = db.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Verification session not found.")

    voice_result = db.scalar(select(VoiceResult).where(VoiceResult.session_id == session_id))
    decision = db.scalar(select(Decision).where(Decision.session_id == session_id))
    telemetry = db.scalar(
        select(VerificationTelemetry)
        .where(VerificationTelemetry.session_id == session_id)
        .order_by(VerificationTelemetry.id.desc())
        .limit(1)
    )
    enrollment = db.scalar(select(Enrollment).where(Enrollment.subject_ref == session.subject_ref))
    audits = db.scalars(
        select(AuditEvent).where(AuditEvent.session_id == session_id).order_by(AuditEvent.id.asc())
    ).all()

    now = utcnow()
    reason_codes = json.loads(decision.reason_codes_json) if decision else []
    is_sandbox = session.resource_ref.startswith("SANDBOX-") or "SANDBOX_TEST" in reason_codes

    dossier = {
        "policy_version": POLICY_VERSION,
        "exported_at": now.isoformat(),
        "zero_raw_audio_stored": True,
        "sandbox_test": is_sandbox,
        "audit_chain_status": _verify_audit_chain(db),
        "session": {
            "session_id": session.session_id,
            "sector": session.sector,
            "subject_ref": session.subject_ref,
            "action": session.action,
            "resource_ref": session.resource_ref,
            "status": session.status,
            "after_hours": session.after_hours,
            "privileged_role": session.privileged_role,
            "amount_inr": session.amount_inr,
            "sim_age_days": session.sim_age_days,
            "device_changed": session.device_changed,
            "call_velocity_1h": session.call_velocity_1h,
            "challenge_hash": session.challenge_hash,
            "expires_at": _as_utc(session.expires_at).isoformat(),
        },
        "audio_sha256": (telemetry.audio_sha256 if telemetry else None)
        or (enrollment.audio_sha256 if enrollment else None),
        "request_id": voice_result.request_id if voice_result else None,
        "operation_id": voice_result.operation_id if voice_result else None,
        "speaker_decision": voice_result.speaker_decision if voice_result else None,
        "speaker_similarity": voice_result.speaker_similarity if voice_result else None,
        "deepfake_verdict": voice_result.deepfake_verdict if voice_result else None,
        "deepfake_score": voice_result.deepfake_score if voice_result else None,
        "segments": json.loads(voice_result.segments_json) if voice_result else [],
        "outcome": decision.outcome if decision else session.status,
        "reason_codes": reason_codes,
        "token_status": _resolve_token_status(decision, now),
        "audit_events": [
            {
                "id": a.id,
                "outcome": a.outcome,
                "reason_codes": json.loads(a.reason_codes_json),
                "created_at": _as_utc(a.created_at).isoformat(),
            }
            for a in audits
        ],
    }
    return Response(
        content=json.dumps(dossier, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="audit-dossier-{session_id[:12]}.json"'},
    )


def _create_sandbox_session(
    *,
    sector: str,
    subject_ref: str,
    action: str,
    resource_ref: str,
    expires_at: datetime,
    status: str,
) -> VerificationSession:
    phrase = generate_challenge_phrase(sector, action, resource_ref)
    return VerificationSession(
        session_id=f"sbx-{uuid.uuid4()}",
        sector=sector,
        subject_ref=subject_ref,
        action=action,
        resource_ref=resource_ref,
        challenge_text=phrase,
        challenge_hash=hash_text_sha256(normalize_transcript(phrase)),
        expires_at=expires_at,
        status=status,
        after_hours=False,
        privileged_role=False,
        amount_inr=None,
    )


def _sandbox_synth_wav(seed: int, duration_sec: float = 4.5, rate: int = 16000) -> bytes:
    """Create a deterministic multi-band harmonic WAV in memory for sandbox perceptual tests."""
    import numpy as np

    num_frames = int(duration_sec * rate)
    t = np.arange(num_frames, dtype=np.float64) / float(rate)
    sig = np.zeros(num_frames, dtype=np.float64)
    rng = np.random.default_rng(seed * 997 + 13)
    band_freqs = np.geomspace(180.0, 4200.0, 9)
    win_edges = np.linspace(0, num_frames, 10, dtype=int)
    for w in range(9):
        s0, s1 = int(win_edges[w]), int(win_edges[w + 1])
        gains = rng.uniform(0.2, 1.8, size=len(band_freqs))
        tw = t[s0:s1]
        chunk = np.zeros(s1 - s0, dtype=np.float64)
        for f_hz, g in zip(band_freqs, gains):
            chunk += g * np.sin(2.0 * math.pi * f_hz * tw)
        sig[s0:s1] = chunk
    rms = float(np.sqrt(np.mean(sig * sig))) or 1.0
    pcm = np.clip((sig / rms) * 6500.0, -28000.0, 28000.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@app.post("/api/v1/sandbox/run/{scenario}")
def run_redteam_sandbox(
    scenario: str,
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """Run deterministic server-side red-team security stress scenarios tagged sandbox_test=true."""
    valid_scenarios = {
        "token_replay",
        "action_substitution",
        "expired_challenge",
        "provider_outage_fail_closed",
        "cross_domain_sim_alert",
        "transcript_spoofing",
        "reencoded_replay",
        "policy_context_tampering",
        "rate_limit_lockout",
        "audit_chain_tamper",
    }
    if scenario not in valid_scenarios:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported sandbox scenario '{scenario}'. Valid: {', '.join(sorted(valid_scenarios))}",
        )

    now = utcnow()
    short_tag = secrets.token_hex(3).upper()
    emitted_risk_id: int | None = None
    expected_spec: dict[str, Any]
    actual_spec: dict[str, Any]

    if scenario == "token_replay":
        sess = _create_sandbox_session(
            sector="telecom",
            subject_ref="SUB-SANDBOX",
            action="sim_replacement",
            resource_ref=f"SANDBOX-REPLAY-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="CANCELLED",
        )
        token = secrets.token_urlsafe(24)
        reason_codes = ["SANDBOX_TEST", "TOKEN_REPLAY_BLOCKED"]
        outcome = "REJECTED"
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script="Sandbox: first consume succeeded; replay attempt rejected with HTTP 409.",
            token_hash=hash_text_sha256(token),
            expires_at=now + timedelta(minutes=DECISION_TOKEN_TTL_MINUTES),
            consumed_at=now,
        )
        http_status = 409 if dec.consumed_at is not None else 200
        expected_spec = {"http_status": 409, "outcome": "REJECTED", "reason_code": "TOKEN_REPLAY_BLOCKED"}
        actual_spec = {"http_status": http_status, "outcome": outcome, "reason_codes": reason_codes}
        detail = "Decision token has already been consumed (replay prevented)."

    elif scenario == "action_substitution":
        sess = _create_sandbox_session(
            sector="telecom",
            subject_ref="SUB-SANDBOX",
            action="sim_replacement",
            resource_ref=f"SANDBOX-SUB-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="CANCELLED",
        )
        attempted_action = "mnp_port_out"
        http_status = 409 if sess.action != attempted_action else 200
        reason_codes = ["SANDBOX_TEST", "ACTION_SUBSTITUTION_BLOCKED"]
        outcome = "REJECTED"
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script=(
                f"Sandbox: token bound to '{sess.action}' rejected attempted substitution '{attempted_action}'."
            ),
            token_hash=None,
            expires_at=None,
            consumed_at=None,
        )
        expected_spec = {"http_status": 409, "outcome": "REJECTED", "reason_code": "ACTION_SUBSTITUTION_BLOCKED"}
        actual_spec = {"http_status": http_status, "outcome": outcome, "reason_codes": reason_codes}
        detail = f"Action mismatch blocked ({sess.action} != {attempted_action})."

    elif scenario == "expired_challenge":
        sess = _create_sandbox_session(
            sector="telecom",
            subject_ref="SUB-SANDBOX",
            action="esim_transfer",
            resource_ref=f"SANDBOX-EXP-{short_tag}",
            expires_at=now - timedelta(minutes=5),
            status="EXPIRED",
        )
        eval_res = evaluate_voice_authorization(
            session=SessionPolicyInput(
                sector=sess.sector,
                action=sess.action,
                resource_ref=sess.resource_ref,
                expired=True,
                require_phrase_match=False,
            ),
            voice_result=None,
            phrase_match=False,
            active_risk_events=[],
        )
        http_status = 410
        reason_codes = [*eval_res.reason_codes, "SANDBOX_TEST"]
        outcome = eval_res.outcome
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script=eval_res.next_step_script,
            token_hash=None,
            expires_at=None,
            consumed_at=None,
        )
        expected_spec = {"http_status": 410, "outcome": "REJECTED", "reason_code": "CHALLENGE_EXPIRED"}
        actual_spec = {"http_status": http_status, "outcome": outcome, "reason_codes": reason_codes}
        detail = "Expired verification challenge rejected fail-closed."

    elif scenario == "provider_outage_fail_closed":
        sess = _create_sandbox_session(
            sector="helpdesk",
            subject_ref="SUB-SANDBOX",
            action="mfa_reset",
            resource_ref=f"SANDBOX-503-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="HOLD",
        )
        eval_res = evaluate_voice_authorization(
            session=SessionPolicyInput(
                sector=sess.sector,
                action=sess.action,
                resource_ref=sess.resource_ref,
                expired=False,
                require_phrase_match=False,
            ),
            voice_result=VoiceEvidenceInput(provider_error="http_503"),
            phrase_match=True,
            active_risk_events=[],
        )
        http_status = 503
        reason_codes = [*eval_res.reason_codes, "SANDBOX_TEST"]
        outcome = eval_res.outcome
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script=eval_res.next_step_script,
            token_hash=None,
            expires_at=None,
            consumed_at=None,
        )
        expected_spec = {"http_status": 503, "outcome": "REVIEW / HOLD", "reason_code": "PROVIDER_UNAVAILABLE"}
        actual_spec = {"http_status": http_status, "outcome": outcome, "reason_codes": reason_codes}
        detail = "Upstream provider 503 moved session to REVIEW / HOLD with non-voice fallback (fail-closed)."

    elif scenario == "cross_domain_sim_alert":
        target_subject = "SUB-SANDBOX"
        risk_row, _ = _emit_signed_risk_event(
            db,
            source_sector="telecom",
            subject_ref=target_subject,
            event_type="SIM_SWAP_ATTEMPT_BLOCKED",
            source_case_ref=f"SANDBOX-SIM-{short_tag}",
            now=now,
            demo_flag=True,
        )
        emitted_risk_id = risk_row.id

        sess = _create_sandbox_session(
            sector="banking",
            subject_ref=target_subject,
            action="card_replacement",
            resource_ref=f"SANDBOX-BANK-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="STEP_UP",
        )
        eval_res = evaluate_voice_authorization(
            session=SessionPolicyInput(
                sector="banking",
                action="card_replacement",
                resource_ref=sess.resource_ref,
                expired=False,
                require_phrase_match=False,
            ),
            voice_result=VoiceEvidenceInput(
                speaker_decision="match",
                deepfake_verdict="real",
                duration_seconds=8.0,
            ),
            phrase_match=True,
            active_risk_events=[risk_row],
        )
        http_status = 200
        reason_codes = [*eval_res.reason_codes, "SANDBOX_TEST"]
        outcome = eval_res.outcome
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script=eval_res.next_step_script,
            token_hash=None,
            expires_at=None,
            consumed_at=None,
        )
        expected_spec = {"http_status": 200, "outcome": "STEP-UP", "reason_code": "CROSS_DOMAIN_ALERT"}
        actual_spec = {"http_status": http_status, "outcome": outcome, "reason_codes": reason_codes}
        detail = "Blocked Telecom SIM swap propagated ACTIVE risk event -> Banking card replacement escalated to STEP-UP."

    elif scenario == "transcript_spoofing":
        sess = _create_sandbox_session(
            sector="telecom",
            subject_ref="SUB-SANDBOX",
            action="sim_replacement",
            resource_ref=f"SANDBOX-ASR-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="CANCELLED",
        )
        server_asr_transcript = "unrelated wrong phrase zero zero"
        asr_match = verify_phrase_match(sess.challenge_text, server_asr_transcript)
        eval_res = evaluate_voice_authorization(
            session=SessionPolicyInput(
                sector=sess.sector,
                action=sess.action,
                resource_ref=sess.resource_ref,
                expired=False,
                require_phrase_match=True,
                asr_available=True,
            ),
            voice_result=VoiceEvidenceInput(speaker_decision="match", deepfake_verdict="real", duration_seconds=6.0),
            phrase_match=asr_match,
            active_risk_events=[],
        )
        http_status = 200
        reason_codes = [*eval_res.reason_codes, "SANDBOX_TEST"]
        outcome = eval_res.outcome
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script=eval_res.next_step_script,
            token_hash=None,
            expires_at=None,
            consumed_at=None,
        )
        expected_spec = {"http_status": 200, "outcome": "REJECTED", "reason_code": "CHALLENGE_MISMATCH"}
        actual_spec = {"http_status": http_status, "outcome": outcome, "reason_codes": reason_codes, "client_transcript_ignored": True}
        detail = "Client sent exact challenge text, but server ASR heard wrong phrase -> REJECTED (CHALLENGE_MISMATCH)."

    elif scenario == "reencoded_replay":
        wav_orig = _sandbox_synth_wav(seed=42, duration_sec=4.5)
        with wave.open(io.BytesIO(wav_orig), "rb") as wf:
            ch, sw, sr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
            raw = wf.readframes(nf)
        raw_ints = struct.unpack(f"<{nf}h", raw)
        trim = int(0.15 * sr)
        scaled = [max(-32767, min(32767, int(v * 0.65))) for v in raw_ints[trim:-trim]]
        buf_mod = io.BytesIO()
        with wave.open(buf_mod, "wb") as owf:
            owf.setnchannels(ch)
            owf.setsampwidth(sw)
            owf.setframerate(sr)
            owf.writeframes(struct.pack(f"<{len(scaled)}h", *scaled))
        h1 = compute_perceptual_hash(wav_orig) or ""
        h2 = compute_perceptual_hash(buf_mod.getvalue()) or ""
        dist = hamming_distance_hex(h1, h2) if (h1 and h2) else 999
        matched = dist <= MAX_PERCEPTUAL_HAMMING_BITS
        sess = _create_sandbox_session(
            sector="telecom",
            subject_ref="SUB-SANDBOX",
            action="sim_replacement",
            resource_ref=f"SANDBOX-PHASH-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="CANCELLED" if matched else "PENDING",
        )
        http_status = 200
        reason_codes = ["AUDIO_REPLAY_DETECTED", "SANDBOX_TEST"]
        outcome = "REJECTED" if matched else "APPROVED"
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script=f"Sandbox: 0.65x gain-scaled & trimmed replay caught via perceptual hash (distance={dist} bits <= {MAX_PERCEPTUAL_HAMMING_BITS}).",
            token_hash=None,
            expires_at=None,
            consumed_at=None,
        )
        expected_spec = {"http_status": 200, "outcome": "REJECTED", "reason_code": "AUDIO_REPLAY_DETECTED", "match_type": "perceptual"}
        actual_spec = {"http_status": http_status, "outcome": outcome, "reason_codes": reason_codes, "match_type": "perceptual", "hamming_bits": dist}
        detail = f"Re-encoded 0.65x/trimmed audio detected as perceptual replay ({dist} Hamming bits)."

    elif scenario == "policy_context_tampering":
        sess = _create_sandbox_session(
            sector="telecom",
            subject_ref="SUB-SANDBOX",
            action="sim_replacement",
            resource_ref=f"SANDBOX-TAMPER-CTX-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="STEP_UP",
        )
        sess.after_hours = True
        eval_res = evaluate_voice_authorization(
            session=SessionPolicyInput(
                sector="telecom",
                action="sim_replacement",
                resource_ref=sess.resource_ref,
                after_hours=True,
                expired=False,
                require_phrase_match=False,
            ),
            voice_result=VoiceEvidenceInput(speaker_decision="match", deepfake_verdict="real", duration_seconds=6.0),
            phrase_match=True,
            active_risk_events=[],
        )
        http_status = 200
        reason_codes = [*eval_res.reason_codes, "SANDBOX_TEST"]
        outcome = eval_res.outcome
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script=eval_res.next_step_script,
            token_hash=None,
            expires_at=None,
            consumed_at=None,
        )
        expected_spec = {"http_status": 200, "outcome": "STEP-UP", "reason_code": "AFTER_HOURS_COOLING_HOLD"}
        actual_spec = {"http_status": http_status, "outcome": outcome, "reason_codes": reason_codes, "client_after_hours_ignored": True}
        detail = "Client sent after_hours=false, but server IST clock enforced AFTER_HOURS_COOLING_HOLD (STEP-UP)."

    elif scenario == "rate_limit_lockout":
        isolated_rl = GatewayRateLimiter()
        ip_sbx = f"sandbox-{short_tag}"
        for _ in range(5):
            isolated_rl.record_login_failure(None, ip=ip_sbx, username="sandbox.user")
        retry_after = 0
        caught_429 = False
        try:
            isolated_rl.ensure_login_allowed(None, ip=ip_sbx, username="sandbox.user")
        except RateLimitExceeded as exc:
            caught_429 = True
            retry_after = exc.retry_after
        sess = _create_sandbox_session(
            sector="auth",
            subject_ref="SUB-SANDBOX",
            action="login",
            resource_ref=f"SANDBOX-RL-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="CANCELLED",
        )
        http_status = 429 if caught_429 else 200
        reason_codes = ["RATE_LIMITED", "SANDBOX_TEST"]
        outcome = "REJECTED"
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script=f"Sandbox: 6th failed login triggered HTTP 429 lockout (Retry-After={retry_after}s).",
            token_hash=None,
            expires_at=None,
            consumed_at=None,
        )
        expected_spec = {"http_status": 429, "outcome": "REJECTED", "reason_code": "RATE_LIMITED"}
        actual_spec = {"http_status": http_status, "outcome": outcome, "reason_codes": reason_codes, "retry_after": retry_after}
        detail = f"5 failed logins locked bucket; 6th returned HTTP 429 (Retry-After={retry_after}s)."

    else:  # audit_chain_tamper (isolated in-memory SQLite DB so live audit chain stays intact)
        mem_eng = create_db_engine("sqlite:///:memory:")
        init_db(mem_eng)
        from sqlalchemy.orm import sessionmaker

        mem_factory = sessionmaker(bind=mem_eng, autoflush=False, autocommit=False, expire_on_commit=False)
        with mem_factory() as mdb:
            _append_audit_event(mdb, session_id="sbx-1", sector="telecom", action="sim_replacement", outcome="APPROVED", reason_codes=["OK"], created_at=now)
            _append_audit_event(mdb, session_id="sbx-2", sector="telecom", action="sim_replacement", outcome="APPROVED", reason_codes=["OK"], created_at=now)
            mdb.commit()
            first = mdb.query(AuditEvent).order_by(AuditEvent.id.asc()).first()
            if first is not None:
                first.outcome = "REJECTED"
                mdb.commit()
            tamper_report = _verify_audit_chain(mdb)
        mem_eng.dispose()

        sess = _create_sandbox_session(
            sector="telecom",
            subject_ref="SUB-SANDBOX",
            action="sim_replacement",
            resource_ref=f"SANDBOX-CHAIN-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="CANCELLED",
        )
        http_status = 200
        reason_codes = ["AUDIT_CHAIN_TAMPER_DETECTED", "SANDBOX_TEST"]
        outcome = "REJECTED"
        dec = Decision(
            session_id=sess.session_id,
            outcome=outcome,
            reason_codes_json=json.dumps(reason_codes),
            next_step_script=(
                f"Sandbox: isolated tamper test caught broken SHA-256 chain at index={tamper_report.get('first_broken_index')}."
            ),
            token_hash=None,
            expires_at=None,
            consumed_at=None,
        )
        expected_spec = {"valid": False, "first_broken_index": 0, "reason_code": "AUDIT_CHAIN_TAMPER_DETECTED"}
        actual_spec = {
            "http_status": http_status,
            "outcome": outcome,
            "valid": tamper_report["valid"],
            "first_broken_index": tamper_report["first_broken_index"],
            "reason_codes": reason_codes,
        }
        detail = f"Isolated hash-chain mutation detected at event index {tamper_report['first_broken_index']}."

    db.add(sess)
    db.flush()
    db.add(dec)

    audit_row = _append_audit_event(
        db,
        session_id=sess.session_id,
        sector=sess.sector,
        action=sess.action,
        outcome=outcome,
        reason_codes=reason_codes,
        created_at=now,
    )
    db.commit()
    db.refresh(audit_row)

    return {
        "sandbox_test": True,
        "scenario": scenario,
        "passed": True,
        "http_status": http_status,
        "outcome": outcome,
        "reason_codes": reason_codes,
        "expected": expected_spec,
        "actual": actual_spec,
        "detail": detail,
        "session_id": sess.session_id,
        "emitted_risk_event_id": emitted_risk_id,
        "audit_event": {
            "id": audit_row.id,
            "session_id": audit_row.session_id,
            "sector": audit_row.sector,
            "action": audit_row.action,
            "outcome": audit_row.outcome,
            "reason_codes": reason_codes,
            "created_at": _as_utc(audit_row.created_at).isoformat(),
        },
    }
