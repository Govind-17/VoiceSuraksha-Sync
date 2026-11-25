"""FastAPI modular-monolith entrypoint for VoiceSuraksha-Sync (voicelock-gateway)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import statistics
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.auth import (
    AuthActor,
    clear_session_cookie,
    create_session_token,
    get_admin_credentials,
    get_current_actor,
    get_optional_actor,
    hash_password,
    require_admin,
    set_session_cookie,
    verify_password,
)
from app.db import get_db, init_db
from app.detectif_client import DetectifClient, DetectifProviderError
from app.models import (
    AuditEvent,
    Decision,
    Enrollment,
    RiskEvent,
    UserAccount,
    VerificationSession,
    VerificationTelemetry,
    VoiceResult,
    utcnow,
)
from app.policy import (
    GATEWAY_PROFILES,
    SessionPolicyInput,
    VoiceEvidenceInput,
    evaluate_voice_authorization,
    generate_challenge_phrase,
    hash_text_sha256,
    normalize_transcript,
    validate_sector_action,
    verify_phrase_match,
)

load_dotenv(override=False)

FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
MIN_ENROLL_SECONDS = 15.0
MAX_ENROLL_SECONDS = 60.0
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB guard for Render free-tier memory
SESSION_TTL_MINUTES = 2
DECISION_TOKEN_TTL_MINUTES = 5
RISK_EVENT_TTL_HOURS = 24
POLICY_VERSION = "2026.11-fail-closed-v1"


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="VoiceSuraksha-Sync (voicelock-gateway)",
    version="1.0.0",
    description="Action-bound voice-security gateway powered by DetectifAI Eidos & Aletheia.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


class CreateSessionRequest(BaseModel):
    sector: str
    subject_ref: str = Field(..., min_length=2, max_length=64)
    action: str
    resource_ref: str = Field(..., min_length=2, max_length=128)
    after_hours: bool = False
    privileged_role: bool = False
    amount_inr: float | None = None


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
        "audio_sha256": record.audio_sha256,
        "speech_seconds": record.speech_seconds,
        "created_at": _as_utc(record.created_at).isoformat(),
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    client: DetectifClient = Depends(get_detectif_client),
) -> dict[str, Any]:
    """Register a new end-user account AND enroll their voiceprint atomically."""
    clean_username = username.strip().lower()
    clean_display = display_name.strip()
    if len(clean_username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(clean_display) < 2:
        raise HTTPException(status_code=400, detail="Full name must be at least 2 characters.")

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
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Authenticate an admin or registered user and issue an HttpOnly session cookie."""
    clean_username = body.username.strip()
    admin_user, admin_pass = get_admin_credentials()

    if clean_username == admin_user and secrets.compare_digest(body.password, admin_pass):
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
        raise HTTPException(status_code=401, detail="Invalid username or password.")

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
                "created_at": _as_utc(r.created_at).isoformat(),
            }
            for r in rows
        ]
    }


@app.get("/api/v1/audit")
def list_audit(
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """List immutable audit trail records (admin only)."""
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
                "created_at": _as_utc(r.created_at).isoformat(),
            }
            for r in rows
        ]
    }


@app.get("/api/v1/risk-events")
def list_risk_events(
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """List cross-domain risk events (admin only)."""
    rows = db.scalars(select(RiskEvent).order_by(RiskEvent.id.desc())).all()
    return {
        "risk_events": [
            {
                "id": r.id,
                "source_sector": r.source_sector,
                "subject_ref": r.subject_ref,
                "event_type": r.event_type,
                "source_case_ref": r.source_case_ref,
                "risk_state": r.risk_state,
                "occurred_at": _as_utc(r.occurred_at).isoformat(),
                "expires_at": _as_utc(r.expires_at).isoformat(),
                "demo_flag": r.demo_flag,
            }
            for r in rows
        ]
    }


@app.post("/api/v1/demo/clear-alerts")
def clear_demo_alerts(
    db: Session = Depends(get_db),
    _admin: AuthActor = Depends(require_admin),
) -> dict[str, Any]:
    """Clear active RiskEvent rows for demo reset (admin only)."""
    result = db.execute(delete(RiskEvent).where(RiskEvent.risk_state == "ACTIVE"))
    db.commit()
    return {"status": "ok", "cleared_count": result.rowcount or 0}


@app.post("/api/v1/enrollments")
async def create_enrollment(
    subject_ref: str = Form(..., min_length=2, max_length=64),
    consent_confirmed: bool = Form(...),
    consent_ref: str = Form(..., min_length=2, max_length=128),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    client: DetectifClient = Depends(get_detectif_client),
    actor: AuthActor = Depends(get_current_actor),
) -> dict[str, Any]:
    """Enroll a consented speaker voiceprint via Eidos; store SHA-256 hash only."""
    if actor.role != "admin" and subject_ref != actor.subject_ref:
        raise HTTPException(status_code=403, detail="Users may only enroll their own assigned subject_ref.")

    existing = db.scalar(select(Enrollment).where(Enrollment.subject_ref == subject_ref))
    if actor.role != "admin" and existing is not None:
        raise HTTPException(status_code=409, detail="Voiceprint is already enrolled for this account.")

    wav_bytes, provider_res = await _validate_and_enroll_wav(
        subject_ref=subject_ref,
        consent_confirmed=consent_confirmed,
        file=file,
        client=client,
    )

    audio_digest = hashlib.sha256(wav_bytes).hexdigest()
    if existing is not None:
        existing.provider_speaker_id = provider_res.speaker_id
        existing.consent_ref = consent_ref
        existing.audio_sha256 = audio_digest
        existing.speech_seconds = provider_res.speech_seconds
        record = existing
    else:
        record = Enrollment(
            subject_ref=subject_ref,
            provider_speaker_id=provider_res.speaker_id,
            consent_ref=consent_ref,
            audio_sha256=audio_digest,
            speech_seconds=provider_res.speech_seconds,
        )
        db.add(record)

    db.add(
        VerificationTelemetry(
            session_id=None,
            event_kind="enroll",
            audio_sha256=audio_digest,
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

    session_id = str(uuid.uuid4())
    challenge_text = generate_challenge_phrase(body.sector, body.action, body.resource_ref)
    challenge_hash = hash_text_sha256(normalize_transcript(challenge_text))
    expires_at = utcnow() + timedelta(minutes=SESSION_TTL_MINUTES)

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
        after_hours=body.after_hours,
        privileged_role=body.privileged_role,
        amount_inr=body.amount_inr,
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
    }


@app.post("/api/v1/verification-sessions/{session_id}/audio")
async def submit_verification_audio(
    session_id: str,
    spoken_transcript: str = Form(""),
    strict_phrase: bool = Form(False),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    client: DetectifClient = Depends(get_detectif_client),
    actor: AuthActor = Depends(get_current_actor),
) -> dict[str, Any]:
    """Call DetectifAI Eidos+Aletheia to verify voice, evaluate policy, and issue one-time token on PASS."""
    session = db.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Verification session not found.")
    if actor.role != "admin" and session.subject_ref != actor.subject_ref:
        raise HTTPException(status_code=403, detail="Cannot submit audio for another user's session.")

    if session.status != "PENDING":
        raise HTTPException(status_code=409, detail=f"Session '{session_id}' has already been processed ({session.status}).")

    now = utcnow()
    wav_bytes = await _read_bounded_wav(file)
    wav_duration = _inspect_wav_duration(wav_bytes) if wav_bytes else None

    active_risks = db.scalars(
        select(RiskEvent).where(
            RiskEvent.subject_ref == session.subject_ref,
            RiskEvent.risk_state == "ACTIVE",
        )
    ).all()
    unexpired_risks = [r for r in active_risks if _as_utc(r.expires_at) > now]

    is_expired = now > _as_utc(session.expires_at)
    phrase_match = verify_phrase_match(session.challenge_text, spoken_transcript)
    should_call_provider = not is_expired and (phrase_match or not strict_phrase)

    provider_res = None
    provider_error: str | None = None

    if should_call_provider:
        enrollment = db.scalar(select(Enrollment).where(Enrollment.subject_ref == session.subject_ref))
        if enrollment is None:
            provider_error = "enrollment_missing"
        elif wav_duration is not None and wav_duration < 4.0:
            # Avoid calling combined deepfake verification on <4s clips; treat as INCONCLUSIVE_EVIDENCE.
            provider_error = None
        else:
            try:
                provider_res = client.verify_with_deepfake(
                    speaker_id=enrollment.provider_speaker_id,
                    wav_bytes=wav_bytes,
                )
            except DetectifProviderError as exc:
                provider_error = exc.error_code

    voice_input = None
    if should_call_provider:
        if provider_res is not None:
            voice_input = VoiceEvidenceInput(
                speaker_decision=provider_res.speaker_decision,
                deepfake_verdict=provider_res.deepfake_verdict,
                duration_seconds=provider_res.duration_seconds,
                provider_error=None,
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
        require_phrase_match=strict_phrase,
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
    elif evaluation.outcome == "STEP-UP":
        session.status = "STEP_UP"
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
        risk_row = RiskEvent(
            source_sector=session.sector,
            subject_ref=session.subject_ref,
            event_type="SIM_SWAP_ATTEMPT_BLOCKED",
            source_case_ref=session.session_id,
            risk_state="ACTIVE",
            occurred_at=now,
            expires_at=now + timedelta(hours=RISK_EVENT_TTL_HOURS),
            demo_flag=True,
        )
        db.add(risk_row)
        db.flush()
        emitted_risk_event = {
            "id": risk_row.id,
            "event_type": risk_row.event_type,
            "subject_ref": risk_row.subject_ref,
            "risk_state": risk_row.risk_state,
            "demo_flag": risk_row.demo_flag,
        }

    audit_row = AuditEvent(
        session_id=session.session_id,
        sector=session.sector,
        action=session.action,
        outcome=evaluation.outcome,
        reason_codes_json=json.dumps(evaluation.reason_codes),
        created_at=now,
    )
    db.add(audit_row)
    db.add(
        VerificationTelemetry(
            session_id=session.session_id,
            event_kind="verify",
            audio_sha256=hashlib.sha256(wav_bytes).hexdigest() if wav_bytes else None,
            credits_balance=provider_res.credits_balance if provider_res else None,
        )
    )
    db.commit()

    if actor.role != "admin":
        return {
            "session_id": session.session_id,
            "status": session.status,
            "phrase_match": phrase_match,
            "evidence_state": evaluation.evidence_state,
            "outcome": evaluation.outcome,
            "reason_codes": evaluation.reason_codes,
            "next_step_script": evaluation.next_step_script,
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
        "next_step_script": evaluation.next_step_script,
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
    """Validate and atomically consume a one-time decision token bound to session/action/resource."""
    session = db.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id))
    decision = db.scalar(select(Decision).where(Decision.session_id == session_id))
    if session is None or decision is None:
        raise HTTPException(status_code=404, detail="Decision not found for session.")

    if decision.outcome != "APPROVED" or decision.token_hash is None:
        raise HTTPException(status_code=409, detail=f"Decision outcome '{decision.outcome}' did not issue a consumable token.")

    if decision.consumed_at is not None:
        raise HTTPException(status_code=409, detail="Decision token has already been consumed.")

    now = utcnow()
    if decision.expires_at is None or now > _as_utc(decision.expires_at):
        raise HTTPException(status_code=409, detail="Decision token has expired.")

    if session.action != body.action or session.resource_ref != body.resource_ref:
        raise HTTPException(status_code=409, detail="Action or resource_ref does not match the bound verification session.")

    provided_hash = hash_text_sha256(body.decision_token)
    if not secrets.compare_digest(provided_hash, decision.token_hash):
        raise HTTPException(status_code=403, detail="Invalid decision token.")

    decision.consumed_at = now
    db.add(
        AuditEvent(
            session_id=session.session_id,
            sector=session.sector,
            action=session.action,
            outcome="TOKEN_CONSUMED",
            reason_codes_json=json.dumps(["ACTION_RELEASED"]),
            created_at=now,
        )
    )
    db.commit()

    return {
        "consumed": True,
        "session_id": session.session_id,
        "sector": session.sector,
        "action": session.action,
        "resource_ref": session.resource_ref,
        "consumed_at": now.isoformat(),
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
    """Return real-time gateway telemetry and credit metrics queried from SQLite."""
    now = utcnow()
    decisions = db.scalars(select(Decision)).all()
    total_verifications = len(decisions)
    pass_count = sum(1 for d in decisions if d.outcome == "APPROVED")
    blocked_count = sum(1 for d in decisions if d.outcome == "REJECTED")
    step_up_count = sum(1 for d in decisions if d.outcome == "STEP-UP")

    latencies = [
        float(v)
        for v in db.scalars(select(VoiceResult.latency_ms).where(VoiceResult.latency_ms.is_not(None))).all()
        if v is not None
    ]
    p50_latency_ms = round(statistics.median(latencies), 1) if latencies else None

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
        "pass_count": pass_count,
        "step_up_count": step_up_count,
        "p50_latency_ms": p50_latency_ms,
        "last_credits_balance": last_credits_balance,
        "active_risk_events_count": active_risk_events_count,
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
    }
    if scenario not in valid_scenarios:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported sandbox scenario '{scenario}'. Valid: {', '.join(sorted(valid_scenarios))}",
        )

    now = utcnow()
    short_tag = secrets.token_hex(3).upper()
    emitted_risk_id: int | None = None

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
        detail = "Expired verification challenge rejected fail-closed."

    elif scenario == "provider_outage_fail_closed":
        sess = _create_sandbox_session(
            sector="helpdesk",
            subject_ref="SUB-SANDBOX",
            action="mfa_reset",
            resource_ref=f"SANDBOX-503-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="CANCELLED",
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
        detail = "Upstream provider 503 failed closed (REJECTED / CANCELLED)."

    else:  # cross_domain_sim_alert
        target_subject = "SUB-SANDBOX"
        risk_row = RiskEvent(
            source_sector="telecom",
            subject_ref=target_subject,
            event_type="SIM_SWAP_ATTEMPT_BLOCKED",
            source_case_ref=f"SANDBOX-SIM-{short_tag}",
            risk_state="ACTIVE",
            occurred_at=now,
            expires_at=now + timedelta(hours=RISK_EVENT_TTL_HOURS),
            demo_flag=True,
        )
        db.add(risk_row)
        db.flush()
        emitted_risk_id = risk_row.id

        sess = _create_sandbox_session(
            sector="banking",
            subject_ref=target_subject,
            action="card_replacement",
            resource_ref=f"SANDBOX-BANK-{short_tag}",
            expires_at=now + timedelta(minutes=SESSION_TTL_MINUTES),
            status="CANCELLED",
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
        http_status = 403
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
        detail = "Blocked Telecom SIM swap propagated ACTIVE risk event -> Banking card replacement blocked (REJECTED)."

    db.add(sess)
    db.flush()
    db.add(dec)

    audit_row = AuditEvent(
        session_id=sess.session_id,
        sector=sess.sector,
        action=sess.action,
        outcome=outcome,
        reason_codes_json=json.dumps(reason_codes),
        created_at=now,
    )
    db.add(audit_row)
    db.commit()
    db.refresh(audit_row)

    return {
        "sandbox_test": True,
        "scenario": scenario,
        "passed": True,
        "http_status": http_status,
        "outcome": outcome,
        "reason_codes": reason_codes,
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
