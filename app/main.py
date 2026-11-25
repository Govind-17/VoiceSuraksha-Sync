"""FastAPI modular-monolith entrypoint for VoiceSuraksha-Sync (voicelock-gateway)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db, init_db
from app.detectif_client import DetectifClient, DetectifProviderError
from app.models import AuditEvent, Decision, Enrollment, RiskEvent, VerificationSession, VoiceResult, utcnow
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
SESSION_TTL_MINUTES = 2
DECISION_TOKEN_TTL_MINUTES = 5
RISK_EVENT_TTL_HOURS = 24


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


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the single-file SOC UI directly at '/'."""
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


@app.get("/api/v1/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return gateway operational mode, available profiles, and enrolled speaker count."""
    demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
    enrolled_count = db.query(Enrollment).count()
    return {
        "status": "ok",
        "service": "voicelock-gateway",
        "mode": "DEMO" if demo_mode else "LIVE",
        "profiles": list(GATEWAY_PROFILES.keys()),
        "enrolled_count": enrolled_count,
    }


@app.get("/api/v1/profiles")
def list_profiles() -> dict[str, Any]:
    """Return the 4 configured GATEWAY_PROFILES."""
    return {"profiles": GATEWAY_PROFILES}


@app.get("/api/v1/enrollments")
def list_enrollments(db: Session = Depends(get_db)) -> dict[str, Any]:
    """List pseudonymous speaker enrollments (metadata only)."""
    rows = db.scalars(select(Enrollment).order_by(Enrollment.created_at.desc())).all()
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
def list_audit(db: Session = Depends(get_db)) -> dict[str, Any]:
    """List immutable audit trail records."""
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
def list_risk_events(db: Session = Depends(get_db)) -> dict[str, Any]:
    """List cross-domain risk events (e.g., simulated SIM_SWAP_ATTEMPT_BLOCKED)."""
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


@app.post("/api/v1/enrollments")
async def create_enrollment(
    subject_ref: str = Form(..., min_length=2, max_length=64),
    consent_confirmed: bool = Form(...),
    consent_ref: str = Form(..., min_length=2, max_length=128),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    client: DetectifClient = Depends(get_detectif_client),
) -> dict[str, Any]:
    """Enroll a consented speaker voiceprint via Eidos; store SHA-256 hash only."""
    if not consent_confirmed:
        raise HTTPException(status_code=400, detail="Explicit speaker consent (consent_confirmed=true) is required.")

    wav_bytes = await file.read()
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

    audio_digest = hashlib.sha256(wav_bytes).hexdigest()
    existing = db.scalar(select(Enrollment).where(Enrollment.subject_ref == subject_ref))
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

    db.commit()
    db.refresh(record)

    return {
        "enrollment": {
            "id": record.id,
            "subject_ref": record.subject_ref,
            "provider_speaker_id": record.provider_speaker_id,
            "consent_ref": record.consent_ref,
            "audio_sha256": record.audio_sha256,
            "speech_seconds": record.speech_seconds,
            "duration_seconds": provider_res.duration_seconds,
            "created_at": _as_utc(record.created_at).isoformat(),
        },
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
) -> dict[str, Any]:
    """Create a short-lived action-bound verification session with a random challenge phrase."""
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    client: DetectifClient = Depends(get_detectif_client),
) -> dict[str, Any]:
    """Verify phrase, call DetectifAI Eidos+Aletheia, run policy engine, and issue one-time token on PASS."""
    session = db.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Verification session not found.")

    if session.status != "PENDING":
        raise HTTPException(status_code=409, detail=f"Session '{session_id}' has already been processed ({session.status}).")

    now = utcnow()
    wav_bytes = await file.read()
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

    provider_res = None
    provider_error: str | None = None

    if not is_expired and phrase_match:
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
    if not is_expired and phrase_match:
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
    )
    evaluation = evaluate_voice_authorization(
        session=policy_session,
        voice_result=voice_input,
        phrase_match=phrase_match,
        active_risk_events=unexpired_risks,
    )

    session.status = "EXPIRED" if is_expired else "COMPLETED"

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
    if evaluation.outcome == "PASS":
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
    db.commit()

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
) -> dict[str, Any]:
    """Validate and atomically consume a one-time decision token bound to session/action/resource."""
    session = db.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id))
    decision = db.scalar(select(Decision).where(Decision.session_id == session_id))
    if session is None or decision is None:
        raise HTTPException(status_code=404, detail="Decision not found for session.")

    if decision.outcome != "PASS" or decision.token_hash is None:
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
