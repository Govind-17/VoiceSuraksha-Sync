"""Gateway integration tests using saved DetectifAI fixtures in tests/fixtures/."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import secrets
import struct
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.asr import ServerASRAdapter, ServerASRResult
from app.audio_fingerprint import MAX_PERCEPTUAL_HAMMING_BITS, compute_perceptual_hash, hamming_distance_hex
from app.auth import PBKDF2_ITERATIONS, hash_password, validate_live_secrets, verify_password
from app.db import Base, CURRENT_SCHEMA_VERSION, create_db_engine, get_database_url, get_storage_backend, init_db, is_storage_durable
from app.detectif_client import DetectifClient
from app.main import _is_ist_after_hours, app, get_asr_adapter, get_db, get_detectif_client
from app.models import AuditEvent, Enrollment, SchemaVersion, VerificationSession, utcnow
from app.policy import normalize_transcript, verify_phrase_match, verify_risk_event_signature
from app.rate_limit import gateway_rate_limiter

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ENROLL_FIXTURE = json.loads((FIXTURES_DIR / "eidos_enroll.json").read_text(encoding="utf-8"))
VERIFY_FIXTURE = json.loads((FIXTURES_DIR / "eidos_verify_with_deepfake.json").read_text(encoding="utf-8"))
ALETHEIA_DETECT_FIXTURE = json.loads((FIXTURES_DIR / "aletheia_detect.json").read_text(encoding="utf-8"))
_WAV_NONCE = 0


def make_pcm_wav_bytes(
    duration_seconds: float,
    sample_rate: int = 16000,
    *,
    silent: bool = False,
    clipped: bool = False,
) -> bytes:
    """Create valid 16kHz mono 16-bit PCM WAV bytes in memory for tests."""
    global _WAV_NONCE
    _WAV_NONCE += 1
    num_frames = int(duration_seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if silent:
            wf.writeframes(b"\x00\x00" * num_frames)
        elif clipped:
            frames = struct.pack(f"<{num_frames}h", *(32767 if i % 2 == 0 else -32768 for i in range(num_frames)))
            wf.writeframes(frames)
        else:
            # Synthesize distinct time-varying formant-like chords per nonce so
            # perceptual spectral fingerprints separate distinct clips while
            # staying invariant under volume scaling and slight trimming.
            rng = np.random.default_rng(_WAV_NONCE * 7919 + 17)
            t = np.arange(num_frames, dtype=np.float64) / float(sample_rate)
            sig = np.zeros(num_frames, dtype=np.float64)
            band_freqs = np.geomspace(180.0, 4200.0, 9)
            win_edges = np.linspace(0, num_frames, 10, dtype=int)
            for w in range(9):
                s0, s1 = int(win_edges[w]), int(win_edges[w + 1])
                gains = rng.uniform(0.15, 1.8, size=len(band_freqs))
                tw = t[s0:s1]
                chunk = np.zeros(s1 - s0, dtype=np.float64)
                for f_hz, g in zip(band_freqs, gains):
                    chunk += g * np.sin(2.0 * math.pi * f_hz * tw)
                sig[s0:s1] = chunk
            rms = float(np.sqrt(np.mean(sig * sig))) or 1.0
            pcm = np.clip((sig / rms) * 6500.0, -28000.0, 28000.0).astype("<i2")
            wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture()
def db_session_factory(tmp_path):
    db_file = tmp_path / "test_voicesuraksha.db"
    engine = create_db_engine(f"sqlite:///{db_file}")
    init_db(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    try:
        yield TestingSessionLocal
    finally:
        engine.dispose()


def make_mock_client(handler) -> DetectifClient:
    transport = httpx.MockTransport(handler)
    return DetectifClient(
        api_key="test-key-never-logged",
        base_url="https://platform.detectif.ai",
        transport=transport,
    )


@pytest.fixture()
def make_test_client(db_session_factory, monkeypatch):
    def _factory(mock_handler, *, auto_login_admin: bool = True, demo_mode: bool = False):
        gateway_rate_limiter.reset()
        monkeypatch.setenv("DEMO_MODE", "true" if demo_mode else "false")
        monkeypatch.setenv("AUTH_SECRET", "pytest-live-auth-secret-32-chars-ok")
        monkeypatch.setenv("WEBHOOK_SIGNING_SECRET", "pytest-live-webhook-secret-32-chars")
        monkeypatch.setenv("ADMIN_PASSWORD", "pytest-strong-admin-password-2026")
        monkeypatch.setenv("ASR_API_KEY", "pytest-hosted-asr-key")
        monkeypatch.setattr("app.main._is_ist_after_hours", lambda _now=None: False)

        asr_cfg: dict = {"available": True, "transcript": None}

        class _TestASRAdapter(ServerASRAdapter):
            def transcribe_wav(self, wav_bytes: bytes) -> ServerASRResult:
                if not asr_cfg["available"]:
                    return ServerASRResult(available=False, transcript=None, provider="sarvam", error="asr_unavailable")
                if asr_cfg["transcript"] is not None:
                    return ServerASRResult(available=True, transcript=asr_cfg["transcript"], provider="sarvam")
                with db_session_factory() as db:
                    latest = db.query(VerificationSession).order_by(VerificationSession.expires_at.desc()).first()
                    txt = latest.challenge_text if latest else ""
                return ServerASRResult(available=True, transcript=txt, provider="sarvam")

        def _override_get_db():
            db = db_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _override_get_db
        app.dependency_overrides[get_detectif_client] = lambda: make_mock_client(mock_handler)
        app.dependency_overrides[get_asr_adapter] = lambda: _TestASRAdapter()
        base_url = "http://testserver" if demo_mode else "https://testserver"
        client = TestClient(app, base_url=base_url)
        client._asr_cfg = asr_cfg  # type: ignore[attr-defined]
        if auto_login_admin:
            from app.auth import get_admin_credentials

            admin_u, admin_p = get_admin_credentials()
            login_resp = client.post("/api/v1/auth/login", json={"username": admin_u, "password": admin_p})
            assert login_resp.status_code == 200, login_resp.text
        return client

    yield _factory
    app.dependency_overrides.clear()


def _default_fixture_handler(verify_override: dict | None = None):
    captured_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.append(dict(request.headers))
        assert "idempotency-key" in request.headers
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            payload = copy.deepcopy(ENROLL_FIXTURE)
            return httpx.Response(200, json=payload)
        if request.url.path == "/v1/speakers/verify":
            payload = copy.deepcopy(verify_override or VERIFY_FIXTURE)
            return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "Unknown path"}})

    return handler, captured_headers


def _enroll_subject(client: TestClient, subject_ref: str = "SUB-104") -> dict:
    wav_16s = make_pcm_wav_bytes(16.0)
    resp = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": subject_ref,
            "consent_confirmed": "true",
            "consent_ref": f"CONSENT-{subject_ref}",
            "consent_text": "Explicit speaker consent confirmed for voiceprint enrollment.",
        },
        files={"file": ("enroll.wav", wav_16s, "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_happy_path_telecom_sim_replacement_and_consume(make_test_client, db_session_factory):
    handler, captured_headers = _default_fixture_handler()
    client = make_test_client(handler)

    # Root HTML and health/profiles endpoints
    assert client.get("/").status_code == 200
    health = client.get("/api/v1/health").json()
    assert health["status"] == "ok"
    assert set(health["profiles"]) == {"telecom", "helpdesk", "banking", "campus"}

    # Consent required (HTTP 400 if false)
    bad_consent = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": "SUB-104",
            "consent_confirmed": "false",
            "consent_ref": "C-0",
            "consent_text": "Explicit speaker consent confirmed for testing.",
        },
        files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert bad_consent.status_code == 400

    # Too short enrollment rejected (HTTP 400)
    short_enroll = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": "SUB-104",
            "consent_confirmed": "true",
            "consent_ref": "C-0",
            "consent_text": "Explicit speaker consent confirmed for testing.",
        },
        files={"file": ("short.wav", make_pcm_wav_bytes(5.0), "audio/wav")},
    )
    assert short_enroll.status_code == 400

    # Enroll SUB-104
    enroll_data = _enroll_subject(client, "SUB-104")
    assert enroll_data["enrollment"]["subject_ref"] == "SUB-104"
    assert len(enroll_data["enrollment"]["audio_sha256"]) == 64

    # Create telecom sim_replacement session
    sess_resp = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-104",
            "action": "sim_replacement",
            "resource_ref": "SIM-MASKED-8841",
        },
    )
    assert sess_resp.status_code == 200
    sess = sess_resp.json()
    session_id = sess["session_id"]

    # Verify audio with matching spoken_transcript
    audio_resp = client.post(
        f"/api/v1/verification-sessions/{session_id}/audio",
        data={"spoken_transcript": sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert audio_resp.status_code == 200
    result = audio_resp.json()
    assert result["outcome"] == "APPROVED"
    assert result["status"] == "APPROVED"
    assert result["evidence_state"] == "ACOUSTIC_PASS"
    assert result["phrase_match"] is True
    assert result["voice_result"]["speaker_decision"] == "match"
    assert result["voice_result"]["deepfake_verdict"] == "real"
    assert result["decision_token"]

    # Consume one-time token bound to action & resource
    consume_resp = client.post(
        f"/api/v1/decisions/{session_id}/consume",
        json={
            "decision_token": result["decision_token"],
            "action": "sim_replacement",
            "resource_ref": "SIM-MASKED-8841",
        },
    )
    assert consume_resp.status_code == 200
    assert consume_resp.json()["consumed"] is True
    assert len(captured_headers) == 3  # deepfake gate + enroll + verify

    # Confirm raw audio is never stored in SQLite
    with db_session_factory() as db:
        enrollment_row = db.query(Enrollment).filter_by(subject_ref="SUB-104").one()
        assert enrollment_row.audio_sha256 == enroll_data["enrollment"]["audio_sha256"]
        assert db.query(AuditEvent).filter_by(session_id=session_id).count() == 2


def test_wrong_phrase_and_session_replay(make_test_client):
    handler, captured_headers = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-201")

    sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-201",
            "action": "sim_replacement",
            "resource_ref": "SIM-MASKED-1102",
        },
    ).json()

    client._asr_cfg["transcript"] = "some completely wrong replayed phrase"
    resp = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": "some completely wrong replayed phrase", "strict_phrase": "true"},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome"] == "REJECTED"
    assert data["status"] == "CANCELLED"
    assert data["reason_codes"] == ["CHALLENGE_MISMATCH"]
    assert data["decision_token"] is None
    assert data["emitted_risk_event"]["event_type"] == "SIM_SWAP_ATTEMPT_BLOCKED"
    # Only enrollment (deepfake detect + enroll) called DetectifAI, not verify
    assert len(captured_headers) == 2

    # Replaying audio against the same session must return HTTP 409
    replay = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert replay.status_code == 409


def test_expired_session_rejected(make_test_client, db_session_factory):
    handler, captured_headers = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-202")

    sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": "SUB-202",
            "action": "mfa_reset",
            "resource_ref": "MFA-DEV-77",
        },
    ).json()

    with db_session_factory() as db:
        row = db.query(VerificationSession).filter_by(session_id=sess["session_id"]).one()
        row.expires_at = utcnow() - timedelta(minutes=1)
        db.commit()

    resp = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome"] == "REJECTED"
    assert data["reason_codes"] == ["CHALLENGE_EXPIRED"]
    assert data["decision_token"] is None
    assert len(captured_headers) == 2  # deepfake detect + enroll only


def test_synthetic_deepfake_verdict_rejected(make_test_client):
    fake_verify = copy.deepcopy(VERIFY_FIXTURE)
    fake_verify["deepfake"]["verdict"] = "fake"
    fake_verify["deepfake"]["is_synthetic"] = True
    fake_verify["deepfake"]["label"] = "synthetic"
    fake_verify["deepfake"]["score"] = 0.91

    handler, _ = _default_fixture_handler(verify_override=fake_verify)
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-203")

    sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-203",
            "action": "esim_transfer",
            "resource_ref": "ESIM-MASKED-4410",
        },
    ).json()

    resp = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome"] == "REJECTED"
    assert data["status"] == "CANCELLED"
    assert data["evidence_state"] == "SYNTHETIC_INDICATOR"
    assert "DEEPFAKE_SIGNAL" in data["reason_codes"]
    assert data["decision_token"] is None
    assert data["emitted_risk_event"]["event_type"] == "SIM_SWAP_ATTEMPT_BLOCKED"


def test_token_reuse_and_action_mismatch_return_409(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-204")

    sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-204",
            "action": "mnp_port_out",
            "resource_ref": "MNP-MASKED-9001",
        },
    ).json()

    audio_data = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    token = audio_data["decision_token"]
    assert token

    # Wrong action/resource must fail with HTTP 409 and NOT consume the token
    mismatch = client.post(
        f"/api/v1/decisions/{sess['session_id']}/consume",
        json={
            "decision_token": token,
            "action": "sim_replacement",
            "resource_ref": "MNP-MASKED-9001",
        },
    )
    assert mismatch.status_code == 409

    # Valid consume succeeds
    first = client.post(
        f"/api/v1/decisions/{sess['session_id']}/consume",
        json={
            "decision_token": token,
            "action": "mnp_port_out",
            "resource_ref": "MNP-MASKED-9001",
        },
    )
    assert first.status_code == 200

    # Reuse must fail with HTTP 409
    reuse = client.post(
        f"/api/v1/decisions/{sess['session_id']}/consume",
        json={
            "decision_token": token,
            "action": "mnp_port_out",
            "resource_ref": "MNP-MASKED-9001",
        },
    )
    assert reuse.status_code == 409


def test_cross_domain_sim_swap_alert_holds_banking(make_test_client):
    fake_verify = copy.deepcopy(VERIFY_FIXTURE)
    fake_verify["deepfake"]["verdict"] = "fake"
    fake_verify["deepfake"]["is_synthetic"] = True
    fake_verify["deepfake"]["score"] = 0.89

    call_mode = {"fake": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            if call_mode["fake"]:
                return httpx.Response(200, json=copy.deepcopy(fake_verify))
            return httpx.Response(200, json=copy.deepcopy(VERIFY_FIXTURE))
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-301")

    # 1. Telecom blocked attempt emits SIM_SWAP_ATTEMPT_BLOCKED
    tel_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-301",
            "action": "sim_replacement",
            "resource_ref": "SIM-MASKED-3010",
        },
    ).json()
    tel_res = client.post(
        f"/api/v1/verification-sessions/{tel_sess['session_id']}/audio",
        data={"spoken_transcript": tel_sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert tel_res["emitted_risk_event"]["event_type"] == "SIM_SWAP_ATTEMPT_BLOCKED"

    risks = client.get("/api/v1/risk-events").json()["risk_events"]
    assert any(r["subject_ref"] == "SUB-301" and r["event_type"] == "SIM_SWAP_ATTEMPT_BLOCKED" for r in risks)

    # 2. Subsequent banking action with matching voice becomes STEP-UP (CROSS_DOMAIN_ALERT), not REJECTED
    call_mode["fake"] = False
    bank_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "banking",
            "subject_ref": "SUB-301",
            "action": "credit_limit_increase",
            "resource_ref": "CARD-MASKED-9912",
            "amount_inr": 50000,
        },
    ).json()
    bank_res = client.post(
        f"/api/v1/verification-sessions/{bank_sess['session_id']}/audio",
        data={"spoken_transcript": bank_sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert bank_res["outcome"] == "STEP-UP"
    assert bank_res["status"] == "STEP_UP"
    assert "CROSS_DOMAIN_ALERT" in bank_res["reason_codes"]
    assert bank_res["decision_token"] is None
    assert len(bank_res["fallback_options"]) == 3


def test_provider_timeout_and_403_fail_closed(make_test_client):
    mode = {"behavior": "timeout"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            if mode["behavior"] == "timeout":
                raise httpx.ReadTimeout("Simulated upstream timeout", request=request)
            return httpx.Response(
                403,
                json={"error": {"code": "insufficient_scope", "message": "Missing deepfake:detect scope"}},
            )
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-401")

    # Timeout case -> fail closed REVIEW / HOLD with non-voice fallbacks
    sess1 = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": "SUB-401",
            "action": "mfa_reset",
            "resource_ref": "MFA-401",
        },
    ).json()
    res1 = client.post(
        f"/api/v1/verification-sessions/{sess1['session_id']}/audio",
        data={"spoken_transcript": sess1["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert res1["outcome"] == "REVIEW / HOLD"
    assert res1["status"] == "HOLD"
    assert res1["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert res1["reason_codes"] == ["PROVIDER_UNAVAILABLE"]
    assert res1["decision_token"] is None
    assert {opt["method"] for opt in res1["fallback_options"]} == {"trusted_app_push", "in_store_kyc", "manager_callback"}

    # 403 insufficient_scope case -> fail closed REVIEW / HOLD with non-voice fallbacks
    mode["behavior"] = "403"
    sess2 = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "campus",
            "subject_ref": "SUB-401",
            "action": "portal_recovery",
            "resource_ref": "CAMPUS-401",
        },
    ).json()
    res2 = client.post(
        f"/api/v1/verification-sessions/{sess2['session_id']}/audio",
        data={"spoken_transcript": sess2["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert res2["outcome"] == "REVIEW / HOLD"
    assert res2["status"] == "HOLD"
    assert res2["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert res2["reason_codes"] == ["PROVIDER_UNAVAILABLE"]
    assert res2["decision_token"] is None
    assert len(res2["fallback_options"]) == 3


def test_contextual_step_ups_and_protective_low_friction(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-501")

    # Set privileged_role server-side via admin endpoint (caller field is ignored)
    assert (
        client.patch("/api/v1/admin/subjects/SUB-501/privilege", json={"privileged_role": True}).status_code
        == 200
    )

    # Privileged helpdesk MFA reset -> STEP-UP even on voice match
    priv_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": "SUB-501",
            "action": "mfa_reset",
            "resource_ref": "MFA-ADMIN-01",
        },
    ).json()
    priv_res = client.post(
        f"/api/v1/verification-sessions/{priv_sess['session_id']}/audio",
        data={"spoken_transcript": priv_sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert priv_res["outcome"] == "STEP-UP"
    assert "PRIVILEGED_APPROVAL_REQUIRED" in priv_res["reason_codes"]
    assert priv_res["decision_token"] is None

    # Protective emergency_sim_block stays PASS even after hours
    emerg_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-501",
            "action": "emergency_sim_block",
            "resource_ref": "SIM-MASKED-501",
            "after_hours": True,
        },
    ).json()
    emerg_res = client.post(
        f"/api/v1/verification-sessions/{emerg_sess['session_id']}/audio",
        data={"spoken_transcript": emerg_sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert emerg_res["outcome"] == "APPROVED"
    assert emerg_res["decision_token"] is not None

    # Short <4s clip -> INCONCLUSIVE_EVIDENCE (REVIEW / HOLD with non-voice fallbacks)
    short_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-501",
            "action": "sim_replacement",
            "resource_ref": "SIM-MASKED-502",
        },
    ).json()
    short_res = client.post(
        f"/api/v1/verification-sessions/{short_sess['session_id']}/audio",
        data={"spoken_transcript": short_sess["challenge_text"]},
        files={"file": ("short.wav", make_pcm_wav_bytes(2.5), "audio/wav")},
    ).json()
    assert short_res["outcome"] == "REVIEW / HOLD"
    assert short_res["status"] == "HOLD"
    assert short_res["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert short_res["reason_codes"] == ["AUDIO_QUALITY_LOW"]
    assert len(short_res["fallback_options"]) == 3


def test_clear_demo_alerts_keeps_enrollments_intact(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-601")

    # Trigger a simulated SIM_SWAP_ATTEMPT_BLOCKED alert via phrase mismatch
    tel_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-601",
            "action": "sim_replacement",
            "resource_ref": "SIM-MASKED-6010",
        },
    ).json()
    client._asr_cfg["transcript"] = "wrong phrase"
    tel_res = client.post(
        f"/api/v1/verification-sessions/{tel_sess['session_id']}/audio",
        data={"spoken_transcript": "wrong phrase", "strict_phrase": "true"},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    client._asr_cfg["transcript"] = None
    assert tel_res["emitted_risk_event"]["event_type"] == "SIM_SWAP_ATTEMPT_BLOCKED"
    assert len(client.get("/api/v1/risk-events").json()["risk_events"]) == 1

    # Clear active demo alerts; enrollment must remain intact
    cleared = client.post("/api/v1/demo/clear-alerts")
    assert cleared.status_code == 200
    assert cleared.json() == {"status": "ok", "cleared_count": 1}
    assert client.get("/api/v1/risk-events").json()["risk_events"] == []
    enrollments = client.get("/api/v1/enrollments").json()["enrollments"]
    assert any(e["subject_ref"] == "SUB-601" for e in enrollments)

    # Subsequent banking action with matching voice now passes cleanly
    bank_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "banking",
            "subject_ref": "SUB-601",
            "action": "credit_limit_increase",
            "resource_ref": "CARD-MASKED-6010",
            "amount_inr": 50000,
        },
    ).json()
    bank_res = client.post(
        f"/api/v1/verification-sessions/{bank_sess['session_id']}/audio",
        data={"spoken_transcript": bank_sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert bank_res["outcome"] == "APPROVED"
    assert bank_res["decision_token"] is not None


def test_critical_actions_enforce_server_asr_while_low_friction_allows_voice_first(make_test_client):
    handler, captured_headers = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-701")

    # Critical action (sim_replacement) rejects when server ASR hears a non-matching phrase,
    # even if client lies in spoken_transcript
    crit_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-701",
            "action": "sim_replacement",
            "resource_ref": "SIM-MASKED-7010",
        },
    ).json()
    assert crit_sess["challenge_text"].startswith("Confirm sim replacement code ")
    client._asr_cfg["transcript"] = "hi my name is attacker replaying wrong text"
    crit_res = client.post(
        f"/api/v1/verification-sessions/{crit_sess['session_id']}/audio",
        data={"spoken_transcript": crit_sess["challenge_text"], "strict_phrase": "false"},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert crit_res["outcome"] == "REJECTED"
    assert crit_res["reason_codes"] == ["CHALLENGE_MISMATCH"]

    # Non-critical protective action (emergency_sim_block, critical=False) works voice-first
    low_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-701",
            "action": "emergency_sim_block",
            "resource_ref": "SIM-MASKED-7011",
        },
    ).json()
    client._asr_cfg["transcript"] = "please block my lost phone immediately"
    res = client.post(
        f"/api/v1/verification-sessions/{low_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["phrase_match"] is False
    assert data["outcome"] == "APPROVED"
    assert data["status"] == "APPROVED"
    assert data["voice_result"]["speaker_decision"] == "match"
    assert data["voice_result"]["deepfake_verdict"] == "real"
    assert data["decision_token"] is not None
    assert len(captured_headers) == 3  # deepfake gate + enroll + low-friction verify


def test_render_readiness_head_health_and_upload_limit(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)

    assert client.head("/").status_code == 200
    assert client.head("/api/v1/health").status_code == 200

    oversized = b"0" * (15 * 1024 * 1024 + 10)
    resp = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": "SUB-BIG",
            "consent_confirmed": "true",
            "consent_ref": "C-BIG",
            "consent_text": "Explicit speaker consent confirmed for testing.",
        },
        files={"file": ("huge.wav", oversized, "audio/wav")},
    )
    assert resp.status_code == 413


def test_unauthenticated_blocked_while_health_and_ui_public(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler, auto_login_admin=False)

    assert client.get("/").status_code == 200
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert "enrolled_count" not in health.json()
    assert client.get("/api/v1/auth/me").json() == {"authenticated": False, "user": None}

    assert client.get("/api/v1/profiles").status_code == 401
    assert client.get("/api/v1/enrollments").status_code == 401
    assert client.get("/api/v1/audit").status_code == 401
    assert client.get("/api/v1/risk-events").status_code == 401
    assert client.get("/api/v1/user/overview").status_code == 401
    assert client.post("/api/v1/demo/clear-alerts").status_code == 401


def test_user_register_login_and_strict_isolation(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler, auto_login_admin=False)

    # Register Alice
    reg_a = client.post(
        "/api/v1/auth/register",
        json={"username": "alice", "display_name": "Alice Rao", "password": "secret123"},
    )
    assert reg_a.status_code == 200, reg_a.text
    alice = reg_a.json()["user"]
    assert alice["role"] == "user"
    alice_sub = alice["subject_ref"]

    # User cannot access admin endpoints or another subject_ref
    assert client.get("/api/v1/audit").status_code == 403
    assert client.get("/api/v1/risk-events").status_code == 403
    assert client.post("/api/v1/demo/clear-alerts").status_code == 403
    assert (
        client.post(
            "/api/v1/enrollments",
            data={
                "subject_ref": "SUB-OTHER",
                "consent_confirmed": "true",
                "consent_ref": "C-1",
                "consent_text": "Explicit speaker consent confirmed for testing.",
            },
            files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
        ).status_code
        == 403
    )

    # Alice enrolls & verifies her own voice
    _enroll_subject(client, alice_sub)
    sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": alice_sub, "action": "sim_replacement", "resource_ref": "SIM-ALICE-1"},
    )
    assert sess.status_code == 200
    alice_sid = sess.json()["session_id"]

    ver = client.post(
        f"/api/v1/verification-sessions/{alice_sid}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert ver.status_code == 200
    ver_json = ver.json()
    assert ver_json["outcome"] == "APPROVED"
    assert ver_json["status"] == "APPROVED"
    assert "decision_token" not in ver_json
    assert ver_json["voice_summary"]["speaker_decision"] == "match"

    overview = client.get("/api/v1/user/overview").json()
    assert overview["enrollment"]["subject_ref"] == alice_sub
    assert len(overview["verifications"]) == 1

    # Logout and register Bob: Bob cannot see Alice's data or touch Alice's session
    client.post("/api/v1/auth/logout")
    reg_b = client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "display_name": "Bob Nair", "password": "secret456"},
    )
    assert reg_b.status_code == 200
    assert client.get("/api/v1/enrollments").json()["enrollments"] == []
    assert client.get("/api/v1/user/overview").json()["verifications"] == []
    assert (
        client.post(
            f"/api/v1/verification-sessions/{alice_sid}/audio",
            data={"spoken_transcript": ""},
            files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
        ).status_code
        == 403
    )

    # Admin login sees everything
    from app.auth import get_admin_credentials

    admin_u, admin_p = get_admin_credentials()
    client.post("/api/v1/auth/logout")
    adm = client.post("/api/v1/auth/login", json={"username": admin_u, "password": admin_p})
    assert adm.status_code == 200
    assert adm.json()["user"]["role"] == "admin"
    assert len(client.get("/api/v1/enrollments").json()["enrollments"]) == 1
    assert len(client.get("/api/v1/audit").json()["audit_events"]) == 1


def test_speaker_no_match_directly_rejected_and_cancelled(make_test_client):
    no_match_verify = copy.deepcopy(VERIFY_FIXTURE)
    no_match_verify["result"]["decision"] = "no_match"
    no_match_verify["result"]["similarity"] = 0.08

    handler, _ = _default_fixture_handler(verify_override=no_match_verify)
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-801")

    sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-801",
            "action": "sim_replacement",
            "resource_ref": "SIM-MASKED-801",
        },
    ).json()

    res = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert res["outcome"] == "REJECTED"
    assert res["status"] == "CANCELLED"
    assert res["reason_codes"] == ["SPEAKER_NO_MATCH"]
    assert res["decision_token"] is None


def test_register_with_voice_saves_voiceprint_and_login_skips_reenroll(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)

    # Short audio at registration is rejected cleanly and does NOT create a half-registered account
    bad = client.post(
        "/api/v1/auth/register-with-voice",
        data={
            "username": "bad.wav.user",
            "display_name": "Bad Wav",
            "password": "password123",
            "consent_confirmed": "true",
            "consent_ref": "REG-CONSENT-0",
            "consent_text": "Explicit speaker consent confirmed for voiceprint registration.",
        },
        files={"file": ("short.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert bad.status_code == 400
    assert (
        client.post("/api/v1/auth/login", json={"username": "bad.wav.user", "password": "password123"}).status_code
        == 401
    )

    # Valid registration enrolls voiceprint atomically and attaches it to the user account
    reg = client.post(
        "/api/v1/auth/register-with-voice",
        data={
            "username": "priya.nair",
            "display_name": "Priya Nair",
            "password": "password123",
            "consent_confirmed": "true",
            "consent_ref": "REG-CONSENT-1",
            "consent_text": "Explicit speaker consent confirmed for voiceprint registration.",
        },
        files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert reg.status_code == 200
    reg_body = reg.json()
    assert reg_body["authenticated"] is True
    subject_ref = reg_body["user"]["subject_ref"]
    assert reg_body["enrollment"]["subject_ref"] == subject_ref

    ov1 = client.get("/api/v1/user/overview")
    assert ov1.status_code == 200
    assert ov1.json()["enrollment"]["subject_ref"] == subject_ref

    # Logout and sign back in: saved voiceprint stays on the account and user cannot re-enroll
    client.post("/api/v1/auth/logout")
    login = client.post("/api/v1/auth/login", json={"username": "priya.nair", "password": "password123"})
    assert login.status_code == 200
    assert login.json()["user"]["subject_ref"] == subject_ref

    ov2 = client.get("/api/v1/user/overview")
    assert ov2.status_code == 200
    assert ov2.json()["enrollment"]["subject_ref"] == subject_ref

    reenroll = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": subject_ref,
            "consent_confirmed": "true",
            "consent_ref": "RETRY",
            "consent_text": "Explicit speaker consent confirmed for re-enrollment.",
        },
        files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert reenroll.status_code == 409


def test_metrics_and_audit_dossier_export(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)

    # Empty state metrics check (and enrollment updates last_credits_balance)
    _enroll_subject(client, "SUB-901")
    empty_metrics = client.get("/api/v1/metrics")
    assert empty_metrics.status_code == 200
    em = empty_metrics.json()
    assert em["total_verifications"] == 0
    assert em["pass_count"] == 0
    assert em["blocked_count"] == 0
    assert em["p50_latency_ms"] is None
    assert em["last_credits_balance"] == ENROLL_FIXTURE["credits"]["balance"]

    # Run one APPROVED verification + consume token + one REJECTED verification
    sess_ok = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-901",
            "action": "sim_replacement",
            "resource_ref": "SIM-901",
        },
    ).json()
    ok_res = client.post(
        f"/api/v1/verification-sessions/{sess_ok['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert ok_res["outcome"] == "APPROVED"

    # Export before consume -> token_status ISSUED
    dossier_before = client.get(f"/api/v1/audit/{sess_ok['session_id']}/export")
    assert dossier_before.status_code == 200
    assert "attachment;" in dossier_before.headers.get("content-disposition", "")
    dbj = dossier_before.json()
    assert dbj["zero_raw_audio_stored"] is True
    assert dbj["policy_version"]
    assert dbj["token_status"] == "ISSUED"
    assert dbj["audio_sha256"] and len(dbj["audio_sha256"]) == 64
    assert dbj["request_id"] == VERIFY_FIXTURE["request_id"]
    assert dbj["operation_id"] == VERIFY_FIXTURE["operation_id"]
    assert dbj["speaker_decision"] == "match"
    assert dbj["deepfake_verdict"] == "real"
    assert len(dbj["segments"]) == len(VERIFY_FIXTURE["deepfake"]["segments"])

    consume = client.post(
        f"/api/v1/decisions/{sess_ok['session_id']}/consume",
        json={
            "decision_token": ok_res["decision_token"],
            "action": "sim_replacement",
            "resource_ref": "SIM-901",
        },
    )
    assert consume.status_code == 200

    # Export after consume -> token_status CONSUMED
    dossier_after = client.get(f"/api/v1/audit/{sess_ok['session_id']}/export").json()
    assert dossier_after["token_status"] == "CONSUMED"
    assert len(dossier_after["audit_events"]) == 2
    assert client.get("/api/v1/audit/missing-session/export").status_code == 404

    # Trigger a blocked verification (wrong phrase -> REJECTED) + a held verification (<4s -> REVIEW / HOLD)
    sess_bad = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": "SUB-901",
            "action": "mfa_reset",
            "resource_ref": "TKT-901",
        },
    ).json()
    client._asr_cfg["transcript"] = "wrong phrase for reject metric"
    bad_res = client.post(
        f"/api/v1/verification-sessions/{sess_bad['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("wrong.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    client._asr_cfg["transcript"] = None
    assert bad_res["outcome"] == "REJECTED"

    sess_hold = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": "SUB-901",
            "action": "mfa_reset",
            "resource_ref": "TKT-902",
        },
    ).json()
    hold_res = client.post(
        f"/api/v1/verification-sessions/{sess_hold['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("short.wav", make_pcm_wav_bytes(2.0), "audio/wav")},
    ).json()
    assert hold_res["outcome"] == "REVIEW / HOLD"

    metrics = client.get("/api/v1/metrics").json()
    assert metrics["total_verifications"] == 3
    assert metrics["pass_count"] == 1
    assert metrics["blocked_count"] == 1
    assert metrics["hold_count"] == 1
    assert metrics["p50_latency_ms"] == round(float(VERIFY_FIXTURE["latency_ms"]), 1)
    assert metrics["last_credits_balance"] == VERIFY_FIXTURE["credits"]["balance"]


def test_redteam_sandbox_all_five_scenarios_and_clear_alerts(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-902")

    expected = {
        "token_replay": (409, "REJECTED", "TOKEN_REPLAY_BLOCKED"),
        "action_substitution": (409, "REJECTED", "ACTION_SUBSTITUTION_BLOCKED"),
        "expired_challenge": (410, "REJECTED", "CHALLENGE_EXPIRED"),
        "provider_outage_fail_closed": (503, "REVIEW / HOLD", "PROVIDER_UNAVAILABLE"),
        "cross_domain_sim_alert": (200, "STEP-UP", "CROSS_DOMAIN_ALERT"),
    }

    for scenario, (exp_http, exp_outcome, exp_reason) in expected.items():
        resp = client.post(f"/api/v1/sandbox/run/{scenario}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["sandbox_test"] is True
        assert data["scenario"] == scenario
        assert data["http_status"] == exp_http
        assert data["outcome"] == exp_outcome
        assert "SANDBOX_TEST" in data["reason_codes"]
        assert exp_reason in data["reason_codes"]
        assert data["audit_event"]["session_id"] == data["session_id"]

        # Confirm audit dossier export works cleanly on sandbox sessions
        exp = client.get(f"/api/v1/audit/{data['session_id']}/export")
        assert exp.status_code == 200
        assert exp.json()["sandbox_test"] is True

    assert client.post("/api/v1/sandbox/run/invalid_scenario").status_code == 404

    # Cross-domain sandbox alert shows in metrics and clears cleanly without touching enrollments
    m_before = client.get("/api/v1/metrics").json()
    assert m_before["active_risk_events_count"] >= 1
    cleared = client.post("/api/v1/demo/clear-alerts")
    assert cleared.status_code == 200
    assert cleared.json()["cleared_count"] >= 1
    m_after = client.get("/api/v1/metrics").json()
    assert m_after["active_risk_events_count"] == 0
    assert len(client.get("/api/v1/enrollments").json()["enrollments"]) == 1


def test_audio_replay_detected_even_when_transcript_matches(make_test_client):
    handler, captured_headers = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-RPL-1")

    reused_wav = make_pcm_wav_bytes(6.0)
    sess1 = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-RPL-1",
            "action": "sim_replacement",
            "resource_ref": "SIM-RPL-1",
        },
    ).json()
    res1 = client.post(
        f"/api/v1/verification-sessions/{sess1['session_id']}/audio",
        data={"spoken_transcript": sess1["challenge_text"]},
        files={"file": ("verify.wav", reused_wav, "audio/wav")},
    ).json()
    assert res1["outcome"] == "APPROVED"
    assert len(captured_headers) == 3

    # Second session submits the IDENTICAL WAV bytes even with matching transcript -> blocked!
    sess2 = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-RPL-1",
            "action": "sim_replacement",
            "resource_ref": "SIM-RPL-2",
        },
    ).json()
    res2 = client.post(
        f"/api/v1/verification-sessions/{sess2['session_id']}/audio",
        data={"spoken_transcript": sess2["challenge_text"]},
        files={"file": ("replay.wav", reused_wav, "audio/wav")},
    ).json()
    assert res2["outcome"] == "REJECTED"
    assert res2["status"] == "CANCELLED"
    assert res2["evidence_state"] == "REPLAY_DETECTED"
    assert res2["reason_codes"] == ["AUDIO_REPLAY_DETECTED"]
    assert res2["decision_token"] is None
    assert len(captured_headers) == 3  # DetectifAI was NOT called again
    assert ServerASRAdapter(provider="disabled", api_key="").transcribe_wav(reused_wav).available is False


def test_hmac_signature_verification_on_sim_swap_alert(make_test_client, monkeypatch):
    monkeypatch.setenv("WEBHOOK_SIGNING_SECRET", "finals-hmac-secret-2026")
    fake_verify = copy.deepcopy(VERIFY_FIXTURE)
    fake_verify["deepfake"]["verdict"] = "fake"
    fake_verify["deepfake"]["is_synthetic"] = True

    handler, _ = _default_fixture_handler(verify_override=fake_verify)
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-HMAC-1")

    sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-HMAC-1",
            "action": "sim_replacement",
            "resource_ref": "SIM-HMAC-1",
        },
    ).json()
    res = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("fake.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    risk = res["emitted_risk_event"]
    assert risk["event_type"] == "SIM_SWAP_ATTEMPT_BLOCKED"
    assert len(risk["hmac_signature"]) == 64
    assert verify_risk_event_signature(risk["signed_payload"], risk["hmac_signature"]) is True

    # Tampered payload must fail HMAC verification
    tampered = dict(risk["signed_payload"], subject_ref="SUB-ATTACKER")
    assert verify_risk_event_signature(tampered, risk["hmac_signature"]) is False

    stored = client.get("/api/v1/risk-events").json()["risk_events"]
    assert stored[0]["hmac_signature"] == risk["hmac_signature"]


def test_offline_sim_swap_callback_and_storage_backend(make_test_client, monkeypatch):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-OFF-1")

    health = client.get("/api/v1/health").json()
    assert health["storage_backend"] == "sqlite_ephemeral"

    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost:5432/voicesuraksha")
    assert get_database_url() == "postgresql://user:pass@localhost:5432/voicesuraksha"
    assert get_storage_backend() == "postgresql"
    monkeypatch.delenv("DATABASE_URL")

    cb = client.post(
        "/api/v1/telecom/offline-swap-callback",
        json={
            "subject_ref": "SUB-OFF-1",
            "store_id": "STORE-BLR-09",
            "sim_iccid_masked": "ICCID-8821",
        },
    )
    assert cb.status_code == 200, cb.text
    body = cb.json()
    assert body["callback_required"] is True
    assert body["trigger"] == "OFFLINE_RETAIL_SIM_SWAP"
    assert body["sector"] == "telecom"
    assert body["action"] == "sim_replacement"
    assert body["resource_ref"] == "OFFLINE-STORE-BLR-09-ICCID-8821"
    assert body["status"] == "PENDING"

    # Voice verification completes the retail callback session cleanly
    ver = client.post(
        f"/api/v1/verification-sessions/{body['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("cb.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert ver["outcome"] == "APPROVED"


def test_silent_and_clipped_audio_quality_low_skips_detectif_provider(make_test_client):
    handler, captured_headers = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-AQ-1")
    assert len(captured_headers) == 2  # deepfake gate + enroll

    # Silent audio (< -45 dBFS) -> REVIEW / HOLD + AUDIO_QUALITY_LOW without calling DetectifAI
    sess_sil = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-AQ-1",
            "action": "sim_replacement",
            "resource_ref": "SIM-AQ-SIL",
        },
    ).json()
    sil_res = client.post(
        f"/api/v1/verification-sessions/{sess_sil['session_id']}/audio",
        data={"spoken_transcript": sess_sil["challenge_text"]},
        files={"file": ("silent.wav", make_pcm_wav_bytes(6.0, silent=True), "audio/wav")},
    ).json()
    assert sil_res["outcome"] == "REVIEW / HOLD"
    assert sil_res["status"] == "HOLD"
    assert sil_res["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert sil_res["reason_codes"] == ["AUDIO_QUALITY_LOW"]
    assert sil_res["audio_quality"]["rms_dbfs"] < -45.0
    assert len(captured_headers) == 2

    # Heavily clipped audio (clipping_ratio > 0.15) -> also routes to REVIEW / HOLD without calling DetectifAI
    sess_clip = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": "SUB-AQ-1",
            "action": "mfa_reset",
            "resource_ref": "TKT-AQ-CLIP",
        },
    ).json()
    clip_res = client.post(
        f"/api/v1/verification-sessions/{sess_clip['session_id']}/audio",
        data={"spoken_transcript": sess_clip["challenge_text"]},
        files={"file": ("clipped.wav", make_pcm_wav_bytes(6.0, clipped=True), "audio/wav")},
    ).json()
    assert clip_res["outcome"] == "REVIEW / HOLD"
    assert clip_res["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert clip_res["reason_codes"] == ["AUDIO_QUALITY_LOW"]
    assert clip_res["audio_quality"]["clipping_ratio"] > 0.15
    assert len(captured_headers) == 2


def test_hosted_asr_adapter_providers_and_fail_closed(make_pcm_fixture=None):
    wav = make_pcm_wav_bytes(5.0)

    # Missing key fails closed
    disabled = ServerASRAdapter(provider="sarvam", api_key="")
    res_dis = disabled.transcribe_wav(wav)
    assert res_dis.available is False
    assert res_dis.error == "asr_not_configured"

    # Sarvam REST contract
    sarvam_transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"transcript": "Confirm sim replacement code blew river forty two"})
        if req.headers.get("api-subscription-key") == "srv-key"
        else httpx.Response(401)
    )
    sarvam = ServerASRAdapter(provider="sarvam", api_key="srv-key", transport=sarvam_transport)
    res_srv = sarvam.transcribe_wav(wav)
    assert res_srv.available is True
    assert res_srv.transcript == "Confirm sim replacement code blew river forty two"

    # Deepgram REST contract
    dg_transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"results": {"channels": [{"alternatives": [{"transcript": "confirm mfa reset code"}]}]}},
        )
    )
    dg = ServerASRAdapter(provider="deepgram", api_key="dg-key", transport=dg_transport)
    assert dg.transcribe_wav(wav).transcript == "confirm mfa reset code"

    # Upstream timeout fails closed
    def _timeout(_req):
        raise httpx.ReadTimeout("ASR timeout")

    timed_out = ServerASRAdapter(provider="sarvam", api_key="srv-key", transport=httpx.MockTransport(_timeout))
    assert timed_out.transcribe_wav(wav).available is False


def test_normalize_transcript_tolerance_keeps_exact_code_tokens():
    challenge = "Confirm sim replacement code blue river 42."
    assert verify_phrase_match(challenge, "Confirm sim replacement code blew river four to") is True
    assert verify_phrase_match(challenge, "confirm sim replacement code blue river forty two") is True
    assert verify_phrase_match(challenge, "confirm sim replacement code green river 42") is False
    assert verify_phrase_match(challenge, "confirm sim replacement code blue river 43") is False
    assert normalize_transcript("Confirm esim transfer code lite tiger 09") == "confirm esim transfer code light tiger 0 9"


def test_live_asr_unavailable_holds_while_client_transcript_ignored(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler, demo_mode=False)
    _enroll_subject(client, "SUB-ASR-1")

    sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-ASR-1",
            "action": "sim_replacement",
            "resource_ref": "SIM-ASR-1",
        },
    ).json()

    # In LIVE mode with ASR unavailable, critical action MUST hold (ASR_UNAVAILABLE) even if client sends phrase
    client._asr_cfg["available"] = False
    res = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": sess["challenge_text"]},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert res["outcome"] == "REVIEW / HOLD"
    assert res["status"] == "HOLD"
    assert res["reason_codes"] == ["ASR_UNAVAILABLE"]
    assert res["decision_token"] is None


def test_server_context_ignores_tampered_after_hours_privileged_and_amount(make_test_client, monkeypatch):
    from datetime import datetime, timezone

    # Verify IST after-hours helper directly (23:30 IST -> True, 14:00 Wed IST -> False, Sat 14:00 IST -> True)
    assert _is_ist_after_hours(datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)) is True  # 23:30 IST
    assert _is_ist_after_hours(datetime(2026, 11, 25, 8, 30, tzinfo=timezone.utc)) is False  # 14:00 Wed IST
    assert _is_ist_after_hours(datetime(2026, 11, 28, 8, 30, tzinfo=timezone.utc)) is True  # 14:00 Sat IST

    handler, _ = _default_fixture_handler()
    client = make_test_client(handler, auto_login_admin=False)

    reg = client.post(
        "/api/v1/auth/register-with-voice",
        data={
            "username": "context.user",
            "display_name": "Context User",
            "password": "password123",
            "consent_confirmed": "true",
            "consent_ref": "CTX-1",
            "consent_text": "Explicit speaker consent confirmed for voiceprint registration.",
        },
        files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert reg.status_code == 200
    sub = reg.json()["user"]["subject_ref"]

    # Non-admin cannot set privileged_role via admin endpoint
    assert client.patch(f"/api/v1/admin/subjects/{sub}/privilege", json={"privileged_role": True}).status_code == 403

    # Caller sends after_hours=False while server clock is IST 23:30 -> server enforces AFTER_HOURS_COOLING_HOLD
    monkeypatch.setattr("app.main._is_ist_after_hours", lambda _now=None: True)
    tel_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": sub,
            "action": "sim_replacement",
            "resource_ref": "SIM-CTX-1",
            "after_hours": False,
        },
    ).json()
    tel_res = client.post(
        f"/api/v1/verification-sessions/{tel_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("v1.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert tel_res["outcome"] == "STEP-UP"
    assert "AFTER_HOURS_COOLING_HOLD" in tel_res["reason_codes"]

    # Caller claims privileged_role=True on non-privileged account -> ignored (helpdesk approves)
    monkeypatch.setattr("app.main._is_ist_after_hours", lambda _now=None: False)
    hd_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": sub,
            "action": "mfa_reset",
            "resource_ref": "TKT-CTX-1",
            "privileged_role": True,
        },
    ).json()
    hd_res = client.post(
        f"/api/v1/verification-sessions/{hd_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("v2.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert hd_res["outcome"] == "APPROVED"

    # Normal user sending amount_inr on banking card_replacement -> ignored -> STEP-UP AMOUNT_REQUIRED
    bk_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "banking",
            "subject_ref": sub,
            "action": "card_replacement",
            "resource_ref": "CARD-CTX-1",
            "amount_inr": 1000,
        },
    ).json()
    bk_res = client.post(
        f"/api/v1/verification-sessions/{bk_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("v3.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert bk_res["outcome"] == "STEP-UP"
    assert "AMOUNT_REQUIRED" in bk_res["reason_codes"]


def test_live_mode_refuses_startup_on_default_or_short_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("AUTH_SECRET", "strong-auth-secret-over-16-chars")
    monkeypatch.setenv("WEBHOOK_SIGNING_SECRET", "strong-webhook-secret-over-16")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin123")
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        validate_live_secrets()

    monkeypatch.setenv("ADMIN_PASSWORD", "strong-admin-password-over-16")
    monkeypatch.setenv("AUTH_SECRET", "short")
    with pytest.raises(RuntimeError, match="AUTH_SECRET"):
        validate_live_secrets()

    monkeypatch.setenv("DEMO_MODE", "true")
    validate_live_secrets()  # Permitted in DEMO_MODE

    # Verify schema_version migration tracking works cleanly on SQLite
    db_file = tmp_path / "migrated.db"
    eng = create_db_engine(f"sqlite:///{db_file}")
    init_db(eng)
    with sessionmaker(bind=eng)() as db:
        row = db.query(SchemaVersion).filter_by(version=CURRENT_SCHEMA_VERSION).one()
        assert row.version == CURRENT_SCHEMA_VERSION
    assert is_storage_durable(f"sqlite:///{db_file}") is False
    eng.dispose()


def test_step_up_completion_single_token_double_complete_409_and_banking_threshold(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler, auto_login_admin=False)

    # Normal user registers with voiceprint
    reg = client.post(
        "/api/v1/auth/register-with-voice",
        data={
            "username": "bank.stepup.user",
            "display_name": "Bank StepUp User",
            "password": "password123",
            "consent_confirmed": "true",
            "consent_ref": "BANK-REG-1",
            "consent_text": "Explicit speaker consent confirmed for voiceprint registration.",
        },
        files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert reg.status_code == 200
    sub = reg.json()["user"]["subject_ref"]

    # User creates Banking card_replacement without admin amount -> STEP-UP (AMOUNT_REQUIRED)
    bk_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "banking",
            "subject_ref": sub,
            "action": "card_replacement",
            "resource_ref": "CARD-SU-1",
        },
    ).json()
    bk_res = client.post(
        f"/api/v1/verification-sessions/{bk_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("v1.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert bk_res["outcome"] == "STEP-UP"
    assert "AMOUNT_REQUIRED" in bk_res["reason_codes"]
    assert "decision_token" not in bk_res

    # Non-admin user cannot supply amount_inr during step-up/complete
    user_attempt = client.post(
        f"/api/v1/decisions/{bk_sess['session_id']}/step-up/complete",
        json={"method": "trusted_app_push", "amount_inr": 1000},
    )
    assert user_attempt.status_code == 403

    # Admin logs in and supplies amount_inr=250000 via trusted_app_push -> re-evaluates high-value threshold (>= 100k)
    from app.auth import get_admin_credentials

    admin_u, admin_p = get_admin_credentials()
    client.post("/api/v1/auth/login", json={"username": admin_u, "password": admin_p})
    high_push = client.post(
        f"/api/v1/decisions/{bk_sess['session_id']}/step-up/complete",
        json={"method": "trusted_app_push", "amount_inr": 250000},
    )
    assert high_push.status_code == 200
    hp = high_push.json()
    assert hp["outcome"] == "STEP-UP"
    assert "HIGH_VALUE_STEP_UP" in hp["reason_codes"]
    assert hp["threshold_evaluation"]["high_value"] is True
    assert hp["decision_token"] is None

    # Admin completes via manager_callback (simulated) -> APPROVED + exactly ONE token issued
    done = client.post(
        f"/api/v1/decisions/{bk_sess['session_id']}/step-up/complete",
        json={"method": "manager_callback", "amount_inr": 250000},
    )
    assert done.status_code == 200
    done_body = done.json()
    assert done_body["outcome"] == "APPROVED"
    assert done_body["simulated"] is True
    assert "STEP_UP_COMPLETED" in done_body["reason_codes"]
    assert "HIGH_VALUE_THRESHOLD_VERIFIED" in done_body["reason_codes"]
    token = done_body["decision_token"]
    assert token

    # Double completion MUST fail with 409
    double_done = client.post(
        f"/api/v1/decisions/{bk_sess['session_id']}/step-up/complete",
        json={"method": "in_store_kyc", "amount_inr": 250000},
    )
    assert double_done.status_code == 409

    # Audit log contains STEP_UP_COMPLETED and single-use token consumes once (409 on replay)
    audits = client.get("/api/v1/audit").json()["audit_events"]
    assert any("STEP_UP_COMPLETED" in a["reason_codes"] for a in audits if a["session_id"] == bk_sess["session_id"])
    assert (
        client.post(
            f"/api/v1/decisions/{bk_sess['session_id']}/consume",
            json={"decision_token": token, "action": "card_replacement", "resource_ref": "CARD-SU-1"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/api/v1/decisions/{bk_sess['session_id']}/consume",
            json={"decision_token": token, "action": "card_replacement", "resource_ref": "CARD-SU-1"},
        ).status_code
        == 409
    )


def test_admin_reviewer_queue_approve_and_reject_with_reason(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-REV-1")

    # 1. Put a session into HOLD via silent audio
    hold_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-REV-1",
            "action": "sim_replacement",
            "resource_ref": "SIM-REV-1",
        },
    ).json()
    hold_res = client.post(
        f"/api/v1/verification-sessions/{hold_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("silent.wav", make_pcm_wav_bytes(6.0, silent=True), "audio/wav")},
    ).json()
    assert hold_res["outcome"] == "REVIEW / HOLD"

    # 2. Put a second session into STEP-UP via privileged role
    client.patch("/api/v1/admin/subjects/SUB-REV-1/privilege", json={"privileged_role": True})
    su_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": "SUB-REV-1",
            "action": "mfa_reset",
            "resource_ref": "TKT-REV-2",
        },
    ).json()
    su_res = client.post(
        f"/api/v1/verification-sessions/{su_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ok.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert su_res["outcome"] == "STEP-UP"

    queue = client.get("/api/v1/admin/review-queue").json()["review_queue"]
    queued_ids = {q["session_id"] for q in queue}
    assert {hold_sess["session_id"], su_sess["session_id"]}.issubset(queued_ids)

    # Empty/whitespace reason is rejected
    bad_rev = client.post(
        f"/api/v1/decisions/{hold_sess['session_id']}/review",
        json={"decision": "APPROVED", "reason": "  "},
    )
    assert bad_rev.status_code in {400, 422}

    # Approve HOLD session -> issues token + logs REVIEWER_APPROVED + timestamp
    app_rev = client.post(
        f"/api/v1/decisions/{hold_sess['session_id']}/review",
        json={"decision": "APPROVED", "reason": "Verified customer over callback"},
    )
    assert app_rev.status_code == 200
    app_body = app_rev.json()
    assert app_body["outcome"] == "APPROVED"
    assert app_body["decision_token"]
    assert app_body["reviewed_at"]
    assert (
        client.post(
            f"/api/v1/decisions/{hold_sess['session_id']}/review",
            json={"decision": "REJECTED", "reason": "Duplicate review"},
        ).status_code
        == 409
    )

    # Reject STEP-UP session -> CANCELLED, no token
    rej_rev = client.post(
        f"/api/v1/decisions/{su_sess['session_id']}/review",
        json={"decision": "REJECTED", "reason": "Manager denied privileged reset"},
    )
    assert rej_rev.status_code == 200
    rej_body = rej_rev.json()
    assert rej_body["outcome"] == "REJECTED"
    assert rej_body["status"] == "CANCELLED"
    assert rej_body["decision_token"] is None


def test_cross_domain_alert_ttl_and_resolve_alert(make_test_client, db_session_factory):
    fake_verify = copy.deepcopy(VERIFY_FIXTURE)
    fake_verify["deepfake"]["verdict"] = "fake"
    fake_verify["deepfake"]["is_synthetic"] = True
    mode = {"fake": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            return httpx.Response(200, json=copy.deepcopy(fake_verify if mode["fake"] else VERIFY_FIXTURE))
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-ALERT-1")

    # Emit SIM_SWAP_ATTEMPT_BLOCKED
    tel = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-ALERT-1", "action": "sim_replacement", "resource_ref": "SIM-A1"},
    ).json()
    tel_res = client.post(
        f"/api/v1/verification-sessions/{tel['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("clone.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    alert_id = tel_res["emitted_risk_event"]["id"]

    # emergency_sim_block remains low-friction APPROVED even while alert is ACTIVE
    mode["fake"] = False
    emerg = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-ALERT-1", "action": "emergency_sim_block", "resource_ref": "SIM-E1"},
    ).json()
    emerg_res = client.post(
        f"/api/v1/verification-sessions/{emerg['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("emerg.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert emerg_res["outcome"] == "APPROVED"

    # Admin resolves the alert -> subsequent banking request passes APPROVED cleanly
    resolved = client.post(f"/api/v1/risk-events/{alert_id}/resolve", json={"resolution_note": "Owner confirmed"})
    assert resolved.status_code == 200
    assert resolved.json()["risk_state"] == "RESOLVED"
    assert client.post(f"/api/v1/risk-events/{alert_id}/resolve").status_code == 409
    assert client.post("/api/v1/risk-events/99999/resolve").status_code == 404

    bank_ok = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "banking",
            "subject_ref": "SUB-ALERT-1",
            "action": "card_replacement",
            "resource_ref": "CARD-A1",
            "amount_inr": 50000,
        },
    ).json()
    assert (
        client.post(
            f"/api/v1/verification-sessions/{bank_ok['session_id']}/audio",
            data={"spoken_transcript": ""},
            files={"file": ("ok2.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
        ).json()["outcome"]
        == "APPROVED"
    )

    # Alert TTL expiry also prevents blocking once expires_at <= now
    from app.models import RiskEvent

    mode["fake"] = True
    tel2 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-ALERT-1", "action": "sim_replacement", "resource_ref": "SIM-A2"},
    ).json()
    tel2_res = client.post(
        f"/api/v1/verification-sessions/{tel2['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("clone2.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    alert2_id = tel2_res["emitted_risk_event"]["id"]

    with db_session_factory() as db:
        row = db.query(RiskEvent).filter_by(id=alert2_id).one()
        row.expires_at = utcnow() - timedelta(minutes=1)
        db.commit()

    risks_list = client.get("/api/v1/risk-events").json()["risk_events"]
    assert any(r["id"] == alert2_id and r["expired"] is True for r in risks_list)

    mode["fake"] = False
    bank_ttl = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "banking",
            "subject_ref": "SUB-ALERT-1",
            "action": "card_replacement",
            "resource_ref": "CARD-A2",
            "amount_inr": 50000,
        },
    ).json()
    assert (
        client.post(
            f"/api/v1/verification-sessions/{bank_ttl['session_id']}/audio",
            data={"spoken_transcript": ""},
            files={"file": ("ok3.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
        ).json()["outcome"]
        == "APPROVED"
    )


def test_every_hold_cause_returns_fallbacks_and_asr_down_still_rejects_fake(make_test_client):
    fake_verify = copy.deepcopy(VERIFY_FIXTURE)
    fake_verify["deepfake"]["verdict"] = "fake"
    fake_verify["deepfake"]["is_synthetic"] = True
    mode = {"fake": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            return httpx.Response(200, json=copy.deepcopy(fake_verify if mode["fake"] else VERIFY_FIXTURE))
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-HOLD-ALL")

    # ASR_UNAVAILABLE with passing voice -> REVIEW / HOLD + 3 fallback options
    client._asr_cfg["available"] = False
    s_asr = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-HOLD-ALL", "action": "sim_replacement", "resource_ref": "SIM-H1"},
    ).json()
    r_asr = client.post(
        f"/api/v1/verification-sessions/{s_asr['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("v_ok.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r_asr["outcome"] == "REVIEW / HOLD"
    assert r_asr["reason_codes"] == ["ASR_UNAVAILABLE"]
    assert {opt["method"] for opt in r_asr["fallback_options"]} == {"trusted_app_push", "in_store_kyc", "manager_callback"}

    # Even with ASR_UNAVAILABLE, fake audio MUST still REJECTED immediately
    mode["fake"] = True
    s_fake = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-HOLD-ALL", "action": "sim_replacement", "resource_ref": "SIM-H2"},
    ).json()
    r_fake = client.post(
        f"/api/v1/verification-sessions/{s_fake['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("v_fake.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r_fake["outcome"] == "REJECTED"
    assert "DEEPFAKE_SIGNAL" in r_fake["reason_codes"]
    assert r_fake["fallback_options"] == []

    # Silent and clipped HOLD responses also carry all 3 non-voice fallbacks
    client._asr_cfg["available"] = True
    mode["fake"] = False
    for bad_wav in (make_pcm_wav_bytes(6.0, silent=True), make_pcm_wav_bytes(6.0, clipped=True)):
        s_q = client.post(
            "/api/v1/verification-sessions",
            json={"sector": "helpdesk", "subject_ref": "SUB-HOLD-ALL", "action": "mfa_reset", "resource_ref": "TKT-H1"},
        ).json()
        r_q = client.post(
            f"/api/v1/verification-sessions/{s_q['session_id']}/audio",
            data={"spoken_transcript": ""},
            files={"file": ("bad.wav", bad_wav, "audio/wav")},
        ).json()
        assert r_q["outcome"] == "REVIEW / HOLD"
        assert r_q["reason_codes"] == ["AUDIO_QUALITY_LOW"]
        assert len(r_q["fallback_options"]) == 3


def _transform_wav_gain_and_trim(wav_bytes: bytes, *, gain: float = 0.6, trim_ms: int = 200) -> bytes:
    """Return a gain-scaled and trimmed PCM WAV copy (breaks exact SHA-256 while preserving spectral identity)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    trim_frames = int((trim_ms / 1000.0) * sr) * ch
    if trim_frames * 2 < samples.size:
        samples = samples[trim_frames:-trim_frames]
    scaled = np.clip(samples * gain, -32767.0, 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as out_wf:
        out_wf.setnchannels(ch)
        out_wf.setsampwidth(sw)
        out_wf.setframerate(sr)
        out_wf.writeframes(scaled.tobytes())
    return out.getvalue()


def test_perceptual_audio_replay_detects_gain_scaled_and_trimmed_copy(make_test_client):
    clip_a = make_pcm_wav_bytes(6.0)
    clip_a_modified = _transform_wav_gain_and_trim(clip_a, gain=0.6, trim_ms=200)
    clip_b = make_pcm_wav_bytes(6.0)

    h_a = compute_perceptual_hash(clip_a)
    h_mod = compute_perceptual_hash(clip_a_modified)
    h_b = compute_perceptual_hash(clip_b)
    assert h_a is not None and h_mod is not None and h_b is not None
    assert hashlib.sha256(clip_a).hexdigest() != hashlib.sha256(clip_a_modified).hexdigest()
    assert hamming_distance_hex(h_a, h_mod) <= MAX_PERCEPTUAL_HAMMING_BITS
    assert hamming_distance_hex(h_a, h_b) > MAX_PERCEPTUAL_HAMMING_BITS

    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-PHASH-1")

    s1 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-PHASH-1", "action": "sim_replacement", "resource_ref": "SIM-P1"},
    ).json()
    r1 = client.post(
        f"/api/v1/verification-sessions/{s1['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("orig.wav", clip_a, "audio/wav")},
    ).json()
    assert r1["outcome"] == "APPROVED"

    s2 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-PHASH-1", "action": "sim_replacement", "resource_ref": "SIM-P2"},
    ).json()
    r2 = client.post(
        f"/api/v1/verification-sessions/{s2['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("reencoded.wav", clip_a_modified, "audio/wav")},
    ).json()
    assert r2["outcome"] == "REJECTED"
    assert r2["reason_codes"] == ["AUDIO_REPLAY_DETECTED"]
    assert r2["match_type"] == "perceptual"
    assert r2["decision_token"] is None


def test_multi_segment_verification_and_disagreement_or_segment_fake(make_test_client):
    verify_calls: list[int] = []
    mode = {"case": "one_fake"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            idx = len(verify_calls)
            verify_calls.append(idx)
            payload = copy.deepcopy(VERIFY_FIXTURE)
            if mode["case"] == "one_fake" and idx % 2 == 1:
                payload["deepfake"]["verdict"] = "fake"
                payload["deepfake"]["is_synthetic"] = True
            elif mode["case"] == "disagree" and idx % 2 == 1:
                payload["result"]["decision"] = "no_match"
                payload["result"]["similarity"] = 0.25
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-MSEC-1")

    # 9s WAV splits into 2 overlapping >=4s slices -> 1 fake slice rejects the whole request
    s1 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "helpdesk", "subject_ref": "SUB-MSEC-1", "action": "mfa_reset", "resource_ref": "TKT-MS1"},
    ).json()
    r1 = client.post(
        f"/api/v1/verification-sessions/{s1['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("long_fake.wav", make_pcm_wav_bytes(9.0), "audio/wav")},
    ).json()
    assert len(verify_calls) == 2
    assert r1["outcome"] == "REJECTED"
    assert "DEEPFAKE_SIGNAL" in r1["reason_codes"]

    # Split speaker disagreement (1 match, 1 no_match, both real) -> STEP-UP with second-challenge followup
    verify_calls.clear()
    mode["case"] = "disagree"
    s2 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "helpdesk", "subject_ref": "SUB-MSEC-1", "action": "mfa_reset", "resource_ref": "TKT-MS2"},
    ).json()
    r2 = client.post(
        f"/api/v1/verification-sessions/{s2['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("long_split.wav", make_pcm_wav_bytes(9.0), "audio/wav")},
    ).json()
    assert len(verify_calls) == 2
    assert r2["outcome"] == "STEP-UP"
    assert r2["reason_codes"] == ["BORDERLINE_SECOND_CHALLENGE"]
    assert r2["followup_session"] is not None
    assert r2["followup_session"]["parent_session_id"] == s2["session_id"]
    assert r2["followup_session"]["challenge_attempt"] == 2


def test_borderline_second_challenge_lineage_and_repeated_hold(make_test_client):
    mode = {"borderline": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            payload = copy.deepcopy(VERIFY_FIXTURE)
            if mode["borderline"]:
                payload["result"]["decision"] = "no_match"
                payload["result"]["similarity"] = 0.49
                payload["result"]["threshold"] = 0.53  # within BORDERLINE_SIMILARITY_MARGIN (0.08)
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-BORD-1")

    # Attempt 1 borderline -> STEP-UP + followup_session (attempt 2); Attempt 2 also borderline -> REVIEW / HOLD
    s1 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-BORD-1", "action": "sim_replacement", "resource_ref": "SIM-B1"},
    ).json()
    r1 = client.post(
        f"/api/v1/verification-sessions/{s1['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("b1.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r1["outcome"] == "STEP-UP"
    assert r1["reason_codes"] == ["BORDERLINE_SECOND_CHALLENGE"]
    followup_id = r1["followup_session"]["session_id"]

    r2 = client.post(
        f"/api/v1/verification-sessions/{followup_id}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("b2.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r2["outcome"] == "REVIEW / HOLD"
    assert r2["reason_codes"] == ["BORDERLINE_VOICE_HOLD"]
    assert r2["followup_session"] is None
    assert len(r2["fallback_options"]) == 3

    # Fresh attempt 1 borderline -> attempt 2 clean match -> APPROVED with token
    s3 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-BORD-1", "action": "sim_replacement", "resource_ref": "SIM-B2"},
    ).json()
    r3 = client.post(
        f"/api/v1/verification-sessions/{s3['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("b3.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r3["outcome"] == "STEP-UP"
    followup_id2 = r3["followup_session"]["session_id"]

    mode["borderline"] = False
    r4 = client.post(
        f"/api/v1/verification-sessions/{followup_id2}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("b4.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r4["outcome"] == "APPROVED"
    assert r4["decision_token"] is not None


def test_rate_limiting_and_login_lockout_429_with_audit(make_test_client, db_session_factory, monkeypatch):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)

    # 1) Verification session rate limit returns 429 + Retry-After + RATE_LIMITED audit row
    _enroll_subject(client, "SUB-RL-1")
    monkeypatch.setenv("RL_SESSIONS_PER_MIN", "2")
    for idx in range(2):
        ok = client.post(
            "/api/v1/verification-sessions",
            json={"sector": "helpdesk", "subject_ref": "SUB-RL-1", "action": "mfa_reset", "resource_ref": f"T-{idx}"},
        )
        assert ok.status_code == 200
    rl_resp = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "helpdesk", "subject_ref": "SUB-RL-1", "action": "mfa_reset", "resource_ref": "T-OVER"},
    )
    assert rl_resp.status_code == 429
    assert int(rl_resp.headers.get("Retry-After", "0")) >= 1
    assert any("RATE_LIMITED" in a["reason_codes"] for a in client.get("/api/v1/audit").json()["audit_events"])

    # 2) Logout and trigger 5 failed logins -> 6th returns HTTP 429 + Retry-After, and SQLite lock survives in-memory reset
    client.post("/api/v1/auth/logout")
    for _ in range(5):
        resp = client.post("/api/v1/auth/login", json={"username": "lockout.target", "password": "wrong-password"})
        assert resp.status_code == 401

    locked = client.post("/api/v1/auth/login", json={"username": "lockout.target", "password": "wrong-password"})
    assert locked.status_code == 429
    assert int(locked.headers.get("Retry-After", "0")) >= 1

    gateway_rate_limiter.reset()
    locked_after_restart = client.post(
        "/api/v1/auth/login",
        json={"username": "lockout.target", "password": "wrong-password"},
    )
    assert locked_after_restart.status_code == 429

    with db_session_factory() as db:
        audits = db.query(AuditEvent).all()
        assert any("RATE_LIMITED" in json.loads(a.reason_codes_json) and a.action == "login" for a in audits)


def test_enrollment_consent_text_verified_reenroll_and_24h_cooldown(make_test_client, monkeypatch):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler, auto_login_admin=False)

    # Registration requires consent_text (min 10 chars)
    no_consent_text = client.post(
        "/api/v1/auth/register-with-voice",
        data={
            "username": "consent.missing",
            "display_name": "No Consent",
            "password": "password123",
            "consent_confirmed": "true",
            "consent_ref": "C-BAD",
            "consent_text": "short",
        },
        files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert no_consent_text.status_code in (400, 422)

    reg = client.post(
        "/api/v1/auth/register-with-voice",
        data={
            "username": "consent.user",
            "display_name": "Consent User",
            "password": "password123",
            "consent_confirmed": "true",
            "consent_ref": "C-OK-1",
            "consent_text": "I consent to biometric voiceprint storage for identity verification.",
        },
        files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert reg.status_code == 200
    sub = reg.json()["user"]["subject_ref"]

    # Complete a verified APPROVED session for this user
    ver_sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "helpdesk", "subject_ref": sub, "action": "mfa_reset", "resource_ref": "TKT-RE-1"},
    ).json()
    ver_res = client.post(
        f"/api/v1/verification-sessions/{ver_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ok1.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert ver_res["outcome"] == "APPROVED"

    # Re-enroll without verified_session_id fails with 409; with verified_session_id succeeds and starts 24h cooldown
    blocked_reenroll = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": sub,
            "consent_confirmed": "true",
            "consent_ref": "C-RE-1",
            "consent_text": "I explicitly authorize updating my biometric voiceprint.",
        },
        files={"file": ("re.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert blocked_reenroll.status_code == 409

    ok_reenroll = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": sub,
            "consent_confirmed": "true",
            "consent_ref": "C-RE-2",
            "consent_text": "I explicitly authorize updating my biometric voiceprint.",
            "verified_session_id": ver_sess["session_id"],
        },
        files={"file": ("re2.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert ok_reenroll.status_code == 200
    assert ok_reenroll.json()["enrollment"]["re_enrolled_at"] is not None

    # Critical telecom sim_replacement during 24h re-enrollment cooldown -> STEP-UP (NEW_ENROLLMENT_COOLDOWN)
    cool_sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": sub, "action": "sim_replacement", "resource_ref": "SIM-COOL-1"},
    ).json()
    cool_res = client.post(
        f"/api/v1/verification-sessions/{cool_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ok2.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert cool_res["outcome"] == "STEP-UP"
    assert "NEW_ENROLLMENT_COOLDOWN" in cool_res["reason_codes"]

    # DEMO_MODE disables NEW_ENROLLMENT_COOLDOWN for fast testing
    monkeypatch.setenv("DEMO_MODE", "true")
    demo_sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": sub, "action": "sim_replacement", "resource_ref": "SIM-COOL-2"},
    ).json()
    demo_res = client.post(
        f"/api/v1/verification-sessions/{demo_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ok3.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert demo_res["outcome"] == "APPROVED"


def test_tamper_evident_audit_hash_chain_and_detection(make_test_client, db_session_factory):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-CHAIN-1")

    for idx in range(2):
        s = client.post(
            "/api/v1/verification-sessions",
            json={"sector": "telecom", "subject_ref": "SUB-CHAIN-1", "action": "sim_replacement", "resource_ref": f"SIM-C{idx}"},
        ).json()
        res = client.post(
            f"/api/v1/verification-sessions/{s['session_id']}/audio",
            data={"spoken_transcript": ""},
            files={"file": (f"c{idx}.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
        ).json()
        assert res["outcome"] == "APPROVED"
        last_sid = s["session_id"]

    chain = client.get("/api/v1/audit/verify-chain")
    assert chain.status_code == 200
    chain_body = chain.json()
    assert chain_body["valid"] is True
    assert chain_body["total_events"] >= 2
    assert chain_body["first_broken_index"] is None

    dossier = client.get(f"/api/v1/audit/{last_sid}/export").json()
    assert dossier["audit_chain_status"]["valid"] is True

    # Tamper directly with the first audit row in SQLite -> verify-chain detects exact break
    with db_session_factory() as db:
        first_row = db.query(AuditEvent).order_by(AuditEvent.id.asc()).first()
        assert first_row is not None
        first_row.outcome = "REJECTED"
        db.commit()

    broken = client.get("/api/v1/audit/verify-chain").json()
    assert broken["valid"] is False
    assert broken["first_broken_index"] == 0
    assert broken["first_broken_id"] is not None


def test_auth_hardening_pbkdf2_600k_legacy_compat_cookies_non_ascii_and_headers(make_test_client, monkeypatch):
    # 1) New password hashes use 600,000 iterations encoded in the 4-part hash string
    hashed = hash_password("StrongPass-2026!")
    parts = hashed.split("$")
    assert len(parts) == 4
    assert parts[0] == "pbkdf2_sha256"
    assert int(parts[1]) == PBKDF2_ITERATIONS == 600_000
    assert verify_password("StrongPass-2026!", hashed) is True
    assert verify_password("wrong", hashed) is False

    # 2) Legacy 3-part 120,000-iteration hashes continue to verify
    legacy_salt = secrets.token_bytes(16)
    legacy_digest = hashlib.pbkdf2_hmac("sha256", b"legacy-pw-123", legacy_salt, 120_000)
    legacy_hash = f"pbkdf2_sha256${legacy_salt.hex()}${legacy_digest.hex()}"
    assert verify_password("legacy-pw-123", legacy_hash) is True
    assert verify_password("not-legacy", legacy_hash) is False

    # 3) Security headers & Secure cookie enforcement in LIVE mode + non-ASCII compare_digest safety
    handler, _ = _default_fixture_handler()
    live_client = make_test_client(handler, auto_login_admin=False, demo_mode=False)

    ui_resp = live_client.get("/")
    assert ui_resp.status_code == 200
    assert ui_resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert ui_resp.headers.get("X-Frame-Options") == "DENY"
    assert ui_resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    csp = ui_resp.headers.get("Content-Security-Policy", "")
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp

    non_ascii = live_client.post(
        "/api/v1/auth/login",
        json={"username": "adm\u00edn-\u0915", "password": "p\u00e4ssword-\u0c95"},
    )
    assert non_ascii.status_code == 401

    from app.auth import get_admin_credentials

    admin_u, admin_p = get_admin_credentials()
    live_login = live_client.post("/api/v1/auth/login", json={"username": admin_u, "password": admin_p})
    assert live_login.status_code == 200
    live_cookie = live_login.headers.get("set-cookie", "")
    assert "; Secure" in live_cookie or "; secure" in live_cookie.lower()

    # Demo mode keeps cookies usable over plain localhost HTTP (no Secure flag)
    demo_client = make_test_client(handler, auto_login_admin=False, demo_mode=True)
    demo_u, demo_p = get_admin_credentials()
    demo_login = demo_client.post("/api/v1/auth/login", json={"username": demo_u, "password": demo_p})
    assert demo_login.status_code == 200
    assert "secure" not in demo_login.headers.get("set-cookie", "").lower()


def test_atomic_decision_token_double_consume_race_returns_single_success(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-RACE-1")

    sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-RACE-1", "action": "sim_replacement", "resource_ref": "SIM-R1"},
    ).json()
    res = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("race.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert res["outcome"] == "APPROVED"
    token = res["decision_token"]
    sid = sess["session_id"]
    admin_cookie = dict(client.cookies)

    n_workers = 6
    barrier = threading.Barrier(n_workers)

    def _worker() -> int:
        tc = TestClient(app, base_url="https://testserver")
        try:
            for k, v in admin_cookie.items():
                tc.cookies.set(k, v)
            barrier.wait(timeout=5)
            r = tc.post(
                f"/api/v1/decisions/{sid}/consume",
                json={"decision_token": token, "action": "sim_replacement", "resource_ref": "SIM-R1"},
            )
            return r.status_code
        finally:
            tc.close()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        statuses = list(pool.map(lambda _i: _worker(), range(n_workers)))

    assert statuses.count(200) == 1
    assert statuses.count(409) == n_workers - 1


def test_contextual_risk_signals_sim_age_device_change_and_call_velocity(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)

    profiles = client.get("/api/v1/profiles").json()["profiles"]
    assert profiles["telecom"]["contextual_rules"]["sim_age_days"]["reason_code"] == "RECENT_SIM_SWAP"
    assert profiles["telecom"]["contextual_rules"]["call_velocity_1h"]["hold_reason_code"] == "CALL_VELOCITY_HOLD"

    _enroll_subject(client, "SUB-CTX-RISK")

    # 1) Recent SIM swap (sim_age_days < 7) -> STEP-UP RECENT_SIM_SWAP
    s1 = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-CTX-RISK",
            "action": "sim_replacement",
            "resource_ref": "SIM-C1",
            "sim_age_days": 2,
        },
    )
    assert s1.status_code == 200
    assert s1.json()["contextual_risk_signals"]["sim_age_simulated"] is True
    r1 = client.post(
        f"/api/v1/verification-sessions/{s1.json()['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ctx1.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r1["outcome"] == "STEP-UP"
    assert "RECENT_SIM_SWAP" in r1["reason_codes"]

    # 2) Device change flag -> STEP-UP DEVICE_CHANGED
    s2 = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-CTX-RISK",
            "action": "sim_replacement",
            "resource_ref": "SIM-C2",
            "device_changed": True,
        },
    ).json()
    r2 = client.post(
        f"/api/v1/verification-sessions/{s2['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ctx2.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r2["outcome"] == "STEP-UP"
    assert "DEVICE_CHANGED" in r2["reason_codes"]

    # 3) High 1-hour call velocity: session #6 -> STEP-UP (HIGH_CALL_VELOCITY), session #8 -> REVIEW / HOLD
    for idx in range(3, 6):
        client.post(
            "/api/v1/verification-sessions",
            json={"sector": "telecom", "subject_ref": "SUB-CTX-RISK", "action": "sim_replacement", "resource_ref": f"SIM-V{idx}"},
        )
    s6 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-CTX-RISK", "action": "sim_replacement", "resource_ref": "SIM-V6"},
    ).json()
    assert s6["contextual_risk_signals"]["call_velocity_1h"] == 6
    r6 = client.post(
        f"/api/v1/verification-sessions/{s6['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ctx6.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r6["outcome"] == "STEP-UP"
    assert "HIGH_CALL_VELOCITY" in r6["reason_codes"]

    client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-CTX-RISK", "action": "sim_replacement", "resource_ref": "SIM-V7"},
    )
    s8 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-CTX-RISK", "action": "sim_replacement", "resource_ref": "SIM-V8"},
    ).json()
    assert s8["contextual_risk_signals"]["call_velocity_1h"] == 8
    r8 = client.post(
        f"/api/v1/verification-sessions/{s8['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ctx8.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r8["outcome"] == "REVIEW / HOLD"
    assert r8["reason_codes"] == ["CALL_VELOCITY_HOLD"]
    assert len(r8["fallback_options"]) == 3

    # Protective emergency_sim_block remains low-friction even under high call velocity
    s9 = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-CTX-RISK", "action": "emergency_sim_block", "resource_ref": "SIM-EMG9"},
    ).json()
    r9 = client.post(
        f"/api/v1/verification-sessions/{s9['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ctx9.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r9["outcome"] == "APPROVED"


def test_metrics_decisions_per_min_p95_latency_and_ui_no_credit_wasted_copy(make_test_client):
    latencies = [180.0, 220.0, 490.0]
    call_idx = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            payload = copy.deepcopy(VERIFY_FIXTURE)
            payload["timing"] = {"total_ms": latencies[min(call_idx["n"], len(latencies) - 1)]}
            call_idx["n"] += 1
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-M95-1")

    for i in range(3):
        s = client.post(
            "/api/v1/verification-sessions",
            json={"sector": "helpdesk", "subject_ref": "SUB-M95-1", "action": "mfa_reset", "resource_ref": f"T-{i}"},
        ).json()
        r = client.post(
            f"/api/v1/verification-sessions/{s['session_id']}/audio",
            data={"spoken_transcript": ""},
            files={"file": (f"m{i}.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
        )
        assert r.status_code == 200

    metrics = client.get("/api/v1/metrics").json()
    assert metrics["decisions_per_min"] >= 3.0
    assert metrics["p50_latency_ms"] is not None
    assert metrics["p95_latency_ms"] is not None
    assert metrics["p95_latency_ms"] >= metrics["p50_latency_ms"]

    html = Path("frontend/index.html").read_text(encoding="utf-8")
    assert "Decisions / min" in html
    assert "p95 Latency" in html
    assert "DetectifAI Credits" not in html
    assert "credits wasted" not in html.lower()


def test_fallback_completion_works_end_to_end_on_hold_and_writes_audit(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler, auto_login_admin=False)

    reg = client.post(
        "/api/v1/auth/register-with-voice",
        data={
            "username": "hold.fallback.user",
            "display_name": "Hold Fallback User",
            "password": "password123",
            "consent_confirmed": "true",
            "consent_ref": "HOLD-FB-1",
            "consent_text": "Explicit speaker consent confirmed for fallback verification testing.",
        },
        files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    ).json()
    sub = reg["user"]["subject_ref"]

    # Silent audio triggers REVIEW / HOLD with 3 fallback options
    sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": sub, "action": "sim_replacement", "resource_ref": "SIM-HOLD-FB"},
    ).json()
    hold_res = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("silent.wav", make_pcm_wav_bytes(6.0, silent=True), "audio/wav")},
    ).json()
    assert hold_res["outcome"] == "REVIEW / HOLD"
    assert len(hold_res["fallback_options"]) == 3

    # User clicks App Push fallback -> completes session end-to-end with simulated flag + single-use token
    fb_done = client.post(
        f"/api/v1/decisions/{sess['session_id']}/step-up/complete",
        json={"method": "trusted_app_push"},
    )
    assert fb_done.status_code == 200
    body = fb_done.json()
    assert body["outcome"] == "APPROVED"
    assert body["simulated"] is True
    assert "STEP_UP_COMPLETED" in body["reason_codes"]
    assert body["decision_token"] is not None

    # Double fallback completion returns 409
    assert client.post(
        f"/api/v1/decisions/{sess['session_id']}/step-up/complete",
        json={"method": "in_store_kyc"},
    ).status_code == 409


def test_expanded_redteam_sandbox_all_10_scenarios_expected_vs_actual(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)

    expected_outcomes = {
        "token_replay": (409, "REJECTED"),
        "action_substitution": (409, "REJECTED"),
        "expired_challenge": (410, "REJECTED"),
        "provider_outage_fail_closed": (503, "REVIEW / HOLD"),
        "cross_domain_sim_alert": (200, "STEP-UP"),
        "transcript_spoofing": (200, "REJECTED"),
        "reencoded_replay": (200, "REJECTED"),
        "policy_context_tampering": (200, "STEP-UP"),
        "rate_limit_lockout": (429, "REJECTED"),
        "audit_chain_tamper": (200, "REJECTED"),
    }
    for scenario, (exp_http, exp_outcome) in expected_outcomes.items():
        resp = client.post(f"/api/v1/sandbox/run/{scenario}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["sandbox_test"] is True
        assert data["passed"] is True
        assert data["http_status"] == exp_http
        assert data["outcome"] == exp_outcome
        assert isinstance(data["expected"], dict) and isinstance(data["actual"], dict)

    # Confirm live audit hash chain remained intact even after running audit_chain_tamper sandbox probe
    assert client.get("/api/v1/audit/verify-chain").json()["valid"] is True


def test_accuracy_report_endpoint_not_run_vs_loaded_json(make_test_client, tmp_path, monkeypatch):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)

    missing_path = tmp_path / "missing_eval.json"
    monkeypatch.setenv("EVAL_REPORT_PATH", str(missing_path))
    empty = client.get("/api/v1/accuracy/report")
    assert empty.status_code == 200
    assert empty.json()["status"] == "not_run"
    assert empty.json()["message"] == "No evaluation run yet"
    assert empty.json()["report"] is None

    sample_report = {
        "total_samples": 4,
        "genuine_total": 2,
        "attack_total": 2,
        "far": 0.0,
        "frr": 0.0,
        "confusion_matrix": {"genuine_accepted_tp": 2, "genuine_rejected_fn": 0, "attack_blocked_tn": 2, "attack_accepted_fp": 0},
    }
    missing_path.write_text(json.dumps(sample_report), encoding="utf-8")
    loaded = client.get("/api/v1/accuracy/report")
    assert loaded.status_code == 200
    assert loaded.json()["status"] == "available"
    assert loaded.json()["report"]["confusion_matrix"]["attack_blocked_tn"] == 2


def test_eval_accuracy_script_computes_real_far_frr_and_confusion_matrix(tmp_path):
    from scripts.eval_accuracy import _discover_dataset, evaluate_labeled_records

    (tmp_path / "genuine" / "hi-IN").mkdir(parents=True)
    (tmp_path / "cloned").mkdir(parents=True)
    (tmp_path / "genuine" / "hi-IN" / "g1.wav").write_bytes(make_pcm_wav_bytes(5.0))
    (tmp_path / "cloned" / "c1.wav").write_bytes(make_pcm_wav_bytes(5.0))

    discovered = _discover_dataset(tmp_path)
    assert [(lbl, acc, p.name) for lbl, acc, p in discovered] == [
        ("cloned", "unspecified", "c1.wav"),
        ("genuine", "hi-IN", "g1.wav"),
    ]

    metrics = evaluate_labeled_records([
        {"label": "genuine", "accent": "hi-IN", "outcome": "APPROVED"},
        {"label": "noisy", "accent": "kn-IN", "outcome": "APPROVED"},
        {"label": "genuine", "accent": "en-IN", "outcome": "REVIEW / HOLD"},
        {"label": "cloned", "accent": "unspecified", "outcome": "REJECTED"},
        {"label": "impostor", "accent": "en-IN", "outcome": "APPROVED"},
    ])
    assert metrics["total_samples"] == 5
    assert metrics["far"] == 0.5
    assert metrics["frr"] == 0.3333
    assert metrics["confusion_matrix"] == {
        "genuine_accepted_tp": 2,
        "genuine_rejected_fn": 1,
        "attack_blocked_tn": 1,
        "attack_accepted_fp": 1,
    }


def test_negative_live_asr_mismatch_ignores_matching_client_transcript(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler, demo_mode=False)
    _enroll_subject(client, "SUB-NEG-ASR")

    client._asr_cfg["transcript"] = "completely different wrong words"
    sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-NEG-ASR", "action": "sim_replacement", "resource_ref": "SIM-N1"},
    ).json()
    res = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": sess["challenge_text"]},  # Attacker spoofs client transcript
        files={"file": ("spoof.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert res["outcome"] == "REJECTED"
    assert "CHALLENGE_MISMATCH" in res["reason_codes"]
    assert res["decision_token"] is None


def test_repo_cleanup_stray_file_removed_and_docs_present():
    assert not Path("0.26.0").exists()
    assert Path("CHANGELOG.md").exists()
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Honest Limitations" in readme
    assert "WEBHOOK_SIGNING_SECRET" in readme
    assert "ASR_PROVIDER" in readme


def test_enrollment_fake_audio_rejected_not_saved_with_audit(make_test_client, db_session_factory):
    fake_detect = copy.deepcopy(ALETHEIA_DETECT_FIXTURE)
    fake_detect["result"]["verdict"] = "fake"
    fake_detect["result"]["is_synthetic"] = True
    fake_detect["result"]["score"] = 0.94

    called_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called_paths.append(request.url.path)
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(fake_detect))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        return httpx.Response(404)

    client = make_test_client(handler)
    r_admin = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": "SUB-FAKE-1",
            "consent_confirmed": "true",
            "consent_ref": "C-FAKE-1",
            "consent_text": "Explicit speaker consent confirmed for synthetic gate test.",
        },
        files={"file": ("fake_enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert r_admin.status_code == 422
    detail = r_admin.json()["detail"]
    assert detail["reason_codes"] == ["ENROLLMENT_SYNTHETIC_INDICATOR"]
    assert "100% FAKE" not in json.dumps(detail).upper()
    assert "/v1/speakers/enroll" not in called_paths

    # Also test register-with-voice path atomically rejects & does NOT create user
    r_reg = client.post(
        "/api/v1/auth/register-with-voice",
        data={
            "username": "fake.voice.user",
            "display_name": "Fake Voice User",
            "password": "password123",
            "consent_confirmed": "true",
            "consent_ref": "C-FAKE-2",
            "consent_text": "Explicit speaker consent confirmed for registration test.",
        },
        files={"file": ("fake_reg.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert r_reg.status_code == 422
    assert r_reg.json()["detail"]["reason_codes"] == ["ENROLLMENT_SYNTHETIC_INDICATOR"]

    with db_session_factory() as db:
        assert db.query(Enrollment).filter_by(subject_ref="SUB-FAKE-1").first() is None
        audits = db.query(AuditEvent).all()
        assert any("ENROLLMENT_SYNTHETIC_INDICATOR" in json.loads(a.reason_codes_json) and a.outcome == "REJECTED" for a in audits)


def test_enrollment_real_audio_accepted_as_active_with_deepfake_metadata(make_test_client, db_session_factory):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    data = _enroll_subject(client, "SUB-REAL-ACTIVE")
    enr = data["enrollment"]
    assert enr["enrollment_status"] == "ACTIVE"
    assert enr["enroll_deepfake_score"] == ALETHEIA_DETECT_FIXTURE["result"]["score"]
    assert enr["enroll_deepfake_threshold"] == ALETHEIA_DETECT_FIXTURE["result"]["threshold"]
    assert enr["enroll_deepfake_request_id"] == ALETHEIA_DETECT_FIXTURE["request_id"]
    assert len(enr["enroll_deepfake_segments"]) == len(ALETHEIA_DETECT_FIXTURE["result"]["segments"])


def test_enrollment_deepfake_gate_provider_error_or_unparseable_fails_closed_503(make_test_client, db_session_factory):
    mode = {"case": "timeout"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            if mode["case"] == "timeout":
                raise httpx.ReadTimeout("Simulated Aletheia timeout", request=request)
            if mode["case"] == "unparseable":
                bad = copy.deepcopy(ALETHEIA_DETECT_FIXTURE)
                bad["result"]["verdict"] = "inconclusive_weird"
                return httpx.Response(200, json=bad)
            return httpx.Response(503, json={"error": {"code": "service_unavailable", "message": "Overloaded"}})
        return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))

    client = make_test_client(handler)
    for err_case in ("timeout", "unparseable", "503"):
        mode["case"] = err_case
        resp = client.post(
            "/api/v1/enrollments",
            data={
                "subject_ref": f"SUB-ERR-{err_case}",
                "consent_confirmed": "true",
                "consent_ref": f"C-{err_case}",
                "consent_text": "Explicit speaker consent confirmed for fail-closed test.",
            },
            files={"file": ("err.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
        )
        assert resp.status_code == 503
        assert resp.json()["detail"]["reason_codes"] == ["PROVIDER_UNAVAILABLE"]

    with db_session_factory() as db:
        assert db.query(Enrollment).count() == 0
        audits = db.query(AuditEvent).all()
        assert any("PROVIDER_UNAVAILABLE" in json.loads(a.reason_codes_json) for a in audits)


def test_enrollment_duplicate_voice_match_second_account_pending_review(make_test_client):
    verify_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            verify_calls.append("called")
            payload = copy.deepcopy(VERIFY_FIXTURE)
            payload["result"]["decision"] = "match"
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    client = make_test_client(handler)
    first = _enroll_subject(client, "SUB-DUP-ORIG")
    assert first["enrollment"]["enrollment_status"] == "ACTIVE"

    second = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": "SUB-DUP-SECOND",
            "consent_confirmed": "true",
            "consent_ref": "C-DUP-2",
            "consent_text": "Explicit speaker consent confirmed for second account.",
        },
        files={"file": ("second_voice.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert second.status_code == 200
    body = second.json()["enrollment"]
    assert body["enrollment_status"] == "PENDING_REVIEW"
    assert body["duplicate_of_subject_ref"] == "SUB-DUP-ORIG"
    assert "DUPLICATE_VOICE_DETECTED" in body["reason_codes"]
    assert len(verify_calls) == 1


def test_enrollment_duplicate_sha256_flagged_without_calling_verify(make_test_client):
    verify_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            verify_calls.append("verify_called")
            return httpx.Response(200, json=copy.deepcopy(VERIFY_FIXTURE))
        return httpx.Response(404)

    client = make_test_client(handler)
    shared_wav = make_pcm_wav_bytes(16.0)
    r1 = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": "SUB-SHA-1",
            "consent_confirmed": "true",
            "consent_ref": "C-SHA-1",
            "consent_text": "Explicit speaker consent confirmed for SHA check.",
        },
        files={"file": ("one.wav", shared_wav, "audio/wav")},
    )
    assert r1.status_code == 200
    assert r1.json()["enrollment"]["enrollment_status"] == "ACTIVE"

    r2 = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": "SUB-SHA-2",
            "consent_confirmed": "true",
            "consent_ref": "C-SHA-2",
            "consent_text": "Explicit speaker consent confirmed for SHA check.",
        },
        files={"file": ("two.wav", shared_wav, "audio/wav")},
    )
    assert r2.status_code == 200
    b2 = r2.json()["enrollment"]
    assert b2["enrollment_status"] == "PENDING_REVIEW"
    assert b2["duplicate_of_subject_ref"] == "SUB-SHA-1"
    assert "DUPLICATE_VOICE_DETECTED" in b2["reason_codes"]
    assert len(verify_calls) == 0


def test_enrollment_duplicate_check_timeout_or_error_fails_closed_incomplete(make_test_client, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            return httpx.Response(503, json={"error": {"code": "service_unavailable", "message": "Down"}})
        return httpx.Response(404)

    client = make_test_client(handler)
    r1 = _enroll_subject(client, "SUB-INC-1")
    assert r1["enrollment"]["enrollment_status"] == "ACTIVE"

    r2 = client.post(
        "/api/v1/enrollments",
        data={
            "subject_ref": "SUB-INC-2",
            "consent_confirmed": "true",
            "consent_ref": "C-INC-2",
            "consent_text": "Explicit speaker consent confirmed for incomplete check test.",
        },
        files={"file": ("two.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert r2.status_code == 200
    b2 = r2.json()["enrollment"]
    assert b2["enrollment_status"] == "PENDING_REVIEW"
    assert b2["duplicate_of_subject_ref"] is None
    assert "DUPLICATE_CHECK_INCOMPLETE" in b2["reason_codes"]


def test_pending_review_user_critical_action_holds_while_emergency_sim_block_approved(make_test_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            return httpx.Response(200, json=copy.deepcopy(VERIFY_FIXTURE))
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-PR-ORIG")

    flagged = _enroll_subject(client, "SUB-PR-USER")
    assert flagged["enrollment"]["enrollment_status"] == "PENDING_REVIEW"

    # Critical action (sim_replacement) routes to REVIEW / HOLD with fallbacks
    crit_sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-PR-USER", "action": "sim_replacement", "resource_ref": "SIM-PR-1"},
    ).json()
    crit_res = client.post(
        f"/api/v1/verification-sessions/{crit_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("v1.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert crit_res["outcome"] == "REVIEW / HOLD"
    assert crit_res["reason_codes"] == ["ENROLLMENT_UNDER_REVIEW"]
    assert crit_res["decision_token"] is None
    assert len(crit_res["fallback_options"]) == 3

    # Protective low-friction emergency_sim_block still works and approves cleanly
    emg_sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-PR-USER", "action": "emergency_sim_block", "resource_ref": "SIM-EMG-PR"},
    ).json()
    emg_res = client.post(
        f"/api/v1/verification-sessions/{emg_sess['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("v2.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert emg_res["outcome"] == "APPROVED"
    assert emg_res["decision_token"] is not None


def test_privacy_end_user_responses_never_contain_duplicate_subject_ref(make_test_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            return httpx.Response(200, json=copy.deepcopy(VERIFY_FIXTURE))
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-PRIV-ORIG")
    client.post("/api/v1/auth/logout")

    # Standard user registers with matching voiceprint -> NEVER sees SUB-PRIV-ORIG anywhere
    reg = client.post(
        "/api/v1/auth/register-with-voice",
        data={
            "username": "privacy.user",
            "display_name": "Privacy User",
            "password": "password123",
            "consent_confirmed": "true",
            "consent_ref": "C-PRIV-1",
            "consent_text": "Explicit speaker consent confirmed for privacy check.",
        },
        files={"file": ("reg.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert reg.status_code == 200
    reg_text = reg.text
    assert "SUB-PRIV-ORIG" not in reg_text
    assert "duplicate_of_subject_ref" not in reg_text
    assert reg.json()["enrollment"]["enrollment_status"] == "PENDING_REVIEW"
    assert (
        reg.json()["enrollment"]["status_message"]
        == "Your voiceprint was saved and needs a quick manual review before high-risk actions are enabled."
    )

    ov_resp = client.get("/api/v1/user/overview")
    list_resp = client.get("/api/v1/enrollments")
    for resp in (ov_resp, list_resp):
        assert "SUB-PRIV-ORIG" not in resp.text
        assert "duplicate_of_subject_ref" not in resp.text

    # Admin sees duplicate_of_subject_ref via admin endpoint
    from app.auth import get_admin_credentials

    admin_u, admin_p = get_admin_credentials()
    client.post("/api/v1/auth/logout")
    client.post("/api/v1/auth/login", json={"username": admin_u, "password": admin_p})
    flagged = client.get("/api/v1/admin/enrollments?status=PENDING_REVIEW").json()["enrollments"]
    assert len(flagged) == 1
    assert flagged[0]["duplicate_of_subject_ref"] == "SUB-PRIV-ORIG"


def test_admin_enrollment_review_approve_and_reject_with_rbac(make_test_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            return httpx.Response(200, json=copy.deepcopy(VERIFY_FIXTURE))
        return httpx.Response(404)

    client = make_test_client(handler)
    _enroll_subject(client, "SUB-REV-ORIG")

    # Enroll two flagged users cleanly by switching to user accounts or admin enroll
    f_app = _enroll_subject(client, "SUB-REV-APP")["enrollment"]
    assert f_app["enrollment_status"] == "PENDING_REVIEW"

    # Non-admin gets 403 on admin enrollment endpoints
    client.post("/api/v1/auth/logout")
    client.post("/api/v1/auth/register", json={"username": "nonadmin.rev", "display_name": "Non Admin", "password": "password123"})
    assert client.get("/api/v1/admin/enrollments?status=PENDING_REVIEW").status_code == 403
    assert client.post("/api/v1/admin/enrollments/SUB-REV-APP/review", json={"decision": "APPROVED", "reason": "Unauthorized"}).status_code == 403

    # Log back in as admin
    from app.auth import get_admin_credentials

    admin_u, admin_p = get_admin_credentials()
    client.post("/api/v1/auth/logout")
    client.post("/api/v1/auth/login", json={"username": admin_u, "password": admin_p})

    # Approve SUB-REV-APP -> ACTIVE, critical sim_replacement works, audit logged
    app_resp = client.post(
        "/api/v1/admin/enrollments/SUB-REV-APP/review",
        json={"decision": "APPROVED", "reason": "Confirmed identical twin in store"},
    )
    assert app_resp.status_code == 200
    assert app_resp.json()["enrollment"]["enrollment_status"] == "ACTIVE"
    assert app_resp.json()["audit_code"] == "ENROLLMENT_APPROVED"

    s_ok = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-REV-APP", "action": "sim_replacement", "resource_ref": "SIM-APP-1"},
    ).json()
    r_ok = client.post(
        f"/api/v1/verification-sessions/{s_ok['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("ok.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
    assert r_ok["outcome"] == "APPROVED"

    # Reject SUB-REV-APP (or another flagged user) -> unusable (403 on new sessions/callbacks)
    rej_resp = client.post(
        "/api/v1/admin/enrollments/SUB-REV-APP/review",
        json={"decision": "REJECTED", "reason": "Fraudulent duplicate confirmed"},
    )
    assert rej_resp.status_code == 200
    assert rej_resp.json()["enrollment"]["enrollment_status"] == "REJECTED"
    assert rej_resp.json()["audit_code"] == "ENROLLMENT_REJECTED"

    blocked_sess = client.post(
        "/api/v1/verification-sessions",
        json={"sector": "telecom", "subject_ref": "SUB-REV-APP", "action": "sim_replacement", "resource_ref": "SIM-BLK-1"},
    )
    assert blocked_sess.status_code == 403
    audits = client.get("/api/v1/audit").json()["audit_events"]
    assert any("ENROLLMENT_APPROVED" in a["reason_codes"] for a in audits)
    assert any("ENROLLMENT_REJECTED" in a["reason_codes"] for a in audits)


def test_enrollment_unrelated_voices_no_match_enroll_as_active(make_test_client):
    verify_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/deepfake/detect":
            return httpx.Response(200, json=copy.deepcopy(ALETHEIA_DETECT_FIXTURE))
        if request.url.path == "/v1/speakers/enroll":
            return httpx.Response(200, json=copy.deepcopy(ENROLL_FIXTURE))
        if request.url.path == "/v1/speakers/verify":
            verify_calls.append("verify")
            payload = copy.deepcopy(VERIFY_FIXTURE)
            payload["result"]["decision"] = "no_match"
            payload["result"]["similarity"] = 0.18
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    client = make_test_client(handler)
    e1 = _enroll_subject(client, "SUB-UNREL-1")["enrollment"]
    e2 = _enroll_subject(client, "SUB-UNREL-2")["enrollment"]
    e3 = _enroll_subject(client, "SUB-UNREL-3")["enrollment"]

    assert e1["enrollment_status"] == "ACTIVE"
    assert e2["enrollment_status"] == "ACTIVE"
    assert e3["enrollment_status"] == "ACTIVE"
    assert e2["duplicate_of_subject_ref"] is None
    assert e3["duplicate_of_subject_ref"] is None
    assert len(verify_calls) == 3  # 1 against SUB-UNREL-1 + 2 against SUB-UNREL-1 & SUB-UNREL-2
