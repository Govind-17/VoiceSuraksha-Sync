"""Gateway integration tests using saved DetectifAI fixtures in tests/fixtures/."""

from __future__ import annotations

import copy
import io
import json
import math
import struct
import wave
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.asr import ServerASRAdapter, ServerASRResult
from app.auth import validate_live_secrets
from app.db import Base, CURRENT_SCHEMA_VERSION, create_db_engine, get_database_url, get_storage_backend, init_db, is_storage_durable
from app.detectif_client import DetectifClient
from app.main import _is_ist_after_hours, app, get_asr_adapter, get_db, get_detectif_client
from app.models import AuditEvent, Enrollment, SchemaVersion, VerificationSession, utcnow
from app.policy import normalize_transcript, verify_phrase_match, verify_risk_event_signature

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ENROLL_FIXTURE = json.loads((FIXTURES_DIR / "eidos_enroll.json").read_text(encoding="utf-8"))
VERIFY_FIXTURE = json.loads((FIXTURES_DIR / "eidos_verify_with_deepfake.json").read_text(encoding="utf-8"))
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
            freq = 220.0 + (_WAV_NONCE % 97) * 7.0
            amp = 6500.0  # ~ -14 dBFS clean non-clipped speech-level tone, unique per call
            samples = [int(amp * math.sin(2.0 * math.pi * freq * i / sample_rate)) for i in range(num_frames)]
            wf.writeframes(struct.pack(f"<{num_frames}h", *samples))
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
        client = TestClient(app)
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
        data={"subject_ref": "SUB-104", "consent_confirmed": "false", "consent_ref": "C-0"},
        files={"file": ("enroll.wav", make_pcm_wav_bytes(16.0), "audio/wav")},
    )
    assert bad_consent.status_code == 400

    # Too short enrollment rejected (HTTP 400)
    short_enroll = client.post(
        "/api/v1/enrollments",
        data={"subject_ref": "SUB-104", "consent_confirmed": "true", "consent_ref": "C-0"},
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
    assert len(captured_headers) == 2

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
    # Verify only enroll called DetectifAI, not verify
    assert len(captured_headers) == 1

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
    assert len(captured_headers) == 1


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

    # 2. Subsequent banking action with matching voice is HELD due to active cross-domain alert
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
    assert bank_res["outcome"] == "REJECTED"
    assert bank_res["status"] == "CANCELLED"
    assert "CROSS_DOMAIN_ALERT" in bank_res["reason_codes"]
    assert bank_res["decision_token"] is None


def test_provider_timeout_and_403_fail_closed(make_test_client):
    mode = {"behavior": "timeout"}

    def handler(request: httpx.Request) -> httpx.Response:
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

    # Timeout case -> fail closed REJECTED (CANCELLED)
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
    assert res1["outcome"] == "REJECTED"
    assert res1["status"] == "CANCELLED"
    assert res1["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert res1["reason_codes"] == ["PROVIDER_UNAVAILABLE"]
    assert res1["decision_token"] is None

    # 403 insufficient_scope case -> fail closed REJECTED (CANCELLED)
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
    assert res2["outcome"] == "REJECTED"
    assert res2["status"] == "CANCELLED"
    assert res2["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert res2["reason_codes"] == ["PROVIDER_UNAVAILABLE"]
    assert res2["decision_token"] is None


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

    # Short <4s clip -> INCONCLUSIVE_EVIDENCE (REJECTED & CANCELLED)
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
    assert short_res["outcome"] == "REJECTED"
    assert short_res["status"] == "CANCELLED"
    assert short_res["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert short_res["reason_codes"] == ["AUDIO_QUALITY_LOW"]


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
    assert len(captured_headers) == 2


def test_render_readiness_head_health_and_upload_limit(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)

    assert client.head("/").status_code == 200
    assert client.head("/api/v1/health").status_code == 200

    oversized = b"0" * (15 * 1024 * 1024 + 10)
    resp = client.post(
        "/api/v1/enrollments",
        data={"subject_ref": "SUB-BIG", "consent_confirmed": "true", "consent_ref": "C-BIG"},
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
            data={"subject_ref": "SUB-OTHER", "consent_confirmed": "true", "consent_ref": "C-1"},
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
        data={"subject_ref": subject_ref, "consent_confirmed": "true", "consent_ref": "RETRY"},
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

    # Trigger a blocked verification (<4s fails closed with REJECTED)
    sess_bad = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": "SUB-901",
            "action": "mfa_reset",
            "resource_ref": "TKT-901",
        },
    ).json()
    bad_res = client.post(
        f"/api/v1/verification-sessions/{sess_bad['session_id']}/audio",
        data={"spoken_transcript": ""},
        files={"file": ("short.wav", make_pcm_wav_bytes(2.0), "audio/wav")},
    ).json()
    assert bad_res["outcome"] == "REJECTED"

    metrics = client.get("/api/v1/metrics").json()
    assert metrics["total_verifications"] == 2
    assert metrics["pass_count"] == 1
    assert metrics["blocked_count"] == 1
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
        "provider_outage_fail_closed": (503, "REJECTED", "PROVIDER_UNAVAILABLE"),
        "cross_domain_sim_alert": (403, "REJECTED", "CROSS_DOMAIN_ALERT"),
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
    assert len(captured_headers) == 2

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
    assert res2["outcome"] == "REJECT_REQUEST"
    assert res2["status"] == "CANCELLED"
    assert res2["evidence_state"] == "REPLAY_DETECTED"
    assert res2["reason_codes"] == ["AUDIO_REPLAY_DETECTED"]
    assert res2["decision_token"] is None
    assert len(captured_headers) == 2  # DetectifAI was NOT called again
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
    assert len(captured_headers) == 1

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
    assert len(captured_headers) == 1

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
    assert len(captured_headers) == 1


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
