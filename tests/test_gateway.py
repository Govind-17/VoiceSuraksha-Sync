"""Gateway integration tests using saved DetectifAI fixtures in tests/fixtures/."""

from __future__ import annotations

import copy
import io
import json
import wave
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.db import Base, create_db_engine
from app.detectif_client import DetectifClient
from app.main import app, get_db, get_detectif_client
from app.models import AuditEvent, Enrollment, VerificationSession, utcnow

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ENROLL_FIXTURE = json.loads((FIXTURES_DIR / "eidos_enroll.json").read_text(encoding="utf-8"))
VERIFY_FIXTURE = json.loads((FIXTURES_DIR / "eidos_verify_with_deepfake.json").read_text(encoding="utf-8"))


def make_pcm_wav_bytes(duration_seconds: float, sample_rate: int = 16000) -> bytes:
    """Create minimal valid 16kHz mono 16-bit PCM WAV bytes in memory for tests."""
    num_frames = int(duration_seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


@pytest.fixture()
def db_session_factory(tmp_path):
    db_file = tmp_path / "test_voicesuraksha.db"
    engine = create_db_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(bind=engine)
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
def make_test_client(db_session_factory):
    def _factory(mock_handler, *, auto_login_admin: bool = True):
        def _override_get_db():
            db = db_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _override_get_db
        app.dependency_overrides[get_detectif_client] = lambda: make_mock_client(mock_handler)
        client = TestClient(app)
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
    assert result["outcome"] == "PASS"
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

    resp = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": "some completely wrong replayed phrase", "strict_phrase": "true"},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome"] == "REJECT_REQUEST"
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
    assert data["outcome"] == "REJECT_REQUEST"
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
    assert data["outcome"] == "REJECT_REQUEST"
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
    assert bank_res["outcome"] == "REVIEW / HOLD"
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

    # Timeout case -> fail closed REVIEW / HOLD
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
    assert res1["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert res1["reason_codes"] == ["PROVIDER_UNAVAILABLE"]
    assert res1["decision_token"] is None

    # 403 insufficient_scope case -> fail closed REVIEW / HOLD
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
    assert res2["evidence_state"] == "INCONCLUSIVE_EVIDENCE"
    assert res2["reason_codes"] == ["PROVIDER_UNAVAILABLE"]
    assert res2["decision_token"] is None


def test_contextual_step_ups_and_protective_low_friction(make_test_client):
    handler, _ = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-501")

    # Privileged helpdesk MFA reset -> STEP-UP even on voice match
    priv_sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "helpdesk",
            "subject_ref": "SUB-501",
            "action": "mfa_reset",
            "resource_ref": "MFA-ADMIN-01",
            "privileged_role": True,
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
    assert emerg_res["outcome"] == "PASS"
    assert emerg_res["decision_token"] is not None

    # Short <4s clip -> INCONCLUSIVE_EVIDENCE (AUDIO_QUALITY_LOW)
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
    tel_res = client.post(
        f"/api/v1/verification-sessions/{tel_sess['session_id']}/audio",
        data={"spoken_transcript": "wrong phrase", "strict_phrase": "true"},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    ).json()
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
    assert bank_res["outcome"] == "PASS"
    assert bank_res["decision_token"] is not None


def test_voice_first_mode_passes_without_exact_transcript(make_test_client):
    handler, captured_headers = _default_fixture_handler()
    client = make_test_client(handler)
    _enroll_subject(client, "SUB-701")

    sess = client.post(
        "/api/v1/verification-sessions",
        json={
            "sector": "telecom",
            "subject_ref": "SUB-701",
            "action": "sim_replacement",
            "resource_ref": "SIM-MASKED-7010",
        },
    ).json()
    assert sess["challenge_text"].startswith("Confirm sim replacement code ")

    # Even with different/empty spoken_transcript, voice-first mode runs Eidos + Aletheia and passes on voice match
    res = client.post(
        f"/api/v1/verification-sessions/{sess['session_id']}/audio",
        data={"spoken_transcript": "hi my name is govind testing our hackathon project"},
        files={"file": ("verify.wav", make_pcm_wav_bytes(6.0), "audio/wav")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["phrase_match"] is False
    assert data["outcome"] == "PASS"
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
    assert ver_json["outcome"] == "PASS"
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
