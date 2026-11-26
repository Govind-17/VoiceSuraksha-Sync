# VoiceSuraksha-Sync (`voicelock-gateway`)

Action-bound voice-security gateway powered by **DetectifAI Eidos** (`/v1/speakers/enroll`, `/v1/speakers/verify`) and **Aletheia** (`/v1/deepfake/detect`) for **Telecom SIM Replacement**, **IT Helpdesk MFA Guard**, and cross-domain **Banking / Campus** presets.

## Architecture

- **Backend**: Modular-monolith FastAPI + Pydantic + httpx + PostgreSQL / SQLite via SQLAlchemy (`app/db.py`), with deterministic policy rules (`app/policy.py`), server-side hosted ASR (`app/asr.py`), NumPy perceptual spectral fingerprinting (`app/audio_fingerprint.py`), and hybrid in-memory + SQL rate limiting (`app/rate_limit.py`).
- **Frontend**: Single-file SOC & User UI served directly at `/` (`frontend/index.html`) with strict security headers (`X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, restrictive `Content-Security-Policy`) and no separate frontend build step.
- **Privacy, Security & UX**: Raw audio is **never** stored in the database; only SHA-256 digests, 64-bit perceptual spectral hashes, pseudonymous `subject_ref` IDs, provider scores/verdicts, and single-use decision token hashes are persisted. Pre-checks (audio replay, silence/clipping quality) block replay attacks and give users fast re-record UX feedback rather than waiting on full downstream verification.

## Outcome Vocabulary (`CANONICAL_OUTCOMES`)

Every backend endpoint, audit event, compliance JSON dossier, and dashboard panel uses one vocabulary:

- **`APPROVED`**: Voice biometrics (`Eidos`), liveness (`Aletheia`), and server-verified spoken challenge passed with zero blocking context (or completed `STEP-UP` / reviewer approval) — issues one single-use `decision_token` consumed atomically via conditional SQL `UPDATE`.
- **`STEP-UP`**: Acoustic match succeeded, but contextual or cross-domain risk signals (`AFTER_HOURS_COOLING_HOLD`, `PRIVILEGED_APPROVAL_REQUIRED`, `AMOUNT_REQUIRED`, `HIGH_VALUE_STEP_UP`, `CROSS_DOMAIN_ALERT`, `BORDERLINE_SECOND_CHALLENGE`, `NEW_ENROLLMENT_COOLDOWN`, `RECENT_SIM_SWAP`, `DEVICE_CHANGED`, `HIGH_CALL_VELOCITY`) require out-of-band confirmation (`trusted_app_push`, `manager_callback`, `in_store_kyc`).
- **`REVIEW / HOLD`**: Evidence was inconclusive or temporarily unavailable (`AUDIO_QUALITY_LOW` for silent/clipped/`<4s` WAV, `PROVIDER_UNAVAILABLE`, `ASR_UNAVAILABLE`, `BORDERLINE_VOICE_HOLD`, `CALL_VELOCITY_HOLD`) — genuine users are never auto-cancelled and receive clickable Non-Voice Accessibility Fallbacks.
- **`REJECTED`**: Clear negative evidence (`DEEPFAKE_SIGNAL`, `SPEAKER_NO_MATCH`, `AUDIO_REPLAY_DETECTED`, `CHALLENGE_MISMATCH`, `CHALLENGE_EXPIRED`, `RATE_LIMITED`, or reviewer rejection).

## Threat Model

| Attack Vector | Control Implemented | Deterministic Outcome |
|---|---|---|
| Spoken transcript spoofing | Client transcript ignored in LIVE mode; server calls `ServerASRAdapter` (`ASR_PROVIDER` / `ASR_API_KEY`) | `REJECTED` (`CHALLENGE_MISMATCH`) or `REVIEW / HOLD` (`ASR_UNAVAILABLE`) |
| Policy context tampering | Server computes IST after-hours window, `privileged_role` from DB, 1h call velocity, and admin-only banking `amount_inr` | `STEP-UP` / `REVIEW / HOLD` enforced regardless of client payload |
| Re-recorded / gain-scaled audio replay | Exact SHA-256 digest + NumPy 64-bit spectral perceptual hash (`<= 8` Hamming bits) across prior verifications | `REJECTED` (`AUDIO_REPLAY_DETECTED`) |
| Single-segment splice / deepfake | WAV `>= 8s` split into 2–3 overlapping slices evaluated concurrently; any `fake` slice rejects, disagreement steps up | `REJECTED` (`DEEPFAKE_SIGNAL`) or `STEP-UP` (`BORDERLINE_SECOND_CHALLENGE`) |
| Token replay & race double-consume | Atomic SQL `UPDATE ... WHERE consumed_at IS NULL AND expires_at > now` + action/resource binding | First `200`, concurrent/replay `409` |
| Brute-force login & endpoint flooding | In-memory + SQL `GatewayRateLimiter` + 5-fail lockout persisted across restarts | HTTP `429` + `Retry-After` + `RATE_LIMITED` audit event |
| Audit log tampering | `AuditEvent` SHA-256 hash chain (`prev_hash` → `entry_hash`) verified by `GET /api/v1/audit/verify-chain` | Exact broken index detected |
| Synthetic voice enrollment | Aletheia `detect_deepfake()` runs BEFORE `enroll_speaker()` | Rejected (`ENROLLMENT_SYNTHETIC_INDICATOR`, not saved) or `503` on outage |
| Multi-account duplicate voice | Exact `audio_sha256` match + parallel Eidos `verify_with_deepfake()` against `ACTIVE` enrollments (~20s budget) | `PENDING_REVIEW` (`DUPLICATE_VOICE_DETECTED` / `DUPLICATE_CHECK_INCOMPLETE`) |

## Enrollment Protection (Aletheia Deepfake Gate + Duplicate-Voice Check)

Every voice enrollment (`/api/v1/auth/register-with-voice` and `POST /api/v1/enrollments`) passes through `_validate_and_enroll_wav`:
1. **Enrollment Deepfake Gate (`detect_deepfake`)**: Aletheia (`POST /v1/deepfake/detect`) scores the incoming WAV before Eidos enrolls it. If `verdict == "fake"`, enrollment is blocked (`ENROLLMENT_SYNTHETIC_INDICATOR`) with calm re-record guidance and an audit entry (no "100% FAKE" claim). Any provider error or unparseable verdict fails closed with HTTP `503` (`PROVIDER_UNAVAILABLE`). Passing `score`, `threshold`, `segments`, and `request_id` metadata are stored for SOC review.
2. **Duplicate-Voice Check (`verify_with_deepfake`)**: Before `enroll_speaker()`, incoming audio is checked against every other `ACTIVE` enrollment via exact `audio_sha256` equality (zero API calls) followed by concurrent `verify_with_deepfake()` checks (max 8 parallel, ~20s budget). Matches or incomplete checks save the enrollment as `PENDING_REVIEW` (`DUPLICATE_VOICE_DETECTED` or `DUPLICATE_CHECK_INCOMPLETE`) rather than hard-rejecting twins/family members. End users see only a neutral manual-review message (matched `subject_ref` is admin-only); critical actions route to `REVIEW / HOLD` (`ENROLLMENT_UNDER_REVIEW`) with fallbacks while `emergency_sim_block` / `wifi_support` remain active. Admins review flagged rows via `GET /api/v1/admin/enrollments?status=PENDING_REVIEW` and `POST /api/v1/admin/enrollments/{subject_ref}/review`.

> **Third-Party Provider Note**: All voice biometric and anti-spoofing evaluations rely exclusively on **DetectifAI Eidos** (`POST /v1/speakers/enroll`, `POST /v1/speakers/verify`) and **DetectifAI Aletheia** (`POST /v1/deepfake/detect`), alongside an optional hosted speech-to-text endpoint configured via `ASR_PROVIDER` (`sarvam`, `deepgram`, or `groq`) — no other external biometric/ML providers or local neural models are used.

## Gateway Profiles (`GATEWAY_PROFILES`)

1. `telecom` (**LIVE MVP**): `sim_replacement`, `esim_transfer`, `mnp_port_out`, `emergency_sim_block` (`protective_low_friction`), IST after-hours step-up, SIM-age contextual rule (`< 7` days simulated), device-change check, and HMAC-signed `SIM_SWAP_ATTEMPT_BLOCKED` alerts.
2. `helpdesk` (**LIVE SECONDARY**): `mfa_reset`, `vpn_token_reissue`, `admin_password_reset`, `wifi_support` (`protective_low_friction`), server-backed `privileged_role` manager approval.
3. `banking` (**CONFIG PRESET**): `card_replacement`, `credit_limit_increase` (reads active unexpired `SIM_SWAP_ATTEMPT_BLOCKED` alerts with TTL, requires admin-verified `amount_inr`, steps up `>= ₹1,00,000`).
4. `campus` (**CONFIG PRESET**): `portal_recovery`, `registered_contact_change`.

## Environment Variables

- `DETECTIF_API_KEY` — Server-side DetectifAI API key (required for live verification).
- `DETECTIF_BASE_URL` — DetectifAI base URL (default `https://platform.detectif.ai`).
- `DEMO_MODE` — `false` (LIVE, strict secret validation + `Secure` cookies) or `true` (local demo helper conveniences).
- `AUTH_SECRET` — `>= 16`-char random secret signing session cookies in LIVE mode.
- `WEBHOOK_SIGNING_SECRET` — `>= 16`-char random secret signing cross-domain risk events via HMAC-SHA256.
- `ADMIN_USERNAME` / `ADMIN_PASSWORD` — SOC Admin credentials (`>= 16`-char password required in LIVE mode; compared via PBKDF2-HMAC-SHA256 600k iterations for users and UTF-8 `compare_digest` for admins).
- `ASR_PROVIDER` / `ASR_API_KEY` / `ASR_LANGUAGE_CODE` — Hosted ASR backend (`sarvam`, `deepgram`, or `groq`, default `en-IN`).
- `DATABASE_URL` (or `VOICESURAKSHA_DB_URL`) — PostgreSQL URL for durable production persistence on Render, or SQLite URL locally.

## Setup, Evaluation & Tests

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Run tests:
```powershell
pytest -v tests/test_gateway.py
```

Run real-audio FAR/FRR evaluation against DetectifAI (no synthetic/fake numbers generated):
```powershell
python scripts/eval_accuracy.py --dataset-dir ./eval_wavs --speaker-id <EIDOS_SPEAKER_ID> --output reports/eval_accuracy.json
```

## Honest Limitations

Voice biometrics and deepfake detection operate strictly as an additional risk and friction layer — never a standalone replacement for carrier KYC, hardware passkeys/OTP, or core bank fraud engines. All partner channels (`trusted_app_push`, `in_store_kyc`, `manager_callback`, banking amount feed, SIM age, device-change flags, and telecom risk propagation) remain explicitly labeled **simulated** integrations in this hackathon MVP. Real-world Eidos/Aletheia accuracy varies with microphone hardware, background noise, codec compression, and regional accents (Indian English, Hindi, Kannada, Hinglish). Local `sqlite:///./voicesuraksha.db` storage is ephemeral across free-tier container redeploys unless managed PostgreSQL (`DATABASE_URL`) is attached. Production telecom deployments would also upgrade batch WAV uploads to real-time SIPREC/WebRTC media streaming and external webhook delivery queues.
