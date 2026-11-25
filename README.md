# VoiceSuraksha-Sync (`voicelock-gateway`)

Action-bound voice-security gateway powered by **DetectifAI Eidos** (`/v1/speakers/enroll`, `/v1/speakers/verify`) and **Aletheia** (`/v1/deepfake/detect`) for **Telecom SIM Replacement**, **IT Helpdesk MFA Guard**, and cross-domain **Banking / Campus** presets.

## Architecture

- **Backend**: Modular-monolith FastAPI + Pydantic + httpx + SQLite (`voicesuraksha.db`) via SQLAlchemy.
- **Frontend**: Single-file SOC UI served directly by FastAPI at `/` (`frontend/index.html`) — no CORS or separate build step.
- **Privacy & Security**: Raw audio is **never** stored in SQLite; only SHA-256 digests, pseudonymous `subject_ref` IDs, provider scores/verdicts, and single-use decision token hashes are persisted. `DETECTIF_API_KEY` is loaded server-side from `.env` only.

## Gateway Profiles (`GATEWAY_PROFILES`)

1. `telecom` (**LIVE MVP**): `sim_replacement`, `esim_transfer`, `mnp_port_out`, `emergency_sim_block` (`after_hours_step_up`, emits simulated `SIM_SWAP_ATTEMPT_BLOCKED` alerts)
2. `helpdesk` (**LIVE SECONDARY**): `mfa_reset`, `vpn_token_reissue`, `admin_password_reset`, `wifi_support` (`privileged_role_manager_approval`)
3. `banking` (**CONFIG PRESET**): `card_replacement`, `credit_limit_increase` (reads active `SIM_SWAP_ATTEMPT_BLOCKED` risk events, `amount_step_up_threshold_inr=100000`)
4. `campus` (**CONFIG PRESET**): `portal_recovery`, `registered_contact_change`

## Setup & Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add your DETECTIF_API_KEY to .env
uvicorn app.main:app --reload
```

## Tests

```powershell
pytest tests/test_gateway.py
```
