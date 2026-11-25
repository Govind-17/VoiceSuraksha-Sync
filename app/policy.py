"""Unified deterministic policy engine and gateway profiles for VoiceSuraksha-Sync."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

from app.auth import is_demo_mode, is_weak_secret

_EPHEMERAL_DEMO_WEBHOOK_SECRET = secrets.token_urlsafe(32)

CANONICAL_OUTCOMES = ("APPROVED", "STEP-UP", "REVIEW / HOLD", "REJECTED")

NON_VOICE_FALLBACK_OPTIONS: list[dict[str, Any]] = [
    {
        "method": "trusted_app_push",
        "label": "Trusted App Push Approval",
        "simulated": True,
    },
    {
        "method": "in_store_kyc",
        "label": "In-Store Biometric KYC",
        "simulated": True,
    },
    {
        "method": "manager_callback",
        "label": "Verified Manager Callback",
        "simulated": True,
    },
]

GATEWAY_PROFILES: dict[str, dict[str, Any]] = {
    "telecom": {
        "name": "telecom",
        "mode": "LIVE MVP",
        "display_name": "Telecom SIM Replacement & Port-Out Guard",
        "actions": {
            "sim_replacement": {"label": "SIM Replacement", "protective_low_friction": False, "critical": True},
            "esim_transfer": {"label": "eSIM Transfer", "protective_low_friction": False, "critical": True},
            "mnp_port_out": {"label": "MNP Port-Out Authorization", "protective_low_friction": False, "critical": True},
            "emergency_sim_block": {"label": "Emergency SIM Block", "protective_low_friction": True, "critical": False},
        },
        "after_hours_step_up": True,
        "simulated_alert_type": "SIM_SWAP_ATTEMPT_BLOCKED",
    },
    "helpdesk": {
        "name": "helpdesk",
        "mode": "LIVE SECONDARY",
        "display_name": "IT Helpdesk MFA & Privileged Access Guard",
        "actions": {
            "mfa_reset": {"label": "MFA Device Reset", "protective_low_friction": False, "critical": True},
            "vpn_token_reissue": {"label": "VPN Token Reissue", "protective_low_friction": False, "critical": True},
            "admin_password_reset": {"label": "Admin Password Reset", "protective_low_friction": False, "critical": True},
            "wifi_support": {"label": "Corporate Wi-Fi Support", "protective_low_friction": True, "critical": False},
        },
        "privileged_role_manager_approval": True,
    },
    "banking": {
        "name": "banking",
        "mode": "CONFIG PRESET",
        "display_name": "Banking Card & Credit Limit Guard",
        "actions": {
            "card_replacement": {"label": "Card Replacement", "protective_low_friction": False, "critical": True},
            "credit_limit_increase": {"label": "Credit Limit Increase", "protective_low_friction": False, "critical": True},
        },
        "reads_risk_event": "SIM_SWAP_ATTEMPT_BLOCKED",
        "amount_step_up_threshold_inr": 100000,
    },
    "campus": {
        "name": "campus",
        "mode": "CONFIG PRESET",
        "display_name": "Campus Portal & Contact Recovery Guard",
        "actions": {
            "portal_recovery": {"label": "Student/Staff Portal Recovery", "protective_low_friction": False, "critical": False},
            "registered_contact_change": {"label": "Registered Contact Change", "protective_low_friction": False, "critical": True},
        },
    },
}

_CHALLENGE_WORDS = [
    "blue", "green", "river", "cloud", "apple", "silver",
    "ocean", "sunny", "tiger", "light", "happy", "music",
]

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "won": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "tree": "3",
    "four": "4",
    "for": "4",
    "fore": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
    "niner": "9",
}

_TENS_WORDS = {
    "twenty": "2",
    "thirty": "3",
    "forty": "4",
    "fifty": "5",
    "sixty": "6",
    "seventy": "7",
    "eighty": "8",
    "ninety": "9",
}

_TEEN_WORDS = {
    "ten": ("1", "0"),
    "eleven": ("1", "1"),
    "twelve": ("1", "2"),
    "thirteen": ("1", "3"),
    "fourteen": ("1", "4"),
    "fifteen": ("1", "5"),
    "sixteen": ("1", "6"),
    "seventeen": ("1", "7"),
    "eighteen": ("1", "8"),
    "nineteen": ("1", "9"),
}

_HOMOPHONE_MAP = {
    "blew": "blue",
    "lite": "light",
}


@dataclass(frozen=True)
class VoiceEvidenceInput:
    """Normalized acoustic evidence passed to the policy evaluator."""

    speaker_decision: str | None = None
    deepfake_verdict: str | None = None
    duration_seconds: float | None = None
    provider_error: str | None = None


@dataclass(frozen=True)
class SessionPolicyInput:
    """Minimal session attributes needed for deterministic policy evaluation."""

    sector: str
    action: str
    resource_ref: str
    after_hours: bool = False
    privileged_role: bool = False
    amount_inr: float | None = None
    expired: bool = False
    require_phrase_match: bool = False
    asr_available: bool = True


@dataclass(frozen=True)
class AuthorizationDecision:
    """Deterministic authorization outcome, reason codes, and calm agent script."""

    outcome: str
    evidence_state: str
    reason_codes: list[str] = field(default_factory=list)
    next_step_script: str = ""
    emit_risk_alert: bool = False


def validate_sector_action(sector: str, action: str) -> dict[str, Any]:
    """Validate that sector and action exist in GATEWAY_PROFILES and return action metadata."""
    profile = GATEWAY_PROFILES.get(sector)
    if profile is None:
        raise ValueError(f"Unsupported sector '{sector}'. Valid sectors: {', '.join(GATEWAY_PROFILES)}")
    action_meta = profile["actions"].get(action)
    if action_meta is None:
        valid_actions = ", ".join(profile["actions"].keys())
        raise ValueError(f"Unsupported action '{action}' for sector '{sector}'. Valid actions: {valid_actions}")
    return action_meta


def generate_challenge_phrase(sector: str, action: str, resource_ref: str) -> str:
    """Generate a simple, short spoken prompt for live voice capture."""
    code_words = [secrets.choice(_CHALLENGE_WORDS) for _ in range(2)]
    digits = "".join(str(secrets.randbelow(10)) for _ in range(2))
    readable_action = action.replace("_", " ")
    return f"Confirm {readable_action} code {' '.join(code_words)} {digits}."


def normalize_transcript(text: str) -> str:
    """Normalize ASR/transcript text with bounded digit-word and homophone tolerance while keeping exact token order."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    cleaned = re.sub(r"\be\s+sim\b", "esim", cleaned)
    cleaned = re.sub(r"\bwi\s+fi\b", "wifi", cleaned)

    raw_tokens = cleaned.split()
    out: list[str] = []
    idx = 0
    while idx < len(raw_tokens):
        tok = _HOMOPHONE_MAP.get(raw_tokens[idx], raw_tokens[idx])
        if tok in _TEEN_WORDS:
            out.extend(_TEEN_WORDS[tok])
        elif tok in _TENS_WORDS:
            out.append(_TENS_WORDS[tok])
            # Only look ahead for unit digit (e.g. "forty two" -> "4", "2"; "forty" alone -> "4", "0")
            if idx + 1 < len(raw_tokens) and _DIGIT_WORDS.get(raw_tokens[idx + 1], raw_tokens[idx + 1]) in set("123456789"):
                out.append(_DIGIT_WORDS.get(raw_tokens[idx + 1], raw_tokens[idx + 1]))
                idx += 1
            else:
                out.append("0")
        elif tok in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[tok])
        elif tok.isdigit():
            out.extend(list(tok))
        else:
            out.append(tok)
        idx += 1
    return " ".join(out)


def hash_text_sha256(value: str) -> str:
    """Return hex SHA-256 hash of a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_webhook_signing_secret() -> str:
    """Load WEBHOOK_SIGNING_SECRET from env; refuse missing/default secrets in LIVE mode."""
    secret = os.getenv("WEBHOOK_SIGNING_SECRET", "").strip()
    if is_weak_secret(secret):
        if is_demo_mode():
            return _EPHEMERAL_DEMO_WEBHOOK_SECRET
        raise RuntimeError(
            "WEBHOOK_SIGNING_SECRET is missing or insecure (<16 chars / known default) in LIVE mode."
        )
    return secret


def sign_risk_event_payload(payload: dict[str, Any], secret: str | None = None) -> str:
    """Compute canonical HMAC-SHA256 signature for an outbound cross-domain RiskEvent webhook."""
    key = (secret if secret is not None else get_webhook_signing_secret()).encode("utf-8")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def verify_risk_event_signature(payload: dict[str, Any], signature: str, secret: str | None = None) -> bool:
    """Verify an HMAC-SHA256 webhook signature in constant time."""
    expected = sign_risk_event_payload(payload, secret=secret)
    return hmac.compare_digest(expected, str(signature or ""))


def verify_phrase_match(challenge_text: str, spoken_transcript: str) -> bool:
    """Verify normalized transcript against the session challenge phrase."""
    expected = normalize_transcript(challenge_text)
    spoken = normalize_transcript(spoken_transcript)
    if not expected or not spoken:
        return False
    return expected == spoken


def _has_active_risk_alert(active_risk_events: list[Any], event_type: str = "SIM_SWAP_ATTEMPT_BLOCKED") -> bool:
    for ev in active_risk_events:
        ev_type = ev.get("event_type") if isinstance(ev, dict) else getattr(ev, "event_type", None)
        ev_state = ev.get("risk_state") if isinstance(ev, dict) else getattr(ev, "risk_state", None)
        if ev_type == event_type and ev_state == "ACTIVE":
            return True
    return False


def evaluate_voice_authorization(
    session: SessionPolicyInput,
    voice_result: VoiceEvidenceInput | None,
    phrase_match: bool,
    active_risk_events: list[Any],
) -> AuthorizationDecision:
    """Deterministic policy evaluator using the 2x2 Acoustic Evidence Matrix + INCONCLUSIVE_EVIDENCE.

    Uses documented DetectifAI decision/verdict fields without invented thresholds.
    """
    action_meta = validate_sector_action(session.sector, session.action)
    protective_low_friction = bool(action_meta.get("protective_low_friction", False))
    critical_action = bool(action_meta.get("critical", False))
    has_cross_domain_alert = _has_active_risk_alert(active_risk_events)

    if session.expired:
        return AuthorizationDecision(
            outcome="REJECTED",
            evidence_state="CHALLENGE_FAILED",
            reason_codes=["CHALLENGE_EXPIRED"],
            next_step_script=(
                "Request cancelled: this verification session has expired. Please start a fresh "
                "verification session and submit voice again."
            ),
            emit_risk_alert=False,
        )

    if session.require_phrase_match and session.asr_available and not phrase_match:
        return AuthorizationDecision(
            outcome="REJECTED",
            evidence_state="CHALLENGE_FAILED",
            reason_codes=["CHALLENGE_MISMATCH"],
            next_step_script=(
                "Request cancelled: the spoken confirmation phrase did not match the session challenge."
            ),
            emit_risk_alert=(session.sector == "telecom" and critical_action),
        )

    if (
        voice_result is None
        or voice_result.provider_error is not None
        or voice_result.speaker_decision not in {"match", "no_match"}
        or voice_result.deepfake_verdict not in {"real", "fake"}
        or (voice_result.duration_seconds is not None and voice_result.duration_seconds < 4.0)
    ):
        is_short_audio = (
            voice_result is not None
            and voice_result.provider_error is None
            and voice_result.duration_seconds is not None
            and voice_result.duration_seconds < 4.0
        )
        reason_code = "AUDIO_QUALITY_LOW" if is_short_audio else "PROVIDER_UNAVAILABLE"
        return AuthorizationDecision(
            outcome="REVIEW / HOLD",
            evidence_state="INCONCLUSIVE_EVIDENCE",
            reason_codes=[reason_code],
            next_step_script=(
                "Audio evidence could not be verified reliably. Genuine users are never auto-cancelled — "
                "complete a Non-Voice Accessibility Fallback: Trusted App Push, In-Store Biometric KYC, or Verified Manager Callback."
            ),
            emit_risk_alert=False,
        )

    speaker_match = voice_result.speaker_decision == "match"
    deepfake_fake = voice_result.deepfake_verdict == "fake"

    # Matrix row: synthetic voice indicator observed (verdict == "fake")
    if deepfake_fake:
        reasons = ["DEEPFAKE_SIGNAL"]
        if not speaker_match:
            reasons.append("SPEAKER_NO_MATCH")
        if has_cross_domain_alert:
            reasons.append("CROSS_DOMAIN_ALERT")

        return AuthorizationDecision(
            outcome="REJECTED",
            evidence_state="SYNTHETIC_INDICATOR",
            reason_codes=reasons,
            next_step_script=(
                "Request cancelled: voice checks flagged a synthetic audio indicator, so this action cannot be authorized."
            ),
            emit_risk_alert=(session.sector == "telecom" and critical_action),
        )

    # Matrix row: real audio, speaker did not match enrolled voiceprint
    if not speaker_match:
        reasons = ["SPEAKER_NO_MATCH"]
        if has_cross_domain_alert:
            reasons.append("CROSS_DOMAIN_ALERT")
        if session.sector == "telecom" and session.after_hours:
            reasons.append("AFTER_HOURS_COOLING_HOLD")
        if session.sector == "helpdesk" and session.privileged_role:
            reasons.append("PRIVILEGED_APPROVAL_REQUIRED")

        return AuthorizationDecision(
            outcome="REJECTED",
            evidence_state="SPEAKER_NO_MATCH",
            reason_codes=reasons,
            next_step_script=(
                "Request cancelled: voice did not match the enrolled speaker profile."
            ),
            emit_risk_alert=(session.sector == "telecom" and critical_action),
        )

    if session.require_phrase_match and not session.asr_available:
        return AuthorizationDecision(
            outcome="REVIEW / HOLD",
            evidence_state="INCONCLUSIVE_EVIDENCE",
            reason_codes=["ASR_UNAVAILABLE"],
            next_step_script=(
                "Spoken challenge verification unavailable in LIVE mode (ASR_UNAVAILABLE). "
                "Route to Non-Voice Accessibility Fallback: Trusted App Push, In-Store Biometric KYC, or Manager Callback."
            ),
            emit_risk_alert=False,
        )

    # Matrix cell: speaker_decision == "match" and deepfake_verdict == "real"
    if protective_low_friction:
        return AuthorizationDecision(
            outcome="APPROVED",
            evidence_state="ACOUSTIC_PASS",
            reason_codes=[],
            next_step_script="Voice matched and verified as genuine. Approved to proceed to the next step.",
            emit_risk_alert=False,
        )

    step_up_reasons: list[str] = []
    if has_cross_domain_alert:
        step_up_reasons.append("CROSS_DOMAIN_ALERT")
    if session.sector == "helpdesk" and session.privileged_role:
        step_up_reasons.append("PRIVILEGED_APPROVAL_REQUIRED")
    if session.sector == "telecom" and session.after_hours:
        step_up_reasons.append("AFTER_HOURS_COOLING_HOLD")
    if session.sector == "banking" and critical_action and session.amount_inr is None:
        step_up_reasons.append("AMOUNT_REQUIRED")
    elif (
        session.sector == "banking"
        and session.amount_inr is not None
        and session.amount_inr >= GATEWAY_PROFILES["banking"]["amount_step_up_threshold_inr"]
    ):
        step_up_reasons.append("HIGH_VALUE_STEP_UP")

    if step_up_reasons:
        return AuthorizationDecision(
            outcome="STEP-UP",
            evidence_state="ACOUSTIC_PASS",
            reason_codes=step_up_reasons,
            next_step_script=(
                "Voice matched, and contextual or cross-domain risk signals require out-of-band step-up confirmation before proceeding."
            ),
            emit_risk_alert=False,
        )

    return AuthorizationDecision(
        outcome="APPROVED",
        evidence_state="ACOUSTIC_PASS",
        reason_codes=[],
        next_step_script="Voice matched and verified as genuine. Approved to proceed to the next step.",
        emit_risk_alert=False,
    )
