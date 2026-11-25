"""Unified deterministic policy engine and gateway profiles for VoiceSuraksha-Sync."""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

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
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
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
    """Normalize SpeechRecognition transcript and challenge text for server-side comparison."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [_DIGIT_WORDS.get(tok, tok) for tok in cleaned.split()]
    return " ".join(tokens)


def hash_text_sha256(value: str) -> str:
    """Return hex SHA-256 hash of a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
            outcome="REJECT_REQUEST",
            evidence_state="CHALLENGE_FAILED",
            reason_codes=["CHALLENGE_EXPIRED"],
            next_step_script=(
                "This verification challenge has expired before completion. Please start a fresh "
                "verification session and read the new challenge phrase."
            ),
            emit_risk_alert=False,
        )

    if session.require_phrase_match and not phrase_match:
        return AuthorizationDecision(
            outcome="REJECT_REQUEST",
            evidence_state="CHALLENGE_FAILED",
            reason_codes=["CHALLENGE_MISMATCH"],
            next_step_script=(
                "The spoken confirmation phrase did not match the active session challenge. "
                "We cannot complete this request on the current recording; please retry with a fresh session."
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
                "Automated voice evidence is currently inconclusive for this request. "
                "Please place the request on hold and complete an alternate out-of-band verification."
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

        if protective_low_friction:
            return AuthorizationDecision(
                outcome="STEP-UP",
                evidence_state="SYNTHETIC_INDICATOR",
                reason_codes=reasons,
                next_step_script=(
                    "Protective safety action noted, but voice authenticity checks returned an anomaly indicator. "
                    "Please confirm via registered callback or OTP before finalizing."
                ),
                emit_risk_alert=False,
            )

        return AuthorizationDecision(
            outcome="REJECT_REQUEST",
            evidence_state="SYNTHETIC_INDICATOR",
            reason_codes=reasons,
            next_step_script=(
                "Voice checks flagged a synthetic audio indicator, so this action cannot be authorized by voice. "
                "No accusation is implied; please direct the requester to in-person or alternate verified channels."
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

        if protective_low_friction:
            return AuthorizationDecision(
                outcome="STEP-UP",
                evidence_state="SPEAKER_NO_MATCH",
                reason_codes=reasons,
                next_step_script=(
                    "Voiceprint evidence did not match the enrolled profile for this protective action. "
                    "Please complete a quick registered-contact OTP or supervisor check to proceed."
                ),
                emit_risk_alert=False,
            )

        return AuthorizationDecision(
            outcome="REVIEW / HOLD",
            evidence_state="SPEAKER_NO_MATCH",
            reason_codes=reasons,
            next_step_script=(
                "Voiceprint evidence did not align with the enrolled reference sample. "
                "Please hold the request and verify identity using an independent secondary channel."
            ),
            emit_risk_alert=(session.sector == "telecom" and critical_action),
        )

    # Matrix cell: speaker_decision == "match" and deepfake_verdict == "real"
    if protective_low_friction:
        return AuthorizationDecision(
            outcome="PASS",
            evidence_state="ACOUSTIC_PASS",
            reason_codes=[],
            next_step_script="Voice and challenge verification passed. Proceed immediately with the protective action.",
            emit_risk_alert=False,
        )

    step_up_reasons: list[str] = []
    if has_cross_domain_alert:
        step_up_reasons.append("CROSS_DOMAIN_ALERT")
    if session.sector == "helpdesk" and session.privileged_role:
        step_up_reasons.append("PRIVILEGED_APPROVAL_REQUIRED")
    if session.sector == "telecom" and session.after_hours:
        step_up_reasons.append("AFTER_HOURS_COOLING_HOLD")
    if (
        session.sector == "banking"
        and session.amount_inr is not None
        and session.amount_inr >= GATEWAY_PROFILES["banking"]["amount_step_up_threshold_inr"]
    ):
        step_up_reasons.append("HIGH_VALUE_STEP_UP")

    if has_cross_domain_alert:
        return AuthorizationDecision(
            outcome="REVIEW / HOLD",
            evidence_state="ACOUSTIC_PASS",
            reason_codes=step_up_reasons,
            next_step_script=(
                "Voice evidence matched, however an active cross-domain SIM swap alert is present for this subject. "
                "Hold the request for mandatory fraud-ops review and out-of-band confirmation."
            ),
            emit_risk_alert=False,
        )

    if step_up_reasons:
        return AuthorizationDecision(
            outcome="STEP-UP",
            evidence_state="ACOUSTIC_PASS",
            reason_codes=step_up_reasons,
            next_step_script=(
                "Voice evidence matched, and contextual policy controls require a secondary step-up approval "
                "(manager sign-off, cooling hold, or high-value OTP) before releasing the action."
            ),
            emit_risk_alert=False,
        )

    return AuthorizationDecision(
        outcome="PASS",
        evidence_state="ACOUSTIC_PASS",
        reason_codes=[],
        next_step_script="Acoustic, deepfake, and challenge checks all passed. One-time action token issued.",
        emit_risk_alert=False,
    )
