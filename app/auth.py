"""Stateless HMAC-signed session auth and role-based access control (admin vs user)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request, Response

load_dotenv(override=False)

SESSION_COOKIE_NAME = "vs_session"
TOKEN_TTL_SECONDS = 8 * 3600
_EPHEMERAL_SECRET = secrets.token_urlsafe(32)


def _get_auth_secret() -> bytes:
    secret = os.getenv("AUTH_SECRET") or _EPHEMERAL_SECRET
    return secret.encode("utf-8")


def get_admin_credentials() -> tuple[str, str]:
    """Return configured admin username and password from environment."""
    load_dotenv(override=False)
    username = os.getenv("ADMIN_USERNAME", "admin").strip() or "admin"
    password = os.getenv("ADMIN_PASSWORD", "admin123")
    return username, password


def hash_password(password: str) -> str:
    """Hash a user password using PBKDF2-HMAC-SHA256 with a random 16-byte salt."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash."""
    try:
        scheme, salt_hex, digest_hex = stored_hash.split("$", 2)
        if scheme != "pbkdf2_sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return hmac.compare_digest(actual, expected)


@dataclass(frozen=True)
class AuthActor:
    """Authenticated user or admin identity extracted from session token."""

    username: str
    role: str  # "admin" or "user"
    display_name: str
    subject_ref: str | None = None


def create_session_token(actor: AuthActor) -> str:
    """Create a signed session token for cookie or Bearer usage."""
    payload = {
        "sub": actor.username,
        "role": actor.role,
        "display_name": actor.display_name,
        "subject_ref": actor.subject_ref,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    sig = hmac.new(_get_auth_secret(), b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
    return f"{b64}.{sig_b64}"


def decode_session_token(token: str) -> AuthActor | None:
    """Validate signature and expiry; return AuthActor or None if invalid."""
    try:
        b64, sig_b64 = token.split(".", 1)
        expected_sig = hmac.new(_get_auth_secret(), b64.encode("ascii"), hashlib.sha256).digest()
        padded_sig = sig_b64 + "=" * (-len(sig_b64) % 4)
        actual_sig = base64.urlsafe_b64decode(padded_sig.encode("ascii"))
        if not hmac.compare_digest(actual_sig, expected_sig):
            return None
        padded_b64 = b64 + "=" * (-len(b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded_b64.encode("ascii")).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        role = payload.get("role")
        if role not in {"admin", "user"}:
            return None
        return AuthActor(
            username=str(payload["sub"]),
            role=str(role),
            display_name=str(payload.get("display_name") or payload["sub"]),
            subject_ref=payload.get("subject_ref"),
        )
    except Exception:
        return None


def set_session_cookie(response: Response, token: str) -> None:
    """Attach HttpOnly session cookie to response."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
    )


def clear_session_cookie(response: Response) -> None:
    """Clear the session cookie on logout."""
    response.delete_cookie(key=SESSION_COOKIE_NAME, samesite="lax")


def get_optional_actor(request: Request) -> AuthActor | None:
    """Return AuthActor if request carries a valid session cookie or Bearer token, else None."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        auth_hdr = request.headers.get("Authorization", "")
        if auth_hdr.startswith("Bearer "):
            token = auth_hdr.removeprefix("Bearer ").strip()
    if not token:
        return None
    return decode_session_token(token)


def get_current_actor(request: Request) -> AuthActor:
    """FastAPI dependency requiring any logged-in user or admin."""
    actor = get_optional_actor(request)
    if actor is None:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    return actor


def require_admin(actor: AuthActor = Depends(get_current_actor)) -> AuthActor:
    """FastAPI dependency requiring admin role."""
    if actor.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required for this resource.")
    return actor
