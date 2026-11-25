"""In-memory + SQLite-backed rate limiter and login lockout enforcement."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import RateLimitCounter, utcnow

MAX_LOGIN_FAILURES = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60


class RateLimitExceeded(Exception):
    """Raised when a caller/IP/subject exceeds rate limits or hits lockout."""

    def __init__(self, *, retry_after: int, detail: str, bucket: str) -> None:
        super().__init__(detail)
        self.retry_after = max(1, int(retry_after))
        self.detail = detail
        self.bucket = bucket


class GatewayRateLimiter:
    """Thread-safe hybrid in-memory sliding window + SQLite lockout persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, deque[float]] = {}
        self._fail_counts: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}

    def reset(self) -> None:
        """Clear in-memory counters cleanly (useful between isolated tests)."""
        with self._lock:
            self._windows.clear()
            self._fail_counts.clear()
            self._locked_until.clear()

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        return int(raw) if raw.isdigit() and int(raw) > 0 else default

    def check_sliding_limit(
        self,
        db: Session | None,
        *,
        key: str,
        limit: int,
        window_seconds: int,
        detail: str,
    ) -> None:
        """Enforce sliding request rate limit across memory and DB."""
        now_mono = time.time()
        now_dt = utcnow()
        cutoff = now_mono - window_seconds

        with self._lock:
            dq = self._windows.setdefault(key, deque())
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= limit:
                retry_after = max(1, int(dq[0] + window_seconds - now_mono) + 1)
                raise RateLimitExceeded(retry_after=retry_after, detail=detail, bucket=key)
            dq.append(now_mono)

        if db is not None:
            row = db.scalar(select(RateLimitCounter).where(RateLimitCounter.key == key))
            if row is None:
                db.add(
                    RateLimitCounter(
                        key=key,
                        window_start=now_dt,
                        count=1,
                        locked_until=None,
                        updated_at=now_dt,
                    )
                )
            else:
                ws = row.window_start if row.window_start.tzinfo else row.window_start.replace(tzinfo=timezone.utc)
                if (now_dt - ws).total_seconds() >= window_seconds:
                    row.window_start = now_dt
                    row.count = 1
                else:
                    row.count += 1
                row.updated_at = now_dt
            db.flush()

    def ensure_login_allowed(self, db: Session | None, *, ip: str, username: str) -> None:
        """Raise RateLimitExceeded if IP or username is currently locked out."""
        now_mono = time.time()
        now_dt = utcnow()
        for key in (f"login_ip:{ip}", f"login_user:{username.lower()}"):
            with self._lock:
                locked = self._locked_until.get(key)
                if locked and locked > now_mono:
                    raise RateLimitExceeded(
                        retry_after=int(locked - now_mono) + 1,
                        detail="Too many failed login attempts. Account/IP temporarily locked.",
                        bucket=key,
                    )
            if db is not None:
                row = db.scalar(select(RateLimitCounter).where(RateLimitCounter.key == key))
                if row and row.locked_until is not None:
                    lu = (
                        row.locked_until
                        if row.locked_until.tzinfo
                        else row.locked_until.replace(tzinfo=timezone.utc)
                    )
                    if lu > now_dt:
                        retry_after = int((lu - now_dt).total_seconds()) + 1
                        with self._lock:
                            self._locked_until[key] = now_mono + retry_after
                        raise RateLimitExceeded(
                            retry_after=retry_after,
                            detail="Too many failed login attempts. Account/IP temporarily locked.",
                            bucket=key,
                        )

    def record_login_failure(self, db: Session | None, *, ip: str, username: str) -> int:
        """Increment failed login counters for IP and username; lock after threshold."""
        max_fails = self._env_int("RL_MAX_LOGIN_FAILS", MAX_LOGIN_FAILURES)
        lock_secs = self._env_int("RL_LOGIN_LOCK_SECONDS", LOGIN_LOCKOUT_SECONDS)
        now_mono = time.time()
        now_dt = utcnow()
        highest = 0

        for key in (f"login_ip:{ip}", f"login_user:{username.lower()}"):
            with self._lock:
                cnt = self._fail_counts.get(key, 0) + 1
                self._fail_counts[key] = cnt
                if cnt >= max_fails:
                    self._locked_until[key] = now_mono + lock_secs
                highest = max(highest, cnt)

            if db is not None:
                row = db.scalar(select(RateLimitCounter).where(RateLimitCounter.key == key))
                if row is None:
                    locked_until = now_dt + timedelta(seconds=lock_secs) if cnt >= max_fails else None
                    db.add(
                        RateLimitCounter(
                            key=key,
                            window_start=now_dt,
                            count=cnt,
                            locked_until=locked_until,
                            updated_at=now_dt,
                        )
                    )
                else:
                    row.count += 1
                    if row.count >= max_fails:
                        row.locked_until = now_dt + timedelta(seconds=lock_secs)
                    row.updated_at = now_dt
                    highest = max(highest, row.count)
                db.flush()
        return highest

    def clear_login_failures(self, db: Session | None, *, ip: str, username: str) -> None:
        """Reset failure counters on successful authentication."""
        now_dt = utcnow()
        for key in (f"login_ip:{ip}", f"login_user:{username.lower()}"):
            with self._lock:
                self._fail_counts.pop(key, None)
                self._locked_until.pop(key, None)
            if db is not None:
                row = db.scalar(select(RateLimitCounter).where(RateLimitCounter.key == key))
                if row is not None:
                    row.count = 0
                    row.locked_until = None
                    row.updated_at = now_dt
                db.flush()


gateway_rate_limiter = GatewayRateLimiter()
