"""Pluggable server-side ASR adapter stub for VoiceSuraksha-Sync.

Disabled by default (ENABLE_SERVER_ASR=false) so Render free-tier (512 MB RAM)
deployments never load heavy local ML models such as faster-whisper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerASRResult:
    """Normalized result returned by the pluggable ServerASRAdapter."""

    enabled: bool
    transcript: str | None
    provider: str
    error: str | None = None


class ServerASRAdapter:
    """Lightweight ASR adapter boundary gated behind ENABLE_SERVER_ASR."""

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = os.getenv("ENABLE_SERVER_ASR", "false").strip().lower() == "true"
        self.enabled = enabled

    def transcribe_wav(self, wav_bytes: bytes) -> ServerASRResult:
        """Transcribe WAV audio when an external server ASR provider is enabled."""
        if not self.enabled:
            return ServerASRResult(enabled=False, transcript=None, provider="disabled")
        if not wav_bytes:
            return ServerASRResult(enabled=True, transcript=None, provider="stub", error="empty_audio")
        # Future remote ASR webhook / API integration hook (no local heavy ML weights)
        return ServerASRResult(
            enabled=True,
            transcript=None,
            provider="remote_stub",
            error="no_remote_asr_endpoint_configured",
        )
