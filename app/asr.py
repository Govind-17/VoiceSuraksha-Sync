"""Provider-agnostic hosted ASR adapter for server-side challenge verification.

Uses lightweight REST calls via httpx (no local Whisper / torch weights) so
VoiceSuraksha-Sync stays well under Render free-tier 512 MB memory limits.
Default provider is Sarvam AI (supports en-IN, hi-IN, kn-IN) with pluggable
support for Deepgram, Groq Whisper, or a custom REST ASR endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

_DEFAULT_ENDPOINTS: dict[str, str] = {
    "sarvam": "https://api.sarvam.ai/speech-to-text",
    "deepgram": "https://api.deepgram.com/v1/listen",
    "groq": "https://api.groq.com/openai/v1/audio/transcriptions",
}


@dataclass(frozen=True)
class ServerASRResult:
    """Normalized result returned by ServerASRAdapter."""

    available: bool
    transcript: str | None
    provider: str
    language_code: str | None = None
    error: str | None = None


class ServerASRAdapter:
    """Provider-agnostic hosted ASR client configured via ASR_PROVIDER and ASR_API_KEY."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        language_code: str | None = None,
        timeout_seconds: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.provider = (
            provider if provider is not None else os.getenv("ASR_PROVIDER", "sarvam")
        ).strip().lower() or "sarvam"
        self.api_key = (api_key if api_key is not None else os.getenv("ASR_API_KEY", "")).strip()
        default_ep = _DEFAULT_ENDPOINTS.get(self.provider, "")
        self.endpoint = (endpoint if endpoint is not None else os.getenv("ASR_ENDPOINT", default_ep)).strip()
        self.language_code = (
            language_code if language_code is not None else os.getenv("ASR_LANGUAGE_CODE", "en-IN")
        ).strip() or "en-IN"
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def _build_client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def transcribe_wav(self, wav_bytes: bytes) -> ServerASRResult:
        """Transcribe WAV bytes via the configured hosted ASR provider."""
        if self.provider == "disabled" or not self.api_key:
            return ServerASRResult(
                available=False,
                transcript=None,
                provider=self.provider,
                language_code=self.language_code,
                error="asr_not_configured",
            )
        if not self.endpoint:
            return ServerASRResult(
                available=False,
                transcript=None,
                provider=self.provider,
                language_code=self.language_code,
                error="asr_endpoint_missing",
            )
        if not wav_bytes:
            return ServerASRResult(
                available=False,
                transcript=None,
                provider=self.provider,
                language_code=self.language_code,
                error="empty_audio",
            )

        try:
            with self._build_client() as client:
                if self.provider == "deepgram":
                    resp = client.post(
                        self.endpoint,
                        headers={
                            "Authorization": f"Token {self.api_key}",
                            "Content-Type": "audio/wav",
                        },
                        params={"model": "nova-2", "language": self.language_code.split("-")[0]},
                        content=wav_bytes,
                    )
                elif self.provider == "sarvam":
                    resp = client.post(
                        self.endpoint,
                        headers={"api-subscription-key": self.api_key},
                        data={"model": "saarika:v2.5", "language_code": self.language_code},
                        files={"file": ("challenge.wav", wav_bytes, "audio/wav")},
                    )
                elif self.provider == "groq":
                    resp = client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data={"model": "whisper-large-v3-turbo", "language": self.language_code.split("-")[0]},
                        files={"file": ("challenge.wav", wav_bytes, "audio/wav")},
                    )
                else:
                    resp = client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data={"language_code": self.language_code},
                        files={"file": ("challenge.wav", wav_bytes, "audio/wav")},
                    )
        except httpx.TimeoutException:
            return ServerASRResult(
                available=False,
                transcript=None,
                provider=self.provider,
                language_code=self.language_code,
                error="asr_timeout",
            )
        except httpx.HTTPError:
            return ServerASRResult(
                available=False,
                transcript=None,
                provider=self.provider,
                language_code=self.language_code,
                error="asr_transport_error",
            )

        if resp.status_code != 200:
            return ServerASRResult(
                available=False,
                transcript=None,
                provider=self.provider,
                language_code=self.language_code,
                error=f"asr_http_{resp.status_code}",
            )

        try:
            payload = resp.json()
        except ValueError:
            return ServerASRResult(
                available=False,
                transcript=None,
                provider=self.provider,
                language_code=self.language_code,
                error="asr_invalid_json",
            )

        transcript: str | None = None
        if isinstance(payload, dict):
            if "transcript" in payload and payload["transcript"] is not None:
                transcript = str(payload["transcript"])
            elif "text" in payload and payload["text"] is not None:
                transcript = str(payload["text"])
            elif isinstance(payload.get("results"), dict):
                channels = payload["results"].get("channels") or []
                if channels and isinstance(channels[0], dict):
                    alts = channels[0].get("alternatives") or []
                    if alts and isinstance(alts[0], dict):
                        transcript = str(alts[0].get("transcript") or "")

        if transcript is None:
            return ServerASRResult(
                available=False,
                transcript=None,
                provider=self.provider,
                language_code=self.language_code,
                error="asr_missing_transcript",
            )

        return ServerASRResult(
            available=True,
            transcript=transcript.strip(),
            provider=self.provider,
            language_code=self.language_code,
            error=None,
        )
