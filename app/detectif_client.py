"""Server-side DetectifAI API client for Eidos and Aletheia.

Loads DETECTIF_API_KEY from environment/.env only and never exposes it.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(override=False)


class DetectifProviderError(Exception):
    """Structured error raised when DetectifAI fails or returns a non-200 status."""

    def __init__(
        self,
        *,
        status_code: int | None,
        error_code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.request_id = request_id


@dataclass(frozen=True)
class ProviderEnrollResult:
    """Parsed fields from POST /v1/speakers/enroll."""

    request_id: str
    operation_id: str
    speaker_id: str
    speech_seconds: float
    duration_seconds: float
    samples: int
    model_name: str | None = None
    model_version: str | None = None
    credits_charged: int | None = None
    credits_balance: int | None = None
    latency_ms: float | None = None


@dataclass(frozen=True)
class ProviderVerifyResult:
    """Parsed fields from POST /v1/speakers/verify (with detect_deepfake=true)."""

    request_id: str
    operation_id: str
    speaker_id: str
    speaker_decision: str
    speaker_similarity: float
    speaker_threshold: float
    duration_seconds: float
    deepfake_verdict: str | None
    deepfake_is_synthetic: bool | None
    deepfake_score: float | None
    deepfake_threshold: float | None
    deepfake_segments: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float | None = None
    credits_balance: int | None = None


@dataclass(frozen=True)
class ProviderDeepfakeResult:
    """Parsed fields from POST /v1/deepfake/detect."""

    request_id: str
    operation_id: str
    verdict: str
    is_synthetic: bool
    score: float
    threshold: float
    duration_seconds: float
    segments: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float | None = None
    credits_balance: int | None = None


class DetectifClient:
    """HTTP client wrapper for DetectifAI Eidos and Aletheia endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        load_dotenv(override=False)
        self._api_key = api_key if api_key is not None else os.getenv("DETECTIF_API_KEY", "")
        self._base_url = (base_url or os.getenv("DETECTIF_BASE_URL", "https://platform.detectif.ai")).rstrip("/")
        env_timeout = os.getenv("DETECTIF_TIMEOUT_SECONDS")
        self._timeout = float(env_timeout) if env_timeout and timeout_seconds == 30.0 else timeout_seconds
        self._transport = transport

    def _build_client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {"base_url": self._base_url, "timeout": self._timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _post_audio(
        self,
        path: str,
        *,
        filename: str,
        wav_bytes: bytes,
        data: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self._api_key:
            raise DetectifProviderError(
                status_code=401,
                error_code="missing_api_key",
                message="DETECTIF_API_KEY is not configured on the server.",
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Idempotency-Key": str(uuid.uuid4()),
        }
        files = {"file": (filename, wav_bytes, "audio/wav")}

        try:
            with self._build_client() as client:
                response = client.post(path, headers=headers, data=data or {}, files=files)
        except httpx.TimeoutException as exc:
            raise DetectifProviderError(
                status_code=None,
                error_code="provider_timeout",
                message=f"DetectifAI request timed out: {exc}",
            ) from exc
        except httpx.HTTPError as exc:
            raise DetectifProviderError(
                status_code=None,
                error_code="provider_transport_error",
                message=f"DetectifAI transport failure: {exc}",
            ) from exc

        req_id = response.headers.get("X-Request-Id")
        if response.status_code != 200:
            error_code = f"http_{response.status_code}"
            message = f"DetectifAI returned HTTP {response.status_code}."
            try:
                err_json = response.json()
                if isinstance(err_json, dict):
                    err_obj = err_json.get("error")
                    if isinstance(err_obj, dict):
                        error_code = str(err_obj.get("code") or error_code)
                        message = str(err_obj.get("message") or message)
                    elif err_json.get("detail"):
                        message = str(err_json["detail"])
                    req_id = err_json.get("request_id") or req_id
            except ValueError:
                pass
            raise DetectifProviderError(
                status_code=response.status_code,
                error_code=error_code,
                message=message,
                request_id=req_id,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise DetectifProviderError(
                status_code=response.status_code,
                error_code="invalid_json",
                message="DetectifAI returned invalid JSON.",
                request_id=req_id,
            ) from exc

        if not isinstance(payload, dict):
            raise DetectifProviderError(
                status_code=response.status_code,
                error_code="invalid_payload",
                message="DetectifAI response payload was not a JSON object.",
                request_id=req_id,
            )
        return payload

    def enroll_speaker(self, speaker_id: str, wav_bytes: bytes) -> ProviderEnrollResult:
        """Call POST /v1/speakers/enroll and parse verified response fields."""
        payload = self._post_audio(
            "/v1/speakers/enroll",
            filename="enroll.wav",
            wav_bytes=wav_bytes,
            data={"speaker_id": speaker_id},
        )
        speaker = payload.get("speaker") or {}
        quality = payload.get("quality") or {}
        model = payload.get("model") or {}
        credits = payload.get("credits") or {}

        if not payload.get("request_id") or not payload.get("operation_id") or not speaker.get("speaker_id"):
            raise DetectifProviderError(
                status_code=200,
                error_code="incomplete_enroll_response",
                message="Missing expected fields in Eidos enroll response.",
                request_id=payload.get("request_id"),
            )

        return ProviderEnrollResult(
            request_id=str(payload["request_id"]),
            operation_id=str(payload["operation_id"]),
            speaker_id=str(speaker["speaker_id"]),
            speech_seconds=float(quality.get("speech_seconds", 0.0)),
            duration_seconds=float(quality.get("duration_seconds", 0.0)),
            samples=int(speaker.get("samples", 1)),
            model_name=model.get("name"),
            model_version=model.get("version"),
            credits_charged=credits.get("charged"),
            credits_balance=credits.get("balance"),
            latency_ms=payload.get("latency_ms"),
        )

    def verify_with_deepfake(self, speaker_id: str, wav_bytes: bytes) -> ProviderVerifyResult:
        """Call POST /v1/speakers/verify with detect_deepfake=true and parse verified fields."""
        payload = self._post_audio(
            "/v1/speakers/verify",
            filename="verify.wav",
            wav_bytes=wav_bytes,
            data={"speaker_id": speaker_id, "detect_deepfake": "true"},
        )
        result = payload.get("result") or {}
        deepfake = payload.get("deepfake")
        credits = payload.get("credits") or {}

        decision = result.get("decision")
        if decision not in {"match", "no_match"}:
            raise DetectifProviderError(
                status_code=200,
                error_code="incomplete_verify_response",
                message="Missing valid result.decision in Eidos verify response.",
                request_id=payload.get("request_id"),
            )

        deepfake_verdict: str | None = None
        deepfake_is_synthetic: bool | None = None
        deepfake_score: float | None = None
        deepfake_threshold: float | None = None
        deepfake_segments: list[dict[str, Any]] = []

        if isinstance(deepfake, dict) and deepfake.get("verdict") in {"real", "fake"}:
            deepfake_verdict = str(deepfake["verdict"])
            deepfake_is_synthetic = bool(deepfake["is_synthetic"])
            deepfake_score = float(deepfake["score"])
            deepfake_threshold = float(deepfake["threshold"])
            raw_segments = deepfake.get("segments")
            if isinstance(raw_segments, list):
                deepfake_segments = raw_segments

        return ProviderVerifyResult(
            request_id=str(payload["request_id"]),
            operation_id=str(payload["operation_id"]),
            speaker_id=str(payload.get("speaker_id") or speaker_id),
            speaker_decision=str(decision),
            speaker_similarity=float(result["similarity"]),
            speaker_threshold=float(result["threshold"]),
            duration_seconds=float(result.get("duration_seconds", 0.0)),
            deepfake_verdict=deepfake_verdict,
            deepfake_is_synthetic=deepfake_is_synthetic,
            deepfake_score=deepfake_score,
            deepfake_threshold=deepfake_threshold,
            deepfake_segments=deepfake_segments,
            latency_ms=payload.get("latency_ms"),
            credits_balance=credits.get("balance"),
        )

    def detect_deepfake(self, wav_bytes: bytes) -> ProviderDeepfakeResult:
        """Call standalone POST /v1/deepfake/detect (Aletheia) and parse verified fields."""
        payload = self._post_audio(
            "/v1/deepfake/detect",
            filename="detect.wav",
            wav_bytes=wav_bytes,
        )
        result = payload.get("result") or {}
        credits = payload.get("credits") or {}
        verdict = result.get("verdict")
        if verdict not in {"real", "fake"}:
            raise DetectifProviderError(
                status_code=200,
                error_code="incomplete_deepfake_response",
                message="Missing valid result.verdict in Aletheia detect response.",
                request_id=payload.get("request_id"),
            )

        return ProviderDeepfakeResult(
            request_id=str(payload["request_id"]),
            operation_id=str(payload["operation_id"]),
            verdict=str(verdict),
            is_synthetic=bool(result["is_synthetic"]),
            score=float(result["score"]),
            threshold=float(result["threshold"]),
            duration_seconds=float(result.get("duration_seconds", 0.0)),
            segments=list(result.get("segments") or []),
            latency_ms=payload.get("latency_ms"),
            credits_balance=credits.get("balance"),
        )
