"""Pure-Python PCM WAV audio quality pre-check using Python stdlib only."""

from __future__ import annotations

import io
import math
import struct
import wave
from dataclasses import dataclass

MIN_RMS_DBFS = -45.0
MAX_CLIPPING_RATIO = 0.15
SILENCE_SAMPLE_THRESHOLD = 0.005
CLIP_SAMPLE_THRESHOLD = 0.98


@dataclass(frozen=True)
class AudioQualityReport:
    """Acoustic quality metrics computed without third-party ML/DSP dependencies."""

    valid_wav: bool
    duration_sec: float
    rms_dbfs: float
    silence_ratio: float
    clipping_ratio: float
    passed: bool
    reason: str | None = None


def _decode_pcm_samples(raw_frames: bytes, sample_width: int, channels: int) -> list[float]:
    """Decode signed/unsigned PCM WAV bytes into normalized mono floats in [-1.0, 1.0]."""
    if sample_width <= 0 or channels <= 0:
        return []
    frame_stride = sample_width * channels
    total_frames = len(raw_frames) // frame_stride
    if total_frames == 0:
        return []

    samples: list[float] = []
    if sample_width == 2:
        unpacked = struct.unpack(f"<{total_frames * channels}h", raw_frames[: total_frames * frame_stride])
        for idx in range(0, len(unpacked), channels):
            mono = sum(unpacked[idx : idx + channels]) / channels
            samples.append(mono / 32768.0)
    elif sample_width == 1:
        unpacked = raw_frames[: total_frames * frame_stride]
        for idx in range(0, len(unpacked), channels):
            mono = sum(unpacked[idx : idx + channels]) / channels
            samples.append((mono - 128.0) / 128.0)
    elif sample_width == 4:
        unpacked = struct.unpack(f"<{total_frames * channels}i", raw_frames[: total_frames * frame_stride])
        for idx in range(0, len(unpacked), channels):
            mono = sum(unpacked[idx : idx + channels]) / channels
            samples.append(mono / 2147483648.0)
    return samples


def inspect_pcm_wav_quality(
    wav_bytes: bytes,
    *,
    min_rms_dbfs: float = MIN_RMS_DBFS,
    max_clipping_ratio: float = MAX_CLIPPING_RATIO,
) -> AudioQualityReport:
    """Inspect duration, RMS dBFS, silence ratio, and clipping ratio of a PCM WAV upload."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            num_frames = wf.getnframes()
            if framerate <= 0:
                return AudioQualityReport(False, 0.0, -180.0, 1.0, 0.0, False, "INVALID_WAV")
            raw_frames = wf.readframes(num_frames)
    except Exception:
        return AudioQualityReport(False, 0.0, -180.0, 1.0, 0.0, False, "INVALID_WAV")

    duration_sec = round(num_frames / float(framerate), 3)
    samples = _decode_pcm_samples(raw_frames, sample_width, channels)
    if not samples:
        return AudioQualityReport(True, duration_sec, -180.0, 1.0, 0.0, False, "SILENT_AUDIO")

    total = len(samples)
    sq_sum = 0.0
    silent_count = 0
    clipped_count = 0
    for s in samples:
        sq_sum += s * s
        mag = abs(s)
        if mag < SILENCE_SAMPLE_THRESHOLD:
            silent_count += 1
        if mag >= CLIP_SAMPLE_THRESHOLD:
            clipped_count += 1

    rms = math.sqrt(sq_sum / total)
    rms_dbfs = round(20.0 * math.log10(max(rms, 1e-9)), 2)
    silence_ratio = round(silent_count / total, 4)
    clipping_ratio = round(clipped_count / total, 4)

    if rms_dbfs < min_rms_dbfs:
        return AudioQualityReport(True, duration_sec, rms_dbfs, silence_ratio, clipping_ratio, False, "SILENT_AUDIO")
    if clipping_ratio > max_clipping_ratio:
        return AudioQualityReport(True, duration_sec, rms_dbfs, silence_ratio, clipping_ratio, False, "CLIPPED_AUDIO")

    return AudioQualityReport(True, duration_sec, rms_dbfs, silence_ratio, clipping_ratio, True, None)
