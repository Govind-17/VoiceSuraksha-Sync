"""NumPy perceptual spectral audio fingerprinting & overlapping WAV segment splitter.

Zero raw audio is stored; only compact 64-bit spectral hashes are persisted.
"""

from __future__ import annotations

import io
import wave

import numpy as np

MAX_PERCEPTUAL_HAMMING_BITS = 8


def _read_pcm_mono(wav_bytes: bytes) -> tuple[np.ndarray, int, int, int] | None:
    """Decode PCM WAV bytes into float64 mono array plus (channels, sampwidth, framerate)."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            num_frames = wf.getnframes()
            if framerate <= 0 or channels <= 0 or num_frames <= 0:
                return None
            raw = wf.readframes(num_frames)
    except Exception:
        return None

    if sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        return None

    if channels > 1:
        usable = (arr.size // channels) * channels
        arr = arr[:usable].reshape(-1, channels).mean(axis=1)
    return arr, channels, sampwidth, framerate


def compute_perceptual_hash(wav_bytes: bytes) -> str | None:
    """Compute a 64-bit (16-hex-char) gain/re-encode-tolerant spectral energy delta fingerprint."""
    decoded = _read_pcm_mono(wav_bytes)
    if decoded is None:
        return None
    samples, _, _, framerate = decoded
    if samples.size < 512:
        return None

    # Peak/RMS normalize so volume scaling cancels cleanly
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms < 1e-6:
        return None
    norm = samples / rms

    # Trim very quiet leading/trailing tails (>5% of peak) to tolerate minor start/end trimming
    thresh = 0.05 * float(np.max(np.abs(norm)))
    active_idx = np.where(np.abs(norm) >= thresh)[0]
    if active_idx.size >= 512:
        norm = norm[active_idx[0] : active_idx[-1] + 1]

    # Partition active speech into 9 overlapping time windows x 9 log-spaced frequency bands -> 8x8=64 bits
    n_windows = 9
    n_bands = 9
    win_edges = np.linspace(0, norm.size, n_windows + 1, dtype=int)
    spec_energies = np.zeros((n_windows, n_bands), dtype=np.float64)

    for t in range(n_windows):
        seg = norm[win_edges[t] : win_edges[t + 1]]
        if seg.size < 64:
            seg = np.pad(seg, (0, 64 - seg.size))
        windowed = seg * np.hanning(seg.size)
        mag2 = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(seg.size, d=1.0 / framerate)
        band_edges = np.geomspace(120.0, min(framerate * 0.45, 7000.0), n_bands + 1)
        for b in range(n_bands):
            mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
            band_power = float(np.mean(mag2[mask])) if np.any(mask) else 1e-12
            spec_energies[t, b] = np.log(max(band_power, 1e-12))

    # 2D spectral-temporal sign differences -> 8 * 8 = 64 bits
    bits = 0
    for t in range(n_windows - 1):
        for b in range(n_bands - 1):
            diff = (spec_energies[t + 1, b] - spec_energies[t, b]) - (
                spec_energies[t + 1, b + 1] - spec_energies[t, b + 1]
            )
            bits = (bits << 1) | (1 if diff > 0.0 else 0)
    return f"{bits:016x}"


def hamming_distance_hex(hash_a: str, hash_b: str) -> int:
    """Return Hamming distance in bits between two 16-char hex perceptual hashes."""
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def split_wav_segments(wav_bytes: bytes, *, min_segment_sec: float = 4.0) -> list[bytes]:
    """Split WAV >= 8s into 2-3 overlapping PCM WAV slices (>=4s each) for multi-segment checking."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except Exception:
        return [wav_bytes]

    if framerate <= 0 or nframes <= 0:
        return [wav_bytes]

    duration_sec = nframes / float(framerate)
    if duration_sec < (min_segment_sec * 2.0):
        return [wav_bytes]

    num_segments = 3 if duration_sec >= 11.0 else 2
    seg_frames = max(int(min_segment_sec * framerate), int((duration_sec * 0.6) * framerate))
    seg_frames = min(seg_frames, nframes)
    stride = channels * sampwidth
    max_start = max(0, nframes - seg_frames)
    starts = [int(round(i * max_start / max(1, num_segments - 1))) for i in range(num_segments)]

    segments: list[bytes] = []
    for st in starts:
        chunk_bytes = raw[st * stride : (st + seg_frames) * stride]
        out = io.BytesIO()
        with wave.open(out, "wb") as out_wf:
            out_wf.setnchannels(channels)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(framerate)
            out_wf.writeframes(chunk_bytes)
        segments.append(out.getvalue())
    return segments
