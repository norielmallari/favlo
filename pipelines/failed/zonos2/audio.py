"""44.1 kHz mono PCM. encode_wav clamps samples outside [-1, 1]."""

from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

import numpy as np

from zonos2.layout import BOS_ID, EOS_ID, LEGACY_SYMBOLS, SAMPLE_RATE, TEXT_VOCAB

_SPEECH_RMS = 0.05
_LOUDNESS_PEAK = 0.99
_SPEECH_FRAME = 882
_SPEECH_GATE = 0.01

_RATE_BUCKETS = 8
_QUALITY_COUNTS = (12, 12, 12, 8, 8, 8)
_BACKGROUND_BUCKETS = 2
_ACCURATE_BUCKETS = 1

# Published 0.2 s silence, 17 aligned frames. The prompt keeps the first 17 delayed rows.
SILENCE_FRAMES = np.array(
    [
        [568, 778, 338, 524, 967, 360, 728, 550, 90],
        [568, 778, 10, 674, 364, 981, 741, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 778, 721, 842, 264, 974, 989, 507, 308],
    ],
    dtype=np.int64,
)


def _conditioning_base() -> int:
    return TEXT_VOCAB - _RATE_BUCKETS - sum(_QUALITY_COUNTS) - _BACKGROUND_BUCKETS - _ACCURATE_BUCKETS


def quality_token(feature: int, bucket: int) -> int:
    """Text row for one quality bucket. Features follow the checkpoint order."""
    return _conditioning_base() + _RATE_BUCKETS + sum(_QUALITY_COUNTS[:feature]) + bucket


def background_token(clean: bool) -> int:
    """Clean background is the first marker. Noisy background is the default."""
    return _conditioning_base() + _RATE_BUCKETS + sum(_QUALITY_COUNTS) + (0 if clean else 1)


def accurate_token() -> int:
    """Accurate mode. Omitting this row is expressive mode."""
    return _conditioning_base() + _RATE_BUCKETS + sum(_QUALITY_COUNTS) + _BACKGROUND_BUCKETS


def text_ids(text: str) -> np.ndarray:
    """BOS, then each UTF-8 byte at row 192 plus that byte, then EOS."""
    if not isinstance(text, str) or not text:
        message = "text is empty"
        raise ValueError(message)
    body = [int(byte) + LEGACY_SYMBOLS for byte in text.encode("utf-8")]
    return np.asarray([BOS_ID, *body, EOS_ID], dtype=np.int64)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """16-bit PCM wav. Stereo is averaged to one channel."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2 or channels < 1 or rate < 1 or not frames:
        message = "the wav is not 16-bit PCM"
        raise ValueError(message)
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), rate


def normalize_loudness(samples: np.ndarray) -> np.ndarray:
    """Scale 20 ms frames at or above 0.01 so their RMS is 0.05. The peak stays at or under 0.99."""
    audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    level = _speech_rms(audio)
    if not np.isfinite(level) or level < 1e-6:
        return audio
    scaled = audio * np.float32(_SPEECH_RMS / level)
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > _LOUDNESS_PEAK:
        scaled = scaled * np.float32(_LOUDNESS_PEAK / peak)
    return np.ascontiguousarray(scaled, dtype=np.float32)


def _speech_rms(audio: np.ndarray) -> float:
    frame = _SPEECH_FRAME
    kept = [
        audio[start : start + frame]
        for start in range(0, max(audio.size - frame + 1, 0), frame)
        if float(np.sqrt(np.mean(audio[start : start + frame] ** 2))) >= _SPEECH_GATE
    ]
    if not kept:
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio * audio)))
    speech = np.concatenate(kept)
    return float(np.sqrt(np.mean(np.square(speech))))


def encode_wav(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.ascontiguousarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()
