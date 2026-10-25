"""16-bit stereo WAV. The samples are already peak-normalized."""

from __future__ import annotations

import struct

import numpy as np

from ace_step_15_turbo.schedule import SAMPLE_RATE


def wav_bytes(channel_first: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    """Interleave ``[2, samples]`` float audio into a RIFF WAV."""
    audio = np.ascontiguousarray(channel_first, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != 2:
        message = "audio must be stereo channel-first"
        raise ValueError(message)
    interleaved = np.empty((audio.shape[1], 2), dtype=np.float32)
    interleaved[:, 0] = audio[0]
    interleaved[:, 1] = audio[1]
    pcm = np.clip(np.rint(interleaved * 32767.0), -32768, 32767).astype("<i2")
    data = pcm.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        2,
        rate,
        rate * 4,
        4,
        16,
        b"data",
        len(data),
    )
    return header + data


def wav_info(payload: bytes) -> tuple[int, int, int]:
    """Return sample rate, channel count, and frame count."""
    if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        message = "not a wav file"
        raise ValueError(message)
    rate, channels = struct.unpack_from("<IH", payload, 24)[0], struct.unpack_from("<H", payload, 22)[0]
    data = payload[44:]
    frames = len(data) // (channels * 2)
    return rate, channels, frames
