"""24 kHz log-mel. Slaney scale, periodic Hann, reflect pad, then log clamp."""

from __future__ import annotations

import math

import numpy as np

RATE = 24000
FFT = 1024
HOP = 256
BINS = 128
FMIN = 0.0
FMAX = 12000.0
_CLAMP = 1e-5
_PAD = (FFT - HOP) // 2


def log_mel(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono samples become ``[frames, 128]`` log-mel features."""
    audio = _resample(np.ascontiguousarray(wave, dtype=np.float64), sample_rate, RATE)
    if audio.size <= _PAD:
        message = "the clip is shorter than the encoder window"
        raise ValueError(message)
    padded = np.pad(audio, (_PAD, _PAD), mode="reflect")
    window = 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(FFT) / FFT)
    frames = 1 + (padded.size - FFT) // HOP
    spectrum = np.empty((FFT // 2 + 1, frames), dtype=np.float64)
    for index in range(frames):
        start = index * HOP
        spectrum[:, index] = np.abs(np.fft.rfft(padded[start : start + FFT] * window))
    power = mel_filterbank() @ spectrum
    return np.log(np.maximum(power, _CLAMP)).astype(np.float32).T


def mel_filterbank() -> np.ndarray:
    """Slaney triangles, shape ``[128, 513]``, normalized by the band width."""
    points = BINS + 2
    low = _hz_to_mel(FMIN)
    high = _hz_to_mel(FMAX)
    hertz = _mel_to_hz(np.linspace(low, high, points))
    bins = np.floor((FFT + 1) * hertz / RATE).astype(np.int64)
    weights = np.zeros((BINS, FFT // 2 + 1), dtype=np.float64)
    for index in range(BINS):
        left, center, right = bins[index], bins[index + 1], bins[index + 2]
        if center <= left or right <= center:
            continue
        rising = np.arange(left, center)
        falling = np.arange(center, right)
        weights[index, left:center] = (rising - left) / (center - left)
        weights[index, center:right] = (right - falling) / (right - center)
    width = hertz[2:] - hertz[:-2]
    scale = np.where(width > 0.0, 2.0 / width, 0.0)
    return weights * scale[:, None]


def _hz_to_mel(hertz: float | np.ndarray) -> np.ndarray:
    values = np.asarray(hertz, dtype=np.float64)
    spacing = 200.0 / 3.0
    corner = 1000.0
    linear = corner / spacing
    step = math.log(6.4) / 27.0
    mel = values / spacing
    bent = values >= corner
    logged = np.array(mel, copy=True)
    logged[bent] = linear + np.log(values[bent] / corner) / step
    return logged


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    spacing = 200.0 / 3.0
    corner = 1000.0
    linear = corner / spacing
    step = math.log(6.4) / 27.0
    hertz = mel * spacing
    bent = mel >= linear
    return np.where(bent, corner * np.exp(step * (mel - linear)), hertz)


def _resample(wave: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return wave
    if source < 1 or target < 1 or wave.size < 2:
        message = "the clip has no samples to resample"
        raise ValueError(message)
    count = int(round(wave.size * target / source))
    spectrum = np.fft.rfft(wave)
    resized = np.zeros(count // 2 + 1, dtype=np.complex128)
    shared = min(spectrum.size, resized.size)
    resized[:shared] = spectrum[:shared]
    output = np.fft.irfft(resized, count)
    return output * (count / wave.size)
