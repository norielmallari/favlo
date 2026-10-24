"""No-reference speech check for a generated clip.

Silence, white noise, and a stuck tone each fail a different signal. The
thresholds follow the ranges used for unreferenced TTS checks: WebRTC's
speech band is 80 Hz to 4 kHz, speech spectral flatness sits between a pure
tone and noise, and a sentence moves its pitch.
"""

from __future__ import annotations

import numpy as np

from zonos2.layout import SAMPLE_RATE

_WINDOW = 2048
_HOP = 512
_F0_MIN = 70.0
_F0_MAX = 400.0
_VOICING = 0.45
_BAND_LOW = 80.0
_BAND_HIGH = 4000.0


def voice_report(samples: np.ndarray, rate: int = SAMPLE_RATE) -> dict[str, float]:
    """Frame-level speech signals. Durations are seconds and ratios are fractions."""
    audio = np.ascontiguousarray(samples, dtype=np.float64).reshape(-1)
    seconds = audio.size / rate if rate else 0.0
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    clip_fraction = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 1.0
    flatness, voiced, pitches, band = _frames(audio, rate)
    voiced_fraction = float(np.mean(voiced)) if voiced.size else 0.0
    median_flatness = float(np.median(flatness)) if flatness.size else 0.0
    band_ratio = float(np.median(band)) if band.size else 0.0
    if pitches.size >= 2:
        semis = 12.0 * np.log2(pitches / np.median(pitches))
        pitch_span = float(np.max(semis) - np.min(semis))
    else:
        pitch_span = 0.0
    return {
        "seconds": seconds,
        "rms": rms,
        "clip_fraction": clip_fraction,
        "frames": float(flatness.size),
        "flatness": median_flatness,
        "voiced": voiced_fraction,
        "pitched": float(pitches.size),
        "pitch_span": pitch_span,
        "band": band_ratio,
    }


def voice_ok(samples: np.ndarray, rate: int = SAMPLE_RATE) -> tuple[bool, dict[str, float]]:
    """True when the clip has level, speech-band energy, harmonic flatness, and a moving pitch."""
    report = voice_report(samples, rate)
    ok = (
        0.75 <= report["seconds"] <= 12.0
        and report["rms"] >= 0.02
        and report["clip_fraction"] < 0.001
        and report["frames"] >= 8
        and 0.03 <= report["flatness"] <= 0.55
        and report["voiced"] >= 0.25
        and report["pitched"] >= 8
        and report["pitch_span"] >= 0.5
        and report["band"] >= 0.55
    )
    return ok, report


def _frames(audio: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if audio.size < _WINDOW:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty
    window = np.hanning(_WINDOW)
    lag_min = max(1, int(rate / _F0_MAX))
    lag_max = min(_WINDOW - 1, int(rate / _F0_MIN))
    freqs = np.fft.rfftfreq(_WINDOW, d=1.0 / rate)
    band = (freqs >= _BAND_LOW) & (freqs <= _BAND_HIGH)
    flatness: list[float] = []
    voiced: list[bool] = []
    pitches: list[float] = []
    bands: list[float] = []
    for start in range(0, audio.size - _WINDOW + 1, _HOP):
        frame = audio[start : start + _WINDOW]
        if float(np.sqrt(np.mean(frame * frame))) < 0.01:
            continue
        shaped = frame * window
        spectrum = np.fft.rfft(shaped)
        power = np.abs(spectrum) ** 2
        total = float(power.sum())
        if total <= 0.0:
            continue
        speech = (freqs >= _BAND_LOW) & (freqs <= _BAND_HIGH)
        magnitude = np.maximum(np.abs(spectrum[speech]), 1e-12)
        flatness.append(float(np.exp(np.mean(np.log(magnitude))) / np.mean(magnitude)))
        bands.append(float(power[band].sum() / total))
        pitch = _pitch(frame, rate, lag_min, lag_max)
        voiced.append(pitch is not None)
        if pitch is not None:
            pitches.append(pitch)
    return (
        np.asarray(flatness, dtype=np.float64),
        np.asarray(voiced, dtype=bool),
        np.asarray(pitches, dtype=np.float64),
        np.asarray(bands, dtype=np.float64),
    )


def _pitch(frame: np.ndarray, rate: int, lag_min: int, lag_max: int) -> float | None:
    centered = frame - np.mean(frame)
    energy = float(np.dot(centered, centered))
    if energy <= 1e-8 or lag_max <= lag_min:
        return None
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    segment = corr[lag_min : lag_max + 1]
    peak = int(np.argmax(segment))
    strength = float(segment[peak] / corr[0])
    if strength < _VOICING:
        return None
    return rate / float(lag_min + peak)
