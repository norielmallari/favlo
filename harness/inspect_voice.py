"""Fundamental frequency of a speech clip. de Cheveigné and Kawahara, YIN, 2002.

The period is the shortest lag whose cumulative mean normalized difference
falls under 0.1. A louder harmonic is not reported as the fundamental.
The same samples always print the same numbers.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

WINDOW = 2048
HOP = 512
F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0
THRESHOLD = 0.1
GATE_RMS = 0.01


def fundamentals(samples: np.ndarray, rate: int) -> np.ndarray:
    """Voiced fundamentals in Hz. Silent and unvoiced frames are left out."""
    audio = np.ascontiguousarray(samples, dtype=np.float64).reshape(-1)
    if rate < 1 or audio.size < WINDOW:
        return np.zeros(0, dtype=np.float64)
    found: list[float] = []
    for start in range(0, audio.size - WINDOW + 1, HOP):
        frame = audio[start : start + WINDOW]
        if float(np.sqrt(np.mean(frame * frame))) < GATE_RMS:
            continue
        period = _period(frame, rate)
        if period is not None:
            found.append(rate / period)
    return np.asarray(found, dtype=np.float64)


def report(samples: np.ndarray, rate: int, reference: np.ndarray | None = None) -> dict[str, float]:
    """Median fundamental, its spread, and the semitone gap to ``reference`` when given."""
    pitches = fundamentals(samples, rate)
    summary = _summary(pitches)
    if reference is None:
        return summary
    other = _summary(fundamentals(reference, rate))
    summary["reference_f0_hz"] = other["f0_hz"]
    if summary["f0_hz"] > 0.0 and other["f0_hz"] > 0.0:
        summary["f0_semitones"] = 12.0 * float(np.log2(summary["f0_hz"] / other["f0_hz"]))
    else:
        summary["f0_semitones"] = float("nan")
    return summary


def format_report(path: Path, summary: dict[str, float]) -> str:
    """Stable text. Frequencies are one decimal. The semitone gap is two."""
    lines = [f"file {path.name}", f"f0_hz {summary['f0_hz']:.1f}"]
    lines.append(f"f0_p10_hz {summary['f0_p10_hz']:.1f}")
    lines.append(f"f0_p90_hz {summary['f0_p90_hz']:.1f}")
    lines.append(f"voiced {int(summary['voiced'])}")
    if "reference_f0_hz" in summary:
        lines.append(f"reference_f0_hz {summary['reference_f0_hz']:.1f}")
        gap = summary["f0_semitones"]
        lines.append(f"f0_semitones {gap:.2f}" if np.isfinite(gap) else "f0_semitones none")
    return "\n".join(lines)


def _summary(pitches: np.ndarray) -> dict[str, float]:
    if pitches.size == 0:
        return {"f0_hz": 0.0, "f0_p10_hz": 0.0, "f0_p90_hz": 0.0, "voiced": 0.0}
    return {
        "f0_hz": float(np.median(pitches)),
        "f0_p10_hz": float(np.percentile(pitches, 10)),
        "f0_p90_hz": float(np.percentile(pitches, 90)),
        "voiced": float(pitches.size),
    }


def _period(frame: np.ndarray, rate: int) -> float | None:
    """Lag, in samples, of the first YIN dip under the absolute threshold."""
    lag_min = max(2, int(rate / F0_MAX_HZ))
    lag_max = min(frame.size - 2, int(rate / F0_MIN_HZ))
    if lag_max <= lag_min:
        return None
    difference = _difference(frame, lag_max)
    normalized = _cumulative_mean(difference)
    dips = np.flatnonzero(normalized[lag_min : lag_max + 1] < THRESHOLD)
    if dips.size == 0:
        return None
    index = int(dips[0]) + lag_min
    while index + 1 <= lag_max and normalized[index + 1] < normalized[index]:
        index += 1
    if index <= 0 or index >= lag_max:
        return float(index)
    left, center, right = normalized[index - 1], normalized[index], normalized[index + 1]
    denom = left - 2.0 * center + right
    shift = 0.5 * (left - right) / denom if denom != 0.0 else 0.0
    return float(index) + float(shift)


def _difference(frame: np.ndarray, lag_max: int) -> np.ndarray:
    values = frame.astype(np.float64)
    values -= values.mean()
    width = values.size
    size = 1 << (2 * width - 1).bit_length()
    spectrum = np.fft.rfft(values, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:width]
    cumulative = np.concatenate((np.zeros(1), np.cumsum(values * values)))
    difference = np.empty(lag_max + 1, dtype=np.float64)
    difference[0] = 0.0
    for lag in range(1, lag_max + 1):
        length = width - lag
        left = cumulative[length] - cumulative[0]
        right = cumulative[width] - cumulative[lag]
        difference[lag] = max(0.0, left + right - 2.0 * correlation[lag])
    return difference


def _cumulative_mean(difference: np.ndarray) -> np.ndarray:
    normalized = np.ones_like(difference)
    running = 0.0
    for lag in range(1, difference.size):
        running += float(difference[lag])
        normalized[lag] = difference[lag] * lag / running if running > 0.0 else 1.0
    return normalized


def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2 or channels < 1 or rate < 1 or not frames:
        message = "the wav is not 16-bit PCM"
        raise ValueError(message)
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(pcm), rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the speech fundamental of a wav.")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args()
    samples, rate = _read(args.wav)
    reference = None
    if args.reference is not None:
        reference, ref_rate = _read(args.reference)
        if ref_rate != rate:
            message = "the reference wav uses another sample rate"
            raise ValueError(message)
    print(format_report(args.wav, report(samples, rate, reference)))


if __name__ == "__main__":
    main()
