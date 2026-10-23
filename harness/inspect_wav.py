"""Turn a WAV into a spectrogram PNG and a deterministic text report.

Numbers are rounded. The STFT uses a fixed Hann window, so the same file
always prints the same report.
"""

from __future__ import annotations

import argparse
import struct
import wave
import zlib
from pathlib import Path

import numpy as np

WINDOW = 2048
HOP = 512
FREQ_BINS = 64
TIME_SLICES = 20
IMAGE_FREQ = 256
IMAGE_TIME = 800
DB_RANGE = 80.0
PITCH_WINDOW = 16384
PITCH_HOP = 4096
PITCH_LOW_HZ = 70.0
PITCH_HIGH_HZ = 2000.0
F0_HIGH_HZ = 800.0
CLASS_SHARE = 0.08
CLASH_SHARE = 0.12
SLICE_RMS = 0.001
BPM_LOW = 60.0
BPM_HIGH = 210.0
BPM_PEAK = 0.12
NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Krumhansl-Kessler probe-tone weights. Index 0 is the tonic.
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


def inspect_file(
    wav: Path,
    png: Path | None = None,
    *,
    keyscale: str | None = None,
    bpm: float | None = None,
    reference: Path | None = None,
) -> str:
    """Write the spectrogram and return the deterministic report."""
    audio, rate = _read(wav)
    image_path = png if png is not None else wav.with_name(wav.stem + ".inspect.png")
    report, image = _inspect(audio, rate)
    _write_png(image_path, image)
    extra = _request_lines(report, audio.mean(axis=1), rate, keyscale=keyscale, bpm=bpm, reference=reference)
    body = report + extra + _output_template(report) + f"spectrogram {image_path.name}\n"
    return body


def _output_template(report: str) -> str:
    try:
        from harness.music_template import output_template
    except ImportError:
        import sys

        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from harness.music_template import output_template

    return output_template(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a WAV file for an agent")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--keyscale", default=None)
    parser.add_argument("--bpm", type=float, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args()
    print(
        inspect_file(
            args.wav,
            args.png,
            keyscale=args.keyscale,
            bpm=args.bpm,
            reference=args.reference,
        ),
        end="",
    )


def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.getnframes()
        payload = handle.readframes(frames)
    if width != 2:
        message = "only 16-bit PCM is supported"
        raise ValueError(message)
    pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32).reshape(frames, channels)
    return pcm / 32768.0, rate


def _inspect(audio: np.ndarray, rate: int) -> tuple[str, np.ndarray]:
    mono = audio.mean(axis=1)
    seconds = audio.shape[0] / rate
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
    clipped = float(np.mean(np.max(np.abs(audio), axis=1) >= 32767.0 / 32768.0)) if audio.size else 0.0
    correlation = _correlation(audio)
    flatness, centroid, grid = _spectrum(mono, rate)
    image = _image(mono, rate)
    lines = [
        f"rate {rate}",
        f"channels {audio.shape[1]}",
        f"seconds {_fmt(seconds)}",
        f"rms {_fmt(rms)}",
        f"peak {_fmt(peak)}",
        f"clip_fraction {_fmt(clipped)}",
        f"channel_correlation {_fmt(correlation)}",
        f"spectral_flatness {_fmt(flatness)}",
        f"spectral_centroid_hz {_fmt(centroid)}",
        "band_hz " + " ".join(str(edge) for edge in _edges(rate)),
        "grid_db time rows, frequency columns, 0 dB is the peak",
    ]
    for row in grid.T:
        lines.append(" ".join(str(int(value)) for value in row))
    envelope = _envelope(mono, rate)
    lines.append("half_second_rms " + " ".join(_fmt(value) for value in envelope))
    lines.extend(_music(mono, rate))
    return "\n".join(lines) + "\n", image


def _spectrum(mono: np.ndarray, rate: int) -> tuple[float, float, np.ndarray]:
    if mono.size < WINDOW:
        padded = np.zeros(WINDOW, dtype=np.float32)
        padded[: mono.size] = mono
        mono = padded
    window = np.hanning(WINDOW).astype(np.float32)
    count = 1 + (mono.size - WINDOW) // HOP
    bins = WINDOW // 2 + 1
    power = np.empty((bins, count), dtype=np.float64)
    for index in range(count):
        start = index * HOP
        frame = mono[start : start + WINDOW] * window
        power[:, index] = np.abs(np.fft.rfft(frame)) ** 2
    total = float(power.mean()) + 1e-20
    flatness = float(np.exp(np.mean(np.log(power + 1e-20))) / total)
    freqs = np.fft.rfftfreq(WINDOW, 1.0 / rate)
    centroid = float(np.sum(freqs * power.mean(axis=1)) / (power.mean(axis=1).sum() + 1e-20))
    reduced = _reduce(power, FREQ_BINS, TIME_SLICES, rate)
    peak = float(reduced.max()) + 1e-20
    db = np.clip(10.0 * np.log10(reduced / peak + 1e-20), -DB_RANGE, 0.0)
    return flatness, centroid, np.rint(db).astype(np.int16)


def _image(mono: np.ndarray, rate: int) -> np.ndarray:
    if mono.size < WINDOW:
        padded = np.zeros(WINDOW, dtype=np.float32)
        padded[: mono.size] = mono
        mono = padded
    window = np.hanning(WINDOW).astype(np.float32)
    count = 1 + (mono.size - WINDOW) // HOP
    power = np.empty((WINDOW // 2 + 1, count), dtype=np.float64)
    for index in range(count):
        start = index * HOP
        power[:, index] = np.abs(np.fft.rfft(mono[start : start + WINDOW] * window)) ** 2
    reduced = _reduce(power, IMAGE_FREQ, min(IMAGE_TIME, count), rate)
    peak = float(reduced.max()) + 1e-20
    db = np.clip(10.0 * np.log10(reduced / peak + 1e-20), -DB_RANGE, 0.0)
    gray = np.rint((db + DB_RANGE) / DB_RANGE * 255.0).astype(np.uint8)
    return gray[::-1]


def _reduce(power: np.ndarray, freq_bins: int, time_slices: int, rate: int) -> np.ndarray:
    edges = _edges(rate, freq_bins)
    freqs = np.fft.rfftfreq(WINDOW, 1.0 / rate)
    bands = np.zeros((freq_bins, power.shape[1]), dtype=np.float64)
    for index in range(freq_bins):
        high = freqs <= edges[index + 1] if index == freq_bins - 1 else freqs < edges[index + 1]
        chosen = (freqs >= edges[index]) & high
        if not np.any(chosen):
            continue
        bands[index] = power[chosen].mean(axis=0)
    slices = np.zeros((freq_bins, time_slices), dtype=np.float64)
    width = bands.shape[1]
    for index in range(time_slices):
        start = index * width // time_slices
        stop = max(start + 1, (index + 1) * width // time_slices)
        slices[:, index] = bands[:, start:stop].mean(axis=1)
    return slices


def _edges(rate: int, freq_bins: int = FREQ_BINS) -> list[int]:
    nyquist = rate / 2.0
    return [int(round(nyquist * index / freq_bins)) for index in range(freq_bins + 1)]


def _envelope(mono: np.ndarray, rate: int) -> list[float]:
    hop = max(1, rate // 2)
    values = []
    for start in range(0, mono.size, hop):
        chunk = mono[start : start + hop]
        values.append(float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0)
    return values


def _correlation(audio: np.ndarray) -> float:
    if audio.shape[1] < 2:
        return 1.0
    left = audio[:, 0] - float(audio[:, 0].mean())
    right = audio[:, 1] - float(audio[:, 1].mean())
    denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(left * right) / denom)


def _music(mono: np.ndarray, rate: int) -> list[str]:
    """Key, pitch, and harmony for each half-second that is above the quiet floor."""
    hop = max(1, rate // 2)
    window = np.hanning(PITCH_WINDOW).astype(np.float64)
    times: list[float] = []
    chromas: list[np.ndarray | None] = []
    pitches: list[tuple[float, str, int] | None] = []
    total = np.zeros(12, dtype=np.float64)
    for index, start in enumerate(range(0, mono.size, hop)):
        chunk = mono[start : start + hop]
        level = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        times.append(index * hop / rate)
        if level < SLICE_RMS or chunk.size < 8:
            chromas.append(None)
            pitches.append(None)
            continue
        chroma, f0 = _slice_tones(chunk.astype(np.float64), rate, window)
        if float(chroma.sum()) <= 0.0 or f0 is None:
            chromas.append(None)
            pitches.append(None)
            continue
        total += chroma
        chromas.append(chroma)
        pitches.append(f0)
    tonic = _best_key(total)[0]
    on_tonic = 0
    clashes = 0
    sounded = 0
    for chroma in chromas:
        if chroma is None:
            continue
        sounded += 1
        if int(np.argmax(chroma)) == tonic:
            on_tonic += 1
        if _clashes(chroma):
            clashes += 1
    quiet = sum(1 for chroma in chromas if chroma is None)
    lines = [
        _key_line(total),
        f"tonic_windows {on_tonic}/{sounded}",
        f"clashes {clashes}/{sounded}",
        f"quiet_slices {quiet}/{len(chromas)}",
        _tempo_line(mono, rate),
    ]
    for when, chroma, f0 in zip(times, chromas, pitches):
        if chroma is None or f0 is None:
            lines.append(f"pitch {when:.1f} none")
            lines.append(f"harmony {when:.1f} none")
            continue
        freq, note, cents = f0
        lines.append(f"pitch {when:.1f} {note} {freq:.1f} {cents:+d}")
        lines.append(f"harmony {when:.1f} {_harmony(chroma)}")
    return lines


def _slice_tones(
    chunk: np.ndarray, rate: int, window: np.ndarray
) -> tuple[np.ndarray, tuple[float, str, int] | None]:
    chroma = np.zeros(12, dtype=np.float64)
    low_freq = 0.0
    low_amp = 0.0
    for start in range(0, chunk.size - PITCH_WINDOW + 1, PITCH_HOP):
        freqs, amps = _peaks(chunk[start : start + PITCH_WINDOW], rate, window)
        if freqs.size == 0:
            continue
        midi = 69.0 + 12.0 * np.log2(freqs / 440.0)
        pc = np.rint(midi).astype(int) % 12
        for note, amp in zip(pc, amps):
            chroma[int(note)] += float(amp)
        low = freqs < F0_HIGH_HZ
        if np.any(low):
            pick = int(np.argmax(amps[low]))
            amp = float(amps[low][pick])
            if amp > low_amp:
                low_amp = amp
                low_freq = float(freqs[low][pick])
    if low_amp <= 0.0:
        return chroma, None
    return chroma, (low_freq, *_named(low_freq))


def _peaks(frame: np.ndarray, rate: int, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = np.abs(np.fft.rfft(frame * window))
    mag = spec[1:-1]
    local = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
    idx = np.flatnonzero(local) + 1
    if idx.size == 0:
        return np.zeros(0), np.zeros(0)
    left, center, right = spec[idx - 1], spec[idx], spec[idx + 1]
    denom = left - 2.0 * center + right
    delta = np.where(np.abs(denom) < 1e-12, 0.0, 0.5 * (left - right) / denom)
    delta = np.clip(delta, -0.5, 0.5)
    freq = (idx + delta) * rate / PITCH_WINDOW
    amp = center - 0.25 * (left - right) * delta
    keep = (freq >= PITCH_LOW_HZ) & (freq <= PITCH_HIGH_HZ) & (amp > 0.0)
    freq, amp = freq[keep], amp[keep]
    if freq.size == 0:
        return freq, amp
    order = np.argsort(amp)[::-1]
    freq, amp = freq[order][:16], amp[order][:16]
    floor = float(amp[0]) * 10 ** (-24.0 / 20.0)
    strong = amp >= floor
    return freq[strong], amp[strong]


def _named(freq: float) -> tuple[str, int]:
    midi = 69.0 + 12.0 * np.log2(freq / 440.0)
    rounded = int(np.rint(midi))
    octave = rounded // 12 - 1
    cents = int(np.rint((midi - rounded) * 100.0))
    return f"{NOTES[rounded % 12]}{octave}", cents


def _best_key(chroma: np.ndarray) -> tuple[int, str, float, int, str, float]:
    """Best key, then the runner-up. An equal score keeps the lower tonic, major first."""
    if float(chroma.sum()) <= 0.0:
        return -1, "none", 0.0, -1, "none", 0.0
    ranked: list[tuple[float, int, str]] = []
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            score = _corr(chroma, np.roll(profile, tonic))
            ranked.append((score, tonic, mode))
    ranked.sort(key=lambda item: (-item[0], item[1], 0 if item[2] == "major" else 1))
    best, runner = ranked[0], ranked[1]
    return best[1], best[2], best[0], runner[1], runner[2], runner[0]


def _key_line(chroma: np.ndarray) -> str:
    tonic, mode, score, other, other_mode, other_score = _best_key(chroma)
    if tonic < 0:
        return "key none"
    margin = score - other_score
    return (
        f"key {NOTES[tonic]} {mode} correlation {_fmt(score)} margin {_fmt(margin)} "
        f"runner_up {NOTES[other]} {other_mode}"
    )


def _harmony(chroma: np.ndarray) -> str:
    share = chroma / chroma.sum()
    order = list(np.argsort(share)[::-1])
    shown = [int(pc) for pc in order if float(share[pc]) >= CLASS_SHARE][:4]
    if not shown:
        shown = [int(order[0])]
    classes = " ".join(f"{NOTES[pc]}:{int(np.rint(share[pc] * 100.0))}" for pc in shown)
    root, mode, margin = _chord(share)
    clash = _clashes(chroma)
    clash_text = "none" if not clash else ", ".join(f"{NOTES[left]}/{NOTES[right]} {gap}" for left, right, gap in clash)
    return f"{classes} chord {NOTES[root]} {mode} margin {_fmt(margin)} clash {clash_text}"


def _chord(share: np.ndarray) -> tuple[int, str, float]:
    scored: list[tuple[float, int, str]] = []
    for root in range(12):
        for mode, intervals in (("major", (0, 4, 7)), ("minor", (0, 3, 7))):
            score = float(sum(share[(root + step) % 12] for step in intervals))
            scored.append((score, root, mode))
    scored.sort(key=lambda item: (-item[0], item[1], 0 if item[2] == "major" else 1))
    return scored[0][1], scored[0][2], scored[0][0] - scored[1][0]


def _clashes(chroma: np.ndarray) -> list[tuple[int, int, int]]:
    share = chroma / chroma.sum()
    strong = [pc for pc in range(12) if float(share[pc]) >= CLASH_SHARE]
    found: list[tuple[int, int, int]] = []
    for index, left in enumerate(strong):
        for right in strong[index + 1 :]:
            gap = min((left - right) % 12, (right - left) % 12)
            if gap in {1, 2}:
                found.append((min(left, right), max(left, right), gap))
    found.sort()
    return found


def _corr(values: np.ndarray, profile: np.ndarray) -> float:
    left = values - float(values.mean())
    right = profile - float(profile.mean())
    denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(left * right) / denom)


def _tempo_line(mono: np.ndarray, rate: int) -> str:
    found = _tempo(mono, rate)
    if found is None:
        return "bpm none"
    bpm, peak = found
    return f"bpm {_fmt(bpm)} peak {_fmt(peak)}"


def _tempo(mono: np.ndarray, rate: int) -> tuple[float, float] | None:
    """BPM from the spectral-flux autocorrelation. The search is 60–210."""
    if mono.size < WINDOW or rate < 1:
        return None
    window = np.hanning(WINDOW).astype(np.float64)
    count = 1 + (mono.size - WINDOW) // HOP
    previous = None
    flux = np.zeros(count, dtype=np.float64)
    for index in range(count):
        start = index * HOP
        frame = mono[start : start + WINDOW].astype(np.float64) * window
        magnitude = np.abs(np.fft.rfft(frame))
        if previous is not None:
            flux[index] = float(np.maximum(magnitude - previous, 0.0).sum())
        previous = magnitude
    flux -= float(flux.mean())
    if float(np.dot(flux, flux)) == 0.0:
        return None
    size = 1 << int(np.ceil(np.log2(count * 2)))
    spectrum = np.fft.rfft(flux, n=size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[:count]
    correlation /= float(correlation[0]) + 1e-20
    frame_rate = rate / HOP
    low = max(1, int(frame_rate * 60.0 / BPM_HIGH))
    high = min(count - 2, int(frame_rate * 60.0 / BPM_LOW))
    if high <= low:
        return None
    lag = low + int(np.argmax(correlation[low : high + 1]))
    left, center, right = correlation[lag - 1], correlation[lag], correlation[lag + 1]
    denom = left - 2.0 * center + right
    delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    peak = float(center)
    if peak < BPM_PEAK:
        return None
    return 60.0 * frame_rate / (lag + delta), peak


def _request_lines(
    report: str,
    mono: np.ndarray,
    rate: int,
    *,
    keyscale: str | None,
    bpm: float | None,
    reference: Path | None,
) -> str:
    lines: list[str] = []
    if keyscale:
        lines.append(_requested_key(report, keyscale))
    if bpm is not None:
        lines.append(_bpm_error(report, bpm))
    if reference is not None:
        lines.append(_reference_line(mono, rate, reference))
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _requested_key(report: str, keyscale: str) -> str:
    asked = _parse_key(keyscale)
    if asked is None:
        return f"requested_key {keyscale} invalid"
    tonic, mode = asked
    detected = _detected_key(report)
    label = f"{NOTES[tonic]} {mode}"
    if detected is None:
        return f"requested_key {label} miss"
    return f"requested_key {label} {'match' if detected == asked else 'miss'}"


def _detected_key(report: str) -> tuple[int, str] | None:
    for line in report.splitlines():
        if not line.startswith("key ") or line == "key none":
            continue
        parts = line.split()
        if len(parts) < 3:
            return None
        return _parse_key(f"{parts[1]} {parts[2]}")
    return None


def _parse_key(text: str) -> tuple[int, str] | None:
    parts = text.strip().split()
    if len(parts) != 2 or parts[1] not in {"major", "minor"}:
        return None
    spelling = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
        "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }
    tonic = spelling.get(parts[0])
    if tonic is None:
        return None
    return tonic, parts[1]


def _bpm_error(report: str, requested: float) -> str:
    estimated = None
    for line in report.splitlines():
        if line.startswith("bpm ") and not line.startswith("bpm_error"):
            token = line.split()[1]
            estimated = None if token == "none" else float(token)
            break
    if estimated is None:
        return "bpm_error none"
    return f"bpm_error {_fmt(estimated - requested)}"


def _reference_line(mono: np.ndarray, rate: int, path: Path) -> str:
    if not path.is_file():
        message = f"missing reference {path.name}"
        raise FileNotFoundError(message)
    other, other_rate = _read(path)
    if other_rate != rate:
        return "reference rate_mismatch"
    target = other.mean(axis=1).astype(np.float64)
    source = mono.astype(np.float64)
    count = min(source.size, target.size)
    if count < WINDOW:
        return "reference short"
    source = source[:count]
    target = target[:count]
    distance = _mel_distance(source, target, rate)
    ratio = _si_sdr(target, source)
    ratio_text = "none" if ratio is None else _fmt(ratio)
    return f"reference mel_distance {_fmt(distance)} si_sdr {ratio_text}"


def _mel_distance(source: np.ndarray, target: np.ndarray, rate: int) -> float:
    left = _log_mel(source, rate)
    right = _log_mel(target, rate)
    frames = min(left.shape[1], right.shape[1])
    diff = left[:, :frames] - right[:, :frames]
    return float(np.mean(diff * diff))


def _log_mel(mono: np.ndarray, rate: int) -> np.ndarray:
    window = np.hanning(WINDOW).astype(np.float64)
    count = 1 + (mono.size - WINDOW) // HOP
    bins = WINDOW // 2 + 1
    power = np.empty((bins, count), dtype=np.float64)
    for index in range(count):
        start = index * HOP
        frame = mono[start : start + WINDOW] * window
        spectrum = np.fft.rfft(frame)
        power[:, index] = spectrum.real**2 + spectrum.imag**2
    filters = _mel_filters(rate, bins)
    mel = filters @ power
    return np.log(mel + 1e-10)


def _mel_filters(rate: int, bins: int, bands: int = 64) -> np.ndarray:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    points = mel_to_hz(np.linspace(hz_to_mel(40.0), hz_to_mel(rate / 2.0), bands + 2))
    centers = np.floor(points / (rate / WINDOW)).astype(int)
    centers = np.clip(centers, 0, bins - 1)
    filters = np.zeros((bands, bins), dtype=np.float64)
    for index in range(bands):
        start, middle, stop = int(centers[index]), int(centers[index + 1]), int(centers[index + 2])
        if middle <= start or stop <= middle:
            filters[index, min(middle, bins - 1)] = 1.0
            continue
        rising = np.arange(start, middle)
        falling = np.arange(middle, stop)
        filters[index, rising] = (rising - start) / (middle - start)
        filters[index, falling] = (stop - falling) / (stop - middle)
    return filters


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    ref = reference - float(reference.mean())
    est = estimate - float(estimate.mean())
    denom = float(np.dot(ref, ref))
    if denom == 0.0:
        return None
    scale = float(np.dot(est, ref) / denom)
    target = scale * ref
    noise = est - target
    target_energy = float(np.dot(target, target))
    noise_energy = float(np.dot(noise, noise))
    if target_energy == 0.0:
        return None
    if noise_energy <= target_energy * 1e-12:
        return 80.0
    return float(10.0 * np.log10(target_energy / noise_energy))


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _write_png(path: Path, gray: np.ndarray) -> None:
    height, width = gray.shape
    raw = b"".join(b"\x00" + gray[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


if __name__ == "__main__":
    main()
