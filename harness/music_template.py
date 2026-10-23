"""Shared lines for a music request and the inspect report.

The input template is the request the generator can be checked against.
The output template is those same fields as the inspect script measured them.
Seconds must agree within 0.05. The key must be the same note and mode.
BPM is an anchor: the estimate may sit within 8 of the request.
"""

from __future__ import annotations

from typing import Any

FIELDS = ("seconds", "key", "bpm")
BPM_TOLERANCE = 8.0
SECONDS_TOLERANCE = 0.05
_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_SPELLING = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def smoke_body(note: dict[str, Any]) -> dict[str, Any]:
    """JSON body for one smoke generation. The measurable fields match :func:`input_template`."""
    return {
        "model": note["model"],
        "prompt": note["prompt"],
        "lyrics": note["lyrics"],
        "seconds": note["seconds"],
        "seed": note["seed"],
        "vocal_language": note["vocal_language"],
        "response_format": "wav",
        "bpm": note["bpm"],
        "keyscale": note["keyscale"],
        "timesignature": note["timesignature"],
    }


def input_template(note: dict[str, Any]) -> str:
    """Measurable request lines. The key spelling is the canonical one."""
    key = _canonical_key(str(note["keyscale"]))
    if key is None:
        message = f"keyscale {note['keyscale']} is not a note plus major or minor"
        raise ValueError(message)
    seconds = float(note["seconds"])
    bpm = float(note["bpm"])
    return _block("template_input", {"seconds": f"{seconds:.4f}", "key": key, "bpm": f"{bpm:.4f}"})


def output_template(report: str) -> str:
    """The same three fields, read from an inspect report."""
    seconds = _first_value(report, "seconds")
    key_line = _first_line(report, "key")
    bpm_line = _first_line(report, "bpm")
    key = "none"
    if key_line is not None and key_line != "key none":
        parts = key_line.split()
        if len(parts) >= 3:
            parsed = _canonical_key(f"{parts[1]} {parts[2]}")
            key = parsed if parsed is not None else "none"
    bpm = "none"
    if bpm_line is not None:
        token = bpm_line.split()[1]
        bpm = token if token == "none" else f"{float(token):.4f}"
    if seconds is None:
        seconds = "none"
    else:
        seconds = f"{float(seconds):.4f}"
    return _block("template_output", {"seconds": seconds, "key": key, "bpm": bpm})


def compare_templates(expected: str, measured: str) -> list[str]:
    """Empty when the output template agrees with the input template."""
    left = _fields(expected, "template_input")
    right = _fields(measured, "template_output")
    mismatches: list[str] = []
    for field in FIELDS:
        if field not in left or field not in right:
            mismatches.append(f"{field} missing")
            continue
        got = right[field]
        want = left[field]
        if field == "key":
            if got != want:
                mismatches.append(f"key {got} is not {want}")
            continue
        if got == "none":
            mismatches.append(f"{field} none")
            continue
        gap = abs(float(got) - float(want))
        limit = SECONDS_TOLERANCE if field == "seconds" else BPM_TOLERANCE
        if gap > limit:
            mismatches.append(f"{field} {got} is {gap:.4f} from {want}")
    return mismatches


def _block(header: str, fields: dict[str, str]) -> str:
    lines = [header]
    for field in FIELDS:
        lines.append(f"{field} {fields[field]}")
    return "\n".join(lines) + "\n"


def _fields(text: str, header: str) -> dict[str, str]:
    lines = text.splitlines()
    try:
        start = lines.index(header)
    except ValueError:
        return {}
    found: dict[str, str] = {}
    for line in lines[start + 1 :]:
        if not line.strip() or line.startswith("template_") or line.startswith("spectrogram "):
            break
        key, _, value = line.partition(" ")
        if key in FIELDS and value:
            found[key] = value.strip()
    return found


def _first_line(report: str, key: str) -> str | None:
    prefix = f"{key} "
    for line in report.splitlines():
        if line.startswith("template_"):
            break
        if line.startswith(prefix) and not line.startswith(f"{key}_"):
            return line
    return None


def _first_value(report: str, key: str) -> str | None:
    line = _first_line(report, key)
    if line is None:
        return None
    return line.split()[1]


def _canonical_key(text: str) -> str | None:
    parts = text.strip().split()
    if len(parts) != 2 or parts[1] not in {"major", "minor"}:
        return None
    tonic = _SPELLING.get(parts[0])
    if tonic is None:
        return None
    return f"{_NOTES[tonic]} {parts[1]}"
