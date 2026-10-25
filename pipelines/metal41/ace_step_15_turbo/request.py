"""Music request fields shared by the OpenAI job and Stability text-to-audio."""

from __future__ import annotations

from dataclasses import dataclass

from ace_step_15_turbo.schedule import MAX_SECONDS, MIN_SECONDS, STEPS

_FORMATS = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}


@dataclass(frozen=True)
class MusicRequest:
    prompt: str
    lyrics: str
    seconds: float
    seed: int | None
    output_format: str
    vocal_language: str
    bpm: int | None
    keyscale: str | None
    timesignature: str | None

    @property
    def media_type(self) -> str:
        return _FORMATS[self.output_format]


def parse_music(body: dict) -> MusicRequest:
    """Accept an OpenAI-style JSON object or a Stability multipart field dict."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        message = "prompt is required"
        raise ValueError(message)
    lyrics = body.get("lyrics", "")
    if lyrics is None:
        lyrics = ""
    if not isinstance(lyrics, str):
        message = "lyrics must be a string"
        raise ValueError(message)
    seconds = _seconds(body)
    seed = _seed(body.get("seed"))
    output_format = _format(body)
    language = body.get("vocal_language", "en")
    if not isinstance(language, str) or not language.strip():
        message = "vocal_language must be a string"
        raise ValueError(message)
    bpm = _optional_int(body.get("bpm"), "bpm")
    keyscale = _optional_str(body.get("keyscale"), "keyscale")
    timesignature = _optional_str(body.get("timesignature"), "timesignature")
    _reject_steps(body.get("steps"))
    _reject_cfg(body.get("cfg_scale"))
    count = body.get("n", 1)
    if count != 1:
        message = "n must be 1"
        raise ValueError(message)
    return MusicRequest(
        prompt=prompt,
        lyrics=lyrics,
        seconds=seconds,
        seed=seed,
        output_format=output_format,
        vocal_language=language.strip(),
        bpm=bpm,
        keyscale=keyscale,
        timesignature=timesignature,
    )


def _seconds(body: dict) -> float:
    raw = body.get("duration", body.get("seconds", 30))
    try:
        seconds = float(raw)
    except (TypeError, ValueError) as exc:
        message = "duration must be a number of seconds"
        raise ValueError(message) from exc
    if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
        message = f"duration must be from {MIN_SECONDS:g} to {MAX_SECONDS:g} seconds"
        raise ValueError(message)
    return seconds


def _seed(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        message = "seed must be an integer"
        raise ValueError(message) from exc
    if seed < 0 or seed > 4_294_967_294:
        message = "seed must be from 0 to 4294967294"
        raise ValueError(message)
    return seed


def _format(body: dict) -> str:
    stability = body.get("output_format")
    speech = body.get("response_format")
    if stability is not None and speech is not None and stability != speech:
        message = "output_format and response_format differ"
        raise ValueError(message)
    chosen = stability if stability is not None else speech
    if chosen is None:
        return "mp3"
    if not isinstance(chosen, str) or chosen not in _FORMATS:
        message = "output_format must be mp3, wav, or flac"
        raise ValueError(message)
    return chosen


def _optional_int(value: object, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        message = f"{name} must be an integer"
        raise ValueError(message) from exc


def _optional_str(value: object, name: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        message = f"{name} must be a string"
        raise ValueError(message)
    return value


def _reject_steps(value: object) -> None:
    if value is None or value == "":
        return
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        message = "steps must be 8"
        raise ValueError(message) from exc
    if steps != STEPS:
        message = "steps must be 8"
        raise ValueError(message)


def _reject_cfg(value: object) -> None:
    if value is None or value == "":
        return
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        message = "cfg_scale must be 1"
        raise ValueError(message) from exc
    if scale != 1.0:
        message = "cfg_scale must be 1"
        raise ValueError(message)
