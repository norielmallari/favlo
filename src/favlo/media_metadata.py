"""Standard media metadata under per-folder ``.favlo_metadata``.

Each directory that holds media files owns its own ``.favlo_metadata`` next to
those files. One sidecar schema for generated files and gallery media.
Generation pipelines store parameters in a nested ``generation`` object. An
analyzer fills the semantic fields and must not overwrite ``generation``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

METADATA_DIR = ".favlo_metadata"

KIND_SUFFIXES = {
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "audio": {".mp3", ".wav"},
    "video": {".mp4"},
}

_MEDIA_SUFFIXES = {suffix for group in KIND_SUFFIXES.values() for suffix in group}

_GENERATION_KEYS = (
    "model",
    "prompt",
    "seed",
    "size",
    "quality",
    "background",
    "negative_prompt",
    "seconds",
    "response_format",
    "lyrics",
    "input",
    "voice",
    "instructions",
)


def kind_for(path: str | Path) -> str | None:
    suffix = Path(path).suffix.lower()
    for kind, suffixes in KIND_SUFFIXES.items():
        if suffix in suffixes:
            return kind
    return None


def is_media(path: str | Path) -> bool:
    return kind_for(path) is not None


def sidecar_path(root: str | Path, relative: str) -> Path:
    """``photos/shot.png`` → ``<root>/photos/.favlo_metadata/shot.png.json``.

    Each media folder owns its ``.favlo_metadata`` directory. Root files use
    ``<root>/.favlo_metadata/<name>.json``.
    """
    base = Path(root)
    rel = _relative(relative)
    return base / rel.parent / METADATA_DIR / f"{rel.name}.json"


def prompt_text(sidecar: dict[str, Any]) -> str:
    """Stable English paragraph for embeddings. Empty fields are omitted."""
    parts: list[str] = []
    filename = str(sidecar.get("filename") or "").strip()
    kind = str(sidecar.get("kind") or "").strip()
    if filename:
        parts.append(f"File {filename}.")
    if kind:
        parts.append(f"Kind {kind}.")
    title = str(sidecar.get("title") or "").strip()
    if title:
        parts.append(title if title.endswith(".") else f"{title}.")
    description = str(sidecar.get("description") or "").strip()
    if description:
        parts.append(description)
    for label, key in (
        ("Place", "place"),
        ("People", "people"),
        ("Time", "time"),
        ("Objects", "objects"),
        ("Style", "style"),
        ("Colors", "colors"),
        ("Visible text", "text_in_image"),
    ):
        value = sidecar.get(key)
        text = _field_text(value)
        if text:
            parts.append(f"{label}: {text}.")
    tags = sidecar.get("tags")
    if isinstance(tags, list):
        cleaned = [str(item).strip() for item in tags if str(item).strip()]
        if cleaned:
            parts.append("Tags: " + ", ".join(cleaned) + ".")
    duration = sidecar.get("duration_seconds")
    if isinstance(duration, (int, float)) and duration > 0:
        parts.append(f"Duration {duration:g} seconds.")
    generation = sidecar.get("generation")
    if isinstance(generation, dict):
        model = str(generation.get("model") or "").strip()
        prompt = str(generation.get("prompt") or generation.get("input") or "").strip()
        if model:
            parts.append(f"Generated with {model}.")
        if prompt:
            parts.append(f"Generation prompt: {prompt}")
    return " ".join(parts).strip()


def build_index_sidecar(
    media_path: str | Path,
    *,
    relative: str,
    mtime_iso: str,
    width: int | None = None,
    height: int | None = None,
    duration_seconds: float | None = None,
    exif_time: str | None = None,
    exif_place: str | None = None,
) -> dict[str, Any]:
    """Sidecar written without a vision model (path and container signals)."""
    path = Path(media_path)
    kind = kind_for(path) or "image"
    stem = path.stem
    title = _humanize(stem)
    size_note = ""
    if width and height:
        size_note = f" {width}x{height} pixels."
    elif duration_seconds:
        size_note = f" About {duration_seconds:g} seconds."
    description = f"A {kind} file at {relative}.{size_note}".strip()
    tags = _tags(relative, stem, kind)
    return clamp(
        {
            "version": 1,
            "filename": path.name,
            "kind": kind,
            "title": title,
            "description": description,
            "tags": tags,
            "place": exif_place or "",
            "people": [],
            "time": exif_time or mtime_iso,
            "objects": [],
            "style": "",
            "colors": [],
            "text_in_image": "",
            "duration_seconds": duration_seconds,
            "nsfw": False,
            "source": "index",
        }
    )


def from_generation(
    *,
    filename: str,
    kind: str,
    generation: dict[str, Any],
    relative: str | None = None,
    width: int | None = None,
    height: int | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Sidecar written by a generation pipeline. Semantic fields start from the prompt."""
    path = Path(filename)
    stem = path.stem
    rel = relative or filename
    prompt = str(generation.get("prompt") or generation.get("input") or "").strip()
    title = _clip(prompt, 120) if prompt else _humanize(stem)
    if width and height:
        size_note = f" {width}x{height} pixels."
    elif duration_seconds:
        size_note = f" About {duration_seconds:g} seconds."
    else:
        size_note = ""
    description = prompt or f"A generated {kind} file at {rel}.{size_note}".strip()
    tags = _tags(rel, stem, kind)
    if "generated" not in tags:
        tags = ["generated", *tags][:24]
    return clamp(
        {
            "version": 1,
            "filename": path.name,
            "kind": kind,
            "title": title,
            "description": description,
            "tags": tags,
            "place": "",
            "people": [],
            "time": "",
            "objects": [],
            "style": "",
            "colors": [],
            "text_in_image": "",
            "duration_seconds": duration_seconds,
            "nsfw": False,
            "source": "generate",
            "generation": generation,
        }
    )


def merge_analyzed(existing: dict[str, Any] | None, analyzed: dict[str, Any]) -> dict[str, Any]:
    """Merge analyzer fields onto an existing sidecar. ``generation`` is preserved."""
    base = dict(analyzed)
    if existing is None:
        return clamp(base)
    generation = existing.get("generation")
    if isinstance(generation, dict) and generation:
        base["generation"] = generation
        if existing.get("source") == "generate":
            base["source"] = "generate"
    for key in ("title", "description"):
        if not str(base.get(key) or "").strip() and str(existing.get(key) or "").strip():
            base[key] = existing[key]
    return clamp(base)


def clamp(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep known fields within size caps."""
    tags = raw.get("tags") if isinstance(raw.get("tags"), list) else []
    people = raw.get("people") if isinstance(raw.get("people"), list) else []
    objects = raw.get("objects") if isinstance(raw.get("objects"), list) else []
    colors = raw.get("colors") if isinstance(raw.get("colors"), list) else []
    duration = raw.get("duration_seconds")
    if not isinstance(duration, (int, float)):
        duration = None
    payload = {
        "version": 1,
        "filename": _clip(str(raw.get("filename") or ""), 240),
        "kind": str(raw.get("kind") or "image"),
        "title": _clip(str(raw.get("title") or ""), 120),
        "description": _clip(str(raw.get("description") or ""), 800),
        "tags": [_clip(str(item), 40).lower() for item in tags][:24],
        "place": _clip(str(raw.get("place") or ""), 120),
        "people": [_clip(str(item), 40) for item in people][:24],
        "time": _clip(str(raw.get("time") or ""), 64),
        "objects": [_clip(str(item), 40) for item in objects][:24],
        "style": _clip(str(raw.get("style") or ""), 80),
        "colors": [_clip(str(item), 40) for item in colors][:24],
        "text_in_image": _clip(str(raw.get("text_in_image") or ""), 400),
        "duration_seconds": None if duration is None else float(duration),
        "nsfw": bool(raw.get("nsfw")),
        "source": str(raw.get("source") or "index"),
    }
    generation = _clamp_generation(raw.get("generation"))
    if generation is not None:
        payload["generation"] = generation
    return payload


def read_sidecar(root: str | Path, relative: str) -> dict[str, Any] | None:
    side = sidecar_path(root, relative)
    if not side.is_file():
        legacy = _legacy_sidecar_path(root, relative)
        side = legacy if legacy is not None and legacy.is_file() else side
    if not side.is_file():
        return None
    try:
        loaded = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return clamp(loaded) if isinstance(loaded, dict) else None


def write_sidecar(root: str | Path, relative: str, payload: dict[str, Any]) -> Path:
    side = sidecar_path(root, relative)
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(clamp(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return side


def remove_sidecar(root: str | Path, relative: str) -> None:
    from favlo.trash import trash_path

    side = sidecar_path(root, relative)
    if not side.is_file():
        legacy = _legacy_sidecar_path(root, relative)
        if legacy is not None and legacy.is_file():
            side = legacy
    if side.is_file():
        trash_path(side)
    _drop_empty_metadata_dir(side.parent)


def move_sidecar(root: str | Path, old_relative: str, new_relative: str) -> None:
    src = sidecar_path(root, old_relative)
    if not src.is_file():
        # Legacy mirror: ``<root>/.favlo_metadata/<old_relative>.json``.
        legacy = _legacy_sidecar_path(root, old_relative)
        if legacy is None or not legacy.is_file():
            return
        src = legacy
    dest = sidecar_path(root, new_relative)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    _drop_empty_metadata_dir(src.parent)


def relocate_metadata_prefix(root: str | Path, old_relative: str, new_relative: str) -> None:
    """After a folder rename/move, migrate any legacy mirrored metadata.

    Per-folder ``.favlo_metadata`` already moved with the directory. A leftover
    ``<root>/.favlo_metadata/<old>/`` tree is copied into ``<new>/.favlo_metadata/``.
    """
    base = Path(root)
    new_rel = new_relative.strip().strip("/")
    if not new_rel:
        return
    legacy = _prefix_dir(base, old_relative)
    if legacy is None or not legacy.exists():
        return
    dest = base / new_rel / METADATA_DIR
    dest.mkdir(parents=True, exist_ok=True)
    for child in list(legacy.iterdir()):
        target = dest / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
    meta_root = (base / METADATA_DIR).resolve()
    _prune_empty(legacy, meta_root)


def remove_metadata_prefix(root: str | Path, relative: str) -> None:
    """Drop a legacy ``<root>/.favlo_metadata/<relative>/`` tree after folder delete."""
    base = Path(root)
    target = _prefix_dir(base, relative)
    if target is None or not target.exists():
        return
    shutil.rmtree(target)
    _prune_empty(target.parent, base / METADATA_DIR)


def _clamp_generation(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    found: dict[str, Any] = {}
    for key in _GENERATION_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if key == "seed" and isinstance(value, bool):
            continue
        if key == "seed" and isinstance(value, int):
            found[key] = value
            continue
        if key in {"prompt", "negative_prompt", "lyrics", "input", "instructions"} and isinstance(
            value, str
        ):
            found[key] = _clip(value, 4000)
            continue
        if isinstance(value, str):
            found[key] = _clip(value, 240)
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            found[key] = value
            continue
        if isinstance(value, bool):
            found[key] = value
    # Preserve extra generation keys the pipeline stored (capped strings/numbers).
    for key, value in raw.items():
        if key in found or not isinstance(key, str) or key.startswith("_"):
            continue
        if isinstance(value, str):
            found[key] = _clip(value, 240)
        elif isinstance(value, bool):
            found[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            found[key] = value
        elif isinstance(value, float):
            found[key] = value
    return found or None


def _prefix_dir(root: Path, relative: str) -> Path | None:
    """Legacy mirrored tree under ``<root>/.favlo_metadata/<relative>/``."""
    cleaned = relative.strip().strip("/")
    if not cleaned:
        return None
    meta_root = (root / METADATA_DIR).resolve()
    candidate = (root / METADATA_DIR / Path(cleaned)).resolve()
    if candidate == meta_root or meta_root not in candidate.parents:
        return None
    return candidate


def _legacy_sidecar_path(root: str | Path, relative: str) -> Path | None:
    """Former layout: ``<root>/.favlo_metadata/<relative>.json``."""
    try:
        rel = _relative(relative)
    except ValueError:
        return None
    return (Path(root) / METADATA_DIR / rel).with_name(rel.name + ".json")


def _drop_empty_metadata_dir(path: Path) -> None:
    """Remove an empty ``.favlo_metadata`` directory after its last sidecar leaves."""
    try:
        if path.name != METADATA_DIR:
            return
        if any(path.iterdir()):
            return
        path.rmdir()
    except OSError:
        return


def _relative(relative: str) -> Path:
    cleaned = relative.strip().replace("\\", "/").strip("/")
    if not cleaned or cleaned in {".", ".."} or ".." in Path(cleaned).parts:
        raise ValueError("A media-relative path is required.")
    if Path(cleaned).is_absolute() or cleaned.startswith("~") or "\x00" in cleaned:
        raise ValueError("A media-relative path is required.")
    parts = Path(cleaned).parts
    if any(part.startswith(".") for part in parts):
        raise ValueError("A media-relative path is required.")
    return Path(cleaned)


def _prune_empty(start: Path, stop: Path) -> None:
    stop_resolved = stop.resolve()
    current = start
    while True:
        try:
            resolved = current.resolve()
        except OSError:
            return
        if resolved == stop_resolved or stop_resolved not in resolved.parents:
            return
        try:
            if any(current.iterdir()):
                return
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _humanize(stem: str) -> str:
    text = re.sub(r"[_\-]+", " ", stem)
    text = re.sub(r"\s+", " ", text).strip()
    return _clip(text, 120)


def _tags(relative: str, stem: str, kind: str) -> list[str]:
    found: list[str] = [kind]
    for piece in Path(relative).parts[:-1]:
        token = piece.strip().lower()
        if token and not token.startswith(".") and not token.isdigit() and len(token) > 1:
            found.append(_clip(token, 40))
    for token in re.split(r"[_\-\s]+", stem.lower()):
        if token and not token.isdigit() and len(token) > 1:
            found.append(_clip(token, 40))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped[:24]


def _field_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str):
        return value.strip()
    return ""


def _clip(value: str, limit: int) -> str:
    cleaned = value.strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip()
