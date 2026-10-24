"""Flavor registry. One file under models/, never committed.

A flavor is a packed record directory. An alias is a second name for that
directory. A variant is a sampling row on the same record.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

_CARD_FIELDS = ("temperature", "top_p", "top_k", "presence_penalty")
DEFAULT_CONTEXT_LENGTH = 131072


def models_dir(root: Path | None = None) -> Path:
    return (root or _repo_root()) / "models"


def registry_path(root: Path | None = None) -> Path:
    return models_dir(root) / "registry.json"


@dataclass
class Selection:
    folder: str
    path: Path
    pipeline: str
    alias: str | None
    requested_id: str
    parent: str
    variant: str | None
    sampling: dict[str, float | int] | None
    kv_cache: str
    resident_percent: float | None
    max_prompt_tokens: int | None
    context_length: int = DEFAULT_CONTEXT_LENGTH
    variants: dict[str, dict[str, float | int]] | None = None
    default_variant: str | None = None
    runtime: str = ""
    custom: bool = False
    parts: dict[str, Path] | None = None


def load_registry(root: Path | None = None) -> dict[str, Any]:
    path = registry_path(root)
    if not path.is_file():
        return {"version": 1, "models": {}}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        message = "registry.json is not an object"
        raise ValueError(message)
    data.setdefault("version", 1)
    data.pop("default", None)
    data.setdefault("models", {})
    if not isinstance(data["models"], dict):
        message = "registry models is not an object"
        raise ValueError(message)
    return data


def save_registry(data: dict[str, Any], root: Path | None = None) -> None:
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def list_unregistered(root: Path | None = None) -> list[str]:
    directory = models_dir(root)
    if not directory.is_dir():
        return []
    known = set(load_registry(root)["models"])
    found: list[str] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or child.name in known:
            continue
        manifest = child / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            body = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            continue
        if body.get("complete"):
            found.append(child.name)
    return found


def format_listing(root: Path | None = None) -> str:
    data = load_registry(root)
    lines: list[str] = []
    for folder, entry in data["models"].items():
        pipeline = _pipeline_name(entry, folder)
        alias = entry.get("alias") or ""
        kv = entry.get("kv_cache") or "e4m3"
        alias_text = f"alias {alias}" if alias else "alias"
        lines.append(f"{folder} pipeline {pipeline} {alias_text} kv {kv}")
    for folder in list_unregistered(root):
        lines.append(f"unregistered {folder}")
    return "\n".join(lines)


def add_model(
    folder: str,
    *,
    alias: str | None = None,
    source: str | None = None,
    root: Path | None = None,
    pipeline: str,
    runtime: str,
    custom: bool = False,
    max_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    data = load_registry(root)
    path = models_dir(root) / folder
    _require_complete(path)
    _check_alias(data, alias, folder)
    if folder in data["models"]:
        message = f"{folder} is already registered"
        raise ValueError(message)
    data["models"][folder] = _entry(
        alias=alias,
        source=source,
        pipeline=pipeline,
        runtime=runtime,
        custom=custom,
        max_prompt_tokens=max_prompt_tokens,
        directory=str(path),
    )
    save_registry(data, root)
    return data


def set_alias(folder: str, alias: str | None, root: Path | None = None) -> dict[str, Any]:
    data = load_registry(root)
    entry = data["models"].get(folder)
    if entry is None:
        message = f"{folder} is not registered"
        raise ValueError(message)
    _check_alias(data, alias, folder)
    entry["alias"] = alias
    save_registry(data, root)
    return data


def register_quantized(
    folder: str,
    *,
    source: Path,
    alias: str | None,
    directory: Path,
    root: Path | None = None,
    pipeline: str,
    runtime: str,
    custom: bool = False,
    max_prompt_tokens: int | None = None,
    parts: Mapping[str, Path] | None = None,
) -> None:
    """Write or update the entry for a record this quantize just produced."""
    data = load_registry(root)
    _check_alias(data, alias, folder)
    source_text = str(source)
    directory_text = str(directory)
    part_text = _part_text(parts)
    variants, default_variant = pipeline_sampling(pipeline, runtime=runtime, custom=custom)
    current = data["models"].get(folder)
    if current is not None:
        stored = current.get("source")
        if stored and _same_path(stored, source):
            current["directory"] = directory_text
            _merge_parts(current, part_text)
            current["variants"] = variants
            current["default_variant"] = default_variant
            current["runtime"] = runtime
            if custom:
                current["custom"] = True
            else:
                current.pop("custom", None)
            if alias is not None:
                current["alias"] = alias
            if max_prompt_tokens is not None:
                current["max_prompt_tokens"] = max_prompt_tokens
                current.pop("max_tokens", None)
            ctx = card_context_length(pipeline, runtime=runtime, custom=custom)
            if ctx is not None:
                current["context_length"] = ctx
            else:
                current.setdefault("context_length", DEFAULT_CONTEXT_LENGTH)
            save_registry(data, root)
            return
        message = f"{folder} is already registered for a different source"
        raise ValueError(message)
    data["models"][folder] = _entry(
        alias=alias,
        source=source_text,
        pipeline=pipeline,
        runtime=runtime,
        custom=custom,
        max_prompt_tokens=max_prompt_tokens,
        directory=directory_text,
        parts=part_text,
    )
    save_registry(data, root)


def set_part(folder: str, name: str, path: Path, root: Path | None = None) -> None:
    """Record one quantized component directory on an existing row."""
    data = load_registry(root)
    entry = data["models"].get(folder)
    if entry is None:
        message = f"{folder} is not registered"
        raise ValueError(message)
    _merge_parts(entry, {name: str(path)})
    save_registry(data, root)


def stored_directory(entry: dict[str, Any], folder: str, root: Path | None = None) -> Path:
    """The quantized record stored on the row."""
    stored = entry.get("directory")
    if isinstance(stored, str) and stored:
        return Path(stored)
    return models_dir(root) / folder


def parts_for_directory(path: Path, root: Path | None = None) -> dict[str, Path]:
    """Component directories for the row whose record is ``path``."""
    wanted = path.expanduser().resolve()
    data = load_registry(root)
    for folder, entry in data["models"].items():
        try:
            stored = stored_directory(entry, folder, root).expanduser().resolve()
        except OSError:
            continue
        if stored == wanted:
            return _parts(entry)
    return {}


def refuse_conflicting_destination(folder: str, source: Path, root: Path | None = None) -> None:
    data = load_registry(root)
    current = data["models"].get(folder)
    if current is None:
        return
    stored = current.get("source")
    if stored and not _same_path(stored, source):
        message = f"{folder} is already registered for a different source"
        raise ValueError(message)


def resolve(model: str, *, root: Path | None = None, resident: str | None = None) -> Selection:
    """Map a client model id onto one folder and an optional sampling row."""
    del resident
    data = load_registry(root)
    text = model.strip()
    if not text:
        message = "model id is empty"
        raise LookupError(message)
    base, variant = _split_variant(text)
    folder = _folder_for(data, base if variant is not None else text)
    if folder is None:
        _unknown(text, data)
    names = _variant_names(data["models"][folder])
    if variant is not None and variant not in names:
        _unknown(text, data)
    return _selection(data, folder, text, variant, root)


def model_objects(root: Path | None = None, *, resident: str | None = None) -> list[dict[str, Any]]:
    data = load_registry(root)
    created = _created(root)
    rows: list[dict[str, Any]] = []
    for folder, entry in data["models"].items():
        names = [folder]
        alias = entry.get("alias")
        if isinstance(alias, str) and alias:
            names.append(alias)
        for name in names:
            rows.append(_object(name, folder, None, entry, created, resident))
            for variant in _variant_names(entry):
                rows.append(_object(f"{name}:{variant}", folder, variant, entry, created, resident))
    return rows


def one_model(
    model_id: str, root: Path | None = None, *, resident: str | None = None
) -> dict[str, Any]:
    selection = resolve(model_id, root=root)
    data = load_registry(root)
    entry = data["models"][selection.folder]
    created = _created(root)
    return _object(
        selection.requested_id,
        selection.folder,
        selection.variant,
        entry,
        created,
        resident,
    )


def choose_complete(
    *,
    model: str | None,
    record: Path | None,
    root: Path | None = None,
    pipeline: str | None = None,
) -> tuple[Path, Selection | None]:
    """The record ``complete`` and ``bf16`` open. ``record`` wins when passed."""
    if record is not None:
        return record, None
    if not pipeline:
        message = "pass the pipeline that owns this command"
        raise LookupError(message)
    data = load_registry(root)
    entries = [
        (folder, entry)
        for folder, entry in data["models"].items()
        if entry.get("pipeline") == pipeline
    ]
    if model:
        selection = resolve(model, root=root)
        return selection.path, selection
    if len(entries) == 1:
        folder = entries[0][0]
        record_path = stored_directory(entries[0][1], folder, root)
        return record_path, _selection(data, folder, folder, None, root)
    if not entries:
        message = (
            "no registered model\n"
            f"pack a checkpoint: favlo --{pipeline} quantize --source <dir>\n"
            f"or register a record: favlo --{pipeline} models add <folder>"
        )
        raise LookupError(message)
    listing = format_listing(root)
    message = f"pass --model\n{listing}"
    raise LookupError(message)


def _context_length(entry: dict[str, Any]) -> int:
    stored = entry.get("context_length")
    if isinstance(stored, int) and not isinstance(stored, bool) and stored > 0:
        return stored
    return DEFAULT_CONTEXT_LENGTH


def _max_prompt_tokens(entry: dict[str, Any]) -> int | None:
    cap = entry.get("max_prompt_tokens")
    if isinstance(cap, bool) or not isinstance(cap, int):
        return None
    return cap


def _pipeline_name(entry: dict[str, Any], folder: str) -> str:
    pipeline = entry.get("pipeline")
    if not isinstance(pipeline, str) or not pipeline:
        message = f"{folder} has no pipeline"
        raise LookupError(message)
    _require_runtime(entry, folder)
    return pipeline


def _require_runtime(entry: dict[str, Any], folder: str) -> str:
    from favlo.cli import RUNTIMES

    runtime = entry.get("runtime")
    if not isinstance(runtime, str) or not runtime:
        message = f"{folder} has no runtime"
        raise LookupError(message)
    if runtime not in RUNTIMES:
        message = f"{folder} runtime is unknown"
        raise LookupError(message)
    return runtime


def _custom(entry: dict[str, Any]) -> bool:
    return bool(entry.get("custom"))


def _entry(
    *,
    alias: str | None,
    source: str | None,
    pipeline: str,
    runtime: str,
    directory: str,
    custom: bool = False,
    max_prompt_tokens: int | None = None,
    parts: dict[str, str] | None = None,
) -> dict[str, Any]:
    variants, default_variant = pipeline_sampling(pipeline, runtime=runtime, custom=custom)
    ctx = card_context_length(pipeline, runtime=runtime, custom=custom)
    entry: dict[str, Any] = {
        "pipeline": pipeline,
        "runtime": runtime,
        "alias": alias,
        "source": source,
        "directory": directory,
        "kv_cache": "e4m3",
        "resident_percent": None,
        "variants": variants,
        "default_variant": default_variant,
        "context_length": ctx if ctx is not None else DEFAULT_CONTEXT_LENGTH,
    }
    if parts:
        entry["parts"] = parts
    if custom:
        entry["custom"] = True
    if max_prompt_tokens is not None:
        entry["max_prompt_tokens"] = max_prompt_tokens
    return entry


def _part_text(parts: Mapping[str, Path] | None) -> dict[str, str] | None:
    if not parts:
        return None
    return {name: str(path) for name, path in parts.items()}


def _merge_parts(entry: dict[str, Any], parts: dict[str, str] | None) -> None:
    if not parts:
        return
    stored = entry.get("parts")
    if not isinstance(stored, dict):
        stored = {}
    stored.update(parts)
    entry["parts"] = stored


def _parts(entry: dict[str, Any]) -> dict[str, Path]:
    raw = entry.get("parts")
    if not isinstance(raw, dict):
        return {}
    found: dict[str, Path] = {}
    for name, path in raw.items():
        if isinstance(name, str) and isinstance(path, str) and path:
            found[name] = Path(path)
    return found


def _require_complete(path: Path) -> None:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        message = f"{path.name} has no manifest.json"
        raise ValueError(message)
    body = json.loads(manifest.read_text())
    if not body.get("complete"):
        message = f"{path.name} is not a finished record"
        raise ValueError(message)


def _check_alias(data: dict[str, Any], alias: str | None, folder: str) -> None:
    if alias is None:
        return
    if ":" in alias or alias == "" or "/" in alias:
        message = f"alias {alias} is not a single name"
        raise ValueError(message)
    if alias == folder or alias in data["models"]:
        message = f"alias {alias} is already a folder name"
        raise ValueError(message)
    for name, entry in data["models"].items():
        if name == folder:
            continue
        if entry.get("alias") == alias:
            message = f"alias {alias} is already used"
            raise ValueError(message)


def _split_variant(text: str) -> tuple[str, str | None]:
    if ":" not in text:
        return text, None
    base, _, suffix = text.rpartition(":")
    if base and suffix:
        return base, suffix
    return text, None


def _unknown(text: str, data: dict[str, Any]) -> NoReturn:
    known = _known_names(data)
    message = f"unknown model {text}"
    if known:
        message = f"{message}; known: {', '.join(known)}"
    raise LookupError(message)


def _folder_for(data: dict[str, Any], base: str) -> str | None:
    if base in data["models"]:
        return base
    pipeline_hits: list[str] = []
    for folder, entry in data["models"].items():
        if entry.get("alias") == base:
            return folder
        if entry.get("pipeline") == base:
            pipeline_hits.append(folder)
    if len(pipeline_hits) == 1:
        return pipeline_hits[0]
    return None


def _selection(
    data: dict[str, Any],
    folder: str,
    requested: str,
    variant: str | None,
    root: Path | None,
) -> Selection:
    entry = data["models"][folder]
    stored = _stored_variants(entry)
    sampling = None
    if variant is not None:
        row = stored.get(variant)
        sampling = dict(row) if isinstance(row, dict) and row else None
    alias = entry.get("alias")
    alias_text = alias if isinstance(alias, str) and alias else None
    parent = alias_text or folder
    if requested == folder or requested.startswith(f"{folder}:"):
        parent = folder
    percent = entry.get("resident_percent")
    return Selection(
        folder=folder,
        path=stored_directory(entry, folder, root),
        pipeline=_pipeline_name(entry, folder),
        alias=alias_text,
        requested_id=requested,
        parent=str(parent),
        variant=variant,
        sampling=sampling,
        kv_cache=str(entry.get("kv_cache") or "e4m3"),
        resident_percent=None if percent is None else float(percent),
        max_prompt_tokens=_max_prompt_tokens(entry),
        context_length=_context_length(entry),
        variants=stored,
        default_variant=_default_variant(entry),
        runtime=_require_runtime(entry, folder),
        custom=_custom(entry),
        parts=_parts(entry),
    )


def _object(
    model_id: str,
    folder: str,
    variant: str | None,
    entry: dict[str, Any],
    created: int,
    resident: str | None,
) -> dict[str, Any]:
    alias = entry.get("alias")
    if model_id == folder or model_id.startswith(f"{folder}:"):
        parent = folder
    else:
        parent = alias or folder
    row: dict[str, Any] = {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": "favlo",
        "root": folder,
        "pipeline": _pipeline_name(entry, folder),
        "loaded": resident == folder,
    }
    cap = _max_prompt_tokens(entry)
    if cap is not None:
        row["max_prompt_tokens"] = cap
    row["context_length"] = _context_length(entry)
    if variant is not None:
        row["parent"] = parent if ":" not in model_id else model_id.split(":", 1)[0]
        row["variant"] = variant
        variants = entry.get("variants") or {}
        row["sampling"] = dict(variants.get(variant) or {})
    return row


def _known_names(data: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for folder, entry in data["models"].items():
        names.append(folder)
        alias = entry.get("alias")
        if isinstance(alias, str) and alias and alias not in names:
            names.append(alias)
        pipeline = entry.get("pipeline")
        if isinstance(pipeline, str) and pipeline and pipeline not in names:
            names.append(pipeline)
    return names


def _created(root: Path | None) -> int:
    path = registry_path(root)
    if not path.is_file():
        return 0
    return int(path.stat().st_mtime)


def _same_path(stored: str, source: Path) -> bool:
    return Path(stored).expanduser().resolve() == source.expanduser().resolve()


def card_context_length(pipeline: str, *, runtime: str, custom: bool = False) -> int | None:
    """``CONTEXT_LENGTH`` on the sampling card, when the pipeline defines one."""
    package = pipeline.replace("-", "_")
    path = _sampling_path(package, runtime=runtime, custom=custom)
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"{package}.sampling_context", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, "CONTEXT_LENGTH", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def pipeline_sampling(
    pipeline: str, *, runtime: str, custom: bool = False
) -> tuple[dict[str, dict[str, float | int]], str]:
    """Load ``VARIANTS`` and ``DEFAULT_VARIANT`` from ``<package>/sampling.py``.

    The module is loaded by path so registration does not import the pipeline.
    """
    package = pipeline.replace("-", "_")
    path = _sampling_path(package, runtime=runtime, custom=custom)
    if not path.is_file():
        message = f"{pipeline} has no sampling card at {package}/sampling.py"
        raise ValueError(message)
    spec = importlib.util.spec_from_file_location(f"{package}.sampling", path)
    if spec is None or spec.loader is None:
        message = f"{pipeline} sampling card could not be loaded"
        raise ValueError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = getattr(module, "VARIANTS", None)
    default = getattr(module, "DEFAULT_VARIANT", None)
    if not isinstance(raw, dict) or not raw:
        message = f"{pipeline} sampling card must define VARIANTS"
        raise ValueError(message)
    if not isinstance(default, str) or default not in raw:
        message = f"{pipeline} DEFAULT_VARIANT must name a row in VARIANTS"
        raise ValueError(message)
    variants: dict[str, dict[str, float | int]] = {}
    for name, row in raw.items():
        if not isinstance(name, str) or not name or ":" in name:
            message = f"{pipeline} variant name must be a single token"
            raise ValueError(message)
        if not isinstance(row, dict) or set(row) != set(_CARD_FIELDS):
            message = f"{pipeline} {name} row must set {', '.join(_CARD_FIELDS)}"
            raise ValueError(message)
        variants[name] = {
            "temperature": float(row["temperature"]),
            "top_p": float(row["top_p"]),
            "top_k": int(row["top_k"]),
            "presence_penalty": float(row["presence_penalty"]),
        }
    return variants, default


def _variant_names(entry: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_stored_variants(entry))


def _stored_variants(entry: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    raw = entry.get("variants")
    if not isinstance(raw, dict):
        return {}
    stored: dict[str, dict[str, float | int]] = {}
    for name, row in raw.items():
        if isinstance(name, str) and isinstance(row, dict):
            stored[name] = dict(row)
    return stored


def _default_variant(entry: dict[str, Any]) -> str | None:
    name = entry.get("default_variant")
    if isinstance(name, str) and name in _stored_variants(entry):
        return name
    return None


def _sampling_path(package: str, *, runtime: str, custom: bool) -> Path:
    from favlo.cli import pipeline_tree

    return pipeline_tree(runtime, custom=custom) / package / "sampling.py"


def _repo_root() -> Path:
    override = os.environ.get("FAVLO_ROOT")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        marker = parent / "src" / "favlo" / "cli.py"
        if (parent / "pyproject.toml").is_file() and marker.is_file():
            return parent
    message = "favlo package is not inside a checkout"
    raise FileNotFoundError(message)
