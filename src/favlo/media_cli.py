"""``favlo media --pipeline <id>`` calls a listener that is already up.

The command posts ``{"kind","body"}`` to ``POST /v1/dispatch`` and prints
each ``mediaUri``. It does not name a pipeline, open a runtime, or start
``favlo serve``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_HOST = "127.0.0.1"
_PORT = 8129


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo media")
    parser.add_argument("--pipeline", required=True, help="pipeline id")
    parser.add_argument("--model", default=None, help="registry folder; omit for the registry default")
    parser.add_argument("--kind", default=None, help="image, video, music, or voice")
    parser.add_argument("--out", type=Path, default=None, help="write the result JSON here")
    parser.add_argument("--host", default=_HOST)
    parser.add_argument("--port", type=int, default=_PORT)
    parser.add_argument("fields", nargs="*", help="body fields as key=value")
    args = parser.parse_args(argv)
    from favlo.daemon import port_open

    if not port_open(args.host, args.port):
        sys.stderr.write(f"serve is not listening on {args.host}:{args.port}\n")
        return 1
    try:
        kind = args.kind or _kind(args.pipeline)
        body = _body(args.fields, args.model, args.pipeline, kind)
        packed = _post(args.host, args.port, kind, body)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    _emit(packed, args.out)
    return 0


def _kind(pipeline: str) -> str:
    from favlo.cli import load_pipeline
    from favlo.registry import load_registry

    data = load_registry()
    runtime = None
    root = None
    for entry in data["models"].values():
        if not isinstance(entry, dict) or entry.get("pipeline") != pipeline:
            continue
        named = entry.get("runtime")
        if isinstance(named, str):
            runtime = named
        stored = entry.get("pipeline_root")
        if isinstance(stored, str):
            root = stored
        break
    if runtime is None:
        message = f"unknown pipeline {pipeline}"
        raise ValueError(message)
    from favlo.cli import pipeline_instance
    from favlo.media_pipeline import MediaPipeline
    from favlo.mcp_server import media_kinds

    module = load_pipeline(pipeline, runtime=runtime, pipeline_root=root)
    kinds = media_kinds(module)
    if not kinds:
        try:
            plugin = pipeline_instance(module)
        except TypeError as exc:
            message = "pass --kind"
            raise ValueError(message) from exc
        if isinstance(plugin, MediaPipeline):
            kinds = [plugin.kind]
    if len(kinds) != 1:
        message = "pass --kind"
        raise ValueError(message)
    return kinds[0]


def _body(fields: list[str], model: str | None, pipeline: str, kind: str) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for field in fields:
        key, sep, raw = field.partition("=")
        if not sep or not key:
            message = f"expected key=value, got {field}"
            raise ValueError(message)
        body[key] = _value(raw)
    if model:
        body["model"] = model
    elif "model" not in body:
        body["model"] = _default_model(pipeline, kind)
    return body


def _value(raw: str) -> Any:
    if raw[:1] in "{[tfn0123456789-":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(parsed, (int, float, bool, list, dict)) or parsed is None:
            return parsed
    return raw


def _default_model(pipeline: str, kind: str) -> str:
    """The registry ``media`` row for this kind when that folder is this pipeline.

    A missing row, or a row for another pipeline, uses the row marked
    ``pipeline_default``. When none is marked, the first folder registered
    for ``pipeline`` is used.
    """
    from favlo.registry import load_registry, media_default, pipeline_default

    named = media_default(kind)
    data = load_registry()
    models = data["models"]
    if named:
        found = _folder(models, named)
        if found is not None:
            entry = models[found]
            if isinstance(entry, dict) and entry.get("pipeline") == pipeline:
                return _public(entry, found)
    marked = pipeline_default(pipeline)
    if marked:
        entry = models.get(marked)
        if isinstance(entry, dict) and entry.get("pipeline") == pipeline:
            return _public(entry, marked)
    for folder, entry in models.items():
        if isinstance(entry, dict) and entry.get("pipeline") == pipeline:
            return _public(entry, folder)
    message = f"unknown pipeline {pipeline}"
    raise ValueError(message)


def _folder(models: dict[str, Any], named: str) -> str | None:
    if named in models:
        return named
    for folder, entry in models.items():
        if isinstance(entry, dict) and entry.get("alias") == named:
            return folder
    return None


def _public(entry: dict[str, Any], folder: str) -> str:
    alias = entry.get("alias")
    if isinstance(alias, str) and alias:
        return alias
    return folder


def _post(host: str, port: int, kind: str, body: dict[str, Any]) -> Any:
    payload = json.dumps({"kind": kind, "body": body}).encode()
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/dispatch",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            parsed = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(detail) from exc
    if not isinstance(parsed, (dict, list)):
        message = "dispatch returned no result"
        raise RuntimeError(message)
    return parsed


def _emit(packed: Any, out: Path | None) -> None:
    rows = packed if isinstance(packed, list) else [packed]
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("mediaUri"), str):
            print(row["mediaUri"], flush=True)
    if out is not None:
        out.write_text(json.dumps(packed) + "\n", encoding="utf-8")
