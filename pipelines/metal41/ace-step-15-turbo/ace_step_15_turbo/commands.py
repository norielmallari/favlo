"""``kernels``, ``quantize``, ``generate``, ``setup``, and ``delete``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np

from ace_step_15_turbo.measure import run_kernels
from ace_step_15_turbo.record import HUB, PACKED_NAME, SOURCE_NAME, quantize
from ace_step_15_turbo.request import MusicRequest
from ace_step_15_turbo.runtime import open_runtime


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --ace-step-15-turbo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("kernels")
    quant = sub.add_parser("quantize")
    quant.add_argument("--source", type=Path, default=None)
    quant.add_argument("--dest", type=Path, default=None)
    gen = sub.add_parser("generate")
    gen.add_argument("--record", type=Path, default=None)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--lyrics", default="")
    gen.add_argument("--seconds", type=float, default=10.0)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--vocal-language", default="en")
    gen.add_argument("--bpm", type=int, default=None)
    gen.add_argument("--keyscale", default=None)
    gen.add_argument("--timesignature", default=None)
    gen.add_argument("--out", type=Path, required=True)
    setup = sub.add_parser("setup")
    setup.add_argument("--dest", type=Path, default=None)
    delete = sub.add_parser("delete", help="remove one packed model")
    delete.add_argument("model")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "kernels":
        run_kernels()
        return 0
    root = _models()
    if args.command == "setup":
        dest = root / SOURCE_NAME if args.dest is None else args.dest
        _download(dest)
        return 0
    if args.command == "quantize":
        source = root / SOURCE_NAME if args.source is None else args.source
        dest = root / PACKED_NAME if args.dest is None else args.dest
        quantize(source, dest)
        from favlo.registry import register_quantized

        register_quantized(
            dest.name,
            source=source,
            alias=None,
            directory=dest,
            pipeline="ace-step-15-turbo",
            runtime="metal41",
        )
        print(dest)
        return 0
    if args.command == "delete":
        return _delete(args)
    record = root / PACKED_NAME if args.record is None else args.record
    runtime = open_runtime(record, kv_cache="e4m3", console=None, resident_percent=None)
    try:
        fields = MusicRequest(
            args.prompt,
            args.lyrics,
            args.seconds,
            args.seed,
            "wav",
            args.vocal_language,
            args.bpm,
            args.keyscale,
            args.timesignature,
        )
        payload = runtime.generate(fields, lambda stage: print(stage, flush=True))
    finally:
        runtime.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(args.out)
    print(args.out)
    return 0


def _delete(args: argparse.Namespace) -> int:
    """Drop one packed model and its registry row. A missing checkout stays missing."""
    from favlo.registry import load_registry, models_dir, save_registry, stored_directory

    text = args.model.strip()
    base = text.split(":", 1)[0] if ":" in text else text
    if not base:
        sys.stderr.write("model id is empty\n")
        return 1
    try:
        data = load_registry()
        folder = _registered_folder(data, base)
        if folder is None:
            sys.stderr.write(f"unknown model {text}\n")
            return 1
        entry = data["models"][folder]
        if entry.get("pipeline") != "ace-step-15-turbo":
            sys.stderr.write(f"{folder} is not an ace-step-15-turbo model\n")
            return 1
        source = entry.get("source")
        record = stored_directory(entry, folder)
        _remove_under_models(record, models_dir())
        if isinstance(source, str) and source and not _source_shared(data, folder, source):
            _remove_under_models(Path(source), models_dir())
        data["models"].pop(folder, None)
        save_registry(data)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"deleted {folder}\n")
    return 0


def _registered_folder(data: dict, base: str) -> str | None:
    models = data["models"]
    if base in models:
        return base
    for folder, entry in models.items():
        if entry.get("alias") == base:
            return folder
    return None


def _source_shared(data: dict, folder: str, source: str) -> bool:
    try:
        wanted = Path(source).expanduser().resolve()
    except OSError:
        return True
    for other, entry in data["models"].items():
        if other == folder:
            continue
        stored = entry.get("source")
        if not isinstance(stored, str) or not stored:
            continue
        try:
            if Path(stored).expanduser().resolve() == wanted:
                return True
        except OSError:
            continue
    return False


def _remove_under_models(path: Path, root: Path) -> None:
    """Remove a directory under ``models/``. A missing path is left missing."""
    try:
        base = root.expanduser().resolve()
        target = path.expanduser().resolve()
    except OSError:
        return
    if target == base or base not in target.parents:
        return
    if path.is_symlink() or not target.is_dir():
        return
    shutil.rmtree(target)


def _models() -> Path:
    from favlo.cli import repo_root

    path = repo_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{HUB}/tree/main?recursive=true"
    request = urllib.request.Request(url, headers={"User-Agent": "favlo"})
    with urllib.request.urlopen(request, timeout=120) as response:
        listing = json.loads(response.read().decode("utf-8"))
    for item in listing:
        if item.get("type") != "file":
            continue
        name = str(item.get("path") or "")
        if not name or name.endswith("silence_latent.pt"):
            continue
        target = dest / name
        size = int(item.get("size") or 0)
        if target.is_file() and (size == 0 or target.stat().st_size == size):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        file_url = f"https://huggingface.co/{HUB}/resolve/main/{name}"
        file_request = urllib.request.Request(file_url, headers={"User-Agent": "favlo"})
        temporary = target.with_name(target.name + ".tmp")
        with urllib.request.urlopen(file_request, timeout=600) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        temporary.replace(target)
        print(name, flush=True)
