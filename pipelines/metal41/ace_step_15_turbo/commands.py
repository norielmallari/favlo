"""``kernels``, ``quantize``, ``generate``, and ``setup``."""

from __future__ import annotations

import argparse
import json
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
            pipeline="ace-step-15-turbo",
            runtime="metal41",
        )
        print(dest)
        return 0
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
