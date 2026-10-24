"""``favlo --voxcpm2`` commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voxcpm2.checkpoint import summarize, tensor_header
from voxcpm2.checks import (
    attention_matches,
    conv_matches,
    linear_matches,
    rmsnorm_matches,
    rope_matches,
)
from voxcpm2.fetch import checkout, download
from voxcpm2.sampling import CONTEXT_LENGTH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --voxcpm2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="download the VoxCPM2 checkout")
    inspect = sub.add_parser("inspect", help="summarize model.safetensors")
    inspect.add_argument("--src", type=Path, default=None)
    sub.add_parser("kernels", help="compile the shader library and check the host formulas")
    speak = sub.add_parser("speak", help="speak one line to a wav file")
    speak.add_argument("--text", required=True)
    speak.add_argument("--out", type=Path, required=True)
    speak.add_argument("--instructions", default="")
    speak.add_argument("--max-len", type=int, default=48)
    speak.add_argument("--seed", type=int, default=None)
    models = sub.add_parser("models", help="list or edit the flavor registry")
    models_sub = models.add_subparsers(dest="models_command")
    add = models_sub.add_parser("add", help="register a finished checkout")
    add.add_argument("folder")
    add.add_argument("--alias", default=None)
    add.add_argument("--source", default=None)
    alias = models_sub.add_parser("alias", help="change the alias of a folder")
    alias.add_argument("folder")
    alias.add_argument("alias")
    args = parser.parse_args(argv)
    if args.command == "setup":
        return _setup()
    if args.command == "inspect":
        return _inspect(args.src)
    if args.command == "kernels":
        return _kernels()
    if args.command == "speak":
        return _speak(args.text, args.out, args.instructions, args.max_len, args.seed)
    if args.command == "models":
        return _models(args)
    return 2


def _setup() -> int:
    from favlo.cli import repo_root

    dest = checkout(repo_root())
    for name in download(dest):
        print(name, flush=True)
    print(dest)
    return 0


def _kernels() -> int:
    rmsnorm_matches()
    print("rmsnorm_row")
    linear_matches()
    print("matmul_f16")
    rope_matches()
    print("rope_half")
    attention_matches()
    print("gqa_attention")
    conv_matches()
    print("causal_conv1d")
    return 0


def _speak(text: str, dest: Path, instructions: str, max_len: int, seed: int | None) -> int:
    from favlo.cli import repo_root
    from voxcpm2.audio import encode_wav
    from voxcpm2.runtime import open_runtime

    runtime = open_runtime(checkout(repo_root()), owner="voxcpm2:speak")
    try:
        samples = runtime.speak(text, instructions=instructions, max_len=max_len, seed=seed)
    finally:
        runtime.close()
    payload = encode_wav(samples, runtime.config.out_sample_rate)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(dest)
    print(dest)
    return 0


def _models(args: argparse.Namespace) -> int:
    from favlo.registry import add_model, card_context_length, format_listing, set_alias

    command = getattr(args, "models_command", None)
    try:
        if command == "add":
            _mark_complete(args.folder)
            add_model(
                args.folder,
                alias=args.alias,
                source=args.source,
                pipeline="voxcpm2",
                runtime="metal41",
                max_prompt_tokens=CONTEXT_LENGTH,
            )
        elif command == "alias":
            set_alias(args.folder, args.alias)
        listing = format_listing()
        if listing:
            sys.stdout.write(listing + "\n")
        length = card_context_length("voxcpm2", runtime="metal41")
        if length is not None:
            sys.stdout.write(f"context {length}\n")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


def _mark_complete(folder: str) -> None:
    from favlo.cli import repo_root

    path = repo_root() / "models" / folder
    weights = path / "model.safetensors"
    config = path / "config.json"
    if not weights.is_file() or not config.is_file():
        message = f"{folder} is not a VoxCPM2 checkout"
        raise ValueError(message)
    manifest = path / "manifest.json"
    if manifest.is_file():
        body = json.loads(manifest.read_text(encoding="utf-8"))
        if body.get("complete") is True:
            return
    manifest.write_text(json.dumps({"complete": True}) + "\n", encoding="utf-8")


def _inspect(src: Path | None) -> int:
    from favlo.cli import repo_root

    path = src if src is not None else checkout(repo_root()) / "model.safetensors"
    if not path.is_file():
        sys.stderr.write(f"{path.name} is missing\n")
        return 1
    print(summarize(tensor_header(path)))
    return 0
