"""``favlo --voxcpm2`` commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voxcpm2.metal41.checkpoint import summarize, tensor_header
from voxcpm2.metal41.checks import (
    activation_matches,
    attention_matches,
    conv_matches,
    linear_matches,
    rmsnorm_matches,
    rope_matches,
    stack_matches,
)
from voxcpm2.metal41.fetch import checkout, download
from voxcpm2.metal41.sampling import CONTEXT_LENGTH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --voxcpm2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="download the VoxCPM2 checkout")
    inspect = sub.add_parser("inspect", help="summarize model.safetensors")
    inspect.add_argument("--src", type=Path, default=None)
    pack = sub.add_parser("pack", help="write kernel-order weights next to the checkout")
    pack.add_argument("--src", type=Path, default=None)
    pack.add_argument("--dest", type=Path, default=None)
    sub.add_parser("kernels", help="compile the shader library and check the host formulas")
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
    if args.command == "pack":
        return _pack(args.src, args.dest)
    if args.command == "kernels":
        return _kernels()
    if args.command == "models":
        return _models(args)
    return 2


def _pack(src: Path | None, dest: Path | None) -> int:
    from favlo.cli import repo_root
    from voxcpm2.metal41.pack import write_pack

    origin = src if src is not None else checkout(repo_root())
    for name in write_pack(origin, dest):
        print(name, flush=True)
    return 0


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
    activation_matches()
    print("silu_f32")
    print("swiglu_f32")
    print("add_f32")
    print("copy_rows")
    print("bias_pack")
    stack_matches()
    print("stack")
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

    mark_complete(repo_root() / "models" / folder, folder)


def mark_complete(path: Path, folder: str) -> None:
    from voxcpm2.metal41.store import WEIGHTS_NAME

    weights = path / "model.safetensors"
    blob = path / WEIGHTS_NAME
    config = path / "config.json"
    if not config.is_file() or not (weights.is_file() or blob.is_file()):
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
