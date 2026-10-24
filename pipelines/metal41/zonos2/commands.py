"""``favlo --zonos2`` commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from zonos2.audio import encode_wav
from zonos2.checkpoint import id_tables, list_tensors
from zonos2.metal import run_checks
from zonos2.pack import quantize as pack_record
from zonos2.runtime import open_runtime
from zonos2.speaker import Encoder


def _inspect(src: Path) -> int:
    weights = src / "model.pth"
    if not weights.is_file():
        sys.stderr.write("model.pth is not in the source directory\n")
        return 2
    tensors = list_tensors(weights)
    tables = id_tables(tensors)
    sys.stdout.write(f"tensors {len(tensors)}\n")
    for tensor in tensors:
        shape = "x".join(str(item) for item in tensor.shape)
        sys.stdout.write(f"{tensor.name} {tensor.dtype} {shape}\n")
    if not tables:
        sys.stderr.write("the checkpoint has no integer token-id table\n")
        return 2
    sys.stdout.write(f"id tables {len(tables)}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --zonos2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("kernels", help="compile the shader library and run the formula checks")
    inspect = sub.add_parser("inspect", help="list model.pth and look for a token-id table")
    inspect.add_argument("--src", type=Path, required=True)
    quantize = sub.add_parser("quantize", help="score MoE layers and pack the record")
    quantize.add_argument("--src", type=Path, required=True)
    quantize.add_argument("--out", type=Path, default=None)
    generate = sub.add_parser("generate", help="write a 44.1 kHz mono WAV")
    generate.add_argument("--src", type=Path, required=True)
    generate.add_argument("--text", required=True)
    generate.add_argument("--speaker", type=Path, default=None)
    generate.add_argument("--out", type=Path, required=True)
    encode = sub.add_parser("encode", help="write a 2048-float speaker vector from a wav")
    encode.add_argument("--wav", type=Path, required=True)
    encode.add_argument("--src", type=Path, default=Path("models/qwen3-speaker"))
    encode.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "kernels":
        run_checks()
        sys.stdout.write("kernels ok\n")
        return 0
    if args.command == "inspect":
        return _inspect(args.src)
    if args.command == "quantize":
        record = pack_record(args.src, args.out)
        sys.stdout.write(f"packed {record.name}\n")
        return 0
    if args.command == "encode":
        vector = Encoder(args.src / "model.safetensors").embed(args.wav)
        np.save(args.out, vector)
        sys.stdout.write(f"wrote {args.out.name}\n")
        return 0
    speaker = None
    if args.speaker is not None:
        speaker = np.load(args.speaker).astype(np.float32)
    runtime = open_runtime(args.src, kv_cache="e4m3", console=None, resident_percent=100.0)
    args.out.write_bytes(encode_wav(runtime.speak(args.text, speaker)))
    sys.stdout.write(f"wrote {args.out.name}\n")
    return 0
