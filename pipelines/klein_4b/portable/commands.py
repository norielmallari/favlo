"""CLI for setup, quantize, and kernels."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from favlo.cli import repo_root
from favlo.smoke_lock import hold, release

from .constants import RECORD

_PIPELINE = "klein_4b"
_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --klein_4b")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="check the cached FLUX.2 [klein] 4B snapshot")
    sub.add_parser("kernels", help="compile the shaders and check int8 and attention")
    quant = sub.add_parser("quantize", help="pack the cached checkpoint as int8")
    quant.add_argument("--weights", "-w", type=Path, help="snapshot directory")
    quant.add_argument("--out", type=Path, help="packed directory")
    quant.add_argument("--name", default=None, help="folder under models/; the registry id")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "setup":
        return _setup()
    if args.command == "kernels":
        return _locked(kernels, "kernels")
    if args.command == "quantize":
        return _locked(lambda: _quantize(args.weights, args.out, args.name), "quantize")
    return 2


def kernels() -> int:
    import numpy as np

    from klein_4b.metal41.metal import Gpu
    from klein_4b.metal41.ops import reference_attention

    from .formats import pack_matrix, quantize_activation, reconstruct_groups, reconstruct_row

    rng = np.random.default_rng(7)
    weight = rng.normal(0, 0.25, size=(7, 200)).astype(np.float32)
    codes, scales = pack_matrix(weight)
    activation = rng.normal(0, 0.5, size=(3, 200)).astype(np.float16)
    activation[0, :] = np.float16(0)
    query = rng.normal(0, 1, size=(5, 2, 128)).astype(np.float32)
    key = rng.normal(0, 1, size=(5, 2, 128)).astype(np.float32)
    value = rng.normal(0, 1, size=(5, 2, 128)).astype(np.float32)
    gpu = Gpu()
    try:
        got = gpu.product_arrays(activation, codes, scales, 7, 200)
        left = reconstruct_groups(*quantize_activation(activation, codes.shape[1]))[:, :200]
        right = reconstruct_row(codes, scales)[:7, :200]
        expect = (left @ right.T).astype(np.float16)
        if not np.allclose(got[:, :7], expect, rtol=1e-2, atol=2e-2):
            delta = float(np.max(np.abs(got[:, :7].astype(np.float32) - expect.astype(np.float32))))
            print(f"kernels int8 mismatch {delta}", flush=True)
            return 1
        attended = gpu.attend(query, key, value, causal=True, valid=4)
        reference = reference_attention(query, key, value, causal=True, valid=4)
        if not np.allclose(attended, reference, rtol=1e-4, atol=1e-4):
            delta = float(np.max(np.abs(attended - reference)))
            print(f"kernels attention mismatch {delta}", flush=True)
            return 1
        if _elementwise(gpu) != 0:
            return 1
    finally:
        gpu.close()
    print("kernels ok", flush=True)
    return 0


def _elementwise(gpu: object) -> int:
    import numpy as np

    from klein_4b.metal41 import ops

    from .rope import apply_dit, apply_text, dit_table, text_table

    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(4, 32)).astype(np.float16)
    weight = rng.normal(size=(32,)).astype(np.float16)
    shift = rng.normal(size=(32,)).astype(np.float16)
    scale = rng.normal(size=(32,)).astype(np.float16)
    wide = rng.normal(size=(3, 16)).astype(np.float16)
    residual = rng.normal(size=(4, 8)).astype(np.float16)
    branch = rng.normal(size=(4, 8)).astype(np.float16)
    gate = rng.normal(size=(8,)).astype(np.float16)
    state = rng.normal(size=(8, 4)).astype(np.float16)
    speed = rng.normal(size=(8, 4)).astype(np.float16)
    query = rng.normal(size=(3, 2, 128)).astype(np.float32)
    text_cos, text_sin = text_table(3)
    ids = np.zeros((2, 4), dtype=np.int64)
    ids[:, 1] = np.arange(2)
    dit_cos, dit_sin = dit_table(ids)
    dit_q = rng.normal(size=(2, 2, 128)).astype(np.float32)
    volume = rng.normal(size=(2, 5, 6)).astype(np.float16)
    group = rng.normal(size=(32, 2, 3)).astype(np.float16)
    group_w = rng.normal(size=(32,)).astype(np.float16)
    group_b = rng.normal(size=(32,)).astype(np.float16)
    checks = (
        ("rmsnorm", gpu.copy_out(gpu.rmsnorm(gpu.copy_in(hidden), gpu.copy_in(weight), store="f16")), ops.as_f16(ops.rmsnorm(hidden, weight))),
        ("modulate", gpu.copy_out(gpu.modulate(gpu.copy_in(hidden), gpu.copy_in(shift), gpu.copy_in(scale))), ops.modulate(hidden, shift, scale)),
        ("swiglu", gpu.copy_out(gpu.swiglu(gpu.copy_in(wide))), ops.swiglu(wide)),
        ("gate", gpu.copy_out(gpu.gate(gpu.copy_in(residual), gpu.copy_in(gate), gpu.copy_in(branch))), ops.gated(residual, gate, branch)),
        ("rope", gpu.copy_out(gpu.rope_text(gpu.copy_in(query), gpu.copy_in(text_cos), gpu.copy_in(text_sin))), apply_text(query, text_cos, text_sin)),
        ("dit rope", gpu.copy_out(gpu.rope_dit(gpu.copy_in(dit_q), gpu.copy_in(dit_cos), gpu.copy_in(dit_sin))), apply_dit(dit_q, dit_cos, dit_sin)),
        ("euler", gpu.copy_out(gpu.euler(gpu.copy_in(state), gpu.copy_in(speed), -0.25)), ops.as_f16(state.astype(np.float32) + np.float32(-0.25) * speed.astype(np.float32))),
        ("groupnorm", gpu.copy_out(gpu.groupnorm(gpu.copy_in(group), gpu.copy_in(group_w), gpu.copy_in(group_b), silu=False)), ops.as_f16(ops.groupnorm(group, group_w, group_b))),
        ("im2col", gpu.copy_out(gpu.im2col(gpu.copy_in(volume), 3, 3, 1, (1, 1, 1, 1))), ops.im2col(volume, 3, 3, 1, 1)),
        (
            "im2col down",
            gpu.copy_out(gpu.im2col(gpu.copy_in(volume), 3, 3, 2, (0, 1, 0, 1))),
            ops.im2col(np.pad(volume, ((0, 0), (0, 1), (0, 1))), 3, 3, 2, 0),
        ),
    )
    for name, got, expect in checks:
        if got.shape != expect.shape or not np.allclose(got, expect, rtol=1e-3, atol=2e-2):
            delta = float(np.max(np.abs(got.astype(np.float32) - expect.astype(np.float32))))
            print(f"kernels {name} mismatch {delta}", flush=True)
            return 1
    return 0


def _setup() -> int:
    from .constants import REPO, WEIGHTS
    from .source import weights_ready

    try:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(REPO, local_files_only=True))
    except Exception as exc:
        sys.stderr.write(f"{REPO} is not in the local Hugging Face cache\n")
        sys.stderr.write(f"{exc}\n")
        return 1
    lines = []
    for name, size in WEIGHTS.items():
        path = root / name
        have = path.stat().st_size if path.is_file() else 0
        lines.append(f"{name} {have}/{size}")
    print("\n".join(lines), flush=True)
    if not weights_ready(root):
        sys.stderr.write("checkpoint is still downloading\n")
        return 1
    print(f"{REPO} ready", flush=True)
    return 0


def _quantize(weights: Path | None, out: Path | None, name: str | None) -> int:
    from favlo.registry import register_quantized

    from .record import quantize
    from .source import snapshot

    try:
        origin = Path(weights) if weights is not None else snapshot()
        dest = _record_dir(name, out)
        quantize(origin, dest)
        register_quantized(
            dest.name,
            source=origin,
            alias=None,
            directory=dest,
            pipeline=_PIPELINE,
            runtime="metal41",
            pipeline_root=_pipeline_root(),
            max_prompt_tokens=512,
        )
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    print(dest.name, flush=True)
    return 0


def _pipeline_root() -> str:
    package = Path(__file__).resolve().parent
    relative = Path(os.path.relpath(package, repo_root()))
    return relative.as_posix()


def _record_dir(name: str | None, out: Path | None) -> Path:
    from favlo.registry import models_dir

    if name is not None and out is not None:
        message = "--name and --out name the same directory"
        raise ValueError(message)
    if name is not None:
        if _FOLDER.fullmatch(name) is None:
            message = f"{name} is not a folder name"
            raise ValueError(message)
        return models_dir() / name
    if out is not None:
        return out
    return models_dir() / RECORD


def _locked(fn, command: str) -> int:
    root = repo_root()
    created = hold(root=root, pipeline=_PIPELINE, command=command)
    try:
        return fn()
    finally:
        if created:
            release(root=root)
