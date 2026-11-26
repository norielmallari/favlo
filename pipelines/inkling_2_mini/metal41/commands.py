"""CLI for the Inkling 2 Mini pipeline. NVFP4 experts are baked once for the open."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from inkling_2_mini.metal41.codec import nvfp4_row
from inkling_2_mini.metal41.layout import EOS
from inkling_2_mini.metal41.session import Session, f32, u32
from inkling_2_mini.portable.record import CARROT_PROMPT

PIPELINE = "inkling_2_mini"
FOLDER = "Inkling-2-Mini-fp4-gptq-foem"
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _folder() -> Path:
    from favlo.registry import models_dir

    return models_dir() / FOLDER


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --inkling_2_mini")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("kernels", help="compile the shader library and run the dispatch checks")
    setup = sub.add_parser(
        "setup",
        help="download the checkpoint, pack the fp4 planes, and register it",
    )
    setup.add_argument(
        "--repo",
        type=_repo_id,
        default=None,
        help="Hugging Face repository org/name, or a huggingface.co link",
    )
    setup.add_argument(
        "--name",
        type=_folder_name,
        default=None,
        help="folder under models/ and the registry id",
    )
    setup.add_argument(
        "--no-calibrate",
        action="store_true",
        help="register without rounding the dense fp4 planes",
    )
    setup.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    complete = sub.add_parser("complete", help="greedy carrot completion from the checkpoint")
    complete.add_argument("--max-memory", default=None)
    complete.add_argument("--limit", type=int, default=64)
    sub.add_parser("models", help="list registered checkpoints")
    calibrate = sub.add_parser(
        "calibrate", help="round NVFP4 experts with GPTQ-FOEM from calibration inputs"
    )
    calibrate.add_argument("--limit", type=int, default=2048)
    calibrate.add_argument(
        "--reuse", action="store_true", help="round again from the saved calibration input sums"
    )
    sub.add_parser("bake", help="requant the NVFP4 experts into the expert fp4 pack")
    quantize = sub.add_parser(
        "quantize",
        help="pack a BF16 checkpoint as mixed fp4 and fp8 experts and seal it",
    )
    quantize.add_argument(
        "--src", type=Path, required=True, help="folder that holds the BF16 shard"
    )
    quantize.add_argument(
        "--name",
        type=_folder_name,
        default="Inkling-2-mini-fp8-e50",
        help="folder under models/ and the registry id",
    )
    quantize.add_argument("--expert-fp8", type=float, default=50.0)
    quantize.add_argument("--promote", choices=("error", "firing"), default="error")
    quantize.add_argument("--firing-db", type=Path, default=None)
    quantize.add_argument("--since", type=int, default=None)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command == "kernels":
        return _kernels()
    if args.command == "setup":
        return _setup(args)
    if args.command == "complete":
        return _complete(args.max_memory, args.limit)
    if args.command == "calibrate":
        return _calibrate(args.limit, args.reuse)
    if args.command == "bake":
        return _bake()
    if args.command == "quantize":
        return _quantize(args)
    return _models()


def _quantize(args: argparse.Namespace) -> int:
    from favlo.registry import models_dir
    from inkling_2_mini.metal41.bf16 import quantize

    dest = models_dir() / args.name
    quantize(
        args.src,
        dest,
        expert_fp8=args.expert_fp8,
        promote=args.promote,
        firing_db=args.firing_db,
        firing_since=args.since,
        log=lambda line: print(line, flush=True),
    )
    return 0


def _bake(folder: Path | None = None) -> int:
    from inkling_2_mini.metal41.record import publish, ready
    from inkling_2_mini.metal41.resident import bake
    from inkling_2_mini.metal41.vision import ensure_pack

    root = folder or _folder()

    def note(line: str) -> None:
        print(line, flush=True)

    if ready(root):
        publish(root, log=note)
        return 0
    bake(root, log=note)
    ensure_pack(root, log=note)
    publish(root, log=note)
    return 0


def _calibrate(limit: int, reuse: bool, folder: Path | None = None) -> int:
    from inkling_2_mini.metal41.calibration import calibrate

    calibrate(
        folder or _folder(),
        limit=limit,
        reuse=reuse,
        log=lambda line: print(line, flush=True),
    )
    return 0


def _kernels() -> int:
    from favlo.memory import memory

    owner = "inkling_2_mini:kernels"
    session = Session(owner)
    try:
        rng = np.random.default_rng(7)
        _check_f16(session, rng)
        _check_fp4(session, rng)
        _check_fp8(session, rng)
        _check_nvfp4(session, rng)
        _check_argmax(session, rng)
        _check_attention(session, rng)
    finally:
        session.close()
        memory.sweep(owner)
    print("kernels ok")
    return 0


_E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _metal_fp4(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Host dequant of Metal fp4 rows: E2M1 nibbles times 2 ** (scale - 127) per 32."""
    low = _E2M1[packed & 0x0F]
    high = _E2M1[packed >> 4]
    values = np.stack([low, high], axis=-1).reshape(packed.shape[0], -1)
    factor = np.repeat(np.exp2(scales.astype(np.float64) - 127.0), 32, axis=1)
    return (values * factor).astype(np.float32)


def _close(found: np.ndarray, expect: np.ndarray, tol: float, label: str) -> None:
    scale = float(np.max(np.abs(expect))) or 1.0
    error = float(np.max(np.abs(found - expect))) / scale
    if not np.isfinite(error) or error > tol:
        message = f"{label} error {error:.4f} is over {tol}"
        raise RuntimeError(message)


def _check_f16_band(
    session: Session,
    rng: np.random.Generator,
    rows: int,
    cols: int,
    inner: int,
    stride: int,
) -> None:
    """Half matmul into a row stride wider than the written band."""
    activation = rng.standard_normal((rows, inner)).astype(np.float16)
    weight = rng.standard_normal((cols, inner)).astype(np.float16)
    expect = activation.astype(np.float32) @ weight.astype(np.float32).T
    dest = session.zeros(rows * stride * 4)
    session.dispatch(
        "matmul_f16_band",
        [session.upload(activation), session.upload(weight), dest],
        [u32(rows), u32(cols), u32(inner), u32(stride)],
        (cols // 64, rows // 64, 1),
    )
    found = session.download_f32(dest, rows * stride).reshape(rows, stride)[:, :cols]
    _close(found, expect, 2e-3, f"half band {rows}x{cols} stride {stride}")


def _check_f16(session: Session, rng: np.random.Generator) -> None:
    rows, cols, inner = 64, 128, 256
    activation = rng.standard_normal((rows, inner)).astype(np.float16)
    weight = rng.standard_normal((cols, inner)).astype(np.float16)
    expect = activation.astype(np.float32) @ weight.astype(np.float32).T
    dest = session.zeros(rows * cols * 4)
    session.dispatch(
        "matmul_f16_m64",
        [session.upload(activation), session.upload(weight), dest],
        [u32(rows), u32(cols), u32(inner)],
        (cols // 64, rows // 64, 1),
    )
    _close(session.download_f32(dest, rows * cols).reshape(rows, cols), expect, 1e-3, "half matmul")
    _check_f16_band(session, rng, 128, 128, 256, 192)
    _check_f16_band(session, rng, 192, 64, 256, 160)
    # The attention plane writes Q into a stride of ATTN_ROWS. Several M tiles.
    _check_f16_band(session, rng, 128, 128, 2304, 256)
    one = activation[0].astype(np.float32)
    gemv = session.zeros(cols * 4)
    session.dispatch(
        "gemv_f16",
        [session.upload(one), session.upload(weight), gemv],
        [u32(cols), u32(inner)],
        (cols, 1, 1),
    )
    _close(session.download_f32(gemv, cols), expect[0], 1e-3, "half gemv")
    _check_attn_f16(session, rng)


def _check_attn_f16(session: Session, rng: np.random.Generator) -> None:
    from inkling_2_mini.metal41.weights import ATTN_ROWS, ROW_K, ROW_V

    inner = 128
    q = rng.standard_normal(inner).astype(np.float32)
    k = rng.standard_normal(inner).astype(np.float32)
    v = rng.standard_normal(inner).astype(np.float32)
    weight = rng.standard_normal((ATTN_ROWS, inner)).astype(np.float16)
    sink = np.zeros((24, inner), dtype=np.float16)
    dest = session.zeros((ATTN_ROWS + 24) * 4)
    session.dispatch(
        "gemv_attn_f16",
        [
            session.upload(q),
            session.upload(k),
            session.upload(v),
            session.upload(np.ascontiguousarray(weight)),
            session.upload(sink),
            session.upload(sink),
            session.upload(sink),
            dest,
        ],
        [u32(ATTN_ROWS), u32(inner)],
        (ATTN_ROWS // 4, 1, 1),
    )
    found = session.download_f32(dest, ATTN_ROWS)
    for row, act in ((0, q), (ROW_K, k), (ROW_V, v), (ROW_K - 4, q)):
        expect = weight[row].astype(np.float32) @ act
        _close(np.array([found[row]]), np.array([expect]), 2e-3, f"half attention gemv row {row}")


def _check_fp8(session: Session, rng: np.random.Generator) -> None:
    from inkling_2_mini.metal41.weights import dequant_mxfp8, quantize_mxfp8

    rows, cols, inner = 64, 128, 256
    codes, scales = quantize_mxfp8(rng.standard_normal((cols, inner)).astype(np.float32))
    weight = dequant_mxfp8(codes, scales)
    activation = rng.standard_normal((rows, inner)).astype(np.float16)
    expect = activation.astype(np.float32) @ weight.T
    data = session.upload(codes)
    scale = session.upload(scales)
    dest = session.zeros(rows * cols * 4)
    session.dispatch(
        "matmul_fp8_m32",
        [session.upload(activation), data, scale, dest],
        [u32(rows), u32(cols), u32(inner), u32(cols)],
        (cols // 128, rows // 32, 1),
    )
    _close(session.download_f32(dest, rows * cols).reshape(rows, cols), expect, 2e-2, "fp8 matmul")
    one = activation[0].astype(np.float32)
    gemv = session.zeros(cols * 4)
    session.dispatch(
        "gemv_fp8_conv_add",
        [
            session.upload(one),
            data,
            scale,
            session.zeros(cols * 4 * 4),
            session.zeros(cols * 3 * 4),
            gemv,
        ],
        [u32(cols), u32(inner)],
        (cols // 4, 1, 1),
    )
    _close(session.download_f32(gemv, cols), expect[0] / np.sqrt(5.0), 2e-2, "fp8 gemv")
    from inkling_2_mini.metal41.gptq import foem_fp8, foem_fp8_device

    matrix = rng.standard_normal((64, 128)).astype(np.float32)
    gram = matrix.T @ matrix + np.eye(128, dtype=np.float32)
    host_codes, host_scales = foem_fp8(matrix, gram, beta=0.1)
    device_codes, device_scales = foem_fp8_device(session, matrix, gram, beta=0.1)
    if not np.array_equal(host_codes, device_codes) or not np.array_equal(
        host_scales, device_scales
    ):
        message = "fp8 foem device does not match the host"
        raise RuntimeError(message)


def _check_fp4(session: Session, rng: np.random.Generator) -> None:
    from inkling_2_mini.metal41.gptq import foem_device_matches
    from inkling_2_mini.metal41.weights import _quantize_fp4

    foem_device_matches(session)

    rows, cols, inner = 64, 256, 512
    packed, scales = _quantize_fp4(rng.standard_normal((cols, inner)).astype(np.float32))
    weight = _metal_fp4(packed, scales)
    activation = rng.standard_normal((rows, inner)).astype(np.float16)
    expect = activation.astype(np.float32) @ weight.T
    data = session.upload(packed)
    scale = session.upload(scales)
    dest = session.zeros(rows * cols * 4)
    session.dispatch(
        "matmul_fp4_m32",
        [session.upload(activation), data, scale, dest],
        [u32(rows), u32(cols), u32(inner), u32(cols)],
        (cols // 128, rows // 32, 1),
    )
    _close(session.download_f32(dest, rows * cols).reshape(rows, cols), expect, 2e-3, "fp4 matmul")
    gemv = session.zeros(cols * 4)
    session.dispatch(
        "gemv_fp4",
        [session.upload(activation[0].astype(np.float32)), data, scale, gemv],
        [u32(cols), u32(inner)],
        (cols // 4, 1, 1),
    )
    _close(session.download_f32(gemv, cols), expect[0], 1e-4, "fp4 gemv")


def _check_nvfp4(session: Session, rng: np.random.Generator) -> None:
    experts, rows, cols = 2, 64, 128
    packed = rng.integers(0, 256, (experts, rows, cols // 2), dtype=np.uint8)
    scales = rng.integers(0x20, 0x40, (experts, rows, cols // 16), dtype=np.uint8)
    global_scale = np.array([0.5, 2.0], dtype=np.float32)
    dest = session.zeros(rows * cols * 2)
    session.launch(
        "dequant_nvfp4",
        [session.upload(packed[0]), session.upload(scales[0]), dest],
        [u32(rows), u32(cols), f32(0.5)],
        rows * cols,
    )
    found = np.frombuffer(session.download(dest, rows * cols * 2), dtype=np.float16).astype(
        np.float32
    )
    expect = np.stack([nvfp4_row(packed[0, row], scales[0, row], 0.5) for row in range(rows)])
    _close(found.reshape(rows, cols), expect, 1e-3, "NVFP4 dequant")
    out_data = session.zeros(experts * rows * cols // 2)
    out_scale = session.zeros(experts * rows * cols // 32)
    session.launch(
        "pack_nvfp4_fp4",
        [
            session.upload(packed),
            session.upload(scales),
            session.upload(global_scale),
            out_data,
            out_scale,
        ],
        [u32(experts), u32(rows), u32(cols), u32(cols)],
        experts * rows * cols // 32,
    )
    got = np.frombuffer(session.download(out_data, experts * rows * cols // 2), dtype=np.uint8)
    got_scale = np.frombuffer(
        session.download(out_scale, experts * rows * cols // 32), dtype=np.uint8
    )
    for expert in range(experts):
        source = np.stack(
            [
                nvfp4_row(packed[expert, row], scales[expert, row], float(global_scale[expert]))
                for row in range(rows)
            ]
        )
        requant = _metal_fp4(
            got.reshape(experts, rows, cols // 2)[expert],
            got_scale.reshape(experts, rows, cols // 32)[expert],
        )
        _close(requant, source, 0.2, "NVFP4 requant")
    _check_nvfp4_decode(session, rng)


def _check_nvfp4_decode(session: Session, rng: np.random.Generator) -> None:
    """Software NVFP4 GEMV and the prefill tile against the host dequant."""
    rows, cols, inner = 32, 128, 128
    packed = rng.integers(0, 256, (cols, inner // 2), dtype=np.uint8)
    scales = rng.integers(1, 127, (cols, inner // 16), dtype=np.uint8)
    tensor_scale = np.float32(0.25)
    weight = np.stack(
        [nvfp4_row(packed[row], scales[row], float(tensor_scale)) for row in range(cols)]
    )
    activation = rng.standard_normal((rows, inner)).astype(np.float16)
    expect = activation.astype(np.float32) @ weight.T
    data = session.upload(packed)
    scale = session.upload(scales)
    jobs = session.upload(np.array([0, 0], dtype=np.uint32))
    expert_of = session.upload(np.zeros(1, dtype=np.uint32))
    row_base = session.upload(np.zeros(1, dtype=np.uint32))
    info = session.upload(np.array([1, 1, rows], dtype=np.uint32))
    offsets = session.upload(np.zeros(1, dtype=np.uint32))
    widths = session.upload(np.zeros(1, dtype=np.uint8))
    dest = session.zeros(rows * cols * 4)
    session.dispatch(
        "expert_nvfp4_tile",
        [
            session.upload(activation),
            data,
            scale,
            dest,
            jobs,
            expert_of,
            row_base,
            info,
            offsets,
            offsets,
            widths,
            session.upload(np.array([tensor_scale], dtype=np.float32)),
        ],
        [u32(inner), u32(cols), u32(cols * inner // 2), u32(cols * inner // 16), u32(0)],
        (cols // 128, 1, 1),
    )
    found = session.download_f32(dest, rows * cols).reshape(rows, cols)
    _close(found, expect, 2e-2, "NVFP4 tile")
    gemv = session.zeros(cols * 4)
    session.dispatch(
        "gemv_nvfp4",
        [session.upload(activation[0].astype(np.float32)), data, scale, gemv],
        [u32(cols), u32(inner), f32(float(tensor_scale))],
        (cols // 4, 1, 1),
    )
    _close(session.download_f32(gemv, cols), expect[0], 2e-4, "NVFP4 gemv")


def _check_argmax(session: Session, rng: np.random.Generator) -> None:
    count = 200064
    logits = rng.standard_normal(count).astype(np.float32)
    logits[123457] = 50.0
    groups = (count + 255) // 256
    values = session.zeros(groups * 4)
    index = session.zeros(groups * 4)
    chosen = session.zeros(16)
    session.dispatch(
        "argmax_span",
        [session.upload(logits), values, index],
        [u32(count), f32(0.0), u32(1)],
        (groups, 1, 1),
    )
    session.dispatch("argmax_index", [values, index, chosen], [u32(groups)], (1, 1, 1))
    picked = int(np.frombuffer(session.download(chosen, 4), dtype=np.uint32)[0])
    if picked != 123457:
        message = f"argmax picked {picked}"
        raise RuntimeError(message)
    # A logit 60 above the rest wins every Gumbel draw unless the noise is infinite.
    draws = 2000
    sharp = np.zeros(count, dtype=np.float32)
    sharp[777] = 60.0
    sharp_handle = session.upload(sharp)
    picks = session.zeros(draws * 4)
    session.begin()
    for draw in range(draws):
        session.dispatch(
            "argmax_span",
            [sharp_handle, values, index],
            [u32(count), f32(1.0), u32(draw * 2654435761 & 0xFFFFFFFF)],
            (groups, 1, 1),
        )
        session.dispatch(
            "argmax_index", [values, index, (picks, draw * 4)], [u32(groups)], (1, 1, 1)
        )
    session.end()
    won = np.frombuffer(session.download(picks, draws * 4), dtype=np.uint32)
    strays = int(np.count_nonzero(won != 777))
    if strays:
        message = f"Gumbel draw picked a token 60 below the top {strays} times in {draws}"
        raise RuntimeError(message)


def _attention_reference(
    q: np.ndarray,
    rel: np.ndarray,
    gate: np.ndarray,
    sink: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    table: np.ndarray,
    position: int,
    window: int,
) -> np.ndarray:
    """One token, 24 query heads over 8 key heads, with sink, relative bias, and gate."""
    lo = max(0, position + 1 - window) if window else 0
    dist = position - np.arange(lo, position + 1)
    near = dist < 1024
    out = np.zeros((24, 128), dtype=np.float64)
    for head in range(24):
        kv = head // 3
        k = keys[lo : position + 1, kv * 128 : kv * 128 + 128].astype(np.float64)
        v = values[lo : position + 1, kv * 128 : kv * 128 + 128].astype(np.float64)
        scores = k @ q[head].astype(np.float64) * 128**-0.5
        bias = rel[head].astype(np.float64) @ table[:, np.minimum(dist, 1023)]
        scores = scores + np.where(near, bias, 0.0)
        logits = np.concatenate([[float(sink[head])], scores])
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        out[head] = weights[1:] @ v / (1.0 + np.exp(-gate[head].astype(np.float64)))
    return out.astype(np.float32)


def _check_attention(session: Session, rng: np.random.Generator) -> None:
    from inkling_2_mini.metal41.weights import ATTN_ROWS, ROW_GATE, ROW_REL

    tokens, pos0, context = 40, 1100, 1140
    keys = (rng.standard_normal((context, 1024)) * 0.3).astype(np.float16)
    values = rng.standard_normal((context, 1024)).astype(np.float16)
    proj = rng.standard_normal((tokens, ATTN_ROWS)).astype(np.float32)
    table = (rng.standard_normal((16, 1024)) * 0.3).astype(np.float32)
    sink = rng.standard_normal((tokens, 64)).astype(np.float32)
    k_cache = session.upload(keys)
    v_cache = session.upload(values)
    proj_h = session.upload(proj)
    table_h = session.upload(table)
    sink_h = session.upload(sink)
    out = session.zeros(tokens * 3072 * 4)

    def heads(row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = proj[row, :3072].reshape(24, 128)
        rel = proj[row, ROW_REL : ROW_REL + 384].reshape(24, 16)
        gate = proj[row, ROW_GATE : ROW_GATE + 3072].reshape(24, 128)
        return q, rel, gate

    for window in (0, 1024):
        session.dispatch(
            "attend_flash",
            [
                proj_h,
                k_cache,
                v_cache,
                (proj_h, ROW_REL * 4),
                table_h,
                sink_h,
                (proj_h, ROW_GATE * 4),
                out,
            ],
            [
                u32(tokens),
                u32(pos0),
                u32(window),
                u32(ATTN_ROWS),
                u32(ATTN_ROWS),
                u32(64),
                u32(ATTN_ROWS),
                u32(0),
            ],
            (24, (tokens + 31) // 32, 1),
        )
        found = session.download_f32(out, tokens * 3072).reshape(tokens, 24, 128)
        for row in (0, tokens - 1):
            q, rel, gate = heads(row)
            expect = _attention_reference(
                q, rel, gate, sink[row], keys, values, table, pos0 + row, window
            )
            _close(found[row], expect, 5e-3, f"prefill attention window {window}")
        position = context - 1
        span = max(256, -(-(position + 1) // 1024) * 256)
        q, rel, gate = heads(0)
        session.dispatch(
            "attend_group",
            [
                proj_h,
                k_cache,
                v_cache,
                (proj_h, ROW_REL * 4),
                table_h,
                sink_h,
                (proj_h, ROW_GATE * 4),
                session.zeros(4 * 24 * 132 * 4),
                session.zeros(64 * 4),
                out,
            ],
            [u32(position), u32(window), u32(span), u32(0)],
            (1, 8, -(-(position + 1 - (max(0, position + 1 - window) if window else 0)) // span)),
        )
        found = session.download_f32(out, 3072).reshape(24, 128)
        expect = _attention_reference(q, rel, gate, sink[0], keys, values, table, position, window)
        _close(found, expect, 1e-4, f"decode attention window {window}")
    _check_window(session, keys, values, table, proj_h, table_h, sink_h, out, heads, sink)


def _check_window(session, keys, values, table, proj_h, table_h, sink_h, out, heads, sink) -> None:
    """A 1024-slot ring gathers in order and attends the same scores as a linear cache."""
    from inkling_2_mini.metal41.weights import ROW_GATE, ROW_REL

    window = 1024
    pos0 = int(keys.shape[0]) - 1
    ring_k = np.zeros((window, 1024), dtype=np.float16)
    ring_v = np.zeros_like(ring_k)
    for key in range(pos0):
        ring_k[key % window] = keys[key]
        ring_v[key % window] = values[key]
    ring_kh = session.upload(ring_k)
    ring_vh = session.upload(ring_v)
    origin = max(0, pos0 + 1 - window) & ~63
    rows = pos0 - origin
    stage_k = session.zeros((rows + 1) * 1024 * 2)
    stage_v = session.zeros((rows + 1) * 1024 * 2)
    session.dispatch(
        "kv_window",
        [stage_k, stage_v, ring_kh, ring_vh],
        [u32(origin), u32(pos0), u32(pos0 + 1), u32(window), u32(0)],
        ((rows + 255) // 256, 1, 1),
    )
    staged = np.frombuffer(session.download(stage_k, rows * 1024 * 2), dtype=np.float16)
    staged = staged.reshape(rows, 1024)
    held = pos0 - window
    expect = np.zeros((rows, 1024), dtype=np.float16)
    for absolute in range(origin, pos0):
        if absolute >= held:
            expect[absolute - origin] = keys[absolute]
    if not np.array_equal(staged, expect):
        message = "window gather did not rebuild the local history"
        raise RuntimeError(message)
    ring_k[pos0 % window] = keys[pos0]
    ring_v[pos0 % window] = values[pos0]
    ring_kh = session.upload(ring_k)
    ring_vh = session.upload(ring_v)
    span = 256
    q, rel, gate = heads(0)
    session.dispatch(
        "attend_group",
        [
            proj_h,
            ring_kh,
            ring_vh,
            (proj_h, ROW_REL * 4),
            table_h,
            sink_h,
            (proj_h, ROW_GATE * 4),
            session.zeros(4 * 24 * 132 * 4),
            session.zeros(64 * 4),
            out,
        ],
        [u32(pos0), u32(window), u32(span), u32(window - 1)],
        (1, 8, -(-window // span)),
    )
    found = session.download_f32(out, 3072).reshape(24, 128)
    reference = _attention_reference(q, rel, gate, sink[0], keys, values, table, pos0, window)
    _close(found, reference, 1e-4, "decode attention ring")


def _repo_id(text: str) -> str:
    """``org/name``, or the same id taken from a huggingface.co model link."""
    raw = text.strip().rstrip("/")
    if "://" in raw or raw.startswith(("huggingface.co/", "www.huggingface.co/", "hf.co/")):
        if "://" not in raw:
            raw = "https://" + raw
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if host not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
            message = "repository link is a huggingface.co model"
            raise argparse.ArgumentTypeError(message)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            message = "repository is org/name or a huggingface.co link"
            raise argparse.ArgumentTypeError(message)
        raw = f"{parts[0]}/{parts[1]}"
    if _REPO_ID.fullmatch(raw) is None:
        message = "repository is org/name or a huggingface.co link"
        raise argparse.ArgumentTypeError(message)
    return raw


def _folder_name(text: str) -> str:
    """One directory name under ``models/``. That name is the registry id."""
    if _FOLDER.fullmatch(text) is None:
        message = "folder name is not a single name"
        raise argparse.ArgumentTypeError(message)
    return text


def register_folder(
    folder: Path,
    *,
    alias: str | None,
    source: Path | None = None,
    root: Path | None = None,
) -> None:
    """Write the registry row. ``source`` is the BF16 folder when it is not ``folder``."""
    from favlo.registry import load_registry, register_quantized, save_registry
    from inkling_2_mini.portable.sampling import CONTEXT_LENGTH, PRESERVE_THINKING

    register_quantized(
        folder.name,
        source=folder if source is None else source,
        alias=alias,
        directory=folder,
        root=root,
        pipeline=PIPELINE,
        runtime="metal41",
        max_prompt_tokens=CONTEXT_LENGTH,
        kv_cache="half",
    )
    data = load_registry(root)
    row = data["models"][folder.name]
    row["preserve_thinking"] = PRESERVE_THINKING
    save_registry(data, root)


def _setup(args: argparse.Namespace) -> int:
    """Download the checkpoint, bake the additive packs, and register that folder."""
    import os

    from favlo.console import Console, publish_count
    from inkling_2_mini.metal41.fetch import (
        REPO,
        download,
        list_files,
        remaining_download,
        remaining_expert,
        shortfall,
    )

    repo = args.repo or REPO
    name = args.name or (FOLDER if repo == REPO else repo.split("/", 1)[1])
    folder = _folder() if name == FOLDER else _folder().parent / name
    ui = Console.from_env(getattr(args, "console", None), pipeline=PIPELINE)
    if os.environ.get("FAVLO_SKIP_PLATE") != "1":
        ui.plate()
    bar = None
    try:
        try:
            files = list_files(repo=repo)
        except (OSError, RuntimeError, ValueError) as exc:
            from inkling_2_mini.metal41.record import ready

            present = (folder / "config.json").is_file() and (
                ready(folder) or (folder / "model-00001-of-00001.safetensors").is_file()
            )
            if not present:
                raise
            files = []
            ui.log(str(exc))
            ui.log(f"using the checkpoint already in {folder.name}")
        need_download = remaining_download(folder, files)
        need_experts = remaining_expert(folder)
        experts_ready = need_experts == 0
        need_record = need_experts
        message = shortfall(need_download, need_record, folder, folder)
        if message is not None:
            ui.log(message)
            return 1
        total = sum(item.size for item in files)
        bar = ui.begin("download", total=total, unit="bytes", detail=repo)

        def _move(done: int, whole: int, label: str) -> None:
            assert bar is not None
            bar.total = whole
            bar.detail = label
            publish_count(ui, bar, done)

        fetched = download(folder, files, _move, repo=repo)
        ui.end(bar, f"downloaded {repo}")
        bar = None
        if any(name.startswith("model-") and name.endswith(".safetensors") for name in fetched):
            from inkling_2_mini.metal41.calibration import HESSIANS_NAME
            from inkling_2_mini.metal41.gptq import PACK_NAME
            from inkling_2_mini.metal41.record import EXPERTS, MANIFEST, MTP, RESIDENT, VISION
            from inkling_2_mini.metal41.resident import EXPERT_PACK
            from inkling_2_mini.metal41.vision import PACK_NAME as VISION_PACK
            from inkling_2_mini.metal41.weights import DENSE_PACK

            for stale in (
                PACK_NAME,
                HESSIANS_NAME,
                EXPERT_PACK,
                VISION_PACK,
                DENSE_PACK,
                MANIFEST,
                RESIDENT,
                EXPERTS,
                VISION,
                MTP,
            ):
                (folder / stale).unlink(missing_ok=True)
            experts_ready = False
        from inkling_2_mini.metal41.repack import repack

        repack(folder, log=ui.log)
        register_folder(folder, alias="inkling_2_mini" if name == FOLDER else None)
        from inkling_2_mini.metal41.record import publish, ready

        if ready(folder):
            publish(folder, log=ui.log)
        elif not experts_ready:
            ui.log("baking the expert fp4 planes")
            _bake(folder)
        else:
            publish(folder, log=ui.log)
        ui.log(f"registered {folder.name}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        ui.report(f"{exc}\n")
        ui.log("the next setup continues from the files already written")
        return 1
    finally:
        if bar is not None and not bar.closed:
            ui.end(bar, "")
        ui.close()


def _complete(max_memory: str | None, limit: int) -> int:
    from inkling_2_mini.portable import open_runtime
    from inkling_2_mini.portable.chat import STOP, decode_text, encode_chat

    folder = _folder()
    runtime = open_runtime(folder, owner="inkling_2_mini:complete", max_memory=max_memory)
    print(f"context {runtime.context}", flush=True)
    try:
        prompt = encode_chat(
            runtime.root,
            [{"role": "user", "content": CARROT_PROMPT}],
            None,
            False,
            effort="none",
        )
        tokens = runtime.generate(prompt, limit, STOP, temperature=0.0, top_p=1.0)
        print(decode_text(runtime.root, tokens))
        if limit >= 32:
            fill = runtime.generate(prompt, limit, STOP, temperature=0.0, top_p=1.0, fill=True)
            print(f"fill {len(fill)} tokens", flush=True)
    finally:
        runtime.close()
    return 0


def _models() -> int:
    from favlo.registry import load_registry

    data = load_registry()
    for folder, entry in data.get("models", {}).items():
        if isinstance(entry, dict) and entry.get("pipeline") == PIPELINE:
            print(f"{folder} alias {entry.get('alias') or ''}")
    return 0


__all__ = ["CARROT_PROMPT", "EOS", "main"]
