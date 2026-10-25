"""Time one-token dispatches on the Qwen3.8-27B gate, 17408 by 5120.

One command buffer, then a wait. The sample is the median of five dispatches
after one discarded sample. Payloads are constant finite codes. An E8M0 plane,
when present, is scale byte 127. The rate divides the bytes moved by the median
seconds and by 2^30.
"""

# ruff: noqa: E501
from __future__ import annotations

import statistics
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import Metal
import numpy as np

from favlo.smoke_lock import hold, release

ROOT = Path(__file__).resolve().parents[1]
SHADER = Path(__file__).resolve().parent / "synthetic_decode.metal"
N = 17408
K = 5120
EXPECT = 5120.0
GIB = 2**30

_BITS = {
    "half": 16,
    "bf16": 16,
    "f32": 32,
    "fp4": 4,
    "e4m3": 8,
    "e5m2": 8,
    "i8": 8,
    "i4": 4,
    "i2": 2,
}
_DEST_BYTES = {"float": 4, "half": 2, "int32": 4}


def plane_bytes(count: int, kind: str) -> int:
    return count * _BITS[kind] // 8


PATTERNS = {
    "half": np.float16(1.0).tobytes(),
    "bf16": struct.pack("<H", 0x3F80),
    "f32": struct.pack("<f", 1.0),
    "fp4": bytes([0x22]),
    "e4m3": bytes([0x38]),
    "e5m2": bytes([0x3C]),
    "e8m0": bytes([127]),
    "i8": bytes([1]),
    "i4": bytes([0x11]),
    "i2": bytes([0x55]),
}


@dataclass(frozen=True)
class Case:
    process: str
    kernel: str
    left: str
    right: str
    dest: str
    scale: str
    rows: int
    tile: str
    threads: int
    layout: str
    act: str
    weight: str
    act_bytes: int
    weight_bytes: int
    dest_bytes: int
    tile_m: int = 0
    tile_n: int = 0
    groups_y: int = 1
    simd: int = 4
    scaled_flag: int = 1
    int2: int = 0
    copy_bytes: int = 0

    def traffic(self) -> int:
        if self.layout == "copy":
            return self.copy_bytes * 2
        rows = 1 if self.layout.startswith("gemv") else self.rows
        act = plane_bytes(rows * K, self.act)
        weight = plane_bytes(N * K, self.weight)
        out_rows = 1 if self.layout.startswith("gemv") else self.rows
        dest = out_rows * N * _DEST_BYTES[self.dest]
        scales = 0
        if self.scale in {"yes", "both"}:
            scales += N * (K // 32)
        if self.scale == "both":
            scales += self.rows * (K // 32)
        return act + weight + scales + dest


def cases() -> list[Case]:
    rows: list[Case] = []

    def add(**kwargs: object) -> None:
        rows.append(Case(**kwargs))  # type: ignore[arg-type]

    def matmul(
        kernel: str,
        left: str,
        right: str,
        dest: str,
        scale: str,
        tile_m: int,
        tile_n: int,
        simd: int,
        layout: str,
        act: str,
        weight: str,
        act_bytes: int,
        weight_bytes: int,
        dest_bytes: int,
        extent: int | None = None,
    ) -> None:
        used = tile_m if extent is None else extent
        add(
            process="matmul2d",
            kernel=kernel,
            left=left,
            right=right,
            dest=dest,
            scale=scale,
            rows=used,
            tile=f"{tile_m}×{tile_n}",
            threads=32 * simd,
            layout=layout,
            act=act,
            weight=weight,
            act_bytes=act_bytes,
            weight_bytes=weight_bytes,
            dest_bytes=dest_bytes,
            tile_m=tile_m,
            tile_n=tile_n,
            simd=simd,
        )

    matmul("mm64_h_fp4", "half", "fp4 E2M1", "float", "yes", 64, 64, 4, "scaled", "half", "fp4", 2, 1, 4)
    matmul("mm64_h_fp4", "half", "fp4 E2M1", "float", "yes", 64, 64, 4, "scaled", "half", "fp4", 2, 1, 4, 1)
    matmul("mm64_h_fp4_raw", "half", "fp4 E2M1", "float", "no", 64, 64, 4, "raw", "half", "fp4", 2, 1, 4)
    matmul("mm64_h_fp4_h", "half", "fp4 E2M1", "half", "yes", 64, 64, 4, "scaled", "half", "fp4", 2, 1, 2)
    matmul("mm64_h_e4", "half", "fp8 E4M3", "float", "yes", 64, 64, 4, "scaled", "half", "e4m3", 2, 1, 4)
    matmul("mm64_h_e4", "half", "fp8 E4M3", "float", "yes", 64, 64, 4, "scaled", "half", "e4m3", 2, 1, 4, 1)
    matmul("mm64_h_e4_raw", "half", "fp8 E4M3", "float", "no", 64, 64, 4, "raw", "half", "e4m3", 2, 1, 4)
    matmul("mm64_h_e5", "half", "fp8 E5M2", "float", "yes", 64, 64, 4, "scaled", "half", "e5m2", 2, 1, 4)
    matmul("mm64_h_h", "half", "half", "float", "no", 64, 64, 4, "plain", "half", "half", 2, 2, 4)
    matmul("mm64_h_h", "half", "half", "float", "no", 64, 64, 4, "plain", "half", "half", 2, 2, 4, 1)
    matmul("mm64_bf_bf", "bfloat", "bfloat", "float", "no", 64, 64, 4, "plain", "bf16", "bf16", 2, 2, 4)
    matmul("mm64_i8", "int8", "int8", "int32", "no", 64, 64, 4, "plain", "i8", "i8", 1, 1, 4)
    matmul("mm64_i8", "int8", "int8", "int32", "no", 64, 64, 4, "plain", "i8", "i8", 1, 1, 4, 1)
    matmul("mm64_fp4_fp4", "fp4 E2M1", "fp4 E2M1", "float", "both", 64, 64, 4, "both", "fp4", "fp4", 1, 1, 4)
    matmul("mm64_e4_e4", "fp8 E4M3", "fp8 E4M3", "float", "both", 64, 64, 4, "both", "e4m3", "e4m3", 1, 1, 4)
    matmul("mm64_h_i4", "half", "int4", "float", "yes", 64, 64, 4, "scaled", "half", "i4", 2, 1, 4)
    matmul("mm64_h_i2", "half", "int2", "float", "yes", 64, 64, 4, "scaled", "half", "i2", 1, 1, 4)
    matmul("mm32_h_fp4", "half", "fp4 E2M1", "float", "yes", 32, 128, 4, "scaled", "half", "fp4", 2, 1, 4)
    matmul("mm32_h_e4", "half", "fp8 E4M3", "float", "yes", 32, 128, 4, "scaled", "half", "e4m3", 2, 1, 4)
    matmul("mm32_i8", "int8", "int8", "int32", "no", 32, 128, 4, "plain", "i8", "i8", 1, 1, 4)
    matmul("mm8_h_fp4", "half", "fp4 E2M1", "float", "yes", 8, 256, 2, "scaled", "half", "fp4", 2, 1, 4)
    matmul("mm8_h_e4", "half", "fp8 E4M3", "float", "yes", 8, 256, 2, "scaled", "half", "e4m3", 2, 1, 4)
    matmul("mm8_i8", "int8", "int8", "int32", "no", 8, 256, 2, "plain", "i8", "i8", 1, 1, 4)
    def gemv(
        kernel: str,
        left: str,
        right: str,
        dest: str,
        scale: str,
        threads: int,
        layout: str,
        act: str,
        weight: str,
        act_bytes: int,
        weight_bytes: int,
        dest_bytes: int,
        **flags: int,
    ) -> None:
        add(
            process="gemv",
            kernel=kernel,
            left=left,
            right=right,
            dest=dest,
            scale=scale,
            rows=1,
            tile=f"{threads} thr",
            threads=threads,
            layout=layout,
            act=act,
            weight=weight,
            act_bytes=act_bytes,
            weight_bytes=weight_bytes,
            dest_bytes=dest_bytes,
            scaled_flag=flags.get("scaled", 1),
            int2=flags.get("int2", 0),
        )

    gemv("gemv_fp4_x4", "float", "fp4 E2M1", "float", "yes", 32, "gemv_fp4", "f32", "fp4", 4, 1, 4)
    gemv("gemv_fp4_x4", "float", "fp4 E2M1", "float", "no", 32, "gemv_fp4", "f32", "fp4", 4, 1, 4, scaled=0)
    gemv("gemv_fp8_x4", "float", "fp8 E4M3", "float", "yes", 160, "gemv_fp8", "f32", "e4m3", 4, 1, 4)
    gemv("gemv_fp8_x4", "float", "fp8 E4M3", "float", "yes", 32, "gemv_fp8", "f32", "e4m3", 4, 1, 4)
    gemv("gemv_sub4", "float", "int4", "float", "yes", 32, "gemv_sub", "f32", "i4", 4, 1, 4)
    gemv("gemv_sub4", "float", "int2", "float", "yes", 32, "gemv_sub", "f32", "i2", 4, 1, 4, int2=1)
    gemv("gemv_i8f_x4", "float", "int8", "float", "no", 160, "gemv_wide", "f32", "i8", 4, 1, 4)
    gemv("gemv_i8_x4", "int8", "int8", "int32", "no", 160, "gemv_wide", "i8", "i8", 1, 1, 4)
    gemv("gemv_half_x4", "float", "half", "float", "no", 160, "gemv_wide", "f32", "half", 4, 2, 4)
    gemv("gemv_bf16_x4", "float", "bfloat", "float", "no", 160, "gemv_wide", "f32", "bf16", 4, 2, 4)

    fp4_image = N * K // 2 + N * (K // 32)
    add(
        process="copy",
        kernel="copy_uint4",
        left="uint4",
        right="uint4",
        dest="uint4",
        scale="no",
        rows=0,
        tile="512 MiB",
        threads=256,
        layout="copy",
        act="",
        weight="",
        act_bytes=0,
        weight_bytes=0,
        dest_bytes=0,
        copy_bytes=512 * 2**20,
    )
    add(
        process="copy",
        kernel="copy_uint4",
        left="uint4",
        right="uint4",
        dest="uint4",
        scale="no",
        rows=0,
        tile="fp4 image",
        threads=256,
        layout="copy",
        act="",
        weight="",
        act_bytes=0,
        weight_bytes=0,
        dest_bytes=0,
        copy_bytes=fp4_image,
    )
    add(
        process="copy",
        kernel="copy_uint4",
        left="uint4",
        right="uint4",
        dest="uint4",
        scale="no",
        rows=0,
        tile="int8 image",
        threads=256,
        layout="copy",
        act="",
        weight="",
        act_bytes=0,
        weight_bytes=0,
        dest_bytes=0,
        copy_bytes=N * K,
    )
    return rows


def error_text(error: object) -> str:
    if error is None:
        return "compile failed"
    text = getattr(error, "localizedDescription", None)
    if callable(text):
        return str(text())
    return str(error)


def compile_library(device: object, source: str) -> tuple[object | None, str]:
    options = Metal.MTLCompileOptions.alloc().init()
    options.setLanguageVersion_((4 << 16) + 1)
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None:
        return None, error_text(error)
    return library, ""


def alloc(device: object, nbytes: int) -> object:
    buffer = device.newBufferWithLength_options_(nbytes, 0)
    if buffer is None:
        message = f"device refused {nbytes} bytes"
        raise RuntimeError(message)
    return buffer


def fill(buf: object, nbytes: int, pattern: bytes) -> None:
    raw = buf.contents().as_buffer(nbytes)
    view = np.frombuffer(raw, dtype=np.uint8)
    if len(pattern) == 1:
        view[:] = pattern[0]
        return
    repeats = nbytes // len(pattern)
    tiled = np.frombuffer(pattern * repeats, dtype=np.uint8)
    view[: tiled.size] = tiled


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def read_dest(buf: object, kind: str) -> float:
    raw = buf.contents().as_buffer(4)
    if kind == "int32":
        return float(struct.unpack_from("<i", raw)[0])
    if kind == "half":
        return float(np.frombuffer(raw[:2], dtype=np.float16)[0])
    return float(struct.unpack_from("<f", raw)[0])


def bind(encoder: object, bufs: list[object], constants: list[bytes], groups: tuple[int, int, int], threads: int) -> None:
    for index, buf in enumerate(bufs):
        encoder.setBuffer_offset_atIndex_(buf, 0, index)
    base = len(bufs)
    for index, blob in enumerate(constants):
        encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(groups[0], groups[1], groups[2]),
        Metal.MTLSizeMake(threads, 1, 1),
    )


def dispatch(queue: object, pipe: object, bufs: list[object], constants: list[bytes], groups: tuple[int, int, int], threads: int) -> tuple[float | None, str]:
    command = queue.commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipe)
    bind(encoder, bufs, constants, groups, threads)
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if int(command.status()) != int(Metal.MTLCommandBufferStatusCompleted):
        return None, error_text(command.error())
    ms = (float(command.GPUEndTime()) - float(command.GPUStartTime())) * 1e3
    return ms, ""


def prepare(case: Case, bufs: dict[str, object]) -> tuple[list[object], list[bytes], tuple[int, int, int]]:
    if case.layout == "copy":
        count = case.copy_bytes // 16
        fill(bufs["copy_src"], case.copy_bytes, bytes([0x11]))
        return [bufs["copy_src"], bufs["copy_dst"]], [u32(count)], (count // 256, 1, 1)
    guard = 1 if case.layout.startswith("gemv") else max(case.tile_m, case.rows)
    fill(bufs["act"], plane_bytes(guard * K, case.act), PATTERNS[case.act])
    fill(bufs["weight"], plane_bytes(N * K, case.weight), PATTERNS[case.weight])
    if case.scale != "no":
        fill(bufs["scale"], N * (K // 32), PATTERNS["e8m0"])
    if case.scale == "both":
        fill(bufs["left_scale"], case.rows * (K // 32), PATTERNS["e8m0"])
    dest_rows = max(case.rows, case.tile_m, 1)
    fill(bufs["dest"], dest_rows * N * _DEST_BYTES[case.dest], bytes([0]))
    if case.layout == "scaled":
        blobs = [bufs["act"], bufs["weight"], bufs["scale"], bufs["dest"]]
        constants = [u32(case.rows), u32(N), u32(K)]
        groups = (N // case.tile_n, 1, 1)
    elif case.layout == "raw":
        blobs = [bufs["act"], bufs["weight"], bufs["dest"]]
        constants = [u32(case.rows), u32(N), u32(K)]
        groups = (N // case.tile_n, 1, 1)
    elif case.layout == "plain":
        blobs = [bufs["act"], bufs["weight"], bufs["dest"]]
        constants = [u32(case.rows), u32(N), u32(K)]
        groups = (N // case.tile_n, 1, 1)
    elif case.layout == "both":
        blobs = [bufs["act"], bufs["weight"], bufs["left_scale"], bufs["scale"], bufs["dest"]]
        constants = [u32(case.rows), u32(N), u32(K)]
        groups = (N // case.tile_n, 1, 1)
    elif case.layout == "gemv_fp4":
        blobs = [bufs["act"], bufs["weight"], bufs["scale"], bufs["dest"]]
        constants = [u32(N), u32(K), u32(case.scaled_flag)]
        groups = (N // 4, 1, 1)
    elif case.layout == "gemv_sub":
        blobs = [bufs["act"], bufs["weight"], bufs["scale"], bufs["dest"]]
        constants = [u32(N), u32(K), u32(case.int2)]
        groups = (N // 4, 1, 1)
    elif case.layout == "gemv_fp8":
        blobs = [bufs["act"], bufs["weight"], bufs["scale"], bufs["dest"]]
        constants = [u32(N), u32(K)]
        groups = (N // 4, 1, 1)
    else:
        blobs = [bufs["act"], bufs["weight"], bufs["dest"]]
        constants = [u32(N), u32(K)]
        groups = (N // 4, 1, 1)
    return blobs, constants, groups


def machine(device: object) -> str:
    mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    return f"{device.name()} with {mem / GIB:.0f} GB"


def main() -> int:
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        print("no Metal device")
        return 1
    print(machine(device))
    library, message = compile_library(device, SHADER.read_text())
    if library is None:
        print(message)
        return 1
    queue = device.newCommandQueue()
    weight_bytes = N * K * 2
    bufs = {
        "act": alloc(device, 64 * K * 4),
        "weight": alloc(device, weight_bytes),
        "scale": alloc(device, N * (K // 32)),
        "left_scale": alloc(device, 64 * (K // 32)),
        "dest": alloc(device, 64 * N * 4),
        "copy_src": alloc(device, 512 * 2**20),
        "copy_dst": alloc(device, 512 * 2**20),
    }
    print("| Process | Kernel | M | Tile | Left | Right | Destination | E8M0 | ms | GiB/s | Check |")
    print("|---|---|---:|---|---|---|---|---|---:|---:|---|")
    for case in cases():
        pipe = library.newFunctionWithName_(case.kernel)
        if pipe is None:
            print(f"| {case.process} | {case.kernel} | {case.rows} | {case.tile} | {case.left} | {case.right} | {case.dest} | {case.scale} |  |  | missing |")
            continue
        state = device.newComputePipelineStateWithFunction_error_(pipe, None)
        if isinstance(state, tuple):
            pipeline, error = state
        else:
            pipeline, error = state, None
        if pipeline is None:
            print(f"| {case.process} | {case.kernel} | {case.rows} | {case.tile} | {case.left} | {case.right} | {case.dest} | {case.scale} |  |  | {error_text(error)} |")
            continue
        samples: list[float] = []
        note = ""
        blobs: list[object] = []
        for _ in range(6):
            blobs, constants, groups = prepare(case, bufs)
            ms, note = dispatch(queue, pipeline, blobs, constants, groups, case.threads)
            if ms is None:
                break
            samples.append(ms)
        if len(samples) < 6:
            detail = note.replace("|", "/")
            print(f"| {case.process} | {case.kernel} | {case.rows} | {case.tile} | {case.left} | {case.right} | {case.dest} | {case.scale} |  |  | {detail} |")
            continue
        median = statistics.median(samples[1:])
        rate = (case.traffic() / (median / 1e3)) / GIB
        check = "copy"
        if case.layout != "copy":
            got = read_dest(blobs[-1] if case.layout != "both" else blobs[4], case.dest)
            # dest is not always the last buffer for both: it is last. For scaled dest is last.
            if case.layout == "both":
                got = read_dest(blobs[4], case.dest)
            check = "ok" if abs(got - EXPECT) <= 0.5 else f"{got:.3g}"
        print(
            f"| {case.process} | {case.kernel} | {case.rows} | {case.tile} | {case.left} | {case.right} | {case.dest} | {case.scale} | {median:.3f} | {rate:.2f} | {check} |"
        )
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    created = hold(root=ROOT, pipeline="shared", command="synthetic-decode")
    try:
        raise SystemExit(main())
    finally:
        if created:
            release(root=ROOT)
