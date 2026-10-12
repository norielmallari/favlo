"""Qwen3.8-Flash-Next pipeline.

This file owns the model. ``madmetal --qwen38-flash-next`` loads it by path.
``probe`` and ``matmul`` measure the GPU record. ``serve`` steps this model's
shaders and answers ``/v1/chat/completions``.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import math
import mmap
import os
import shutil
import struct
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import objc

BLOCK = 32
SENTINEL = 0x5A
HOST_BYTES = 4096
PAGE = 16384
MIB = 1024 * 1024
EXPERT_SLOT = 3 * MIB
NGRAM_SLOT = 128
NGRAM_WIDTH = 160
RECORD_VERSION = 1
DEFAULT_RECORD = Path.home() / ".madmetal" / "qwen38-flash-next"

# MTLTensorDataType values from the macOS 27 SDK headers.
TENSOR_INT8 = 45
TENSOR_INT4 = 143
TENSOR_E8M0 = 145
TENSOR_E5M2 = 141
TENSOR_E4M3 = 142
TENSOR_INT2 = 150
PLANE_DATA = 0
PLANE_SCALES = 1

# fmt: off
INT4_ORDER_PATTERN = [0, 1, 2, 4, 8, 15, 7, 3, 5, 6, 9, 10, 11, 12, 13, 14] * 2
INT2_ORDER_PATTERN = [0, 1, 2, 3, 1, 0, 3, 2, 2, 2, 1, 1, 0, 3, 0, 1] * 2
# Packed bytes the device stores. Nibble and crumb order are not in this layout;
# the shader unpack decides which bits are element 0.
INT4_PACKED = bytes([
    0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
    0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
])
INT2_PACKED = bytes([0x1B, 0xE4, 0xA7, 0x5B, 0x0F, 0xF0, 0x36, 0xC9])
INT8_PATTERN = [
    -128, -127, -2, -1, 0, 1, 2, 126, 127, 50, -50, 64, -64, 7, -8, 15,
    -15, 32, -32, 100, -100, 3, -3, 9, -9, 16, -16, 40, -40, 5, -5, 8,
]
FP8_PATTERN = [
    0x00, 0x80, 0x38, 0x40, 0x3C, 0x3E, 0x48, 0xC8,
    0x1C, 0x9C, 0x04, 0x01, 0x7E, 0xFE, 0x7F, 0xFF,
    0x30, 0xB0, 0x28, 0xA8, 0x18, 0x98, 0x08, 0x88,
    0x44, 0xC4, 0x50, 0xD0, 0x60, 0xE0, 0x70, 0xF0,
]
# fmt: on

SHADER = r"""
#include <metal_stdlib>
using namespace metal;

kernel void copy_bytes(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dst[gid] = src[gid];
    }
}

kernel void read_int8(
    device const char* src [[buffer(0)]],
    device int* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    for (uint i = 0; i < 32; ++i) {
        dst[i] = int(src[i]);
    }
}

kernel void unpack_int4(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    for (uint group = 0; group < 2; ++group) {
        packed_vec<uchar, 8> bytes;
        for (uint i = 0; i < 8; ++i) {
            bytes[i] = src[group * 8 + i];
        }
        auto values = unpack<float>(packed_int4b<16>(bytes));
        for (uint i = 0; i < 16; ++i) {
            dst[group * 16 + i] = values[i];
        }
    }
}

kernel void unpack_e4m3(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    for (uint group = 0; group < 4; ++group) {
        packed_vec<uchar, 8> bytes;
        for (uint i = 0; i < 8; ++i) {
            bytes[i] = src[group * 8 + i];
        }
        auto values = unpack<float>(packed_metal_fp8_e4m3<8>(bytes));
        for (uint i = 0; i < 8; ++i) {
            dst[group * 8 + i] = values[i];
        }
    }
}

kernel void unpack_e5m2(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    for (uint group = 0; group < 4; ++group) {
        packed_vec<uchar, 8> bytes;
        for (uint i = 0; i < 8; ++i) {
            bytes[i] = src[group * 8 + i];
        }
        auto values = unpack<float>(packed_metal_fp8_e5m2<8>(bytes));
        for (uint i = 0; i < 8; ++i) {
            dst[group * 8 + i] = values[i];
        }
    }
}

kernel void reconstruct_int4(
    device const uchar* src [[buffer(0)]],
    device const uchar* scale [[buffer(1)]],
    device float* dst [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    float factor = exp2(float(scale[0]) - 127.0f);
    for (uint group = 0; group < 2; ++group) {
        packed_vec<uchar, 8> bytes;
        for (uint i = 0; i < 8; ++i) {
            bytes[i] = src[group * 8 + i];
        }
        auto values = unpack<float>(packed_int4b<16>(bytes));
        for (uint i = 0; i < 16; ++i) {
            dst[group * 16 + i] = values[i] * factor;
        }
    }
}
"""


def pack_int4(codes: Sequence[int], *, low_nibble_first: bool) -> bytes:
    if len(codes) % 2:
        message = "int4 packing needs an even number of codes"
        raise ValueError(message)
    out = bytearray()
    for index in range(0, len(codes), 2):
        first = codes[index] & 0xF
        second = codes[index + 1] & 0xF
        if low_nibble_first:
            out.append(first | (second << 4))
        else:
            out.append((first << 4) | second)
    return bytes(out)


def unpack_int4(raw: bytes, *, low_nibble_first: bool, count: int = BLOCK) -> list[int]:
    codes: list[int] = []
    for byte in raw:
        if low_nibble_first:
            pair = (byte & 0xF, (byte >> 4) & 0xF)
        else:
            pair = ((byte >> 4) & 0xF, byte & 0xF)
        codes.extend(pair)
        if len(codes) >= count:
            return codes[:count]
    return codes


def pack_int2(codes: Sequence[int], *, low_bits_first: bool) -> bytes:
    if len(codes) % 4:
        message = "int2 packing needs a multiple of four codes"
        raise ValueError(message)
    out = bytearray()
    for index in range(0, len(codes), 4):
        group = [codes[index + offset] & 0x3 for offset in range(4)]
        byte = 0
        for offset, code in enumerate(group):
            shift = 2 * offset if low_bits_first else 6 - 2 * offset
            byte |= code << shift
        out.append(byte)
    return bytes(out)


def unpack_int2(raw: bytes, *, low_bits_first: bool, count: int = BLOCK) -> list[int]:
    codes: list[int] = []
    for byte in raw:
        for offset in range(4):
            shift = 2 * offset if low_bits_first else 6 - 2 * offset
            codes.append((byte >> shift) & 0x3)
            if len(codes) >= count:
                return codes[:count]
    return codes


def sign_extend(code: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    masked = code & ((1 << bits) - 1)
    return masked - (1 << bits) if masked & sign else masked


def decode_e8m0(byte: int) -> float:
    if byte & 0xFF == 0xFF:
        return math.nan
    return 2.0 ** ((byte & 0xFF) - 127)


# Positive endpoint of the signed code. int4 is −8..7, int8 is −128..127, int2 is −2..1.
QMAX = {2: 1, 4: 7, 8: 127}


def ceil_log2(value: float) -> int:
    """Smallest integer p such that 2^p >= value, for value > 0."""
    mantissa, exponent = math.frexp(value)
    if mantissa == 0.5:
        return exponent - 1
    return exponent


def e8m0_scale_byte(amax: float, qmax: int) -> int:
    """Power of two at or above amax/qmax, stored with bias 127. Never 0xFF."""
    if qmax <= 0:
        message = "qmax must be positive"
        raise ValueError(message)
    if not math.isfinite(amax) or amax <= 0.0:
        return 0
    power = ceil_log2(amax / qmax)
    biased = power + 127
    if biased < 0:
        return 0
    if biased > 254:
        return 254
    return biased


def quantize_block(values: Sequence[float], bits: int) -> tuple[list[int], int]:
    """One block of 32. Returns signed codes and the E8M0 scale byte.

    The scale rounds up so every finite input fits in the signed range.
    Codes use round-ties-to-even, then saturate.
    """
    if bits not in QMAX:
        message = f"unsupported element width: {bits}"
        raise ValueError(message)
    if len(values) != BLOCK:
        message = f"a block is {BLOCK} values"
        raise ValueError(message)
    qmax = QMAX[bits]
    qmin = -(1 << (bits - 1))
    amax = 0.0
    for value in values:
        if not math.isfinite(value):
            message = "a block value must be finite"
            raise ValueError(message)
        magnitude = abs(value)
        if magnitude > amax:
            amax = magnitude
    scale_byte = e8m0_scale_byte(amax, qmax)
    scale = decode_e8m0(scale_byte)
    codes: list[int] = []
    for value in values:
        # Python 3 round is ties-to-even, which is the MX rounding mode.
        code = int(round(value / scale))
        if code < qmin:
            code = qmin
        elif code > qmax:
            code = qmax
        codes.append(code)
    return codes, scale_byte


def decode_e4m3(byte: int) -> float:
    """OCP E4M3FN. Exponent 15 is finite except mantissa 7, which is NaN."""
    value = byte & 0xFF
    sign = -1.0 if value & 0x80 else 1.0
    exponent = (value >> 3) & 0xF
    mantissa = value & 0x7
    if exponent == 0:
        return sign * (mantissa / 8.0) * 2.0**-6
    if exponent == 0xF and mantissa == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def decode_e5m2(byte: int) -> float:
    """OCP E5M2. Exponent 31 is inf when the mantissa is 0, otherwise NaN."""
    value = byte & 0xFF
    sign = -1.0 if value & 0x80 else 1.0
    exponent = (value >> 2) & 0x1F
    mantissa = value & 0x3
    if exponent == 0:
        return sign * (mantissa / 4.0) * 2.0**-14
    if exponent == 0x1F:
        if mantissa == 0:
            return math.copysign(math.inf, sign)
        return math.copysign(math.nan, sign)
    return sign * (1.0 + mantissa / 4.0) * 2.0 ** (exponent - 15)


def _finite_equal(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    if math.isinf(left) or math.isinf(right):
        return left == right
    return abs(left - right) <= 1e-4 * max(1.0, abs(right))


def classify_subbyte(
    raw: bytes,
    codes: Sequence[int],
    *,
    bits: int,
) -> str | None:
    packed = bytes(raw[: len(codes) * bits // 8])
    if bits == 4:
        if packed == pack_int4(codes, low_nibble_first=True):
            return "low_nibble_is_element_0"
        if packed == pack_int4(codes, low_nibble_first=False):
            return "high_nibble_is_element_0"
        return None
    if bits == 2:
        if packed == pack_int2(codes, low_bits_first=True):
            return "low_bits_are_element_0"
        if packed == pack_int2(codes, low_bits_first=False):
            return "high_bits_are_element_0"
        return None
    message = f"unsupported sub-byte width: {bits}"
    raise ValueError(message)


def json_number(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _metal():
    import Metal

    return Metal


def _extents(values: Sequence[int]) -> Any:
    metal = _metal()
    objc.registerMetaDataForSelector(
        b"MTLTensorExtents",
        b"initWithRank:values:",
        {
            "arguments": {
                3: {
                    "type": b"^q",
                    "type_modifier": b"n",
                    "c_array_length_in_arg": 2,
                }
            }
        },
    )
    created = metal.MTLTensorExtents.alloc().initWithRank_values_(len(values), list(values))
    if created is None:
        message = f"MTLTensorExtents rejected {list(values)}"
        raise RuntimeError(message)
    return created


def _buffer_view(buffer: Any, count: int) -> memoryview:
    return memoryview(buffer.contents().as_buffer(count))


def _fill_buffer(buffer: Any, count: int, value: int) -> None:
    _buffer_view(buffer, count)[:] = bytes([value]) * count


def _write_u32(buffer: Any, value: int) -> None:
    _buffer_view(buffer, 4)[:] = value.to_bytes(4, "little")


def _buffer_bytes(buffer: Any, count: int) -> bytes:
    return bytes(_buffer_view(buffer, count))


def _objc_pair(result: Any) -> tuple[Any, str | None]:
    if isinstance(result, tuple) and len(result) == 2:
        value, error = result
        if error is not None:
            return None, str(error)
        return value, None
    return result, None


def _device_name(device: Any) -> str:
    name = device.name
    return str(name() if callable(name) else name)


class Roundtrip:
    """One shared device and one shader library for the block probe."""

    def __init__(self) -> None:
        metal = _metal()
        device = metal.MTLCreateSystemDefaultDevice()
        if device is None:
            message = "this Mac has no Metal device"
            raise RuntimeError(message)
        self.metal = metal
        self.device = device
        options = metal.MTLCompileOptions.alloc().init()
        # Metal 4.1. PyObjC does not export MTLLanguageVersion4_1.
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = _objc_pair(
            device.newLibraryWithSource_options_error_(SHADER, options, None)
        )
        if library is None:
            message = f"shader library failed to compile: {error}"
            raise RuntimeError(message)
        self.library = library

    def measure(self) -> dict[str, Any]:
        formats = {
            "int4": self._subbyte(TENSOR_INT4, 4, INT4_PACKED, "unpack_int4"),
            "int2": self._subbyte(TENSOR_INT2, 2, INT2_PACKED, None),
            "int8": self._int8(),
            "fp8_e4m3": self._fp8("fp8_e4m3", TENSOR_E4M3, decode_e4m3, "unpack_e4m3"),
            "fp8_e5m2": self._fp8("fp8_e5m2", TENSOR_E5M2, decode_e5m2, "unpack_e5m2"),
        }
        scale = self._scale_plane()
        ok = all(item.get("ok") for item in formats.values()) and bool(scale.get("ok"))
        return {
            "device": _device_name(self.device),
            "block": BLOCK,
            "ok": ok,
            "formats": formats,
            "scale": scale,
        }

    def _subbyte(
        self,
        data_type: int,
        bits: int,
        payload: bytes,
        unpack_kernel: str | None,
    ) -> dict[str, Any]:
        """Host slices are packed bytes. A shader unpack, when the SDK has one, names element 0."""
        try:
            written = self._write_read(data_type, payload, with_scale=False)
            packed_len = bits * BLOCK // 8
            raw = written["raw"][:packed_len]
            read_back = written["readback"][:packed_len]
            copy = self._shader_bytes(written["raw_buffer"], 0, packed_len)
            identity = raw == payload and read_back == payload and copy == payload
            shader_values: list[float] | None = None
            shader_order = None
            endpoints = None
            if unpack_kernel is not None:
                shader_values = self._shader_floats(unpack_kernel, written["raw_buffer"], 0)
                shader_order = _shader_int_order(raw, shader_values, bits)
                if shader_order is not None:
                    endpoints = _endpoint_table(raw, shader_values, shader_order)
            order_known = unpack_kernel is None or shader_order is not None
            report = {
                "ok": identity and order_known,
                "data_type": data_type,
                "host_transfer": "packed_bytes",
                "packed_bytes": packed_len,
                "storage": shader_order if shader_order is not None else "packed_bytes",
                "element_order": shader_order,
                "raw_hex": raw.hex(),
                "readback_hex": read_back.hex(),
                "gpu_copy_matches": copy == payload,
                "shader_unpack": (
                    None
                    if shader_values is None
                    else [json_number(value) for value in shader_values]
                ),
                "signed_codes": endpoints,
                "readback_size": written["readback_size"],
            }
            if bits == 2:
                report["note"] = (
                    "Metal 4.1 unpack() does not accept int2b, and this SDK has no "
                    "compute-encoder tensor binding, so element order is not measured"
                )
            return report
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _int8(self) -> dict[str, Any]:
        try:
            result = self._write_read(TENSOR_INT8, _byte_payload(INT8_PATTERN), with_scale=False)
            logical = _logical_ints(result["readback"], INT8_PATTERN)
            shader = self._shader_ints("read_int8", result["raw_buffer"], 0)
            return {
                "ok": logical == INT8_PATTERN and shader == INT8_PATTERN,
                "data_type": TENSOR_INT8,
                "storage": "one_byte_per_element",
                "logical": logical,
                "logical_matches_pattern": logical == INT8_PATTERN,
                "raw_hex": result["raw"][:BLOCK].hex(),
                "shader": shader,
                "readback_size": result["readback_size"],
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _fp8(self, name: str, data_type: int, decode, kernel: str) -> dict[str, Any]:
        del name
        try:
            result = self._write_read(data_type, bytes(FP8_PATTERN), with_scale=False)
            logical = list(result["readback"][:BLOCK])
            shader = self._shader_floats(kernel, result["raw_buffer"], 0)
            expected = [decode(byte) for byte in FP8_PATTERN]
            matches = all(
                _finite_equal(got, want) for got, want in zip(shader, expected, strict=True)
            )
            identity = logical == FP8_PATTERN and result["raw"][:BLOCK] == bytes(FP8_PATTERN)
            return {
                "ok": identity and matches,
                "data_type": data_type,
                "storage": "one_byte_per_element",
                "logical": logical,
                "raw_hex": result["raw"][:BLOCK].hex(),
                "byte_identity": identity,
                "shader_unpack": [json_number(value) for value in shader],
                "host_decode": [json_number(value) for value in expected],
                "shader_matches_ocp": matches,
                "readback_size": result["readback_size"],
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _scale_plane(self) -> dict[str, Any]:
        """One E8M0 scale for 32 int4 codes, then shader reconstruct."""
        scale_byte = 130
        try:
            written = self._write_read(
                TENSOR_INT4,
                INT4_PACKED,
                with_scale=True,
                scale=scale_byte,
            )
            shader = self._shader_floats(
                "reconstruct_int4",
                written["raw_buffer"],
                0,
                scale_buffer=written["raw_buffer"],
                scale_offset=128,
            )
            factor = decode_e8m0(scale_byte)
            order, expected, matches = _scaled_int4(written["raw"], shader, factor)
            scale_read = written["scale_read"]
            return {
                "ok": scale_read == scale_byte and matches and order is not None,
                "scale_wrote": scale_byte,
                "scale_read": scale_read,
                "exponent_bias": 127,
                "reconstruct": "code * 2^(e-127)",
                "storage": order,
                "shader": [json_number(value) for value in shader],
                "expected": [json_number(value) for value in expected],
                "shader_matches_reconstruct": matches,
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _buffer(self, length: int) -> Any:
        mode = self.metal.MTLResourceStorageModeShared
        buffer = self.device.newBufferWithLength_options_(length, mode)
        if buffer is None:
            message = f"Metal buffer of {length} bytes was not created"
            raise RuntimeError(message)
        return buffer

    def _write_read(
        self,
        data_type: int,
        payload: bytes,
        *,
        with_scale: bool,
        scale: int = 127,
    ) -> dict[str, Any]:
        metal = self.metal
        length = 256
        buffer = self._buffer(length)
        if buffer is None:
            message = "newBufferWithLength failed"
            raise RuntimeError(message)
        _fill_buffer(buffer, length, 0)
        descriptor = metal.MTLTensorDescriptor.alloc().init()
        descriptor.setDimensions_(_extents([BLOCK]))
        descriptor.setStrides_(_extents([1]))
        descriptor.setDataType_(data_type)
        descriptor.setUsage_(metal.MTLTensorUsageCompute)
        descriptor.setStorageMode_(metal.MTLStorageModeShared)
        attachments = metal.MTLTensorBufferAttachments.alloc().init()
        attachments.setBuffer_offset_forPlane_(buffer, 0, PLANE_DATA)
        if with_scale:
            plane = metal.MTLTensorAuxiliaryPlaneDescriptor.alloc().init()
            plane.setDataType_(TENSOR_E8M0)
            plane.setBlockFactors_(_extents([BLOCK]))
            planes = metal.MTLTensorAuxiliaryPlaneDescriptorMap.alloc().init()
            planes.setDescriptor_forPlane_(plane, PLANE_SCALES)
            descriptor.setAuxiliaryPlanes_(planes)
            attachments.setBuffer_offset_forPlane_(buffer, 128, PLANE_SCALES)
        tensor, error = _objc_pair(
            self.device.newTensorWithDescriptor_attachments_error_(descriptor, attachments, None)
        )
        if tensor is None:
            message = f"tensor create failed for data type {data_type}: {error}"
            raise RuntimeError(message)
        origin = _extents([0])
        dimensions = _extents([BLOCK])
        strides = _extents([1])
        host = payload + bytes([SENTINEL]) * (HOST_BYTES - len(payload))
        tensor.replaceSliceOrigin_sliceDimensions_withBytes_strides_(
            origin, dimensions, host, strides
        )
        if with_scale:
            scale_host = bytes([scale]) + bytes([SENTINEL]) * 63
            scale_holder = (ctypes.c_uint8 * len(scale_host)).from_buffer_copy(scale_host)
            tensor.replaceSliceOrigin_sliceDimensions_plane_withBytes_strides_(
                _extents([0]),
                _extents([1]),
                PLANE_SCALES,
                ctypes.addressof(scale_holder),
                _extents([1]),
            )
        readback = self._get_bytes(tensor, origin, dimensions, strides)
        scale_read = None
        if with_scale:
            scale_blob = self._get_plane(
                tensor,
                _extents([0]),
                _extents([1]),
                _extents([1]),
                PLANE_SCALES,
            )
            scale_read = scale_blob[0] if scale_blob else None
        return {
            "raw": _buffer_bytes(buffer, 64),
            "raw_buffer": buffer,
            "readback": readback,
            "readback_size": _changed_prefix(readback),
            "scale_read": scale_read,
        }

    def _get_bytes(self, tensor: Any, origin: Any, dimensions: Any, strides: Any) -> bytes:
        dest = (ctypes.c_uint8 * 256)()
        ctypes.memset(dest, SENTINEL, 256)
        tensor.getBytes_strides_fromSliceOrigin_sliceDimensions_(dest, strides, origin, dimensions)
        return bytes(dest)

    def _get_plane(
        self,
        tensor: Any,
        origin: Any,
        dimensions: Any,
        strides: Any,
        plane: int,
    ) -> bytes:
        dest = (ctypes.c_uint8 * 64)()
        ctypes.memset(dest, SENTINEL, 64)
        tensor.getBytes_strides_fromSliceOrigin_sliceDimensions_plane_(
            ctypes.addressof(dest), strides, origin, dimensions, plane
        )
        return bytes(dest)

    def _shader_floats(
        self,
        kernel: str,
        source: Any,
        source_offset: int,
        *,
        scale_buffer: Any | None = None,
        scale_offset: int = 0,
    ) -> list[float]:
        out = self._dispatch(kernel, source, source_offset, 128, scale_buffer, scale_offset)
        return list((ctypes.c_float * BLOCK).from_buffer_copy(out))

    def _shader_ints(self, kernel: str, source: Any, source_offset: int) -> list[int]:
        out = self._dispatch(kernel, source, source_offset, 128, None, 0)
        return list((ctypes.c_int * BLOCK).from_buffer_copy(out))

    def _shader_bytes(self, source: Any, source_offset: int, count: int) -> bytes:
        count_buf = self._buffer(4)
        _write_u32(count_buf, count)
        out = self._buffer(max(count, 16))
        _fill_buffer(out, max(count, 16), 0)
        encoder, command = self._encoder("copy_bytes")
        encoder.setBuffer_offset_atIndex_(source, source_offset, 0)
        encoder.setBuffer_offset_atIndex_(out, 0, 1)
        encoder.setBuffer_offset_atIndex_(count_buf, 0, 2)
        self._commit(encoder, command, max(count, 1))
        return _buffer_bytes(out, count)

    def _dispatch(
        self,
        kernel: str,
        source: Any,
        source_offset: int,
        out_bytes: int,
        scale_buffer: Any | None,
        scale_offset: int,
    ) -> bytes:
        out = self._buffer(out_bytes)
        _fill_buffer(out, out_bytes, 0)
        encoder, command = self._encoder(kernel)
        encoder.setBuffer_offset_atIndex_(source, source_offset, 0)
        if scale_buffer is None:
            encoder.setBuffer_offset_atIndex_(out, 0, 1)
        else:
            encoder.setBuffer_offset_atIndex_(scale_buffer, scale_offset, 1)
            encoder.setBuffer_offset_atIndex_(out, 0, 2)
        self._commit(encoder, command, 1)
        return _buffer_bytes(out, out_bytes)

    def _encoder(self, kernel: str) -> tuple[Any, Any]:
        function = self.library.newFunctionWithName_(kernel)
        if function is None:
            message = f"shader is missing {kernel}"
            raise RuntimeError(message)
        created, error = _objc_pair(
            self.device.newComputePipelineStateWithFunction_error_(function, None)
        )
        if created is None:
            message = f"pipeline state failed for {kernel}: {error}"
            raise RuntimeError(message)
        queue = self.device.newCommandQueue()
        command = queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(created)
        return encoder, command

    def _commit(self, encoder: Any, command: Any, threads: int) -> None:
        size = self.metal.MTLSizeMake(threads, 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(size, size)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()


def _byte_payload(codes: Sequence[int]) -> bytes:
    return bytes(code & 0xFF for code in codes)


def _changed_prefix(blob: bytes) -> int:
    count = 0
    while count < len(blob) and blob[count] != SENTINEL:
        count += 1
    return count


def _logical_ints(blob: bytes, pattern: Sequence[int]) -> list[int]:
    """Read the host slice as one signed byte per element when that width was written."""
    if _changed_prefix(blob) >= BLOCK:
        return [sign_extend(blob[index], 8) for index in range(BLOCK)]
    del pattern
    return list(blob[:BLOCK])


def _endpoint_table(raw: bytes, values: Sequence[float], order: str) -> dict[str, float | str]:
    low_first = order == "low_nibble_is_element_0"
    codes = unpack_int4(raw, low_nibble_first=low_first)
    table: dict[str, float | str] = {}
    for code, value in zip(codes, values, strict=True):
        table.setdefault(str(code), json_number(float(value)))
    return table


def _scaled_int4(
    raw: bytes, values: Sequence[float], factor: float
) -> tuple[str | None, list[float], bool]:
    for order, low_first in (
        ("low_nibble_is_element_0", True),
        ("high_nibble_is_element_0", False),
    ):
        signed = [sign_extend(code, 4) for code in unpack_int4(raw, low_nibble_first=low_first)]
        expected = [code * factor for code in signed]
        if all(_finite_equal(got, want) for got, want in zip(values, expected, strict=True)):
            return order, expected, True
    return None, [], False


def _shader_int_order(raw: bytes, values: Sequence[float], bits: int) -> str | None:
    if bits != 4:
        return None
    for name, low_first in (
        ("low_nibble_is_element_0", True),
        ("high_nibble_is_element_0", False),
    ):
        signed = [sign_extend(code, 4) for code in unpack_int4(raw, low_nibble_first=low_first)]
        if all(abs(values[index] - signed[index]) < 1e-3 for index in range(BLOCK)):
            return name
    return None


def summarize(report: dict[str, Any]) -> str:
    lines = [f"device: {report['device']}", f"block: {report['block']}"]
    for name, item in report["formats"].items():
        if item.get("error"):
            lines.append(f"{name}: {item['error']}")
            continue
        state = "ok" if item.get("ok") else "mismatch"
        order = item.get("element_order")
        if order:
            codes = item.get("signed_codes") or {}
            lines.append(
                f"{name}: {state}, {order}, "
                f"codes 0={codes.get('0')} 7={codes.get('7')} "
                f"8={codes.get('8')} 15={codes.get('15')}"
            )
        elif item.get("note"):
            lines.append(f"{name}: {state}, {item['packed_bytes']} packed bytes; {item['note']}")
        else:
            lines.append(f"{name}: {state}, storage {item.get('storage')}")
    scale = report["scale"]
    if scale.get("error"):
        lines.append(f"e8m0: {scale['error']}")
    else:
        state = "ok" if scale.get("ok") else "mismatch"
        lines.append(
            f"e8m0: {state}, wrote {scale.get('scale_wrote')} read {scale.get('scale_read')}, "
            f"reconstruct {scale.get('reconstruct')}"
        )
    return "\n".join(lines)


# Prefill is mpp::tensor_ops::matmul2d. The E8M0 plane requires transpose_right.
# Extents are (inner, outer) with the inner dimension contiguous, which is
# row-major [rows, inner] for the activation and [cols, inner] for the weight.
# WWDC26 session 330. The scale block is 32 along the inner dimension.
MATMUL_SHADER = r"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

kernel void prefill_int4_e8m0(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device int4b_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});

    constexpr auto desc = matmul2d_descriptor(
        64, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    // Extents are (inner, cols). The 32-wide output tile walks cols, not K.
    auto tile_right = right.slice(0, int(tgid.x) * 32);
    auto tile_out = out.slice(int(tgid.x) * 32, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
}

kernel void decode_int4_e8m0(
    device const half* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    uint blocks = inner >> 5;
    device const uchar* packed = weights + row * (inner >> 1);
    device const uchar* row_scale = scales + row * blocks;
    float acc = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        float factor = exp2(float(row_scale[block]) - 127.0f);
        uint element = (block << 5) + lane;
        uchar byte = packed[element >> 1];
        uint nibble = (element & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        int code = int(nibble);
        if (code > 7) {
            code -= 16;
        }
        acc += float(code) * factor * float(activation[element]);
    }
    float sum = simd_sum(acc);
    if (lane == 0) {
        destination[row] = sum;
    }
}
"""


def _shared_buffer(device: Any, blob: bytes) -> Any:
    metal = _metal()
    buffer = device.newBufferWithLength_options_(
        max(len(blob), 16), metal.MTLResourceStorageModeShared
    )
    if blob:
        _buffer_view(buffer, len(blob))[:] = blob
    return buffer


class Int4Gemm:
    """Prefill matmul2d and decode shader dot for int4 weights with E8M0 scales."""

    def __init__(self) -> None:
        metal = _metal()
        device = metal.MTLCreateSystemDefaultDevice()
        if device is None:
            message = "this Mac has no Metal device"
            raise RuntimeError(message)
        options = metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = _objc_pair(
            device.newLibraryWithSource_options_error_(MATMUL_SHADER, options, None)
        )
        if library is None:
            message = f"matmul shader failed to compile: {error}"
            raise RuntimeError(message)
        self.metal = metal
        self.device = device
        self.prefill = self._pipeline(library, "prefill_int4_e8m0")
        self.decode = self._pipeline(library, "decode_int4_e8m0")

    def _pipeline(self, library: Any, name: str) -> Any:
        function = library.newFunctionWithName_(name)
        pipeline, error = _objc_pair(
            self.device.newComputePipelineStateWithFunction_error_(function, None)
        )
        if pipeline is None:
            message = f"{name} pipeline failed: {error}"
            raise RuntimeError(message)
        return pipeline

    def prefill_int4(
        self, activation: bytes, weights: bytes, scales: bytes, rows: int, cols: int, inner: int
    ) -> bytes:
        """rows×inner half activation times cols×inner int4. Returns rows×cols float32."""
        if rows % 64 or cols % 32 or inner % 32:
            message = "prefill tile requires rows multiple of 64 and cols, inner multiple of 32"
            raise ValueError(message)
        destination = _shared_buffer(self.device, b"\x00" * (rows * cols * 4))
        self._dispatch_prefill(activation, weights, scales, destination, rows, cols, inner)
        return _buffer_bytes(destination, rows * cols * 4)

    def decode_int4(
        self, activation: bytes, weights: bytes, scales: bytes, cols: int, inner: int
    ) -> bytes:
        """One half vector of length inner times cols int4 rows. Returns cols float32."""
        if cols < 1 or inner % 32:
            message = "decode requires a positive column count and inner multiple of 32"
            raise ValueError(message)
        destination = _shared_buffer(self.device, b"\x00" * (cols * 4))
        self._dispatch_decode(activation, weights, scales, destination, cols, inner)
        return _buffer_bytes(destination, cols * 4)

    def _dispatch_prefill(
        self,
        activation: bytes,
        weights: bytes,
        scales: bytes,
        destination: Any,
        rows: int,
        cols: int,
        inner: int,
    ) -> None:
        buffers = (
            _shared_buffer(self.device, activation),
            _shared_buffer(self.device, weights),
            _shared_buffer(self.device, scales),
            destination,
            _shared_buffer(self.device, rows.to_bytes(4, "little")),
            _shared_buffer(self.device, cols.to_bytes(4, "little")),
            _shared_buffer(self.device, inner.to_bytes(4, "little")),
        )
        width = self.prefill.threadExecutionWidth()
        self._encode(
            self.prefill,
            buffers,
            self.metal.MTLSize(cols // 32, rows // 64, 1),
            self.metal.MTLSize(width * 4, 1, 1),
        )

    def _dispatch_decode(
        self,
        activation: bytes,
        weights: bytes,
        scales: bytes,
        destination: Any,
        cols: int,
        inner: int,
    ) -> None:
        buffers = (
            _shared_buffer(self.device, activation),
            _shared_buffer(self.device, weights),
            _shared_buffer(self.device, scales),
            destination,
            _shared_buffer(self.device, cols.to_bytes(4, "little")),
            _shared_buffer(self.device, inner.to_bytes(4, "little")),
        )
        self._encode(
            self.decode,
            buffers,
            self.metal.MTLSize(cols, 1, 1),
            self.metal.MTLSize(self.decode.threadExecutionWidth(), 1, 1),
        )

    def _encode(self, pipeline: Any, buffers: tuple[Any, ...], groups: Any, threads: Any) -> None:
        command = self.device.newCommandQueue().commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buffer in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()


def matmul_reference_tile() -> dict[str, Any]:
    """One 64×32×32 int4 tile. Activation is 1 on inner 0. Column n stores code (n%7)-3."""
    import numpy as np

    rows, cols, inner = 64, 32, 32
    activation = np.zeros((rows, inner), dtype=np.float16)
    activation[:, 0] = np.float16(1.0)
    codes = np.zeros((cols, inner), dtype=np.int8)
    column = np.array([(index % 7) - 3 for index in range(cols)], dtype=np.int8)
    codes[:, 0] = column
    packed = bytearray()
    flat = codes.reshape(-1).astype(np.uint8)
    for index in range(0, flat.size, 2):
        packed.append(int(flat[index] & 0xF) | (int(flat[index + 1] & 0xF) << 4))
    scales = bytes([130]) * (cols * (inner // BLOCK))
    expected = column.astype(np.float32) * np.float32(8.0)
    return {
        "rows": rows,
        "cols": cols,
        "inner": inner,
        "activation": np.ascontiguousarray(activation).tobytes(),
        "weights": bytes(packed),
        "scales": scales,
        "expected": expected,
    }


def check_int4_gemm() -> dict[str, Any]:
    tile = matmul_reference_tile()
    gemm = Int4Gemm()
    prefill = np_from_f32(
        gemm.prefill_int4(
            tile["activation"],
            tile["weights"],
            tile["scales"],
            tile["rows"],
            tile["cols"],
            tile["inner"],
        )
    ).reshape(tile["rows"], tile["cols"])
    decode = np_from_f32(
        gemm.decode_int4(
            tile["activation"][: tile["inner"] * 2],
            tile["weights"],
            tile["scales"],
            tile["cols"],
            tile["inner"],
        )
    )
    expected = tile["expected"]
    prefill_ok = bool((prefill == expected).all())
    decode_ok = bool((decode == expected).all())
    wide_ok = _wide_prefill_matches(gemm)
    return {
        "device": _device_name(gemm.device),
        "ok": prefill_ok and decode_ok and wide_ok,
        "prefill": prefill_ok,
        "decode": decode_ok,
        "wide": wide_ok,
    }


def _wide_prefill_matches(gemm: Int4Gemm) -> bool:
    """Two row tiles and two column tiles. A 32-column tile hides a bad N offset."""
    import numpy as np

    rows, cols, inner = 128, 64, 64
    rng = np.random.default_rng(0)
    activation = rng.normal(size=(rows, inner)).astype(np.float16)
    weight = rng.normal(size=(cols, inner)).astype(np.float32)
    packed, scales = quantize_matrix(weight)
    got = np_from_f32(
        gemm.prefill_int4(activation.tobytes(), packed, scales, rows, cols, inner)
    ).reshape(rows, cols)
    restored = dequant_int4(packed, scales, cols, inner)
    expect = activation.astype(np.float32) @ restored.T
    return bool(np.allclose(got, expect, rtol=1e-3, atol=1e-3))


def np_from_f32(blob: bytes) -> Any:
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32).copy()


def probe(*, as_json: bool) -> int:
    try:
        report = Roundtrip().measure()
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    if as_json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(summarize(report) + "\n")
    return 0 if report["ok"] else 1


def matmul() -> int:
    try:
        report = check_int4_gemm()
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    sys.stdout.write(
        f"device: {report['device']}\n"
        f"prefill matmul2d int4 e8m0: {'ok' if report['prefill'] else 'mismatch'}\n"
        f"prefill matmul2d 128x64x64: {'ok' if report['wide'] else 'mismatch'}\n"
        f"decode shader dot int4 e8m0: {'ok' if report['decode'] else 'mismatch'}\n"
    )
    return 0 if report["ok"] else 1


def quantize_matrix(weight: np.ndarray) -> tuple[bytes, bytes]:
    """Pack [rows, inner] float32 as int4, low nibble first, with one E8M0 byte per 32."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    blocks = matrix.reshape(rows, inner // BLOCK, BLOCK).astype(np.float64)
    amax = np.max(np.abs(blocks), axis=-1)
    ratio = amax / QMAX[4]
    mantissa, exponent = np.frexp(np.maximum(ratio, np.finfo(np.float64).tiny))
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float64), biased.astype(np.int32) - 127)
    codes = np.rint(blocks / scale[..., None])
    codes = np.clip(codes, -8, 7).astype(np.int32)
    nibbles = (codes & 0xF).astype(np.uint8)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return (
        np.ascontiguousarray(packed).tobytes(),
        np.ascontiguousarray(biased).tobytes(),
    )


def dequant_int4(packed: bytes, scales: bytes, rows: int, inner: int) -> np.ndarray:
    raw = np.frombuffer(packed, dtype=np.uint8).reshape(rows, inner // 2)
    codes = np.empty((rows, inner), dtype=np.int32)
    codes[:, 0::2] = raw & 0xF
    codes[:, 1::2] = raw >> 4
    codes = np.where(codes > 7, codes - 16, codes)
    factors = np.frombuffer(scales, dtype=np.uint8).astype(np.int32).reshape(rows, inner // BLOCK)
    restored = codes.reshape(rows, inner // BLOCK, BLOCK).astype(np.float32)
    restored *= np.ldexp(np.ones(factors.shape, dtype=np.float32), factors - 127)[..., None]
    return restored.reshape(rows, inner)


def stored_k_int4(inner: int, *, align_k: bool) -> int:
    """int4 row bytes must be a multiple of 128, so stored K is a multiple of 256."""
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    if not align_k:
        return inner
    return ((inner + 255) // 256) * 256


def int4_plane_sizes(rows: int, stored_inner: int) -> tuple[int, int]:
    return rows * (stored_inner // 2), rows * (stored_inner // BLOCK)


def quantize_rows(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack [rows, stored_k] float32. Same rule as ``quantize_matrix``."""
    rows, inner = block.shape
    blocks = block.reshape(rows, inner // BLOCK, BLOCK)
    amax = np.max(np.abs(blocks), axis=-1).astype(np.float64)
    ratio = amax / QMAX[4]
    mantissa, exponent = np.frexp(np.maximum(ratio, np.finfo(np.float64).tiny))
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    codes = np.rint(blocks / scale[..., None])
    codes = np.clip(codes, -8, 7).astype(np.int32)
    nibbles = (codes & 0xF).astype(np.uint8)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return np.ascontiguousarray(packed), np.ascontiguousarray(biased)


def quantize_int4_image(weight: np.ndarray, *, align_k: bool = True) -> tuple[bytes, int]:
    """Data plane, then tight E8M0 scales. ``align_k`` pads K out to a multiple of 256."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    stored = stored_k_int4(inner, align_k=align_k)
    if stored != inner:
        matrix = np.pad(matrix, ((0, 0), (0, stored - inner)))
    data_n, scale_n = int4_plane_sizes(rows, stored)
    image = bytearray(data_n + scale_n)
    step = 256
    data_row = stored // 2
    scale_row = stored // BLOCK
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        packed, scales = quantize_rows(matrix[start:stop])
        count = stop - start
        image[start * data_row : stop * data_row] = packed.reshape(count, data_row).tobytes()
        base = data_n + start * scale_row
        image[base : base + count * scale_row] = scales.reshape(count, scale_row).tobytes()
    return bytes(image), stored


def ngram_slots(weight: np.ndarray) -> bytes:
    """One 128-byte slot per row: 80 int4 bytes, five E8M0 bytes, then padding."""
    rows, inner = weight.shape
    if inner != NGRAM_WIDTH:
        message = f"n-gram width {inner} is not {NGRAM_WIDTH}"
        raise ValueError(message)
    image, stored = quantize_int4_image(weight, align_k=False)
    if stored != NGRAM_WIDTH:
        message = f"n-gram stored K {stored} is not {NGRAM_WIDTH}"
        raise RuntimeError(message)
    data_n, _scale_n = int4_plane_sizes(rows, stored)
    data = np.frombuffer(image[:data_n], dtype=np.uint8).reshape(rows, 80)
    scales = np.frombuffer(image[data_n:], dtype=np.uint8).reshape(rows, 5)
    slots = np.zeros((rows, NGRAM_SLOT), dtype=np.uint8)
    slots[:, :80] = data
    slots[:, 80:85] = scales
    return np.ascontiguousarray(slots).tobytes()


def expert_live_bytes() -> int:
    """Middle-tier gate_up image followed by the down image."""
    gate_up = sum(int4_plane_sizes(1280, stored_k_int4(2560, align_k=True)))
    down = sum(int4_plane_sizes(2560, stored_k_int4(640, align_k=True)))
    return gate_up + down


VECTOR_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

kernel void gemv_f16(
    device const half* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    device const half* rowp = weights + row * inner;
    float acc = 0.0f;
    for (uint i = lane; i < inner; i += 32u) {
        acc += float(rowp[i]) * float(activation[i]);
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

kernel void gemv_f32(
    device const float* activation [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    device const float* rowp = weights + row * inner;
    float acc = 0.0f;
    for (uint i = lane; i < inner; i += 32u) {
        acc += rowp[i] * activation[i];
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

kernel void gdn_delta(
    device float* state [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device const float* value [[buffer(3)]],
    device const float* alpha [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float key_h[128];
    threadgroup float query_h[128];
    if (tid < 128u) {
        key_h[tid] = key[head * 128u + tid];
        query_h[tid] = query[head * 128u + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* shard = state + head * 128u * 128u;
    float dot = 0.0f;
    for (uint dk = 0u; dk < 128u; ++dk) {
        dot += shard[dk * 128u + tid] * key_h[dk];
    }
    float decay = alpha[head];
    float error = value[head * 128u + tid] - decay * dot;
    float write = beta[head];
    for (uint dk = 0u; dk < 128u; ++dk) {
        uint index = dk * 128u + tid;
        shard[index] = decay * shard[index] + write * key_h[dk] * error;
    }
    float acc = 0.0f;
    for (uint dk = 0u; dk < 128u; ++dk) {
        acc += shard[dk * 128u + tid] * query_h[dk];
    }
    output[head * 128u + tid] = acc;
}

kernel void attend_gqa(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& kv_heads [[buffer(6)]],
    constant uint& dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint kv = head / (24u / kv_heads);
    device const float* qh = query + head * dim;
    for (uint t = 0u; t < tokens; ++t) {
        device const float* kt = key + (t * kv_heads + kv) * dim;
        float acc = 0.0f;
        for (uint d = lane; d < dim; d += 32u) {
            acc += qh[d] * kt[d];
        }
        float score = simd_sum(acc) * scale;
        if (lane == 0u) {
            scores[head * tokens + t] = score;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    float maxs = -INFINITY;
    for (uint t = 0u; t < tokens; ++t) {
        maxs = max(maxs, scores[head * tokens + t]);
    }
    if (lane == 0u) {
        for (uint t = 0u; t < tokens; ++t) {
            scores[head * tokens + t] = exp(scores[head * tokens + t] - maxs);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    float denom = 0.0f;
    for (uint t = 0u; t < tokens; ++t) {
        denom += scores[head * tokens + t];
    }
    for (uint d = lane; d < dim; d += 32u) {
        float acc = 0.0f;
        for (uint t = 0u; t < tokens; ++t) {
            device const float* vt = value + (t * kv_heads + kv) * dim;
            acc += scores[head * tokens + t] / denom * vt[d];
        }
        output[head * dim + d] = acc;
    }
}
"""


def _log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _tok_s(label: str, tokens: int, seconds: float) -> str:
    if tokens <= 0 or seconds <= 0:
        return f"{label} 0 tok"
    return f"{label} {tokens / seconds:.2f} tok/s ({tokens} tok in {seconds:.2f}s)"


def _append_transcript(path: Path | None, prompt: str, output: str) -> None:
    """Append one prompt and its completion. Token rates stay on the console."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = f"prompt:\n{prompt.rstrip()}\n\noutput:\n{output.rstrip()}\n"
    existing = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8") as handle:
        if existing:
            handle.write("\n")
        handle.write(record)


def _prompt_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return json.dumps(messages, ensure_ascii=False)
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"{message.get('role', '')}: {content}")
    return "\n".join(lines)


def _check_f32_gemv(gpu: MetalSession) -> None:
    weight = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    vector = np.arange(4, dtype=np.float32) + 1.0
    got = gpu.f32(gpu.load_f32(weight), vector)
    expect = weight @ vector
    if not np.allclose(got, expect):
        message = "f32 shader dot diverged"
        raise RuntimeError(message)


def bf16_to_f32(raw: bytes) -> np.ndarray:
    units = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (units << 16).view(np.float32)


class Int4Matrix:
    def __init__(
        self,
        rows: int,
        inner: int,
        packed_rows: int,
        weights: Any,
        scales: Any,
        stored_inner: int | None = None,
    ) -> None:
        self.rows = rows
        self.inner = inner
        self.stored_inner = inner if stored_inner is None else stored_inner
        self.packed_rows = packed_rows
        self.weights = weights
        self.scales = scales


class F16Matrix:
    def __init__(self, rows: int, inner: int, weights: Any) -> None:
        self.rows = rows
        self.inner = inner
        self.weights = weights


class F32Matrix:
    """One BF16 matrix, expanded to float32 and uploaded for the shader dot."""

    def __init__(self, rows: int, inner: int, weights: Any) -> None:
        self.rows = rows
        self.inner = inner
        self.weights = weights


class MetalSession:
    """Decode dots, prefill matmul2d, the GDN delta, and grouped-query attention."""

    def __init__(self) -> None:
        self.gemm = Int4Gemm()
        self.device = self.gemm.device
        self.metal = self.gemm.metal
        width = self.gemm.decode.threadExecutionWidth()
        if int(width) != 32:
            message = f"shader dot assumes simd width 32, device reported {width}"
            raise RuntimeError(message)
        self.queue = self.device.newCommandQueue()
        options = self.metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = _objc_pair(
            self.device.newLibraryWithSource_options_error_(VECTOR_SHADER, options, None)
        )
        if library is None:
            message = f"vector shader failed to compile: {error}"
            raise RuntimeError(message)
        self.f16_pipe = self._pipeline(library, "gemv_f16")
        self.f32_pipe = self._pipeline(library, "gemv_f32")
        self.gdn_pipe = self._pipeline(library, "gdn_delta")
        self.attend_pipe = self._pipeline(library, "attend_gqa")
        self.activation = self.alloc(10240 * 2)
        self.f32_activation = self.alloc(10240 * 4)
        self.destination = self.alloc(248320 * 4)
        self.prefill_act = self.alloc(64 * 10240 * 2)
        self.prefill_dest = self.alloc(64 * 12288 * 4)
        self.scores = self.alloc(24 * 64 * 4)
        self.attend_q = self.alloc(24 * 256 * 4)
        self.attend_k = self.alloc(64 * 2 * 256 * 4)
        self.attend_v = self.alloc(64 * 2 * 256 * 4)
        self.attend_out = self.alloc(24 * 256 * 4)
        self.gdn_q = self.alloc(48 * 128 * 4)
        self.gdn_k = self.alloc(48 * 128 * 4)
        self.gdn_v = self.alloc(48 * 128 * 4)
        self.gdn_a = self.alloc(48 * 4)
        self.gdn_b = self.alloc(48 * 4)
        self.gdn_y = self.alloc(48 * 128 * 4)

    def _pipeline(self, library: Any, name: str) -> Any:
        function = library.newFunctionWithName_(name)
        pipeline, error = _objc_pair(
            self.device.newComputePipelineStateWithFunction_error_(function, None)
        )
        if pipeline is None:
            message = f"{name} pipeline failed: {error}"
            raise RuntimeError(message)
        return pipeline

    def alloc(self, nbytes: int) -> Any:
        return self.device.newBufferWithLength_options_(
            max(nbytes, 16), self.metal.MTLResourceStorageModeShared
        )

    def upload(self, blob: bytes) -> Any:
        buffer = self.alloc(len(blob))
        if blob:
            _buffer_view(buffer, len(blob))[:] = blob
        return buffer

    def zeros(self, nbytes: int) -> Any:
        buffer = self.alloc(nbytes)
        _fill_buffer(buffer, nbytes, 0)
        return buffer

    def load_int4(
        self, image: bytes | memoryview, rows: int, logical_inner: int, stored_inner: int
    ) -> Int4Matrix:
        """Upload a record image. Zero rows pad the output width to a multiple of 32."""
        data_n, scale_n = int4_plane_sizes(rows, stored_inner)
        if len(image) != data_n + scale_n:
            message = f"int4 image is {len(image)} bytes, expected {data_n + scale_n}"
            raise ValueError(message)
        packed_rows = ((rows + 31) // 32) * 32
        weights = memoryview(image)[:data_n]
        scales = memoryview(image)[data_n:]
        if packed_rows != rows:
            row_b = stored_inner // 2
            scale_b = stored_inner // BLOCK
            w = bytearray(weights)
            w.extend(b"\x00" * ((packed_rows - rows) * row_b))
            s = bytearray(scales)
            s.extend(b"\x00" * ((packed_rows - rows) * scale_b))
            weights = memoryview(w)
            scales = memoryview(s)
        return Int4Matrix(
            rows,
            logical_inner,
            packed_rows,
            self.upload(weights),
            self.upload(scales),
            stored_inner,
        )

    def load_f32(self, matrix: np.ndarray) -> F32Matrix:
        array = np.ascontiguousarray(matrix, dtype=np.float32)
        rows, inner = int(array.shape[0]), int(array.shape[1])
        return F32Matrix(rows, inner, self.upload(array.tobytes()))

    def f32(self, matrix: F32Matrix, vector: np.ndarray) -> np.ndarray:
        if vector.shape != (matrix.inner,):
            message = f"f32 vector {vector.shape} does not match inner {matrix.inner}"
            raise ValueError(message)
        if matrix.inner > 10240:
            message = f"f32 inner {matrix.inner} exceeds the activation buffer"
            raise ValueError(message)
        self._write_f32(self.f32_activation, vector)
        self._encode(
            self.f32_pipe,
            (self.f32_activation, matrix.weights, self.destination),
            (struct.pack("<I", matrix.rows), struct.pack("<I", matrix.inner)),
            self.metal.MTLSize(matrix.rows, 1, 1),
            self.metal.MTLSize(32, 1, 1),
        )
        raw = _buffer_bytes(self.destination, matrix.rows * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def int4(self, matrix: Int4Matrix, vector: np.ndarray) -> np.ndarray:
        if isinstance(matrix, F32Matrix):
            return self.f32(matrix, vector)
        if vector.shape != (matrix.inner,):
            message = f"int4 vector {vector.shape} does not match inner {matrix.inner}"
            raise ValueError(message)
        activation = np.zeros(matrix.stored_inner, dtype=np.float32)
        activation[: matrix.inner] = vector
        half = np.clip(activation, -65504.0, 65504.0).astype(np.float16)
        _buffer_view(self.activation, half.nbytes)[:] = half.tobytes()
        self._encode(
            self.gemm.decode,
            (self.activation, matrix.weights, matrix.scales, self.destination),
            (struct.pack("<I", matrix.rows), struct.pack("<I", matrix.stored_inner)),
            self.metal.MTLSize(matrix.rows, 1, 1),
            self.metal.MTLSize(32, 1, 1),
        )
        raw = _buffer_bytes(self.destination, matrix.rows * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def int4_rows(self, matrix: Int4Matrix, rows: np.ndarray) -> np.ndarray:
        if isinstance(matrix, F32Matrix):
            return np.stack([self.f32(matrix, row) for row in rows], axis=0)
        count = rows.shape[0]
        if count == 1:
            return self.int4(matrix, rows[0])[None, :]
        padded = ((count + 63) // 64) * 64
        cols = matrix.packed_rows
        act = np.zeros((padded, matrix.stored_inner), dtype=np.float16)
        act[:count, : matrix.inner] = np.clip(rows, -65504.0, 65504.0).astype(np.float16)
        self.prefill_act = self._grow(self.prefill_act, act.nbytes)
        self.prefill_dest = self._grow(self.prefill_dest, padded * cols * 4)
        _buffer_view(self.prefill_act, act.nbytes)[:] = act.tobytes()
        _fill_buffer(self.prefill_dest, padded * cols * 4, 0)
        self._encode(
            self.gemm.prefill,
            (self.prefill_act, matrix.weights, matrix.scales, self.prefill_dest),
            (
                struct.pack("<I", padded),
                struct.pack("<I", cols),
                struct.pack("<I", matrix.stored_inner),
            ),
            self.metal.MTLSize(cols // 32, padded // 64, 1),
            self.metal.MTLSize(32 * 4, 1, 1),
        )
        flat = np.frombuffer(
            _buffer_bytes(self.prefill_dest, padded * cols * 4), dtype=np.float32
        ).reshape(padded, cols)
        return flat[:count, : matrix.rows].copy()

    def f16(self, matrix: F16Matrix, vector: np.ndarray) -> np.ndarray:
        if isinstance(matrix, F32Matrix):
            return self.f32(matrix, vector)
        half = np.clip(vector, -65504.0, 65504.0).astype(np.float16)
        _buffer_view(self.activation, half.nbytes)[:] = half.tobytes()
        self._encode(
            self.f16_pipe,
            (self.activation, matrix.weights, self.destination),
            (struct.pack("<I", matrix.rows), struct.pack("<I", matrix.inner)),
            self.metal.MTLSize(matrix.rows, 1, 1),
            self.metal.MTLSize(32, 1, 1),
        )
        raw = _buffer_bytes(self.destination, matrix.rows * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def f16_rows(self, matrix: F16Matrix, rows: np.ndarray) -> np.ndarray:
        return np.stack([self.f16(matrix, row) for row in rows], axis=0)

    def gdn(
        self,
        state: Any,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        self._write_f32(self.gdn_q, query)
        self._write_f32(self.gdn_k, key)
        self._write_f32(self.gdn_v, value)
        self._write_f32(self.gdn_a, alpha)
        self._write_f32(self.gdn_b, beta)
        self._encode(
            self.gdn_pipe,
            (state, self.gdn_q, self.gdn_k, self.gdn_v, self.gdn_a, self.gdn_b, self.gdn_y),
            (),
            self.metal.MTLSize(48, 1, 1),
            self.metal.MTLSize(128, 1, 1),
        )
        return np.frombuffer(_buffer_bytes(self.gdn_y, 48 * 128 * 4), dtype=np.float32).copy()

    def attend(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        tokens = key.shape[0]
        self.scores = self._grow(self.scores, 24 * tokens * 4)
        self.attend_k = self._grow(self.attend_k, key.nbytes)
        self.attend_v = self._grow(self.attend_v, value.nbytes)
        self._write_f32(self.attend_q, query)
        self._write_f32(self.attend_k, key)
        self._write_f32(self.attend_v, value)
        scale = struct.pack("<f", 1.0 / math.sqrt(256.0))
        self._encode(
            self.attend_pipe,
            (self.attend_q, self.attend_k, self.attend_v, self.scores, self.attend_out),
            (
                struct.pack("<I", tokens),
                struct.pack("<I", 2),
                struct.pack("<I", 256),
                scale,
            ),
            self.metal.MTLSize(24, 1, 1),
            self.metal.MTLSize(32, 1, 1),
        )
        return np.frombuffer(_buffer_bytes(self.attend_out, 24 * 256 * 4), dtype=np.float32).copy()

    def _grow(self, buffer: Any, nbytes: int) -> Any:
        if buffer.length() >= nbytes:
            return buffer
        size = 16
        while size < nbytes:
            size *= 2
        return self.alloc(size)

    def _write_f32(self, buffer: Any, array: np.ndarray) -> None:
        blob = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        if buffer.length() < len(blob):
            message = f"buffer {buffer.length()} is smaller than {len(blob)}"
            raise RuntimeError(message)
        _buffer_view(buffer, len(blob))[:] = blob

    def _encode(
        self,
        pipeline: Any,
        buffers: tuple[Any, ...],
        constants: tuple[bytes, ...],
        groups: Any,
        threads: Any,
    ) -> None:
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buffer in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        for offset, blob in enumerate(constants):
            encoder.setBytes_length_atIndex_(blob, len(blob), len(buffers) + offset)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if int(command.status()) != 4:
            message = f"Metal command status {command.status()} {command.error()}"
            raise RuntimeError(message)


def check_state_kernels() -> None:
    """GDN delta and grouped-query attention match the tech-report recurrences on one step."""
    session = MetalSession()
    rng = np.random.default_rng(0)
    state = rng.normal(size=(48, 128, 128)).astype(np.float32)
    query = rng.normal(size=(48, 128)).astype(np.float32)
    key = rng.normal(size=(48, 128)).astype(np.float32)
    value = rng.normal(size=(48, 128)).astype(np.float32)
    alpha = rng.uniform(0.2, 0.9, size=48).astype(np.float32)
    beta = rng.uniform(0.1, 0.8, size=48).astype(np.float32)
    buffer = session.upload(np.ascontiguousarray(state).tobytes())
    got = session.gdn(buffer, query, key, value, alpha, beta)
    raw = _buffer_bytes(buffer, state.nbytes)
    updated = np.frombuffer(raw, dtype=np.float32).reshape(state.shape)
    tilde = alpha[:, None, None] * state
    error = value - np.einsum("hdv,hd->hv", tilde, key)
    expect_state = tilde + beta[:, None, None] * key[:, :, None] * error[:, None, :]
    expect_y = np.einsum("hdv,hd->hv", expect_state, query).reshape(-1)
    if not np.allclose(updated, expect_state, rtol=1e-4, atol=1e-4):
        message = "GDN state shader diverged"
        raise RuntimeError(message)
    if not np.allclose(got, expect_y, rtol=1e-4, atol=1e-4):
        message = "GDN output shader diverged"
        raise RuntimeError(message)
    tokens = 5
    q = rng.normal(size=(24, 256)).astype(np.float32)
    k = rng.normal(size=(tokens, 2, 256)).astype(np.float32)
    v = rng.normal(size=(tokens, 2, 256)).astype(np.float32)
    attended = session.attend(q, k, v).reshape(24, 256)
    scale = 1.0 / math.sqrt(256.0)
    expect = np.empty_like(q)
    for head in range(24):
        kv = head // 12
        scores = k[:, kv] @ q[head] * scale
        scores = scores - scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        expect[head] = probs @ v[:, kv]
    if not np.allclose(attended, expect, rtol=1e-4, atol=1e-4):
        message = "attention shader diverged"
        raise RuntimeError(message)


def _rms(
    values: np.ndarray, weight: np.ndarray, eps: float, *, zero_centered: bool = True
) -> np.ndarray:
    variance = np.mean(np.square(values), axis=-1, keepdims=True)
    # Hyper-connection norms are Gemma-style offsets. The GDN head norm is a
    # multiplicative gain, initialized at one, so the stored value is the gain.
    gain = (1.0 + weight) if zero_centered else weight
    return values * np.reciprocal(np.sqrt(variance + eps)) * gain


def _silu(values: np.ndarray) -> np.ndarray:
    # A large negative input overflows exp and saturates to zero, which is SiLU.
    with np.errstate(over="ignore"):
        return values / (1.0 + np.exp(-values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    # A large negative input overflows exp and saturates to zero, which is sigmoid.
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-values))


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def _l2norm(values: np.ndarray, eps: float) -> np.ndarray:
    return values / np.sqrt(np.sum(np.square(values), axis=-1, keepdims=True) + eps)


def _rope(values: np.ndarray, position: int, rotary_dim: int, theta: float) -> np.ndarray:
    half = rotary_dim // 2
    freqs = np.float64(theta) ** (-np.arange(half, dtype=np.float64) / half)
    angles = position * freqs
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    out = values.copy()
    first = values[:, :half]
    second = values[:, half:rotary_dim]
    out[:, :half] = first * cos - second * sin
    out[:, half:rotary_dim] = first * sin + second * cos
    return out


def _causal_conv(history: list[np.ndarray], current: np.ndarray, weight: np.ndarray) -> np.ndarray:
    kernel = weight.shape[1]
    frames = [*history, current]
    if len(frames) < kernel:
        zero = np.zeros_like(current)
        frames = [zero] * (kernel - len(frames)) + frames
    mixed = np.zeros_like(current)
    for tap, frame in enumerate(frames[-kernel:]):
        mixed += weight[:, tap] * frame
    history.append(current)
    del history[:-kernel + 1]
    return mixed


def _sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int, rng: Any) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    scores = logits.astype(np.float64) / temperature
    if 0 < top_k < scores.size:
        cutoff = np.partition(scores, -top_k)[-top_k]
        scores = np.where(scores < cutoff, -np.inf, scores)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    total = float(probs.sum())
    if not math.isfinite(total) or total <= 0.0:
        return int(np.argmax(logits))
    probs /= total
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        remove = cumulative > top_p
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        probs[order[remove]] = 0.0
        probs /= probs.sum()
    return int(rng.choice(probs.size, p=probs))


_I64_MASK = (1 << 64) - 1
_I64_SIGN = 1 << 63


def _earlier_token(tokens: Sequence[int], shift: int, eos: int) -> int:
    """Token `shift` steps before the last one.

    Shift 0 is the current token, kept even when it is EOS. A lookback that
    starts before this segment, or that crosses an earlier EOS, is the EOS id.
    """
    if shift == 0:
        return int(tokens[-1])
    last = len(tokens) - 1
    segment_start = 0
    for index, item in enumerate(tokens[:-1]):
        if int(item) == eos:
            segment_start = index + 1
    if last - segment_start < shift:
        return eos
    return int(tokens[last - shift])


def _ngram_row(
    tokens: Sequence[int],
    head: int,
    multipliers: np.ndarray,
    offsets: np.ndarray,
    vocabs: np.ndarray,
    *,
    heads_per_ngram: int,
    ngram_size: int,
    eos: int,
) -> int:
    """Bigram heads, then trigram heads. Wrapping int64 XOR, positive remainder.

    Shift 0 is the current token and uses the first multiplier. Each older
    token uses the next multiplier. A cleared lookback contributes the EOS id.
    """
    order = 2 + head // heads_per_ngram
    mults = [int(value) for value in np.reshape(multipliers, -1).tolist()]
    mixed = 0
    for shift in range(order):
        product = (_earlier_token(tokens, shift, eos) * mults[shift]) & _I64_MASK
        mixed ^= product
    signed = mixed - (1 << 64) if mixed >= _I64_SIGN else mixed
    return int(offsets[head]) + signed % int(vocabs[head])


def _dilated_conv(
    history: list[np.ndarray], current: np.ndarray, weight: np.ndarray, dilation: int
) -> np.ndarray:
    """Causal depthwise conv. Weight tap 0 is the oldest sample."""
    kernel = int(weight.shape[1])
    span = (kernel - 1) * dilation
    frames = [*history, current]
    if len(frames) < span + 1:
        zero = np.zeros_like(current)
        frames = [zero] * (span + 1 - len(frames)) + frames
    window = frames[-(span + 1) :]
    mixed = np.zeros_like(current)
    for tap in range(kernel):
        mixed += weight[:, tap] * window[tap * dilation]
    history.append(np.array(current, copy=True))
    del history[:-span]
    return mixed


class ShardStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self._fds: dict[str, int] = {}

    def _locate(self, key: str) -> tuple[int, int, str, list[int]]:
        shard = self.weight_map[key]
        if shard not in self._headers:
            path = self.root / shard
            with path.open("rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            self._headers[shard] = (8 + header_len, header)
            fd = os.open(path, os.O_RDONLY)
            _bypass_cache(fd)
            self._fds[shard] = fd
        base, header = self._headers[shard]
        meta = header[key]
        start, _end = meta["data_offsets"]
        return self._fds[shard], base + start, meta["dtype"], meta["shape"]

    def read_bf16(
        self, key: str, elem_offset: int, elem_count: int, shape: tuple[int, ...]
    ) -> np.ndarray:
        fd, start, dtype, _shape = self._locate(key)
        if dtype != "BF16":
            message = f"{key} is {dtype}, expected BF16"
            raise RuntimeError(message)
        raw = os.pread(fd, elem_count * 2, start + elem_offset * 2)
        return bf16_to_f32(raw).reshape(shape)

    def matrix(self, key: str) -> np.ndarray:
        _fd, _start, _dtype, shape = self._locate(key)
        count = int(np.prod(shape))
        return self.read_bf16(key, 0, count, tuple(shape))

    def i64(self, key: str) -> np.ndarray:
        fd, start, dtype, shape = self._locate(key)
        if dtype != "I64":
            message = f"{key} is {dtype}, expected I64"
            raise RuntimeError(message)
        count = int(np.prod(shape))
        raw = os.pread(fd, count * 8, start)
        return np.frombuffer(raw, dtype=np.int64).reshape(shape).copy()

    def expert(self, layer: int, name: str, expert: int) -> np.ndarray:
        key = f"model.language_model.layers.{layer}.mlp.experts.{name}"
        _fd, _start, _dtype, shape = self._locate(key)
        _experts, out, inner = shape
        count = out * inner
        return self.read_bf16(key, expert * count, count, (out, inner))

    def ngram_row(self, row: int, rows_per_shard: int, width: int) -> np.ndarray:
        shard = row // rows_per_shard
        local = row % rows_per_shard
        key = (
            "model.language_model.layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{shard}.weight"
        )
        return self.read_bf16(key, local * width, width, (width,))


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
_LIBC.pread.restype = ctypes.c_ssize_t


def _bypass_cache(fd: int) -> None:
    flag = getattr(fcntl, "F_NOCACHE", None)
    if flag is not None:
        fcntl.fcntl(fd, flag, 1)


def pread_exact(fd: int, nbytes: int, offset: int, scratch: mmap.mmap) -> memoryview:
    if len(scratch) < nbytes:
        message = f"scratch {len(scratch)} is smaller than {nbytes}"
        raise RuntimeError(message)
    holder = ctypes.c_char.from_buffer(scratch)
    addr = ctypes.addressof(holder)
    done = 0
    while done < nbytes:
        got = _LIBC.pread(fd, addr + done, nbytes - done, offset + done)
        if got < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        if got == 0:
            message = f"short read at {offset + done}"
            raise OSError(message)
        done += int(got)
    del holder
    return memoryview(scratch)[:nbytes]


def _expert_matrix(store: ShardStore, key: str, expert: int) -> np.ndarray:
    _fd, _start, _dtype, shape = store._locate(key)
    _count, out, inner = shape
    return store.read_bf16(key, expert * out * inner, out * inner, (out, inner))


class ResidentWriter:
    def __init__(self, path: Path) -> None:
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(self.fd, 0)
        self.offset = 0
        self.int4: dict[str, dict[str, int]] = {}
        self.f16: dict[str, dict[str, int]] = {}
        self.f32: dict[str, dict[str, Any]] = {}
        self.i64: dict[str, dict[str, Any]] = {}

    def _write(self, blob: bytes) -> int:
        start = self.offset
        os.pwrite(self.fd, blob, start)
        self.offset = start + len(blob)
        pad = (128 - self.offset % 128) % 128
        if pad:
            os.pwrite(self.fd, b"\x00" * pad, self.offset)
            self.offset += pad
        return start

    def add_int4(self, key: str, matrix: np.ndarray) -> None:
        image, stored = quantize_int4_image(matrix, align_k=True)
        rows, logical = int(matrix.shape[0]), int(matrix.shape[1])
        data_n, scale_n = int4_plane_sizes(rows, stored)
        self.int4[key] = {
            "offset": self._write(image),
            "rows": rows,
            "logical_inner": logical,
            "stored_inner": stored,
            "data": data_n,
            "scales": scale_n,
        }

    def add_f16(self, key: str, matrix: np.ndarray) -> None:
        half = np.clip(np.ascontiguousarray(matrix, dtype=np.float32), -65504.0, 65504.0)
        blob = np.ascontiguousarray(half, dtype=np.float16).tobytes()
        self.f16[key] = {
            "offset": self._write(blob),
            "rows": int(matrix.shape[0]),
            "inner": int(matrix.shape[1]),
        }

    def add_f32(self, key: str, array: np.ndarray) -> None:
        blob = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        self.f32[key] = {"offset": self._write(blob), "shape": list(array.shape)}

    def add_i64(self, key: str, array: np.ndarray) -> None:
        blob = np.ascontiguousarray(array, dtype=np.int64).tobytes()
        self.i64[key] = {"offset": self._write(blob), "shape": list(array.shape)}

    def close(self) -> None:
        os.fsync(self.fd)
        os.close(self.fd)


def _progress_path(dest: Path) -> Path:
    return dest / "quantize_progress.json"


def _load_progress(dest: Path) -> dict[str, Any]:
    path = _progress_path(dest)
    if not path.is_file():
        return {
            "version": RECORD_VERSION,
            "resident": False,
            "layers": [],
            "mtp": False,
            "ngram_rows": 0,
        }
    progress = json.loads(path.read_text())
    if progress.get("version") != RECORD_VERSION:
        message = f"delete {dest}; it was started with record version {progress.get('version')}"
        raise RuntimeError(message)
    return progress


def _save_progress(dest: Path, progress: dict[str, Any]) -> None:
    path = _progress_path(dest)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress))
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    directory = os.open(dest, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def ngram_resume_row(saved: int, file_size: int) -> int:
    """Keep the checkpoint only for rows that are already in the file."""
    durable = max(file_size, 0) // NGRAM_SLOT
    if saved > durable:
        return durable
    return max(saved, 0)


def _expert_start(
    progress: dict[str, Any],
    *,
    done: bool,
    inflight_key: int | str,
    path: Path,
    experts: int,
) -> int:
    """First expert still to write. ``experts`` means the file is finished."""
    full = experts * EXPERT_SLOT
    size = path.stat().st_size if path.is_file() else 0
    if done and size >= full:
        return experts
    if progress.get("inflight_key") == inflight_key and size >= full:
        nxt = int(progress.get("inflight_expert", -1)) + 1
        if nxt >= experts:
            return experts
        if nxt > 0:
            return nxt
    return 0


def _clear_inflight(progress: dict[str, Any]) -> None:
    progress.pop("inflight_key", None)
    progress.pop("inflight_expert", None)


def _layer_int4_keys(prefix: str, kind: str) -> list[str]:
    keys = [
        f"{prefix}.attn_hyper_connection.input_mix_weight_down.weight",
        f"{prefix}.attn_hyper_connection.input_mix_weight_up.weight",
        f"{prefix}.attn_hyper_connection.block_inject_weight.weight",
        f"{prefix}.mlp_hyper_connection.input_mix_weight_down.weight",
        f"{prefix}.mlp_hyper_connection.input_mix_weight_up.weight",
        f"{prefix}.mlp_hyper_connection.block_inject_weight.weight",
        f"{prefix}.mlp.shared_expert.gate_proj.weight",
        f"{prefix}.mlp.shared_expert.up_proj.weight",
        f"{prefix}.mlp.shared_expert.down_proj.weight",
        f"{prefix}.mlp.shared_expert_gate.weight",
    ]
    if kind == "linear_attention":
        keys.extend(
            [
                f"{prefix}.linear_attn.in_proj_qkv.weight",
                f"{prefix}.linear_attn.in_proj_z.weight",
                f"{prefix}.linear_attn.in_proj_a.weight",
                f"{prefix}.linear_attn.in_proj_b.weight",
                f"{prefix}.linear_attn.out_proj.weight",
            ]
        )
    elif kind == "full_attention":
        keys.extend(
            [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.indexer.index_qk_proj.weight",
            ]
        )
    else:
        message = f"unknown layer type {kind}"
        raise RuntimeError(message)
    return keys


def _layer_f32_keys(prefix: str, kind: str) -> list[str]:
    keys = [
        f"{prefix}.attn_hyper_connection.hc_norm.weight",
        f"{prefix}.mlp_hyper_connection.hc_norm.weight",
    ]
    if kind == "linear_attention":
        keys.extend(
            [
                f"{prefix}.linear_attn.conv1d.weight",
                f"{prefix}.linear_attn.A_log",
                f"{prefix}.linear_attn.dt_bias",
                f"{prefix}.linear_attn.norm.weight",
            ]
        )
    else:
        keys.extend(
            [
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
            ]
        )
    return keys


def _write_expert_file(
    store: ShardStore,
    path: Path,
    gate_key: str,
    down_key: str,
    experts: int,
    start: int = 0,
    commit: Callable[[int], None] | None = None,
) -> None:
    live = expert_live_bytes()
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        full = experts * EXPERT_SLOT
        if start == 0 or os.fstat(fd).st_size < full:
            os.ftruncate(fd, full)
        for expert in range(start, experts):
            gate = _expert_matrix(store, gate_key, expert)
            down = _expert_matrix(store, down_key, expert)
            if gate.shape != (1280, 2560) or down.shape != (2560, 640):
                message = f"{gate_key} shapes {gate.shape} {down.shape}"
                raise RuntimeError(message)
            image = quantize_int4_image(gate)[0] + quantize_int4_image(down)[0]
            if len(image) != live:
                message = f"expert image is {len(image)} bytes, expected {live}"
                raise RuntimeError(message)
            os.pwrite(fd, image, expert * EXPERT_SLOT)
            if expert % 64 == 0:
                _log(f"{path.name} expert {expert}")
            if commit is not None and (expert % 64 == 63 or expert == experts - 1):
                os.fsync(fd)
                commit(expert)
    finally:
        os.close(fd)


def quantize_checkpoint(source: Path, dest: Path) -> None:
    """Write the Metal record once. A later run continues from quantize_progress.json."""
    if not (source / "config.json").is_file():
        message = f"checkpoint config is missing: {source}"
        raise FileNotFoundError(message)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        done = json.loads(manifest_path.read_text())
        if done.get("complete") and done.get("version") == RECORD_VERSION:
            _log(f"record already complete: {dest}")
            return
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(source / name, dest / name)
    progress = _load_progress(dest)
    store = ShardStore(source)
    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    kinds = list(text["layer_types"])
    experts = int(text["num_experts"])
    if progress["resident"] and not (dest / "resident.bin").is_file():
        progress["resident"] = False
    if not progress["resident"]:
        _log("quantize resident matrices")
        writer = ResidentWriter(dest / "resident.bin")
        try:
            writer.add_int4("lm_head.weight", store.matrix("lm_head.weight"))
            embed_key = "model.language_model.embed_tokens.weight"
            writer.add_int4(embed_key, store.matrix(embed_key))
            for layer, kind in enumerate(kinds):
                prefix = f"model.language_model.layers.{layer}"
                for key in _layer_int4_keys(prefix, kind):
                    writer.add_int4(key, store.matrix(key))
                gate_key = f"{prefix}.mlp.gate.weight"
                writer.add_f16(gate_key, store.matrix(gate_key))
                for key in _layer_f32_keys(prefix, kind):
                    writer.add_f32(key, store.matrix(key))
                _log(f"resident layer {layer}")
            ple = "model.language_model.layers.1.ple"
            table = f"{ple}.ple_embedding"
            for key in (f"{ple}.key_proj.weight", f"{ple}.value_proj.weight"):
                writer.add_int4(key, store.matrix(key))
            for key in (
                f"{ple}.conv1d.weight",
                f"{ple}.norm_conv.weight",
                f"{ple}.norm_key.weight",
                f"{ple}.norm_query.weight",
            ):
                writer.add_f32(key, store.matrix(key))
            for key in (
                f"{table}.layer_multipliers",
                f"{table}.ngram_heads_offsets",
                f"{table}.ngram_heads_vocab_sizes",
            ):
                writer.add_i64(key, store.i64(key))
            mixer = "model.language_model.hyper_connection_mixer"
            writer.add_f32(f"{mixer}.hc_norm.weight", store.matrix(f"{mixer}.hc_norm.weight"))
            writer.add_int4(
                f"{mixer}.input_mix_weight_down.weight",
                store.matrix(f"{mixer}.input_mix_weight_down.weight"),
            )
            writer.add_int4(
                f"{mixer}.input_mix_weight_up.weight",
                store.matrix(f"{mixer}.input_mix_weight_up.weight"),
            )
            mtp = "mtp.layers.0"
            for key in _layer_int4_keys(mtp, "full_attention"):
                writer.add_int4(key, store.matrix(key))
            writer.add_f16(f"{mtp}.mlp.gate.weight", store.matrix(f"{mtp}.mlp.gate.weight"))
            for key in _layer_f32_keys(mtp, "full_attention"):
                writer.add_f32(key, store.matrix(key))
            for key in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"):
                writer.add_int4(key, store.matrix(key))
            mtp_mix = "mtp.hyper_connection_mixer"
            writer.add_f32(f"{mtp_mix}.hc_norm.weight", store.matrix(f"{mtp_mix}.hc_norm.weight"))
            writer.add_int4(
                f"{mtp_mix}.input_mix_weight_down.weight",
                store.matrix(f"{mtp_mix}.input_mix_weight_down.weight"),
            )
            writer.add_int4(
                f"{mtp_mix}.input_mix_weight_up.weight",
                store.matrix(f"{mtp_mix}.input_mix_weight_up.weight"),
            )
            for key in ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"):
                writer.add_f32(key, store.matrix(key))
        finally:
            writer.close()
        progress["resident"] = True
        progress["catalog"] = {
            "int4": writer.int4,
            "f16": writer.f16,
            "f32": writer.f32,
            "i64": writer.i64,
        }
        _save_progress(dest, progress)
        _log("resident record written")
    catalog = progress["catalog"]
    expert_dir = dest / "experts"
    expert_dir.mkdir(exist_ok=True)
    done_layers = set(progress["layers"])
    for layer in range(len(kinds)):
        path = expert_dir / f"{layer:02d}.bin"
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        start = _expert_start(
            progress,
            done=layer in done_layers,
            inflight_key=layer,
            path=path,
            experts=experts,
        )
        if start >= experts:
            if layer not in done_layers:
                progress["layers"].append(layer)
                done_layers.add(layer)
                _clear_inflight(progress)
                _save_progress(dest, progress)
            continue
        if layer in done_layers:
            progress["layers"] = [item for item in progress["layers"] if item != layer]
            done_layers.discard(layer)
        if start:
            _log(f"resume experts layer {layer} at {start}")
        else:
            _log(f"quantize experts layer {layer}")

        def commit_expert(expert: int, layer: int = layer) -> None:
            progress["inflight_key"] = layer
            progress["inflight_expert"] = expert
            _save_progress(dest, progress)

        _write_expert_file(
            store,
            path,
            f"{prefix}.gate_up_proj",
            f"{prefix}.down_proj",
            experts,
            start,
            commit_expert,
        )
        progress["layers"].append(layer)
        done_layers.add(layer)
        _clear_inflight(progress)
        _save_progress(dest, progress)
    mtp_path = dest / "mtp-experts.bin"
    mtp_start = _expert_start(
        progress,
        done=bool(progress["mtp"]),
        inflight_key="mtp",
        path=mtp_path,
        experts=experts,
    )
    if mtp_start >= experts:
        if not progress["mtp"]:
            progress["mtp"] = True
            _clear_inflight(progress)
            _save_progress(dest, progress)
    else:
        if progress["mtp"]:
            progress["mtp"] = False
        if mtp_start:
            _log(f"resume mtp experts at {mtp_start}")
        else:
            _log("quantize mtp experts")

        def commit_mtp(expert: int) -> None:
            progress["inflight_key"] = "mtp"
            progress["inflight_expert"] = expert
            _save_progress(dest, progress)

        _write_expert_file(
            store,
            mtp_path,
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
            experts,
            mtp_start,
            commit_mtp,
        )
        progress["mtp"] = True
        _clear_inflight(progress)
        _save_progress(dest, progress)
    _write_ngram(store, dest, progress)
    manifest = {
        "version": RECORD_VERSION,
        "complete": True,
        "expert_slot": EXPERT_SLOT,
        "expert_live": expert_live_bytes(),
        "experts": experts,
        "layers": len(kinds),
        "ngram_rows": 128 * 2_500_012,
        "ngram_slot": NGRAM_SLOT,
        "ngram_width": NGRAM_WIDTH,
        "embed_key": "model.language_model.embed_tokens.weight",
        "head_key": "lm_head.weight",
        **catalog,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest))
    manifest_fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(manifest_fd)
    finally:
        os.close(manifest_fd)
    os.replace(tmp, manifest_path)
    _log(f"record complete: {dest}")


def _write_ngram(store: ShardStore, dest: Path, progress: dict[str, Any]) -> None:
    path = dest / "ngram.bin"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        rows_per = 2_500_012
        shards = 128
        total = shards * rows_per
        saved = int(progress["ngram_rows"])
        written = ngram_resume_row(saved, os.fstat(fd).st_size)
        if written != saved:
            progress["ngram_rows"] = written
            _save_progress(dest, progress)
            _log(f"n-gram cursor rewound to {written}")
        if written >= total:
            return
        if written:
            _log(f"resume n-gram at row {written}/{total}")
        shard = written // rows_per
        local = written % rows_per
        while shard < shards:
            key = (
                "model.language_model.layers.1.ple.ple_embedding."
                f"ngram_embedding.shard_{shard}.weight"
            )
            _fd, _start, _dtype, shape = store._locate(key)
            if list(shape) != [rows_per, NGRAM_WIDTH]:
                message = f"{key} shape {shape}"
                raise RuntimeError(message)
            while local < rows_per:
                count = min(4096, rows_per - local)
                matrix = store.read_bf16(
                    key, local * NGRAM_WIDTH, count * NGRAM_WIDTH, (count, NGRAM_WIDTH)
                )
                blob = ngram_slots(matrix)
                os.pwrite(fd, blob, (shard * rows_per + local) * NGRAM_SLOT)
                local += count
                written += count
                if local % (4096 * 16) == 0 or local == rows_per:
                    os.fsync(fd)
                    progress["ngram_rows"] = written
                    _save_progress(dest, progress)
                    _log(f"n-gram rows {written}/{total}")
            shard += 1
            local = 0
        os.fsync(fd)
        progress["ngram_rows"] = total
        _save_progress(dest, progress)
    finally:
        os.close(fd)


class QuantRecord:
    """The file the shaders read. It does not contain BF16 weights."""

    def __init__(self, root: Path) -> None:
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            message = (
                f"quantized record is missing: {manifest_path}\n"
                "madmetal --qwen38-flash-next quantize writes it. The request does not quantize."
            )
            raise FileNotFoundError(message)
        self.root = root
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("version") != RECORD_VERSION or not self.manifest.get("complete"):
            message = f"quantized record is incomplete: {root}"
            raise RuntimeError(message)
        self.resident_fd = os.open(root / "resident.bin", os.O_RDONLY)
        self.expert_fds = [
            os.open(root / "experts" / f"{layer:02d}.bin", os.O_RDONLY)
            for layer in range(int(self.manifest["layers"]))
        ]
        for fd in self.expert_fds:
            _bypass_cache(fd)
        self.ngram_fd = os.open(root / "ngram.bin", os.O_RDONLY)
        _bypass_cache(self.ngram_fd)
        self.live = int(self.manifest["expert_live"])
        self.slot = int(self.manifest["expert_slot"])
        self.expert_scratch = mmap.mmap(-1, self.live)
        self.page_scratch = mmap.mmap(-1, PAGE)
        self.pages: dict[int, bytes] = {}
        self.gate_up_bytes = sum(int4_plane_sizes(1280, stored_k_int4(2560, align_k=True)))

    def read_resident(self, offset: int, nbytes: int) -> bytes:
        blob = os.pread(self.resident_fd, nbytes, offset)
        if len(blob) != nbytes:
            message = f"resident read at {offset} returned {len(blob)} of {nbytes}"
            raise OSError(message)
        return blob

    def expert_image(self, layer: int, expert: int) -> bytes:
        view = pread_exact(
            self.expert_fds[layer], self.live, expert * self.slot, self.expert_scratch
        )
        blob = bytes(view)
        del view
        return blob

    def ngram_vector(self, row: int) -> np.ndarray:
        if row < 0 or row >= int(self.manifest["ngram_rows"]):
            message = f"n-gram row {row} is outside the table"
            raise IndexError(message)
        slot_off = row * NGRAM_SLOT
        page = (slot_off // PAGE) * PAGE
        cached = self.pages.get(page)
        if cached is None:
            cached = bytes(pread_exact(self.ngram_fd, PAGE, page, self.page_scratch))
            if len(self.pages) > 4096:
                self.pages.clear()
            self.pages[page] = cached
        local = slot_off - page
        slot = cached[local : local + NGRAM_SLOT]
        return dequant_int4(slot[:80], slot[80:85], 1, NGRAM_WIDTH)[0]


class _Bf16Ngram:
    """Gather one BF16 n-gram row from the checkpoint shards."""

    def __init__(self, store: ShardStore) -> None:
        self.store = store

    def ngram_vector(self, row: int) -> np.ndarray:
        return self.store.ngram_row(row, 2_500_012, NGRAM_WIDTH)


class Qwen38Runtime:
    """One sequence of Qwen3.8-Flash-Next. Linear maps are Metal shader dots."""

    def __init__(self, root: Path, *, bf16: bool = False) -> None:
        self.gpu = MetalSession()
        _log(f"shaders ready on {_device_name(self.gpu.device)}")
        self.root = root
        config = json.loads((root / "config.json").read_text())
        text = config["text_config"]
        self.hidden = int(text["hidden_size"])
        self.layers = int(text["num_hidden_layers"])
        self.experts = int(text["num_experts"])
        self.top_k = int(text["num_experts_per_tok"])
        self.moe = int(text["moe_intermediate_size"])
        self.vocab = int(text["vocab_size"])
        self.eos = int(text["eos_token_id"])
        self.stop_ids = {self.eos}
        self.eps = float(text["rms_norm_eps"])
        rope = text["rope_parameters"]
        self.rope_theta = float(rope["rope_theta"])
        self.rotary_dim = int(256 * float(rope["partial_rotary_factor"]))
        self.kinds = list(text["layer_types"])
        self.ple_layers = {int(item) for item in text["ple_layer_ids"]}
        self.branches = int(text["hc_count"])
        self.heads_per_ngram = int(text["heads_per_ngram"])
        self.ngram_size = int(text["ngram_size"])
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self._bf16 = bf16
        self.store: ShardStore | None = None
        self.files: QuantRecord | _Bf16Ngram
        self.weights: dict[str, Int4Matrix] = {}
        self.f16_weights: dict[str, F16Matrix] = {}
        self.f32: dict[str, np.ndarray] = {}
        self.i64: dict[str, np.ndarray] = {}
        self._embed_meta: dict[str, int] = {}
        self._embed_image = b""
        self.expert_cache: dict[tuple[int, int], tuple[Int4Matrix, Int4Matrix]] = {}
        self.tokens: list[int] = []
        self.last_logits: np.ndarray | None = None
        self._decode_tokens = 0
        self._decode_seconds = 0.0
        self.rng = np.random.default_rng()
        self.gdn_state = {
            index: self.gpu.zeros(48 * 128 * 128 * 4)
            for index, kind in enumerate(self.kinds)
            if kind == "linear_attention"
        }
        self.conv_hist: dict[int, list[np.ndarray]] = {index: [] for index in self.gdn_state}
        self.ple_hist: list[np.ndarray] = []
        self.kv: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            index: [] for index, kind in enumerate(self.kinds) if kind == "full_attention"
        }
        self.indexer_k: dict[int, list[np.ndarray]] = {index: [] for index in self.kv}
        self._bundle: dict[int, dict[str, Any]] = {}
        self._final: dict[str, Any] | None = None
        self._head: Int4Matrix | None = None
        self._ple: dict[str, Any] | None = None
        if bf16:
            self.store = ShardStore(root)
            self.files = _Bf16Ngram(self.store)
            self._load_bf16()
        else:
            self.files = QuantRecord(root)
            self._load_record()
        tok_cfg = json.loads((root / "tokenizer_config.json").read_text())
        self.chat_template = tok_cfg["chat_template"]
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._add_stop_token(tok_cfg.get("eos_token"))
        generation = root / "generation_config.json"
        if generation.is_file():
            eos_ids = json.loads(generation.read_text()).get("eos_token_id")
            if isinstance(eos_ids, int):
                self.stop_ids.add(eos_ids)
            elif isinstance(eos_ids, list):
                self.stop_ids.update(int(item) for item in eos_ids)

    def _add_stop_token(self, piece: Any) -> None:
        """A single-id end marker stops generation. The n-gram boundary stays `eos`."""
        if not isinstance(piece, str) or piece == "":
            return
        encoded = self.tokenizer.encode(piece).ids
        if len(encoded) == 1:
            self.stop_ids.add(int(encoded[0]))

    def reset(self) -> None:
        self.tokens = []
        self.last_logits = None
        for buffer in self.gdn_state.values():
            _fill_buffer(buffer, 48 * 128 * 128 * 4, 0)
        for history in self.conv_hist.values():
            history.clear()
        self.ple_hist.clear()
        for cache in self.kv.values():
            cache.clear()
        for cache in self.indexer_k.values():
            cache.clear()

    def align(self, prompt: list[int]) -> int:
        if prompt[: len(self.tokens)] != self.tokens:
            _log("prefix miss")
            self.reset()
            return 0
        _log(f"prefix hit {len(self.tokens)}")
        return len(self.tokens)

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[int]:
        start = self.align(prompt)
        if start == len(prompt):
            if self.last_logits is None:
                message = "the cached prefix has no logits"
                raise RuntimeError(message)
            logits = self.last_logits
            _log("prefill 0 tok")
        else:
            logits = self.forward(prompt[start:])
        tokens = list(self._new_tokens(logits, max_tokens, temperature, top_p, top_k))
        self._log_decode()
        return tokens

    def _new_tokens(
        self,
        logits: np.ndarray,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        self._decode_tokens = 0
        self._decode_seconds = 0.0
        for _ in range(max_tokens):
            token = _sample(logits, temperature, top_p, top_k, self.rng)
            if token in self.stop_ids:
                break
            yield token
            logits = self.forward([token], decode=True)

    def _log_decode(self) -> None:
        _log(_tok_s("decode", self._decode_tokens, self._decode_seconds))

    def forward(self, tokens: list[int], *, decode: bool = False) -> np.ndarray:
        started = time.perf_counter()
        residuals = [self._embed(token) for token in tokens]
        count = len(tokens)
        for layer in range(self.layers):
            # Command buffers retain the expert weights they read. Drain the
            # pool after the layer drops those weights, or a long BF16 decode
            # keeps every expert from the prompt resident.
            with objc.autorelease_pool():
                if (layer + 1) in self.ple_layers:
                    for index, token_residual in enumerate(residuals):
                        history = self.tokens + tokens[: index + 1]
                        memory = self._ple_memory(token_residual, history)
                        residuals[index] = token_residual + memory
                bundle = self._layer(layer)
                hats = np.stack(
                    [_rms(item, bundle["attn_norm"], self.eps) for item in residuals]
                )
                xs = self._gr_read(bundle["attn_down"], bundle["attn_up"], hats)
                if self.kinds[layer] == "linear_attention":
                    mixed = self._gdn_out(layer, bundle, xs)
                elif self.kinds[layer] == "full_attention":
                    mixed = self._qsa_out(layer, bundle, xs)
                else:
                    message = f"unknown layer type {self.kinds[layer]}"
                    raise RuntimeError(message)
                inject = self.gpu.int4_rows(bundle["attn_inject"], hats.reshape(count, -1))
                scales = 2.0 * _sigmoid(inject * 0.25)
                for index in range(count):
                    residuals[index] = residuals[index] + scales[index][:, None] * mixed[index]
                hats = np.stack(
                    [_rms(item, bundle["mlp_norm"], self.eps) for item in residuals]
                )
                xs = self._gr_read(bundle["mlp_down"], bundle["mlp_up"], hats)
                mixed = self._moe(layer, bundle, xs)
                if self._bf16:
                    held = len(self.expert_cache)
                    self.expert_cache.clear()
                inject = self.gpu.int4_rows(bundle["mlp_inject"], hats.reshape(count, -1))
                scales = 2.0 * _sigmoid(inject * 0.25)
                for index in range(count):
                    residuals[index] = residuals[index] + scales[index][:, None] * mixed[index]
                if self._bf16:
                    _log(f"layer {layer} stepped {count} experts {held}")
        final = self._final_bundle()
        hat = _rms(residuals[-1], final["norm"], self.eps)[None, ...]
        hidden = self._gr_read(final["down"], final["up"], hat)[0]
        logits = self.gpu.int4(self._lm_head(), hidden)
        self.tokens.extend(tokens)
        self.last_logits = logits
        elapsed = time.perf_counter() - started
        if decode:
            self._decode_tokens += len(tokens)
            self._decode_seconds += elapsed
        else:
            _log(_tok_s("prefill", len(tokens), elapsed))
        return logits

    def _load_record(self) -> None:
        _log("loading quantized record")
        manifest = self.files.manifest
        for key, meta in manifest["int4"].items():
            image = self.files.read_resident(meta["offset"], meta["data"] + meta["scales"])
            if key == manifest["embed_key"]:
                self._embed_meta = meta
                self._embed_image = image
                continue
            self.weights[key] = self.gpu.load_int4(
                image, meta["rows"], meta["logical_inner"], meta["stored_inner"]
            )
        for key, meta in manifest["f16"].items():
            blob = self.files.read_resident(meta["offset"], meta["rows"] * meta["inner"] * 2)
            self.f16_weights[key] = F16Matrix(meta["rows"], meta["inner"], self.gpu.upload(blob))
        for key, meta in manifest["f32"].items():
            count = int(np.prod(meta["shape"]))
            blob = self.files.read_resident(meta["offset"], count * 4)
            self.f32[key] = np.frombuffer(blob, dtype=np.float32).copy().reshape(meta["shape"])
        for key, meta in manifest["i64"].items():
            count = int(np.prod(meta["shape"]))
            blob = self.files.read_resident(meta["offset"], count * 8)
            self.i64[key] = np.frombuffer(blob, dtype=np.int64).copy().reshape(meta["shape"])
        _log(f"loaded {len(self.weights)} int4 matrices")

    def _load_bf16(self) -> None:
        """Upload the checkpoint's own values. Experts and n-gram rows stay on disk."""
        store = self.store
        if store is None:
            message = "BF16 load has no shard store"
            raise RuntimeError(message)
        _check_f32_gemv(self.gpu)
        _log("loading BF16 checkpoint onto Metal")
        self._head = self.gpu.load_f32(store.matrix("lm_head.weight"))
        for layer, kind in enumerate(self.kinds):
            prefix = f"model.language_model.layers.{layer}"
            for key in _layer_int4_keys(prefix, kind):
                self.weights[key] = self.gpu.load_f32(store.matrix(key))
            gate_key = f"{prefix}.mlp.gate.weight"
            self.f16_weights[gate_key] = self.gpu.load_f32(store.matrix(gate_key))
            for key in _layer_f32_keys(prefix, kind):
                self.f32[key] = store.matrix(key)
            _log(f"bf16 layer {layer}")
        ple = "model.language_model.layers.1.ple"
        table = f"{ple}.ple_embedding"
        for key in (f"{ple}.key_proj.weight", f"{ple}.value_proj.weight"):
            self.weights[key] = self.gpu.load_f32(store.matrix(key))
        for key in (
            f"{ple}.conv1d.weight",
            f"{ple}.norm_conv.weight",
            f"{ple}.norm_key.weight",
            f"{ple}.norm_query.weight",
        ):
            self.f32[key] = store.matrix(key)
        for key in (
            f"{table}.layer_multipliers",
            f"{table}.ngram_heads_offsets",
            f"{table}.ngram_heads_vocab_sizes",
        ):
            self.i64[key] = store.i64(key)
        mixer = "model.language_model.hyper_connection_mixer"
        self.f32[f"{mixer}.hc_norm.weight"] = store.matrix(f"{mixer}.hc_norm.weight")
        for key in (
            f"{mixer}.input_mix_weight_down.weight",
            f"{mixer}.input_mix_weight_up.weight",
        ):
            self.weights[key] = self.gpu.load_f32(store.matrix(key))
        _log(f"loaded {len(self.weights)} bf16 matrices")

    def _embed(self, token: int) -> np.ndarray:
        if self._bf16:
            if self.store is None:
                message = "BF16 embed has no shard store"
                raise RuntimeError(message)
            key = "model.language_model.embed_tokens.weight"
            vector = self.store.read_bf16(key, token * self.hidden, self.hidden, (self.hidden,))
            return np.broadcast_to(vector, (4, self.hidden)).copy()
        meta = self._embed_meta
        stored = meta["stored_inner"]
        data_row = stored // 2
        scale_row = stored // BLOCK
        data = self._embed_image[token * data_row : (token + 1) * data_row]
        scale_at = meta["data"] + token * scale_row
        scales = self._embed_image[scale_at : scale_at + scale_row]
        vector = dequant_int4(bytes(data), bytes(scales), 1, stored)[0, : self.hidden]
        return np.broadcast_to(vector, (4, self.hidden)).copy()

    def _gr_read(self, down: Int4Matrix, up: Int4Matrix, hats: np.ndarray) -> np.ndarray:
        count = hats.shape[0]
        # Tech report eq. (31): SiLU((1/n_r) W_d vec(R̂)). The branch mean is the
        # second 1/n_r.
        scale = 1.0 / self.branches
        projected = self.gpu.int4_rows(down, hats.reshape(count, -1))
        hidden = _silu(projected * scale)
        gate = _sigmoid(self.gpu.int4_rows(up, hidden)).reshape(
            count, self.branches, self.hidden
        )
        return np.mean(gate * hats, axis=1)

    def _layer(self, layer: int) -> dict[str, Any]:
        cached = self._bundle.get(layer)
        if cached is not None:
            return cached
        prefix = f"model.language_model.layers.{layer}"
        bundle: dict[str, Any] = {
            "attn_norm": self._norm(f"{prefix}.attn_hyper_connection.hc_norm.weight"),
            "attn_down": self._int4(f"{prefix}.attn_hyper_connection.input_mix_weight_down.weight"),
            "attn_up": self._int4(f"{prefix}.attn_hyper_connection.input_mix_weight_up.weight"),
            "attn_inject": self._int4(f"{prefix}.attn_hyper_connection.block_inject_weight.weight"),
            "mlp_norm": self._norm(f"{prefix}.mlp_hyper_connection.hc_norm.weight"),
            "mlp_down": self._int4(f"{prefix}.mlp_hyper_connection.input_mix_weight_down.weight"),
            "mlp_up": self._int4(f"{prefix}.mlp_hyper_connection.input_mix_weight_up.weight"),
            "mlp_inject": self._int4(f"{prefix}.mlp_hyper_connection.block_inject_weight.weight"),
            "shared_gate": self._int4(f"{prefix}.mlp.shared_expert.gate_proj.weight"),
            "shared_up": self._int4(f"{prefix}.mlp.shared_expert.up_proj.weight"),
            "shared_down": self._int4(f"{prefix}.mlp.shared_expert.down_proj.weight"),
            "shared_coeff": self._int4(f"{prefix}.mlp.shared_expert_gate.weight"),
        }
        if self.kinds[layer] == "linear_attention":
            bundle.update(
                {
                    "qkv": self._int4(f"{prefix}.linear_attn.in_proj_qkv.weight"),
                    "z": self._int4(f"{prefix}.linear_attn.in_proj_z.weight"),
                    "a": self._int4(f"{prefix}.linear_attn.in_proj_a.weight"),
                    "b": self._int4(f"{prefix}.linear_attn.in_proj_b.weight"),
                    "out": self._int4(f"{prefix}.linear_attn.out_proj.weight"),
                    "conv": self._f32(f"{prefix}.linear_attn.conv1d.weight").reshape(10240, 4),
                    "a_log": self._f32(f"{prefix}.linear_attn.A_log").reshape(-1),
                    "dt_bias": self._f32(f"{prefix}.linear_attn.dt_bias").reshape(-1),
                    "norm": self._f32(f"{prefix}.linear_attn.norm.weight").reshape(-1),
                }
            )
        else:
            bundle.update(
                {
                    "q_proj": self._int4(f"{prefix}.self_attn.q_proj.weight"),
                    "k_proj": self._int4(f"{prefix}.self_attn.k_proj.weight"),
                    "v_proj": self._int4(f"{prefix}.self_attn.v_proj.weight"),
                    "o_proj": self._int4(f"{prefix}.self_attn.o_proj.weight"),
                    "q_norm": self._f32(f"{prefix}.self_attn.q_norm.weight").reshape(-1),
                    "k_norm": self._f32(f"{prefix}.self_attn.k_norm.weight").reshape(-1),
                    "index_qk": self._int4(f"{prefix}.self_attn.indexer.index_qk_proj.weight"),
                }
            )
        self._bundle[layer] = bundle
        return bundle

    def _norm(self, key: str) -> np.ndarray:
        return self._f32(key).reshape(4, self.hidden)

    def _f32(self, key: str) -> np.ndarray:
        return self.f32[key]

    def _int4(self, key: str) -> Int4Matrix:
        return self.weights[key]

    def _gdn_out(self, layer: int, bundle: dict[str, Any], xs: np.ndarray) -> np.ndarray:
        qkv = self.gpu.int4_rows(bundle["qkv"], xs)
        z = self.gpu.int4_rows(bundle["z"], xs)
        raw_a = self.gpu.int4_rows(bundle["a"], xs)
        raw_b = self.gpu.int4_rows(bundle["b"], xs)
        heads = []
        for index in range(xs.shape[0]):
            mixed = _silu(_causal_conv(self.conv_hist[layer], qkv[index], bundle["conv"]))
            query = _l2norm(mixed[:2048].reshape(16, 128), self.eps)
            key = _l2norm(mixed[2048:4096].reshape(16, 128), self.eps)
            # L2 norm, then the query is scaled by 1/sqrt(head_dim). The key is not.
            query = query * (query.shape[-1] ** -0.5)
            value = mixed[4096:].reshape(48, 128)
            query = np.repeat(query, 3, axis=0)
            key = np.repeat(key, 3, axis=0)
            beta = _sigmoid(raw_b[index])
            alpha = np.exp(-np.exp(bundle["a_log"]) * _softplus(raw_a[index] + bundle["dt_bias"]))
            y = self.gpu.gdn(self.gdn_state[layer], query, key, value, alpha, beta)
            y = _rms(y.reshape(48, 128), bundle["norm"], self.eps, zero_centered=False).reshape(-1)
            heads.append(y * _sigmoid(z[index]))
        return self.gpu.int4_rows(bundle["out"], np.stack(heads))

    def _qsa_out(self, layer: int, bundle: dict[str, Any], xs: np.ndarray) -> np.ndarray:
        query = self.gpu.int4_rows(bundle["q_proj"], xs)
        key = self.gpu.int4_rows(bundle["k_proj"], xs)
        value = self.gpu.int4_rows(bundle["v_proj"], xs)
        index_qk = self.gpu.int4_rows(bundle["index_qk"], xs)
        gated = []
        base = len(self.tokens)
        for index in range(xs.shape[0]):
            position = base + index
            # Fused q_proj is per head [query, sigmoid gate], 256 + 256.
            # The tech report splits this projection at head granularity.
            packed = query[index].reshape(24, 2, 256)
            q = _rms(packed[:, 0], bundle["q_norm"], self.eps)
            k = _rms(key[index].reshape(2, 256), bundle["k_norm"], self.eps)
            v = value[index].reshape(2, 256)
            q = _rope(q, position, self.rotary_dim, self.rope_theta)
            k = _rope(k, position, self.rotary_dim, self.rope_theta)
            self.kv[layer].append((k, v))
            self.indexer_k[layer].append(index_qk[index, 512:640].copy())
            tokens = len(self.kv[layer])
            if tokens // 4 > 512:
                message = "sequence is past the 2048-token indexer budget"
                raise RuntimeError(message)
            k_stack = np.stack([item[0] for item in self.kv[layer]])
            v_stack = np.stack([item[1] for item in self.kv[layer]])
            context = self.gpu.attend(q, k_stack, v_stack).reshape(24, 256)
            gate = _sigmoid(packed[:, 1])
            gated.append((context * gate).reshape(-1))
        return self.gpu.int4_rows(bundle["o_proj"], np.stack(gated))

    def _moe(self, layer: int, bundle: dict[str, Any], xs: np.ndarray) -> np.ndarray:
        gate = _silu(self.gpu.int4_rows(bundle["shared_gate"], xs))
        up = self.gpu.int4_rows(bundle["shared_up"], xs)
        shared = self.gpu.int4_rows(bundle["shared_down"], gate * up)
        coeff = _sigmoid(self.gpu.int4_rows(bundle["shared_coeff"], xs))
        shared = shared * coeff
        router = self._router(layer)
        outputs = []
        for index in range(xs.shape[0]):
            logits = self.gpu.f16(router, xs[index])
            choice, weight = _topk(logits, self.top_k)
            picked = [int(expert) for expert in choice.tolist()]
            for expert in sorted(picked):
                self._ensure_expert(layer, expert)
            mixed = shared[index].copy()
            for expert, scale in zip(picked, weight.tolist(), strict=True):
                mixed += scale * self._routed(layer, expert, xs[index])
            outputs.append(mixed)
        return np.stack(outputs)

    def _router(self, layer: int) -> F16Matrix:
        return self.f16_weights[f"model.language_model.layers.{layer}.mlp.gate.weight"]

    def _ensure_expert(self, layer: int, expert: int) -> tuple[Int4Matrix, Int4Matrix]:
        cached = self.expert_cache.get((layer, expert))
        if cached is not None:
            return cached
        if self._bf16:
            if self.store is None:
                message = "BF16 expert load has no shard store"
                raise RuntimeError(message)
            gate = self.gpu.load_f32(self.store.expert(layer, "gate_up_proj", expert))
            down = self.gpu.load_f32(self.store.expert(layer, "down_proj", expert))
            cached = (gate, down)
            self.expert_cache[(layer, expert)] = cached
            return cached
        image = self.files.expert_image(layer, expert)
        split = self.files.gate_up_bytes
        stored_down = stored_k_int4(self.moe, align_k=True)
        gate_up = self.gpu.load_int4(image[:split], 1280, self.hidden, self.hidden)
        down = self.gpu.load_int4(image[split:], self.hidden, self.moe, stored_down)
        cached = (gate_up, down)
        self.expert_cache[(layer, expert)] = cached
        return cached

    def _routed(self, layer: int, expert: int, vector: np.ndarray) -> np.ndarray:
        gate_up, down = self._ensure_expert(layer, expert)
        hidden = self.gpu.int4(gate_up, vector)
        activated = _silu(hidden[: self.moe]) * hidden[self.moe :]
        return self.gpu.int4(down, activated)

    def _ple_memory(self, residual: np.ndarray, tokens: list[int]) -> np.ndarray:
        """N-gram memory injected before this layer's residual read."""
        ple = self._ple_bundle()
        rows = []
        for head in range(self.ngram_heads):
            index = _ngram_row(
                tokens,
                head,
                ple["mult"],
                ple["offset"],
                ple["vocab"],
                heads_per_ngram=self.heads_per_ngram,
                ngram_size=self.ngram_size,
                eos=self.eos,
            )
            rows.append(self.files.ngram_vector(index))
        embedding = np.concatenate(rows)
        key = self.gpu.int4(ple["key"], embedding).reshape(self.branches, self.hidden)
        value = self.gpu.int4(ple["value"], embedding)
        query = _rms(residual, ple["norm_query"].reshape(self.branches, self.hidden), self.eps)
        keyed = _rms(key, ple["norm_key"].reshape(self.branches, self.hidden), self.eps)
        scaled = np.sum(query * keyed, axis=-1) / math.sqrt(self.hidden)
        score = np.sign(scaled) * np.sqrt(np.maximum(np.abs(scaled), 1e-6))
        gate = _sigmoid(score)
        updated = gate[:, None] * value
        normed = _rms(updated, ple["norm_conv"].reshape(self.branches, self.hidden), self.eps)
        conv = _dilated_conv(self.ple_hist, normed.reshape(-1), ple["conv"], self.ngram_size)
        return updated + _silu(conv.reshape(self.branches, self.hidden))

    def _ple_bundle(self) -> dict[str, Any]:
        if self._ple is not None:
            return self._ple
        base = "model.language_model.layers.1.ple"
        table = "model.language_model.layers.1.ple.ple_embedding"
        self._ple = {
            "key": self._int4(f"{base}.key_proj.weight"),
            "value": self._int4(f"{base}.value_proj.weight"),
            "conv": self._f32(f"{base}.conv1d.weight").reshape(10240, 4),
            "norm_conv": self._f32(f"{base}.norm_conv.weight").reshape(-1),
            "norm_key": self._f32(f"{base}.norm_key.weight").reshape(-1),
            "norm_query": self._f32(f"{base}.norm_query.weight").reshape(-1),
            "mult": self.i64[f"{table}.layer_multipliers"],
            "offset": self.i64[f"{table}.ngram_heads_offsets"],
            "vocab": self.i64[f"{table}.ngram_heads_vocab_sizes"],
        }
        return self._ple

    def _final_bundle(self) -> dict[str, Any]:
        if self._final is None:
            base = "model.language_model.hyper_connection_mixer"
            self._final = {
                "norm": self._norm(f"{base}.hc_norm.weight"),
                "down": self._int4(f"{base}.input_mix_weight_down.weight"),
                "up": self._int4(f"{base}.input_mix_weight_up.weight"),
            }
        return self._final

    def _lm_head(self) -> Int4Matrix:
        if self._head is None:
            self._head = self._int4("lm_head.weight")
        return self._head

    def encode_chat(
        self, messages: list[dict[str, Any]], tools: Any, enable_thinking: bool
    ) -> list[int]:
        from jinja2 import Environment

        def raise_exception(message: str) -> None:
            raise RuntimeError(message)

        template = Environment().from_string(self.chat_template)
        text = template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            raise_exception=raise_exception,
        )
        return list(self.tokenizer.encode(text).ids)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)


def _topk(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    probs = np.exp(shifted)
    probs /= probs.sum()
    choice = np.argpartition(probs, -k)[-k:]
    choice = choice[np.argsort(probs[choice])[::-1]]
    weight = probs[choice]
    weight = weight / weight.sum()
    return choice, weight.astype(np.float32)


def _checkpoint(path: Path | None) -> Path:
    if path is None:
        message = "pass --source with the checkpoint directory"
        raise FileNotFoundError(message)
    if not (path / "config.json").is_file():
        message = f"checkpoint config is missing: {path}"
        raise FileNotFoundError(message)
    return path


def _record(path: Path | None) -> Path:
    return path or DEFAULT_RECORD


def quantize_main(args: Any) -> int:
    try:
        quantize_checkpoint(_checkpoint(args.source), args.output)
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    return 0


def bf16_complete(args: Any) -> int:
    """Same forward as complete, with Metal dots of the checkpoint BF16 values."""
    try:
        runtime = Qwen38Runtime(_checkpoint(args.source), bf16=True)
        if args.raw:
            prompt = list(runtime.tokenizer.encode(args.prompt).ids)
        else:
            messages = [{"role": "user", "content": args.prompt}]
            prompt = runtime.encode_chat(messages, None, enable_thinking=args.thinking)
        _log(f"prompt tokens {len(prompt)} ids {prompt}")
        logits = runtime.forward(prompt)
        order = np.argsort(logits)[::-1]
        top = " ".join(
            f"{float(logits[i]):.2f}:{int(i)}:{runtime.decode([int(i)])!r}" for i in order[:8]
        )
        _log(f"top {top}")
        four = runtime.tokenizer.encode("4").ids
        if len(four) == 1:
            rank = int(np.flatnonzero(order == four[0])[0])
            _log(f"4 rank {rank} logit {float(logits[four[0]]):.2f}")
        produced: list[int] = []
        shown = ""
        for token in runtime._new_tokens(
            logits, args.max_tokens, args.temperature, args.top_p, args.top_k
        ):
            produced.append(token)
            text = runtime.decode(produced)
            sys.stdout.write(text[len(shown) :])
            sys.stdout.flush()
            shown = text
        sys.stdout.write("\n")
        sys.stdout.flush()
        runtime._log_decode()
        _append_transcript(args.transcript, args.prompt, shown)
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    return 0


def complete(args: Any) -> int:
    try:
        runtime = Qwen38Runtime(_record(args.record))
        if args.raw:
            prompt = list(runtime.tokenizer.encode(args.prompt).ids)
        else:
            messages = [{"role": "user", "content": args.prompt}]
            prompt = runtime.encode_chat(messages, None, enable_thinking=not args.thinking)
        _log(f"prompt tokens {len(prompt)} ids {prompt}")
        logits = runtime.forward(prompt)
        order = np.argsort(logits)[::-1]
        top = " ".join(
            f"{float(logits[i]):.2f}:{int(i)}:{runtime.decode([int(i)])!r}" for i in order[:8]
        )
        _log(f"top {top}")
        four = runtime.tokenizer.encode("4").ids
        if len(four) == 1:
            rank = int(np.flatnonzero(order == four[0])[0])
            _log(f"4 rank {rank} logit {float(logits[four[0]]):.2f}")
        produced: list[int] = []
        shown = ""
        for token in runtime._new_tokens(
            logits, args.max_tokens, args.temperature, args.top_p, args.top_k
        ):
            produced.append(token)
            text = runtime.decode(produced)
            sys.stdout.write(text[len(shown) :])
            sys.stdout.flush()
            shown = text
        sys.stdout.write("\n")
        sys.stdout.flush()
        runtime._log_decode()
        _append_transcript(args.transcript, args.prompt, shown)
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    return 0


def serve(args: Any) -> int:
    try:
        runtime = Qwen38Runtime(_record(args.record))
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *log_args: Any) -> None:
            _log(fmt % log_args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/v1/models":
                self._json(404, {"error": {"message": "not found"}})
                return
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen38-flash-next",
                            "object": "model",
                            "owned_by": "madmetal",
                        }
                    ],
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return
            self._started = False
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                messages = body["messages"]
                tools = body.get("tools")
                enable_thinking = bool(body.get("enable_thinking", False))
                max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 64)
                temperature = float(body.get("temperature", 0.0))
                top_p = float(body.get("top_p", 1.0))
                top_k = int(body.get("top_k", 0))
                stream = bool(body.get("stream", False))
                with lock:
                    prompt = runtime.encode_chat(messages, tools, enable_thinking)
                    _log(f"prompt tokens {len(prompt)}")
                    if stream:
                        self._stream(
                            prompt,
                            max_tokens,
                            temperature,
                            top_p,
                            top_k,
                            _prompt_text(messages),
                        )
                    else:
                        tokens = runtime.generate(
                            prompt, max_tokens, temperature, top_p, top_k
                        )
                        text = runtime.decode(tokens)
                        _log(f"completion: {text}")
                        _append_transcript(args.transcript, _prompt_text(messages), text)
                        reason = "length" if len(tokens) == max_tokens else "stop"
                        self._json(200, _completion(text, len(prompt), len(tokens), reason))
            except Exception as exc:
                traceback.print_exc()
                if self._started:
                    return
                self._json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})

        def _stream(
            self,
            prompt: list[int],
            max_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
            prompt_text: str,
        ) -> None:
            self._started = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._event(
                {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            )
            start = runtime.align(prompt)
            if start == len(prompt):
                if runtime.last_logits is None:
                    message = "the cached prefix has no logits"
                    raise RuntimeError(message)
                logits = runtime.last_logits
                _log("prefill 0 tok")
            else:
                logits = runtime.forward(prompt[start:])
            produced: list[int] = []
            previous = ""
            for token in runtime._new_tokens(logits, max_tokens, temperature, top_p, top_k):
                produced.append(token)
                text = runtime.decode(produced)
                self._event(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text[len(previous) :]},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                previous = text
            reason = "length" if len(produced) == max_tokens else "stop"
            self._event({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            runtime._log_decode()
            _append_transcript(args.transcript, prompt_text, previous)

        def _event(self, payload: dict[str, Any]) -> None:
            blob = json.dumps(payload).encode()
            self.wfile.write(b"data: " + blob + b"\n\n")
            self.wfile.flush()

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            blob = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    _log(f"listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def _completion(
    text: str, prompt_tokens: int, completion_tokens: int, finish_reason: str
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-madmetal",
        "object": "chat.completion",
        "model": "qwen38-flash-next",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="madmetal --qwen38-flash-next")
    sub = parser.add_subparsers(dest="command", required=True)
    probe_parser = sub.add_parser("probe", help="measure one block of 32 on this GPU")
    probe_parser.add_argument("--json", action="store_true", help="print the measurement as JSON")
    sub.add_parser("matmul", help="check int4 E8M0 prefill tiles against the decode dot")
    sub.add_parser("kernels", help="check the GDN delta and the attention shader")
    quant_parser = sub.add_parser("quantize", help="write the Metal record from BF16 shards")
    quant_parser.add_argument("--source", type=Path, default=None)
    quant_parser.add_argument("--output", type=Path, default=DEFAULT_RECORD)
    serve_parser = sub.add_parser("serve", help="serve the quantized record")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--record", type=Path, default=None)
    serve_parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="append each prompt and completion to this file",
    )
    bf16_parser = sub.add_parser("bf16", help="complete from the BF16 checkpoint with Metal dots")
    bf16_parser.add_argument("--prompt", required=True)
    bf16_parser.add_argument("--max-tokens", type=int, default=8)
    bf16_parser.add_argument("--temperature", type=float, default=0.0)
    bf16_parser.add_argument("--top-p", type=float, default=1.0)
    bf16_parser.add_argument("--top-k", type=int, default=0)
    bf16_parser.add_argument("--thinking", action="store_true")
    bf16_parser.add_argument(
        "--raw", action="store_true", help="tokenize the prompt with no chat template"
    )
    bf16_parser.add_argument("--source", type=Path, default=None)
    bf16_parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="append the prompt and the completion to this file",
    )
    complete_parser = sub.add_parser("complete", help="print one completion from the record")
    complete_parser.add_argument("--prompt", required=True)
    complete_parser.add_argument("--max-tokens", type=int, default=8)
    complete_parser.add_argument("--temperature", type=float, default=0.0)
    complete_parser.add_argument("--top-p", type=float, default=1.0)
    complete_parser.add_argument("--top-k", type=int, default=0)
    complete_parser.add_argument("--thinking", action="store_true")
    complete_parser.add_argument(
        "--raw", action="store_true", help="tokenize the prompt with no chat template"
    )
    complete_parser.add_argument("--record", type=Path, default=None)
    complete_parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="append the prompt and the completion to this file",
    )
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command == "probe":
        return probe(as_json=args.json)
    if args.command == "matmul":
        return matmul()
    if args.command == "kernels":
        try:
            check_state_kernels()
        except Exception as exc:
            sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
            return 1
        sys.stdout.write("gdn delta and attention shaders: ok\n")
        return 0
    if args.command == "quantize":
        return quantize_main(args)
    if args.command == "serve":
        return serve(args)
    if args.command == "bf16":
        return bf16_complete(args)
    if args.command == "complete":
        return complete(args)
    return 2
