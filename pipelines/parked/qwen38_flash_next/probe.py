"""Block-of-32 roundtrip and the int4 prefill tile check."""

from __future__ import annotations

import ctypes
import json
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import objc

from favlo.memory import memory
from qwen38_flash_next.formats import (
    BLOCK,
    FP8_PATTERN,
    HOST_BYTES,
    INT2_PACKED,
    INT4_PACKED,
    INT8_PATTERN,
    PLANE_DATA,
    PLANE_SCALES,
    SENTINEL,
    TENSOR_E4M3,
    TENSOR_E5M2,
    TENSOR_E8M0,
    TENSOR_INT2,
    TENSOR_INT4,
    TENSOR_INT8,
    _finite_equal,
    decode_e4m3,
    decode_e5m2,
    decode_e8m0,
    dequant_int4,
    json_number,
    pack_int2,
    pack_int4,
    quantize_matrix,
    sign_extend,
    unpack_int4,
)
from qwen38_flash_next.session import (
    Int4Gemm,
    _buffer_bytes,
    _device_name,
    _fill_buffer,
    _metal,
    _objc_pair,
    _write_u32,
)
from qwen38_flash_next.shader_text import shader_text

SHADER = shader_text("probe.metal")


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


class Roundtrip:
    """One shared device and one shader library for the block probe."""

    def __init__(self) -> None:
        metal = _metal()
        self._owner = f"qwen38-flash-next:{id(self)}"
        self.metal = metal
        self.device = memory.device
        options = metal.MTLCompileOptions.alloc().init()
        # Metal 4.1. PyObjC does not export MTLLanguageVersion4_1.
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = _objc_pair(
            self.device.newLibraryWithSource_options_error_(SHADER, options, None)
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
        return memory.allocate(max(int(length), 16), owner=self._owner, zero=True)

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
        memory.bind_plane(attachments, PLANE_DATA, buffer, 0)
        if with_scale:
            plane = metal.MTLTensorAuxiliaryPlaneDescriptor.alloc().init()
            plane.setDataType_(TENSOR_E8M0)
            plane.setBlockFactors_(_extents([BLOCK]))
            planes = metal.MTLTensorAuxiliaryPlaneDescriptorMap.alloc().init()
            planes.setDescriptor_forPlane_(plane, PLANE_SCALES)
            descriptor.setAuxiliaryPlanes_(planes)
            memory.bind_plane(attachments, PLANE_SCALES, buffer, 128)
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
        memory.bind(encoder, 0, source, source_offset)
        memory.bind(encoder, 1, out, 0)
        memory.bind(encoder, 2, count_buf, 0)
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
        memory.bind(encoder, 0, source, source_offset)
        if scale_buffer is None:
            memory.bind(encoder, 1, out, 0)
        else:
            memory.bind(encoder, 1, scale_buffer, scale_offset)
            memory.bind(encoder, 2, out, 0)
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
        queue = memory.make_queue()
        command = memory.command_buffer(queue)
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(created)
        return encoder, command

    def _commit(self, encoder: Any, command: Any, threads: int) -> None:
        size = self.metal.MTLSizeMake(threads, 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(size, size)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        memory.retire(command)


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


def matmul_reference_tile() -> dict[str, Any]:
    """One 64×32×32 fp4 tile. Activation is 1 on inner 0. Column n stores magnitude n%8."""

    rows, cols, inner = 64, 32, 32
    activation = np.zeros((rows, inner), dtype=np.float16)
    activation[:, 0] = np.float16(1.0)
    magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    column = magnitudes[np.arange(cols) % 8]
    codes = np.zeros((cols, inner), dtype=np.uint8)
    codes[:, 0] = np.arange(cols, dtype=np.uint8) % 8
    packed = bytearray()
    flat = codes.reshape(-1)
    for index in range(0, flat.size, 2):
        packed.append(int(flat[index] & 0xF) | (int(flat[index + 1] & 0xF) << 4))
    scales = bytes([130]) * (cols * (inner // BLOCK))
    expected = column * np.float32(8.0)
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
    try:
        return _check_int4_gemm(gemm, tile)
    finally:
        memory.sweep(gemm._owner)


def _check_int4_gemm(gemm: Int4Gemm, tile: dict[str, Any]) -> dict[str, Any]:
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
    fp8_ok = _fp8_prefill_matches(gemm)
    return {
        "device": _device_name(gemm.device),
        "ok": prefill_ok and decode_ok and wide_ok and fp8_ok,
        "prefill": prefill_ok,
        "decode": decode_ok,
        "wide": wide_ok,
        "fp8": fp8_ok,
    }


def _fp8_prefill_matches(gemm: Int4Gemm) -> bool:
    """One fp8 E4M3 tile. Code 0x38 is 1.0 and scale byte 127 is 2^0."""

    rows, cols, inner = 64, 32, 32
    activation = np.zeros((rows, inner), dtype=np.float16)
    activation[:, 0] = np.float16(1.0)
    weights = bytearray(cols * inner)
    for row in range(cols):
        weights[row * inner] = 0x38
    scales = bytes([127]) * (cols * (inner // BLOCK))
    got = np_from_f32(
        gemm.run_prefill_fp8(activation.tobytes(), bytes(weights), scales, rows, cols, inner)
    ).reshape(rows, cols)
    return bool(np.allclose(got, np.float32(1.0), rtol=1e-3, atol=1e-3))


def _wide_prefill_matches(gemm: Int4Gemm) -> bool:
    """Two row tiles and two column tiles. A 32-column tile hides a bad N offset."""

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

    return np.frombuffer(blob, dtype=np.float32).copy()


def probe(*, as_json: bool) -> int:
    trip = Roundtrip()
    try:
        try:
            report = trip.measure()
        except Exception as exc:
            sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
            return 1
        if as_json:
            json.dump(report, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            sys.stdout.write(summarize(report) + "\n")
        return 0 if report["ok"] else 1
    finally:
        memory.sweep(trip._owner)


def matmul() -> int:
    try:
        report = check_int4_gemm()
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    sys.stdout.write(
        f"device: {report['device']}\n"
        f"prefill matmul2d fp4 e8m0: {'ok' if report['prefill'] else 'mismatch'}\n"
        f"prefill matmul2d fp8 e8m0: {'ok' if report['fp8'] else 'mismatch'}\n"
        f"prefill matmul2d 128x64x64: {'ok' if report['wide'] else 'mismatch'}\n"
        f"decode shader dot fp4 e8m0: {'ok' if report['decode'] else 'mismatch'}\n"
    )
    return 0 if report["ok"] else 1
