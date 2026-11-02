"""Key and value prefix for Gemma 4.

Local and global regions each store keys and values. The filled length is
``count`` tokens. The live cache is Metal ``half``, and the prefix file stores
that same device layout for positions ``0 .. count-1``. Keys and values are
token-major ``[count, heads, dim]``. An exact-length prefix is not a sample:
the next token still has to be stepped.
"""

from __future__ import annotations

from favlo.memory import device_buffer, memory
from favlo.prefix import PrefixHeader, Region, read_regions
from gemma4_26b_a4b.geometry import (
    GLOBAL_DIM,
    GLOBAL_HEADS,
    GLOBAL_LAYERS,
    LOCAL_DIM,
    LOCAL_HEADS,
    local_layers,
)
from gemma4_26b_a4b.planes import pack_matrix, restore_matrix

LOCAL = "local"
GLOBAL = "global"


class HostCache:
    """Filled key and value planes. Keys are ``[layers, heads, count, dim]``."""

    def __init__(self) -> None:
        self.count = 0
        self.local_k: bytes = b""
        self.local_v: bytes = b""
        self.global_k: bytes = b""


def half_plane_nbytes(count: int, heads: int, dim: int) -> int:
    """Metal half bytes for one filled cache plane."""
    if count <= 0 or heads <= 0 or dim <= 0:
        return 0
    return count * heads * dim * 2


def local_nbytes(count: int) -> int:
    layers = len(local_layers())
    one = half_plane_nbytes(count, LOCAL_HEADS, LOCAL_DIM)
    return layers * one * 2


def global_nbytes(count: int) -> int:
    one = half_plane_nbytes(count, GLOBAL_HEADS, GLOBAL_DIM)
    return len(GLOBAL_LAYERS) * one * 2


def _plane_bytes(plane: object, layers: int, heads: int, count: int, dim: int) -> bytes:
    """One plane in the live device order: each layer is ``[count, heads, dim]``."""
    import numpy as np

    values = _as_half(plane, (layers, heads, count, dim))
    parts: list[bytes] = []
    for layer in range(layers):
        token_major = np.ascontiguousarray(np.transpose(values[layer], (1, 0, 2)))
        parts.append(token_major.tobytes())
    return b"".join(parts)


def pack_local(keys: object, values: object, count: int) -> bytes:
    layers = len(local_layers())
    return _plane_bytes(keys, layers, LOCAL_HEADS, count, LOCAL_DIM) + _plane_bytes(
        values, layers, LOCAL_HEADS, count, LOCAL_DIM
    )


def pack_global(keys: object, values: object, count: int) -> bytes:
    layers = len(GLOBAL_LAYERS)
    return _plane_bytes(keys, layers, GLOBAL_HEADS, count, GLOBAL_DIM) + _plane_bytes(
        values, layers, GLOBAL_HEADS, count, GLOBAL_DIM
    )


def pack_token_major(raw: object, heads: int, dim: int, count: int) -> bytes:
    """Pack one ``[count, heads, dim]`` Metal half plane in device order."""
    if count <= 0 or heads <= 0:
        return b""
    import numpy as np

    plane = np.ndarray((count, heads, dim), dtype=np.float16, buffer=raw)
    return np.ascontiguousarray(plane).tobytes()


def _as_half(plane: object, shape: tuple[int, ...]) -> object:
    import numpy as np

    values = np.asarray(plane, dtype=np.float16)
    if values.shape != shape:
        message = f"expected {shape}, got {values.shape}"
        raise ValueError(message)
    return np.ascontiguousarray(values)


class RuntimeAdapter:
    """Prefix adapter over a runtime that keeps the live cache on device."""

    def __init__(self, runtime: object) -> None:
        self.runtime = runtime

    def capture(self) -> list[Region]:
        count = int(self.runtime.filled)
        if count < 1:
            return [Region(LOCAL, b""), Region(GLOBAL, b"")]
        local_n = local_nbytes(count)
        global_n = global_nbytes(count)
        scratch = self._scratch(local_n + global_n)
        cursor = _copy_handles_out(
            scratch,
            0,
            self.runtime.local_k,
            count,
            LOCAL_HEADS,
            LOCAL_DIM,
        )
        cursor = _copy_handles_out(
            scratch,
            cursor,
            self.runtime.local_v,
            count,
            LOCAL_HEADS,
            LOCAL_DIM,
        )
        cursor = _copy_handles_out(
            scratch,
            cursor,
            self.runtime.global_k,
            count,
            GLOBAL_HEADS,
            GLOBAL_DIM,
        )
        _copy_handles_out(
            scratch,
            cursor,
            self.runtime.global_v,
            count,
            GLOBAL_HEADS,
            GLOBAL_DIM,
        )
        return [
            Region(LOCAL, memory.load(scratch, 0, local_n)),
            Region(GLOBAL, memory.load(scratch, local_n, global_n)),
        ]

    def install(self, header: PrefixHeader) -> bool:
        if not self.accept(header, list(header.tokens) + [0]):
            return False
        blobs = read_regions(header, (LOCAL, GLOBAL))
        if blobs is None:
            return False
        count = len(header.tokens)
        self.runtime._ensure(count)
        local = blobs[LOCAL]
        packed = self._scratch(len(local) + len(blobs[GLOBAL]))
        memory.store(packed, 0, local)
        memory.store(packed, len(local), blobs[GLOBAL])
        cursor = _copy_handles_in(
            packed,
            0,
            self.runtime.local_k,
            count,
            LOCAL_HEADS,
            LOCAL_DIM,
        )
        cursor = _copy_handles_in(
            packed,
            cursor,
            self.runtime.local_v,
            count,
            LOCAL_HEADS,
            LOCAL_DIM,
        )
        cursor = _copy_handles_in(
            packed,
            cursor,
            self.runtime.global_k,
            count,
            GLOBAL_HEADS,
            GLOBAL_DIM,
        )
        _copy_handles_in(
            packed,
            cursor,
            self.runtime.global_v,
            count,
            GLOBAL_HEADS,
            GLOBAL_DIM,
        )
        self.runtime.filled = count
        return True

    def reset(self) -> None:
        self.runtime.reset()

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        return _accept(header, prompt)

    def _scratch(self, nbytes: int) -> object:
        runtime = self.runtime
        held = int(getattr(runtime, "_prefix_scratch_n", 0))
        current = getattr(runtime, "_prefix_scratch", None)
        if current is not None and held >= nbytes:
            return current
        if current is not None:
            memory.release(current)
        runtime._prefix_scratch = device_buffer(runtime._owner, nbytes)
        runtime._prefix_scratch_n = nbytes
        return runtime._prefix_scratch


def encode_pack_kv(
    session: object,
    source: object,
    dest: object,
    dest_at: int,
    count: int,
    heads: int,
    dim: int,
    *,
    transposed: bool,
) -> int:
    """Dispatch ``pack_kv_fp8``. Returns the packed byte count.

    Kept for the shader check. The live prefix path stores Metal half.
    """
    rows, k_pad, nbytes = fp8_layout(count, heads, dim, transposed=transposed)
    blocks = heads * rows * (k_pad // 32)
    if blocks < 1:
        return 0
    session.encode(
        "pack_kv_fp8",
        [source, dest],
        [
            session._u32(count),
            session._u32(heads),
            session._u32(dim),
            session._u32(k_pad),
            session._u32(rows),
            session._u32(1 if transposed else 0),
        ],
        ((blocks + 255) // 256, 1, 1),
        (256, 1, 1),
        origins=[0, dest_at],
    )
    return nbytes


def encode_unpack_kv(
    session: object,
    packed: object,
    cache: object,
    packed_at: int,
    count: int,
    heads: int,
    dim: int,
    *,
    transposed: bool,
) -> int:
    """Dispatch ``unpack_kv_fp8``. Returns the packed byte count consumed.

    Kept for the shader check. The live prefix path stores Metal half.
    """
    rows, k_pad, nbytes = fp8_layout(count, heads, dim, transposed=transposed)
    live = count * heads * dim
    if live < 1 or nbytes < 1:
        return 0
    session.encode(
        "unpack_kv_fp8",
        [packed, cache],
        [
            session._u32(count),
            session._u32(heads),
            session._u32(dim),
            session._u32(k_pad),
            session._u32(rows),
            session._u32(1 if transposed else 0),
        ],
        ((live + 255) // 256, 1, 1),
        (256, 1, 1),
        origins=[packed_at, 0],
    )
    return nbytes


def fp8_layout(count: int, heads: int, dim: int, *, transposed: bool) -> tuple[int, int, int]:
    """Matrix rows, padded K, and the packed bytes of one fp8 cache plane."""
    rows = dim if transposed else count
    active = count if transposed else dim
    k_pad = (active + 127) & ~127
    one = rows * k_pad + rows * (k_pad // 32)
    return rows, k_pad, heads * one


def _copy_handles_out(
    dest: object,
    offset: int,
    handles: list[object],
    count: int,
    heads: int,
    dim: int,
) -> int:
    nbytes = half_plane_nbytes(count, heads, dim)
    for handle in handles:
        if handle is None:
            continue
        if nbytes:
            memory.copy(handle, 0, dest, offset, nbytes)
        offset += nbytes
    return offset


def _copy_handles_in(
    source: object,
    offset: int,
    handles: list[object],
    count: int,
    heads: int,
    dim: int,
) -> int:
    nbytes = half_plane_nbytes(count, heads, dim)
    for handle in handles:
        if handle is None:
            continue
        if nbytes:
            memory.copy(source, offset, handle, 0, nbytes)
        offset += nbytes
    return offset


def _unpack_plane(
    blob: bytes, layers: int, heads: int, count: int, dim: int
) -> object:
    import numpy as np

    one = half_plane_nbytes(count, heads, dim)
    out = np.zeros((layers, heads, count, dim), dtype=np.float16)
    for layer in range(layers):
        piece = np.frombuffer(blob[layer * one : (layer + 1) * one], dtype=np.float16)
        token_major = piece.reshape(count, heads, dim)
        out[layer] = np.transpose(token_major, (1, 0, 2))
    return out.copy()


def unpack_local(blob: bytes, count: int) -> tuple[object, object]:
    layers = len(local_layers())
    one = half_plane_nbytes(count, LOCAL_HEADS, LOCAL_DIM) * layers
    return (
        _unpack_plane(blob[:one], layers, LOCAL_HEADS, count, LOCAL_DIM),
        _unpack_plane(blob[one : one * 2], layers, LOCAL_HEADS, count, LOCAL_DIM),
    )


def unpack_global(blob: bytes, count: int) -> tuple[object, object]:
    layers = len(GLOBAL_LAYERS)
    one = half_plane_nbytes(count, GLOBAL_HEADS, GLOBAL_DIM) * layers
    return (
        _unpack_plane(blob[:one], layers, GLOBAL_HEADS, count, GLOBAL_DIM),
        _unpack_plane(blob[one : one * 2], layers, GLOBAL_HEADS, count, GLOBAL_DIM),
    )


def _accept(header: PrefixHeader, prompt: list[int]) -> bool:
    count = len(header.tokens)
    if count == 0 or count >= len(prompt):
        return False
    if list(prompt[:count]) != list(header.tokens):
        return False
    local = header.region(LOCAL)
    global_region = header.region(GLOBAL)
    if local is None or global_region is None:
        return False
    return local.length == local_nbytes(count) and global_region.length == global_nbytes(count)


class CacheAdapter:
    """Prefix adapter over :class:`HostCache`.

    ``accept`` refuses an exact-length file. The next token is still stepped.
    """

    def __init__(self, cache: HostCache) -> None:
        self.cache = cache

    def capture(self) -> list[Region]:
        count = self.cache.count
        return [
            Region(LOCAL, self.cache.local_k or _empty_local(count)),
            Region(GLOBAL, self.cache.global_k or _empty_global(count)),
        ]

    def install(self, header: PrefixHeader) -> bool:
        if not self.accept(header, list(header.tokens) + [0]):
            return False
        blobs = read_regions(header, (LOCAL, GLOBAL))
        if blobs is None:
            return False
        self.cache.local_k = blobs[LOCAL]
        self.cache.global_k = blobs[GLOBAL]
        self.cache.count = len(header.tokens)
        return True

    def reset(self) -> None:
        self.cache.count = 0
        self.cache.local_k = b""
        self.cache.global_k = b""

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        return _accept(header, prompt)


def _empty_local(count: int) -> bytes:
    import numpy as np

    layers = len(local_layers())
    keys = np.zeros((layers, LOCAL_HEADS, count, LOCAL_DIM), dtype=np.float16)
    values = np.zeros((layers, LOCAL_HEADS, count, LOCAL_DIM), dtype=np.float16)
    return pack_local(keys, values, count)


def _empty_global(count: int) -> bytes:
    import numpy as np

    shape = (len(GLOBAL_LAYERS), GLOBAL_HEADS, count, GLOBAL_DIM)
    keys = np.zeros(shape, dtype=np.float16)
    values = np.zeros(shape, dtype=np.float16)
    return pack_global(keys, values, count)


def roundtrip_head(matrix: object) -> object:
    """Pack and restore one head. Tests use this. The forward does not."""
    import numpy as np

    values = np.asarray(matrix, dtype=np.float32)
    blob = pack_matrix(values, "fp8")
    return restore_matrix(blob, values.shape[0], values.shape[1], "fp8")
