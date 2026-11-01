"""Host pack of the key and value prefix.

Local and global regions each store keys and values. The filled length is
``count`` tokens. fp8 pads the contraction axis. One scale byte covers 32
channels, packed tightly. An exact-length prefix is not a sample: the next
token still has to be stepped.
"""

from __future__ import annotations

from favlo.memory import memory
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


def _keys(plane: object, layers: int, heads: int, count: int, dim: int) -> bytes:
    values = _as_array(plane, (layers, heads, count, dim))
    parts: list[bytes] = []
    for layer in range(layers):
        for head in range(heads):
            parts.append(pack_matrix(values[layer, head], "fp8"))
    return b"".join(parts)


def _values(plane: object, layers: int, heads: int, count: int, dim: int) -> bytes:
    values = _as_array(plane, (layers, heads, dim, count))
    parts: list[bytes] = []
    for layer in range(layers):
        for head in range(heads):
            parts.append(pack_matrix(values[layer, head], "fp8"))
    return b"".join(parts)


def pack_local(keys: object, values: object, count: int) -> bytes:
    layers = len(local_layers())
    key_bytes = _keys(keys, layers, LOCAL_HEADS, count, LOCAL_DIM)
    value_bytes = _values(values, layers, LOCAL_HEADS, count, LOCAL_DIM)
    return key_bytes + value_bytes


def pack_global(keys: object, values: object, count: int) -> bytes:
    layers = len(GLOBAL_LAYERS)
    return _keys(keys, layers, GLOBAL_HEADS, count, GLOBAL_DIM) + _keys(
        values, layers, GLOBAL_HEADS, count, GLOBAL_DIM
    )


def pack_token_major(raw: object, heads: int, dim: int, count: int, *, transposed: bool) -> bytes:
    """Pack one ``[count, heads, dim]`` float16 plane. One head is staged at a time."""
    if count <= 0 or heads <= 0:
        return b""
    import numpy as np

    plane = np.ndarray((count, heads, dim), dtype=np.float16, buffer=raw)
    parts: list[bytes] = []
    for head in range(heads):
        matrix = plane[:, head, :].astype(np.float32)
        if transposed:
            matrix = np.ascontiguousarray(matrix.T)
        parts.append(pack_matrix(matrix, "fp8"))
    return b"".join(parts)


def local_nbytes(count: int) -> int:
    layers = len(local_layers())
    one_key = len(pack_matrix(_zeros((count, LOCAL_DIM)), "fp8"))
    one_value = len(pack_matrix(_zeros((LOCAL_DIM, count)), "fp8"))
    return layers * LOCAL_HEADS * (one_key + one_value)


def global_nbytes(count: int) -> int:
    one_key = len(pack_matrix(_zeros((count, GLOBAL_DIM)), "fp8"))
    return len(GLOBAL_LAYERS) * GLOBAL_HEADS * one_key * 2


def _zeros(shape: tuple[int, ...]) -> object:
    import numpy as np

    return np.zeros(shape, dtype=np.float32)


def _as_array(plane: object, shape: tuple[int, ...]) -> object:
    import numpy as np

    values = np.asarray(plane, dtype=np.float32)
    if values.shape != shape:
        message = f"expected {shape}, got {values.shape}"
        raise ValueError(message)
    return values


class RuntimeAdapter:
    """Prefix adapter over a runtime that keeps the live cache on device."""

    def __init__(self, runtime: object) -> None:
        self.runtime = runtime

    def capture(self) -> list[Region]:
        count = int(self.runtime.filled)
        return [
            Region(LOCAL, _pack_local(self.runtime, count)),
            Region(GLOBAL, _pack_global(self.runtime, count)),
        ]

    def install(self, header: PrefixHeader) -> bool:
        if not self.accept(header, list(header.tokens) + [0]):
            return False
        blobs = read_regions(header, (LOCAL, GLOBAL))
        if blobs is None:
            return False
        count = len(header.tokens)
        keys, values = unpack_local(blobs[LOCAL], count)
        self.runtime.import_local(keys, values)
        self.runtime.import_global(*unpack_global(blobs[GLOBAL], count))
        self.runtime.filled = count
        return True

    def reset(self) -> None:
        self.runtime.reset()

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool:
        return _accept(header, prompt)


def _pack_local(runtime: object, count: int) -> bytes:
    keys: list[bytes] = []
    values: list[bytes] = []
    width = count * LOCAL_HEADS * LOCAL_DIM * 2
    for layer in local_layers():
        keys.append(
            pack_token_major(
                _plane(getattr(runtime, "local_k")[layer], width),
                LOCAL_HEADS,
                LOCAL_DIM,
                count,
                transposed=False,
            )
        )
        values.append(
            pack_token_major(
                _plane(getattr(runtime, "local_v")[layer], width),
                LOCAL_HEADS,
                LOCAL_DIM,
                count,
                transposed=True,
            )
        )
    return b"".join(keys) + b"".join(values)


def _pack_global(runtime: object, count: int) -> bytes:
    keys: list[bytes] = []
    values: list[bytes] = []
    width = count * GLOBAL_HEADS * GLOBAL_DIM * 2
    for layer in GLOBAL_LAYERS:
        keys.append(
            pack_token_major(
                _plane(getattr(runtime, "global_k")[layer], width),
                GLOBAL_HEADS,
                GLOBAL_DIM,
                count,
                transposed=False,
            )
        )
        values.append(
            pack_token_major(
                _plane(getattr(runtime, "global_v")[layer], width),
                GLOBAL_HEADS,
                GLOBAL_DIM,
                count,
                transposed=False,
            )
        )
    return b"".join(keys) + b"".join(values)


def _plane(handle: object, nbytes: int) -> memoryview:
    if nbytes < 1:
        return memoryview(b"")
    return memory.map(handle, 0, nbytes)


def unpack_local(blob: bytes, count: int) -> tuple[object, object]:
    import numpy as np

    layers = len(local_layers())
    key_n = len(pack_matrix(np.zeros((count, LOCAL_DIM), dtype=np.float32), "fp8"))
    value_n = len(pack_matrix(np.zeros((LOCAL_DIM, count), dtype=np.float32), "fp8"))
    keys = np.zeros((layers, LOCAL_HEADS, count, LOCAL_DIM), dtype=np.float32)
    values = np.zeros((layers, LOCAL_HEADS, count, LOCAL_DIM), dtype=np.float32)
    offset = 0
    for layer in range(layers):
        for head in range(LOCAL_HEADS):
            piece = blob[offset : offset + key_n]
            keys[layer, head] = restore_matrix(piece, count, LOCAL_DIM, "fp8")
            offset += key_n
    for layer in range(layers):
        for head in range(LOCAL_HEADS):
            restored = restore_matrix(blob[offset : offset + value_n], LOCAL_DIM, count, "fp8")
            values[layer, head] = restored.T
            offset += value_n
    return keys, values


def unpack_global(blob: bytes, count: int) -> tuple[object, object]:
    import numpy as np

    layers = len(GLOBAL_LAYERS)
    key_n = len(pack_matrix(np.zeros((count, GLOBAL_DIM), dtype=np.float32), "fp8"))
    keys = np.zeros((layers, GLOBAL_HEADS, count, GLOBAL_DIM), dtype=np.float32)
    values = np.zeros((layers, GLOBAL_HEADS, count, GLOBAL_DIM), dtype=np.float32)
    offset = 0
    for plane in (keys, values):
        for layer in range(layers):
            for head in range(GLOBAL_HEADS):
                piece = blob[offset : offset + key_n]
                plane[layer, head] = restore_matrix(piece, count, GLOBAL_DIM, "fp8")
                offset += key_n
    return keys, values


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
    keys = np.zeros((layers, LOCAL_HEADS, count, LOCAL_DIM), dtype=np.float32)
    values = np.zeros((layers, LOCAL_HEADS, LOCAL_DIM, count), dtype=np.float32)
    return pack_local(keys, values, count)


def _empty_global(count: int) -> bytes:
    import numpy as np

    shape = (len(GLOBAL_LAYERS), GLOBAL_HEADS, count, GLOBAL_DIM)
    keys = np.zeros(shape, dtype=np.float32)
    values = np.zeros(shape, dtype=np.float32)
    return pack_global(keys, values, count)


def roundtrip_head(matrix: object) -> object:
    """Pack and restore one head. Tests use this. The forward does not."""
    import numpy as np

    values = np.asarray(matrix, dtype=np.float32)
    blob = pack_matrix(values, "fp8")
    return restore_matrix(blob, values.shape[0], values.shape[1], "fp8")
