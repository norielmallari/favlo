"""One MiniCPM-4 tower. Prefill and a later step share the key/value cache.

A forward encodes every layer and the final norm into one command buffer.
Values between those kernels stay float32. The matmul casts its activation
to half. The cache is a float32 device buffer.
"""

from __future__ import annotations

import numpy as np

from favlo.memory import memory
from voxcpm2.metal41.block import rope_angles
from voxcpm2.metal41.metal import Plane, Session
from voxcpm2.metal41.store import Store


class Cache:
    """Key and value rows already rotated, one float32 buffer per layer."""

    def __init__(self, layers: int, kv_heads: int, dim: int, capacity: int | None = None) -> None:
        self.layers = layers
        self.kv_heads = kv_heads
        self.dim = dim
        self.capacity = capacity
        self.length = 0
        self.k: list[Plane] | None = None
        self.v: list[Plane] | None = None

    def ensure(self, session: Session, seq: int) -> None:
        """Allocate the prefix once. Later steps append inside that allocation."""
        if seq < 1:
            message = "a cache step needs a row"
            raise ValueError(message)
        if self.k is None or self.v is None:
            cap = seq if self.capacity is None else int(self.capacity)
            if seq > cap:
                message = f"the cache holds {cap} rows and this step needs {seq}"
                raise RuntimeError(message)
            self.capacity = cap
            cols = self.kv_heads * self.dim
            self.k = [session.plane(cap, cols) for _ in range(self.layers)]
            self.v = [session.plane(cap, cols) for _ in range(self.layers)]
            return
        if self.length + seq > int(self.capacity):
            message = (
                f"the cache holds {self.capacity} rows and this step needs {self.length + seq}"
            )
            raise RuntimeError(message)

    def release(self) -> None:
        planes: list[Plane] = []
        if self.k is not None:
            planes.extend(self.k)
        if self.v is not None:
            planes.extend(self.v)
        self.k = None
        self.v = None
        self.length = 0
        for plane in planes:
            memory.release(plane.handle)


class _Room:
    """Planes one layer reuses. They stay live until the command buffer finishes."""

    def __init__(self, session: Session, seq: int, stack: Stack) -> None:
        self.normed = session.plane(seq, stack.hidden)
        self.query = session.plane(seq, stack.heads * stack.head_dim)
        self.key = session.plane(seq, stack.kv_heads * stack.head_dim)
        self.value = session.plane(seq, stack.kv_heads * stack.head_dim)
        self.mixed = session.plane(seq, stack.heads * stack.head_dim)
        self.gate = session.plane(seq, stack.ffn)
        self.up = session.plane(seq, stack.ffn)
        self.update = session.plane(seq, stack.hidden)

    def planes(self) -> list[Plane]:
        return [
            self.normed,
            self.query,
            self.key,
            self.value,
            self.mixed,
            self.gate,
            self.up,
            self.update,
        ]


class Stack:
    """Pre-norm attention and SwiGLU. ``use_mup`` is false, so the residual is a plain add."""

    def __init__(
        self,
        session: Session,
        store: Store,
        prefix: str,
        norm: str,
        *,
        layers: int,
        hidden: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        eps: float,
        rope: bool,
        theta: float,
        factors: tuple[float, ...],
        scale: float,
    ) -> None:
        k_shape = store.header[f"{prefix}.0.self_attn.k_proj.weight"]["shape"]
        q_shape = store.header[f"{prefix}.0.self_attn.q_proj.weight"]["shape"]
        gate = store.header[f"{prefix}.0.mlp.gate_proj.weight"]["shape"]
        if int(k_shape[0]) != kv_heads * head_dim or int(q_shape[0]) != heads * head_dim:
            message = f"{prefix} head width does not match the config"
            raise ValueError(message)
        if int(gate[1]) != hidden:
            message = f"{prefix} feed-forward width does not match the config"
            raise ValueError(message)
        self.session = session
        self.store = store
        self.prefix = prefix
        self.norm = norm
        self.layers = layers
        self.hidden = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.ffn = int(gate[0])
        self.eps = eps
        self.rope = rope
        self.theta = theta
        self.factors = factors
        self.scale = scale

    def prepare(self, seq: int, kv_len: int) -> None:
        """Size the scratch before a command buffer opens."""
        span = max(self.hidden, self.ffn, self.heads * self.head_dim)
        self.session.prepare_linear(seq, span, span, seq * self.heads * kv_len)

    def forward(
        self,
        hidden: np.ndarray,
        positions: np.ndarray,
        causal: bool,
        cache: Cache,
    ) -> np.ndarray:
        """Run every layer and the final norm. A vector input comes back as a vector."""
        data = np.asarray(hidden, dtype=np.float32)
        squeeze = data.ndim == 1
        if squeeze:
            data = data.reshape(1, -1)
        seq = int(data.shape[0])
        if self.rope and len(positions) != seq:
            message = f"rope positions are {len(positions)}, the sequence is {seq}"
            raise ValueError(message)
        src = self.session.load(data)
        owned = [src]
        handles: list[object] = []
        try:
            out = self.encode(src, positions, causal, cache, owned, handles)
            result = self.session.read(out)
            cache.length += seq
        finally:
            self.session.release_all(owned, handles)
        if squeeze:
            return result.reshape(-1)
        return result

    def encode(
        self,
        data: Plane,
        positions: np.ndarray,
        causal: bool,
        cache: Cache,
        owned: list[Plane],
        handles: list[object],
    ) -> Plane:
        """Encode the tower onto the open command buffer. ``data`` is the residual stream."""
        seq = data.rows
        if data.cols != self.hidden:
            message = f"{self.prefix} width is {data.cols}, the stack is {self.hidden}"
            raise ValueError(message)
        if self.rope and len(positions) != seq:
            message = f"rope positions are {len(positions)}, the sequence is {seq}"
            raise ValueError(message)
        self.prepare(seq, cache.length + seq)
        cache.ensure(self.session, seq)
        if cache.k is None or cache.v is None:
            message = "the cache has no rows"
            raise RuntimeError(message)
        room = _Room(self.session, seq, self)
        owned.extend(room.planes())
        cos = sin = None
        if self.rope:
            cos_tab, sin_tab = _angles(
                positions, self.head_dim, self.theta, self.factors, self.scale
            )
            cos = self.session.upload(np.ascontiguousarray(cos_tab, dtype=np.float32).tobytes())
            handles.append(cos)
            sin = self.session.upload(np.ascontiguousarray(sin_tab, dtype=np.float32).tobytes())
            handles.append(sin)
        base = cache.length
        for layer in range(self.layers):
            self._layer(layer, data, room, cache, base, causal, cos, sin)
        self._norm(data, room.normed, self.norm)
        return room.normed

    def _layer(
        self,
        layer: int,
        data: Plane,
        room: _Room,
        cache: Cache,
        base: int,
        causal: bool,
        cos: object | None,
        sin: object | None,
    ) -> None:
        seq = data.rows
        stem = f"{self.prefix}.{layer}"
        keys = cache.k
        values = cache.v
        if keys is None or values is None:
            message = "the cache has no rows"
            raise RuntimeError(message)
        self._norm(data, room.normed, f"{stem}.input_layernorm.weight")
        self._project(room.normed, f"{stem}.self_attn.q_proj.weight", room.query)
        self._project(room.normed, f"{stem}.self_attn.k_proj.weight", room.key)
        self._project(room.normed, f"{stem}.self_attn.v_proj.weight", room.value)
        if cos is not None and sin is not None:
            self.session.rotate(room.query, cos, sin, self.heads, self.head_dim)
            self.session.rotate(room.key, cos, sin, self.kv_heads, self.head_dim)
        self.session.copy(room.key, keys[layer], seq, 0, base)
        self.session.copy(room.value, values[layer], seq, 0, base)
        self.session.attend(
            room.query,
            keys[layer],
            values[layer],
            room.mixed,
            kv_len=base + seq,
            heads=self.heads,
            kv_heads=self.kv_heads,
            dim=self.head_dim,
            causal=causal,
        )
        self._project(room.mixed, f"{stem}.self_attn.o_proj.weight", room.update)
        self.session.add(data, room.update, data)
        self._norm(data, room.normed, f"{stem}.post_attention_layernorm.weight")
        self._project(room.normed, f"{stem}.mlp.gate_proj.weight", room.gate)
        self._project(room.normed, f"{stem}.mlp.up_proj.weight", room.up)
        self.session.swiglu(room.gate, room.up, room.gate)
        self._project(room.gate, f"{stem}.mlp.down_proj.weight", room.update)
        self.session.add(data, room.update, data)

    def _project(self, src: Plane, name: str, dst: Plane) -> None:
        self.session.project(src, self.store.weight(self.session, name), None, dst)

    def _norm(self, src: Plane, dst: Plane, name: str) -> None:
        self.session.norm(src, self.store.gain(self.session, name), dst, self.eps)


def _angles(
    positions: np.ndarray,
    dim: int,
    theta: float,
    factors: tuple[float, ...],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos = np.empty((len(positions), dim), dtype=np.float32)
    sin = np.empty((len(positions), dim), dtype=np.float32)
    for index, position in enumerate(positions):
        cosine, sine = rope_angles(
            int(position),
            dim=dim,
            base=theta,
            factors=factors,
            scale=scale,
        )
        cos[index] = np.asarray(cosine, dtype=np.float32)
        sin[index] = np.asarray(sine, dtype=np.float32)
    return cos, sin
