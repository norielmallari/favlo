"""One MiniCPM-4 tower. Prefill and a later step share the key/value cache."""

from __future__ import annotations

import numpy as np

from voxcpm2.block import residual_add, rope_angles, swiglu
from voxcpm2.metal import Session
from voxcpm2.store import Store


class Cache:
    """Key and value rows already rotated, one array per layer."""

    def __init__(self, layers: int, kv_heads: int, dim: int) -> None:
        self.k = [np.zeros((0, kv_heads, dim), dtype=np.float32) for _ in range(layers)]
        self.v = [np.zeros((0, kv_heads, dim), dtype=np.float32) for _ in range(layers)]


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
        if int(k_shape[0]) != kv_heads * head_dim or int(q_shape[0]) != heads * head_dim:
            message = f"{prefix} head width does not match the config"
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
        self.eps = eps
        self.rope = rope
        self.theta = theta
        self.factors = factors
        self.scale = scale

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
        cos = sin = None
        if self.rope:
            cos, sin = _angles(positions, self.head_dim, self.theta, self.factors, self.scale)
        for layer in range(self.layers):
            data = self._layer(layer, data, causal, cache, cos, sin)
        data = self._norm(data, self.norm)
        if squeeze:
            return data.reshape(-1)
        return data

    def _layer(
        self,
        layer: int,
        data: np.ndarray,
        causal: bool,
        cache: Cache,
        cos: np.ndarray | None,
        sin: np.ndarray | None,
    ) -> np.ndarray:
        seq = int(data.shape[0])
        stem = f"{self.prefix}.{layer}"
        normed = self._norm(data, f"{stem}.input_layernorm.weight")
        query = self.store.linear(self.session, f"{stem}.self_attn.q_proj.weight", normed)
        key = self.store.linear(self.session, f"{stem}.self_attn.k_proj.weight", normed)
        value = self.store.linear(self.session, f"{stem}.self_attn.v_proj.weight", normed)
        query = query.reshape(seq, self.heads, self.head_dim)
        key = key.reshape(seq, self.kv_heads, self.head_dim)
        value = value.reshape(seq, self.kv_heads, self.head_dim)
        if cos is not None and sin is not None:
            query = self.session.rope(query, cos, sin)
            key = self.session.rope(key, cos, sin)
        cache.k[layer] = np.concatenate((cache.k[layer], key), axis=0)
        cache.v[layer] = np.concatenate((cache.v[layer], value), axis=0)
        mixed = self.session.attention(query, cache.k[layer], cache.v[layer], causal=causal)
        mixed = mixed.reshape(seq, self.heads * self.head_dim)
        update = self.store.linear(self.session, f"{stem}.self_attn.o_proj.weight", mixed)
        data = residual_add(data, update, use_mup=False, scale_depth=1.0, layers=self.layers)
        normed = self._norm(data, f"{stem}.post_attention_layernorm.weight")
        gate = self.store.linear(self.session, f"{stem}.mlp.gate_proj.weight", normed)
        up = self.store.linear(self.session, f"{stem}.mlp.up_proj.weight", normed)
        down = self.store.linear(self.session, f"{stem}.mlp.down_proj.weight", swiglu(gate, up))
        return residual_add(data, down, use_mup=False, scale_depth=1.0, layers=self.layers)

    def _norm(self, hidden: np.ndarray, name: str) -> np.ndarray:
        weight = self.store.vector(name)
        return self.session.rmsnorm(hidden, weight, self.eps).astype(np.float32)


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
