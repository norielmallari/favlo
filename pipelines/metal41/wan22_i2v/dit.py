"""One Euler evaluation of a packed expert."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from wan22_i2v.formats import TILE_M, LinearPack, pad_to, slim_pack, stored_k
from wan22_i2v.metal import Gpu
from wan22_i2v.schedule import rope_axes


@dataclass
class Expert:
    """Host arrays for one noise expert."""

    packs: dict[str, LinearPack]
    halves: dict[str, np.ndarray]


@dataclass
class Resident:
    """Device copy of one expert. Drop it before loading the other expert."""

    packs: dict[str, LinearPack]
    weights: dict[str, dict[str, Any]]
    halves: dict[str, Any]
    host: dict[str, np.ndarray]
    owned: list[Any]


def upload_expert(
    gpu: Gpu, expert: Expert, on_file: Callable[[], None] | None = None
) -> Resident:
    weights: dict[str, dict[str, Any]] = {}
    halves: dict[str, Any] = {}
    owned: list[Any] = []
    for name, pack in expert.packs.items():
        buffers = {
            "packed": gpu.keep(gpu.upload(pack.packed)),
            "scales": gpu.keep(gpu.upload(pack.scales)),
            "bias": gpu.keep(gpu.upload(pack.bias)),
            "perm": gpu.keep(gpu.upload(pack.perm)),
        }
        weights[name] = buffers
        owned.extend(buffers.values())
        if on_file is not None:
            on_file()
    for name, array in expert.halves.items():
        buf = gpu.keep(gpu.upload(np.ascontiguousarray(array, dtype=np.float16)))
        halves[name] = buf
        owned.append(buf)
        if on_file is not None:
            on_file()
    packs = {name: slim_pack(pack) for name, pack in expert.packs.items()}
    return Resident(
        packs=packs,
        weights=weights,
        halves=halves,
        host={},
        owned=owned,
    )


def drop_expert(gpu: Gpu, resident: Resident) -> None:
    for buf in resident.owned:
        gpu.unkeep(buf)
    resident.owned.clear()
    resident.weights.clear()
    resident.halves.clear()
    resident.host.clear()
    resident.packs.clear()
    gpu.reclaim()


_FORWARD_BUFFERS = (
    "half",
    "x",
    "nrm",
    "q",
    "k",
    "v",
    "attn",
    "proj",
    "hidden",
    "wide",
    "head_out",
    "text",
    "embed",
    "time_src",
    "volume",
    "shift",
    "scale",
    "gate",
    "ones",
    "time_embed",
    "mixed",
    "vel",
)


def forward_bytes(config: dict[str, Any], tokens: int, volume_size: int) -> int:
    """Bytes ``Forward`` allocates for this token count and latent volume."""
    dim = int(config["dim"])
    ffn = int(config["ffn"])
    rows = pad_to(tokens, TILE_M)
    text_rows = pad_to(int(config["text_len"]), TILE_M)
    freq = int(config["freq_dim"])
    text_dim = int(config["text_dim"])
    span = stored_k(max(dim, ffn, text_dim, freq))
    return (
        rows * span * 2
        + 7 * rows * dim * 4
        + rows * ffn * 4
        + 64 * dim * 6 * 4
        + rows * 64 * 4
        + text_rows * dim * 4
        + text_rows * text_dim * 4
        + 64 * freq * 4
        + volume_size * 4
        + 5 * dim * 4
        + dim * 6 * 4
        + (volume_size // 36) * 16 * 4
    )


class Forward:
    """Device scratch for one geometry. The latent grid stays on the device."""

    def __init__(self, gpu: Gpu, config: dict[str, Any], tokens: int, volume_size: int) -> None:
        self.gpu = gpu
        self.config = config
        dim = int(config["dim"])
        ffn = int(config["ffn"])
        text_len = int(config["text_len"])
        self.tokens = tokens
        self.rows = pad_to(tokens, TILE_M)
        self.text_rows = pad_to(text_len, TILE_M)
        self.dim = dim
        self.ffn = ffn
        self.heads = int(config["heads"])
        self.head_dim = dim // self.heads
        self.eps = float(config["eps"])
        self.freq = int(config["freq_dim"])
        self.text_dim = int(config["text_dim"])
        self.text_len = text_len
        span = stored_k(max(dim, ffn, self.text_dim, self.freq))
        self.half = gpu.keep(gpu.empty(self.rows * span * 2))
        self.x = gpu.keep(gpu.empty(self.rows * dim * 4))
        self.nrm = gpu.keep(gpu.empty(self.rows * dim * 4))
        self.q = gpu.keep(gpu.empty(self.rows * dim * 4))
        self.k = gpu.keep(gpu.empty(self.rows * dim * 4))
        self.v = gpu.keep(gpu.empty(self.rows * dim * 4))
        self.attn = gpu.keep(gpu.empty(self.rows * dim * 4))
        self.proj = gpu.keep(gpu.empty(self.rows * dim * 4))
        self.hidden = gpu.keep(gpu.empty(self.rows * ffn * 4))
        self.wide = gpu.keep(gpu.empty(64 * dim * 6 * 4))
        self.head_out = gpu.keep(gpu.empty(self.rows * 64 * 4))
        self.text = gpu.keep(gpu.empty(self.text_rows * dim * 4))
        self.embed = gpu.keep(gpu.empty(self.text_rows * self.text_dim * 4))
        self.time_src = gpu.keep(gpu.empty(64 * self.freq * 4))
        self.volume = gpu.keep(gpu.empty(volume_size * 4))
        self.shift = gpu.keep(gpu.empty(dim * 4))
        self.scale = gpu.keep(gpu.empty(dim * 4))
        self.gate = gpu.keep(gpu.empty(dim * 4))
        self.ones = gpu.keep(gpu.empty(dim * 4))
        self.time_embed = gpu.keep(gpu.empty(dim * 4))
        self.mixed = gpu.keep(gpu.empty(dim * 6 * 4))
        self.vel = gpu.keep(gpu.empty((volume_size // 36) * 16 * 4))
        gpu.store_f32(self.ones, np.ones(dim, dtype=np.float32))
        self.axes = rope_axes(self.head_dim)

    def prepare(
        self, noise: Any, mask: Any, latent: Any, time: int, height: int, width: int
    ) -> None:
        """Stack noise, the mask, and the VAE latent into the 36-channel volume."""
        self.gpu.condition(noise, mask, latent, self.volume, time, height, width)

    def release(self) -> None:
        """Drop the Python references to the scratch buffers."""
        for name in _FORWARD_BUFFERS:
            setattr(self, name, None)

    def detach(self, gpu: Gpu) -> None:
        """Release the scratch buffers back to the device."""
        buffers = [getattr(self, name) for name in _FORWARD_BUFFERS]
        self.release()
        gpu.drop_many([buf for buf in buffers if buf is not None])

    def velocity(
        self,
        resident: Resident,
        embed: np.ndarray,
        timestep: float,
        grid: tuple[int, int, int],
        on_layer: Callable[[], None] | None = None,
    ) -> None:
        """Predict the flow velocity into ``vel``. ``prepare`` filled the volume."""
        patch_grid = (grid[0], grid[1] // 2, grid[2] // 2)
        self._text(resident, embed)
        self._time(resident, timestep)
        self._patch(resident, grid)
        self.gpu.commit()
        for layer in range(int(self.config["layers"])):
            self._block(resident, layer, patch_grid)
            self.gpu.commit()
            if on_layer is not None:
                on_layer()
        self._head(resident)
        self.gpu.unpatchify(self.head_out, self.vel, patch_grid[0], patch_grid[1], patch_grid[2])
        self.gpu.end()

    def _linear(
        self, src: Any, src_cols: int, rows: int, name: str, dest: Any, resident: Resident
    ) -> None:
        pack = resident.packs[name]
        self.gpu.project(src, src_cols, rows, pack, dest, resident.weights[name], self.half)

    def _text(self, resident: Resident, table: np.ndarray) -> None:
        self.gpu.store_f32(self.embed, table)
        self._linear(
            self.embed,
            self.text_dim,
            self.text_rows,
            "text_embedding.0.weight",
            self.proj,
            resident,
        )
        self.gpu.pointwise("gelu_tanh", self.proj, self.text_rows, self.dim, self.dim)
        self._linear(
            self.proj, self.dim, self.text_rows, "text_embedding.2.weight", self.text, resident
        )

    def _time(self, resident: Resident, timestep: float) -> None:
        self.gpu.zero(self.time_src, 64 * self.freq * 4)
        self.gpu.sinusoid(self.time_src, self.freq, float(timestep))
        self._linear(self.time_src, self.freq, 64, "time_embedding.0.weight", self.nrm, resident)
        self.gpu.pointwise("silu", self.nrm, 64, self.dim, self.dim)
        self._linear(self.nrm, self.dim, 64, "time_embedding.2.weight", self.proj, resident)
        self.gpu.copy_f32(self.proj, 0, self.time_embed, self.dim)
        self.gpu.pointwise("silu", self.proj, 64, self.dim, self.dim)
        self._linear(self.proj, self.dim, 64, "time_projection.1.weight", self.wide, resident)

    def _patch(self, resident: Resident, grid: tuple[int, int, int]) -> None:
        self.gpu.zero(self.x, self.rows * self.dim * 4)
        time, height, width = grid
        self.gpu.patch(
            self.volume,
            resident.halves["patch_embedding.weight"],
            resident.halves["patch_embedding.bias"],
            self.x,
            36,
            time,
            height,
            width,
            self.dim,
        )

    def _block(self, resident: Resident, layer: int, grid: tuple[int, int, int]) -> None:
        prefix = f"blocks.{layer}"
        self.gpu.add_half(
            self.wide, resident.halves[f"{prefix}.modulation"], self.mixed, 6 * self.dim
        )
        self._self_attn(resident, prefix, grid)
        self.gpu.layernorm_affine(
            self.x,
            resident.halves[f"{prefix}.norm3.weight"],
            resident.halves[f"{prefix}.norm3.bias"],
            self.nrm,
            self.rows,
            self.dim,
            self.eps,
        )
        self._cross(resident, prefix)
        self.gpu.layernorm(self.x, self.nrm, self.rows, self.dim, self.eps)
        self._bind(3, 4)
        self.gpu.modulate(self.nrm, self.shift, self.scale, self.rows, self.dim, self.dim)
        self._linear(self.nrm, self.dim, self.rows, f"{prefix}.ffn.0.weight", self.hidden, resident)
        self.gpu.pointwise("gelu_tanh", self.hidden, self.rows, self.ffn, self.ffn)
        self._linear(
            self.hidden, self.ffn, self.rows, f"{prefix}.ffn.2.weight", self.proj, resident
        )
        self._bind_gate(5)
        self.gpu.gate(self.x, self.proj, self.gate, self.rows, self.dim, self.dim)

    def _self_attn(self, resident: Resident, prefix: str, grid: tuple[int, int, int]) -> None:
        self.gpu.layernorm(self.x, self.nrm, self.rows, self.dim, self.eps)
        self._bind(0, 1)
        self.gpu.modulate(self.nrm, self.shift, self.scale, self.rows, self.dim, self.dim)
        stem = f"{prefix}.self_attn"
        self._linear(self.nrm, self.dim, self.rows, f"{stem}.q.weight", self.q, resident)
        self._linear(self.nrm, self.dim, self.rows, f"{stem}.k.weight", self.k, resident)
        self._linear(self.nrm, self.dim, self.rows, f"{stem}.v.weight", self.v, resident)
        self._rms(self.q, resident.halves[f"{stem}.norm_q.weight"], self.rows)
        self._rms(self.k, resident.halves[f"{stem}.norm_k.weight"], self.rows)
        self.gpu.rope(self.q, self.tokens, self.heads, self.head_dim, grid, self.axes)
        self.gpu.rope(self.k, self.tokens, self.heads, self.head_dim, grid, self.axes)
        self.gpu.attend(
            self.q, self.k, self.v, self.attn, self.tokens, self.tokens, self.heads, self.head_dim
        )
        self._linear(self.attn, self.dim, self.rows, f"{stem}.o.weight", self.proj, resident)
        self._bind_gate(2)
        self.gpu.gate(self.x, self.proj, self.gate, self.rows, self.dim, self.dim)

    def _cross(self, resident: Resident, prefix: str) -> None:
        stem = f"{prefix}.cross_attn"
        self._linear(self.nrm, self.dim, self.rows, f"{stem}.q.weight", self.q, resident)
        self._rms(self.q, resident.halves[f"{stem}.norm_q.weight"], self.rows)
        self._linear(self.text, self.dim, self.text_rows, f"{stem}.k.weight", self.k, resident)
        self._linear(self.text, self.dim, self.text_rows, f"{stem}.v.weight", self.v, resident)
        self._rms(self.k, resident.halves[f"{stem}.norm_k.weight"], self.text_rows)
        self.gpu.attend(
            self.q, self.k, self.v, self.attn, self.tokens, self.text_len, self.heads, self.head_dim
        )
        self._linear(self.attn, self.dim, self.rows, f"{stem}.o.weight", self.proj, resident)
        self.gpu.gate(self.x, self.proj, self.ones, self.rows, self.dim, self.dim)

    def _head(self, resident: Resident) -> None:
        self.gpu.add_embed_half(
            resident.halves["head.modulation"], self.time_embed, self.mixed, 2, self.dim
        )
        self.gpu.layernorm(self.x, self.nrm, self.rows, self.dim, self.eps)
        self._bind(0, 1)
        self.gpu.modulate(self.nrm, self.shift, self.scale, self.rows, self.dim, self.dim)
        self._linear(self.nrm, self.dim, self.rows, "head.head.weight", self.head_out, resident)

    def _rms(self, values: Any, weight: Any, rows: int) -> None:
        self.gpu.rmsnorm(values, weight, values, rows, self.dim, self.eps)

    def _bind(self, shift_at: int, scale_at: int) -> None:
        self.gpu.copy_f32(self.mixed, shift_at * self.dim, self.shift, self.dim)
        self.gpu.copy_f32(self.mixed, scale_at * self.dim, self.scale, self.dim)

    def _bind_gate(self, index: int) -> None:
        self.gpu.copy_f32(self.mixed, index * self.dim, self.gate, self.dim)
