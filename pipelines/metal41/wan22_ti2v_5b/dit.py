"""Wan2.2 TI2V-5B diffusion transformer.

Self-attention, cross-attention, and the feed-forward are packed. Time,
text, patch, and the head stay float16. One forward is one knot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wan22_ti2v_5b.activations import (
    apply_rope,
    gelu_erf,
    gelu_tanh,
    layernorm,
    merge_heads,
    qk_norm,
    silu,
    sinusoidal,
    split_heads,
)
from wan22_ti2v_5b.gpu import Gpu
from wan22_ti2v_5b.spec import HEADS, IN_DIM
from wan22_ti2v_5b.weights import apply, attend, load_matrix


class DiT:
    def __init__(self, root: Path, gpu: Gpu) -> None:
        self.root = root
        self.gpu = gpu
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.gelu = gelu_erf if manifest.get("gelu") == "erf" else gelu_tanh
        self.layers = int(manifest["layers"])
        self._cached_text: np.ndarray | None = None
        self.patch = load_matrix(root / "patch.npy")
        self.text0 = load_matrix(root / "text0.npy")
        self.text2 = load_matrix(root / "text2.npy")
        self.time0 = load_matrix(root / "time0.npy")
        self.time2 = load_matrix(root / "time2.npy")
        self.time_proj = load_matrix(root / "time_proj.npy")
        self.head = load_matrix(root / "head.npy")
        self.head_mod = np.load(root / "head_mod.npy").astype(np.float32)
        self._blocks = [
            _Block.load(root / "blocks" / f"{index:02d}") for index in range(self.layers)
        ]

    def velocity(
        self,
        latent: np.ndarray,
        token_time: np.ndarray,
        text: np.ndarray,
        text_mask: np.ndarray,
        grid: tuple[int, int, int],
    ) -> np.ndarray:
        """``latent`` is ``[48, T, H, W]``. ``token_time`` matches the token grid."""
        time, height, width = grid
        tokens = self._patch(latent)
        if self._cached_text is not text:
            context = self._embed_text(text, text_mask)
            self._cached_text = text
            for block in self._blocks:
                projected = apply(self.gpu, context, block.cross_k)
                block.cross_key = qk_norm(projected, block.cross_norm_k)
                block.cross_value = apply(self.gpu, context, block.cross_v)
        temb, proj = self._times(token_time.reshape(-1))
        for index in range(self.layers):
            tokens = self._block(tokens, temb, proj, text, text_mask, index, height, width)
        out = self._head(tokens, temb)
        return _unpatch(out, time, height, width)

    def _patch(self, latent: np.ndarray) -> np.ndarray:
        channels, frames, height, width = latent.shape
        folded = latent.reshape(channels, frames, height // 2, 2, width // 2, 2)
        folded = np.transpose(folded, (1, 2, 4, 0, 3, 5))
        rows = folded.reshape(-1, IN_DIM * 4).astype(np.float32)
        return apply(self.gpu, rows, self.patch)

    def _embed_text(self, text: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        hidden = self.gelu(apply(self.gpu, text, self.text0))
        return apply(self.gpu, hidden, self.text2)

    def _times(self, token_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.unique(token_time.astype(np.float32))
        temb_of: dict[float, np.ndarray] = {}
        proj_of: dict[float, np.ndarray] = {}
        for value in values:
            temb = self._time_embed(float(value))
            temb_of[float(value)] = temb
            proj_of[float(value)] = self._time_proj(temb)
        temb_rows = np.stack([temb_of[float(value)] for value in token_time])
        proj_rows = np.stack([proj_of[float(value)] for value in token_time])
        return temb_rows, proj_rows

    def _time_embed(self, value: float) -> np.ndarray:
        wave = sinusoidal(value)[None, :]
        hidden = silu(apply(self.gpu, wave, self.time0))
        return apply(self.gpu, hidden, self.time2)[0]

    def _time_proj(self, temb: np.ndarray) -> np.ndarray:
        return apply(self.gpu, silu(temb)[None, :], self.time_proj)[0]

    def _block(
        self,
        tokens: np.ndarray,
        temb: np.ndarray,
        proj: np.ndarray,
        text: np.ndarray,
        text_mask: np.ndarray,
        index: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        del text
        del text_mask
        block = self._blocks[index]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = np.split(
            proj + block.modulation, 6, axis=1
        )
        normed = layernorm(tokens)
        normed = normed * (1.0 + scale_msa) + shift_msa
        query = qk_norm(apply(self.gpu, normed, block.self_q), block.self_norm_q)
        key = qk_norm(apply(self.gpu, normed, block.self_k), block.self_norm_k)
        value = apply(self.gpu, normed, block.self_v)
        query = apply_rope(query, height, width)
        key = apply_rope(key, height, width)
        mixed = attend(
            self.gpu,
            split_heads(query, HEADS),
            split_heads(key, HEADS),
            split_heads(value, HEADS),
        )
        update = apply(self.gpu, merge_heads(mixed), block.self_o)
        tokens = tokens + update * gate_msa
        normed = layernorm(tokens, block.norm3, block.norm3_bias)
        query = qk_norm(apply(self.gpu, normed, block.cross_q), block.cross_norm_q)
        mixed = attend(
            self.gpu,
            split_heads(query, HEADS),
            split_heads(block.cross_key, HEADS),
            split_heads(block.cross_value, HEADS),
        )
        tokens = tokens + apply(self.gpu, merge_heads(mixed), block.cross_o)
        normed = layernorm(tokens)
        normed = normed * (1.0 + scale_mlp) + shift_mlp
        hidden = self.gelu(apply(self.gpu, normed, block.ffn0))
        tokens = tokens + apply(self.gpu, hidden, block.ffn2) * gate_mlp
        return tokens

    def _head(self, tokens: np.ndarray, temb: np.ndarray) -> np.ndarray:
        shift = temb + self.head_mod[0]
        scale = temb + self.head_mod[1]
        normed = layernorm(tokens)
        normed = normed * (1.0 + scale) + shift
        return apply(self.gpu, normed, self.head)


class _Block:
    def __init__(self, root: Path) -> None:
        self.modulation = np.load(root / "modulation.npy").astype(np.float32).reshape(1, -1)
        self.norm3 = np.load(root / "norm3.npy")
        self.norm3_bias = np.load(root / "norm3.bias.npy")
        self.self_norm_q = np.load(root / "self_norm_q.npy")
        self.self_norm_k = np.load(root / "self_norm_k.npy")
        self.cross_norm_q = np.load(root / "cross_norm_q.npy")
        self.cross_norm_k = np.load(root / "cross_norm_k.npy")
        self.self_q = load_matrix(root / "self_q.npz")
        self.self_k = load_matrix(root / "self_k.npz")
        self.self_v = load_matrix(root / "self_v.npz")
        self.self_o = load_matrix(root / "self_o.npz")
        self.cross_q = load_matrix(root / "cross_q.npz")
        self.cross_k = load_matrix(root / "cross_k.npz")
        self.cross_v = load_matrix(root / "cross_v.npz")
        self.cross_o = load_matrix(root / "cross_o.npz")
        self.ffn0 = load_matrix(root / "ffn0.npz")
        self.ffn2 = load_matrix(root / "ffn2.npz")

    @staticmethod
    def load(root: Path) -> _Block:
        return _Block(root)


def _unpatch(tokens: np.ndarray, time: int, height: int, width: int) -> np.ndarray:
    data = tokens.reshape(time, height, width, 1, 2, 2, IN_DIM)
    data = np.transpose(data, (6, 0, 3, 1, 4, 2, 5))
    return data.reshape(IN_DIM, time, height * 2, width * 2)
