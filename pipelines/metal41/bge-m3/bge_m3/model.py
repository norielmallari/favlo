"""XLM-RoBERTa encoder for BGE-M3.

The graph is the checkpoint's encoder plus the ColBERT projection. Attention
is one head at a time. The pad fill is -40000, which is the finite value the
Neural Engine softmax accepts. The tanh pooler is not loaded. Quantize is
the only importer.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from bge_m3.text import HIDDEN

LAYERS = 24
HEADS = 16
HEAD = 64
INTERMEDIATE = 4096
VOCAB = 250002
POSITIONS = 8194
EPS = 1e-5


class _Pointwise(nn.Module):
    """A 1×1 convolution. The sequence axis stays whole so the quantize pair can fuse."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden.transpose(1, 2).unsqueeze(2)
        value = self.conv(value)
        return value.squeeze(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = _Pointwise(HIDDEN, HIDDEN)
        self.key = _Pointwise(HIDDEN, HIDDEN)
        self.value = _Pointwise(HIDDEN, HIDDEN)
        self.out = _Pointwise(HIDDEN, HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN, eps=EPS)

    def forward(self, hidden: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        parts: list[torch.Tensor] = []
        for head in range(HEADS):
            start = head * HEAD
            stop = start + HEAD
            scores = torch.matmul(query[:, :, start:stop], key[:, :, start:stop].transpose(-1, -2))
            scores = scores * query.new_tensor(HEAD**-0.5) + key_mask
            weights = torch.softmax(scores, dim=-1)
            parts.append(torch.matmul(weights, value[:, :, start:stop]))
        mixed = self.out(torch.cat(parts, dim=-1))
        return self.norm(hidden + mixed)


class _FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = _Pointwise(HIDDEN, INTERMEDIATE)
        self.down = _Pointwise(INTERMEDIATE, HIDDEN)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(HIDDEN, eps=EPS)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden + self.down(self.act(self.up(hidden))))


class Encoder(nn.Module):
    """One bidirectional encode.

    ``input_ids`` and ``position_ids`` are ``[1, length]``. ``key_mask`` is
    ``[1, 1, length]`` with 0 on real tokens and -40000 on pads. Position ids
    and the mask are host inputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.word = nn.Embedding(VOCAB, HIDDEN)
        self.position = nn.Embedding(POSITIONS, HIDDEN)
        self.type_row = nn.Parameter(torch.zeros(1, 1, HIDDEN))
        self.embed_norm = nn.LayerNorm(HIDDEN, eps=EPS)
        self.attention = nn.ModuleList(_Attention() for _ in range(LAYERS))
        self.feed_forward = nn.ModuleList(_FeedForward() for _ in range(LAYERS))
        self.colbert = _Pointwise(HIDDEN, HIDDEN)

    def embed_tokens(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Token, position, and type rows, then the embedding LayerNorm."""
        hidden = self.word(input_ids) + self.position(position_ids) + self.type_row
        return self.embed_norm(hidden)

    def encode_hidden(
        self,
        hidden: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The transformer body. ``hidden`` is the embedding LayerNorm output."""
        for attention, feed_forward in zip(self.attention, self.feed_forward, strict=True):
            hidden = feed_forward(attention(hidden, key_mask))
        return hidden, self.colbert(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens(input_ids, position_ids)
        return self.encode_hidden(hidden, key_mask)


def load_encoder(source: Path) -> Encoder:
    """Load the float32 encoder and the ColBERT matrix. The pooler stays out."""
    encoder = Encoder()
    state = _load_state(source / "pytorch_model.bin")
    _copy_embeddings(encoder, state)
    for layer in range(LAYERS):
        prefix = f"encoder.layer.{layer}."
        attention = encoder.attention[layer]
        _copy_linear(attention.query, state, prefix + "attention.self.query.")
        _copy_linear(attention.key, state, prefix + "attention.self.key.")
        _copy_linear(attention.value, state, prefix + "attention.self.value.")
        _copy_linear(attention.out, state, prefix + "attention.output.dense.")
        _copy_norm(attention.norm, state, prefix + "attention.output.LayerNorm.")
        feed = encoder.feed_forward[layer]
        _copy_linear(feed.up, state, prefix + "intermediate.dense.")
        _copy_linear(feed.down, state, prefix + "output.dense.")
        _copy_norm(feed.norm, state, prefix + "output.LayerNorm.")
    colbert = _load_state(source / "colbert_linear.pt")
    _copy_linear(encoder.colbert, colbert, "")
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def load_sparse(source: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """The lexical map, weight ``[1024]`` and a scalar bias."""
    state = _load_state(source / "sparse_linear.pt")
    weight = _lookup(state, "weight").detach().reshape(-1).to(torch.float32)
    bias = _lookup(state, "bias").detach().reshape(-1).to(torch.float32)
    if weight.numel() != HIDDEN or bias.numel() != 1:
        message = "sparse_linear.pt is not a 1024 to 1 map"
        raise ValueError(message)
    return weight, bias


def _copy_embeddings(encoder: Encoder, state: dict[str, torch.Tensor]) -> None:
    encoder.word.weight.data.copy_(_lookup(state, "embeddings.word_embeddings.weight"))
    encoder.position.weight.data.copy_(_lookup(state, "embeddings.position_embeddings.weight"))
    row = _lookup(state, "embeddings.token_type_embeddings.weight")
    encoder.type_row.data.copy_(row.reshape(1, 1, HIDDEN))
    _copy_norm(encoder.embed_norm, state, "embeddings.LayerNorm.")


def _copy_linear(layer: _Pointwise, state: dict[str, torch.Tensor], prefix: str) -> None:
    weight = _lookup(state, prefix + "weight")
    layer.conv.weight.data.copy_(weight.reshape(weight.shape[0], weight.shape[1], 1, 1))
    layer.conv.bias.data.copy_(_lookup(state, prefix + "bias").reshape(-1))


def _copy_norm(layer: nn.LayerNorm, state: dict[str, torch.Tensor], prefix: str) -> None:
    layer.weight.data.copy_(_lookup(state, prefix + "weight"))
    layer.bias.data.copy_(_lookup(state, prefix + "bias"))


def _lookup(state: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    if suffix in state:
        return state[suffix]
    matches = [key for key in state if key.endswith("." + suffix)]
    if len(matches) != 1:
        message = f"checkpoint is missing {suffix}"
        raise KeyError(message)
    return state[matches[0]]


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        message = f"{path.name} is not in the checkout"
        raise FileNotFoundError(message)
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if hasattr(blob, "state_dict"):
        blob = blob.state_dict()
    if not isinstance(blob, dict):
        message = f"{path.name} is not a state dict"
        raise ValueError(message)
    state: dict[str, torch.Tensor] = {}
    for key, value in blob.items():
        if torch.is_tensor(value):
            state[str(key)] = value
    if not state:
        message = f"{path.name} has no tensors"
        raise ValueError(message)
    return state
