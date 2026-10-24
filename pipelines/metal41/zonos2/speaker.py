"""ECAPA-TDNN speaker encoder. Desplanques et al., shapes from the Qwen3-TTS 1.7B module."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from zonos2.audio import read_wav
from zonos2.codec import _compile, _dispatch_conv
from zonos2.layout import SPEAKER_DIM
from zonos2.mel import log_mel

_SCALE = 8
_DILATIONS = (2, 3, 4)


class Encoder:
    """One reference clip becomes the 2048-float vector the decoder conditions on."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            message = "speaker encoder weights are not in models/qwen3-speaker/model.safetensors"
            raise FileNotFoundError(message)
        weights = _load(path)
        self._owner = f"zonos2:{id(self)}"
        self._device, self._queue, pipelines = _compile()
        self._conv = pipelines["conv1d"]
        self._stem = _pair(weights, "blocks.0.conv")
        self._blocks = tuple(_block(weights, index) for index in (1, 2, 3))
        self._mfa = _pair(weights, "mfa.conv")
        self._asp_tdnn = _pair(weights, "asp.tdnn.conv")
        self._asp = _pair(weights, "asp.conv")
        self._fc = _pair(weights, "fc")

    def embed(self, path: Path) -> np.ndarray:
        """Read a PCM wav and return one 2048-float vector."""
        wave, rate = read_wav(path)
        features = log_mel(wave, rate)
        hidden = self._relu(self._same(features, *self._stem, 1))
        maps = []
        for block, dilation in zip(self._blocks, _DILATIONS, strict=True):
            hidden = self._se_res2(hidden, block, dilation)
            maps.append(hidden)
        joined = np.concatenate(maps, axis=1)
        hidden = self._relu(self._same(joined, *self._mfa, 1))
        vector = self._same(self._pool(hidden)[None, :], *self._fc, 1)[0]
        if vector.shape != (SPEAKER_DIM,) or not np.isfinite(vector).all():
            message = "the speaker vector is not 2048 finite floats"
            raise RuntimeError(message)
        return vector

    def _se_res2(self, hidden: np.ndarray, block: dict, dilation: int) -> np.ndarray:
        residual = hidden
        hidden = self._relu(self._same(hidden, block["tdnn1"][0], block["tdnn1"][1], 1))
        hidden = self._res2(hidden, block["branches"], dilation)
        hidden = self._relu(self._same(hidden, block["tdnn2"][0], block["tdnn2"][1], 1))
        hidden = self._squeeze(hidden, block["se1"], block["se2"])
        return hidden + residual

    def _res2(self, hidden: np.ndarray, branches: tuple, dilation: int) -> np.ndarray:
        parts = np.split(hidden, _SCALE, axis=1)
        outputs = [parts[0]]
        carried = None
        for index, (weight, bias) in enumerate(branches):
            incoming = parts[index + 1]
            if carried is not None:
                incoming = incoming + carried
            carried = self._relu(self._same(incoming, weight, bias, dilation))
            outputs.append(carried)
        return np.concatenate(outputs, axis=1)

    def _squeeze(self, hidden: np.ndarray, first: tuple, second: tuple) -> np.ndarray:
        pooled = hidden.mean(axis=0, keepdims=True)
        excited = self._relu(self._same(pooled, first[0], first[1], 1))
        gate = 1.0 / (1.0 + np.exp(-self._same(excited, second[0], second[1], 1)))
        return hidden * gate

    def _pool(self, hidden: np.ndarray) -> np.ndarray:
        """Channel attention, then the weighted mean and standard deviation."""
        center = hidden.mean(axis=0)
        spread = np.sqrt(np.maximum(np.mean(hidden * hidden, axis=0) - center * center, 0.0))
        context = np.concatenate(
            [hidden, np.broadcast_to(center, hidden.shape), np.broadcast_to(spread, hidden.shape)],
            axis=1,
        )
        # The TDNN block is ReLU. The attentive pool then applies tanh before the score.
        attended = np.tanh(self._relu(self._same(context, *self._asp_tdnn, 1)))
        scores = self._same(attended, *self._asp, 1)
        scores -= scores.max(axis=0, keepdims=True)
        attention = np.exp(scores)
        attention /= attention.sum(axis=0, keepdims=True)
        mean = (attention * hidden).sum(axis=0)
        square = (attention * hidden * hidden).sum(axis=0)
        deviation = np.sqrt(np.maximum(square - mean * mean, 0.0))
        return np.concatenate([mean, deviation]).astype(np.float32)

    def _same(
        self, values: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int
    ) -> np.ndarray:
        kernel = int(weight.shape[-1])
        pad = dilation * (kernel - 1) // 2
        length = values.shape[0]
        if pad:
            if length <= pad:
                message = "the clip is shorter than the encoder window"
                raise ValueError(message)
            values = np.pad(values, ((pad, pad), (0, 0)), mode="reflect")
        out = _dispatch_conv(
            self._device,
            self._queue,
            self._conv,
            values,
            weight,
            bias,
            0,
            dilation,
            self._owner,
        )
        return out[:length]

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)


def _block(weights: dict[str, np.ndarray], index: int) -> dict:
    prefix = f"blocks.{index}"
    names = (f"{prefix}.res2net_block.blocks.{branch}.conv" for branch in range(_SCALE - 1))
    branches = tuple(_pair(weights, name) for name in names)
    return {
        "tdnn1": _pair(weights, f"{prefix}.tdnn1.conv"),
        "tdnn2": _pair(weights, f"{prefix}.tdnn2.conv"),
        "branches": branches,
        "se1": _pair(weights, f"{prefix}.se_block.conv1"),
        "se2": _pair(weights, f"{prefix}.se_block.conv2"),
    }


def _pair(weights: dict[str, np.ndarray], name: str) -> tuple[np.ndarray, np.ndarray]:
    return weights[f"{name}.weight"], weights[f"{name}.bias"]


def _load(path: Path) -> dict[str, np.ndarray]:
    blob = path.read_bytes()
    count = struct.unpack_from("<Q", blob, 0)[0]
    header = json.loads(blob[8 : 8 + count])
    data = memoryview(blob)[8 + count :]
    found = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "BF16":
            message = f"{name} is not bfloat16"
            raise ValueError(message)
        start, end = meta["data_offsets"]
        bits = np.frombuffer(data[start:end], dtype="<u2").astype(np.uint32) << 16
        values = np.array(bits.view(np.float32), copy=True).reshape(meta["shape"])
        found[name] = values
    return found
