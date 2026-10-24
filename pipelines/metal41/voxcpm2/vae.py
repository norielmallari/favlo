"""AudioVAE V2 decoder. Weights are float32. The convolutions run on the device."""

from __future__ import annotations

import collections
import io
import math
import pickle
import zipfile
from pathlib import Path

import numpy as np

from voxcpm2.metal import Session


class Decoder:
    """Depthwise causal decoder. Sample-rate conditioning is a scale and a bias per stage."""

    def __init__(
        self,
        session: Session,
        path: Path,
        rates: tuple[int, ...],
        boundaries: tuple[int, ...],
        rate: int,
    ) -> None:
        self.session = session
        self.rates = rates
        self.state = _load(path)
        self.sr_index = _bucket(rate, boundaries)
        _normalize(self.state)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """``latent`` is ``[channels, time]``. The waveform is one float32 channel."""
        data = np.ascontiguousarray(latent, dtype=np.float32)
        first = self.state["decoder.model.0.weight"]
        if tuple(int(item) for item in first.shape) != (int(data.shape[0]), 1, 7):
            message = f"decoder.model.0 weight is {tuple(first.shape)}"
            raise ValueError(message)
        data = self._conv(data, "decoder.model.0", left=6, dilation=1, groups=int(data.shape[0]))
        data = self._conv(data, "decoder.model.1", left=0, dilation=1, groups=1)
        channels = int(data.shape[0])
        for index, stride in enumerate(self.rates):
            block = 2 + index
            data = self._film(data, block)
            data = self._snake(data, f"decoder.model.{block}.block.0.alpha")
            data = self._transpose(data, f"decoder.model.{block}.block.1", int(stride))
            for unit, dilation in ((2, 1), (3, 3), (4, 9)):
                data = self._residual(data, f"decoder.model.{block}.block.{unit}", dilation)
            channels = channels // 2
        if int(data.shape[0]) != channels:
            message = f"decoder width is {data.shape[0]}, the rates leave {channels}"
            raise RuntimeError(message)
        data = self._snake(data, "decoder.model.8.alpha")
        data = self._conv(data, "decoder.model.9", left=6, dilation=1, groups=1)
        return np.tanh(data[0]).astype(np.float32)

    def _film(self, data: np.ndarray, index: int) -> np.ndarray:
        scale = self.state[f"decoder.sr_cond_model.{index}.scale_embed.weight"][self.sr_index]
        bias = self.state[f"decoder.sr_cond_model.{index}.bias_embed.weight"][self.sr_index]
        return data * scale.reshape(-1, 1) + bias.reshape(-1, 1)

    def _snake(self, data: np.ndarray, name: str) -> np.ndarray:
        alpha = self.state[name].reshape(-1, 1)
        if int(alpha.shape[0]) != int(data.shape[0]):
            message = f"{name} covers {alpha.shape[0]} channels, the activation has {data.shape[0]}"
            raise ValueError(message)
        return data + np.sin(alpha * data) ** 2 / (alpha + 1e-9)

    def _residual(self, data: np.ndarray, prefix: str, dilation: int) -> np.ndarray:
        pad = ((7 - 1) * dilation) // 2
        hidden = self._snake(data, f"{prefix}.block.0.alpha")
        hidden = self._conv(
            hidden,
            f"{prefix}.block.1",
            left=pad * 2,
            dilation=dilation,
            groups=int(data.shape[0]),
        )
        hidden = self._snake(hidden, f"{prefix}.block.2.alpha")
        hidden = self._conv(hidden, f"{prefix}.block.3", left=0, dilation=1, groups=1)
        return data + hidden

    def _conv(
        self,
        data: np.ndarray,
        prefix: str,
        *,
        left: int,
        dilation: int,
        groups: int,
    ) -> np.ndarray:
        return self.session.conv1d(
            data,
            self.state[f"{prefix}.weight"],
            self.state[f"{prefix}.bias"],
            left=left,
            dilation=dilation,
            groups=groups,
        )

    def _transpose(self, data: np.ndarray, prefix: str, stride: int) -> np.ndarray:
        weight = self.state[f"{prefix}.weight"]
        if int(weight.shape[-1]) != stride * 2:
            message = f"{prefix} kernel is {weight.shape[-1]}, the stride is {stride}"
            raise ValueError(message)
        bias = self.state[f"{prefix}.bias"]
        return self.session.conv_transpose1d(data, weight, bias, stride=stride)


def _bucket(rate: int, boundaries: tuple[int, ...]) -> int:
    index = 0
    for edge in boundaries:
        if int(rate) >= int(edge):
            index += 1
        else:
            break
    return index


def _normalize(state: dict[str, np.ndarray]) -> None:
    names = [name[: -len(".weight_v")] for name in state if name.endswith(".weight_v")]
    for name in names:
        direction = state.pop(f"{name}.weight_v")
        gain = state.pop(f"{name}.weight_g")
        axes = tuple(range(1, direction.ndim))
        norm = np.linalg.norm(direction, axis=axes, keepdims=True)
        state[f"{name}.weight"] = (direction * (gain / norm)).astype(np.float32)


def _load(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("vae/data.pkl")
        reader = _Unpickler(io.BytesIO(payload))
        reader.archive = archive
        loaded = reader.load()
    state = loaded["state_dict"]
    arrays: dict[str, np.ndarray] = {}
    for name, meta in state.items():
        if not isinstance(meta, dict) or "size" not in meta:
            continue
        arrays[name] = meta["array"]
    return arrays


class _Unpickler(pickle.Unpickler):
    archive: zipfile.ZipFile

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if name == "_rebuild_tensor_v2":
            return _rebuild
        if name == "_rebuild_parameter":
            return lambda data, requires_grad, backward_hooks: data

        class _Dummy:
            def __setstate__(self, state: object) -> None:
                if isinstance(state, dict):
                    self.__dict__.update(state)

        return _Dummy

    def persistent_load(self, pid: object) -> object:
        if not isinstance(pid, tuple) or len(pid) < 5:
            message = "AudioVAE storage id is missing"
            raise ValueError(message)
        key = str(pid[2])
        count = int(pid[4])
        raw = self.archive.read(f"vae/data/{key}")
        if len(raw) != count * 4:
            message = f"AudioVAE storage {key} is {len(raw)} bytes for {count} values"
            raise ValueError(message)
        return np.frombuffer(raw, dtype="<f4")


def _rebuild(
    storage: np.ndarray,
    offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    hooks: object,
    metadata: object = None,
) -> dict[str, np.ndarray]:
    del requires_grad, hooks, metadata
    shape = tuple(int(item) for item in size)
    steps = tuple(int(item) for item in stride)
    if steps != _contiguous(shape):
        message = f"AudioVAE tensor stride {steps} is not contiguous"
        raise ValueError(message)
    count = int(math.prod(shape))
    start = int(offset)
    array = np.array(storage[start : start + count], dtype=np.float32).reshape(shape)
    return {"size": shape, "array": array}


def _contiguous(shape: tuple[int, ...]) -> tuple[int, ...]:
    steps: list[int] = []
    acc = 1
    for dim in reversed(shape):
        steps.append(acc)
        acc *= int(dim)
    steps.reverse()
    return tuple(steps)
