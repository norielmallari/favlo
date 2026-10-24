"""44.1 kHz DAC decoder. Kumar et al., shapes from the published 9-codebook checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import objc

from favlo.memory import memory
from zonos2.layout import CODEBOOKS, HOP, SAMPLE_RATE

_LANGUAGE = (4 << 16) + 1
_SOURCE = Path(__file__).resolve().parent / "shaders" / "codec.metal"
_RATIOS = (8, 8, 4, 2)
_DILATIONS = (1, 3, 9)


class Decoder:
    """Sum the nine codebook projections, then the decoder stack, then tanh."""

    def __init__(self, path: Path) -> None:
        from safetensors import safe_open

        if not path.is_file():
            message = "DAC weights are not in models/dac/model.safetensors"
            raise FileNotFoundError(message)
        self._device, self._queue, self._pipelines = _compile()
        with safe_open(path, framework="numpy") as handle:
            self._conv1 = _load_conv(handle, "decoder.conv1")
            self._conv2 = _load_conv(handle, "decoder.conv2")
            self._final_snake = _alpha(handle, "decoder.snake1.alpha")
            self._blocks = tuple(_load_block(handle, index) for index in range(len(_RATIOS)))
            self._books = tuple(_load_book(handle, index) for index in range(CODEBOOKS))

    def waveform(self, codes: np.ndarray) -> np.ndarray:
        """Aligned codes ``[frames, 9]`` in ``0..1023`` become one mono clip."""
        frames, books = codes.shape
        if books != CODEBOOKS or frames < 1:
            message = "aligned codes are empty"
            raise ValueError(message)
        if int(codes.min()) < 0 or int(codes.max()) > 1023:
            message = "a DAC code is outside the codebook"
            raise ValueError(message)
        latent = np.zeros((frames, 1024), dtype=np.float32)
        for book, (table, weight, bias) in enumerate(self._books):
            looked = table[codes[:, book]]
            latent += self._conv(looked, weight, bias, 0)
        hidden = self._conv(latent, *self._conv1, 3)
        for block in self._blocks:
            hidden = self._block(hidden, block)
        hidden = self._snake(hidden, self._final_snake)
        wave = self._conv(hidden, *self._conv2, 3)
        return np.tanh(wave[:, 0])

    def _block(self, hidden: np.ndarray, block: dict) -> np.ndarray:
        activated = self._snake(hidden, block["snake"])
        stride = block["stride"]
        hidden = self._transpose(activated, block["weight"], block["bias"], stride, stride // 2)
        for unit, dilation in zip(block["units"], _DILATIONS, strict=True):
            residual = hidden
            hidden = self._snake(hidden, unit["snake1"])
            span = dilation * (unit["conv1"].shape[-1] - 1) // 2
            hidden = self._conv(hidden, unit["conv1"], unit["bias1"], span, dilation)
            hidden = self._snake(hidden, unit["snake2"])
            hidden = self._conv(hidden, unit["conv2"], unit["bias2"], 0)
            hidden = residual + hidden
        return hidden

    def _conv(
        self,
        values: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        padding: int,
        dilation: int = 1,
    ) -> np.ndarray:
        return _dispatch_conv(
            self._device, self._queue, self._pipelines["conv1d"], values, weight, bias, padding, dilation
        )

    def _transpose(
        self, values: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int, padding: int
    ) -> np.ndarray:
        return _dispatch_transpose(
            self._device,
            self._queue,
            self._pipelines["conv_transpose1d"],
            values,
            weight,
            bias,
            stride,
            padding,
        )

    def _snake(self, values: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return _dispatch_snake(self._device, self._queue, self._pipelines["snake"], values, alpha)


def seconds(frames: int) -> float:
    return frames * HOP / SAMPLE_RATE


def check_codec() -> None:
    """One conv, one transposed conv, and Snake against the host formula."""
    device, queue, pipelines = _compile()
    rng = np.random.default_rng(4)
    values = rng.standard_normal((5, 3)).astype(np.float32)
    weight = rng.standard_normal((4, 3, 3)).astype(np.float32)
    bias = rng.standard_normal(4).astype(np.float32)
    got = _dispatch_conv(device, queue, pipelines["conv1d"], values, weight, bias, 1, 1)
    _close(got, _conv_host(values, weight, bias, 1, 1), "conv")
    values = rng.standard_normal((12, 2)).astype(np.float32)
    weight = rng.standard_normal((3, 2, 3)).astype(np.float32)
    bias = rng.standard_normal(3).astype(np.float32)
    got = _dispatch_conv(device, queue, pipelines["conv1d"], values, weight, bias, 3, 3)
    _close(got, _conv_host(values, weight, bias, 3, 3), "dilated conv")
    values = rng.standard_normal((3, 2)).astype(np.float32)
    weight = rng.standard_normal((2, 3, 4)).astype(np.float32)
    bias = rng.standard_normal(3).astype(np.float32)
    got = _dispatch_transpose(device, queue, pipelines["conv_transpose1d"], values, weight, bias, 2, 1)
    _close(got, _transpose_host(values, weight, bias, 2, 1), "transpose")
    values = rng.standard_normal((4, 3)).astype(np.float32)
    alpha = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    got = _dispatch_snake(device, queue, pipelines["snake"], values, alpha)
    wave = np.sin(alpha * values)
    _close(got, values + wave * wave / alpha, "snake")


def _compile():
    import Metal

    device = memory.device
    options = Metal.MTLCompileOptions.alloc().init()
    options.setLanguageVersion_(_LANGUAGE)
    library, error = device.newLibraryWithSource_options_error_(
        _SOURCE.read_text(encoding="utf-8"), options, None
    )
    if library is None:
        message = str(error) if error is not None else "codec shaders failed"
        raise RuntimeError(message)
    pipelines = {}
    for name in ("conv1d", "conv_transpose1d", "snake"):
        function = library.newFunctionWithName_(name)
        pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            message = str(error) if error is not None else f"{name} did not compile"
            raise RuntimeError(message)
        pipelines[name] = pipeline
    queue = memory.make_queue()
    return device, queue, pipelines


def _load_conv(handle, name: str) -> tuple[np.ndarray, np.ndarray]:
    weight = np.ascontiguousarray(handle.get_tensor(f"{name}.weight"), dtype=np.float32)
    bias = np.ascontiguousarray(handle.get_tensor(f"{name}.bias"), dtype=np.float32)
    return weight, bias


def _load_block(handle, index: int) -> dict:
    prefix = f"decoder.block.{index}"
    weight, bias = _load_conv(handle, f"{prefix}.conv_t1")
    kernel = weight.shape[-1]
    units = []
    for unit in range(1, 4):
        conv1, bias1 = _load_conv(handle, f"{prefix}.res_unit{unit}.conv1")
        conv2, bias2 = _load_conv(handle, f"{prefix}.res_unit{unit}.conv2")
        units.append(
            {
                "conv1": conv1,
                "bias1": bias1,
                "conv2": conv2,
                "bias2": bias2,
                "snake1": _alpha(handle, f"{prefix}.res_unit{unit}.snake1.alpha"),
                "snake2": _alpha(handle, f"{prefix}.res_unit{unit}.snake2.alpha"),
            }
        )
    return {
        "weight": weight,
        "bias": bias,
        "stride": kernel // 2,
        "snake": _alpha(handle, f"{prefix}.snake1.alpha"),
        "units": units,
    }


def _load_book(handle, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"quantizer.quantizers.{index}"
    table = np.ascontiguousarray(handle.get_tensor(f"{prefix}.codebook.weight"), dtype=np.float32)
    weight, bias = _load_conv(handle, f"{prefix}.out_proj")
    return table, weight, bias


def _alpha(handle, name: str) -> np.ndarray:
    return np.ascontiguousarray(handle.get_tensor(name).reshape(-1), dtype=np.float32)


def _dispatch_conv(device, queue, pipeline, values, weight, bias, padding: int, dilation: int) -> np.ndarray:

    data = np.ascontiguousarray(values, dtype=np.float32)
    kernel = np.ascontiguousarray(weight, dtype=np.float32)
    offset = np.ascontiguousarray(bias, dtype=np.float32)
    length, in_channels = data.shape
    out_channels = kernel.shape[0]
    out = np.zeros((length, out_channels), dtype=np.float32)
    buffers = [_buffer(device, item) for item in (data, kernel, offset, out)]
    scalars = [
        np.uint32(length),
        np.uint32(in_channels),
        np.uint32(out_channels),
        np.uint32(kernel.shape[-1]),
        np.int32(padding),
        np.int32(dilation),
    ]
    _encode(queue, device, pipeline, buffers, scalars, (length, out_channels, 1), out)
    return out.reshape(length, out_channels)


def _dispatch_transpose(device, queue, pipeline, values, weight, bias, stride: int, padding: int) -> np.ndarray:
    data = np.ascontiguousarray(values, dtype=np.float32)
    kernel = np.ascontiguousarray(weight, dtype=np.float32)
    offset = np.ascontiguousarray(bias, dtype=np.float32)
    in_length, in_channels = data.shape
    out_channels = kernel.shape[1]
    taps = kernel.shape[-1]
    out_length = (in_length - 1) * stride - 2 * padding + taps
    out = np.zeros((out_length, out_channels), dtype=np.float32)
    buffers = [_buffer(device, item) for item in (data, kernel, offset, out)]
    scalars = [
        np.uint32(in_length),
        np.uint32(out_length),
        np.uint32(in_channels),
        np.uint32(out_channels),
        np.uint32(taps),
        np.uint32(stride),
        np.int32(padding),
    ]
    _encode(queue, device, pipeline, buffers, scalars, (out_length, out_channels, 1), out)
    return out.reshape(out_length, out_channels)


def _dispatch_snake(device, queue, pipeline, values, alpha) -> np.ndarray:
    data = np.ascontiguousarray(values, dtype=np.float32)
    scale = np.ascontiguousarray(alpha, dtype=np.float32)
    out = np.zeros_like(data)
    buffers = [_buffer(device, item) for item in (data, scale, out)]
    _encode(queue, device, pipeline, buffers, [np.uint32(data.shape[1])], (data.size, 1, 1), out)
    return out.reshape(data.shape)


def _encode(queue, device, pipeline, buffers, scalars, grid, out) -> None:
    import Metal

    with objc.autorelease_pool():
        extra = [_buffer(device, np.array([value], dtype=value.dtype)) for value in scalars]
        command = memory.command_buffer(queue)
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buffer in enumerate(buffers + extra):
            memory.bind(encoder, index, buffer, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSize(*grid), Metal.MTLSize(min(grid[0], 64), 1, 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        memory.retire(command)
        view = buffers[-1 if out is not None else 0].contents().as_buffer(out.nbytes)
        out[:] = np.frombuffer(view, dtype=np.float32).reshape(out.shape)


def _buffer(device, array: np.ndarray):
    del device
    return memory.upload(np.ascontiguousarray(array), owner="zonos")


def _conv_host(values, weight, bias, padding: int, dilation: int) -> np.ndarray:
    length, _in_channels = values.shape
    out_channels, _, kernel = weight.shape
    out = np.zeros((length, out_channels), dtype=np.float32)
    for time in range(length):
        acc = bias.astype(np.float32).copy()
        for tap in range(kernel):
            index = time + tap * dilation - padding
            if 0 <= index < length:
                acc += values[index] @ weight[:, :, tap].T
        out[time] = acc
    return out


def _transpose_host(values, weight, bias, stride: int, padding: int) -> np.ndarray:
    """Adjoint of cross-correlation. Tap ``k`` lands at ``source * stride + k - padding``."""
    in_length, _in_channels = values.shape
    _cin, out_channels, kernel = weight.shape
    out_length = (in_length - 1) * stride - 2 * padding + kernel
    out = np.zeros((out_length, out_channels), dtype=np.float32)
    for time in range(out_length):
        acc = bias.astype(np.float32).copy()
        for tap in range(kernel):
            numerator = time - tap + padding
            if numerator < 0 or numerator % stride != 0:
                continue
            source = numerator // stride
            if source >= in_length:
                continue
            acc += values[source] @ weight[:, :, tap]
        out[time] = acc
    return out


def _close(got: np.ndarray, expect: np.ndarray, name: str) -> None:
    if not np.allclose(got, expect, rtol=1e-4, atol=1e-4):
        message = f"{name} disagrees with the host formula"
        raise RuntimeError(message)
