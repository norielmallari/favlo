"""Packed VoxCPM2 weights. These checks use a temp directory and no GPU."""

from __future__ import annotations

import collections
import io
import json
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np

from voxcpm2.metal41.commands import mark_complete
from voxcpm2.metal41.metal import Resident, Span, _binding, device_argument, pad_linear
from voxcpm2.metal41.pack import write_pack
from voxcpm2.metal41.runtime import _SCRATCH, Runtime, open_device
from voxcpm2.metal41.store import (
    EMBED_NAME,
    PACK_NAME,
    WEIGHTS_NAME,
    Store,
    _bf16_f32,
    read_into,
    weight_bytes,
)
from voxcpm2.metal41.vae import Decoder, _contiguous

_CONFIG = {
    "patch_size": 4,
    "feat_dim": 64,
    "scalar_quantization_latent_dim": 512,
    "scalar_quantization_scale": 9,
    "residual_lm_num_layers": 8,
    "residual_lm_no_rope": True,
    "max_length": 8192,
    "lm_config": {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 2,
        "kv_channels": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "vocab_size": 73448,
        "max_position_embeddings": 32768,
        "use_mup": False,
        "scale_emb": 12,
        "scale_depth": 1.4,
        "rope_scaling": {
            "long_factor": [1.0, 1.0],
            "short_factor": [1.0, 1.0],
            "original_max_position_embeddings": 32768,
        },
    },
}


def test_pack_bytes_match_the_kernel_layout(tmp_path: Path) -> None:
    linear = np.array([[0.1, -1.3, 2.2, 4.0], [0.5, 0.25, -0.75, 1.5], [3.0, -0.1, 0.2, 0.3]])
    gain = np.array([0.1, 1.5, -0.25], dtype=np.float32)
    bias = np.array([0.1, -2.0, 0.5], dtype=np.float32)
    embed = np.array(
        [[0.1, 0.2, 0.3, 0.4], [-1.25, 0.5, 2.0, 0.0]],
        dtype=np.float32,
    )
    special = np.array([0.1, -0.2, 0.3, 4.0], dtype=np.float32).reshape(1, 1, 1, 4)
    direction = np.array([[[1.0, 0.0, 0.0]], [[0.0, 3.0, 4.0]]], dtype=np.float32)
    scale = np.array([[[2.0]], [[3.0]]], dtype=np.float32)
    vae_bias = np.array([0.25, -0.5], dtype=np.float32)
    alpha = np.array([0.1, 1.25], dtype=np.float32)
    _save_bf16(
        tmp_path / "model.safetensors",
        {
            "proj.weight": linear,
            "block.norm.weight": gain,
            EMBED_NAME: embed,
            "proj.bias": bias,
            "feat_encoder.special_token": special,
        },
    )
    _save_vae(
        tmp_path / "audiovae.pth",
        {
            "decoder.model.0.weight_v": direction,
            "decoder.model.0.weight_g": scale,
            "decoder.model.0.bias": vae_bias,
            "decoder.snake.alpha": alpha,
        },
    )
    before = (tmp_path / "model.safetensors").stat().st_size
    assert write_pack(tmp_path) == [WEIGHTS_NAME, PACK_NAME]
    assert (tmp_path / "model.safetensors").stat().st_size == before
    assert (tmp_path / "audiovae.pth").is_file()

    blob = (tmp_path / WEIGHTS_NAME).read_bytes()
    body = json.loads((tmp_path / PACK_NAME).read_text(encoding="utf-8"))
    rows = body["tensors"]
    assert body["version"] == 1
    assert body["file"] == WEIGHTS_NAME
    assert "decoder.model.0.weight_v" not in rows
    order = [rows[name]["offset"] for name in (
        "proj.weight",
        "block.norm.weight",
        "proj.bias",
        EMBED_NAME,
        "feat_encoder.special_token",
        "decoder.model.0.bias",
        "decoder.model.0.weight",
        "decoder.snake.alpha",
    )]
    assert order == sorted(order)
    assert len(set(order)) == len(order)
    previous = 0
    for name in sorted(rows, key=lambda item: rows[item]["offset"]):
        row = rows[name]
        offset = int(row["offset"])
        nbytes = int(row["nbytes"])
        assert offset % 128 == 0
        assert blob[previous:offset] == b"\0" * (offset - previous)
        assert len(blob[offset : offset + nbytes]) == nbytes
        previous = offset + nbytes
    assert previous == len(blob)

    promoted = _promoted(linear)
    padded, out, inner, out_pad, inner_pad = pad_linear(promoted)
    linear_row = rows["proj.weight"]
    assert linear_row["kind"] == "linear"
    assert linear_row["dtype"] == "F16"
    assert linear_row["shape"] == [out, inner]
    assert linear_row["pad"] == [out_pad, inner_pad]
    assert _slice(blob, linear_row) == padded.tobytes()
    gain_f16 = np.ascontiguousarray(_promoted(gain), dtype=np.float16).tobytes()
    bias_f32 = np.ascontiguousarray(_promoted(bias), dtype=np.float32).tobytes()
    embed_f32 = np.ascontiguousarray(_promoted(embed), dtype=np.float32).tobytes()
    assert _slice(blob, rows["block.norm.weight"]) == gain_f16
    assert _slice(blob, rows["proj.bias"]) == bias_f32
    assert _slice(blob, rows[EMBED_NAME]) == embed_f32
    assert _slice(blob, rows["feat_encoder.special_token"]) == (
        np.ascontiguousarray(_promoted(special), dtype=np.float32).tobytes()
    )
    axes = tuple(range(1, direction.ndim))
    norm = np.linalg.norm(direction, axis=axes, keepdims=True)
    fused = (direction * (scale / norm)).astype(np.float32)
    assert _slice(blob, rows["decoder.model.0.weight"]) == np.ascontiguousarray(fused).tobytes()
    assert _slice(blob, rows["decoder.model.0.bias"]) == vae_bias.tobytes()
    assert _slice(blob, rows["decoder.snake.alpha"]) == alpha.tobytes()
    assert weight_bytes(tmp_path) == (tmp_path / WEIGHTS_NAME).stat().st_size
    runtime = Runtime.__new__(Runtime)
    runtime.path = tmp_path
    assert runtime.working_set_bytes(None) == weight_bytes(tmp_path) + _SCRATCH


def test_packed_store_binds_spans_in_the_blob(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    raw = (root / WEIGHTS_NAME).read_bytes()
    filled = bytearray(len(raw))
    (root / "copy.bin").write_bytes(raw)
    read_into(root / "copy.bin", memoryview(filled), chunk=7)
    assert bytes(filled) == raw
    body = json.loads((root / PACK_NAME).read_text(encoding="utf-8"))
    handle = "blob"
    store = Store.bind(filled, body["tensors"], handle)
    try:
        table = _promoted(
            np.array([[0.1, 0.2, 0.3, 0.4], [-1.25, 0.5, 2.0, 0.0]], dtype=np.float32)
        )
        gathered = store.embed(np.array([1, 0]))
        assert np.array_equal(gathered, table[[1, 0]])
        resident = store.weight(None, "proj.weight")
        assert isinstance(resident, Resident)
        assert resident.handle is handle
        assert resident.offset == body["tensors"]["proj.weight"]["offset"]
        assert resident.rows == 3 and resident.cols == 4
        assert resident.rows_pad == 32 and resident.cols_pad == 256
        assert store.header["proj.weight"]["shape"] == [3, 4]
        gain = store.gain(None, "block.norm.weight")
        bias = store.bias_handle(None, "proj.bias")
        assert isinstance(gain, Span) and isinstance(bias, Span)
        assert gain.handle is handle and bias.handle is handle
        assert _binding(gain) == (handle, gain.offset)
        assert _binding(resident) == (handle, resident.offset)
        special = store.vector("feat_encoder.special_token")
        assert special.shape == (4,)
        assert np.allclose(special, _promoted(np.array([0.1, -0.2, 0.3, 4.0], dtype=np.float32)))
    finally:
        store.close()


def test_packed_conv_does_not_release_the_blob(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    raw = bytearray((root / WEIGHTS_NAME).read_bytes())
    body = json.loads((root / PACK_NAME).read_text(encoding="utf-8"))
    handle = "blob"
    decoder = Decoder.from_records(
        _Fake(),
        raw,
        body["tensors"],
        handle,
        (2,),
        (20000,),
        48000,
    )
    weight = decoder._spans["decoder.model.0.weight"]
    bias = decoder._spans["decoder.model.0.bias"]
    bound, release = device_argument(_Boom(), weight)
    assert bound is weight
    assert release is False
    assert bound.handle is handle
    data = np.zeros((2, 1), dtype=np.float32)
    out = decoder._conv(data, "decoder.model.0", left=0, dilation=1, groups=1)
    assert out is data
    assert decoder.session.weight is weight
    assert decoder.session.bias is bias
    assert decoder.session.weight is not handle


def test_unpacked_working_set_counts_the_source_files(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"abcd")
    (tmp_path / "audiovae.pth").write_bytes(b"ef")
    assert weight_bytes(tmp_path) == 6
    runtime = Runtime.__new__(Runtime)
    runtime.path = tmp_path
    assert runtime.working_set_bytes(None) == 6 + _SCRATCH


def test_safetensors_embed_still_reads_bfloat16(tmp_path: Path) -> None:
    embed = np.array([[0.1, -1.5], [2.0, 0.25]], dtype=np.float32)
    path = tmp_path / "model.safetensors"
    _save_bf16(path, {EMBED_NAME: embed})
    store = Store(path)
    try:
        got = store.embed(np.array([1, 0]))
    finally:
        store.close()
    assert np.array_equal(got, _promoted(embed)[::-1])


def test_complete_accepts_the_blob_or_the_safetensors(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "config.json").write_text("{}", encoding="utf-8")
    try:
        mark_complete(bare, "bare")
    except ValueError as exc:
        assert "not a VoxCPM2 checkout" in str(exc)
    else:
        raise AssertionError("a tree with only config.json was marked complete")

    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "config.json").write_text("{}", encoding="utf-8")
    (sources / "model.safetensors").write_bytes(b"weights")
    mark_complete(sources, "sources")
    assert json.loads((sources / "manifest.json").read_text(encoding="utf-8"))["complete"] is True

    packed = tmp_path / "packed"
    packed.mkdir()
    (packed / "config.json").write_text("{}", encoding="utf-8")
    (packed / WEIGHTS_NAME).write_bytes(b"blob")
    mark_complete(packed, "packed")
    assert json.loads((packed / "manifest.json").read_text(encoding="utf-8"))["complete"] is True


def test_open_accepts_the_blob_without_safetensors(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    try:
        open_device(tmp_path)
    except FileNotFoundError as exc:
        assert "model.safetensors" in str(exc)
    else:
        raise AssertionError("an empty tree opened")

    packed = tmp_path / "packed"
    packed.mkdir()
    (packed / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
    (packed / WEIGHTS_NAME).write_bytes(b"blob")
    runtime = open_device(packed, resident_percent=None)
    assert runtime.path == packed
    assert not (packed / "model.safetensors").exists()


def _checkout(path: Path) -> Path:
    root = path / "checkout"
    root.mkdir()
    linear = np.array(
        [[0.1, -1.3, 2.2, 4.0], [0.5, 0.25, -0.75, 1.5], [3.0, -0.1, 0.2, 0.3]],
        dtype=np.float32,
    )
    _save_bf16(
        root / "model.safetensors",
        {
            "proj.weight": linear,
            "block.norm.weight": np.array([0.1, 1.5, -0.25], dtype=np.float32),
            EMBED_NAME: np.array(
                [[0.1, 0.2, 0.3, 0.4], [-1.25, 0.5, 2.0, 0.0]],
                dtype=np.float32,
            ),
            "proj.bias": np.array([0.1, -2.0, 0.5], dtype=np.float32),
            "feat_encoder.special_token": np.array([0.1, -0.2, 0.3, 4.0], dtype=np.float32).reshape(
                1, 1, 1, 4
            ),
        },
    )
    _save_vae(
        root / "audiovae.pth",
        {
            "decoder.model.0.weight_v": np.ones((2, 1, 3), dtype=np.float32),
            "decoder.model.0.weight_g": np.ones((2, 1, 1), dtype=np.float32),
            "decoder.model.0.bias": np.zeros(2, dtype=np.float32),
            "decoder.snake.alpha": np.ones(2, dtype=np.float32),
        },
    )
    write_pack(root)
    return root


def _promoted(value: np.ndarray) -> np.ndarray:
    raw = _bf16_payload(value)
    return _bf16_f32(raw).reshape(np.asarray(value).shape)


def _bf16_payload(value: np.ndarray) -> bytes:
    data = np.ascontiguousarray(value, dtype=np.float32)
    return (data.view(np.uint32) >> 16).astype("<u2").tobytes()


def _slice(blob: bytes, row: dict[str, object]) -> bytes:
    offset = int(row["offset"])
    return blob[offset : offset + int(row["nbytes"])]


def _save_bf16(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    cursor = 0
    for name, value in tensors.items():
        raw = _bf16_payload(value)
        header[name] = {
            "dtype": "BF16",
            "shape": list(np.asarray(value).shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        blobs.append(raw)
        cursor += len(raw)
    payload = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"".join(blobs))


class _Storage:
    def __init__(self, key: str, count: int) -> None:
        self.key = key
        self.count = count


class _Tensor:
    def __init__(self, key: str, array: np.ndarray) -> None:
        self.key = key
        self.array = array

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        shape = tuple(int(item) for item in self.array.shape)
        storage = _Storage(self.key, int(self.array.size))
        return (
            _rebuild_tensor_v2,
            (storage, 0, shape, _contiguous(shape), False, None, None),
        )


def _rebuild_tensor_v2(
    storage: object,
    offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: object,
    metadata: object = None,
) -> object:
    del storage, offset, size, stride, requires_grad, backward_hooks, metadata
    message = "AudioVAE dump helper"
    raise RuntimeError(message)


def _save_vae(path: Path, arrays: dict[str, np.ndarray]) -> None:
    storages: dict[str, bytes] = {}
    tensors = collections.OrderedDict()
    for index, (name, value) in enumerate(arrays.items()):
        key = str(index)
        data = np.ascontiguousarray(value, dtype="<f4")
        storages[key] = data.tobytes()
        tensors[name] = _Tensor(key, data)

    class _Pickler(pickle.Pickler):
        def persistent_id(self, obj: object) -> tuple[object, ...] | None:
            if isinstance(obj, _Storage):
                return ("storage", "FloatStorage", obj.key, "cpu", obj.count)
            return None

    buf = io.BytesIO()
    _Pickler(buf, protocol=4).dump({"state_dict": tensors})
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("vae/data.pkl", buf.getvalue())
        for key, raw in storages.items():
            archive.writestr(f"vae/data/{key}", raw)


class _Fake:
    def conv1d(
        self,
        source: np.ndarray,
        weight: object,
        bias: object,
        **kwargs: object,
    ) -> np.ndarray:
        del kwargs
        self.weight = weight
        self.bias = bias
        return source


class _Boom:
    def upload(self, raw: bytes) -> object:
        del raw
        message = "a packed span was uploaded"
        raise AssertionError(message)
