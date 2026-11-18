"""Raw weight blobs. Temp directories, no device."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wan22_i2v.metal41.blob import load_index, read_into, write_blob
from wan22_i2v.metal41.formats import LinearPack
from wan22_i2v.metal41.metal import BufferView
from wan22_i2v.metal41.record import load_half, load_linear, quantize_int8, write_graph_source
from wan22_i2v.metal41.runtime import expert_bytes


def test_plane_pad_is_not_nbytes(tmp_path: Path) -> None:
    first = np.arange(3, dtype=np.uint8)
    second = np.arange(4, dtype=np.float32)
    write_blob(tmp_path, [("a", first), ("b", second)])
    index = load_index(tmp_path)
    assert index is not None
    assert index["a"].nbytes == 3
    assert index["a"].offset == 0
    assert index["b"].offset == 128
    assert index["b"].nbytes == 16
    raw = (tmp_path / "weights.bin").read_bytes()
    assert raw[3:128] == b"\x00" * 125
    assert raw[128:144] == second.tobytes()


def test_read_into_uses_short_steps(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 4
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    dest = bytearray(len(payload))
    read_into(path, memoryview(dest), chunk=8)
    assert bytes(dest) == payload


def test_int8_expert_blob_copies_codes(tmp_path: Path) -> None:
    source = write_graph_source(tmp_path / "source")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    quantize_int8(
        source / "high.safetensors",
        source / "low.safetensors",
        tmp_path / "record",
        config,
    )
    high = tmp_path / "record" / "high"
    low = tmp_path / "record" / "low"
    assert (high / "weights.bin").is_file()
    assert (low / "weights.bin").is_file()
    index = load_index(high)
    assert index is not None
    name = "blocks.0.ffn.0.weight"
    packed = index[f"{name}.packed"]
    scales = index[f"{name}.scales"]
    assert packed.offset % 128 == 0
    assert scales.offset % 128 == 0
    assert scales.offset != packed.offset
    assert packed.kind == "int8"
    with np.load(high / f"{name}.npz") as blob:
        assert load_linear(high, name).packed.tobytes() == np.ascontiguousarray(blob["packed"]).tobytes()
        assert load_linear(high, name).scales.tobytes() == np.ascontiguousarray(blob["scales"]).tobytes()
    half_name = "head.head.weight"
    assert np.array_equal(load_half(high, half_name), np.load(high / f"{half_name}.npy"))
    high_blob = (high / "weights.bin").stat().st_size
    low_blob = (low / "weights.bin").stat().st_size
    assert expert_bytes(tmp_path / "record") == max(high_blob, low_blob)
    (high / "weights.bin").unlink()
    (high / "planes.json").unlink()
    assert load_linear(high, name).kind == "int8"


def test_upload_binds_a_view_without_a_second_copy() -> None:
    from wan22_i2v.metal41.dit import upload_half, upload_linear

    pack = LinearPack(
        kind="int8",
        packed=np.zeros((4, 8), dtype=np.int8),
        scales=np.ones(4, dtype=np.float32),
        bias=np.zeros(4, dtype=np.float16),
        perm=np.arange(8, dtype=np.uint32),
        logical_n=4,
        logical_k=8,
        padded_n=4,
        stored_k=8,
    )
    views = {
        "packed": BufferView("span", 0),
        "scales": BufferView("span", 128),
        "bias": BufferView("span", 256),
        "perm": BufferView("span", 384),
    }

    class _Gpu:
        def upload(self, array: np.ndarray) -> None:
            del array
            raise AssertionError("uploaded")

        def keep(self, buf: object) -> object:
            raise AssertionError("kept")

    slim, bound = upload_linear(_Gpu(), pack, views=views)  # type: ignore[arg-type]
    assert bound["scales"].offset == 128
    assert bound["scales"].base == "span"
    assert slim.packed.size == 0
    view = upload_half(_Gpu(), np.zeros(1, dtype=np.float16), view=BufferView("span", 64))  # type: ignore[arg-type]
    assert view.offset == 64


def test_expert_bytes_ignores_the_per_file_planes_when_the_blob_is_present(tmp_path: Path) -> None:
    folder = tmp_path / "high"
    folder.mkdir()
    (folder / "weights.bin").write_bytes(b"0123456789")
    (folder / "planes.json").write_text("[]", encoding="utf-8")
    (folder / "blocks.0.ffn.0.weight.npz").write_bytes(b"x" * 1000)
    assert expert_bytes(tmp_path, high=folder, low=tmp_path / "missing") == 10


def test_encoder_blob_binds_views_and_casts_half_positions(tmp_path: Path) -> None:
    from wan22_i2v.metal41.text import _FFN, _LAYERS, _WIDTH
    from wan22_i2v.metal41.umt5_pack import (
        _BlobSource,
        _FileSource,
        _open_source,
        _require_encoder,
        _stems,
    )

    planes: list[tuple[str, np.ndarray]] = [
        ("embed", np.arange(8, dtype=np.float16).reshape(4, 2)),
        ("norm", np.ones(4, dtype=np.float16)),
    ]
    for layer in range(_LAYERS):
        planes.append((f"blocks.{layer}.norm1", np.ones(4, dtype=np.float16)))
        planes.append((f"blocks.{layer}.norm2", np.ones(4, dtype=np.float16)))
        planes.append((f"blocks.{layer}.pos", np.arange(4, dtype=np.float16)))
        for stem in _stems(layer):
            planes.append((f"{stem}.packed", np.zeros((8, 256), dtype=np.int8)))
            planes.append((f"{stem}.scales", np.ones(8, dtype=np.float32)))
            width = 3 if stem.endswith("ffn.fc2") else 5
            planes.append((f"{stem}.bias", np.ones(width, dtype=np.float16)))
    write_blob(tmp_path, planes)
    index = load_index(tmp_path)
    assert index is not None
    _require_encoder(index)
    source = _BlobSource("span", index)
    pack, uploaded = source.linear(None, "blocks.0.ffn.fc2.weight")  # type: ignore[arg-type]
    assert pack.logical_k == _FFN
    assert pack.logical_n == 3
    assert pack.padded_n == 8
    assert pack.stored_k == 256
    assert isinstance(uploaded["packed"], BufferView)
    assert uploaded["packed"].base == "span"
    query, _ = source.linear(None, "blocks.0.attn.q.weight")  # type: ignore[arg-type]
    assert query.logical_k == _WIDTH
    assert isinstance(source.norm(None, "norm"), BufferView)  # type: ignore[arg-type]

    copied: list[np.ndarray] = []

    class _Gpu:
        def upload(self, array: np.ndarray) -> str:
            copied.append(array)
            return "copied"

    half = _BlobSource(
        "span",
        {
            "blocks.0.pos": index["blocks.0.pos"],
        },
    )

    def _rows(buf: object, plane: object) -> np.ndarray:
        del buf, plane
        return np.ones(4, dtype=np.float16)

    import wan22_i2v.metal41.umt5_pack as umt5_pack

    original = umt5_pack._span_array
    umt5_pack._span_array = _rows  # type: ignore[assignment]
    try:
        assert half.pos(_Gpu(), 0) == "copied"  # type: ignore[arg-type]
    finally:
        umt5_pack._span_array = original
    assert copied[0].dtype == np.float32

    wide = _BlobSource(
        "span",
        {
            "blocks.0.pos": type(index["blocks.0.pos"])(
                name="blocks.0.pos",
                dtype="float32",
                offset=128,
                nbytes=16,
                shape=(4,),
            )
        },
    )
    view = wide.pos(_Gpu(), 0)  # type: ignore[arg-type]
    assert isinstance(view, BufferView)
    assert view.offset == 128
    dropped: list[object] = []

    class _Drop:
        def drop(self, buf: object) -> None:
            dropped.append(buf)

    source.close(_Drop())  # type: ignore[arg-type]
    assert dropped == ["span"]
    assert isinstance(_open_source(tmp_path / "absent", None), _FileSource)  # type: ignore[arg-type]


def test_partial_encoder_blob_does_not_open_a_span(tmp_path: Path) -> None:
    from wan22_i2v.metal41.umt5_pack import _open_source

    write_blob(tmp_path, [("embed", np.zeros(4, dtype=np.float16))])
    with pytest.raises(KeyError, match="norm"):
        _open_source(tmp_path, None)  # type: ignore[arg-type]


def test_vae_blob_is_upload_order_and_skips_the_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from wan22_i2v.metal41.vae import iter_vae_planes, kernel_array, load_vae, write_vae_pack

    class _Weights(dict[str, np.ndarray]):
        def __contains__(self, key: object) -> bool:
            return isinstance(key, str)

        def __getitem__(self, key: str) -> np.ndarray:
            if key.endswith("bias") or key.endswith("gamma"):
                return np.arange(4, dtype=np.float32)
            if "resample" in key and key.endswith("weight"):
                return np.ones((2, 2, 3, 3), dtype=np.float32)
            if "shortcut" in key and key.endswith("weight"):
                return np.ones((4, 4), dtype=np.float32)
            if key.endswith("weight"):
                return np.ones((2, 2, 1, 3, 3), dtype=np.float32)
            raise KeyError(key)

    weights = _Weights()
    planes = iter_vae_planes(weights)
    names = [name for name, _, _ in planes]
    assert names[0] == "encoder.conv1.weight"
    assert names[1] == "encoder.conv1.bias"
    resample = "encoder.downsamples.5.resample.1.weight#t"
    later = "encoder.downsamples.5.time_conv.weight"
    assert names.index(resample) < names.index(later)
    resample_array = next(array for name, array, _ in planes if name == resample)
    assert resample_array.shape == (2, 2, 1, 3, 3)
    assert np.array_equal(
        resample_array, kernel_array(np.ones((2, 2, 3, 3), dtype=np.float32), time=True)
    )
    shortcut = next(array for name, array, _ in planes if name == "encoder.downsamples.0.shortcut.weight")
    assert shortcut.shape == (4, 4, 1, 1, 1)
    write_vae_pack(weights, tmp_path)

    def _boom(path: Path) -> dict[str, np.ndarray]:
        del path
        raise AssertionError("unpickle")

    monkeypatch.setattr("wan22_i2v.metal41.vae._unpickle_vae", _boom)
    loaded = load_vae(tmp_path / "Wan2.1_VAE.pth")
    assert loaded[resample].shape == (2, 2, 1, 3, 3)
    assert loaded["encoder.conv1.bias"].dtype == np.float32


def test_vae_without_a_blob_unpickles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from wan22_i2v.metal41.vae import load_vae

    seen: dict[str, Path] = {}

    def _fake(path: Path) -> dict[str, np.ndarray]:
        seen["path"] = path
        return {"conv1.weight": np.ones(2, dtype=np.float32)}

    monkeypatch.setattr("wan22_i2v.metal41.vae._unpickle_vae", _fake)
    path = tmp_path / "Wan2.1_VAE.pth"
    loaded = load_vae(path)
    assert seen["path"] == path
    assert loaded["conv1.weight"].shape == (2,)
