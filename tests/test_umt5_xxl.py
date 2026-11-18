"""umT5-XXL pack paths, int8 roundtrip, tokenizer copy, and encoder formats."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from umt5_xxl.metal41.formats import pack_int8, relative_error, restore_int8
from umt5_xxl.metal41.quantize import _copy_tokenizer, _drop_source, _write_planes, quantize
from umt5_xxl.metal41.source import ENCODER_FILE, TOKENIZER_FILE, pack_dir, source_dir
from umt5_xxl.metal41.weights import hf_key, open_weights


def test_pack_paths_live_under_models_google() -> None:
    assert source_dir().parts[-2:] == ("google", "umt5-xxl")
    assert pack_dir().parts[-2:] == ("google", "umt5-xxl-int8")


def test_pack_int8_restores_the_peak() -> None:
    weight = np.linspace(-2.0, 2.0, 128 * 64, dtype=np.float32).reshape(128, 64)
    pack = pack_int8(weight)
    restored = restore_int8(pack)
    assert pack.kind == "int8"
    assert pack.packed.dtype == np.int8
    assert restored.shape == weight.shape
    assert relative_error(restored, weight) < 0.01


def test_quantize_copies_the_tokenizer_into_the_pack(tmp_path: Path) -> None:
    source = tmp_path / "google" / "umt5_xxl"
    pack = tmp_path / "google" / "umt5-xxl-int8"
    source.mkdir(parents=True)
    pack.mkdir(parents=True)
    (source / TOKENIZER_FILE).write_text('{"version":"1.0"}', encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    _copy_tokenizer(source, pack)
    assert (pack / TOKENIZER_FILE).read_text(encoding="utf-8") == '{"version":"1.0"}'
    assert (pack / "tokenizer_config.json").read_text(encoding="utf-8") == "{}"
    assert (pack / "special_tokens_map.json").read_text(encoding="utf-8") == "{}"


def test_quantize_refuses_a_missing_tokenizer(tmp_path: Path) -> None:
    source = tmp_path / "src"
    pack = tmp_path / "pack"
    source.mkdir()
    pack.mkdir()
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        _copy_tokenizer(source, pack)


def test_quantize_removes_the_bf16_checkout(tmp_path: Path) -> None:
    source = tmp_path / "google" / "umt5_xxl"
    pack = tmp_path / "google" / "umt5-xxl-int8"
    source.mkdir(parents=True)
    pack.mkdir(parents=True)
    (source / ENCODER_FILE).write_bytes(b"bf16")
    (source / TOKENIZER_FILE).write_text("{}", encoding="utf-8")
    _drop_source(source, pack)
    assert not source.exists()
    assert pack.is_dir()


def test_quantize_keeps_the_pack_when_it_is_the_source(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / ENCODER_FILE).write_bytes(b"bf16")
    _drop_source(pack, pack)
    assert pack.is_dir()


def test_hf_key_maps_the_wan_names() -> None:
    assert hf_key("token_embedding.weight") == "shared.weight"
    assert hf_key("norm.weight") == "encoder.final_layer_norm.weight"
    assert hf_key("blocks.3.attn.q.weight") == (
        "encoder.block.3.layer.0.SelfAttention.q.weight"
    )
    assert hf_key("blocks.3.ffn.gate.0.weight") == (
        "encoder.block.3.layer.1.DenseReluDense.wi_0.weight"
    )
    assert hf_key("blocks.3.pos_embedding.embedding.weight") == (
        "encoder.block.3.layer.0.SelfAttention.relative_attention_bias.weight"
    )


def test_hf_safetensors_reads_wan_layout_names(tmp_path: Path) -> None:
    path = tmp_path / "encoder.safetensors"
    values = {
        "shared.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
        "encoder.final_layer_norm.weight": np.linspace(0.1, 0.4, 4, dtype=np.float32),
        "encoder.block.0.layer.0.SelfAttention.q.weight": np.eye(4, dtype=np.float32),
    }
    _write_bf16_safetensors(path, values)
    reader = open_weights("hf", path)
    try:
        assert reader.embed_shape() == (2, 4)
        assert np.allclose(reader.read_embed_rows(0, 2), values["shared.weight"], atol=1e-2)
        assert np.allclose(
            reader.tensor("norm.weight"),
            values["encoder.final_layer_norm.weight"],
            atol=1e-2,
        )
        assert np.allclose(
            reader.tensor("blocks.0.attn.q.weight"),
            values["encoder.block.0.layer.0.SelfAttention.q.weight"],
            atol=1e-2,
        )
    finally:
        reader.close()


def test_commands_quantize_defaults_to_wan_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from umt5_xxl.metal41.commands import main

    seen: dict[str, object] = {}

    def quantize(
        src: Path | None = None,
        dest: Path | None = None,
        *,
        encoder_format: str,
        tokenizer_src: Path | None = None,
        drop_source: bool = True,
    ) -> float:
        del src, tokenizer_src, drop_source
        seen["format"] = encoder_format
        assert dest is not None
        dest.mkdir(parents=True, exist_ok=True)
        (dest / TOKENIZER_FILE).write_text("{}", encoding="utf-8")
        return 0.0

    monkeypatch.setattr("umt5_xxl.metal41.source.models_dir", lambda root=None: tmp_path / "models")
    monkeypatch.setattr("umt5_xxl.metal41.quantize.quantize", quantize)
    assert main(["quantize"]) == 0
    assert seen["format"] == "wan"


def test_commands_quantize_writes_the_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from umt5_xxl.metal41.commands import main

    seen: dict[str, object] = {}

    def quantize(
        src: Path | None = None,
        dest: Path | None = None,
        *,
        encoder_format: str,
        tokenizer_src: Path | None = None,
        drop_source: bool = True,
    ) -> float:
        seen["src"] = src
        seen["dest"] = dest
        seen["format"] = encoder_format
        seen["tokenizer"] = tokenizer_src
        seen["drop"] = drop_source
        assert dest is not None
        dest.mkdir(parents=True, exist_ok=True)
        (dest / TOKENIZER_FILE).write_text("{}", encoding="utf-8")
        return 0.0123

    monkeypatch.setattr("umt5_xxl.metal41.source.models_dir", lambda root=None: tmp_path / "models")
    monkeypatch.setattr("umt5_xxl.metal41.quantize.quantize", quantize)
    packed = main(["quantize", "--format", "wan"])
    assert packed == 0
    assert seen["dest"] == tmp_path / "models" / "google" / "umt5-xxl-int8"
    assert seen["format"] == "wan"
    assert seen["drop"] is True
    kept = main(
        [
            "quantize",
            "--format",
            "hf",
            "--keep-source",
            "--src",
            str(tmp_path / "enc.safetensors"),
            "--out",
            str(tmp_path / "custom"),
            "--tokenizer",
            str(tmp_path / "tok"),
        ]
    )
    assert kept == 0
    assert seen["dest"] == tmp_path / "custom"
    assert seen["format"] == "hf"
    assert seen["src"] == tmp_path / "enc.safetensors"
    assert seen["tokenizer"] == tmp_path / "tok"
    assert seen["drop"] is False


def test_quantize_writes_aligned_planes_beside_the_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from umt5_xxl.metal41 import quantize as quantize_mod

    source = tmp_path / "src"
    pack = tmp_path / "pack"
    source.mkdir()
    encoder = source / "encoder.safetensors"
    _write_bf16_safetensors(encoder, _tiny_encoder())
    (source / TOKENIZER_FILE).write_text("{}", encoding="utf-8")
    quantize(
        src=encoder,
        dest=pack,
        encoder_format="hf",
        tokenizer_src=source,
        drop_source=False,
    )
    planes = json.loads((pack / "planes.json").read_text(encoding="utf-8"))
    blob = (pack / "weights.bin").read_bytes()
    assert [plane["name"] for plane in planes] == _plane_names()
    assert not blob.startswith(b"\x93NUMPY")
    assert not blob.startswith(b"PK")
    cursor = 0
    for plane in planes:
        assert list(plane) == ["name", "dtype", "offset", "nbytes", "shape"]
        offset = int(plane["offset"])
        nbytes = int(plane["nbytes"])
        assert offset % 128 == 0
        assert offset >= cursor
        assert blob[cursor:offset] == b"\x00" * (offset - cursor)
        assert blob[offset : offset + nbytes] == _stored_plane(pack, str(plane["name"]))
        assert nbytes == len(_stored_plane(pack, str(plane["name"])))
        cursor = offset + nbytes
    assert cursor == len(blob)
    assert (pack / "blocks.0.attn.o.weight.npz").is_file()
    assert (pack / "blocks.0.attn.q.weight.npz").is_file()
    assert (pack / TOKENIZER_FILE).is_file()
    assert (pack / "manifest.json").is_file()
    with np.load(pack / "blocks.0.attn.q.weight.npz") as stored:
        assert "perm" in stored.files
    by_name = {str(plane["name"]): plane for plane in planes}
    assert by_name["embed"]["dtype"] == "float16"
    assert by_name["blocks.0.norm1"]["dtype"] == "float16"
    assert by_name["blocks.0.attn.q.weight.packed"]["dtype"] == "int8"
    assert by_name["blocks.0.attn.q.weight.scales"]["dtype"] == "float32"
    assert by_name["blocks.0.attn.q.weight.bias"]["dtype"] == "float16"
    assert by_name["blocks.0.pos"]["dtype"] == "float16"
    assert by_name["blocks.0.norm2"]["dtype"] == "float16"
    assert by_name["norm"]["dtype"] == "float16"
    pos = np.load(pack / "blocks.0.pos.npy")
    assert pos.dtype == np.float16
    assert by_name["blocks.0.pos"]["nbytes"] == int(pos.nbytes)
    assert by_name["blocks.0.pos"]["shape"] == [int(dim) for dim in pos.shape]
    first_blob = blob
    first_planes = (pack / "planes.json").read_text(encoding="utf-8")

    def fail_pack(*_args, **_kwargs):
        raise AssertionError("codes were recomputed")

    monkeypatch.setattr(quantize_mod, "pack_int8", fail_pack)
    quantize(
        src=encoder,
        dest=pack,
        encoder_format="hf",
        tokenizer_src=source,
        drop_source=False,
    )
    assert (pack / "weights.bin").read_bytes() == first_blob
    assert (pack / "planes.json").read_text(encoding="utf-8") == first_planes


def test_raw_planes_leave_no_blob_when_a_file_is_missing(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    np.save(pack / "embed.npy", np.zeros((2, 2), dtype=np.float16))
    with pytest.raises(FileNotFoundError, match="norm1"):
        _write_planes(pack)
    assert {path.name for path in pack.iterdir()} == {"embed.npy"}


def _plane_names() -> list[str]:
    names = ["embed"]
    fields = ("packed", "scales", "bias")
    for layer in range(24):
        prefix = f"blocks.{layer}"
        names.append(f"{prefix}.norm1")
        for stem in ("attn.q", "attn.k", "attn.v", "attn.o"):
            names.extend(f"{prefix}.{stem}.weight.{field}" for field in fields)
        names.append(f"{prefix}.pos")
        names.append(f"{prefix}.norm2")
        for stem in ("ffn.gate.0", "ffn.fc1", "ffn.fc2"):
            names.extend(f"{prefix}.{stem}.weight.{field}" for field in fields)
    names.append("norm")
    return names


def _stored_plane(pack: Path, name: str) -> bytes:
    for field in ("packed", "scales", "bias"):
        suffix = f".{field}"
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            with np.load(pack / f"{stem}.npz") as loaded:
                return np.ascontiguousarray(loaded[field]).tobytes()
    array = np.load(pack / f"{name}.npy", mmap_mode="r")
    try:
        return np.ascontiguousarray(array).tobytes()
    finally:
        del array


def _tiny_encoder() -> dict[str, np.ndarray]:
    tensors = {
        "token_embedding.weight": np.arange(9, dtype=np.float32).reshape(3, 3),
        "norm.weight": np.linspace(0.1, 0.8, 8, dtype=np.float32),
    }
    for layer in range(24):
        prefix = f"blocks.{layer}"
        tensors[f"{prefix}.norm1.weight"] = np.full(8, 0.1 * (layer + 1), dtype=np.float32)
        tensors[f"{prefix}.norm2.weight"] = np.full(8, -0.05 * (layer + 1), dtype=np.float32)
        pos = np.arange(8, dtype=np.float32).reshape(2, 4) + layer
        tensors[f"{prefix}.pos_embedding.embedding.weight"] = pos
        for suffix in (
            "attn.q",
            "attn.k",
            "attn.v",
            "attn.o",
            "ffn.gate.0",
            "ffn.fc1",
            "ffn.fc2",
        ):
            matrix = np.linspace(-1.0, 1.0, 15, dtype=np.float32).reshape(3, 5)
            tensors[f"{prefix}.{suffix}.weight"] = matrix + layer
    return {hf_key(name): value for name, value in tensors.items()}


def _write_bf16_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    cursor = 0
    blobs: list[bytes] = []
    for name, array in tensors.items():
        flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        bits = (flat.view(np.uint32) >> np.uint32(16)).astype("<u2")
        raw = bits.tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(array.shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        blobs.append(raw)
        cursor += len(raw)
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for blob in blobs:
            handle.write(blob)
