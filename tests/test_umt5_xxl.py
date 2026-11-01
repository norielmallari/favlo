"""umT5-XXL pack paths, int8 roundtrip, and tokenizer copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from umt5_xxl.formats import pack_int8, relative_error, restore_int8
from umt5_xxl.quantize import _copy_tokenizer, _drop_source
from umt5_xxl.source import ENCODER_FILE, TOKENIZER_FILE, pack_dir, source_dir


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
    source = tmp_path / "google" / "umt5-xxl"
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
    source = tmp_path / "google" / "umt5-xxl"
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


def test_commands_quantize_writes_the_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from umt5_xxl.commands import main

    seen: dict[str, object] = {}

    def quantize(
        src: Path | None = None,
        dest: Path | None = None,
        *,
        tokenizer_src: Path | None = None,
        drop_source: bool = True,
    ) -> float:
        del tokenizer_src
        seen["src"] = src
        seen["dest"] = dest
        seen["drop"] = drop_source
        assert dest is not None
        dest.mkdir(parents=True, exist_ok=True)
        (dest / TOKENIZER_FILE).write_text("{}", encoding="utf-8")
        return 0.0123

    monkeypatch.setattr("umt5_xxl.source.models_dir", lambda root=None: tmp_path / "models")
    monkeypatch.setattr("umt5_xxl.quantize.quantize", quantize)
    packed = main(["quantize"])
    assert packed == 0
    assert seen["dest"] == tmp_path / "models" / "google" / "umt5-xxl-int8"
    assert seen["drop"] is True
    kept = main(["quantize", "--keep-source", "--out", str(tmp_path / "custom")])
    assert kept == 0
    assert seen["dest"] == tmp_path / "custom"
    assert seen["drop"] is False
