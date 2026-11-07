"""Pack the DiT. Self-attention, cross-attention, and the feed-forward are int8.

One float scale per output row. The VAE is not packed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from favlo.registry import load_registry, models_dir, register_quantized, save_registry
from wan22_ti2v_5b.activations import gelu_erf, gelu_tanh
from wan22_ti2v_5b.checkpoint import Archive, load_state
from wan22_ti2v_5b.quant import pack_int8, relative_rms
from wan22_ti2v_5b.spec import LAYERS

_FOLDER = "Wan2.2-TI2V-5B-int8"
_STALE = "Wan2.2-TI2V-5B-fp8-e50"
_SCORED = (
    "self_attn.q.weight",
    "self_attn.k.weight",
    "self_attn.v.weight",
    "self_attn.o.weight",
    "cross_attn.q.weight",
    "cross_attn.k.weight",
    "cross_attn.v.weight",
    "cross_attn.o.weight",
    "ffn.0.weight",
    "ffn.2.weight",
)
_PAIRS = (
    ("self_q", "self_attn.q"),
    ("self_k", "self_attn.k"),
    ("self_v", "self_attn.v"),
    ("self_o", "self_attn.o"),
    ("cross_q", "cross_attn.q"),
    ("cross_k", "cross_attn.k"),
    ("cross_v", "cross_attn.v"),
    ("cross_o", "cross_attn.o"),
    ("ffn0", "ffn.0"),
    ("ffn2", "ffn.2"),
)


def quantize(weights: Path, copy: str | None, root: Path | None = None) -> Path:
    if not weights.is_file():
        message = f"DiT checkpoint is missing: {weights.name}"
        raise FileNotFoundError(message)
    archive = load_state(weights)
    try:
        chosen = archive.choose(copy)
        print(f"state dict {chosen}", flush=True)
        _require_blocks(archive)
        directory = models_dir(root) / _FOLDER
        directory.mkdir(parents=True, exist_ok=True)
        gelu = _gelu_kind(archive)
        print(f"gelu {gelu}", flush=True)
        _save_float(archive, directory)
        worst = (0.0, "")
        for index in range(LAYERS):
            worst = _save_block(archive, directory / "blocks" / f"{index:02d}", index, worst)
            print(f"block {index} int8", flush=True)
        print(f"worst {worst[1]} {worst[0]:.6f}", flush=True)
        vae = "models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
        manifest = {
            "layers": LAYERS,
            "gelu": gelu,
            "quant": "int8",
            "vae": vae,
            "fps": 24,
        }
        text = json.dumps(manifest, indent=2) + "\n"
        (directory / "manifest.json").write_text(text, encoding="utf-8")
        (directory / "source.json").write_text(
            json.dumps({"weights": str(weights.resolve())}) + "\n",
            encoding="utf-8",
        )
        umt5 = models_dir(root) / "google" / "umt5-xxl-int8"
        _release_stale_alias(root)
        register_quantized(
            _FOLDER,
            source=weights.resolve(),
            alias="wan22-ti2v-5b",
            directory=directory,
            root=root,
            pipeline="wan22-ti2v-5b",
            runtime="metal41",
            parts={"umt5": umt5},
        )
        print(_FOLDER, flush=True)
        return directory
    finally:
        archive.close()


def _require_blocks(archive: Archive) -> None:
    missing = [name for name in _SCORED if f"blocks.0.{name}" not in set(archive.keys())]
    if not missing:
        return
    sample = [key for key in archive.keys() if key.startswith("blocks.0.")][:40]
    print("missing " + ", ".join(missing), flush=True)
    print("found " + ", ".join(sample), flush=True)
    message = "DiT state dict does not use the Wan block names"
    raise RuntimeError(message)


def _release_stale_alias(root: Path | None) -> None:
    data = load_registry(root)
    entry = data["models"].get(_STALE)
    if not isinstance(entry, dict):
        return
    directory = Path(str(entry.get("directory", "")))
    if directory.is_dir():
        return
    del data["models"][_STALE]
    save_registry(data, root)


def _gelu_kind(archive: Archive) -> str:
    weight = _matrix(archive, "blocks.0.ffn.0.weight")
    bias = _matrix(archive, "blocks.0.ffn.0.bias")
    row = np.random.default_rng(0).standard_normal((1, weight.shape[1])).astype(np.float32)
    pre = row @ weight.T + bias.reshape(1, -1)
    error = float(np.max(np.abs(gelu_tanh(pre) - gelu_erf(pre))))
    print(f"gelu tanh vs erf {error:.6f}", flush=True)
    if error > 0.5:
        return "erf"
    return "tanh"


def _save_float(archive: Archive, directory: Path) -> None:
    patch = _matrix(archive, "patch_embedding.weight")
    patch = patch.reshape(patch.shape[0], -1)
    _save_linear(directory, "patch", patch, _matrix(archive, "patch_embedding.bias"))
    _save_named(archive, directory, "text0", "text_embedding.0")
    _save_named(archive, directory, "text2", "text_embedding.2")
    _save_named(archive, directory, "time0", "time_embedding.0")
    _save_named(archive, directory, "time2", "time_embedding.2")
    _save_linear(
        directory,
        "time_proj",
        _matrix(archive, "time_projection.1.weight"),
        _bias(archive, "time_projection.1.bias"),
    )
    _save_named(archive, directory, "head", "head.head")
    head_mod = _matrix(archive, "head.modulation").reshape(2, -1)
    np.save(directory / "head_mod.npy", head_mod.astype(np.float16))


def _save_block(
    archive: Archive,
    directory: Path,
    index: int,
    worst: tuple[float, str],
) -> tuple[float, str]:
    directory.mkdir(parents=True, exist_ok=True)
    mod = _matrix(archive, f"blocks.{index}.modulation").reshape(6, -1)
    np.save(directory / "modulation.npy", mod.astype(np.float16))
    norm3 = _matrix(archive, f"blocks.{index}.norm3.weight")
    norm3_bias = _matrix(archive, f"blocks.{index}.norm3.bias")
    np.save(directory / "norm3.npy", norm3.astype(np.float16))
    np.save(directory / "norm3.bias.npy", norm3_bias.astype(np.float16))
    for stem in ("self_attn", "cross_attn"):
        for side in ("q", "k"):
            weight = _matrix(archive, f"blocks.{index}.{stem}.norm_{side}.weight")
            name = "self" if stem == "self_attn" else "cross"
            np.save(directory / f"{name}_norm_{side}.npy", weight.astype(np.float16))
    for filename, stem in _PAIRS:
        weight = _matrix(archive, f"blocks.{index}.{stem}.weight")
        bias = _bias(archive, f"blocks.{index}.{stem}.bias")
        label = f"blocks.{index}.{stem}"
        error = _save_packed(directory / f"{filename}.npz", weight, bias)
        print(f"{label} {error:.6f}", flush=True)
        if error >= worst[0]:
            worst = (error, label)
    return worst


def _save_named(archive: Archive, directory: Path, name: str, stem: str) -> None:
    weight = _matrix(archive, f"{stem}.weight")
    _save_linear(directory, name, weight, _bias(archive, f"{stem}.bias"))


def _save_linear(directory: Path, name: str, weight: np.ndarray, bias: np.ndarray) -> None:
    np.save(directory / f"{name}.npy", np.ascontiguousarray(weight, dtype=np.float16))
    np.save(directory / f"{name}.bias.npy", np.ascontiguousarray(bias, dtype=np.float16))


def _save_packed(path: Path, weight: np.ndarray, bias: np.ndarray) -> float:
    if bias.shape[0] != weight.shape[0]:
        bias = np.zeros(weight.shape[0], dtype=np.float32)
    packed, scales, dequant = pack_int8(np.ascontiguousarray(weight, dtype=np.float32))
    np.savez(
        path,
        kind=np.array("int8"),
        packed=packed,
        scales=scales,
        bias=np.ascontiguousarray(bias, dtype=np.float16),
        logical_n=np.int32(weight.shape[0]),
        logical_k=np.int32(weight.shape[1]),
        padded_n=np.int32(packed.shape[0]),
        stored_k=np.int32(packed.shape[1]),
    )
    return relative_rms(weight, dequant)


def _matrix(archive: Archive, name: str) -> np.ndarray:
    return np.asarray(archive.tensor(name), dtype=np.float32)


def _bias(archive: Archive, name: str) -> np.ndarray:
    if name not in set(archive.keys()):
        return np.zeros(1, dtype=np.float32)
    return _matrix(archive, name).reshape(-1)
