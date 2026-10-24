"""Pack an F16 GGUF into an int2 record.

Every matrix is int2 with an E8M0 scale every 32 along K. Norms, ``A_log``,
``dt_bias``, and the depthwise convolution stay float16. The file is GGUF.
A safetensors checkout is not a source.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from ternary_bonsai_2.formats import pack_int2_matrix, stored_rows
from ternary_bonsai_2.gguf import Gguf

RECORD_NAME = "Ternary-Bonsai-2-int2"
_WIDTHS = (5120, 6144, 17408)
_TOKEN_DIR = Path("models/Qwen3.8-27B-fp8-fp4")
_KEEP = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
)


def _align(value: int) -> int:
    return (value + 127) & ~127


def _append(path: Path, blob: bytes) -> int:
    with path.open("ab") as handle:
        end = handle.seek(0, 2)
        pad = _align(end) - end
        if pad:
            handle.write(b"\x00" * pad)
            end += pad
        handle.write(blob)
    return end


def _f16_entry(array: np.ndarray) -> tuple[bytes, dict]:
    flat = np.ascontiguousarray(array, dtype=np.float16)
    body = {"dtype": "f16", "shape": list(array.shape), "stored_shape": list(flat.shape)}
    return flat.tobytes(), body


def _int2_entry(array: np.ndarray) -> tuple[bytes, dict]:
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    logical = list(matrix.shape)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    rows = stored_rows(matrix.shape[0])
    data, scales, stored_r, stored_c = pack_int2_matrix(matrix, rows=rows)
    pad = _align(len(data)) - len(data)
    blob = data + (b"\x00" * pad) + scales
    body = {
        "dtype": "int2",
        "shape": logical,
        "stored_shape": [stored_r, stored_c],
        "data_nbytes": len(data),
        "scale_nbytes": len(scales),
        "scale_pad": pad,
    }
    return blob, body


def _keep_f16(name: str, array: np.ndarray) -> bool:
    if name.endswith("conv1d.weight") or name.endswith("A_log") or name.endswith("dt_bias"):
        return True
    if name.endswith(".bias") or array.ndim == 1:
        return True
    return False


def _language_name(ggml_name: str) -> str:
    """The record name for one GGUF tensor. Absent names are not invented."""
    if ggml_name == "token_embd.weight":
        return "model.language_model.embed_tokens.weight"
    if ggml_name == "output.weight":
        return "lm_head.weight"
    if ggml_name == "output_norm.weight":
        return "model.language_model.norm.weight"
    parts = ggml_name.split(".")
    if len(parts) < 3 or parts[0] != "blk":
        message = f"{ggml_name} is not a language tensor"
        raise ValueError(message)
    layer = parts[1]
    tail = ".".join(parts[2:])
    table = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight",
        "ssm_conv1d.weight": "linear_attn.conv1d.weight",
        "ssm_a": "linear_attn.A_log",
        "ssm_dt.bias": "linear_attn.dt_bias",
        "ssm_norm.weight": "linear_attn.norm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    rest = table.get(tail)
    if rest is None:
        message = f"{ggml_name} has no record name"
        raise ValueError(message)
    return f"model.language_model.layers.{layer}.{rest}"


def _layer_types(names: set[str]) -> list[str]:
    layers = sorted({int(name.split(".")[1]) for name in names if name.startswith("blk.")})
    kinds: list[str] = []
    for layer in layers:
        full = f"blk.{layer}.attn_q.weight" in names
        linear = f"blk.{layer}.attn_qkv.weight" in names
        if full == linear:
            message = f"layer {layer} is neither full attention nor Gated DeltaNet"
            raise ValueError(message)
        kinds.append("full_attention" if full else "linear_attention")
    return kinds


def _signs(gguf: Gguf) -> dict[str, np.ndarray]:
    values = gguf.metadata.get("prism.hadamard.sign_values")
    widths = gguf.metadata.get("prism.hadamard.sign_widths")
    if not isinstance(values, list) or not isinstance(widths, list):
        message = "the GGUF has no explicit Hadamard signs"
        raise ValueError(message)
    if [int(width) for width in widths] != list(_WIDTHS):
        message = f"Hadamard widths {widths} are not {_WIDTHS}"
        raise ValueError(message)
    cursor = 0
    out: dict[str, np.ndarray] = {}
    for width in _WIDTHS:
        chunk = values[cursor : cursor + width]
        if len(chunk) != width or any(int(item) not in (-1, 1) for item in chunk):
            message = f"Hadamard signs for {width} are not ±1"
            raise ValueError(message)
        out[f"hadamard.sign.{width}"] = np.asarray(chunk, dtype=np.float32)
        cursor += width
    if cursor != len(values):
        message = "Hadamard sign vector does not match its widths"
        raise ValueError(message)
    return out


def _copy_tokenizer(output: Path) -> None:
    if not _TOKEN_DIR.is_dir():
        return
    for name in _KEEP:
        src = _TOKEN_DIR / name
        target = output / name
        if src.is_file() and not target.exists():
            shutil.copyfile(src, target)


def quantize(src: Path, output: Path | None = None, *, ffn_fp8: float | None = None) -> Path:
    """Pack ``src``. ``src`` is the F16 GGUF, or a directory that contains it."""
    if ffn_fp8 not in (None, 0, 0.0):
        message = "this record is int2"
        raise ValueError(message)
    src = src.expanduser().resolve()
    if src.is_dir():
        matches = sorted(src.glob("*F16.gguf"))
        if len(matches) != 1:
            message = f"{src} does not contain one F16 GGUF"
            raise FileNotFoundError(message)
        src = matches[0]
    gguf = Gguf(src)
    if gguf.metadata.get("general.architecture") != "qwen35":
        message = f"{src.name} architecture is not qwen35"
        raise ValueError(message)
    if any(name.startswith("mtp") for name in gguf.tensors):
        message = f"{src.name} contains an MTP tensor"
        raise ValueError(message)
    kinds = _layer_types(set(gguf.tensors))
    if output is None:
        output = Path("models") / RECORD_NAME
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("complete"):
        return output
    progress_path = output / "quantize_progress.json"
    done: dict[str, dict] = {}
    if progress_path.is_file():
        done = json.loads(progress_path.read_text()).get("tensors", {})
    resident = output / "resident.bin"
    order = sorted(gguf.tensors, key=lambda name: gguf.tensors[name].offset)
    for ggml_name in order:
        name = _language_name(ggml_name)
        if name in done:
            continue
        array = gguf.array(ggml_name)
        print(f"  {name}", flush=True)
        if _keep_f16(name, array):
            blob, body = _f16_entry(array)
        else:
            blob, body = _int2_entry(array)
        offset = _append(resident, blob)
        done[name] = {**body, "file": "resident.bin", "offset": offset}
        del array
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"tensors": done}))
        os.replace(temporary, progress_path)
    for name, array in _signs(gguf).items():
        if name in done:
            continue
        blob, body = _f16_entry(array)
        offset = _append(resident, blob)
        done[name] = {**body, "file": "resident.bin", "offset": offset}
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"tensors": done}))
        os.replace(temporary, progress_path)
    hidden = int(gguf.metadata["qwen35.embedding_length"])
    manifest = {
        "complete": True,
        "pipeline": "ternary-bonsai-2",
        "source": src.name,
        "tensors": done,
        "kv_cache": "e4m3",
        "weight": "int2",
        "layers": len(kinds),
        "hidden": hidden,
        "vocab": int(gguf.tensors["token_embd.weight"].dims[1]),
        "layer_types": kinds,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest))
    os.replace(temporary, manifest_path)
    _copy_tokenizer(output)
    return output
