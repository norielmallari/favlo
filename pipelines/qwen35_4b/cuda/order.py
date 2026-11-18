"""Checkpoint tensor order for the CUDA pack. Shapes are the published config."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_LAYOUT = Path(__file__).resolve().parent / "layout.py"
_spec = importlib.util.spec_from_file_location("qwen35_cuda_layout_order", _LAYOUT)
if _spec is None or _spec.loader is None:
    message = f"cannot load {_LAYOUT}"
    raise ImportError(message)
_layout = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _layout
_spec.loader.exec_module(_layout)
HIDDEN = _layout.HIDDEN
INTERMEDIATE = _layout.INTERMEDIATE
LAYERS = _layout.LAYERS
MERGER_DIM = _layout.MERGER_DIM
PATCH_DIM = _layout.PATCH_DIM
VISION_DEPTH = _layout.VISION_DEPTH
VISION_HIDDEN = _layout.VISION_HIDDEN
VISION_INTERMEDIATE = _layout.VISION_INTERMEDIATE
VISION_OUT = _layout.VISION_OUT
VISION_POSITIONS = _layout.VISION_POSITIONS
VOCAB = _layout.VOCAB
is_full = _layout.is_full


def tensors(*, vision: bool) -> list[tuple[str, str, str, tuple[int, ...]]]:
    """(checkpoint name, logical name, kind, shape). ``kind`` is ``int`` or ``half``."""
    rows: list[tuple[str, str, str, tuple[int, ...]]] = [
        ("model.language_model.embed_tokens.weight", "embedding", "int", (VOCAB, HIDDEN)),
    ]
    for index in range(LAYERS):
        stem = f"model.language_model.layers.{index}."
        if is_full(index):
            rows.extend(
                (
                    (stem + "self_attn.q_proj.weight", f"layer.{index}.q", "int", (8192, HIDDEN)),
                    (stem + "self_attn.k_proj.weight", f"layer.{index}.k", "int", (1024, HIDDEN)),
                    (stem + "self_attn.v_proj.weight", f"layer.{index}.v", "int", (1024, HIDDEN)),
                    (stem + "self_attn.o_proj.weight", f"layer.{index}.o", "int", (HIDDEN, 4096)),
                    (stem + "self_attn.q_norm.weight", f"layer.{index}.q_norm", "half", (256,)),
                    (stem + "self_attn.k_norm.weight", f"layer.{index}.k_norm", "half", (256,)),
                )
            )
        else:
            rows.extend(
                (
                    (
                        stem + "linear_attn.in_proj_qkv.weight",
                        f"layer.{index}.qkv",
                        "int",
                        (8192, HIDDEN),
                    ),
                    (
                        stem + "linear_attn.in_proj_z.weight",
                        f"layer.{index}.z",
                        "int",
                        (4096, HIDDEN),
                    ),
                    (
                        stem + "linear_attn.in_proj_a.weight",
                        f"layer.{index}.a",
                        "int",
                        (32, HIDDEN),
                    ),
                    (
                        stem + "linear_attn.in_proj_b.weight",
                        f"layer.{index}.b",
                        "int",
                        (32, HIDDEN),
                    ),
                    (
                        stem + "linear_attn.out_proj.weight",
                        f"layer.{index}.out",
                        "int",
                        (HIDDEN, 4096),
                    ),
                    (
                        stem + "linear_attn.conv1d.weight",
                        f"layer.{index}.conv",
                        "half",
                        (8192, 1, 4),
                    ),
                    (stem + "linear_attn.A_log", f"layer.{index}.A_log", "half", (32,)),
                    (stem + "linear_attn.dt_bias", f"layer.{index}.dt_bias", "half", (32,)),
                    (stem + "linear_attn.norm.weight", f"layer.{index}.gdn_norm", "half", (128,)),
                )
            )
        rows.extend(
            (
                (
                    stem + "mlp.gate_proj.weight",
                    f"layer.{index}.gate",
                    "int",
                    (INTERMEDIATE, HIDDEN),
                ),
                (
                    stem + "mlp.up_proj.weight",
                    f"layer.{index}.up",
                    "int",
                    (INTERMEDIATE, HIDDEN),
                ),
                (
                    stem + "mlp.down_proj.weight",
                    f"layer.{index}.down",
                    "int",
                    (HIDDEN, INTERMEDIATE),
                ),
                (stem + "input_layernorm.weight", f"layer.{index}.in_norm", "half", (HIDDEN,)),
                (
                    stem + "post_attention_layernorm.weight",
                    f"layer.{index}.post_norm",
                    "half",
                    (HIDDEN,),
                ),
            )
        )
    rows.append(("model.language_model.norm.weight", "final_norm", "half", (HIDDEN,)))
    if vision:
        rows.extend(_vision())
    return rows


def _vision() -> list[tuple[str, str, str, tuple[int, ...]]]:
    root = "model.visual."
    rows: list[tuple[str, str, str, tuple[int, ...]]] = [
        (root + "patch_embed.proj.weight", "visual.patch", "int", (VISION_HIDDEN, PATCH_DIM)),
        (root + "patch_embed.proj.bias", "visual.patch_bias", "half", (VISION_HIDDEN,)),
        (root + "pos_embed.weight", "visual.pos", "int", (VISION_POSITIONS, VISION_HIDDEN)),
    ]
    for index in range(VISION_DEPTH):
        stem = f"{root}blocks.{index}."
        name = f"visual.{index}."
        rows.extend(
            (
                (stem + "norm1.weight", name + "norm1", "half", (VISION_HIDDEN,)),
                (stem + "norm1.bias", name + "norm1_bias", "half", (VISION_HIDDEN,)),
                (
                    stem + "attn.qkv.weight",
                    name + "qkv",
                    "int",
                    (VISION_HIDDEN * 3, VISION_HIDDEN),
                ),
                (stem + "attn.qkv.bias", name + "qkv_bias", "half", (VISION_HIDDEN * 3,)),
                (stem + "attn.proj.weight", name + "proj", "int", (VISION_HIDDEN, VISION_HIDDEN)),
                (stem + "attn.proj.bias", name + "proj_bias", "half", (VISION_HIDDEN,)),
                (stem + "norm2.weight", name + "norm2", "half", (VISION_HIDDEN,)),
                (stem + "norm2.bias", name + "norm2_bias", "half", (VISION_HIDDEN,)),
                (
                    stem + "mlp.linear_fc1.weight",
                    name + "fc1",
                    "int",
                    (VISION_INTERMEDIATE, VISION_HIDDEN),
                ),
                (stem + "mlp.linear_fc1.bias", name + "fc1_bias", "half", (VISION_INTERMEDIATE,)),
                (
                    stem + "mlp.linear_fc2.weight",
                    name + "fc2",
                    "int",
                    (VISION_HIDDEN, VISION_INTERMEDIATE),
                ),
                (stem + "mlp.linear_fc2.bias", name + "fc2_bias", "half", (VISION_HIDDEN,)),
            )
        )
    rows.extend(
        (
            (root + "merger.norm.weight", "visual.merger_norm", "half", (VISION_HIDDEN,)),
            (root + "merger.norm.bias", "visual.merger_norm_bias", "half", (VISION_HIDDEN,)),
            (
                root + "merger.linear_fc1.weight",
                "visual.merger_fc1",
                "int",
                (MERGER_DIM, MERGER_DIM),
            ),
            (root + "merger.linear_fc1.bias", "visual.merger_fc1_bias", "half", (MERGER_DIM,)),
            (
                root + "merger.linear_fc2.weight",
                "visual.merger_fc2",
                "int",
                (VISION_OUT, MERGER_DIM),
            ),
            (root + "merger.linear_fc2.bias", "visual.merger_fc2_bias", "half", (VISION_OUT,)),
        )
    )
    return rows
