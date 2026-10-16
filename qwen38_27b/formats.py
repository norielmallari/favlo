"""Shapes read from this checkpoint's ``config.json``. No other model's layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextShapes:
    hidden: int
    layers: int
    vocab: int
    intermediate: int
    full_heads: int
    kv_heads: int
    head_dim: int
    linear_q_heads: int
    linear_v_heads: int
    linear_head_dim: int
    conv_kernel: int
    rotary_pairs: int
    context: int
    rms_eps: float
    rope_theta: float
    mrope_section: tuple[int, int, int]
    layer_types: tuple[str, ...]
    output_gate: str
    eos: tuple[int, ...]


def text_shapes(config: dict) -> TextShapes:
    text = config["text_config"]
    rope = text["rope_parameters"]
    section = tuple(int(v) for v in rope["mrope_section"])
    if len(section) != 3:
        message = "mrope_section must be three widths"
        raise ValueError(message)
    layers = tuple(text["layer_types"])
    if len(layers) != int(text["num_hidden_layers"]):
        message = "layer_types length does not match num_hidden_layers"
        raise ValueError(message)
    return TextShapes(
        hidden=int(text["hidden_size"]),
        layers=int(text["num_hidden_layers"]),
        vocab=int(text["vocab_size"]),
        intermediate=int(text["intermediate_size"]),
        full_heads=int(text["num_attention_heads"]),
        kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        linear_q_heads=int(text["linear_num_key_heads"]),
        linear_v_heads=int(text["linear_num_value_heads"]),
        linear_head_dim=int(text["linear_key_head_dim"]),
        conv_kernel=int(text["linear_conv_kernel_dim"]),
        rotary_pairs=sum(section),
        context=int(text["max_position_embeddings"]),
        rms_eps=float(text["rms_norm_eps"]),
        rope_theta=float(rope["rope_theta"]),
        mrope_section=(section[0], section[1], section[2]),
        layer_types=layers,
        output_gate=str(text["output_gate_type"]),
        eos=(),
    )


def load_text_shapes(source: Path) -> TextShapes:
    config = json.loads((source / "config.json").read_text())
    shapes = text_shapes(config)
    generation = json.loads((source / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    if isinstance(eos, int):
        eos = (eos,)
    else:
        eos = tuple(int(v) for v in eos)
    return TextShapes(
        hidden=shapes.hidden,
        layers=shapes.layers,
        vocab=shapes.vocab,
        intermediate=shapes.intermediate,
        full_heads=shapes.full_heads,
        kv_heads=shapes.kv_heads,
        head_dim=shapes.head_dim,
        linear_q_heads=shapes.linear_q_heads,
        linear_v_heads=shapes.linear_v_heads,
        linear_head_dim=shapes.linear_head_dim,
        conv_kernel=shapes.conv_kernel,
        rotary_pairs=shapes.rotary_pairs,
        context=shapes.context,
        rms_eps=shapes.rms_eps,
        rope_theta=shapes.rope_theta,
        mrope_section=shapes.mrope_section,
        layer_types=shapes.layer_types,
        output_gate=shapes.output_gate,
        eos=eos,
    )
