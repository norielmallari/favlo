"""Load the packed record into resident matmul weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zonos2.layout import (
    DIM,
    EXPERTS,
    FFN,
    HEAD_DIM,
    HEADS,
    KV_HEADS,
    LAYERS,
    ROUTER_DIM,
    ROUTER_OUT,
    SPEAKER_DIM,
    SPEAKER_LDA,
)
from zonos2.session import Session, Weight


@dataclass
class Layer:
    attention_norm: np.ndarray
    wq: Weight
    wk: Weight
    wv: Weight
    wo: Weight
    gater: Weight
    alpha: np.ndarray
    ffn_norm: np.ndarray
    up: Weight | None
    gate: Weight | None
    down: Weight | None
    experts: tuple[tuple[Weight, Weight, Weight], ...] | None
    router_down: Weight | None
    router_down_bias: np.ndarray | None
    router_norm: np.ndarray | None
    router_mlp0: Weight | None
    router_mlp0_bias: np.ndarray | None
    router_mlp2: Weight | None
    router_mlp2_bias: np.ndarray | None
    router_mlp4: Weight | None
    router_scale: np.ndarray | None
    balancing: np.ndarray | None


@dataclass
class Record:
    layers: tuple[Layer, ...]
    text: np.ndarray
    audio: tuple[np.ndarray, ...]
    head: Weight
    out_norm: np.ndarray
    speaker_lda: Weight
    speaker_lda_bias: np.ndarray
    speaker: Weight
    speaker_bias: np.ndarray


def load_record(session: Session, path: Path) -> Record:
    weights = path / "weights"
    if not weights.is_dir():
        message = "zonos2 record has no weights"
        raise FileNotFoundError(message)
    layers = tuple(_layer(session, weights, index) for index in range(LAYERS))
    audio = tuple(_matrix(weights, f"multi_embedder.embedders.{book}.weight", 1026, DIM) for book in range(9))
    return Record(
        layers=layers,
        text=_matrix(weights, "multi_embedder.embedders.9.weight", 520, DIM),
        audio=audio,
        head=_quant(session, weights, "multi_output.weight", 9234, DIM),
        out_norm=_vector(weights, "out_norm.weight", DIM),
        speaker_lda=_half(session, weights, "speaker_lda_projection.weight", SPEAKER_LDA, SPEAKER_DIM),
        speaker_lda_bias=_vector(weights, "speaker_lda_projection.bias", SPEAKER_LDA),
        speaker=_half(session, weights, "speaker_projection.weight", SPEAKER_DIM, SPEAKER_LDA),
        speaker_bias=_vector(weights, "speaker_projection.bias", SPEAKER_DIM),
    )


def _layer(session: Session, weights: Path, index: int) -> Layer:
    prefix = f"layers.{index}"
    kv = KV_HEADS * HEAD_DIM
    experts = None
    up = gate = down = None
    router_down = router_norm = None
    router_mlp0 = router_mlp2 = router_mlp4 = None
    router_down_bias = router_mlp0_bias = router_mlp2_bias = None
    router_scale = balancing = None
    expert_dir = weights / f"{prefix}.expert.0.gate.fp4"
    expert_fp8 = weights / f"{prefix}.expert.0.gate.fp8"
    if expert_dir.is_file() or expert_fp8.is_file():
        experts = tuple(
            (
                _quant(session, weights, f"{prefix}.expert.{expert}.gate", FFN, DIM),
                _quant(session, weights, f"{prefix}.expert.{expert}.up", FFN, DIM),
                _quant(session, weights, f"{prefix}.expert.{expert}.down", DIM, FFN),
            )
            for expert in range(EXPERTS)
        )
        router = f"{prefix}.feed_forward.router"
        router_down = _half(session, weights, f"{router}.down_proj.weight", ROUTER_DIM, DIM)
        router_down_bias = _vector(weights, f"{router}.down_proj.bias", ROUTER_DIM)
        router_norm = _vector(weights, f"{router}.rmsnorm_eda.weight", ROUTER_DIM)
        router_mlp0 = _half(session, weights, f"{router}.router_mlp.0.weight", ROUTER_DIM, ROUTER_DIM)
        router_mlp0_bias = _vector(weights, f"{router}.router_mlp.0.bias", ROUTER_DIM)
        router_mlp2 = _half(session, weights, f"{router}.router_mlp.2.weight", ROUTER_DIM, ROUTER_DIM)
        router_mlp2_bias = _vector(weights, f"{router}.router_mlp.2.bias", ROUTER_DIM)
        router_mlp4 = _half(session, weights, f"{router}.router_mlp.4.weight", ROUTER_OUT, ROUTER_DIM)
        balancing = _vector(weights, f"{router}.balancing_biases", ROUTER_OUT)
        scale = weights / f"{router}.router_states_scale.f16"
        if scale.is_file():
            router_scale = _vector(weights, f"{router}.router_states_scale", ROUTER_DIM)
    else:
        feed = f"{prefix}.feed_forward"
        up = _quant(session, weights, f"{feed}.w_in.weight.0", FFN, DIM)
        gate = _quant(session, weights, f"{feed}.w_in.weight.1", FFN, DIM)
        down = _quant(session, weights, f"{feed}.w_out.weight", DIM, FFN)
    return Layer(
        attention_norm=_vector(weights, f"{prefix}.attention_norm.weight", DIM),
        wq=_quant(session, weights, f"{prefix}.attention.wq.weight", DIM, DIM),
        wk=_quant(session, weights, f"{prefix}.attention.wkv.weight.0", kv, DIM),
        wv=_quant(session, weights, f"{prefix}.attention.wkv.weight.1", kv, DIM),
        wo=_quant(session, weights, f"{prefix}.attention.wo.weight", DIM, DIM),
        gater=_half(session, weights, f"{prefix}.attention.gater.weight", HEADS, DIM),
        alpha=_vector(weights, f"{prefix}.attention.temp", HEADS),
        ffn_norm=_vector(weights, f"{prefix}.ffn_norm.weight", DIM),
        up=up,
        gate=gate,
        down=down,
        experts=experts,
        router_down=router_down,
        router_down_bias=router_down_bias,
        router_norm=router_norm,
        router_mlp0=router_mlp0,
        router_mlp0_bias=router_mlp0_bias,
        router_mlp2=router_mlp2,
        router_mlp2_bias=router_mlp2_bias,
        router_mlp4=router_mlp4,
        router_scale=router_scale,
        balancing=balancing,
    )


def _quant(session: Session, weights: Path, name: str, rows: int, inner: int) -> Weight:
    if (weights / f"{name}.f16").is_file():
        return session.half(_matrix(weights, name, rows, inner))
    scales = _bytes(weights / f"{name}.scale").reshape(rows, inner // 32)
    fp8 = weights / f"{name}.fp8"
    if fp8.is_file():
        return session.fp8(_bytes(fp8).reshape(rows, inner), scales, inner)
    return session.fp4(_bytes(weights / f"{name}.fp4").reshape(rows, inner // 2), scales, inner)


def _half(session: Session, weights: Path, name: str, rows: int, inner: int) -> Weight:
    return session.half(_matrix(weights, name, rows, inner))


def _matrix(weights: Path, name: str, rows: int, inner: int) -> np.ndarray:
    return _raw(weights / f"{name}.f16").reshape(rows, inner)


def _vector(weights: Path, name: str, count: int) -> np.ndarray:
    return _raw(weights / f"{name}.f16").reshape(count).astype(np.float32)


def _raw(path: Path) -> np.ndarray:
    if not path.is_file():
        message = "a packed weight is missing"
        raise FileNotFoundError(message)
    return np.fromfile(path, dtype=np.float16)


def _bytes(path: Path) -> np.ndarray:
    if not path.is_file():
        message = "a packed weight is missing"
        raise FileNotFoundError(message)
    return np.fromfile(path, dtype=np.uint8)
