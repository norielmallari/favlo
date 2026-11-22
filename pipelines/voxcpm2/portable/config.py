"""Checkpoint config. Fields are the published ``config.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RopeScaling:
    long_factor: tuple[float, ...]
    short_factor: tuple[float, ...]
    original_max_position_embeddings: int


@dataclass(frozen=True)
class LmConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    kv_channels: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    max_position_embeddings: int
    use_mup: bool
    scale_emb: float
    scale_depth: float
    rope_scaling: RopeScaling


@dataclass(frozen=True)
class Tower:
    hidden: int
    ffn: int
    heads: int
    layers: int
    kv_channels: int


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int
    feat_dim: int
    fsq_dim: int
    fsq_scale: int
    residual_layers: int
    residual_no_rope: bool
    max_length: int
    cfg_rate: float
    mean_mode: bool
    decoder_rates: tuple[int, ...]
    sr_boundaries: tuple[int, ...]
    out_sample_rate: int
    encoder: Tower
    dit: Tower
    lm: LmConfig


def load_config(path: Path) -> ModelConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    lm = raw["lm_config"]
    rope = lm["rope_scaling"]
    encoder = raw.get("encoder_config") or {}
    dit = raw.get("dit_config") or {}
    cfm = dit.get("cfm_config") or {}
    audio = raw.get("audio_vae_config") or {}
    return ModelConfig(
        patch_size=int(raw["patch_size"]),
        feat_dim=int(raw["feat_dim"]),
        fsq_dim=int(raw["scalar_quantization_latent_dim"]),
        fsq_scale=int(raw["scalar_quantization_scale"]),
        residual_layers=int(raw["residual_lm_num_layers"]),
        residual_no_rope=bool(raw["residual_lm_no_rope"]),
        max_length=int(raw["max_length"]),
        cfg_rate=float(cfm.get("inference_cfg_rate", 2.0)),
        mean_mode=bool(dit.get("mean_mode", False)),
        decoder_rates=tuple(int(item) for item in audio.get("decoder_rates", [8, 6, 5, 2, 2, 2])),
        sr_boundaries=tuple(
            int(item) for item in audio.get("sr_bin_boundaries", [20000, 30000, 40000])
        ),
        out_sample_rate=int(audio.get("out_sample_rate", 48000)),
        encoder=Tower(
            hidden=int(encoder.get("hidden_dim", 1024)),
            ffn=int(encoder.get("ffn_dim", 4096)),
            heads=int(encoder.get("num_heads", 16)),
            layers=int(encoder.get("num_layers", 12)),
            kv_channels=int(encoder.get("kv_channels", 128)),
        ),
        dit=Tower(
            hidden=int(dit.get("hidden_dim", 1024)),
            ffn=int(dit.get("ffn_dim", 4096)),
            heads=int(dit.get("num_heads", 16)),
            layers=int(dit.get("num_layers", 12)),
            kv_channels=int(dit.get("kv_channels", 128)),
        ),
        lm=LmConfig(
            hidden_size=int(lm["hidden_size"]),
            intermediate_size=int(lm["intermediate_size"]),
            num_attention_heads=int(lm["num_attention_heads"]),
            num_hidden_layers=int(lm["num_hidden_layers"]),
            num_key_value_heads=int(lm["num_key_value_heads"]),
            kv_channels=int(lm["kv_channels"]),
            rms_norm_eps=float(lm["rms_norm_eps"]),
            rope_theta=float(lm["rope_theta"]),
            vocab_size=int(lm["vocab_size"]),
            max_position_embeddings=int(lm["max_position_embeddings"]),
            use_mup=bool(lm["use_mup"]),
            scale_emb=float(lm["scale_emb"]),
            scale_depth=float(lm["scale_depth"]),
            rope_scaling=RopeScaling(
                long_factor=tuple(float(item) for item in rope["long_factor"]),
                short_factor=tuple(float(item) for item in rope["short_factor"]),
                original_max_position_embeddings=int(rope["original_max_position_embeddings"]),
            ),
        ),
    )
