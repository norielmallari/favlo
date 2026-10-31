"""Checkpoint shapes for Gemma 4 26B-A4B.

The sizes are the safetensors header of
``google/gemma-4-26B-A4B-it-qat-q4_0-unquantized``. Global layers are the
``full_attention`` entries in that config.
"""

from __future__ import annotations

LAYERS = 30
EXPERTS = 128
GLOBAL_LAYERS = (5, 11, 17, 23, 29)
VISION_LAYERS = 27
HIDDEN = 2816
VOCAB = 262144
EXPERT_WIDTH = 704
DENSE_WIDTH = 2112
WINDOW = 1024
CONTEXT = 131072
CHECKPOINT_CONTEXT = 262144
LOCAL_HEADS = 8
LOCAL_DIM = 256
GLOBAL_HEADS = 2
GLOBAL_DIM = 512
VISION_WIDTH = 1152
VISION_MLP = 4304
VISION_HEAD = 72
RMS_EPS = 1e-6
SOFTCAP = 30.0
TOP_K = 8
Q_HEADS = 16
ROPE_PARTIAL = 0.25
ROPE_LOCAL = 10000.0
ROPE_GLOBAL = 1000000.0

Role = str

RESIDENT = "resident"
EXPERT_UP = "expert-up"
EXPERT_DOWN = "expert-down"
FP16 = "fp16"


def local_layers() -> tuple[int, ...]:
    skip = set(GLOBAL_LAYERS)
    return tuple(layer for layer in range(LAYERS) if layer not in skip)


def attention_shapes(layer: int) -> dict[str, tuple[int, ...]]:
    if layer in GLOBAL_LAYERS:
        return {
            "q_proj": (16 * GLOBAL_DIM, HIDDEN),
            "k_proj": (GLOBAL_HEADS * GLOBAL_DIM, HIDDEN),
            "o_proj": (HIDDEN, 16 * GLOBAL_DIM),
            "q_norm": (GLOBAL_DIM,),
            "k_norm": (GLOBAL_DIM,),
        }
    return {
        "q_proj": (16 * LOCAL_DIM, HIDDEN),
        "k_proj": (LOCAL_HEADS * LOCAL_DIM, HIDDEN),
        "v_proj": (LOCAL_HEADS * LOCAL_DIM, HIDDEN),
        "o_proj": (HIDDEN, 16 * LOCAL_DIM),
        "q_norm": (LOCAL_DIM,),
        "k_norm": (LOCAL_DIM,),
    }


def expected() -> dict[str, tuple[Role, tuple[int, ...]]]:
    """Every tensor in the text decoder and the vision tower."""
    found: dict[str, tuple[Role, tuple[int, ...]]] = {
        "model.language_model.embed_tokens.weight": (RESIDENT, (VOCAB, HIDDEN)),
        "model.language_model.norm.weight": (FP16, (HIDDEN,)),
        "model.embed_vision.embedding_projection.weight": (RESIDENT, (HIDDEN, VISION_WIDTH)),
        "model.vision_tower.patch_embedder.input_proj.weight": (
            RESIDENT,
            (VISION_WIDTH, 16 * 16 * 3),
        ),
        "model.vision_tower.patch_embedder.position_embedding_table": (
            FP16,
            (2, 10240, VISION_WIDTH),
        ),
        "model.vision_tower.std_bias": (FP16, (VISION_WIDTH,)),
        "model.vision_tower.std_scale": (FP16, (VISION_WIDTH,)),
    }
    for layer in range(LAYERS):
        base = f"model.language_model.layers.{layer}"
        for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "pre_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm.weight",
            "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight",
        ):
            found[f"{base}.{name}"] = (FP16, (HIDDEN,))
        found[f"{base}.layer_scalar"] = (FP16, (1,))
        found[f"{base}.router.proj.weight"] = (FP16, (EXPERTS, HIDDEN))
        found[f"{base}.router.scale"] = (FP16, (HIDDEN,))
        found[f"{base}.router.per_expert_scale"] = (FP16, (EXPERTS,))
        found[f"{base}.mlp.gate_proj.weight"] = (RESIDENT, (DENSE_WIDTH, HIDDEN))
        found[f"{base}.mlp.up_proj.weight"] = (RESIDENT, (DENSE_WIDTH, HIDDEN))
        found[f"{base}.mlp.down_proj.weight"] = (RESIDENT, (HIDDEN, DENSE_WIDTH))
        found[f"{base}.experts.gate_up_proj"] = (
            EXPERT_UP,
            (EXPERTS, EXPERT_WIDTH * 2, HIDDEN),
        )
        found[f"{base}.experts.down_proj"] = (
            EXPERT_DOWN,
            (EXPERTS, HIDDEN, EXPERT_WIDTH),
        )
        for name, shape in attention_shapes(layer).items():
            kind = FP16 if name.endswith("norm") else RESIDENT
            found[f"{base}.self_attn.{name}.weight"] = (kind, shape)
    for layer in range(VISION_LAYERS):
        base = f"model.vision_tower.encoder.layers.{layer}"
        for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            found[f"{base}.{name}"] = (FP16, (VISION_WIDTH,))
        found[f"{base}.self_attn.q_norm.weight"] = (FP16, (VISION_HEAD,))
        found[f"{base}.self_attn.k_norm.weight"] = (FP16, (VISION_HEAD,))
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            found[f"{base}.self_attn.{name}.linear.weight"] = (
                RESIDENT,
                (VISION_WIDTH, VISION_WIDTH),
            )
        found[f"{base}.mlp.gate_proj.linear.weight"] = (RESIDENT, (VISION_MLP, VISION_WIDTH))
        found[f"{base}.mlp.up_proj.linear.weight"] = (RESIDENT, (VISION_MLP, VISION_WIDTH))
        found[f"{base}.mlp.down_proj.linear.weight"] = (RESIDENT, (VISION_WIDTH, VISION_MLP))
    return found
