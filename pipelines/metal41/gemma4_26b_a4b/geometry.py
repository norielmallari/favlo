"""Checkpoint shapes for Gemma 4 26B-A4B.

The sizes are the safetensors header of
``google/gemma-4-26B-A4B-it``. Global layers are the
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
VISION_HEADS = 16
PATCH = 16
POOL = 3
VISION_SOFT_MAX = 280
IMAGE_TOKEN = 258880
VISION_ROPE = 100.0
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
HALF = "half"


def pooled_size(height: int, width: int, *, n_max: int = VISION_SOFT_MAX) -> tuple[int, int, int]:
    """Resized height, width, and soft-token count.

    Algorithm 1 of the Gemma 4 technical report. The pooled patch is ``POOL``
    times ``PATCH``. Each side rounds down to a multiple of that patch. A side
    that rounds to zero becomes one pooled patch, and the longer side shrinks
    until the count is at most ``n_max``.
    """
    if height < 1 or width < 1 or n_max < 1:
        message = f"image {height}x{width} cannot be resized for {n_max} soft tokens"
        raise ValueError(message)
    pooled = POOL * PATCH
    scale = (n_max * pooled * pooled / (height * width)) ** 0.5

    def snap(side: int) -> int:
        snapped = int((scale * side) // pooled) * pooled
        if snapped < pooled:
            return pooled
        return snapped

    target_h = snap(height)
    target_w = snap(width)
    while (target_h // pooled) * (target_w // pooled) > n_max:
        if target_h >= target_w and target_h > pooled:
            target_h -= pooled
        elif target_w > pooled:
            target_w -= pooled
        else:
            break
    soft = (target_h // pooled) * (target_w // pooled)
    return target_h, target_w, soft


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
        "model.language_model.norm.weight": (HALF, (HIDDEN,)),
        "model.embed_vision.embedding_projection.weight": (RESIDENT, (HIDDEN, VISION_WIDTH)),
        "model.vision_tower.patch_embedder.input_proj.weight": (
            RESIDENT,
            (VISION_WIDTH, 16 * 16 * 3),
        ),
        "model.vision_tower.patch_embedder.position_embedding_table": (
            HALF,
            (2, 10240, VISION_WIDTH),
        ),
        "model.vision_tower.std_bias": (HALF, (VISION_WIDTH,)),
        "model.vision_tower.std_scale": (HALF, (VISION_WIDTH,)),
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
            found[f"{base}.{name}"] = (HALF, (HIDDEN,))
        found[f"{base}.layer_scalar"] = (HALF, (1,))
        found[f"{base}.router.proj.weight"] = (HALF, (EXPERTS, HIDDEN))
        found[f"{base}.router.scale"] = (HALF, (HIDDEN,))
        found[f"{base}.router.per_expert_scale"] = (HALF, (EXPERTS,))
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
            kind = HALF if name.endswith("norm") else RESIDENT
            found[f"{base}.self_attn.{name}.weight"] = (kind, shape)
    for layer in range(VISION_LAYERS):
        base = f"model.vision_tower.encoder.layers.{layer}"
        for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            found[f"{base}.{name}"] = (HALF, (VISION_WIDTH,))
        found[f"{base}.self_attn.q_norm.weight"] = (HALF, (VISION_HEAD,))
        found[f"{base}.self_attn.k_norm.weight"] = (HALF, (VISION_HEAD,))
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            found[f"{base}.self_attn.{name}.linear.weight"] = (
                RESIDENT,
                (VISION_WIDTH, VISION_WIDTH),
            )
        found[f"{base}.mlp.gate_proj.linear.weight"] = (RESIDENT, (VISION_MLP, VISION_WIDTH))
        found[f"{base}.mlp.up_proj.linear.weight"] = (RESIDENT, (VISION_MLP, VISION_WIDTH))
        found[f"{base}.mlp.down_proj.linear.weight"] = (RESIDENT, (VISION_WIDTH, VISION_MLP))
    return found
