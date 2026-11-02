"""Shapes from the Qwen3.5-4B ``config.json``. No checkpoint is read here."""

from __future__ import annotations

STEM = "qwen35-4b"
FOLDER = "Qwen3.5-4B-fp8"
REPO = "Qwen/Qwen3.5-4B"

HIDDEN = 2560
LAYERS = 32
INTERMEDIATE = 9216
VOCAB = 248_320
CONTEXT = 262_144
SERVED_CONTEXT = 131_072
CHUNK = 2048
BLOCK = 32
RMS_EPS = 1e-6

# Full attention (every 4th layer: 3, 7, ..., 31).
ATTN_HEADS = 16
KV_HEADS = 4
HEAD_DIM = 256
ROPE_DIM = 64
THETA = 10_000_000.0
ATTN_SCALE = HEAD_DIM**-0.5
MROPE_SECTION = (11, 11, 10)
FULL = tuple(range(3, LAYERS, 4))
FULL_SET = frozenset(FULL)

# Gated DeltaNet.
LINEAR_KEY_HEADS = 16
LINEAR_VALUE_HEADS = 32
LINEAR_HEAD_DIM = 128
CONV_KERNEL = 4
GDN_QKV = LINEAR_KEY_HEADS * LINEAR_HEAD_DIM * 2 + LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM
GDN_Z = LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM
GDN_STATE_BYTES = (
    (LAYERS - len(FULL)) * LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * 4
)

# Special token ids from the published config.
EOS = 248_044
IM_END = 248_046
THINK_START = 248_068
THINK_END = 248_069
VISION_START = 248_053
VISION_END = 248_054
IMAGE_TOKEN = 248_056
VIDEO_TOKEN = 248_057

# Vision tower. The Conv3d patch weight is stored as [hidden, channels * temporal * patch * patch].
VISION_DEPTH = 24
VISION_HIDDEN = 1024
VISION_INTERMEDIATE = 4096
VISION_HEADS = 16
VISION_OUT = HIDDEN
VISION_CHANNELS = 3
VISION_POSITIONS = 2304
PATCH = 16
SPATIAL_MERGE = 2
TEMPORAL_PATCH = 2
PATCH_DIM = VISION_CHANNELS * TEMPORAL_PATCH * PATCH * PATCH
MERGER_DIM = VISION_HIDDEN * SPATIAL_MERGE * SPATIAL_MERGE


def is_full(layer: int) -> bool:
    return layer in FULL_SET


def fp8_bytes(rows: int, inner: int) -> int:
    """One E4M3 byte per element, plus one E8M0 byte per 32 elements along K."""
    return rows * inner + rows * (inner // BLOCK)


def half_bytes(count: int) -> int:
    return count * 2


KV_MODES = ("half", "fp8_e4m3")


def kv_mode(name: str | None) -> str:
    """``half`` or ``fp8_e4m3``. Empty means Metal ``half``."""
    text = "half" if name is None or name == "" else str(name)
    if text not in KV_MODES:
        message = f"kv cache {name} is not half or fp8_e4m3"
        raise ValueError(message)
    return text


def kv_token_bytes(mode: str = "half") -> int:
    """Key and value planes of the eight full-attention layers, one token.

    ``half`` is two bytes per element. ``fp8_e4m3`` is one Metal
    ``metal_fp8_e4m3`` byte plus one E8M0 byte per 32 channels.
    """
    chosen = kv_mode(mode)
    if chosen == "half":
        per_plane = KV_HEADS * HEAD_DIM * 2
    else:
        per_plane = KV_HEADS * (HEAD_DIM + HEAD_DIM // BLOCK)
    return len(FULL) * 2 * per_plane


def cache_bytes(tokens: int, mode: str = "half") -> int:
    return kv_token_bytes(mode) * tokens
