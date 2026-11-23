"""Shapes and byte counts for the FLUX.2 [klein] 4B CUDA record."""

from __future__ import annotations

from pathlib import Path

REPO = "black-forest-labs/FLUX.2-klein-4B"
RECORD = "FLUX.2-klein-4B-int4"
SOURCE = "FLUX.2-klein-4B"
FORMAT = "flux2-klein-4b-int4"

PROMPT_LENGTH = 512
PAD_ID = 151643
TEXT_LAYERS = 27
TEXT_CAPTURE = (8, 17, 26)
TEXT_HIDDEN = 2560
TEXT_HEADS = 32
TEXT_KV_HEADS = 8
TEXT_HEAD_DIM = 128
TEXT_INTERMEDIATE = 9728
TEXT_THETA = 1_000_000.0
TEXT_EPS = 1e-6

DIT_DIM = 3072
DIT_HEADS = 24
DIT_HEAD_DIM = 128
DIT_DOUBLE = 5
DIT_SINGLE = 20
TEXT_JOINT = 7680
LATENT_CHANNELS = 128
DIT_THETA = 2000.0
DIT_AXES = (32, 32, 32, 32)
ROPE_SCALE = 10
NORM_EPS = 1e-6

VAE_GROUPS = 32
VAE_EPS = 1e-6
BN_EPS = 1e-4
VAE_CHANNELS = (128, 256, 512, 512)

MIN_SIDE = 16
MAX_SIDE = 2048
MAX_INPUT_SIDE = 1024
ALIGN = 16
DEFAULT_STEPS = 4
DEFAULT_SIDE = 1024
MAX_STEPS = 50
MAX_IMAGES = 4
PACK_ALIGN = 128

TEXT_RECORD = 1_561_187_328
DIT_RECORD = 1_941_851_648
VAE_RECORD = 42_486_342
CAPTION_BYTES = 512 * 7680 * 2

SOURCE_BYTES = {
    "transformer/diffusion_pytorch_model.safetensors": 7_751_109_744,
    "text_encoder/model-00001-of-00002.safetensors": 4_967_215_360,
    "text_encoder/model-00002-of-00002.safetensors": 3_077_766_632,
    "vae/diffusion_pytorch_model.safetensors": 168_120_878,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def models_dir() -> Path:
    return repo_root() / "models"


def source_dir() -> Path:
    return models_dir() / SOURCE


def record_dir() -> Path:
    return models_dir() / RECORD


def align_up(value: int, multiple: int = PACK_ALIGN) -> int:
    return (value + multiple - 1) // multiple * multiple


def grid_tokens(height: int, width: int) -> int:
    return (height // ALIGN) * (width // ALIGN)


def joint_sequence(target: int, refs: int) -> int:
    return PROMPT_LENGTH + target + refs


def denoise_bytes(dit: int, seq: int) -> int:
    """DiT weights, the joint residual, and one extra single-block projection."""
    planes = seq * (27648 + 3072) * 2
    extra = seq * 27648 * 2
    return dit + planes + extra + CAPTION_BYTES


def decode_bytes(vae: int, height: int, width: int) -> int:
    """Decoder weights plus the input volume and the output volume at full size."""
    volume = 128 * height * width * 4
    return vae + volume * 2


def opening_bytes(height: int, width: int, ref_tokens: int, stages: dict[str, int] | None = None) -> int:
    text = TEXT_RECORD
    dit = DIT_RECORD
    vae = VAE_RECORD
    if stages:
        text = int(stages.get("text", text))
        dit = int(stages.get("dit", dit))
        vae = int(stages.get("vae", vae))
    seq = joint_sequence(grid_tokens(height, width), ref_tokens)
    return max(text, denoise_bytes(dit, seq), decode_bytes(vae, height, width))
