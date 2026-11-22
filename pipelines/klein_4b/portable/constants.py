"""Shapes and names for FLUX.2 [klein] 4B."""

from __future__ import annotations

REPO = "black-forest-labs/FLUX.2-klein-4B"
RECORD = "FLUX.2-klein-4B-int8"
FORMAT = "flux2-klein-4b-int8"

PROMPT_LENGTH = 512
TEXT_LAYERS = 27
TEXT_CAPTURE = (8, 17, 26)
TEXT_HIDDEN = 2560
TEXT_HEADS = 32
TEXT_KV_HEADS = 8
TEXT_HEAD_DIM = 128
TEXT_INTERMEDIATE = 9728
TEXT_THETA = 1_000_000.0
PAD_ID = 151643

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

VAE_GROUPS = 32
VAE_EPS = 1e-6
BN_EPS = 1e-4
NORM_EPS = 1e-6

MIN_SIDE = 16
MAX_SIDE = 2048
MAX_INPUT_SIDE = 1024
ALIGN = 16
DEFAULT_STEPS = 4
DEFAULT_SIDE = 1024
MAX_STEPS = 50
MAX_IMAGES = 4

GROUP = 128
TILE_M = 64
TILE_N = 128

WEIGHTS = {
    "transformer/diffusion_pytorch_model.safetensors": 7_751_109_744,
    "text_encoder/model-00001-of-00002.safetensors": 4_967_215_360,
    "text_encoder/model-00002-of-00002.safetensors": 3_077_766_632,
    "vae/diffusion_pytorch_model.safetensors": 168_120_878,
}

SCRATCH_BYTES = 1_500_000_000
