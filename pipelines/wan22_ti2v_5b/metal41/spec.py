"""Published widths for Wan2.2 TI2V-5B.

The numbers are the base ``config.json`` and the Diffusers VAE config.
"""

from __future__ import annotations

DIM = 3072
FFN = 14336
HEADS = 24
HEAD = 128
LAYERS = 30
TEXT_LEN = 512
TEXT_DIM = 4096
FREQ_DIM = 256
IN_DIM = 48
PATCH = (1, 2, 2)
EPS = 1e-6
FPS = 24
ROPE_LEN = 1024
LATENT_CHANNELS = 48
FP16_MAX = 65504.0
TILE_M = 64
TILE_N = 32
TILE_N_I8 = 128

# Head-dim split from the published frequency construction: d - 4*(d//6), 2*(d//6), 2*(d//6).
ROPE_T = HEAD - 4 * (HEAD // 6)
ROPE_H = 2 * (HEAD // 6)
ROPE_W = 2 * (HEAD // 6)
