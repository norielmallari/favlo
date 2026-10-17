"""Language shapes this pipeline's shaders are written for.

The numbers are the Qwen3.8-27B text config: hidden 5120, 64 layers in a 3:1
mix, SwiGLU intermediate 17408, and the Gated DeltaNet and full-attention
widths in that card. ``AB`` is the stored row count of ``in_proj_a`` and
``in_proj_b`` after padding 48 up to the 64-row ``matmul2d`` tile.
"""

HIDDEN = 5120
LAYERS = 64
LINEAR_LAYERS = 48
FULL_LAYERS = 16
MTP_SLOT = FULL_LAYERS
VOCAB = 248320
INTERMEDIATE = 17408
Q_HEADS = 24
KV_HEADS = 4
HEAD = 256
GROUP = 6
LINEAR_Q = 16
LINEAR_V = 48
LINEAR_DIM = 128
MIX = 10240
Z_WIDTH = 6144
V_OFFSET = 4096
Q_PROJ = 12288
KV_PROJ = 1024
ATTN_OUT = 6144
AB = 64
AB_LOGICAL = 48
CONV = 4
CHUNK = 2048
EPS = 1e-6
THETA = 10_000_000.0
CONTEXT = 262144
