"""Sizes from the Inkling 2 Mini config. No weights are read here."""

from __future__ import annotations

HIDDEN = 2304
HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
D_REL = 16
REL_EXTENT = 1024
LAYERS = 24
EXPERTS = 128
TOP_K = 8
INTERMEDIATE = 1152
VOCAB = 201024
UNPADDED_VOCAB = 200058
EOS = 200006
WINDOW = 1024
ROUTE_SCALE = 8.0
GATE_SOFTCAP = 4.0
UP_SOFTCAP = 25.0
LOGIT_SOFTCAP = 30.0
Q_L2 = 22.627416997969522
REL_L2 = 8.0
EPS = 1e-6
FULL_LAYERS = frozenset({5, 11, 17, 23})
CORE_START = 6
CORE_END = 18
BLOCKS = ((0, 6), (6, 12), (12, 18), (18, 24))
NVFP4_LAYERS = 20
OUTER_SCALE_DENOMINATOR = 2688.0


def stored_k(inner: int) -> int:
    """fp4 rows are 128-byte groups, so K is a multiple of 256."""
    return ((int(inner) + 255) // 256) * 256


Q_DIM = HEADS * HEAD_DIM
KV_DIM = KV_HEADS * HEAD_DIM
REL_DIM = HEADS * D_REL
FUSED = INTERMEDIATE * 2
PAIR = HIDDEN * 2
