"""Shapes from the ZONOS2 report and params.json."""

from __future__ import annotations

LAYERS = 28
DIM = 2048
HEADS = 16
KV_HEADS = 4
HEAD_DIM = 128
FFN = 3072
EXPERTS = 16
MOE_START = 3
MOE_END = 27
TOP2_LAYER = 26
ROUTER_DIM = 128
CODEBOOKS = 9
CODEBOOK_SIZE = 1024
EOA_ID = 1024
PAD_ID = 1025
AUDIO_ROWS = PAD_ID + 1
TEXT_VOCAB = 519
TEXT_ROWS = 520
LEGACY_SYMBOLS = 192
BOS_ID = 2
EOS_ID = 3
QK_EPS = 1e-6
CONTEXT = 6144
ROPE_THETA = 10000.0
NORM_EPS = 1e-5
SOFTCAP = 15.0
SAMPLE_RATE = 44100
HOP = 512
SPEAKER_DIM = 2048
SPEAKER_LDA = 1024
FP8_PERCENT = 20.0
OUTPUT_ROWS = CODEBOOKS * AUDIO_ROWS
ROUTER_HIDDEN = 128
ROUTER_OUT = EXPERTS


def dense_layers() -> tuple[int, ...]:
    return tuple(layer for layer in range(LAYERS) if layer < MOE_START or layer >= MOE_END)


def moe_layers() -> tuple[int, ...]:
    return tuple(range(MOE_START, MOE_END))


def topk(layer: int) -> int:
    if layer == TOP2_LAYER:
        return 2
    if MOE_START <= layer < MOE_END:
        return 1
    return 0
