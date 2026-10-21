"""Device scratch for one image and a 1536-token caption.

``MAX_EDGE`` is 1280. The publisher's last training stage keeps each edge
within 256 pixels of the 1024 grid, from 768 to 1280, and a multiple of 16.
1280 is the largest square in that stage.
"""

from __future__ import annotations

from z_image_turbo.formats import TILE_M, pad_to
from z_image_turbo.metal import Gpu

MAX_EDGE = 1280
MAX_CAPTION = 1536
TEXT_HIDDEN = 2560
TEXT_Q = 4096
TEXT_FFN = 9728
DIT_HIDDEN = 3840
DIT_FFN = 10240


class Scratch:
    def __init__(self, gpu: Gpu) -> None:
        image = (MAX_EDGE // 16) ** 2
        self.cap_rows = pad_to(MAX_CAPTION, TILE_M)
        self.image_rows = pad_to(image, TILE_M)
        self.seq_rows = pad_to(MAX_CAPTION + image, TILE_M)
        self.text_h = gpu.keep(gpu.empty(self.cap_rows * TEXT_HIDDEN * 4))
        self.text_h2 = gpu.keep(gpu.empty(self.cap_rows * TEXT_HIDDEN * 4))
        self.text_q = gpu.keep(gpu.empty(self.cap_rows * TEXT_Q * 4))
        self.text_k = gpu.keep(gpu.empty(self.cap_rows * TEXT_Q * 4))
        self.text_v = gpu.keep(gpu.empty(self.cap_rows * TEXT_Q * 4))
        self.text_a = gpu.keep(gpu.empty(self.cap_rows * TEXT_Q * 4))
        self.text_ffn = gpu.keep(gpu.empty(self.cap_rows * TEXT_FFN * 4))
        self.text_ffn2 = gpu.keep(gpu.empty(self.cap_rows * TEXT_FFN * 4))
        self.text_out = gpu.keep(gpu.empty(self.cap_rows * TEXT_FFN * 4))
        self.cap = gpu.keep(gpu.empty(self.cap_rows * DIT_HIDDEN * 4))
        self.image = gpu.keep(gpu.empty(self.image_rows * DIT_HIDDEN * 4))
        self.image2 = gpu.keep(gpu.empty(self.image_rows * DIT_HIDDEN * 4))
        self.seq = gpu.keep(gpu.empty(self.seq_rows * DIT_HIDDEN * 4))
        self.seq2 = gpu.keep(gpu.empty(self.seq_rows * DIT_HIDDEN * 4))
        self.q = gpu.keep(gpu.empty(self.seq_rows * DIT_HIDDEN * 4))
        self.k = gpu.keep(gpu.empty(self.seq_rows * DIT_HIDDEN * 4))
        self.v = gpu.keep(gpu.empty(self.seq_rows * DIT_HIDDEN * 4))
        self.attn = gpu.keep(gpu.empty(self.seq_rows * DIT_HIDDEN * 4))
        self.ffn = gpu.keep(gpu.empty(self.seq_rows * DIT_FFN * 4))
        self.ffn2 = gpu.keep(gpu.empty(self.seq_rows * DIT_FFN * 4))
        self.out = gpu.keep(gpu.empty(self.seq_rows * DIT_FFN * 4))
        self.wide = gpu.keep(gpu.empty(64 * 15360 * 4))
        self.mod = gpu.keep(gpu.empty(64 * 15360 * 4))
        self.cond = gpu.keep(gpu.empty(64 * 1024 * 4))
        self.time = gpu.keep(gpu.empty(64 * 256 * 4))
        self.tokens = gpu.keep(gpu.empty(self.image_rows * 256 * 4))

    def release(self) -> None:
        """Drop every scratch buffer. The device then can free that memory."""
        for name in _BUFFERS:
            setattr(self, name, None)


_BUFFERS = (
    "text_h",
    "text_h2",
    "text_q",
    "text_k",
    "text_v",
    "text_a",
    "text_ffn",
    "text_ffn2",
    "text_out",
    "cap",
    "image",
    "image2",
    "seq",
    "seq2",
    "q",
    "k",
    "v",
    "attn",
    "ffn",
    "ffn2",
    "out",
    "wide",
    "mod",
    "cond",
    "time",
    "tokens",
)
