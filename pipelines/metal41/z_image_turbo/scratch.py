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


def scratch_bytes(width: int, height: int) -> int:
    """Bytes of the scratch buffers for one image of this size."""
    return sum(_sizes(width, height).values())


def _sizes(width: int, height: int) -> dict[str, int]:
    image = (width // 16) * (height // 16)
    cap_rows = pad_to(MAX_CAPTION, TILE_M)
    image_rows = pad_to(image, TILE_M)
    seq_rows = pad_to(MAX_CAPTION + image, TILE_M)
    latent_n = 16 * (height // 8) * (width // 8)
    pos_bytes = seq_rows * 3 * 4
    return {
        "text_h": cap_rows * TEXT_HIDDEN * 4,
        "text_h2": cap_rows * TEXT_HIDDEN * 4,
        "text_q": cap_rows * TEXT_Q * 4,
        "text_k": cap_rows * TEXT_Q * 4,
        "text_v": cap_rows * TEXT_Q * 4,
        "text_a": cap_rows * TEXT_Q * 4,
        "text_ffn": cap_rows * TEXT_FFN * 4,
        "text_ffn2": cap_rows * TEXT_FFN * 4,
        "text_out": cap_rows * TEXT_FFN * 4,
        "cap": cap_rows * DIT_HIDDEN * 4,
        "image": image_rows * DIT_HIDDEN * 4,
        "image2": image_rows * DIT_HIDDEN * 4,
        "seq": seq_rows * DIT_HIDDEN * 4,
        "seq2": seq_rows * DIT_HIDDEN * 4,
        "q": seq_rows * DIT_HIDDEN * 4,
        "k": seq_rows * DIT_HIDDEN * 4,
        "v": seq_rows * DIT_HIDDEN * 4,
        "attn": seq_rows * DIT_HIDDEN * 4,
        "ffn": seq_rows * DIT_FFN * 4,
        "ffn2": seq_rows * DIT_FFN * 4,
        "out": seq_rows * DIT_FFN * 4,
        "wide": 64 * 15360 * 4,
        "mod": 64 * 15360 * 4,
        "cond": 64 * 1024 * 4,
        "time": 64 * 256 * 4,
        "tokens": image_rows * 256 * 4,
        "latent": latent_n * 4,
        "vel": latent_n * 4,
        "pos_noise": pos_bytes,
        "pos_text": pos_bytes,
        "pos_main": pos_bytes,
    }


class Scratch:
    def __init__(self, gpu: Gpu, width: int = MAX_EDGE, height: int = MAX_EDGE) -> None:
        sizes = _sizes(width, height)
        image = (width // 16) * (height // 16)
        self.width = width
        self.height = height
        self._gpu = gpu
        self.cap_rows = pad_to(MAX_CAPTION, TILE_M)
        self.image_rows = pad_to(image, TILE_M)
        self.seq_rows = pad_to(MAX_CAPTION + image, TILE_M)
        self.latent_count = 16 * (height // 8) * (width // 8)
        for name, nbytes in sizes.items():
            setattr(self, name, gpu.keep(gpu.empty(nbytes)))

    def covers(self, width: int, height: int) -> bool:
        """True when these buffers can hold an image of this size."""
        image = (width // 16) * (height // 16)
        latent = 16 * (height // 8) * (width // 8)
        return (
            self.image_rows >= pad_to(image, TILE_M)
            and self.seq_rows >= pad_to(MAX_CAPTION + image, TILE_M)
            and self.latent_count >= latent
        )

    def release(self) -> None:
        """Drop every scratch buffer. The device then can free that memory."""
        gpu = getattr(self, "_gpu", None)
        self._gpu = None
        for name in _BUFFERS:
            buf = getattr(self, name, None)
            setattr(self, name, None)
            forget = getattr(gpu, "forget", None)
            if buf is not None and callable(forget):
                forget(buf)


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
    "latent",
    "vel",
    "pos_noise",
    "pos_text",
    "pos_main",
)
