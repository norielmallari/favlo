"""Euler grid, 320p geometry, and the Wan 2.2 I2V widths."""

from __future__ import annotations

import math

import numpy as np

FPS = 16
SHIFT = 5.0
STEPS = 4
MAX_STEPS = 40
BOUNDARY = 900.0
SAMPLE_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
SHORT_FRAMES = 5
FIVE_SEC_FRAMES = 81


def frames_from_duration(seconds: float) -> int:
    """Largest ``4n+1`` frame count that fits in ``seconds`` at ``FPS``."""
    if seconds <= 0:
        message = "duration must be positive"
        raise ValueError(message)
    counted = max(int(seconds * FPS), SHORT_FRAMES)
    return 1 + 4 * ((counted - 1) // 4)

P240_WIDTH = 416
P240_HEIGHT = 240
P320_WIDTH = 576
P320_HEIGHT = 320
P480_WIDTH = 832
P480_HEIGHT = 480
P720_WIDTH = 1280
P720_HEIGHT = 720
_BUCKETS = (
    ("240p", P240_WIDTH, P240_HEIGHT),
    ("320p", P320_WIDTH, P320_HEIGHT),
    ("480p", P480_WIDTH, P480_HEIGHT),
    ("720p", P720_WIDTH, P720_HEIGHT),
)
TEXT_LEN = 512
PROMOTE_PERCENT = 20

OFFICIAL = {
    "dim": 5120,
    "layers": 40,
    "heads": 40,
    "ffn": 13824,
    "text_len": TEXT_LEN,
    "text_dim": 4096,
    "in_dim": 36,
    "out_dim": 16,
    "freq_dim": 256,
    "eps": 1e-6,
    "boundary": BOUNDARY,
    "fps": FPS,
}

GRAPH = {
    "dim": 128,
    "layers": 1,
    "heads": 4,
    "ffn": 256,
    "text_len": 64,
    "text_dim": 64,
    "in_dim": 36,
    "out_dim": 16,
    "freq_dim": 64,
    "eps": 1e-6,
    "boundary": BOUNDARY,
    "fps": FPS,
}

LATENT_MEAN = np.array(
    [
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    ],
    dtype=np.float32,
)
LATENT_STD = np.array(
    [
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    ],
    dtype=np.float32,
)


def video_modes() -> tuple[str, ...]:
    """This pipeline is image-to-video only."""
    return ("i2v",)


def list_resolutions() -> list[dict[str, str]]:
    """Landscape and portrait sizes for each served bucket."""
    rows: list[dict[str, str]] = []
    for name, width, height in _BUCKETS:
        rows.append({"name": name, "orientation": "landscape", "size": f"{width}x{height}"})
        rows.append({"name": name, "orientation": "portrait", "size": f"{height}x{width}"})
    return rows


def sigmas(steps: int = STEPS, shift: float = SHIFT) -> np.ndarray:
    """Shifted flow knots from 1 to 0, including the terminal 0."""
    if steps < 1:
        message = f"steps must be positive, got {steps}"
        raise ValueError(message)
    base = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    return shifted.astype(np.float32)


def timesteps(steps: int = STEPS, shift: float = SHIFT) -> np.ndarray:
    """Network timesteps. The solver still subtracts the unit-interval knots."""
    return sigmas(steps, shift) * np.float32(1000.0)


def expert_name(timestep: float, boundary: float = BOUNDARY) -> str:
    """High-noise expert at and above the boundary, then the low-noise expert."""
    if timestep >= boundary:
        return "high"
    return "low"


def require_steps(steps: int) -> int:
    """Euler steps on the shifted grid. The distill default is four."""
    if steps < 1 or steps > MAX_STEPS:
        message = f"steps must be from 1 to {MAX_STEPS}, got {steps}"
        raise ValueError(message)
    return steps


def require_guidance(low: float, high: float) -> tuple[float, float]:
    """Low-noise scale, then the high-noise scale. Both 1 skip the uncond forward."""
    pair = (float(low), float(high))
    if not math.isfinite(pair[0]) or not math.isfinite(pair[1]):
        message = f"guidance scales must be finite, got {low}, {high}"
        raise ValueError(message)
    return pair


def guide_scale(timestep: float, low: float, high: float, boundary: float = BOUNDARY) -> float:
    """High-noise scale at and above the boundary, then the low-noise scale."""
    if timestep >= boundary:
        return high
    return low


def velocity_walks(
    times: np.ndarray, low: float, high: float, boundary: float = BOUNDARY
) -> int:
    """DiT walks. A step whose scale is 1 runs once. Any other scale runs twice."""
    walks = 0
    for timestep in times[:-1]:
        scale = guide_scale(float(timestep), low, high, boundary)
        walks += 1 if scale == 1.0 else 2
    return walks


def resolved_negative(text: str) -> str:
    """A blank negative is Wan's published sample string."""
    stripped = text.strip()
    if stripped:
        return stripped
    return SAMPLE_NEG_PROMPT


def require_geometry(frames: int, height: int, width: int) -> None:
    if frames < 1 or (frames - 1) % 4:
        message = f"frames must be 4n+1, got {frames}"
        raise ValueError(message)
    if height % 16 or width % 16 or height < 16 or width < 16:
        message = f"edges must be multiples of 16, got {width}x{height}"
        raise ValueError(message)


def geometry(frames: int, height: int, width: int) -> tuple[int, int, int, int]:
    """Latent time, height, width, and the patch-token count."""
    require_geometry(frames, height, width)
    lat_t = 1 + (frames - 1) // 4
    lat_h = height // 8
    lat_w = width // 8
    tokens = lat_t * (lat_h // 2) * (lat_w // 2)
    return lat_t, lat_h, lat_w, tokens


def rope_axes(head_dim: int) -> tuple[int, int, int]:
    """Even split of one head: time, then height, then width."""
    if head_dim % 2:
        message = f"head dim must be even, got {head_dim}"
        raise ValueError(message)
    unit = head_dim // 6
    time_axis = head_dim - 4 * unit
    height_axis = 2 * unit
    width_axis = head_dim - time_axis - height_axis
    return time_axis, height_axis, width_axis


def promote_count(layers: int, percent: int = PROMOTE_PERCENT) -> int:
    """How many blocks move to fp8. Ties are resolved by the caller."""
    if layers < 0 or percent < 0:
        message = f"layers and percent must be non-negative, got {layers}, {percent}"
        raise ValueError(message)
    return (percent * layers) // 100


def mask_latent(frames: int, *, end: bool = False) -> np.ndarray:
    """Four mask channels. The first latent time is the condition frame.

    ``end`` marks the last pixel as well. The first pixel is repeated four
    times, then the rest of the pixels are grouped by four. The last group
    of an ended clip is ``[0, 0, 0, 1]``.
    """
    lat_t = 1 + (frames - 1) // 4
    pixel = np.zeros(frames, dtype=np.float32)
    pixel[0] = np.float32(1)
    if end:
        pixel[-1] = np.float32(1)
    packed = np.concatenate((np.ones(4, dtype=np.float32), pixel[1:]))
    time_major = packed.reshape(lat_t, 4)
    return np.transpose(time_major, (1, 0))


def apply_split(
    frames: int, split: int | None, overlap: int = 1
) -> tuple[int, int | None]:
    """Return ``(frames, context)`` for ``split`` chunks that continue through ``overlap`` latents.

    ``None`` and ``1`` keep every frame and return no window. A larger split
    shortens the clip until the chunks tile it, then returns one chunk's length
    in ``4n+1`` pixel frames. The overlap is the finished tail of one chunk and
    the conditioned start of the next. ``overlap`` defaults to 1.
    """
    if split is None or split == 1:
        return frames, None
    if split < 1:
        message = f"split must be at least 1, got {split}"
        raise ValueError(message)
    if overlap < 1:
        message = f"overlap must be at least 1, got {overlap}"
        raise ValueError(message)
    lat = 1 + (frames - 1) // 4
    min_lat = split + overlap
    if lat < min_lat:
        need = 1 + 4 * (min_lat - 1)
        message = (
            f"split {split} with overlap {overlap} needs at least {need} frames, "
            f"got {frames}"
        )
        raise ValueError(message)
    while lat >= min_lat:
        if (lat - overlap) % split == 0:
            win = (lat + (split - 1) * overlap) // split
            if win > overlap:
                context = 1 + (win - 1) * 4
                plan = latent_windows(lat, context, overlap)
                if plan is not None and len(plan[1]) == split:
                    return 1 + (lat - 1) * 4, context
        lat -= 1
    message = f"split {split} with overlap {overlap} does not land on the clip"
    raise ValueError(message)


def windows(lat: int, win: int, overlap: int) -> list[int]:
    """Latent starts that cover ``lat`` frames with a window of ``win`` and ``overlap`` shared."""
    if win < 2 or win > lat:
        message = f"a window must fit in the clip, got {win} of {lat}"
        raise ValueError(message)
    if overlap < 1 or overlap >= win:
        message = f"overlap must be from 1 to {win - 1}, got {overlap}"
        raise ValueError(message)
    stride = win - overlap
    starts: list[int] = []
    start = 0
    while start + win <= lat:
        starts.append(start)
        start += stride
    last = lat - win
    if starts[-1] != last:
        starts.append(last)
    return starts


def latent_windows(lat: int, context: int | None, overlap: int) -> tuple[int, list[int]] | None:
    """``(window, starts)`` when ``context`` splits the clip. One call returns ``None``."""
    if context is None:
        return None
    win = 1 + (context - 1) // 4
    if win >= lat:
        return None
    return win, windows(lat, win, overlap)


def handoff_frames(overlap: int) -> int:
    """Pixel frames that re-encode into ``overlap`` latent slots.

    The first slot is one pixel. Each later slot is four. Encoding this many
    frames from scratch makes latent 0 a real first frame.
    """
    if overlap < 1:
        message = f"overlap must be at least 1, got {overlap}"
        raise ValueError(message)
    return 1 + (overlap - 1) * 4


def chunk_overlap(starts: list[int], win: int) -> int:
    """Latent tail one chunk hands to the next. One chunk shares nothing."""
    if len(starts) < 2:
        return 0
    share = win - (starts[1] - starts[0])
    if share < 1 or share >= win:
        message = f"overlap must sit inside the chunk, got {share} of {win}"
        raise ValueError(message)
    return share


def commit_span(start: int, win: int, overlap: int) -> tuple[int, int]:
    """``(write_at, count)`` for the latents a finished chunk owns.

    The first chunk writes its whole window. A later chunk leaves the
    conditioned tail in place and writes only the new latents after it.
    """
    if start < 0 or win < 1 or overlap < 0 or overlap >= win:
        message = f"a chunk needs a start, a length, and an overlap, got {start}, {win}, {overlap}"
        raise ValueError(message)
    if start == 0:
        return 0, win
    return start + overlap, win - overlap
