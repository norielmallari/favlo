"""Isolated int8 prefill kernel times for the Gemma 4 ceiling.

Each sample is one command buffer, then a wait. The first sample is discarded.
The figure is the median of the next four. Expert and attention rows that did
not change keep the times already in ``docs/gemma4-26b-a4b/ceilings.md``.
"""

from __future__ import annotations

import numpy as np

from gemma4_26b_a4b.planes import pack_matrix
from gemma4_26b_a4b.session import Session

_CHUNK = 1024
_HIDDEN = 2816


def run() -> None:
    session = Session("gemma4-26b-a4b:measure")
    try:
        _i8_rows(session)
    finally:
        session.close()


def _i8_rows(session: Session) -> None:
    shapes = (
        ("q_local", 4096, _HIDDEN, 25),
        ("kv_local", 2048, _HIDDEN, 50),
        ("o_local", _HIDDEN, 4096, 25),
        ("q_global", 8192, _HIDDEN, 5),
        ("kv_global", 1024, _HIDDEN, 5),
        ("o_global", _HIDDEN, 8192, 5),
        ("gate_up", 2112, _HIDDEN, 60),
        ("down", _HIDDEN, 2176, 30),
    )
    for name, cols, inner, count in shapes:
        weight = np.zeros((cols, inner), dtype=np.float32)
        weight[:, 0] = 0.25
        blob = pack_matrix(weight, "fp8")
        packed = np.frombuffer(blob, dtype=np.uint8)
        data = session.upload(bytes(packed[: cols * inner]))
        scales = session.upload(bytes(packed[cols * inner :]))
        stage = session.zeros(cols * inner + cols * 4)
        act_f = session.zeros(_CHUNK * (inner if inner <= 8192 else 8192) * 4)
        # Source width is the active K the forward passes (may be tighter than pad).
        active = 2112 if inner == 2176 else inner
        act_f = session.zeros(_CHUNK * active * 4)
        act_i8 = session.zeros(_CHUNK * inner)
        act_scale = session.zeros(_CHUNK * 4)
        dest = session.zeros(_CHUNK * cols * 4)
        width = int(session.pipes["prefill_i8_n64"].threadExecutionWidth())

        def restore(
            data=data,
            scales=scales,
            stage=stage,
            cols=cols,
            inner=inner,
        ) -> None:
            session.encode(
                "restore_fp8_i8",
                [data, scales, stage, stage],
                [session._u32(cols), session._u32(inner)],
                (cols, 1, 1),
                (256, 1, 1),
                origins=[0, 0, 0, cols * inner],
            )

        def quantize(
            act_f=act_f,
            act_i8=act_i8,
            act_scale=act_scale,
            active=active,
            inner=inner,
        ) -> None:
            session.encode(
                "quantize_float_i8",
                [act_f, act_i8, act_scale],
                [session._u32(_CHUNK), session._u32(active), session._u32(inner)],
                (_CHUNK, 1, 1),
                (256, 1, 1),
            )

        def tile(
            act_i8=act_i8,
            stage=stage,
            dest=dest,
            cols=cols,
            inner=inner,
            width=width,
        ) -> None:
            session.encode(
                "prefill_i8_n64",
                [act_i8, stage, dest],
                [session._u32(_CHUNK), session._u32(cols), session._u32(inner)],
                (cols // 64, _CHUNK // 64, 1),
                (width * 4, 1, 1),
            )

        def scale(
            dest=dest,
            act_scale=act_scale,
            stage=stage,
            cols=cols,
            inner=inner,
        ) -> None:
            n = _CHUNK * cols
            session.encode(
                "scale_i8_f32",
                [dest, act_scale, stage],
                [session._u32(_CHUNK), session._u32(cols)],
                ((n + 255) // 256, 1, 1),
                (256, 1, 1),
                origins=[0, 0, cols * inner],
            )

        restore_us = session.measure_us(restore)
        quant_us = session.measure_us(quantize)
        tile_us = session.measure_us(tile)
        scale_us = session.measure_us(scale)
        total = restore_us + quant_us + tile_us + scale_us
        print(
            f"i8 {name} count={count} {_CHUNK}x{cols}x{inner} "
            f"restore {restore_us:.1f} quantize_f32 {quant_us:.1f} "
            f"tile {tile_us:.1f} scale {scale_us:.1f} sum {total:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    run()
