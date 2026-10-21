"""Shared projection. The matmul tile is 64 by 32."""

from __future__ import annotations

from typing import Any

from z_image_turbo.formats import TILE_N, pad_to
from z_image_turbo.metal import Gpu, linear
from z_image_turbo.record import Plane


def project(
    gpu: Gpu,
    act: Any,
    weight: Plane,
    bias: Plane | None,
    dest_f32: Any,
    dest_f16: Any,
    rows: int,
    inner: int,
) -> None:
    cols = int(weight.shape[0])
    if weight.stored_k != inner:
        message = f"{weight.name} stored K {weight.stored_k} != {inner}"
        raise ValueError(message)
    linear(
        gpu,
        act,
        weight.data,
        weight.scale,
        dest_f32,
        dest_f16,
        rows,
        cols,
        inner,
        None if bias is None else bias.data,
        rows,
        pad_to(cols, TILE_N),
    )
