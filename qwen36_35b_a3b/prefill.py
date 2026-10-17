"""Chunked prompt prefill. Projections are fp4 matmul2d tiles of 64 rows.

Decode stays a GEMV. A chunk is at most 2048 tokens so a routed expert's
group is large enough for that tile. The Gated DeltaNet scan is one dispatch
per layer. Float attention is one dispatch per layer. fp8 and fp4 attention
scores and value mixes are ``matmul2d`` tiles. Expert groups are gathered,
multiplied, and scattered back onto the token rows.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from qwen36_35b_a3b.attn import attend_plane
from qwen36_35b_a3b.convert import GATE_UP_BYTES
from qwen36_35b_a3b.formats import EXPERT_IMAGE

CHUNK = 2048
HIDDEN = 2048
HEAD = 256
EPS = 1e-6
_GATE_SCALE = 1_048_576
_DOWN_SCALE = 524_288
_WAVE = 32
_MAX_PAD = 128


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)


def _write(buffer: Any, blob: bytes) -> None:
    if not blob:
        return
    memoryview(buffer.contents().as_buffer(len(blob)))[: len(blob)] = blob


class _Workspace:
    def __init__(self, session: Any) -> None:
        cap = CHUNK
        zeros = session.zeros
        self.hidden = zeros(cap * HIDDEN * 4)
        self.normed = zeros(cap * HIDDEN * 4)
        self.half = zeros(cap * 4096 * 2)
        self.wide = zeros(cap * 8192 * 4)
        self.z = zeros(cap * 4096 * 4)
        self.ab = zeros(cap * 32 * 4)
        self.ab2 = zeros(cap * 32 * 4)
        self.y = zeros(cap * 4096 * 4)
        self.query = zeros(cap * 4096 * 4)
        self.gate = zeros(cap * 4096 * 4)
        self.kv = zeros(cap * 512 * 4)
        self.vv = zeros(cap * 512 * 4)
        self.attn = zeros(cap * 4096 * 4)
        self.router = zeros(cap * 256 * 4)
        self.coeff = zeros(cap * 4)
        self.ids = zeros(cap * 8 * 4)
        self.mix = zeros(cap * 8 * 4)
        self.gate_up = zeros(cap * 1024 * 4)
        self.gated = zeros(cap * 512 * 4)
        self.down = zeros(cap * HIDDEN * 4)
        self.pack = zeros(cap * HIDDEN * 2)
        self.expert_pack = zeros(_WAVE * _MAX_PAD * HIDDEN * 2)
        self.expert_out = zeros(_WAVE * _MAX_PAD * HIDDEN * 4)
        self.expert_gated = zeros(_WAVE * _MAX_PAD * 512 * 4)
        # Every wave's jobs stay live until the command buffer commits.
        self.jobs = zeros(256 * (_MAX_PAD // 64) * 2 * 4 + 256)
        self.expert_of = zeros(8 * _WAVE * 4)
        self.slot_n = zeros(8 * _WAVE * 4)
        self.slot_csr = zeros(8 * _WAVE * 4)
        self.tokens = zeros(cap * 4)
        # Alignment padding between experts can add up to three slots each.
        self.csr_i = zeros((cap * 8 + 1024) * 4)
        self.csr_w = zeros((cap * 8 + 1024) * 4)
        self.iota = zeros(cap * 4)
        _write(self.iota, np.arange(cap, dtype=np.uint32).tobytes())


def workspace_bytes() -> int:
    """Bytes of the chunk workspace, reserved before the atlas is sized."""
    cap = CHUNK
    return (
        cap * HIDDEN * 4
        + cap * HIDDEN * 4
        + cap * 4096 * 2
        + cap * 8192 * 4
        + cap * 4096 * 4
        + cap * 32 * 4
        + cap * 32 * 4
        + cap * 4096 * 4
        + cap * 4096 * 4
        + cap * 4096 * 4
        + cap * 512 * 4
        + cap * 512 * 4
        + cap * 4096 * 4
        + cap * 256 * 4
        + cap * 4
        + cap * 8 * 4
        + cap * 8 * 4
        + cap * 1024 * 4
        + cap * 512 * 4
        + cap * HIDDEN * 4
        + cap * HIDDEN * 2
        + _WAVE * _MAX_PAD * HIDDEN * 2
        + _WAVE * _MAX_PAD * HIDDEN * 4
        + _WAVE * _MAX_PAD * 512 * 4
        + 256 * (_MAX_PAD // 64) * 2 * 4
        + 256
        + 8 * _WAVE * 4
        + 8 * _WAVE * 4
        + 8 * _WAVE * 4
        + cap * 4
        + (cap * 8 + 1024) * 4
        + (cap * 8 + 1024) * 4
        + cap * 4
    )


def prefill_ids(runtime: Any, ids: list[int], pos0: int = 0) -> None:
    """Advance state and the KV cache through ``ids``. The last prompt token is separate."""
    if not ids:
        return
    if pos0 + len(ids) >= runtime.max_pos:
        message = f"position {pos0 + len(ids)} is past {runtime.max_pos}"
        raise RuntimeError(message)
    workspace = getattr(runtime, "_prefill", None)
    if workspace is None:
        workspace = _Workspace(runtime.session)
        runtime._prefill = workspace
    cursor = pos0
    remaining = ids
    while remaining:
        take = min(CHUNK, len(remaining))
        _chunk(runtime, workspace, remaining[:take], cursor)
        cursor += take
        remaining = remaining[take:]


def _chunk(runtime: Any, work: _Workspace, ids: list[int], pos0: int) -> None:
    count = len(ids)
    rows = (count + 63) // 64 * 64
    _write(work.tokens, np.asarray(ids, dtype=np.uint32).tobytes())
    runtime.session.begin()
    _zero(runtime, work.hidden, rows * HIDDEN)
    _embed(runtime, work, count)
    runtime.session.end()
    for layer in range(40):
        runtime.session.begin()
        _zero(runtime, work.normed, rows * HIDDEN)
        _rms(runtime, work.hidden, _input_norm(layer), work.normed, count)
        if layer % 4 == 3:
            _attend(runtime, work, layer, count, rows, pos0)
        else:
            _gdn(runtime, work, layer, count, rows)
        _zero(runtime, work.normed, rows * HIDDEN)
        _rms(runtime, work.hidden, _post_norm(layer), work.normed, count)
        _route(runtime, work, layer, count, rows)
        runtime.session.end()
        _experts(runtime, work, layer, count, rows)


def _embed(runtime: Any, work: _Workspace, count: int) -> None:
    buf, offset, entry = runtime._loc("model.language_model.embed_tokens.weight")
    _rows, inner = entry["stored_shape"]
    runtime.session.encode(
        "embed_fp4_rows",
        [
            (buf, offset),
            (buf, runtime._scale(entry, offset)),
            (work.tokens, 0),
            (work.hidden, 0),
        ],
        [_u32(count), _u32(int(inner))],
        (count, 1, 1),
        (32, 1, 1),
    )


def _gdn(runtime: Any, work: _Workspace, layer: int, count: int, rows: int) -> None:
    base = f"model.language_model.layers.{layer}.linear_attn."
    _cast(runtime, work.normed, work.half, rows * HIDDEN)
    _matmul(runtime, work.half, base + "in_proj_qkv.weight", work.wide, rows)
    _matmul(runtime, work.half, base + "in_proj_z.weight", work.z, rows)
    _matmul(runtime, work.half, base + "in_proj_a.weight", work.ab, rows)
    _matmul(runtime, work.half, base + "in_proj_b.weight", work.ab2, rows)
    slot = layer - (layer // 4)
    hist = slot * 3 * 8192 * 2
    weight, woff, _entry = runtime._loc(base + "conv1d.weight")
    runtime.session.encode(
        "gdn_conv_rows",
        [(runtime.history, hist), (work.wide, 0), (weight, woff), (work.wide, 0)],
        [_u32(count), _u32(8192)],
        (8192, 1, 1),
        (1, 1, 1),
    )
    along, aoff, _entry = runtime._loc(base + "A_log")
    bias, boff, _entry = runtime._loc(base + "dt_bias")
    norm, noff, _entry = runtime._loc(base + "norm.weight")
    state = slot * 32 * 128 * 128 * 4
    _zero(runtime, work.y, rows * 4096)
    runtime.session.encode(
        "gdn_state_rows",
        [
            (runtime.state, state),
            (work.wide, 0),
            (work.ab, 0),
            (work.ab2, 0),
            (work.z, 0),
            (along, aoff),
            (bias, boff),
            (norm, noff),
            (work.y, 0),
        ],
        [_f32(EPS), _u32(count)],
        (32, 1, 1),
        (128, 1, 1),
    )
    _cast(runtime, work.y, work.half, rows * 4096)
    _matmul(runtime, work.half, base + "out_proj.weight", work.down, rows)
    _add(runtime, work, count)


def _attend(
    runtime: Any, work: _Workspace, layer: int, count: int, rows: int, pos0: int
) -> None:
    base = f"model.language_model.layers.{layer}.self_attn."
    _cast(runtime, work.normed, work.half, rows * HIDDEN)
    _matmul(runtime, work.half, base + "q_proj.weight", work.wide, rows)
    _matmul(runtime, work.half, base + "k_proj.weight", work.kv, rows)
    _matmul(runtime, work.half, base + "v_proj.weight", work.vv, rows)
    runtime.session.encode(
        "split_q_gate_rows",
        [(work.wide, 0), (work.query, 0), (work.gate, 0)],
        [_u32(count)],
        (count * 16, 1, 1),
        (256, 1, 1),
    )
    q_buf, q_off, _entry = runtime._loc(base + "q_norm.weight")
    k_buf, k_off, _entry = runtime._loc(base + "k_norm.weight")
    runtime.session.encode(
        "qk_norm_rope_rows",
        [(work.query, 0), (work.kv, 0), (q_buf, q_off), (k_buf, k_off)],
        [_f32(1e7), _f32(EPS), _u32(pos0)],
        (count, 18, 1),
        (HEAD, 1, 1),
    )
    slot = layer // 4
    _zero(runtime, work.attn, rows * 4096)
    if runtime.kv_cache == "f32":
        stride = runtime.max_pos * HEAD
        runtime.session.encode(
            "store_kv_rows",
            [
                (runtime.k_cache[slot], 0),
                (runtime.v_cache[slot], 0),
                (work.kv, 0),
                (work.vv, 0),
            ],
            [_u32(pos0), _u32(stride)],
            (count, 2, 1),
            (HEAD, 1, 1),
        )
        runtime.session.encode(
            "attend_prefill2",
            [
                (work.query, 0),
                (runtime.k_cache[slot], 0),
                (runtime.v_cache[slot], 0),
                (work.gate, 0),
                (work.attn, 0),
            ],
            [_u32(count), _u32(pos0), _u32(stride)],
            ((count + 1) // 2, 16, 1),
            (32, 1, 1),
        )
    else:
        attend_plane(
            runtime,
            query=work.query,
            key=work.kv,
            value=work.vv,
            gate=work.gate,
            dest=work.attn,
            pos0=pos0,
            count=count,
            slot=slot,
        )
    _cast(runtime, work.attn, work.half, rows * 4096)
    _matmul(runtime, work.half, base + "o_proj.weight", work.down, rows)
    _add(runtime, work, count)


def _route(runtime: Any, work: _Workspace, layer: int, count: int, rows: int) -> None:
    prefix = f"model.language_model.layers.{layer}.mlp."
    router, roff, entry = runtime._loc(prefix + "gate.weight")
    _rows, inner = entry["shape"]
    _cast(runtime, work.normed, work.half, rows * HIDDEN)
    runtime.session.encode(
        "prefill_f16",
        [(work.half, 0), (router, roff), (work.router, 0)],
        [_u32(rows), _u32(int(_rows)), _u32(int(inner))],
        (int(_rows) // 32, rows // 64, 1),
        _tile_threads(runtime),
    )
    coeff, coff, centry = runtime._loc(prefix + "shared_expert_gate.weight")
    _crows, cinner = centry["shape"]
    runtime.session.encode(
        "coeff_rows",
        [(work.normed, 0), (coeff, coff), (work.coeff, 0)],
        [_u32(count), _u32(int(cinner))],
        (count, 1, 1),
        (32, 1, 1),
    )
    runtime.session.encode(
        "router8_rows",
        [(work.router, 0), (work.ids, 0), (work.mix, 0)],
        [_u32(count)],
        (count, 1, 1),
        (1, 1, 1),
    )


def _experts(runtime: Any, work: _Workspace, layer: int, count: int, rows: int) -> None:
    chosen = np.frombuffer(
        runtime.session.download(work.ids, count * 8 * 4), dtype=np.uint32
    ).reshape(count, 8)
    mix = runtime.session.download_f32(work.mix, count * 8).reshape(count, 8)
    flat_e = np.ascontiguousarray(chosen).reshape(-1)
    flat_w = np.ascontiguousarray(mix).reshape(-1)
    flat_t = np.repeat(np.arange(count, dtype=np.uint32), 8)
    order = np.argsort(flat_e, kind="stable")
    flat_e = flat_e[order]
    flat_t = flat_t[order]
    flat_w = flat_w[order]
    counts = np.bincount(flat_e, minlength=256)
    offsets: list[int] = []
    chunks_i: list[np.ndarray] = []
    chunks_w: list[np.ndarray] = []
    cursor = 0
    begin = 0
    for expert in range(256):
        pad = (4 - cursor % 4) % 4
        if pad:
            chunks_i.append(np.zeros(pad, dtype=np.uint32))
            chunks_w.append(np.zeros(pad, dtype=np.float32))
            cursor += pad
        offsets.append(cursor)
        n = int(counts[expert])
        if n:
            chunks_i.append(flat_t[begin : begin + n])
            chunks_w.append(flat_w[begin : begin + n].astype(np.float32))
            cursor += n
        begin += n
    tokens_np = np.concatenate(chunks_i) if chunks_i else np.zeros(0, dtype=np.uint32)
    weights_np = np.concatenate(chunks_w) if chunks_w else np.zeros(0, dtype=np.float32)
    if tokens_np.size:
        _write(work.csr_i, tokens_np.tobytes())
        _write(work.csr_w, weights_np.tobytes())
    prefix = f"model.language_model.layers.{layer}.mlp."
    file = runtime.expert_buffer(layer)
    runtime.session.begin()
    _cast(runtime, work.normed, work.half, rows * HIDDEN)
    _matmul(runtime, work.half, prefix + "shared_expert.gate_up", work.gate_up, rows)
    runtime.session.encode(
        "silu_mul_rows",
        [(work.gate_up, 0), (work.gated, 0)],
        [_u32(rows)],
        (rows * 2, 1, 1),
        (256, 1, 1),
    )
    _cast(runtime, work.gated, work.half, rows * 512)
    _matmul(runtime, work.half, prefix + "shared_expert.down_proj.weight", work.down, rows)
    runtime.session.encode(
        "scatter_add_rows",
        [(work.hidden, 0), (work.down, 0), (work.iota, 0), (work.coeff, 0)],
        [_u32(count), _u32(HIDDEN)],
        (count, 1, 1),
        (256, 1, 1),
    )
    small: list[int] = []
    big: list[int] = []
    for expert in range(256):
        n = int(counts[expert])
        if n == 0:
            continue
        pad = ((n + 63) // 64) * 64
        (big if pad > _MAX_PAD else small).append(expert)
    waves = [small[start : start + _WAVE] for start in range(0, len(small), _WAVE)]
    job_words: list[int] = []
    owners = np.zeros((max(len(waves), 1), _WAVE), dtype=np.uint32)
    slot_n = np.zeros((max(len(waves), 1), _WAVE), dtype=np.uint32)
    slot_csr = np.zeros((max(len(waves), 1), _WAVE), dtype=np.uint32)
    wave_job_off: list[int] = []
    wave_njobs: list[int] = []
    for wave_index, wave in enumerate(waves):
        while len(job_words) % 4:
            job_words.append(0)
        wave_job_off.append(len(job_words) * 4)
        before = len(job_words)
        for slot, expert in enumerate(wave):
            owners[wave_index, slot] = expert
            slot_n[wave_index, slot] = int(counts[expert])
            slot_csr[wave_index, slot] = offsets[expert]
            pad = ((int(counts[expert]) + 63) // 64) * 64
            for tile in range(pad // 64):
                job_words.extend((slot, tile))
        wave_njobs.append((len(job_words) - before) // 2)
    if job_words:
        _write(work.jobs, np.asarray(job_words, dtype=np.uint32).tobytes())
        _write(work.expert_of, np.ascontiguousarray(owners).tobytes())
        _write(work.slot_n, np.ascontiguousarray(slot_n).tobytes())
        _write(work.slot_csr, np.ascontiguousarray(slot_csr).tobytes())
    for expert in big:
        _expert_one(runtime, work, file, expert, int(counts[expert]), offsets[expert])
    for wave_index, wave in enumerate(waves):
        n_slots = len(wave)
        meta = wave_index * _WAVE * 4
        _zero(runtime, work.expert_pack, n_slots * _MAX_PAD * (HIDDEN // 2))
        runtime.session.encode(
            "gather_half_slots",
            [
                (work.normed, 0),
                (work.csr_i, 0),
                (work.expert_pack, 0),
                (work.slot_n, meta),
                (work.slot_csr, meta),
            ],
            [_u32(_MAX_PAD), _u32(HIDDEN)],
            (_MAX_PAD, n_slots, 1),
            (256, 1, 1),
        )
        _expert_tiles(
            runtime,
            work,
            file,
            wave_njobs[wave_index],
            wave_job_off[wave_index],
            wave_index * _WAVE * 4,
            1024,
            HIDDEN,
            0,
            _GATE_SCALE,
        )
        span = n_slots * _MAX_PAD * 512
        runtime.session.encode(
            "silu_wave",
            [(work.expert_out, 0), (work.expert_gated, 0)],
            [_u32(_MAX_PAD)],
            (span // 256, 1, 1),
            (256, 1, 1),
        )
        _cast(runtime, work.expert_gated, work.expert_pack, span)
        _expert_tiles(
            runtime,
            work,
            file,
            wave_njobs[wave_index],
            wave_job_off[wave_index],
            wave_index * _WAVE * 4,
            HIDDEN,
            512,
            GATE_UP_BYTES,
            GATE_UP_BYTES + _DOWN_SCALE,
        )
        for slot, expert in enumerate(wave):
            runtime.session.encode(
                "scatter_add_rows",
                [
                    (work.hidden, 0),
                    (work.expert_out, slot * _MAX_PAD * HIDDEN * 4),
                    (work.csr_i, offsets[expert] * 4),
                    (work.csr_w, offsets[expert] * 4),
                ],
                [_u32(int(counts[expert])), _u32(HIDDEN)],
                (int(counts[expert]), 1, 1),
                (256, 1, 1),
            )
    runtime.session.end()


def _matmul(runtime: Any, half: Any, name: str, dest: Any, rows: int) -> None:
    buf, offset, entry = runtime._loc(name)
    cols, inner = (int(value) for value in entry["stored_shape"])
    _fp4(runtime, half, buf, offset, runtime._scale(entry, offset), dest, rows, cols, inner)


def _fp4(
    runtime: Any,
    half: Any,
    buf: Any,
    data_off: int,
    scale_off: int,
    dest: Any,
    rows: int,
    cols: int,
    inner: int,
) -> None:
    wide = cols % 64 == 0
    runtime.session.encode(
        "prefill_fp4_n64" if wide else "prefill_fp4_e8m0",
        [(half, 0), (buf, data_off), (buf, scale_off), (dest, 0)],
        [_u32(rows), _u32(cols), _u32(inner)],
        (cols // (64 if wide else 32), rows // 64, 1),
        _tile_threads(runtime),
    )


def _expert_tiles(
    runtime: Any,
    work: _Workspace,
    file: Any,
    n_jobs: int,
    job_off: int,
    owner_off: int,
    cols: int,
    inner: int,
    image_data: int,
    image_scale: int,
) -> None:
    runtime.session.encode(
        "prefill_fp4_expert_tile",
        [
            (work.expert_pack, 0),
            (file, 0),
            (work.expert_out, 0),
            (work.jobs, job_off),
            (work.expert_of, owner_off),
        ],
        [
            _u32(inner),
            _u32(cols),
            _u32(_MAX_PAD),
            _u32(image_data),
            _u32(image_scale),
            _u32(EXPERT_IMAGE),
        ],
        (cols // 64, n_jobs, 1),
        _tile_threads(runtime),
    )


def _expert_one(
    runtime: Any, work: _Workspace, file: Any, expert: int, n: int, start: int
) -> None:
    pad = ((n + 63) // 64) * 64
    _zero(runtime, work.pack, pad * (HIDDEN // 2))
    runtime.session.encode(
        "gather_half_rows",
        [(work.normed, 0), (work.csr_i, start * 4), (work.pack, 0)],
        [_u32(n), _u32(HIDDEN), _u32(HIDDEN)],
        (n, 1, 1),
        (256, 1, 1),
    )
    _fp4(
        runtime,
        work.pack,
        file,
        expert * EXPERT_IMAGE,
        expert * EXPERT_IMAGE + _GATE_SCALE,
        work.gate_up,
        pad,
        1024,
        HIDDEN,
    )
    runtime.session.encode(
        "silu_mul_rows",
        [(work.gate_up, 0), (work.gated, 0)],
        [_u32(pad)],
        (pad * 2, 1, 1),
        (256, 1, 1),
    )
    _cast(runtime, work.gated, work.pack, pad * 512)
    down = expert * EXPERT_IMAGE + GATE_UP_BYTES
    _fp4(
        runtime,
        work.pack,
        file,
        down,
        down + _DOWN_SCALE,
        work.down,
        pad,
        HIDDEN,
        512,
    )
    runtime.session.encode(
        "scatter_add_rows",
        [
            (work.hidden, 0),
            (work.down, 0),
            (work.csr_i, start * 4),
            (work.csr_w, start * 4),
        ],
        [_u32(n), _u32(HIDDEN)],
        (n, 1, 1),
        (256, 1, 1),
    )


def _zero_at(runtime: Any, buf: Any, byte_off: int, n_floats: int) -> None:
    runtime.session.encode(
        "zero_f32",
        [(buf, byte_off)],
        [],
        (n_floats // 256, 1, 1),
        (256, 1, 1),
    )


def _cast_at(
    runtime: Any, src: Any, src_off: int, dst: Any, dst_off: int, n: int
) -> None:
    runtime.session.encode(
        "cast_f32_f16",
        [(src, src_off), (dst, dst_off)],
        [],
        (n // 256, 1, 1),
        (256, 1, 1),
    )


def _tile_threads(runtime: Any) -> tuple[int, int, int]:
    width = int(runtime.session.pipes["prefill_fp4_e8m0"].threadExecutionWidth())
    return (width * 4, 1, 1)


def _cast(runtime: Any, src: Any, dst: Any, n: int) -> None:
    runtime.session.encode(
        "cast_f32_f16",
        [(src, 0), (dst, 0)],
        [],
        (n // 256, 1, 1),
        (256, 1, 1),
    )


def _zero(runtime: Any, buf: Any, n: int) -> None:
    runtime.session.encode(
        "zero_f32",
        [(buf, 0)],
        [],
        (n // 256, 1, 1),
        (256, 1, 1),
    )


def _rms(runtime: Any, src: Any, name: str, dest: Any, rows: int) -> None:
    buf, offset, _entry = runtime._loc(name)
    runtime.session.encode(
        "rmsnorm_rows",
        [(src, 0), (buf, offset), (dest, 0)],
        [_u32(HIDDEN), _f32(EPS), _u32(rows)],
        (rows, 1, 1),
        (256, 1, 1),
    )


def _add(runtime: Any, work: _Workspace, count: int) -> None:
    runtime.session.encode(
        "residual_rows",
        [(work.hidden, 0), (work.down, 0), (work.hidden, 0)],
        [_u32(count * HIDDEN)],
        (count * 8, 1, 1),
        (256, 1, 1),
    )


def _input_norm(layer: int) -> str:
    return f"model.language_model.layers.{layer}.input_layernorm.weight"


def _post_norm(layer: int) -> str:
    return f"model.language_model.layers.{layer}.post_attention_layernorm.weight"
