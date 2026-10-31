"""Two-row language verify. Dense weights are read once for the token and its draft.

Gated DeltaNet and attention run in token order. Routed experts stay on the
token that selected them. The shared expert is one weight read. A rejected
draft copies the DeltaNet state and the convolution history back to the
confirm point, which is the state after the first token of the pair.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

HIDDEN = 2048
VOCAB = 248320
_STATE_WORDS = 30 * 32 * 128 * 128
_HISTORY_WORDS = 30 * 3 * 8192 * 2 // 4
_LAYERS = 40
_TOP_THREADS = 256
_TOP_CAP = 32
_PEN_CAP = 4096
_ARGMAX = (VOCAB + 255) // 256


class PairStep:
    """One paired verify. ``extra`` is the draft to append when it was kept.

    ``follow`` is the next argmax when temperature is 0. ``device`` means the
    bonus logits stay on ``span_logits`` for the next sample.
    """

    def __init__(
        self,
        accepted: bool,
        extra: int | None,
        carry: int | None,
        logits: np.ndarray | None,
        follow: int | None = None,
        device: bool = False,
    ) -> None:
        self.accepted = accepted
        self.extra = extra
        self.carry = carry
        self.logits = logits
        self.follow = follow
        self.device = device


def paired_step(
    runtime: Any,
    token: int,
    pos: int,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    seen: set[int],
    banned: set[int] | None,
    frequency: float,
    counts: dict[int, int] | None,
    stop: set[int],
) -> PairStep:
    """Draft one token at ``pos - 1`` and verify ``token`` together with that draft."""
    from qwen36_35b_a3b.speculate import candidate_probs, keep_draft, penalized

    runtime._drop_pending()
    host_draft: int | None = None
    q_idx = np.asarray([0], dtype=np.int64)
    q_pr = np.asarray([1.0], dtype=np.float64)
    greedy = temperature == 0.0
    if not greedy:
        runtime._draft_chain(int(token), int(pos) - 1, 1, snap=False)
        picked = _candidates(
            runtime,
            runtime.logits,
            temperature,
            top_p,
            top_k,
            presence,
            seen,
            banned,
            frequency,
            counts,
        )
        if picked is None:
            drafted = penalized(
                runtime.session.download_f32(runtime.logits, VOCAB),
                presence,
                seen,
                banned,
                frequency,
                counts,
            )
            q_idx, q_pr = candidate_probs(drafted, temperature, top_p, top_k)
        else:
            q_idx, q_pr = picked
        pick = 0 if q_idx.size == 1 else int(np.random.choice(q_idx.size, p=q_pr))
        host_draft = int(q_idx[pick])
    on_device = (not greedy) and 0 < int(top_k) <= _TOP_CAP
    runtime.session.begin()
    try:
        if host_draft is None:
            runtime._draft_chain(
                int(token), int(pos) - 1, 1, opened=True, snap=False
            )
            runtime._embed_id(runtime._draft_ids, 0, runtime.span_h, HIDDEN * 4)
        else:
            runtime._embed(int(host_draft), runtime.span_h, HIDDEN * 4)
        runtime._embed(int(token), runtime.span_h, 0)
        _layers(runtime, int(pos))
        if greedy:
            _stage_adjust(runtime, runtime.logits, presence, seen, banned, frequency, counts)
            _argmax_at(runtime, runtime.logits, 1)
            if not _penalized(presence, seen, banned, frequency, counts):
                _argmax_at(runtime, runtime.span_logits, 2)
        elif on_device and _stage_adjust(
            runtime, runtime.logits, presence, seen, banned, frequency, counts
        ):
            _stage_top(runtime, runtime.logits, int(top_k))
        else:
            on_device = False
        runtime.session.commit()
    except Exception:
        runtime.session.abort()
        runtime.session.wait_inflight()
        raise
    runtime.session.wait_inflight()
    follow: int | None = None
    if host_draft is None:
        raw = runtime.session.download(runtime._draft_ids, 12)
        words = np.frombuffer(raw, dtype=np.uint32)
        draft_id = int(words[0])
        corrected = int(words[1])
        follow = int(words[2]) if _penalized(presence, seen, banned, frequency, counts) is False else None
        runtime._draft_count = 0
        accepted = corrected == draft_id
    else:
        draft_id = int(host_draft)
        if on_device:
            p_idx, p_pr = _read_top(runtime, int(top_k), temperature, top_p)
        else:
            target = penalized(
                runtime.session.download_f32(runtime.logits, VOCAB),
                presence,
                seen,
                banned,
                frequency,
                counts,
            )
            p_idx, p_pr = candidate_probs(target, temperature, top_p, top_k)
        accepted, corrected = keep_draft(draft_id, p_idx, p_pr, q_idx, q_pr)
    if accepted and int(draft_id) in stop:
        _restore(runtime)
        _routes(runtime, 1)
        runtime.filled = int(pos) + 1
        return PairStep(False, None, int(draft_id), None)
    if not accepted:
        _restore(runtime)
        _routes(runtime, 1)
        runtime.filled = int(pos) + 1
        return PairStep(False, None, int(corrected), None)
    _keep_row(runtime, 1)
    _routes(runtime, 2)
    runtime.filled = int(pos) + 2
    runtime._mtp_snapped = False
    if greedy and follow is not None:
        return PairStep(True, int(draft_id), None, None, follow=follow)
    if greedy:
        bonus = runtime.session.download_f32(runtime.span_logits, VOCAB)
        return PairStep(True, int(draft_id), None, bonus)
    if on_device:
        return PairStep(True, int(draft_id), None, None, device=True)
    bonus = runtime.session.download_f32(runtime.span_logits, VOCAB)
    return PairStep(True, int(draft_id), None, bonus)


def sample_buffer(
    runtime: Any,
    scores: Any,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    seen: set[int],
    banned: set[int] | None,
    frequency: float,
    counts: dict[int, int] | None,
) -> int:
    """Draw the next token from a logit buffer that is still on device."""
    from qwen36_35b_a3b.speculate import candidate_probs, penalized

    picked = _candidates(
        runtime, scores, temperature, top_p, top_k, presence, seen, banned, frequency, counts
    )
    if picked is None:
        rows = penalized(
            runtime.session.download_f32(scores, VOCAB),
            presence,
            seen,
            banned,
            frequency,
            counts,
        )
        idx, probs = candidate_probs(rows, temperature, top_p, top_k)
    else:
        idx, probs = picked
    if idx.size == 1:
        return int(idx[0])
    return int(idx[int(np.random.choice(idx.size, p=probs))])


def _penalized(
    presence: float,
    seen: set[int],
    banned: set[int] | None,
    frequency: float,
    counts: dict[int, int] | None,
) -> bool:
    return bool((presence and seen) or (frequency and counts) or banned)


def _adjustments(
    presence: float,
    seen: set[int],
    banned: set[int] | None,
    frequency: float,
    counts: dict[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    delta: dict[int, float] = {}
    if presence and seen:
        for token in seen:
            delta[int(token)] = delta.get(int(token), 0.0) - float(presence)
    if frequency and counts:
        for token, count in counts.items():
            if count > 0:
                delta[int(token)] = delta.get(int(token), 0.0) - float(frequency) * float(count)
    if banned:
        for token in banned:
            delta[int(token)] = float("-inf")
    if not delta:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)
    ids = np.fromiter(delta.keys(), dtype=np.uint32, count=len(delta))
    values = np.fromiter(delta.values(), dtype=np.float32, count=len(delta))
    return ids, values


def _poke(buf: Any, raw: bytes) -> None:
    view = memoryview(buf.contents().as_buffer(len(raw)))
    view[:] = raw


def _stage_adjust(
    runtime: Any,
    scores: Any,
    presence: float,
    seen: set[int],
    banned: set[int] | None,
    frequency: float,
    counts: dict[int, int] | None,
) -> bool:
    ids, values = _adjustments(presence, seen, banned, frequency, counts)
    if ids.size == 0:
        return True
    if ids.size > _PEN_CAP:
        return False
    _poke(runtime.span_pen_i, ids.tobytes())
    _poke(runtime.span_pen_d, values.tobytes())
    runtime.session.encode(
        "logit_adjust",
        [(scores, 0), (runtime.span_pen_i, 0), (runtime.span_pen_d, 0)],
        [_u32(int(ids.size))],
        (int(ids.size), 1, 1),
        (1, 1, 1),
    )
    return True


def _stage_top(runtime: Any, scores: Any, k: int) -> None:
    runtime.session.encode(
        "topk_partial",
        [(scores, 0), (runtime.span_top_v, 0), (runtime.span_top_i, 0)],
        [_u32(VOCAB), _u32(int(k))],
        (1, 1, 1),
        (_TOP_THREADS, 1, 1),
    )


def _read_top(
    runtime: Any, k: int, temperature: float, top_p: float
) -> tuple[np.ndarray, np.ndarray]:
    from qwen36_35b_a3b.speculate import merge_top

    count = _TOP_THREADS * int(k)
    values = np.frombuffer(
        runtime.session.download(runtime.span_top_v, count * 4), dtype=np.float32
    ).copy()
    ids = np.frombuffer(
        runtime.session.download(runtime.span_top_i, count * 4), dtype=np.uint32
    )
    return merge_top(values, ids, int(k), temperature, top_p)


def _candidates(
    runtime: Any,
    scores: Any,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    seen: set[int],
    banned: set[int] | None,
    frequency: float,
    counts: dict[int, int] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    k = int(top_k)
    if k <= 0 or k > _TOP_CAP:
        return None
    runtime.session.begin()
    try:
        if not _stage_adjust(runtime, scores, presence, seen, banned, frequency, counts):
            runtime.session.abort()
            return None
        _stage_top(runtime, scores, k)
        runtime.session.commit()
    except Exception:
        runtime.session.abort()
        runtime.session.wait_inflight()
        raise
    runtime.session.wait_inflight()
    return _read_top(runtime, k, temperature, top_p)


def _argmax_at(runtime: Any, scores: Any, slot: int) -> None:
    runtime.session.encode(
        "argmax_span",
        [(scores, 0), (runtime._argmax_v, 0), (runtime._argmax_i, 0)],
        [_u32(VOCAB)],
        (_ARGMAX, 1, 1),
        (256, 1, 1),
    )
    runtime.session.encode(
        "argmax_index",
        [
            (runtime._argmax_v, 0),
            (runtime._argmax_i, 0),
            (runtime._draft_ids, slot * 4),
        ],
        [_u32(_ARGMAX)],
        (1, 1, 1),
        (256, 1, 1),
    )


def _yield(runtime: Any) -> None:
    """Commit the open buffer and keep recording. The queue runs that buffer while the next one is recorded."""
    runtime.session.commit()
    runtime.session.begin()


def _layers(runtime: Any, pos: int) -> None:
    for layer in range(_LAYERS):
        norm = f"model.language_model.layers.{layer}.input_layernorm.weight"
        runtime._rms(runtime.span_h, 0, norm, HIDDEN, runtime.span_n, 0)
        runtime._rms(runtime.span_h, HIDDEN * 4, norm, HIDDEN, runtime.span_n, HIDDEN * 4)
        if layer % 4 == 3:
            _attn(runtime, layer, pos)
        else:
            _gdn(runtime, layer)
        post = f"model.language_model.layers.{layer}.post_attention_layernorm.weight"
        runtime._rms(runtime.span_h, 0, post, HIDDEN, runtime.span_n, 0)
        runtime._rms(runtime.span_h, HIDDEN * 4, post, HIDDEN, runtime.span_n, HIDDEN * 4)
        _moe(runtime, layer)
        if layer == 19:
            _yield(runtime)
    final = "model.language_model.norm.weight"
    runtime._rms(runtime.span_h, 0, final, HIDDEN, runtime.span_n, 0)
    runtime._rms(runtime.span_h, HIDDEN * 4, final, HIDDEN, runtime.span_n, HIDDEN * 4)
    runtime._gemv_pair(
        runtime.span_n,
        0,
        runtime.span_n,
        HIDDEN * 4,
        "lm_head.weight",
        runtime.logits,
        0,
        runtime.span_logits,
        0,
    )


def _gdn(runtime: Any, layer: int) -> None:
    base = f"model.language_model.layers.{layer}.linear_attn."
    qkv_w, qkv_s = runtime._weight(base + "in_proj_qkv.weight")
    z_w, z_s = runtime._weight(base + "in_proj_z.weight")
    a_w, a_s = runtime._weight(base + "in_proj_a.weight")
    b_w, b_s = runtime._weight(base + "in_proj_b.weight")
    runtime.session.encode(
        "gdn_in_proj_fp8_pair",
        [
            (runtime.span_n, 0),
            (runtime.span_n, HIDDEN * 4),
            qkv_w,
            qkv_s,
            z_w,
            z_s,
            a_w,
            a_s,
            b_w,
            b_s,
            (runtime.span_qkv, 0),
            (runtime.span_z, 0),
            (runtime.span_a, 0),
            (runtime.span_b, 0),
        ],
        [],
        (12352, 1, 1),
        (32, 1, 1),
    )
    _load_gdn(runtime, 0)
    runtime._gdn_recur(layer)
    runtime._copy_f32(runtime.gdn_y, 0, runtime.span_y, 0, 4096)
    _snap_slot(runtime, layer)
    _load_gdn(runtime, 1)
    runtime._gdn_recur(layer)
    runtime._copy_f32(runtime.gdn_y, 0, runtime.span_y, 4096 * 4, 4096)
    runtime._gemv_pair(
        runtime.span_y,
        0,
        runtime.span_y,
        4096 * 4,
        base + "out_proj.weight",
        runtime.span_h,
        0,
        runtime.span_h,
        HIDDEN * 4,
        acc=True,
    )


def _attn(runtime: Any, layer: int, pos: int) -> None:
    base = f"model.language_model.layers.{layer}.self_attn."
    q_w, q_s = runtime._weight(base + "q_proj.weight")
    k_w, k_s = runtime._weight(base + "k_proj.weight")
    v_w, v_s = runtime._weight(base + "v_proj.weight")
    runtime.session.encode(
        "attn_qkv_fp8_pair",
        [
            (runtime.span_n, 0),
            (runtime.span_n, HIDDEN * 4),
            q_w,
            q_s,
            k_w,
            k_s,
            v_w,
            v_s,
            (runtime.span_aq, 0),
            (runtime.span_ak, 0),
            (runtime.span_av, 0),
        ],
        [],
        (9216, 1, 1),
        (32, 1, 1),
    )
    _load_attn(runtime, 0)
    runtime._attn_recur(layer, pos)
    runtime._copy_f32(runtime.attn, 0, runtime.span_attn, 0, 4096)
    _load_attn(runtime, 1)
    runtime._attn_recur(layer, pos + 1)
    runtime._copy_f32(runtime.attn, 0, runtime.span_attn, 4096 * 4, 4096)
    runtime._gemv_pair(
        runtime.span_attn,
        0,
        runtime.span_attn,
        4096 * 4,
        base + "o_proj.weight",
        runtime.span_h,
        0,
        runtime.span_h,
        HIDDEN * 4,
        acc=True,
    )


def _moe(runtime: Any, layer: int) -> None:
    prefix = f"model.language_model.layers.{layer}.mlp."
    for row in (0, 1):
        off = row * HIDDEN * 4
        runtime._copy_f32(runtime.span_n, off, runtime.normed, 0, HIDDEN)
        runtime._copy_f32(runtime.span_h, off, runtime.hidden, 0, HIDDEN)
        runtime._route_token(layer, prefix, runtime.span_coeff, row * 4)
        if row == 0:
            runtime.session.encode(
                "copy_u32",
                [(runtime.route_ids, layer * 8 * 4), (runtime.span_route, layer * 8 * 4)],
                [_u32(8)],
                (1, 1, 1),
                (32, 1, 1),
            )
        runtime._routed_only(layer, prefix)
        runtime._copy_f32(runtime.hidden, 0, runtime.span_h, off, HIDDEN)
    gate_w, gate_s = runtime._weight(prefix + "shared_expert.gate_up")
    runtime.session.encode(
        "shared_gate_silu_pair",
        [
            (runtime.span_n, 0),
            (runtime.span_n, HIDDEN * 4),
            gate_w,
            gate_s,
            (runtime.span_shared, 0),
            (runtime.span_shared, 512 * 4),
        ],
        [],
        (512, 1, 1),
        (32, 1, 1),
    )
    runtime._gemv_pair(
        runtime.span_shared,
        0,
        runtime.span_shared,
        512 * 4,
        prefix + "shared_expert.down_proj.weight",
        runtime.span_h,
        0,
        runtime.span_h,
        HIDDEN * 4,
        coeff0=runtime.span_coeff,
        coeff1=runtime.span_coeff,
        coeff1_off=4,
    )


def _load_gdn(runtime: Any, row: int) -> None:
    runtime._copy_f32(runtime.span_qkv, row * 8192 * 4, runtime.wide, 0, 8192)
    runtime._copy_f32(runtime.span_z, row * 4096 * 4, runtime.z, 0, 4096)
    runtime._copy_f32(runtime.span_a, row * 32 * 4, runtime.ab, 0, 32)
    runtime._copy_f32(runtime.span_b, row * 32 * 4, runtime.ab2, 0, 32)


def _load_attn(runtime: Any, row: int) -> None:
    runtime._copy_f32(runtime.span_aq, row * 8192 * 4, runtime.wide, 0, 8192)
    runtime._copy_f32(runtime.span_ak, row * 512 * 4, runtime.kv, 0, 512)
    runtime._copy_f32(runtime.span_av, row * 512 * 4, runtime.vv, 0, 512)


def _snap_slot(runtime: Any, layer: int) -> None:
    slot = layer - (layer // 4)
    state = slot * 32 * 128 * 128 * 4
    hist = slot * 3 * 8192 * 2
    runtime._copy_f32(runtime.state, state, runtime.span_state, state, 32 * 128 * 128)
    runtime._copy_f32(runtime.history, hist, runtime.span_history, hist, 3 * 8192 * 2 // 4)


def _restore(runtime: Any) -> None:
    runtime.session.begin()
    try:
        runtime._copy_f32(runtime.span_state, 0, runtime.state, 0, _STATE_WORDS)
        runtime._copy_f32(runtime.span_history, 0, runtime.history, 0, _HISTORY_WORDS)
        runtime._copy_f32(runtime.span_h, 0, runtime.hidden, 0, HIDDEN)
        runtime.session.commit()
        runtime.session.wait_inflight()
    except Exception:
        runtime.session.abort()
        runtime.session.wait_inflight()
        raise
    runtime._mtp_snapped = False


def _keep_row(runtime: Any, row: int) -> None:
    runtime.session.begin()
    try:
        runtime._copy_f32(runtime.span_h, row * HIDDEN * 4, runtime.hidden, 0, HIDDEN)
        runtime.session.commit()
        runtime.session.wait_inflight()
    except Exception:
        runtime.session.abort()
        runtime.session.wait_inflight()
        raise


def _routes(runtime: Any, tokens: int) -> None:
    routes = runtime.routes
    if routes is None or tokens < 1:
        return
    raw0 = np.frombuffer(
        runtime.session.download(runtime.span_route, _LAYERS * 8 * 4), dtype=np.uint32
    ).reshape(_LAYERS, 1, 8)
    if tokens == 1:
        routes.note_decode(raw0)
        routes.take_decode(1)
        return
    raw1 = np.frombuffer(
        runtime.session.download(runtime.route_ids, _LAYERS * 8 * 4), dtype=np.uint32
    ).reshape(_LAYERS, 1, 8)
    routes.note_decode(np.concatenate((raw0, raw1), axis=1))
    routes.take_decode(2)


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)
