"""Device checks for the CUDA language step."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pipelines" / "qwen35_4b" / "cuda" / "engine.py"


def _engine():
    from favlo.host import cuda_home

    if cuda_home() is None:
        pytest.skip("CUDA Toolkit is not installed")
    spec = importlib.util.spec_from_file_location("qwen35_cuda_engine_test", ENGINE)
    if spec is None or spec.loader is None:
        message = f"cannot load {ENGINE}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Engine()


def _quant(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = values.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    scale = np.where(amax > 0.0, amax / 127.0, 0.0).astype(np.float32)
    inv = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale != 0.0)
    codes = np.clip(np.rint(blocks * inv[:, None]), -127, 127).astype(np.int32)
    return codes, scale


def test_alloc_is_device_memory() -> None:
    engine = _engine()
    pointer = engine.alloc(4096)
    try:
        assert engine.on_device(pointer)
    finally:
        engine.free(pointer)
        engine.close()


def test_int8_gemv_uses_row_and_block_scales() -> None:
    engine = _engine()
    try:
        rng = np.random.default_rng(4)
        rows, inner = 3, 64
        weight = rng.normal(size=(rows, inner)).astype(np.float32)
        row_scale = (np.max(np.abs(weight), axis=1) / 127.0).astype(np.float32)
        codes = np.clip(np.rint(weight / row_scale[:, None]), -127, 127).astype(np.int8)
        activation = rng.normal(size=inner).astype(np.float32)
        act_codes, act_scale = _quant(activation)
        dots = (codes.astype(np.int32).reshape(rows, -1, 32) * act_codes).sum(axis=2)
        expect = (dots.astype(np.float32) * act_scale).sum(axis=1) * row_scale
        weight_p = engine.alloc(codes.nbytes + row_scale.nbytes)
        source = engine.alloc(activation.nbytes)
        dest = engine.alloc(rows * 4)
        engine.h2d(weight_p, codes)
        engine.h2d(weight_p + codes.nbytes, row_scale)
        engine.h2d(source, activation)
        engine.gemv(weight_p, weight_p + codes.nbytes, source, dest, rows, inner)
        engine.sync()
        found = np.empty(rows, dtype=np.float32)
        engine.d2h(found, dest)
        np.testing.assert_allclose(found, expect, rtol=1e-4, atol=1e-4)
    finally:
        engine.close()


def test_int8_gemm_matches_the_row_scales() -> None:
    engine = _engine()
    try:
        rng = np.random.default_rng(5)
        rows, inner, tokens = 4, 64, 3
        weight = rng.normal(size=(rows, inner)).astype(np.float32)
        row_scale = (np.max(np.abs(weight), axis=1) / 127.0).astype(np.float32)
        codes = np.clip(np.rint(weight / row_scale[:, None]), -127, 127).astype(np.int8)
        activation = rng.normal(size=(tokens, inner)).astype(np.float32)
        expect = np.empty((tokens, rows), dtype=np.float32)
        for token in range(tokens):
            act_codes, act_scale = _quant(activation[token])
            dots = (codes.astype(np.int32).reshape(rows, -1, 32) * act_codes).sum(axis=2)
            expect[token] = (dots.astype(np.float32) * act_scale).sum(axis=1) * row_scale
        weight_p = engine.alloc(codes.nbytes + row_scale.nbytes)
        source = engine.alloc(activation.nbytes)
        dest = engine.alloc(tokens * rows * 4)
        engine.h2d(weight_p, codes)
        engine.h2d(weight_p + codes.nbytes, row_scale)
        engine.h2d(source, activation)
        engine.gemm(weight_p, weight_p + codes.nbytes, source, dest, rows, inner, tokens)
        engine.sync()
        found = np.empty((tokens, rows), dtype=np.float32)
        engine.d2h(found, dest)
        np.testing.assert_allclose(found, expect, rtol=1e-4, atol=1e-4)
    finally:
        engine.close()


def _packed_weight(engine, matrix: np.ndarray) -> tuple[int, np.ndarray]:
    row_scale = (np.max(np.abs(matrix), axis=1) / 127.0).astype(np.float32)
    row_scale = np.where(row_scale > 0.0, row_scale, np.float32(1.0))
    codes = np.clip(np.rint(matrix / row_scale[:, None]), -127, 127).astype(np.int8)
    pointer = engine.alloc(codes.nbytes + row_scale.nbytes)
    engine.h2d(pointer, codes)
    engine.h2d(pointer + codes.nbytes, row_scale)
    return pointer, row_scale


def _expect_gemm(
    codes: np.ndarray, row_scale: np.ndarray, activation: np.ndarray
) -> np.ndarray:
    rows = codes.shape[0]
    tokens = activation.shape[0]
    expect = np.empty((tokens, rows), dtype=np.float32)
    for token in range(tokens):
        act_codes, act_scale = _quant(activation[token])
        dots = (codes.astype(np.int32).reshape(rows, -1, 32) * act_codes).sum(axis=2)
        expect[token] = (dots.astype(np.float32) * act_scale).sum(axis=1) * row_scale
    return expect


def test_paired_gemm_matches_each_product() -> None:
    engine = _engine()
    try:
        rng = np.random.default_rng(6)
        inner, tokens = 64, 3
        left = rng.normal(size=(4, inner)).astype(np.float32)
        right = rng.normal(size=(8, inner)).astype(np.float32)
        activation = rng.normal(size=(tokens, inner)).astype(np.float32)
        left_p, left_scale = _packed_weight(engine, left)
        right_p, right_scale = _packed_weight(engine, right)
        source = engine.alloc(activation.nbytes)
        dest_l = engine.alloc(tokens * 4 * 4)
        dest_r = engine.alloc(tokens * 8 * 4)
        engine.h2d(source, activation)
        engine.gemm_pair(
            left_p,
            left_p + left.size,
            dest_l,
            4,
            right_p,
            right_p + right.size,
            dest_r,
            8,
            source,
            inner,
            tokens,
        )
        engine.sync()
        found_l = np.empty((tokens, 4), dtype=np.float32)
        found_r = np.empty((tokens, 8), dtype=np.float32)
        engine.d2h(found_l, dest_l)
        engine.d2h(found_r, dest_r)
        left_codes = np.clip(np.rint(left / left_scale[:, None]), -127, 127).astype(np.int8)
        right_codes = np.clip(np.rint(right / right_scale[:, None]), -127, 127).astype(np.int8)
        np.testing.assert_allclose(
            found_l, _expect_gemm(left_codes, left_scale, activation), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            found_r, _expect_gemm(right_codes, right_scale, activation), rtol=1e-4, atol=1e-4
        )
        engine.capture_begin(1)
        engine.gemm_pair(
            left_p,
            left_p + left.size,
            dest_l,
            4,
            right_p,
            right_p + right.size,
            dest_r,
            8,
            source,
            inner,
            tokens,
        )
        engine.capture_end(1)
        engine.zero(dest_l, tokens * 4 * 4)
        engine.zero(dest_r, tokens * 8 * 4)
        engine.replay(1)
        engine.sync()
        engine.d2h(found_l, dest_l)
        engine.d2h(found_r, dest_r)
        np.testing.assert_allclose(
            found_l, _expect_gemm(left_codes, left_scale, activation), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            found_r, _expect_gemm(right_codes, right_scale, activation), rtol=1e-4, atol=1e-4
        )
    finally:
        engine.close()


def test_gdn_step_matches_the_recurrence() -> None:
    engine = _engine()
    try:
        rng = np.random.default_rng(7)
        heads, dim = 32, 128
        mixed = rng.normal(size=8192).astype(np.float32)
        proj_a = rng.normal(size=heads).astype(np.float32)
        proj_b = rng.normal(size=heads).astype(np.float32)
        a_log = rng.normal(size=heads).astype(np.float16)
        dt_bias = rng.normal(size=heads).astype(np.float16)
        state0 = rng.normal(size=(heads, dim, dim)).astype(np.float32) * 0.01
        expect = _gdn_reference(state0.copy(), mixed, proj_a, proj_b, a_log, dt_bias)
        state = engine.alloc(state0.nbytes)
        mixed_p = engine.alloc(mixed.nbytes)
        a_p = engine.alloc(proj_a.nbytes)
        b_p = engine.alloc(proj_b.nbytes)
        log_p = engine.alloc(a_log.nbytes)
        bias_p = engine.alloc(dt_bias.nbytes)
        out = engine.alloc(heads * dim * 4)
        engine.h2d(state, state0)
        engine.h2d(mixed_p, mixed)
        engine.h2d(a_p, proj_a)
        engine.h2d(b_p, proj_b)
        engine.h2d(log_p, a_log)
        engine.h2d(bias_p, dt_bias)
        engine.gdn(state, mixed_p, a_p, b_p, log_p, bias_p, out, 16, heads)
        engine.sync()
        found = np.empty(heads * dim, dtype=np.float32)
        engine.d2h(found, out)
        np.testing.assert_allclose(found, expect, rtol=1e-4, atol=1e-4)
    finally:
        engine.close()


def _gdn_reference(
    state: np.ndarray,
    mixed: np.ndarray,
    proj_a: np.ndarray,
    proj_b: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
) -> np.ndarray:
    dim = 128
    key_heads = 16
    output = np.empty(32 * dim, dtype=np.float32)
    for head in range(32):
        key = head // 2
        query = mixed[key * dim : (key + 1) * dim]
        key_row = mixed[key_heads * dim + key * dim : key_heads * dim + (key + 1) * dim]
        value = mixed[2 * key_heads * dim + head * dim : 2 * key_heads * dim + (head + 1) * dim]
        query = query / np.sqrt(np.dot(query, query) + 1e-6) / np.sqrt(dim)
        key_row = key_row / np.sqrt(np.dot(key_row, key_row) + 1e-6)
        beta = 1.0 / (1.0 + np.exp(-proj_b[head]))
        raw = float(proj_a[head] + np.float32(dt_bias[head]))
        soft = raw if raw > 20.0 else float(np.log1p(np.exp(raw)))
        alpha = float(np.exp(-np.exp(np.float32(a_log[head])) * soft))
        matrix = state[head]
        sk = matrix.T @ key_row
        delta = beta * (value - alpha * sk)
        updated = alpha * matrix + np.outer(key_row, delta)
        state[head] = updated
        output[head * dim : (head + 1) * dim] = updated.T @ query
    return output


def _draw(
    logits: np.ndarray,
    counts: np.ndarray,
    top_k: int,
    temperature: float,
    top_p: float,
    presence: float,
    frequency: float,
    draw: float,
) -> int:
    """Host twin of the device nucleus. Argmax keeps the smaller index on a tie."""
    penalized = logits.astype(np.float32).copy()
    seen = counts > 0
    penalty = np.float32(presence) + np.float32(frequency) * counts[seen]
    penalized[seen] -= penalty
    if temperature <= 0.0:
        return int(np.argmax(penalized))
    rest = penalized.copy()
    chosen: list[int] = []
    for _ in range(top_k):
        best = int(np.argmax(rest))
        chosen.append(best)
        rest[best] = np.float32(-1.0e30)
    values = penalized[np.array(chosen)]
    weights = np.exp((values - values[0]) / np.float32(max(temperature, 1.0e-5)))
    total = float(weights.sum())
    keep = len(chosen)
    if 0.0 < top_p < 1.0 and total > 0.0:
        cumulative = np.cumsum(weights / np.float32(total))
        keep = int(np.searchsorted(cumulative, top_p, side="left")) + 1
    kept = weights[:keep]
    target = draw * float(kept.sum())
    run = 0.0
    pick = chosen[keep - 1]
    for index, weight in enumerate(kept):
        run += float(weight)
        if target < run:
            pick = chosen[index]
            break
    return int(pick)


def _quant_query(query: np.ndarray) -> np.ndarray:
    """Match the prefill score: one int8 scale for every 32 query channels."""
    out = np.empty_like(query)
    for start in range(0, query.shape[0], 32):
        chunk = query[start : start + 32]
        peak = float(np.max(np.abs(chunk)))
        factor = np.float32(peak / 127.0) if peak > 0.0 else np.float32(0.0)
        inverse = np.float32(0.0) if factor == 0.0 else np.float32(1.0 / factor)
        codes = np.clip(np.rint(chunk * inverse), -127, 127)
        out[start : start + 32] = (codes * factor).astype(np.float32)
    return out


def _softmax_mix(key: np.ndarray, value: np.ndarray, query: np.ndarray, scale: float) -> np.ndarray:
    scores = key.astype(np.float32) @ query
    scores *= np.float32(scale)
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    return (weights @ value.astype(np.float32)) * np.float32(0.5)


def _online_half_mix(key: np.ndarray, value: np.ndarray, query: np.ndarray,
    scale: float) -> np.ndarray:
    """Match prefill: each 16-key tile rounds the weight and the value to fp16."""
    scores = key.astype(np.float32) @ query.astype(np.float32)
    scores *= np.float32(scale)
    acc = np.zeros(value.shape[1], np.float32)
    running_max = np.float32(-1.0e30)
    running_sum = np.float32(0.0)
    values = value.astype(np.float32)
    for base in range(0, scores.shape[0], 16):
        tile = scores[base : base + 16]
        tile_max = np.float32(np.max(tile))
        if tile_max <= np.float32(-1.0e20):
            continue
        nxt = np.maximum(running_max, tile_max)
        alpha = np.exp(running_max - nxt)
        prob = np.exp(tile - nxt).astype(np.float16).astype(np.float32)
        block = values[base : base + 16].astype(np.float16).astype(np.float32)
        acc = acc * alpha + prob @ block
        running_sum = running_sum * alpha + np.float32(prob.sum())
        running_max = nxt
    if running_sum == 0.0:
        return acc
    return acc / running_sum * np.float32(0.5)


def test_attention_mixes_every_key_on_each_lane() -> None:
    """A warp that walks more than one key has to share the full score."""
    engine = _engine()
    try:
        rng = np.random.default_rng(7)
        heads, kv_heads, dim = 4, 2, 256
        scale = float(dim**-0.5)
        for count, length in ((1, 40), (4, 24), (7, 16), (128, 32)):
            seen_max = length + count - 1
            query = rng.normal(size=(count, heads, dim)).astype(np.float32) * np.float32(0.05)
            gate = np.zeros((count, heads, dim), np.float32)
            packed = np.concatenate([query, gate], axis=2)
            key = rng.integers(-2, 3, size=(seen_max, kv_heads, dim), dtype=np.int8)
            value = rng.integers(-2, 3, size=(seen_max, kv_heads, dim), dtype=np.int8)
            kscale = np.full((seen_max, kv_heads, dim // 32), 127, np.uint8)
            query_p = engine.alloc(packed.nbytes)
            key_p = engine.alloc(key.nbytes)
            value_p = engine.alloc(value.nbytes)
            scale_p = engine.alloc(kscale.nbytes)
            out_p = engine.alloc(count * heads * dim * 4)
            engine.h2d(query_p, packed)
            engine.h2d(key_p, key)
            engine.h2d(value_p, value)
            engine.h2d(scale_p, kscale)
            engine.attend(
                query_p, key_p, scale_p, value_p, scale_p, out_p,
                length, heads, kv_heads, dim, scale, count,
            )
            engine.sync()
            found = np.empty((count, heads, dim), np.float32)
            engine.d2h(found, out_p)
            for token in (0, count - 1):
                seen = length + token
                for head in (0, heads - 1):
                    kv = head // (heads // kv_heads)
                    scored = query[token, head]
                    if count > 1:
                        scored = _quant_query(scored)
                    if count == 1:
                        expect = _softmax_mix(key[:seen, kv], value[:seen, kv], scored, scale)
                    else:
                        expect = _online_half_mix(key[:seen, kv], value[:seen, kv], scored, scale)
                    np.testing.assert_allclose(
                        found[token, head], expect, rtol=1e-4, atol=1e-4
                    )
    finally:
        engine.close()


def test_sample_matches_penalized_nucleus() -> None:
    engine = _engine()
    try:
        logits = np.array([0.0, 3.0, 1.0, 2.0], dtype=np.float32)
        counts = np.zeros(4, dtype=np.int32)
        logits_p = engine.alloc(logits.nbytes)
        counts_p = engine.alloc(counts.nbytes)
        token_p = engine.alloc(4)
        engine.h2d(logits_p, logits)
        engine.h2d(counts_p, counts)
        engine.sample(logits_p, counts_p, token_p, 4, 2, 0.0, 1.0, 0.0, 0.0, 0.0)
        found = np.empty(1, dtype=np.int32)
        engine.d2h(found, token_p)
        assert int(found[0]) == 1
        engine.h2d(counts_p, np.array([0, 1, 0, 0], dtype=np.int32))
        engine.sample(logits_p, counts_p, token_p, 4, 2, 0.0, 1.0, 1.5, 0.0, 0.0)
        engine.d2h(found, token_p)
        assert int(found[0]) == 3
        fresh = np.zeros(4, dtype=np.int32)
        for draw, top_p in ((0.0, 1.0), (0.99, 1.0), (0.0, 0.5)):
            engine.h2d(counts_p, fresh)
            engine.sample(logits_p, counts_p, token_p, 4, 2, 1.0, top_p, 0.0, 0.0, draw)
            engine.d2h(found, token_p)
            expect = _draw(logits, fresh, 2, 1.0, top_p, 0.0, 0.0, draw)
            assert int(found[0]) == expect
    finally:
        engine.close()
