"""K and V cache widths. The store test is one GPU. The rest are host checks."""

import numpy as np
from qwen38_flash_next.context import kv_bytes_per_token
from qwen38_flash_next.formats import (
    INDEX_BF16_BYTES,
    KV_BF16_BYTES,
    KV_CHANNELS,
    KV_E4M3_BYTES,
    decode_e4m3,
    encode_bf16,
    encode_e4m3,
    kv_store_groups,
    pack_bf16,
    quantize_e4m3_token,
)
from qwen38_flash_next.session import MetalSession, _buffer_slice


def test_e4m3_bytes_include_the_block_scales() -> None:
    assert KV_E4M3_BYTES == 528
    assert kv_bytes_per_token(12) == 12 * (KV_E4M3_BYTES * 2 + INDEX_BF16_BYTES)
    assert kv_bytes_per_token(12, cache="bf16") == 12 * (KV_BF16_BYTES * 2 + INDEX_BF16_BYTES)


def test_finite_e4m3_codes_roundtrip() -> None:
    for code in range(256):
        if code & 0x7F == 0x7F:
            continue
        assert encode_e4m3(decode_e4m3(code)) == code


def test_store_group_counts() -> None:
    assert kv_store_groups(6, 0, "e4m3", 0) == 4 * 32
    assert kv_store_groups(6, 0, "e4m3", 1) == 2 * 32
    assert kv_store_groups(3, 6, "e4m3", 0) == 4 * 32
    assert kv_store_groups(3, 6, "e4m3", 1) == 1 * 32
    assert kv_store_groups(2, 1, "bf16", 0) == 2 * 32
    assert kv_store_groups(2, 1, "bf16", 1) == 0


def _read(session: MetalSession, buf: object, nbytes: int) -> bytes:
    session.flush()
    return _buffer_slice(buf, 0, nbytes)


def _commit_host(
    body: bytearray, tail: list[np.ndarray], base: int, incoming: np.ndarray
) -> tuple[bytearray, list[np.ndarray]]:
    count = int(incoming.shape[0])
    old_complete = base & ~3
    new_end = base + count
    new_complete = new_end & ~3
    window: list[np.ndarray] = []
    for token in range(old_complete, new_end):
        if token < base:
            window.append(tail[token - old_complete])
        else:
            window.append(incoming[token - base])
    quant = new_complete - old_complete
    out = bytearray(body)
    need = new_complete * KV_E4M3_BYTES
    if len(out) < need:
        out.extend(bytes(need - len(out)))
    for index in range(quant):
        start = (old_complete + index) * KV_E4M3_BYTES
        out[start : start + KV_E4M3_BYTES] = quantize_e4m3_token(window[index])
    if new_complete == old_complete:
        extended = list(tail)
        for row in incoming:
            _raw, back = pack_bf16(row)
            extended.append(back.reshape(KV_CHANNELS))
        return out, extended
    remainder = []
    for row in window[quant:]:
        _raw, back = pack_bf16(row)
        remainder.append(back.reshape(KV_CHANNELS))
    return out, remainder


def test_e4m3_store_matches_the_host_blocks() -> None:
    session = MetalSession()
    rng = np.random.default_rng(4)
    body_k = session.alloc(8 * KV_E4M3_BYTES)
    body_v = session.alloc(8 * KV_E4M3_BYTES)
    tail_k = session.alloc(3 * KV_BF16_BYTES)
    tail_v = session.alloc(3 * KV_BF16_BYTES)
    host_k = bytearray()
    host_v = bytearray()
    tail_state_k: list[np.ndarray] = []
    tail_state_v: list[np.ndarray] = []
    base = 0
    for count in (6, 3):
        key = rng.normal(size=(count, KV_CHANNELS)).astype(np.float32)
        value = rng.normal(size=(count, KV_CHANNELS)).astype(np.float32) * 3
        session.kv_store(
            session.dev_from(key),
            session.dev_from(value),
            body_k,
            body_v,
            tail_k,
            tail_v,
            count,
            base,
            "e4m3",
            0,
        )
        session.kv_store(
            session.dev_from(key),
            session.dev_from(value),
            body_k,
            body_v,
            tail_k,
            tail_v,
            count,
            base,
            "e4m3",
            1,
        )
        host_k, tail_state_k = _commit_host(host_k, tail_state_k, base, key)
        host_v, tail_state_v = _commit_host(host_v, tail_state_v, base, value)
        base += count
    assert base == 9
    assert _read(session, body_k, len(host_k)) == bytes(host_k)
    assert _read(session, body_v, len(host_v)) == bytes(host_v)
    tail_bytes_k, _back = pack_bf16(np.stack(tail_state_k))
    tail_bytes_v, _back = pack_bf16(np.stack(tail_state_v))
    assert _read(session, tail_k, len(tail_bytes_k)) == tail_bytes_k
    assert _read(session, tail_v, len(tail_bytes_v)) == tail_bytes_v


def test_bf16_store_matches_the_host_words() -> None:
    session = MetalSession()
    key = np.arange(2 * KV_CHANNELS, dtype=np.float32).reshape(2, KV_CHANNELS) / 7
    value = -key
    body_k = session.alloc(7 * KV_BF16_BYTES)
    body_v = session.alloc(7 * KV_BF16_BYTES)
    tail = session.alloc(64)
    session.kv_store(
        session.dev_from(key),
        session.dev_from(value),
        body_k,
        body_v,
        tail,
        tail,
        2,
        5,
        "bf16",
        0,
    )
    words_k = np.frombuffer(_read(session, body_k, 7 * KV_BF16_BYTES), dtype=np.uint16)
    words_v = np.frombuffer(_read(session, body_v, 7 * KV_BF16_BYTES), dtype=np.uint16)
    flat_k = key.reshape(-1)
    flat_v = value.reshape(-1)
    for token in range(2):
        for channel in range(KV_CHANNELS):
            index = (5 + token) * KV_CHANNELS + channel
            assert int(words_k[index]) == encode_bf16(float(flat_k[token * KV_CHANNELS + channel]))
            assert int(words_v[index]) == encode_bf16(float(flat_v[token * KV_CHANNELS + channel]))
