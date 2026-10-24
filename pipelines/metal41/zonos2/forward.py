"""One delayed frame. Appendix C of Clark et al., then the DAC decoder."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from zonos2.audio import SILENCE_FRAMES, accurate_token, background_token, quality_token, text_ids
from zonos2.checks import rms_norm, softcap, swiglu
from zonos2.codec import Decoder
from zonos2.delay import shear, unshear
from zonos2.layout import (
    CODEBOOKS,
    CONTEXT,
    DIM,
    EOA_ID,
    HEAD_DIM,
    HEADS,
    KV_HEADS,
    LAYERS,
    NORM_EPS,
    PAD_ID,
    QK_EPS,
    ROPE_THETA,
    SPEAKER_DIM,
    TEXT_VOCAB,
)
from zonos2.record import Layer, Record, load_record
from zonos2.sampling import REPETITION_CODEBOOKS, sample_code
from zonos2.session import Session


class Model:
    def __init__(self, record: Record, session: Session, decoder: Decoder) -> None:
        self.record = record
        self.session = session
        self.decoder = decoder
        self.keys = [np.zeros((0, KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(LAYERS)]
        self.values = [np.zeros((0, KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(LAYERS)]
        self.router_state: list[np.ndarray | None] = [None] * LAYERS

    @classmethod
    def open(cls, path: Path, owner: str) -> Model:
        session = Session(owner)
        record = load_record(session, path)
        return cls(record, session, Decoder(path.parent / "dac" / "model.safetensors", owner))

    def speak(self, text: str, speaker: np.ndarray | None = None, max_frames: int | None = None) -> np.ndarray:
        prefix = self._prefix(text, speaker)
        hidden = prefix[0]
        for position, vector in enumerate(prefix):
            hidden = self._stack(vector, position)
        logits = self._logits(hidden)
        history: list[np.ndarray] = []
        rng = np.random.default_rng()
        eoa = None
        limit = CONTEXT - len(prefix) - 1
        if max_frames is not None:
            limit = min(limit, max_frames)
        while len(history) < limit:
            step = len(history)
            row = np.empty(CODEBOOKS, dtype=np.int64)
            for book in range(CODEBOOKS):
                seen = [int(past[book]) for past in history]
                if book >= REPETITION_CODEBOOKS:
                    seen = []
                row[book] = sample_code(logits[book], seen, rng, allow_eoa=True)
            history.append(row)
            if eoa is None and int(row[0]) == EOA_ID:
                eoa = step
            if eoa is not None and len(history) >= eoa + CODEBOOKS - 1:
                break
            if len(history) == limit:
                break
            if len(history) % 25 == 0:
                sys.stderr.write(f"zonos2 frame {len(history)}\n")
            hidden = self._stack(self._audio_row(row), len(prefix) + len(history) - 1)
            logits = self._logits(hidden)
        delayed = np.stack(history) if history else np.zeros((0, CODEBOOKS), dtype=np.int64)
        if delayed.shape[0] < CODEBOOKS - 1:
            message = "the model ended before an audio frame"
            raise RuntimeError(message)
        aligned = unshear(delayed)
        if eoa is not None:
            aligned = aligned[:eoa]
        usable = np.minimum(aligned, EOA_ID - 1)
        if usable.size == 0:
            message = "the model ended before an audio frame"
            raise RuntimeError(message)
        return self.decoder.waveform(usable.astype(np.int64))

    def _prefix(self, text: str, speaker: np.ndarray | None) -> list[np.ndarray]:
        source = np.zeros(SPEAKER_DIM, dtype=np.float32) if speaker is None else speaker.astype(np.float32)
        if source.shape != (SPEAKER_DIM,):
            message = "the speaker vector is 2048 floats"
            raise ValueError(message)
        projected = self.session.gemm(source.reshape(1, -1), self.record.speaker_lda)[0]
        projected += self.record.speaker_lda_bias
        voice = self.session.gemm(projected.reshape(1, -1), self.record.speaker)[0]
        voice += self.record.speaker_bias
        rows = [voice, self._text_row(background_token(False)), self._text_row(accurate_token())]
        rows.append(self._text_row(quality_token(5, 3)))
        for code in text_ids(text):
            rows.append(self._text_row(int(code)))
        opened = shear(SILENCE_FRAMES)[: SILENCE_FRAMES.shape[0]]
        for frame in opened:
            rows.append(self._audio_row(frame))
        return rows

    def _text_row(self, token: int) -> np.ndarray:
        vector = self._pad_sum()
        vector += self.record.text[token].astype(np.float32)
        return vector

    def _audio_row(self, codes: np.ndarray) -> np.ndarray:
        vector = self.record.text[TEXT_VOCAB].astype(np.float32).copy()
        for book, code in enumerate(codes):
            vector += self.record.audio[book][int(code)].astype(np.float32)
        return vector

    def _pad_sum(self) -> np.ndarray:
        vector = np.zeros(DIM, dtype=np.float32)
        for book in range(CODEBOOKS):
            vector += self.record.audio[book][PAD_ID].astype(np.float32)
        return vector

    def _stack(self, hidden: np.ndarray, position: int) -> np.ndarray:
        hidden = rms_norm(hidden, np.ones(DIM, dtype=np.float32))
        for index, layer in enumerate(self.record.layers):
            hidden = self._layer(layer, index, hidden, position)
        return hidden

    def _layer(self, layer: Layer, index: int, hidden: np.ndarray, position: int) -> np.ndarray:
        normed = rms_norm(hidden, layer.attention_norm)
        query = self._heads(self.session.gemm(normed.reshape(1, -1), layer.wq)[0], HEADS)
        key = self._heads(self.session.gemm(normed.reshape(1, -1), layer.wk)[0], KV_HEADS)
        value = self._heads(self.session.gemm(normed.reshape(1, -1), layer.wv)[0], KV_HEADS)
        query = _qk(query) * np.abs(layer.alpha).reshape(HEADS, 1)
        key = _qk(key)
        query = _rope(query, position)
        key = _rope(key, position)
        self.keys[index] = np.concatenate((self.keys[index], key.reshape(1, KV_HEADS, HEAD_DIM)), axis=0)
        self.values[index] = np.concatenate((self.values[index], value.reshape(1, KV_HEADS, HEAD_DIM)), axis=0)
        mixed = self.session.attend(query, self.keys[index], self.values[index])
        gate = 1.0 / (1.0 + np.exp(-self.session.gemm(normed.reshape(1, -1), layer.gater)[0]))
        mixed = mixed * gate.reshape(HEADS, 1)
        attended = self.session.gemm(mixed.reshape(1, -1), layer.wo)[0]
        residual = hidden + attended
        activated = rms_norm(residual, layer.ffn_norm)
        if layer.experts is None:
            up = self.session.gemm(activated.reshape(1, -1), layer.up)[0]
            gate = self.session.gemm(activated.reshape(1, -1), layer.gate)[0]
            delta = self.session.gemm(swiglu(gate, up).reshape(1, -1), layer.down)[0]
            return residual + delta
        state = self.session.gemm(activated.reshape(1, -1), layer.router_down)[0]
        state = state + layer.router_down_bias
        previous = self.router_state[index - 1] if index else None
        if layer.router_scale is not None and previous is not None:
            state = state + layer.router_scale * previous
        self.router_state[index] = state
        routed = rms_norm(state, layer.router_norm, NORM_EPS)
        routed = _gelu(self.session.gemm(routed.reshape(1, -1), layer.router_mlp0)[0] + layer.router_mlp0_bias)
        routed = _gelu(self.session.gemm(routed.reshape(1, -1), layer.router_mlp2)[0] + layer.router_mlp2_bias)
        logits = self.session.gemm(routed.reshape(1, -1), layer.router_mlp4)[0]
        count = 2 if index == 26 else 1
        picked, probs = choose_experts(logits, layer.balancing, count)
        delta = np.zeros(DIM, dtype=np.float32)
        for expert in picked:
            gate_w, up_w, down_w = layer.experts[int(expert)]
            up = self.session.gemm(activated.reshape(1, -1), up_w)[0]
            gate = self.session.gemm(activated.reshape(1, -1), gate_w)[0]
            piece = self.session.gemm(swiglu(gate, up).reshape(1, -1), down_w)[0]
            delta += piece * np.float32(probs[int(expert)])
        return residual + delta

    def _logits(self, hidden: np.ndarray) -> np.ndarray:
        normed = rms_norm(hidden, self.record.out_norm)
        raw = self.session.gemm(normed.reshape(1, -1), self.record.head)[0]
        return softcap(raw).reshape(CODEBOOKS, -1)

    def _heads(self, values: np.ndarray, count: int) -> np.ndarray:
        return values.reshape(count, HEAD_DIM)


def _rope(heads: np.ndarray, position: int) -> np.ndarray:
    pairs = heads.reshape(heads.shape[0], -1, 2).astype(np.float64)
    index = np.arange(pairs.shape[1], dtype=np.float64)
    freq = position / np.power(ROPE_THETA, 2.0 * index / (pairs.shape[1] * 2))
    cos = np.cos(freq)
    sin = np.sin(freq)
    left = pairs[:, :, 0] * cos - pairs[:, :, 1] * sin
    right = pairs[:, :, 0] * sin + pairs[:, :, 1] * cos
    out = np.empty_like(pairs)
    out[:, :, 0] = left
    out[:, :, 1] = right
    return out.reshape(heads.shape).astype(np.float32)


def _qk(heads: np.ndarray) -> np.ndarray:
    flat = heads.astype(np.float64)
    rms = np.sqrt(np.mean(flat * flat, axis=1, keepdims=True) + QK_EPS)
    return (flat / rms).astype(np.float32)


def choose_experts(logits: np.ndarray, bias: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Softmax, then top-k on those probabilities plus the legacy balancing bias.

    The mixture weight is the softmax probability. The bias only breaks the ranking.
    """
    shifted = logits.astype(np.float64)
    shifted -= np.max(shifted)
    probs = np.exp(shifted)
    probs /= probs.sum()
    routing = probs + bias.astype(np.float64)
    picked = np.argpartition(routing, -count)[-count:]
    order = picked[np.argsort(routing[picked])[::-1]]
    return order.astype(np.int64), probs


def _gelu(values: np.ndarray) -> np.ndarray:
    """erf GELU, the published router activation."""
    scale = math.sqrt(2.0)
    data = values.astype(np.float64)
    activated = [0.5 * item * (1.0 + math.erf(item / scale)) for item in data]
    return np.asarray(activated, dtype=np.float32)
