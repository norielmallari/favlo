"""On-device argmax and nucleus sample. The token id comes back as four bytes."""

from __future__ import annotations

import random
import struct
from typing import Any

from favlo.memory import memory

from spark_x26_1_7b.layout import VOCAB
from spark_x26_1_7b.session import Session, f32, u32

_GROUPS = 256
_THREADS = 256
_BLOCKS = VOCAB // _THREADS


class Sampler:
    """One vocabulary. Counts stay on device for the presence and frequency penalties."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.counts = session.allocate(VOCAB * 4, zero=True)
        self.partial_f = session.allocate(_GROUPS * 4, zero=True)
        self.partial_u = session.allocate(_GROUPS * 4, zero=True)
        self.indexes = session.allocate(_GROUPS * 4, zero=True)
        self.scalar = session.allocate(16, zero=True)
        self.token = session.allocate(4, zero=True)
        self.probs = session.allocate(VOCAB * 4, zero=True)
        self.pairs = session.allocate(VOCAB * 8, zero=True)
        self.block_totals = session.allocate(_BLOCKS * 4, zero=True)
        self.block_prefix = session.allocate(_BLOCKS * 4, zero=True)
        self._rng = random.Random(0)

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(int(seed))

    def clear(self) -> None:
        self.session.launch(
            "zero_u32",
            [self.counts],
            [u32(VOCAB)],
            (VOCAB // _THREADS, 1, 1),
            (_THREADS, 1, 1),
        )

    def draw(
        self,
        logits: Any,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        frequency: float,
    ) -> int:
        if temperature <= 0.0 or top_p <= 0.0:
            return self._argmax(logits)
        self.session.begin()
        self._penalize(logits, presence, frequency)
        if top_k > 0:
            self._mask_top_k(logits, top_k)
        self._softmax(logits, temperature)
        self._sort()
        self._scan()
        unit = self._rng.random()
        if unit <= 0.0:
            unit = 1e-7
        self.session.launch(
            "pick_nucleus",
            [self.probs, self.pairs, self.token],
            [struct.pack("<Iff", VOCAB, float(top_p), float(unit))],
            (1, 1, 1),
            (1, 1, 1),
        )
        self.session.launch(
            "bump_count",
            [self.counts, self.token],
            [],
            (1, 1, 1),
            (1, 1, 1),
        )
        self.session.end()
        return self._token()

    def _argmax(self, logits: Any) -> int:
        self.session.begin()
        self.session.launch(
            "argmax_partial",
            [logits, self.partial_f, self.indexes],
            [u32(VOCAB)],
            (_GROUPS, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "argmax_final",
            [self.partial_f, self.indexes, self.token],
            [],
            (1, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "bump_count",
            [self.counts, self.token],
            [],
            (1, 1, 1),
            (1, 1, 1),
        )
        self.session.end()
        return self._token()

    def _penalize(self, logits: Any, presence: float, frequency: float) -> None:
        self.session.launch(
            "penalize",
            [logits, self.counts],
            [u32(VOCAB), f32(presence), f32(frequency)],
            (VOCAB // _THREADS, 1, 1),
            (_THREADS, 1, 1),
        )

    def _mask_top_k(self, logits: Any, top_k: int) -> None:
        self.session.launch(
            "partial_min",
            [logits, self.partial_f],
            [u32(VOCAB)],
            (_GROUPS, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "final_min",
            [self.partial_f, self.scalar],
            [],
            (1, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "partial_max",
            [logits, self.partial_f],
            [u32(VOCAB)],
            (_GROUPS, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "final_max",
            [self.partial_f, self.scalar],
            [],
            (1, 1, 1),
            (_THREADS, 1, 1),
            offsets=[0, 4],
        )
        self.session.end()
        lo = struct.unpack_from("<f", bytes(memory.map(self.scalar, 0, 4)))[0]
        hi = struct.unpack_from("<f", bytes(memory.map(self.scalar, 4, 4)))[0]
        self.session.begin()
        for _ in range(16):
            mid = (lo + hi) * 0.5
            self.session.launch(
                "partial_count",
                [logits, self.partial_u],
                [u32(VOCAB), f32(mid)],
                (_GROUPS, 1, 1),
                (_THREADS, 1, 1),
            )
            self.session.launch(
                "final_count",
                [self.partial_u, self.token],
                [],
                (1, 1, 1),
                (_THREADS, 1, 1),
            )
            self.session.end()
            count = struct.unpack_from("<I", bytes(memory.map(self.token, 0, 4)))[0]
            self.session.begin()
            if count > top_k:
                lo = mid
            else:
                hi = mid
        self.session.launch(
            "mask_below",
            [logits],
            [u32(VOCAB), f32(hi)],
            (VOCAB // _THREADS, 1, 1),
            (_THREADS, 1, 1),
        )

    def _softmax(self, logits: Any, temperature: float) -> None:
        self.session.launch(
            "partial_max",
            [logits, self.partial_f],
            [u32(VOCAB)],
            (_GROUPS, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "final_max",
            [self.partial_f, self.scalar],
            [],
            (1, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "partial_exp",
            [logits, self.partial_f, self.scalar],
            [u32(VOCAB), f32(temperature)],
            (_GROUPS, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "final_sum",
            [self.partial_f, self.scalar],
            [],
            (1, 1, 1),
            (_THREADS, 1, 1),
            offsets=[0, 4],
        )
        self.session.launch(
            "write_probs",
            [logits, self.probs, self.scalar, self.scalar],
            [u32(VOCAB), f32(temperature)],
            (VOCAB // _THREADS, 1, 1),
            (_THREADS, 1, 1),
            offsets=[0, 0, 0, 4],
        )

    def _sort(self) -> None:
        self.session.launch(
            "fill_pairs",
            [self.probs, self.pairs],
            [u32(VOCAB)],
            (VOCAB // _THREADS, 1, 1),
            (_THREADS, 1, 1),
        )
        stage = 2
        while stage <= VOCAB:
            stride = stage >> 1
            while stride > 0:
                self.session.launch(
                    "bitonic",
                    [self.pairs],
                    [u32(stage), u32(stride)],
                    (VOCAB // _THREADS, 1, 1),
                    (_THREADS, 1, 1),
                )
                stride >>= 1
            stage <<= 1

    def _scan(self) -> None:
        self.session.launch(
            "export_probs",
            [self.pairs, self.probs],
            [],
            (VOCAB // _THREADS, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "block_scan",
            [self.probs, self.block_totals],
            [],
            (_BLOCKS, 1, 1),
            (_THREADS, 1, 1),
        )
        self.session.launch(
            "scan_blocks",
            [self.block_totals, self.block_prefix],
            [],
            (1, 1, 1),
            (_BLOCKS, 1, 1),
        )
        self.session.launch(
            "add_block_prefix",
            [self.probs, self.block_prefix],
            [],
            (VOCAB // _THREADS, 1, 1),
            (_THREADS, 1, 1),
        )

    def _token(self) -> int:
        raw = bytes(memory.map(self.token, 0, 4))
        return struct.unpack_from("<I", raw)[0]
