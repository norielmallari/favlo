"""Open a VoxCPM2 checkout and speak one line."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from favlo.media_memory import begin_release, finish_release, open_gate, stamp_owner
from favlo.media_pipeline import MediaPaused
from favlo.memory import memory
from voxcpm2.metal41.metal import Session
from voxcpm2.metal41.speak import generate
from voxcpm2.metal41.store import (
    WEIGHTS_NAME,
    Store,
    load_pack,
    pack_ready,
    read_blob,
    weight_bytes,
)
from voxcpm2.metal41.vae import Decoder
from voxcpm2.portable.config import ModelConfig, load_config
from voxcpm2.portable.progress import PATCH_CAP, VoiceProgress
from voxcpm2.portable.text import style_line
from voxcpm2.portable.tokenize import LineTokenizer

_SCRATCH = 256 << 20


class Runtime:
    """The published checkout. A pack is read at open. An unpacked tree uploads on first use."""

    def __init__(
        self,
        path: Path,
        config: ModelConfig,
        *,
        owner: str,
        kv_cache: str,
        resident_percent: float | None,
        console: Any,
    ) -> None:
        self.path = path
        self.config = config
        self._owner = owner
        self.kv_cache = kv_cache
        self.resident_percent = resident_percent
        self.console = console
        self._session: Session | None = None
        self._store: Store | None = None
        self._decoder: Decoder | None = None
        self._tokenizer: LineTokenizer | None = None

    def ensure_open(self, request: Any) -> None:
        """Wait for prior releases, then open when the line's working set fits."""
        del request
        need = self.working_set_bytes(None)
        open_gate(need, detail="voice line")
        self._open()

    def working_set_bytes(self, request: Any) -> int:
        """Weight files plus the fixed scratch reservation."""
        del request
        return weight_bytes(self.path) + _SCRATCH

    def close(self) -> None:
        if (
            self._session is None
            and self._store is None
            and self._decoder is None
            and self._tokenizer is None
        ):
            return
        before, held = begin_release(self._owner)
        try:
            self._decoder = None
            if self._store is not None:
                self._store.close()
                self._store = None
            if self._session is not None:
                self._session.close()
                self._session = None
            self._tokenizer = None
        finally:
            finish_release(self._owner, before, held)

    def speak(
        self,
        text: str,
        *,
        persona: str,
        max_len: int | None = None,
        min_len: int = 2,
        steps: int = 10,
        seed: int | None = None,
        stop: Any = None,
        job: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """One waveform. ``persona`` is a parenthetical phrase in front of ``text``."""
        spoken = text.strip()
        if not spoken:
            message = "input is required"
            raise ValueError(message)
        line = style_line(spoken, persona)
        session, store, decoder, tokenizer = self._open()
        progress = VoiceProgress(self.console, job)
        try:
            samples = generate(
                session,
                store,
                decoder,
                tokenizer,
                self.config,
                line,
                max_len=PATCH_CAP if max_len is None else max_len,
                min_len=min_len,
                steps=steps,
                seed=seed,
                progress=progress,
                stop=stop,
            )
        except MediaPaused as paused:
            label = "sample paused" if paused.reason == "preempt" else "sample cancelled"
            progress.close(label)
            raise
        except Exception:
            progress.close(progress.summary(failed=True))
            raise
        progress.complete()
        summary = progress.summary()
        spent = time.perf_counter() - progress.started
        try:
            progress.note(
                f"speak {spent:.2f}s patches {progress.patches} decode {progress.decode_s:.2f}s"
            )
        finally:
            progress.close(summary)
        return samples

    def _open(self) -> tuple[Session, Store, Decoder, LineTokenizer]:
        ready = (
            self._session is not None
            and self._store is not None
            and self._decoder is not None
            and self._tokenizer is not None
        )
        if ready:
            return self._session, self._store, self._decoder, self._tokenizer
        tokenizer_path = self.path / "tokenizer.json"
        if not tokenizer_path.is_file():
            message = f"{self.path.name} has no tokenizer.json"
            raise FileNotFoundError(message)
        need = self.working_set_bytes(None)
        if memory.room() < need:
            message = f"a voice line does not fit in {memory.room()} bytes"
            raise RuntimeError(message)
        if pack_ready(self.path):
            return self._open_packed(tokenizer_path)
        vae_path = self.path / "audiovae.pth"
        weights = self.path / "model.safetensors"
        if (self.path / WEIGHTS_NAME).is_file() and not weights.is_file():
            message = f"{self.path.name} pack is incomplete"
            raise ValueError(message)
        if not vae_path.is_file():
            message = f"{self.path.name} has no audiovae.pth"
            raise FileNotFoundError(message)
        if not weights.is_file():
            message = f"{self.path.name} has no model.safetensors"
            raise FileNotFoundError(message)
        self._session = Session(self._owner)
        try:
            self._store = Store(weights)
            self._decoder = Decoder(
                self._session,
                vae_path,
                self.config.decoder_rates,
                self.config.sr_boundaries,
                self.config.out_sample_rate,
            )
            self._tokenizer = LineTokenizer(tokenizer_path)
        except Exception:
            self.close()
            raise
        return self._session, self._store, self._decoder, self._tokenizer

    def _open_packed(self, tokenizer_path: Path) -> tuple[Session, Store, Decoder, LineTokenizer]:
        body = load_pack(self.path)
        records = body["tensors"]
        if not isinstance(records, dict):
            message = f"{self.path.name} pack has no tensors"
            raise ValueError(message)
        blob = self.path / str(body["file"])
        self._session = Session(self._owner)
        try:
            handle, view = read_blob(blob, self._owner)
            self._store = Store.bind(view, records, handle)
            self._decoder = Decoder.from_records(
                self._session,
                view,
                records,
                handle,
                self.config.decoder_rates,
                self.config.sr_boundaries,
                self.config.out_sample_rate,
            )
            self._tokenizer = LineTokenizer(tokenizer_path)
        except Exception:
            self.close()
            raise
        return self._session, self._store, self._decoder, self._tokenizer


def open_device(
    path: str | Path,
    *,
    kv_cache: str = "e4m3",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | None = None,
    owner: str | None = None,
) -> Runtime:
    """Load ``config.json``. The tree has ``model.safetensors`` or ``weights.bin``."""
    checkout = Path(path)
    weights = checkout / "model.safetensors"
    blob = checkout / WEIGHTS_NAME
    if not weights.is_file() and not blob.is_file():
        message = f"{checkout.name} has no model.safetensors"
        raise FileNotFoundError(message)
    config_path = checkout / "config.json"
    if not config_path.is_file():
        message = f"{checkout.name} has no config.json"
        raise FileNotFoundError(message)
    memory.set_max_memory(max_memory)
    percent = None if resident_percent is None else float(resident_percent)
    return Runtime(
        checkout,
        load_config(config_path),
        owner=stamp_owner("voxcpm2", owner, id(checkout)),
        kv_cache=kv_cache,
        resident_percent=percent,
        console=console,
    )

