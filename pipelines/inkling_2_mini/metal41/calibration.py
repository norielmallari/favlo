"""Calibrate the dense fp4 planes: capture plane inputs, then round with GPTQ.

Chat rows come from ``HuggingFaceH4/ultrachat_200k`` ``train_sft`` and are
stored under ``artifacts/``. Tool transcripts live in this file. Each row is
rendered with the checkpoint template and cut at 2048 tokens. Nothing is
generated. The capture prefill sums ``X^T X``. Each finished sequence is
written before the next one starts. An out-of-memory capture closes, reopens
with fewer rows, and continues. Each rounded plane is written before the next
one starts. The runtime then closes, and every dense plane is rounded against
the Hessian of its own inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from inkling_2_mini.metal41.gptq import (
    PACK_FORMAT,
    PACK_NAME,
    Capture,
    config_digest,
    dequant,
    gptq_fp4,
    write_pack,
)
from inkling_2_mini.metal41.layout import HIDDEN, LAYERS, Q_DIM, UNPADDED_VOCAB
from inkling_2_mini.metal41.weights import (
    ATTN_ROWS,
    HEAD_ROWS,
    ROW_K,
    ROW_V,
    Tensors,
    _pad_rows,
    _quantize_fp4,
    attn_matrix,
    checkpoint_file,
    head_matrix,
    wo_matrix,
)

HESSIANS_NAME = "dense-fp4-inputs.npz"
ROUND_PARTIAL = "dense-fp4-partial.npz"
SEQUENCE_LENGTH = 2048
_CAPTURE_CONTEXT = 4096
_ROW_STEPS = (2048, 1024, 512, 256)
_CHAT_SEQUENCES = 112
_POOL = 400
_ROWS = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=HuggingFaceH4/ultrachat_200k&config=default&split=train_sft"
    "&offset={offset}&length=100"
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative path."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite and return the summary line.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string", "description": "Test name filter."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_event",
            "description": "Add an event to the calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string", "description": "ISO date."},
                    "minutes": {"type": "integer"},
                },
                "required": ["title", "date"],
            },
        },
    },
]

_TOOL_TURNS: list[tuple[str, str, dict[str, Any], str, str]] = [
    (
        "What's the weather in Lisbon right now?",
        "get_weather",
        {"city": "Lisbon", "unit": "celsius"},
        "18 C and clear.",
        "Lisbon is 18 C and clear.",
    ),
    (
        "Open README.md and tell me what the project does.",
        "read_file",
        {"path": "README.md"},
        "A local inference server.",
        "The project is a local inference server.",
    ),
    (
        "Run the tests named parser and report the result.",
        "run_tests",
        {"pattern": "parser"},
        "12 passed.",
        "The parser tests passed, 12 of them.",
    ),
    (
        "Put a 30 minute dentist visit on 2026-03-14.",
        "add_event",
        {"title": "dentist", "date": "2026-03-14", "minutes": 30},
        "saved.",
        "The dentist visit is on 2026-03-14 for 30 minutes.",
    ),
    (
        "Read the file src/main.py.",
        "read_file",
        {"path": "src/main.py"},
        "def main():\n    return 0",
        "src/main.py defines main and returns 0.",
    ),
    (
        "Is it colder in Oslo or in Helsinki today? Use fahrenheit.",
        "get_weather",
        {"city": "Oslo", "unit": "fahrenheit"},
        "30 F.",
        "Oslo is 30 F. Helsinki needs a second lookup.",
    ),
    (
        "What is the weather in Kyoto tomorrow morning?",
        "get_weather",
        {"city": "Kyoto", "unit": "celsius"},
        "9 C and cloudy.",
        "Kyoto tomorrow morning is 9 C and cloudy.",
    ),
    (
        "Schedule team sync for 2026-05-02, 45 minutes.",
        "add_event",
        {"title": "team sync", "date": "2026-05-02", "minutes": 45},
        "saved.",
        "Team sync is on 2026-05-02 for 45 minutes.",
    ),
    (
        "Read notes/plan.md.",
        "read_file",
        {"path": "notes/plan.md"},
        "Ship the pack.",
        "notes/plan.md says to ship the pack.",
    ),
    (
        "Run the full test suite.",
        "run_tests",
        {"pattern": ""},
        "400 passed.",
        "The full suite passed, 400 tests.",
    ),
    (
        "Add a one hour review on 2026-06-01.",
        "add_event",
        {"title": "review", "date": "2026-06-01", "minutes": 60},
        "saved.",
        "The review is on 2026-06-01 for one hour.",
    ),
    (
        "How warm is Cairo this afternoon?",
        "get_weather",
        {"city": "Cairo", "unit": "celsius"},
        "31 C.",
        "Cairo this afternoon is 31 C.",
    ),
    (
        "Show me the text of Makefile.",
        "read_file",
        {"path": "Makefile"},
        "test:\n\tpytest",
        "The Makefile runs pytest for test.",
    ),
    (
        "Run tests matching chat.",
        "run_tests",
        {"pattern": "chat"},
        "8 passed, 1 failed.",
        "Eight chat tests passed and one failed.",
    ),
    (
        "Put lunch on 2026-04-02 for 45 minutes.",
        "add_event",
        {"title": "lunch", "date": "2026-04-02", "minutes": 45},
        "saved.",
        "Lunch is on 2026-04-02 for 45 minutes.",
    ),
    (
        "Read the weather in Reykjavik in fahrenheit.",
        "get_weather",
        {"city": "Reykjavik", "unit": "fahrenheit"},
        "41 F and windy.",
        "Reykjavik is 41 F and windy.",
    ),
]


def _artifact() -> Path:
    """Downloaded chat rows. The directory is gitignored."""
    root = Path(__file__).resolve().parents[3]
    return root / "artifacts" / "inkling_2_mini" / "ultrachat-train-sft.json"


def _accept(messages: Any) -> list[dict[str, str]] | None:
    """A user/assistant dialog long enough to calibrate, or None."""
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    kept: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            return None
        if _EMAIL.search(content):
            return None
        kept.append({"role": role, "content": content})
    if sum(len(item["content"]) for item in kept) < 800:
        return None
    return kept


def _download(path: Path) -> None:
    """Fetch ``_POOL`` train_sft rows into ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    found: list[list[dict[str, str]]] = []
    offset = 0
    while len(found) < _POOL and offset < 5000:
        request = urllib.request.Request(
            _ROWS.format(offset=offset), headers={"User-Agent": "favlo"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode())
        rows = body.get("rows") or []
        if not rows:
            break
        for item in rows:
            row = item.get("row") if isinstance(item, dict) else None
            messages = _accept(row.get("messages") if isinstance(row, dict) else None)
            if messages is not None:
                found.append(messages)
            if len(found) == _POOL:
                break
        offset += len(rows)
    if len(found) < _POOL:
        message = f"ultrachat returned {len(found)} rows, need {_POOL}"
        raise RuntimeError(message)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(found, handle)
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def chat_rows() -> list[list[dict[str, str]]]:
    """The downloaded dialogs, fetched once into ``artifacts/``."""
    path = _artifact()
    if not path.is_file():
        _download(path)
    found = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(found, list) or len(found) < _POOL:
        _download(path)
        found = json.loads(path.read_text(encoding="utf-8"))
    rows = found[:_POOL]
    if len(rows) < _POOL:
        message = f"ultrachat file has {len(rows)} rows"
        raise RuntimeError(message)
    return rows


def _tool_dialogs() -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    dialogs = []
    for user, name, args, result, reply in _TOOL_TURNS:
        call = "call-1"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are an agent. Use the tools when they help."},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": call, "type": "function", "function": {"name": name, "arguments": args}}
                ],
            },
            {"role": "tool", "tool_call_id": call, "content": result},
            {"role": "assistant", "content": reply},
        ]
        dialogs.append((messages, _TOOLS))
    return dialogs


def _sequences(root: Path, limit: int, log: Any) -> list[list[int]]:
    from inkling_2_mini.metal41.chat import encode_chat

    cap = SEQUENCE_LENGTH if limit <= 0 else int(limit)
    sequences: list[list[int]] = []
    pending: list[int] = []
    for messages in chat_rows():
        if len(sequences) == _CHAT_SEQUENCES:
            break
        pending.extend(encode_chat(root, messages, None, False, effort="none", opening=""))
        while len(pending) >= cap and len(sequences) < _CHAT_SEQUENCES:
            sequences.append(pending[:cap])
            pending = pending[cap:]
    if len(sequences) < _CHAT_SEQUENCES:
        message = f"ultrachat packed {len(sequences)} sequences, need {_CHAT_SEQUENCES}"
        raise RuntimeError(message)
    for messages, tools in _tool_dialogs():
        ids = encode_chat(root, messages, tools, False, effort="none", opening="")
        sequences.append(ids[:cap])
    for index, ids in enumerate(sequences):
        log(f"sequence {index + 1} tokens {len(ids)}")
    return sequences


def _source(sequences: list[list[int]], limit: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(limit)).encode())
    for sequence in sequences:
        digest.update(np.asarray(sequence, dtype=np.uint32).tobytes())
    return digest.hexdigest()


def _open_saved(path: Path, source: str) -> Capture | None:
    if not path.is_file():
        return None
    loaded = Capture.load(path)
    if loaded.source != source:
        return None
    return loaded


def _out_of_memory(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "insufficient memory" in text


def _load_round(path: Path, source: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path) as held:
        stored = str(held["source"]) if "source" in held.files else ""
        if stored != source:
            return {}
        return {key: np.array(held[key]) for key in held.files if key != "source"}


def _store_round(path: Path, source: str, tensors: dict[str, np.ndarray]) -> None:
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    try:
        np.savez(name, source=np.array(source), **tensors)
        os.replace(name, path)
    except Exception:
        os.unlink(name)
        raise


def _unpack(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    nibbles = np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(packed.shape[0], -1)
    factor = np.repeat(np.exp2(scales.astype(np.float32) - 127.0), 32, axis=1)
    return dequant(nibbles) * factor


def weighted_error(
    matrix: np.ndarray, packed: np.ndarray, scales: np.ndarray, hessian: np.ndarray
) -> float:
    """Relative output error ``tr(E H E^T) / tr(W H W^T)`` on up to 512 rows."""
    step = max(1, matrix.shape[0] // 512)
    rows = slice(0, matrix.shape[0], step)
    weight = matrix[rows].astype(np.float64)
    error = weight - _unpack(packed[rows], scales[rows]).astype(np.float64)
    return float(np.sum((error @ hessian) * error) / np.sum((weight @ hessian) * weight))


def _round(
    name: str, matrix: np.ndarray, hessian: np.ndarray, log: Any
) -> tuple[np.ndarray, np.ndarray]:
    started = time.perf_counter()
    packed, scales = gptq_fp4(matrix, hessian)
    nearest = _quantize_fp4(matrix)
    before = weighted_error(matrix, *nearest, hessian)
    after = weighted_error(matrix, packed, scales, hessian)
    log(
        f"{name} output error nearest {before**0.5:.4f} gptq {after**0.5:.4f} "
        f"{time.perf_counter() - started:.1f}s"
    )
    return packed, scales


def _attention(
    tensors: Tensors, capture: Capture, index: int, log: Any
) -> tuple[np.ndarray, np.ndarray]:
    matrix = attn_matrix(tensors, index)
    parts = []
    for stream, lo, hi in (("q", 0, ROW_K), ("k", ROW_K, ROW_V), ("v", ROW_V, ATTN_ROWS)):
        parts.append(
            _round(f"layer {index} {stream}", matrix[lo:hi], capture.hessian(index, stream), log)
        )
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _capture(folder: Path, limit: int, log: Any) -> Capture:
    from inkling_2_mini.metal41.forward import InklingRuntime

    sequences = _sequences(folder, limit, log)
    source = _source(sequences, limit)
    saved = folder / HESSIANS_NAME
    capture = _open_saved(saved, source) or Capture()
    capture.source = source
    if capture.done > len(sequences):
        capture.done = len(sequences)
    if capture.done:
        log(f"resuming capture at {min(capture.done + 1, len(sequences))}/{len(sequences)}")
    if capture.done >= len(sequences):
        return capture
    step = 0
    while capture.done < len(sequences):
        runtime = None
        try:
            runtime = InklingRuntime(
                folder,
                owner="inkling_2_mini:calibrate",
                calibrated=False,
                context_limit=_CAPTURE_CONTEXT,
                rows_limit=_ROW_STEPS[step],
            )
            runtime._prefix_scratch = "inkling-calibrate"
            runtime._prefix_budget = 1
            while capture.done < len(sequences):
                scratch = Capture()
                runtime.capture_prefill(sequences[capture.done], scratch)
                capture.absorb(scratch)
                capture.done += 1
                capture.save(saved)
                log(f"captured {capture.done}/{len(sequences)} tokens {capture.tokens}")
        except (RuntimeError, MemoryError) as exc:
            if not _out_of_memory(exc) or step + 1 >= len(_ROW_STEPS):
                raise
            step += 1
            log(
                f"out of memory at sequence {capture.done + 1}, "
                f"reopening with rows {_ROW_STEPS[step]}"
            )
        else:
            break
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as exc:
                    log(f"close failed after capture: {exc}")
    return capture


def calibrate(
    folder: Path, *, limit: int = SEQUENCE_LENGTH, reuse: bool = False, log: Any = print
) -> Path:
    """Write the GPTQ pack. ``reuse`` rounds again from the saved input sums."""
    folder = Path(folder)
    saved = folder / HESSIANS_NAME
    if reuse and saved.is_file():
        capture = Capture.load(saved)
        log(f"loaded {saved.name} from {capture.tokens} calibration tokens")
    else:
        capture = _capture(folder, limit, log)
    tensors = Tensors(checkpoint_file(folder))
    partial = folder / ROUND_PARTIAL
    out = _load_round(partial, capture.source)
    if out:
        log(f"resuming round with {len(out)} saved tensors")
    try:
        for index in range(LAYERS):
            data = f"layers.{index}.attn.data"
            if data not in out:
                out[data], out[f"layers.{index}.attn.scale"] = _attention(
                    tensors, capture, index, log
                )
                _store_round(partial, capture.source, out)
            wo_name = f"layers.{index}.wo.data"
            if wo_name not in out:
                wo = wo_matrix(tensors, index)
                out[wo_name], out[f"layers.{index}.wo.scale"] = _round(
                    f"layer {index} wo", wo, capture.hessian(index, "wo"), log
                )
                _store_round(partial, capture.source, out)
            for stream in ("shared", "q", "k", "v", "wo"):
                capture.sums.pop((index, stream), None)
        if "unembed.data" not in out:
            head = head_matrix(tensors)
            packed, scales = _round("head", head, capture.hessian(-1, "head"), log)
            out["unembed.data"] = _pad_rows(packed, HEAD_ROWS)
            out["unembed.scale"] = _pad_rows(scales, HEAD_ROWS)
            _store_round(partial, capture.source, out)
    finally:
        tensors.close()
    if out["unembed.data"].shape != (HEAD_ROWS, HIDDEN // 2) or UNPADDED_VOCAB > HEAD_ROWS:
        message = "head pack shape is wrong"
        raise RuntimeError(message)
    if out["layers.0.wo.data"].shape != (HIDDEN, Q_DIM // 2):
        message = "wo pack shape is wrong"
        raise RuntimeError(message)
    path = Path(folder) / PACK_NAME
    write_pack(
        path,
        out,
        {"format": PACK_FORMAT, "config": config_digest(folder), "tokens": str(capture.tokens)},
    )
    log(f"wrote {path.name} from {capture.tokens} calibration tokens")
    partial.unlink(missing_ok=True)
    return path


__all__ = ["SEQUENCE_LENGTH", "calibrate", "chat_rows", "weighted_error"]
