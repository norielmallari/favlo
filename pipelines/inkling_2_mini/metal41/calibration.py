"""Calibrate the dense fp4 planes: capture plane inputs, then round with GPTQ.

Sequences are packed to 2048 tokens and rendered with the checkpoint template.
Resolved software-agent traces come first, then coding instructions, then
function-calling dialogs. General chat fills only the remaining sequences.
Downloads are stored under ``artifacts/`` and are not committed. Nothing is
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
_QUOTAS = (("agent", 56), ("code", 40), ("tool", 24), ("chat", 8))
_PIECES = {"agent": 4, "code": 2, "tool": 2, "chat": 2}
_POOLS = {"agent": 32, "code": 400, "tool": 120, "chat": 40}
_SOURCES = {
    "agent": ("nvidia/Open-SWE-Traces", "v1.0", "sweagent", 10),
    "code": ("ise-uiuc/Magicoder-Evol-Instruct-110K", "default", "train", 100),
    "tool": ("NousResearch/hermes-function-calling-v1", "func_calling", "train", 20),
    "chat": ("HuggingFaceH4/ultrachat_200k", "default", "train_sft", 100),
}
_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
_FILES = {
    "agent": "open-swe-traces.json",
    "code": "magicoder-evol-instruct.json",
    "tool": "hermes-function-calling.json",
    "chat": "ultrachat-train-sft.json",
}
_ROWS = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def _artifact(name: str) -> Path:
    """Downloaded rows. The directory is gitignored."""
    root = Path(__file__).resolve().parents[3]
    return root / "artifacts" / "inkling_2_mini" / name


def _tagged(text: str, tag: str) -> list[Any] | None:
    needle = f"<{tag}>"
    close = f"</{tag}>"
    found: list[Any] = []
    cursor = 0
    while True:
        begin = text.find(needle, cursor)
        if begin < 0:
            return found
        stop = text.find(close, begin + len(needle))
        if stop < 0:
            return None
        raw = text[begin + len(needle) : stop].strip()
        try:
            found.append(json.loads(raw))
        except json.JSONDecodeError:
            return None
        cursor = stop + len(close)


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return None


def _agent_dialog(row: Any) -> dict[str, Any] | None:
    """One resolved permissive trajectory, or None."""
    if not isinstance(row, dict) or row.get("resolved") != 1:
        return None
    if row.get("license") not in _LICENSES:
        return None
    tools: list[dict[str, Any]] = []
    for raw in row.get("tools") or []:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if isinstance(raw, dict):
            tools.append(raw)
    messages = _openai_messages(row.get("messages"))
    if messages is None or not tools:
        return None
    return {"messages": messages, "tools": tools}


def _openai_messages(messages: Any) -> list[dict[str, Any]] | None:
    if not isinstance(messages, list):
        return None
    pending: list[str] = []
    kept: list[dict[str, Any]] = []
    number = 0
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        calls = message.get("tool_calls")
        if role == "assistant" and isinstance(calls, list) and calls:
            shaped = []
            for call in calls:
                if not isinstance(call, dict):
                    return None
                function = call.get("function")
                if not isinstance(function, dict) or not function.get("name"):
                    return None
                number += 1
                ident = str(call.get("id") or f"call-{number}")
                shaped.append(
                    {
                        "id": ident,
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": function.get("arguments")
                            if function.get("arguments") is not None
                            else {},
                        },
                    }
                )
                pending.append(ident)
            item: dict[str, Any] = {"role": "assistant", "tool_calls": shaped}
            content = _text(message.get("content"))
            if content is None:
                return None
            if content.strip():
                item["content"] = content
            kept.append(item)
            continue
        if role == "tool":
            if not pending:
                return None
            content = _text(message.get("content"))
            if content is None:
                return None
            kept.append({"role": "tool", "tool_call_id": pending.pop(0), "content": content})
            continue
        if role not in {"system", "user", "assistant"}:
            continue
        content = _text(message.get("content"))
        if content is None:
            return None
        if role in {"system", "user"} and _EMAIL.search(content):
            return None
        if content.strip():
            kept.append({"role": role, "content": content})
    while pending and kept and kept[-1].get("role") == "assistant":
        calls = kept[-1].get("tool_calls") or []
        extra = len(pending)
        if len(calls) <= extra:
            pending = pending[: extra - len(calls)]
            kept.pop()
        else:
            kept[-1]["tool_calls"] = calls[: len(calls) - extra]
            pending = []
    if pending or len(kept) < 2:
        return None
    return kept


def _hermes_dialog(conversations: Any, tools: Any) -> dict[str, Any] | None:
    """One Hermes function-calling dialog, or None."""
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None
    if not isinstance(conversations, list) or not isinstance(tools, list) or not tools:
        return None
    pending: list[str] = []
    kept: list[dict[str, Any]] = []
    number = 0
    for turn in conversations:
        if not isinstance(turn, dict):
            return None
        who = turn.get("from")
        value = turn.get("value")
        if not isinstance(value, str) or _EMAIL.search(value):
            return None
        if who == "system":
            continue
        if who == "human":
            if pending or not value.strip():
                return None
            kept.append({"role": "user", "content": value})
            continue
        if who == "gpt":
            calls = _tagged(value, "tool_call")
            if calls is None:
                return None
            if calls:
                shaped = []
                for call in calls:
                    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                        return None
                    number += 1
                    ident = f"call-{number}"
                    shaped.append(
                        {
                            "id": ident,
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call.get("arguments") or {},
                            },
                        }
                    )
                    pending.append(ident)
                kept.append({"role": "assistant", "tool_calls": shaped})
            elif value.strip():
                kept.append({"role": "assistant", "content": value.strip()})
            continue
        if who == "tool":
            responses = _tagged(value, "tool_response")
            if not responses or len(responses) > len(pending):
                return None
            for response in responses:
                if not isinstance(response, dict):
                    return None
                content = _text(response.get("content", ""))
                if content is None:
                    return None
                kept.append({"role": "tool", "tool_call_id": pending.pop(0), "content": content})
            continue
        return None
    if pending or len(kept) < 2:
        return None
    return {"messages": kept, "tools": tools}


def _code_dialog(instruction: Any, response: Any) -> list[dict[str, str]] | None:
    if not isinstance(instruction, str) or not isinstance(response, str):
        return None
    if _EMAIL.search(instruction) or _EMAIL.search(response):
        return None
    if len(instruction) + len(response) < 800:
        return None
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]


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


def _save(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(rows, stream)
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _page(dataset: str, config: str, split: str, offset: int, length: int) -> list[Any]:
    url = _ROWS.format(dataset=dataset, config=config, split=split, offset=offset, length=length)
    request = urllib.request.Request(url, headers={"User-Agent": "favlo"})
    with urllib.request.urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode())
    return body.get("rows") or []


def _collect(kind: str) -> list[Any]:
    dataset, config, split, length = _SOURCES[kind]
    need = _POOLS[kind]
    found: list[Any] = []
    offset = 0
    while len(found) < need and offset < 8000:
        rows = _page(dataset, config, split, offset, length)
        if not rows:
            break
        for item in rows:
            row = item.get("row") if isinstance(item, dict) else None
            kept = _keep(kind, row)
            if kept is not None:
                found.append(kept)
            if len(found) == need:
                break
        offset += len(rows)
    if len(found) < need:
        message = f"{kind} returned {len(found)} rows, need {need}"
        raise RuntimeError(message)
    return found


def _keep(kind: str, row: Any) -> Any:
    if not isinstance(row, dict):
        return None
    if kind == "agent":
        return _agent_dialog(row)
    if kind == "code":
        return _code_dialog(row.get("instruction"), row.get("response"))
    if kind == "tool":
        return _hermes_dialog(row.get("conversations"), row.get("tools"))
    return _accept(row.get("messages"))


def _rows(kind: str) -> list[Any]:
    path = _artifact(_FILES[kind])
    need = _POOLS[kind]
    if path.is_file():
        found = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(found, list) and len(found) >= need:
            return found[:need]
    found = _collect(kind)
    _save(path, found)
    return found


def chat_rows() -> list[list[dict[str, str]]]:
    """The general-chat remainder, fetched once into ``artifacts/``."""
    return _rows("chat")


def _call_start(root: Path, ids: list[int]) -> int | None:
    """Token index a little before the first tool call, or None."""
    from inkling_2_mini.portable.chat import decode_text
    from inkling_2_mini.portable.markup import TOOL

    if TOOL not in decode_text(root, ids):
        return None
    lo = 0
    hi = len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if TOOL in decode_text(root, ids[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return max(0, lo - 128)


def _window(root: Path, kind: str, row: Any, cap: int) -> list[int]:
    from inkling_2_mini.portable.chat import encode_chat

    messages = row["messages"] if isinstance(row, dict) else row
    tools = row.get("tools") if isinstance(row, dict) else None
    ids = encode_chat(root, messages, tools, False, effort="none", opening="")
    start = _call_start(root, ids) if kind == "agent" else 0
    if start is None:
        return []
    return ids[start : start + _PIECES[kind] * cap]


def _pack(root: Path, kind: str, quota: int, cap: int) -> list[list[int]]:
    pending: list[int] = []
    sequences: list[list[int]] = []
    for row in _rows(kind):
        if len(sequences) >= quota:
            break
        try:
            ids = _window(root, kind, row, cap)
        except (RuntimeError, json.JSONDecodeError, ValueError, KeyError):
            continue
        pending.extend(ids)
        while len(pending) >= cap and len(sequences) < quota:
            sequences.append(pending[:cap])
            pending = pending[cap:]
    return sequences


def _sequences(root: Path, limit: int, log: Any) -> list[list[int]]:
    cap = SEQUENCE_LENGTH if limit <= 0 else int(limit)
    sequences: list[list[int]] = []
    for kind, quota in _QUOTAS:
        got = _pack(root, kind, quota, cap)
        if len(got) < quota:
            message = f"{kind} packed {len(got)} sequences, need {quota}"
            raise RuntimeError(message)
        for ids in got:
            sequences.append(ids)
            log(f"sequence {len(sequences)} {kind} tokens {len(ids)}")
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
