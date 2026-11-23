"""Calibrate NVFP4 experts: capture their inputs, then round with FOEM.

Sequences are packed to 2048 tokens from ``artifacts/calibration/`` and rendered
with the checkpoint template. SWE traces come first, then coding instructions,
then function-calling dialogs. General chat fills the remaining sequences.
Nothing is generated. The capture prefill sums ``X^T X`` for each expert on
the device. Several layers share one prefill, as many as the budget holds.
Those layers are rounded before the next group starts.
Attention and the output projection are nearest fp8 and are not part of this
capture.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from favlo.calibration_corpus import QUOTAS as _QUOTAS
from inkling_2_mini.metal41.codec import dequant_nvfp4_matrix
from inkling_2_mini.metal41.gptq import (
    BETA,
    Capture,
    _inverse_factor_stable,
    config_digest,
    dequant,
    foem_device_matches,
    foem_fp4_encode,
    foem_fp4_read,
    pack_nibbles,
)
from inkling_2_mini.metal41.layout import (
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    NVFP4_LAYERS,
    OUTER_SCALE_DENOMINATOR,
    stored_k,
)
from inkling_2_mini.metal41.weights import Tensors, _quantize_fp4, checkpoint_file, ensure_dense

HESSIANS_NAME = "dense-fp4-inputs.npz"
ROUND_PARTIAL = "dense-fp4-partial.npz"
SEQUENCE_LENGTH = 2048
_CAPTURE_CONTEXT = 4096
_ROW_STEPS = (2048, 1024, 512, 256)
_FOEM_WAVE = 16
_PIECES = {"swe": 4, "code": 2, "tool": 2, "chat": 2}
_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


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


def _rows(kind: str) -> list[Any]:
    from favlo.calibration_corpus import load

    return load(kind)


def chat_rows() -> list[Any]:
    """The general-chat rows from the shared calibration corpus."""
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
    start = _call_start(root, ids) if kind == "swe" else 0
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


def _keep_round(
    name: str,
    matrix: np.ndarray,
    hessian: np.ndarray,
    nearest: tuple[np.ndarray, np.ndarray],
    foem: tuple[np.ndarray, np.ndarray],
    log: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the lower output error. The check runs on the host."""
    before = weighted_error(matrix, *nearest, hessian)
    after = weighted_error(matrix, *foem, hessian)
    kept = "foem"
    packed, scales = foem
    if after > before:
        packed, scales = nearest
        kept = "nearest"
    log(f"{name} output error nearest {before**0.5:.4f} foem {after**0.5:.4f} kept {kept}")
    return packed, scales


_W13_K = HIDDEN
_W2_K = stored_k(INTERMEDIATE)
_GRAM_BYTES = EXPERTS * (_W13_K * _W13_K + _W2_K * _W2_K) * 4


class ExpertGrams:
    """``X^T X`` for one NVFP4 layer. ``up`` feeds w13 and ``down`` feeds w2."""

    def __init__(self, layer: int) -> None:
        self.layer = int(layer)
        self.up = np.zeros((EXPERTS, _W13_K, _W13_K), dtype=np.float32)
        self.down = np.zeros((EXPERTS, _W2_K, _W2_K), dtype=np.float32)
        self.tokens = 0
        self.done = 0
        self.saved_tokens = 0
        self.saved_done = 0
        self.source = ""
        self.up_buf: Any = None
        self.down_buf: Any = None

    def add_up(self, hidden: np.ndarray, ids: np.ndarray) -> None:
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        if hidden.ndim != 2 or hidden.shape[1] != _W13_K:
            message = f"up input shape {hidden.shape} is not tokens by {_W13_K}"
            raise ValueError(message)
        for expert in range(EXPERTS):
            rows = hidden[np.any(ids == expert, axis=1)]
            if len(rows) == 0:
                continue
            self.up[expert] += rows.T @ rows

    def add_down(
        self,
        gated: np.ndarray,
        expert_of: np.ndarray,
        slot_n: np.ndarray,
        row_base: np.ndarray,
    ) -> None:
        gated = np.asarray(gated, dtype=np.float32)
        if gated.ndim != 2 or gated.shape[1] != _W2_K:
            message = f"down input shape {gated.shape} is not rows by {_W2_K}"
            raise ValueError(message)
        for slot, expert in enumerate(np.asarray(expert_of).reshape(-1)):
            count = int(slot_n[slot])
            if count <= 0:
                continue
            base = int(row_base[slot])
            rows = gated[base : base + count]
            self.down[int(expert)] += rows.T @ rows

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.npz")
        np.savez(
            partial,
            up=self.up,
            down=self.down,
            tokens=np.int64(self.tokens),
            done=np.int64(self.done),
            layer=np.int64(self.layer),
            source=np.array(self.source),
        )
        partial.replace(path)

    @classmethod
    def load(cls, path: Path, source: str) -> ExpertGrams | None:
        if not path.is_file():
            return None
        with np.load(path) as held:
            found = str(held["source"]) if "source" in held.files else ""
            if found != source:
                return None
            grams = cls.__new__(cls)
            grams.layer = int(held["layer"])
            grams.up = np.array(held["up"], dtype=np.float32)
            grams.down = np.array(held["down"], dtype=np.float32)
            grams.tokens = int(held["tokens"])
            grams.done = int(held["done"])
            grams.saved_tokens = grams.tokens
            grams.saved_done = grams.done
            grams.source = found
            grams.up_buf = None
            grams.down_buf = None
        if grams.up.shape != (EXPERTS, _W13_K, _W13_K):
            return None
        if grams.down.shape != (EXPERTS, _W2_K, _W2_K):
            return None
        return grams

    def bind(self, session: Any) -> None:
        """Allocate this layer's Grams on the device. A resumed Gram is uploaded."""
        from favlo.memory import memory

        up_n = EXPERTS * _W13_K * _W13_K
        down_n = EXPERTS * _W2_K * _W2_K
        self.up_buf = session.zeros(up_n * 4)
        self.down_buf = session.zeros(down_n * 4)
        if self.saved_done:
            memory.store(self.up_buf, 0, np.ascontiguousarray(self.up).tobytes())
            memory.store(self.down_buf, 0, np.ascontiguousarray(self.down).tobytes())

    def snapshot(self, session: Any) -> tuple[np.ndarray, np.ndarray]:
        """Read the device Grams without moving the resume point."""
        up_n = EXPERTS * _W13_K * _W13_K
        down_n = EXPERTS * _W2_K * _W2_K
        up = session.download_f32(self.up_buf, up_n).reshape(EXPERTS, _W13_K, _W13_K)
        down = session.download_f32(self.down_buf, down_n).reshape(EXPERTS, _W2_K, _W2_K)
        return up, down

    def keep(self, up: np.ndarray, down: np.ndarray) -> None:
        """Store a snapshot and mark it as the resume point."""
        self.up = up
        self.down = down
        self.saved_done = self.done
        self.saved_tokens = self.tokens

    def rewind(self) -> None:
        """Drop sequences that were not copied off the device."""
        self.done = self.saved_done
        self.tokens = self.saved_tokens


def _capture_plan(limit_layers: int, limit_wave: int, rows: int) -> tuple[int, int]:
    """Size the resident layers and the concurrent expert wave from leftover RAM.

    One layer Gram is the floor. A full wave is reserved before a second layer
    is added, and any bytes still free become a wider wave, up to every expert.
    """
    from favlo.memory import memory

    room = min(memory.room(), max(int(memory.wire_room()), 0))
    usable = max(0, room - (2 << 30))
    pad = (int(rows) + 63) & ~63
    per_expert = HIDDEN * pad * 2 + HIDDEN * HIDDEN * 4
    cap_layers = max(1, int(limit_layers))
    cap_wave = max(1, min(int(limit_wave), EXPERTS))
    layers = 1
    if usable >= _GRAM_BYTES + cap_wave * per_expert:
        layers = max(1, min(cap_layers, (usable - cap_wave * per_expert) // _GRAM_BYTES))
    left = usable - layers * _GRAM_BYTES
    wave = 1
    if left >= per_expert:
        wave = max(1, min(cap_wave, left // per_expert))
    return layers, wave


def _layer_label(group: list[ExpertGrams]) -> str:
    first = group[0].layer
    last = group[-1].layer
    if first == last:
        return str(first)
    return f"{first}-{last}"


def _capture_group(
    folder: Path, cohort: list[ExpertGrams], sequences: list[list[int]], log: Any
) -> list[ExpertGrams]:
    """Prefill until every Gram in the resident group has seen every sequence."""
    from inkling_2_mini.metal41.forward import InklingRuntime

    if not cohort or cohort[0].done >= len(sequences):
        return list(cohort)
    step = 0
    limit_layers = len(cohort)
    limit_wave = EXPERTS
    group = cohort[:1]
    while group[0].done < len(sequences):
        runtime = None
        try:
            runtime = InklingRuntime(
                folder,
                owner="inkling_2_mini:calibrate",
                calibrated=False,
                context_limit=_CAPTURE_CONTEXT,
                rows_limit=_ROW_STEPS[step],
            )
            if not runtime.atlas.complete:
                message = "expert calibration needs the complete expert atlas"
                raise RuntimeError(message)
            fit, wave = _capture_plan(limit_layers, limit_wave, runtime.rows)
            limit_layers = fit
            limit_wave = wave
            group = cohort[:fit]
            runtime._gram_wave = wave
            for grams in group:
                grams.bind(runtime.session)
            runtime._prefix_scratch = "inkling-calibrate"
            runtime._prefix_budget = 1
            label = _layer_label(group)
            log(f"layers {label} expert wave {wave}")
            resident = {grams.layer: grams for grams in group}
            while group[0].done < len(sequences):
                runtime.capture_experts(sequences[group[0].done], resident)
                log(
                    f"layers {label} captured {group[0].done}/{len(sequences)} "
                    f"tokens {group[0].tokens}"
                )
            copies = [grams.snapshot(runtime.session) for grams in group]
            for grams, (up, down) in zip(group, copies, strict=True):
                grams.keep(up, down)
                grams.save(folder / "expert-foem" / f"layer-{grams.layer}-inputs.npz")
            log(f"layers {label} saved")
        except (RuntimeError, MemoryError) as exc:
            for grams in group:
                grams.rewind()
            if not _out_of_memory(exc) or (
                limit_layers == 1 and limit_wave == 1 and step + 1 >= len(_ROW_STEPS)
            ):
                raise
            if limit_wave > 1:
                limit_wave = max(1, limit_wave // 2)
                log(f"out of memory, expert wave {limit_wave}")
            elif limit_layers > 1:
                limit_layers -= 1
                log(f"out of memory, capturing {limit_layers} layers")
            else:
                step += 1
                log(f"out of memory, reopening with rows {_ROW_STEPS[step]}")
        else:
            break
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as exc:
                    log(f"close failed after capture: {exc}")
    return group


def _byte_tensor(tensors: Tensors, name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw, info = tensors.raw(name)
    if tuple(info["shape"]) != shape or info["dtype"] not in {"U8", "F8_E4M3"}:
        message = f"{name} is not 1-byte {shape}"
        raise ValueError(message)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def _projection(
    tensors: Tensors, layer: int, proj: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"model.llm.layers.{layer}.mlp.experts.{proj}"
    rows = INTERMEDIATE * 2 if proj == "w13" else HIDDEN
    cols = HIDDEN if proj == "w13" else INTERMEDIATE
    packed = _byte_tensor(tensors, f"{stem}_weight", (EXPERTS, rows, cols // 2))
    scales = _byte_tensor(tensors, f"{stem}_weight_scale", (EXPERTS, rows, cols // 16))
    amax = tensors.vector(f"{stem}_weight_amax", (EXPERTS,))
    denom = tensors.vector(f"{stem}_weight_outer_scale_denominator", (1,))
    if float(denom[0]) != OUTER_SCALE_DENOMINATOR:
        message = f"{stem} outer scale denominator is not {OUTER_SCALE_DENOMINATOR}"
        raise ValueError(message)
    return packed, scales, (amax / denom[0]).astype(np.float32)


def _expert_matrix(
    packed: np.ndarray, scales: np.ndarray, global_scale: float, rows: int, cols: int
) -> np.ndarray:
    values = dequant_nvfp4_matrix(packed, scales, global_scale)
    wide = stored_k(cols)
    if wide == cols:
        return values
    out = np.zeros((rows, wide), dtype=np.float32)
    out[:, :cols] = values
    return out


def _round_projection(
    layer: int,
    proj: str,
    packed_all: np.ndarray,
    scales_all: np.ndarray,
    global_scale: np.ndarray,
    hessian_rows: np.ndarray,
    rows: int,
    cols: int,
    log: Any,
    session: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Round every expert. The host factors the next expert while a wave is on the device."""
    wide = stored_k(cols)
    data = np.zeros((EXPERTS, rows, wide // 2), dtype=np.uint8)
    scale_out = np.zeros((EXPERTS, rows, wide // 32), dtype=np.uint8)
    prepared: queue.Queue = queue.Queue()
    failures: list[BaseException] = []

    def produce() -> None:
        try:
            for expert in range(EXPERTS):
                name = f"layer {layer} {proj} expert {expert}"
                hessian = hessian_rows[expert]
                matrix = _expert_matrix(
                    packed_all[expert],
                    scales_all[expert],
                    float(global_scale[expert]),
                    rows,
                    cols,
                )
                nearest = _quantize_fp4(matrix)
                if float(np.trace(hessian)) <= 0.0:
                    prepared.put(("skip", expert, name, nearest))
                    continue
                upper, dead = _inverse_factor_stable(np.asarray(hessian, dtype=np.float64), 0.01)
                prepared.put(
                    (
                        "gpu",
                        expert,
                        name,
                        matrix,
                        upper,
                        np.ascontiguousarray(dead, dtype=np.uint8),
                        nearest,
                        hessian,
                    )
                )
        except BaseException as exc:
            failures.append(exc)
        finally:
            prepared.put(None)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    pending: list[tuple[Any, ...]] = []
    inflight: list[tuple[Any, list[tuple[Any, ...]]]] = []
    opened = False

    def finish_oldest() -> None:
        session.wait_oldest()
        job, wave = inflight.pop(0)
        nibbles, scales = foem_fp4_read(session, job)
        for index, (expert, name, matrix, hessian, nearest) in enumerate(wave):
            packed = pack_nibbles(nibbles[index])
            chosen, held = _keep_round(
                name, matrix, hessian, nearest, (packed, scales[index]), log
            )
            data[expert, :, : chosen.shape[1]] = chosen
            scale_out[expert, :, : held.shape[1]] = held

    def submit(force: bool) -> None:
        nonlocal opened
        if not pending or (len(pending) < _FOEM_WAVE and not force):
            return
        wave = pending[:_FOEM_WAVE]
        del pending[:_FOEM_WAVE]
        count = len(wave)
        sample = wave[0][2]
        height, inner = sample.shape
        work = np.empty((count, height, inner), dtype=np.float32)
        original = np.empty((count, height, inner), dtype=np.float32)
        uppers = np.empty((count, inner, inner), dtype=np.float32)
        deads = np.empty((count, inner), dtype=np.uint8)
        meta: list[tuple[Any, ...]] = []
        for index, item in enumerate(wave):
            expert, name, matrix, upper, dead, nearest, hessian = item
            original[index] = matrix
            work[index] = matrix
            work[index, :, dead.astype(bool)] = 0.0
            uppers[index] = upper
            deads[index] = dead
            meta.append((expert, name, matrix, hessian, nearest))
        if not opened:
            session.begin()
            opened = True
        started = time.perf_counter()
        job = foem_fp4_encode(session, work, original, uppers, deads, beta=BETA)
        session.flush()
        log(f"layer {layer} {proj} wave {count} queued {time.perf_counter() - started:.1f}s")
        inflight.append((job, meta))
        if len(inflight) > 1:
            finish_oldest()

    try:
        while True:
            item = prepared.get()
            if item is None:
                break
            if item[0] == "skip":
                _, expert, name, nearest = item
                log(f"{name} has no inputs, kept nearest")
                packed, held = nearest
                data[expert, :, : packed.shape[1]] = packed
                scale_out[expert, :, : held.shape[1]] = held
                continue
            pending.append(item[1:])
            submit(False)
        submit(True)
        while inflight:
            finish_oldest()
        if opened:
            session.end()
    finally:
        thread.join()
    if failures:
        raise failures[0]
    return data, scale_out


def _round_layer(grams: ExpertGrams, tensors: Tensors, dest: Path, log: Any) -> None:
    from inkling_2_mini.metal41.session import Session

    layer = grams.layer
    session = Session("inkling_2_mini:gptq")
    stored: dict[str, np.ndarray] = {}
    try:
        for proj, hessian_rows, rows, cols in (
            ("w13", grams.up, INTERMEDIATE * 2, HIDDEN),
            ("w2", grams.down, HIDDEN, INTERMEDIATE),
        ):
            packed_all, scales_all, global_scale = _projection(tensors, layer, proj)
            data, scale = _round_projection(
                layer,
                proj,
                packed_all,
                scales_all,
                global_scale,
                hessian_rows,
                rows,
                cols,
                log,
                session,
            )
            stored[f"{proj}.data"] = data
            stored[f"{proj}.scale"] = scale
    finally:
        session.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".partial.npz")
    np.savez(partial, source=np.array(grams.source), **stored)
    partial.replace(dest)


def _source_name(layer: int, proj: str, kind: str) -> str:
    stem = f"model.llm.layers.{layer}.mlp.experts.{proj}"
    if kind == "data":
        return f"{stem}_weight"
    return f"{stem}_weight_scale"


def _assemble(folder: Path, tensors: Tensors, log: Any) -> Path:
    from inkling_2_mini.metal41.resident import EXPERT_FORMAT, EXPERT_PACK, Sink, catalog

    work = folder / "expert-foem"
    specs = catalog()
    sink = Sink(
        folder / EXPERT_PACK,
        [(spec.name, spec.shape) for spec in specs],
        {"format": EXPERT_FORMAT, "config": config_digest(folder), "quant": "foem"},
    )
    opened: Any = None
    current = -1
    try:
        for spec in specs:
            if spec.nvfp4:
                if spec.layer != current:
                    if opened is not None:
                        opened.close()
                    opened = np.load(work / f"layer-{spec.layer}.npz")
                    current = spec.layer
                array = np.ascontiguousarray(opened[f"{spec.proj}.{spec.kind}"])
                if array.shape != spec.shape:
                    message = f"{spec.name} shape {array.shape} is not {spec.shape}"
                    raise RuntimeError(message)
                sink.write_view(spec.name, memoryview(array.reshape(-1)))
                continue
            file_off, nbytes = tensors.span(
                _source_name(spec.layer, spec.proj, spec.kind), spec.shape
            )
            sink.copy_fd(spec.name, tensors.fileno(), file_off, nbytes)
        sink.close()
    except Exception:
        sink.abort()
        raise
    finally:
        if opened is not None:
            opened.close()
    path = folder / EXPERT_PACK
    log(f"wrote {path.name}")
    return path


def _drop_dense_fp4(folder: Path) -> None:
    for name in (HESSIANS_NAME, ROUND_PARTIAL, "dense-fp4.safetensors"):
        (folder / name).unlink(missing_ok=True)


def calibrate(
    folder: Path, *, limit: int = SEQUENCE_LENGTH, reuse: bool = False, log: Any = print
) -> Path:
    """Write Metal fp4 experts. Layers 0–19 are GPTQ-FOEM. Layers 20–23 are copied."""
    from inkling_2_mini.metal41.session import Session

    folder = Path(folder)
    _drop_dense_fp4(folder)
    ensure_dense(folder, log=log)
    probe = Session("inkling_2_mini:gptq")
    try:
        foem_device_matches(probe)
    finally:
        probe.close()
    log("foem device matches")
    sequences = _sequences(folder, limit, log)
    source = _source(sequences, limit)
    work = folder / "expert-foem"
    work.mkdir(parents=True, exist_ok=True)
    tensors = Tensors(checkpoint_file(folder))
    try:
        pending: list[ExpertGrams] = []
        for layer in range(NVFP4_LAYERS):
            packed = work / f"layer-{layer}.npz"
            grams_path = work / f"layer-{layer}-inputs.npz"
            if packed.is_file() and not reuse:
                with np.load(packed) as held:
                    found = str(held["source"]) if "source" in held.files else ""
                if found == source:
                    log(f"layer {layer} pack is current")
                    continue
            if reuse:
                grams = ExpertGrams.load(grams_path, source)
                if grams is None:
                    message = f"layer {layer} has no saved inputs for this corpus"
                    raise RuntimeError(message)
                log(f"layer {layer} reusing {grams.tokens} calibration tokens")
                _round_layer(grams, tensors, packed, log)
                grams_path.unlink(missing_ok=True)
                continue
            grams = ExpertGrams.load(grams_path, source)
            if grams is None or grams.layer != layer:
                grams = ExpertGrams(layer)
                grams.source = source
            if grams.done > len(sequences):
                grams.done = len(sequences)
                grams.saved_done = grams.done
            pending.append(grams)
        while pending:
            ready = [grams for grams in pending if grams.done >= len(sequences)]
            if not ready:
                done = min(grams.done for grams in pending)
                cohort = [grams for grams in pending if grams.done == done]
                ready = _capture_group(folder, cohort, sequences, log)
            for grams in ready:
                packed = work / f"layer-{grams.layer}.npz"
                _round_layer(grams, tensors, packed, log)
                (work / f"layer-{grams.layer}-inputs.npz").unlink(missing_ok=True)
            pending = [grams for grams in pending if grams not in ready]
        path = _assemble(folder, tensors, log)
    finally:
        tensors.close()
    return path


__all__ = ["SEQUENCE_LENGTH", "calibrate", "chat_rows", "weighted_error"]
