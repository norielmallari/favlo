"""Calibrate the dense fp4 planes: capture plane inputs, then round with GPTQ.

The model answers each calibration prompt itself, with thinking on and off,
from the round-to-nearest planes. Each prompt plus its answer is prefilled
once more with the capture hooks on. The runtime then closes, and every dense
plane is rounded against the Hessian of its own inputs.
"""

from __future__ import annotations

import time
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

PROMPTS: list[tuple[str, str, bool]] = [
    (
        "You are a helpful assistant.",
        "Explain how a binary search works, with a short example.",
        False,
    ),
    (
        "You are a helpful assistant.",
        "Write a Python function that merges two sorted lists.",
        False,
    ),
    ("You are a concise assistant.", "What causes the seasons on Earth?", False),
    (
        "You are a helpful assistant.",
        "Summarize the plot of a heist story in five sentences.",
        False,
    ),
    (
        "You are a careful math tutor.",
        "A train covers 180 km in 2.5 hours. What is its average speed in m/s?",
        False,
    ),
    ("You are a helpful assistant.", "Give three tips for writing clear commit messages.", False),
    (
        "You are a helpful assistant.",
        "Translate 'The library opens at nine' into French and Spanish.",
        False,
    ),
    (
        "You are a code reviewer.",
        "Review this function:\n\ndef add(items=[]):\n    items.append(1)\n    return items",
        False,
    ),
    (
        "You are a helpful assistant.",
        "Write a haiku about autumn rain, then explain its imagery.",
        False,
    ),
    ("You are a helpful assistant.", "Compare TCP and UDP in a short table.", False),
    (
        "You are a helpful assistant.",
        "How do I reverse a linked list in place? Show code in C.",
        False,
    ),
    ("You are a helpful assistant.", "List the planets in order and one fact about each.", False),
    (
        "You are a JSON API. Reply with JSON only.",
        "Return an object with keys name, age, and city for a fictional person.",
        False,
    ),
    ("You are a helpful assistant.", "Solve for x: 3x + 7 = 2x - 5. Show each step.", False),
    (
        "You are a helpful assistant.",
        "What is the difference between a process and a thread?",
        False,
    ),
    (
        "You are a helpful assistant.",
        "Draft a polite email asking to move a meeting to Friday.",
        False,
    ),
    (
        "You are an agent. Use the tools when they help.",
        "What's the weather in Lisbon right now?",
        True,
    ),
    (
        "You are an agent. Use the tools when they help.",
        "Open README.md and tell me what the project does.",
        True,
    ),
    (
        "You are an agent. Use the tools when they help.",
        "Run the tests named parser and report the result.",
        True,
    ),
    (
        "You are an agent. Use the tools when they help.",
        "Put a 30 minute dentist visit on 2026-03-14.",
        True,
    ),
    ("Call a tool. Do not describe the call.", "Read the file src/main.py.", True),
    (
        "You are an agent. Use the tools when they help.",
        "Is it colder in Oslo or in Helsinki today? Use fahrenheit.",
        True,
    ),
    (
        "You are an agent. Use the tools when they help.",
        "Thanks! Can you explain what a unit test is?",
        True,
    ),
    ("Use one tool call, then stop.", "Schedule 'team sync' for 2026-05-02, 45 minutes.", True),
]


def _sequences(runtime: Any, limit: int, log: Any) -> list[list[int]]:
    from inkling_2_mini.metal41.chat import STOP, encode_chat

    sequences: list[list[int]] = []
    for index, (system, user, tools) in enumerate(PROMPTS):
        for thinking in (True, False):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            declared = _TOOLS if tools else None
            prompt = encode_chat(
                runtime.root, messages, declared, thinking, effort="medium" if thinking else "none"
            )
            produced = runtime.generate(prompt, limit, 0.7, 1.0, 0, 0.0, STOP)
            sequences.append(prompt + produced)
            log(f"sequence {len(sequences)} prompt {index} tokens {len(sequences[-1])}")
    return sequences


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

    capture = Capture()
    runtime = InklingRuntime(folder, owner="inkling_2_mini:calibrate", calibrated=False)
    runtime._prefix_scratch = "inkling-calibrate"
    runtime._prefix_budget = 1
    try:
        sequences = _sequences(runtime, limit, log)
        for number, sequence in enumerate(sequences):
            runtime.capture_prefill(sequence, capture)
            log(f"captured {number + 1}/{len(sequences)} tokens {capture.tokens}")
    finally:
        runtime.close()
    return capture


def calibrate(folder: Path, *, limit: int = 384, reuse: bool = False, log: Any = print) -> Path:
    """Write the GPTQ pack. ``reuse`` rounds again from the saved input sums."""
    saved = Path(folder) / HESSIANS_NAME
    if reuse and saved.is_file():
        capture = Capture.load(saved)
        log(f"loaded {saved.name} from {capture.tokens} calibration tokens")
    else:
        capture = _capture(Path(folder), limit, log)
        capture.save(saved)
    tensors = Tensors(checkpoint_file(folder))
    out: dict[str, np.ndarray] = {}
    try:
        for index in range(LAYERS):
            out[f"layers.{index}.attn.data"], out[f"layers.{index}.attn.scale"] = _attention(
                tensors, capture, index, log
            )
            wo = wo_matrix(tensors, index)
            out[f"layers.{index}.wo.data"], out[f"layers.{index}.wo.scale"] = _round(
                f"layer {index} wo", wo, capture.hessian(index, "wo"), log
            )
            for stream in ("shared", "q", "k", "v", "wo"):
                capture.sums.pop((index, stream), None)
        head = head_matrix(tensors)
        packed, scales = _round("head", head, capture.hessian(-1, "head"), log)
        out["unembed.data"] = _pad_rows(packed, HEAD_ROWS)
        out["unembed.scale"] = _pad_rows(scales, HEAD_ROWS)
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
    return path


__all__ = ["PROMPTS", "calibrate", "weighted_error"]
