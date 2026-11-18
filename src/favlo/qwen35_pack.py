"""One Qwen3.5-4B quantize command for Windows, macOS, and Linux.

``fp8`` writes the Metal record. ``int8`` and ``int4`` write the CUDA record.
Both are host NumPy. A missing GPU does not stop the command: the default
plane is then ``fp8``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from favlo.cli import repo_root
from favlo.host import device_budget

FP8_CONTEXT = 262_144
INT4_CONTEXT = 65_536


@dataclass(frozen=True)
class Target:
    plane: str
    vision: bool
    context: int


def choose_target(plane: str | None) -> Target:
    """The record this host should write. ``plane`` overrides the default."""
    if plane == "fp8":
        return Target("fp8", True, FP8_CONTEXT)
    if plane == "int4":
        return Target("int4", False, INT4_CONTEXT)
    if plane == "int8":
        budget = device_budget()
        opened = _budget_pack(6 << 30 if budget is None else budget)
        return Target("int8", True, int(opened.context))
    budget = device_budget()
    if budget is None:
        return Target("fp8", True, FP8_CONTEXT)
    opened = _budget_pack(budget)
    return Target(opened.plane, bool(opened.vision), int(opened.context))


def quantize(src: Path, dest: Path, target: Target, on_progress: Any = None) -> Path:
    """Pack ``src`` into ``dest`` for ``target``. The same bytes on every OS."""
    if target.plane == "fp8":
        return _module("metal41", "qwen35_metal_convert").quantize(Path(src), Path(dest))
    module = _module("cuda", "qwen35_cuda_convert")
    return module.quantize(
        Path(src),
        Path(dest),
        plane=target.plane,
        vision=target.vision,
        context=target.context,
        on_progress=on_progress,
    )


def record_folder(plane: str) -> str:
    folders = {"fp8": "Qwen3.5-4B-fp8", "int8": "Qwen3.5-4B-int8", "int4": "Qwen3.5-4B-int4"}
    folder = folders.get(plane)
    if folder is None:
        message = f"weight plane {plane} is not fp8, int8, or int4"
        raise ValueError(message)
    return folder


def runtime_for(plane: str) -> str:
    """The runtime that serves this record. Packing itself does not use it."""
    if plane == "fp8":
        return "metal41"
    return "cuda"


def _budget_pack(budget: int) -> Any:
    return _module("cuda", "qwen35_cuda_layout_budget", "layout.py").pack_for_budget(budget)


def _module(runtime: str, name: str, filename: str = "convert.py") -> Any:
    path = repo_root() / "pipelines" / "qwen35_4b" / runtime / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        message = f"cannot load {path}"
        raise ImportError(message)
    module = sys.modules.get(name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module
