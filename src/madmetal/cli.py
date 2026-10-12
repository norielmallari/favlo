"""Select one pipeline file and run it.

A model flag is the filename without ``.py``. ``--qwen38-flash-next`` loads
``qwen38-flash-next.py`` by path. The hyphenated file is not an import.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

PIPELINES = ("qwen38-flash-next",)

HELP = """\
usage: madmetal --qwen38-flash-next [args]

Select one model pipeline. The flag is the pipeline filename without .py.

  --qwen38-flash-next    Qwen3.8-Flash-Next
"""


def repo_root() -> Path:
    override = os.environ.get("MADMETAL_ROOT")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "qwen38-flash-next.py").is_file():
            return parent
    message = "qwen38-flash-next.py is not next to the madmetal package"
    raise FileNotFoundError(message)


def load_pipeline(name: str, root: Path | None = None) -> ModuleType:
    path = (root or repo_root()) / f"{name}.py"
    if not path.is_file():
        message = f"pipeline file is missing: {path}"
        raise FileNotFoundError(message)
    spec = importlib.util.spec_from_file_location(f"{name.replace('-', '_')}_pipeline", path)
    if spec is None or spec.loader is None:
        message = f"cannot load pipeline file: {path}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pipeline_main(module: ModuleType) -> Callable[[Sequence[str]], int]:
    entry = getattr(module, "main", None)
    if not callable(entry):
        message = f"{module.__file__} does not define main()"
        raise TypeError(message)
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        sys.stderr.write(HELP)
        return 2
    if args[0] in {"-h", "--help"}:
        sys.stdout.write(HELP)
        return 0
    flag = args[0]
    if not flag.startswith("--") or flag == "--":
        sys.stderr.write(HELP)
        sys.stderr.write(f"\nunknown argument: {flag}\n")
        return 2
    name = flag[2:]
    if name not in PIPELINES:
        known = ", ".join(f"--{item}" for item in PIPELINES)
        sys.stderr.write(f"unknown model flag: {flag}\nknown: {known}\n")
        return 2
    try:
        module = load_pipeline(name)
        entry = pipeline_main(module)
    except (FileNotFoundError, ImportError, TypeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return int(entry(args[1:]))
