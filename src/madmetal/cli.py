"""Select one pipeline file and run it.

A model flag is the filename without ``.py``. ``--<pipeline>`` loads
``<pipeline>.py`` by path, and that file loads the model's package.
The hyphenated file is not an import.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType


def repo_root() -> Path:
    override = os.environ.get("MADMETAL_ROOT")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        marker = parent / "src" / "madmetal" / "cli.py"
        if (parent / "pyproject.toml").is_file() and marker.is_file():
            return parent
    message = "madmetal package is not inside a checkout"
    raise FileNotFoundError(message)


def pipeline_names(root: Path | None = None) -> tuple[str, ...]:
    """Entry files at the checkout root. Each sets ``PIPELINE`` to its filename stem.

    A new runtime is a new file. Discovering it does not edit this module.
    """
    found: list[str] = []
    for path in sorted((root or repo_root()).glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id != "PIPELINE":
                continue
            if isinstance(node.value, ast.Constant) and node.value.value == path.stem:
                found.append(path.stem)
                break
    return tuple(found)


def _help(names: Sequence[str]) -> str:
    lines = ["usage: madmetal serve"]
    lines.extend(f"       madmetal --{name} [args]" for name in names)
    lines.append("")
    lines.append("  serve                  listen, then load the model a request names")
    width = max((len(name) for name in names), default=0)
    for name in names:
        lines.append(f"  --{name.ljust(width)}    {name}")
    return "\n".join(lines) + "\n"


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
    names = pipeline_names()
    if not args:
        sys.stderr.write(_help(names))
        return 2
    if args[0] in {"-h", "--help"}:
        sys.stdout.write(_help(names))
        return 0
    if args[0] == "serve":
        from madmetal.listener import serve_main

        return int(serve_main(args[1:]))
    flag = args[0]
    if not flag.startswith("--") or flag == "--":
        sys.stderr.write(_help(names))
        sys.stderr.write(f"\nunknown argument: {flag}\n")
        return 2
    name = flag[2:]
    if name not in names:
        known = ", ".join(f"--{item}" for item in names)
        sys.stderr.write(f"unknown model flag: {flag}\nknown: {known}\n")
        return 2
    try:
        module = load_pipeline(name)
        entry = pipeline_main(module)
    except (FileNotFoundError, ImportError, TypeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return int(entry(args[1:]))
