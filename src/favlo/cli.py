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
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

RUNTIMES = ("metal41", "cuda")
_IMPORTS = threading.Lock()


def repo_root() -> Path:
    override = os.environ.get("FAVLO_ROOT")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        marker = parent / "src" / "favlo" / "cli.py"
        if (parent / "pyproject.toml").is_file() and marker.is_file():
            return parent
    message = "favlo package is not inside a checkout"
    raise FileNotFoundError(message)


def pipeline_tree(runtime: str, *, custom: bool = False, root: Path | None = None) -> Path:
    """The directory that holds one runtime's entry files."""
    if runtime not in RUNTIMES:
        message = f"unknown runtime: {runtime}"
        raise ValueError(message)
    base = "custom-pipelines" if custom else "pipelines"
    return (root or repo_root()) / base / runtime


def pipeline_names(root: Path | None = None) -> tuple[str, ...]:
    """Entry stems under the four pipeline trees.

    A new runtime is a new file. Discovering it does not edit this module.
    Two files with the same ``PIPELINE`` stem are an error.
    """
    return tuple(sorted(pipeline_entries(root)))


def pipeline_entries(root: Path | None = None) -> dict[str, Path]:
    """Stem to entry file. A duplicate stem raises ``ValueError``."""
    found: dict[str, Path] = {}
    for custom in (False, True):
        for runtime in RUNTIMES:
            directory = pipeline_tree(runtime, custom=custom, root=root)
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.py")):
                stem = _pipeline_stem(path)
                if stem is None:
                    continue
                previous = found.get(stem)
                if previous is not None:
                    message = f"duplicate pipeline {stem}: {previous} and {path}"
                    raise ValueError(message)
                found[stem] = path
    return found


def _pipeline_stem(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "PIPELINE":
            continue
        if isinstance(node.value, ast.Constant) and node.value.value == path.stem:
            return path.stem
    return None


def _help(names: Sequence[str]) -> str:
    lines = ["usage: favlo serve [--non-interactive]"]
    lines.append("       favlo web")
    lines.extend(f"       favlo --{name} [args]" for name in names)
    lines.append("")
    lines.append("  serve                  chat on a terminal; the server is a daemon")
    lines.append("  serve --non-interactive")
    lines.append("                         start the server daemon and exit")
    lines.append("  web                    start the API and the Expo app")
    width = max((len(name) for name in names), default=0)
    for name in names:
        lines.append(f"  --{name.ljust(width)}    {name}")
    return "\n".join(lines) + "\n"


def load_pipeline(
    name: str,
    root: Path | None = None,
    *,
    runtime: str | None = None,
    custom: bool = False,
) -> ModuleType:
    if runtime is None:
        path = pipeline_entries(root).get(name)
        if path is None:
            message = f"pipeline file is missing: {name}"
            raise FileNotFoundError(message)
    else:
        path = pipeline_tree(runtime, custom=custom, root=root) / f"{name}.py"
        if not path.is_file():
            message = f"pipeline file is missing: {path}"
            raise FileNotFoundError(message)
    with _IMPORTS:
        spec = importlib.util.spec_from_file_location(f"{name.replace('-', '_')}_pipeline", path)
        if spec is None or spec.loader is None:
            message = f"cannot load pipeline file: {path}"
            raise ImportError(message)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def pipeline_roots(root: Path | None = None) -> tuple[Path, ...]:
    """Pipeline trees that exist in this checkout."""
    found: list[Path] = []
    for custom in (False, True):
        for runtime in RUNTIMES:
            directory = pipeline_tree(runtime, custom=custom, root=root)
            if directory.is_dir():
                found.append(directory.resolve())
    return tuple(found)


def drop_pipeline_modules(root: Path | None = None) -> int:
    """Drop imported pipeline packages so the next load reads the files again.

    The lock is the same one ``load_pipeline`` holds. A module under ``src/favlo`` stays.
    """
    with _IMPORTS:
        roots = pipeline_roots(root)
        names = [
            name
            for name, module in list(sys.modules.items())
            if module is not None and _module_under(module, roots)
        ]
        for name in names:
            sys.modules.pop(name, None)
        return len(names)


def _module_under(module: ModuleType, roots: tuple[Path, ...]) -> bool:
    file = getattr(module, "__file__", None)
    if isinstance(file, str) and file and _path_under(Path(file), roots):
        return True
    paths = getattr(module, "__path__", None)
    if paths is None:
        return False
    return any(_path_under(Path(str(item)), roots) for item in paths)


def _path_under(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved.is_relative_to(root) for root in roots)


def pipeline_main(module: ModuleType) -> Callable[[Sequence[str]], int]:
    entry = getattr(module, "main", None)
    if not callable(entry):
        message = f"{module.__file__} does not define main()"
        raise TypeError(message)
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        names = pipeline_names()
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if not args:
        sys.stderr.write(_help(names))
        return 2
    if args[0] in {"-h", "--help"}:
        sys.stdout.write(_help(names))
        return 0
    if args[0] == "serve":
        from favlo.listener import serve_main

        return int(serve_main(args[1:]))
    if args[0] == "web":
        from favlo.web_host import main as web_main

        return int(web_main(args[1:]))
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
