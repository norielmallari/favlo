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


def resolve_pipeline_root(pipeline_root: str, *, root: Path | None = None) -> Path:
    """One pipeline's directory. A relative value resolves against the checkout."""
    text = pipeline_root.strip()
    if not text:
        message = "pipeline_root is empty"
        raise ValueError(message)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (root or repo_root()) / path
    return path.resolve()


def pipeline_tree(
    runtime: str,
    *,
    pipeline: str,
    pipeline_root: str | None = None,
    root: Path | None = None,
) -> Path:
    """The pipeline directory. It contains ``<pipeline>.py`` and the package."""
    if runtime not in RUNTIMES:
        message = f"unknown runtime: {runtime}"
        raise ValueError(message)
    text = pipeline_root.strip() if isinstance(pipeline_root, str) else ""
    if not text or text == "pipelines":
        text = f"pipelines/{runtime}/{pipeline}"
    return resolve_pipeline_root(text, root=root)


def _registry_pipeline_roots(root: Path | None = None) -> tuple[str, ...]:
    """Each ``pipeline_root`` stored on a registry row."""
    names: list[str] = []
    try:
        from favlo.registry import load_registry

        data = load_registry(root)
    except (OSError, ValueError):
        return ()
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return ()
    for entry in models.values():
        if not isinstance(entry, dict):
            continue
        stored = entry.get("pipeline_root")
        if isinstance(stored, str) and stored.strip() and stored.strip() not in names:
            names.append(stored.strip())
    return tuple(names)


def pipeline_names(root: Path | None = None) -> tuple[str, ...]:
    """Entry stems under every pipeline root.

    A new runtime is a new file. Discovering it does not edit this module.
    Two files with the same ``PIPELINE`` stem are an error.
    """
    return tuple(sorted(pipeline_entries(root)))


def pipeline_entries(root: Path | None = None) -> dict[str, Path]:
    """Stem to entry file. A duplicate stem raises ``ValueError``."""
    base = root or repo_root()
    found: dict[str, Path] = {}

    def consider(path: Path) -> None:
        stem = _pipeline_stem(path)
        if stem is None:
            return
        previous = found.get(stem)
        if previous is not None and previous.resolve() != path.resolve():
            message = f"duplicate pipeline {stem}: {previous} and {path}"
            raise ValueError(message)
        found[stem] = path

    for runtime in RUNTIMES:
        tree = base / "pipelines" / runtime
        if not tree.is_dir():
            continue
        for child in sorted(path for path in tree.iterdir() if path.is_dir()):
            entry = child / f"{child.name}.py"
            if entry.is_file():
                consider(entry)
    for stored in _registry_pipeline_roots(root):
        if stored == "pipelines":
            continue
        try:
            directory = resolve_pipeline_root(stored, root=base)
        except ValueError:
            continue
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            consider(path)
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
    lines.append("  serve                  listen on this terminal")
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
    pipeline_root: str | None = None,
) -> ModuleType:
    if runtime is None:
        path = pipeline_entries(root).get(name)
        if path is None:
            message = f"pipeline file is missing: {name}"
            raise FileNotFoundError(message)
    else:
        path = pipeline_tree(
            runtime, pipeline=name, pipeline_root=pipeline_root, root=root
        ) / f"{name}.py"
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
    """Directories that hold a discovered pipeline entry and its package."""
    found: list[Path] = []
    seen: set[Path] = set()
    for path in pipeline_entries(root).values():
        directory = path.parent.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        found.append(directory)
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
    from favlo.logs import install_rolling_stdio

    install_rolling_stdio()
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
