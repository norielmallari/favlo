"""Discover pipeline packages and run them.

Layout is ``pipelines/<model>/<runtime>/``. The model id is the CLI flag
stem. Loading imports ``{model}.{runtime}``.
"""

from __future__ import annotations

import importlib
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
    """One runtime package directory. A relative value resolves against the checkout."""
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
    """The runtime package directory for one model."""
    if runtime not in RUNTIMES:
        message = f"unknown runtime: {runtime}"
        raise ValueError(message)
    text = pipeline_root.strip() if isinstance(pipeline_root, str) else ""
    if not text or text == "pipelines":
        text = f"pipelines/{pipeline}/{runtime}"
    return resolve_pipeline_root(text, root=root)


def _ensure_import_root(path: Path) -> None:
    """Put the parent of the model package on ``sys.path``."""
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def ensure_pipelines_path(*, root: Path | None = None) -> Path:
    """Checkout ``pipelines/`` on ``sys.path`` once."""
    path = (root or repo_root()) / "pipelines"
    _ensure_import_root(path)
    return path


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


def _read_pipeline_constant(init: Path) -> str | None:
    try:
        text = init.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("PIPELINE"):
            continue
        if "=" not in stripped:
            continue
        _, _, rhs = stripped.partition("=")
        value = rhs.strip().strip("\"'")
        if value:
            return value
    return None


def _consider_runtime(
    found: dict[str, dict[str, Path]],
    *,
    model: str,
    runtime: str,
    directory: Path,
) -> None:
    init = directory / "__init__.py"
    if not init.is_file():
        return
    pipeline = _read_pipeline_constant(init)
    if pipeline != model:
        return
    runtimes = found.setdefault(model, {})
    previous = runtimes.get(runtime)
    if previous is not None and previous.resolve() != directory.resolve():
        message = f"duplicate pipeline {model}.{runtime}: {previous} and {directory}"
        raise ValueError(message)
    runtimes[runtime] = directory


def pipeline_runtimes(root: Path | None = None) -> dict[str, dict[str, Path]]:
    """Model id to runtime name to package directory."""
    base = root or repo_root()
    found: dict[str, dict[str, Path]] = {}
    tree = base / "pipelines"
    if tree.is_dir():
        for model_dir in sorted(path for path in tree.iterdir() if path.is_dir()):
            if model_dir.name.startswith(".") or model_dir.name == "__pycache__":
                continue
            if not (model_dir / "__init__.py").is_file():
                continue
            for runtime in RUNTIMES:
                runtime_dir = model_dir / runtime
                if runtime_dir.is_dir():
                    _consider_runtime(
                        found,
                        model=model_dir.name,
                        runtime=runtime,
                        directory=runtime_dir,
                    )
    for stored in _registry_pipeline_roots(root):
        if stored == "pipelines":
            continue
        try:
            directory = resolve_pipeline_root(stored, root=base)
        except ValueError:
            continue
        if not directory.is_dir() or not (directory / "__init__.py").is_file():
            continue
        runtime = directory.name
        if runtime not in RUNTIMES:
            continue
        model = directory.parent.name
        _ensure_import_root(directory.parent.parent)
        _consider_runtime(found, model=model, runtime=runtime, directory=directory)
    return found


def pipeline_names(root: Path | None = None) -> tuple[str, ...]:
    """Discovered model ids (CLI flag stems)."""
    return tuple(sorted(pipeline_runtimes(root)))


def pipeline_entries(root: Path | None = None) -> dict[str, Path]:
    """Model id to a representative runtime package directory (preferred metal41)."""
    found: dict[str, Path] = {}
    for model, runtimes in pipeline_runtimes(root).items():
        if "metal41" in runtimes:
            found[model] = runtimes["metal41"]
        else:
            found[model] = runtimes[sorted(runtimes)[0]]
    return found


def default_runtime(name: str, root: Path | None = None) -> str:
    """Sole runtime for a model, else ``metal41`` when present."""
    runtimes = pipeline_runtimes(root).get(name) or {}
    if not runtimes:
        message = f"pipeline is missing: {name}"
        raise FileNotFoundError(message)
    if "metal41" in runtimes:
        return "metal41"
    if len(runtimes) == 1:
        return next(iter(runtimes))
    message = f"pipeline {name} has multiple runtimes; pass runtime explicitly"
    raise ValueError(message)


def _help(names: Sequence[str]) -> str:
    lines = ["usage: favlo serve [--non-interactive]"]
    lines.append("       favlo web")
    lines.append("       favlo media --pipeline <id> [key=value ...]")
    lines.extend(f"       favlo --{name} [args]" for name in names)
    lines.append("")
    lines.append("  serve                  listen on this terminal")
    lines.append("  serve --non-interactive")
    lines.append("                         start the server daemon and exit")
    lines.append("  web                    start the API and the Expo app")
    lines.append("  media --pipeline <id>  run one media pipeline on a listening serve")
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
    base = root or repo_root()
    ensure_pipelines_path(root=base)
    if pipeline_root:
        directory = resolve_pipeline_root(pipeline_root, root=base)
        if directory.name in RUNTIMES:
            model_dir = directory.parent
            chosen = runtime or directory.name
            if chosen != directory.name:
                directory = model_dir / chosen
        else:
            # Older rows may store the model directory; prefer an explicit runtime.
            model_dir = directory
            chosen = runtime or default_runtime(name, root=base)
            directory = model_dir / chosen
        if not (directory / "__init__.py").is_file():
            message = f"pipeline file is missing: {directory}"
            raise FileNotFoundError(message)
        _ensure_import_root(model_dir.parent)
        module_name = f"{model_dir.name}.{directory.name}"
    else:
        chosen = runtime or default_runtime(name, root=base)
        runtimes = pipeline_runtimes(root=base).get(name) or {}
        if chosen not in runtimes:
            message = f"pipeline file is missing: {name}.{chosen}"
            raise FileNotFoundError(message)
        module_name = f"{name}.{chosen}"
    with _IMPORTS:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            message = f"pipeline file is missing: {module_name}"
            raise FileNotFoundError(message) from exc


def pipeline_roots(root: Path | None = None) -> tuple[Path, ...]:
    """Directories that hold a discovered runtime package."""
    found: list[Path] = []
    seen: set[Path] = set()
    for runtimes in pipeline_runtimes(root).values():
        for directory in runtimes.values():
            resolved = directory.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return tuple(found)


def drop_pipeline_modules(root: Path | None = None) -> int:
    """Drop imported pipeline packages so the next load reads the files again.

    The lock is the same one ``load_pipeline`` holds. A module under ``src/favlo`` stays.
    """
    with _IMPORTS:
        roots = pipeline_roots(root)
        model_roots = {
            path.parent.resolve() for path in roots
        }
        names = [
            name
            for name, module in list(sys.modules.items())
            if module is not None
            and (
                _module_under(module, roots)
                or _module_under(module, tuple(model_roots))
            )
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
    if args[0] == "media":
        from favlo.media_cli import main as media_main

        return int(media_main(args[1:]))
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
    except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return int(entry(args[1:]))
