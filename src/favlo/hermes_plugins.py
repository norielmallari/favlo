"""Install Favlo's Hermes image and video backends into every profile.

Hermes reads ``plugins.enabled`` from the active profile's ``config.yaml``.
A named profile's home is ``~/.hermes/profiles/<name>``, and that home discovers
``plugins/`` on its own. The default profile is ``~/.hermes``. The backends are
copied there. Each other profile symlinks those two directories. Enabling still
happens in that profile's own config.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PLUGIN_KEYS = ("image_gen/favlo", "video_gen/favlo")
SECTIONS = {
    "image_gen": ("provider: favlo_image_gen", "model: z-image-turbo-fp4"),
    "video_gen": ("provider: favlo_video_gen", "model: wan22-i2v"),
}
OLD_PROVIDER = "favlo"


def plugin_source(root: Path | None = None) -> Path:
    from favlo.cli import repo_root

    return (root or repo_root()) / "hermes-agent-plugins"


def hermes_root(path: Path | None = None) -> Path:
    """The Hermes home that contains ``profiles/``. A profile path unwraps."""
    if path is not None:
        home = path.expanduser().resolve()
    else:
        raw = os.environ.get("HERMES_HOME", "").strip()
        home = Path(raw).expanduser().resolve() if raw else Path.home() / ".hermes"
    if home.parent.name == "profiles":
        return home.parent.parent
    return home


def profile_homes(root: Path) -> list[Path]:
    """The default home, then each directory under ``profiles/``."""
    homes = [root]
    profiles = root / "profiles"
    if not profiles.is_dir():
        return homes
    homes.extend(
        sorted(
            child
            for child in profiles.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )
    )
    return homes


def install(root: Path, source: Path | None = None) -> list[str]:
    """Copy both backends into the default home. Other profiles symlink there."""
    tree = source or plugin_source()
    for key in PLUGIN_KEYS:
        _copy_plugin(tree / key, root / "plugins" / key)
    labels: list[str] = []
    for home in profile_homes(root):
        if home != root:
            for key in PLUGIN_KEYS:
                _link_plugin(root / "plugins" / key, home / "plugins" / key)
        config = home / "config.yaml"
        if config.is_file():
            updated = edit_config(config.read_text(encoding="utf-8"))
            config.write_text(updated, encoding="utf-8")
        labels.append("default" if home == root else home.name)
    return labels


def edit_config(text: str) -> str:
    """Enable both backends and select them when a profile has no provider yet."""
    lines = text.splitlines()
    lines = _ensure_enabled(lines, PLUGIN_KEYS)
    lines = _ensure_toolset(lines, "platform_toolsets", "image_gen", "video_gen")
    lines = _ensure_cli_tools(lines, ("image_gen", "video_gen"))
    for name, fields in SECTIONS.items():
        lines = _rename_provider(lines, name, OLD_PROVIDER, fields[0].split(": ", 1)[1])
        lines = _ensure_section(lines, name, fields)
    body = "\n".join(lines)
    if text.endswith("\n") or body:
        body += "\n"
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="favlo hermes-plugins",
        description="Install the Favlo image and video backends into every Hermes profile.",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Hermes home. The default is HERMES_HOME, or ~/.hermes.",
    )
    args = parser.parse_args(argv)
    root = hermes_root(args.home)
    if not root.is_dir():
        sys.stderr.write(f"hermes home is missing: {root}\n")
        return 1
    source = plugin_source()
    if not (source / "image_gen" / "favlo" / "plugin.yaml").is_file():
        sys.stderr.write(f"plugin source is missing: {source}\n")
        return 1
    for label in install(root, source):
        print(f"installed image_gen/favlo and video_gen/favlo for {label}", flush=True)
    return 0


def _copy_plugin(source: Path, dest: Path) -> None:
    _remove(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _link_plugin(target: Path, dest: Path) -> None:
    """Point one profile's backend at the default home's copy."""
    _remove(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(Path(os.path.relpath(target, dest.parent)))


def _remove(dest: Path) -> None:
    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)


def _ensure_enabled(lines: list[str], keys: tuple[str, ...]) -> list[str]:
    span = _top_span(lines, "plugins")
    if span is None:
        return [*lines, "plugins:", "  enabled:", *[f"    - {key}" for key in keys]]
    start, end = span
    enabled = None
    for index in range(start + 1, end):
        if lines[index].strip() == "enabled:" or lines[index].strip() == "enabled: []":
            enabled = index
            break
    if enabled is None:
        insert = ["  enabled:", *[f"    - {key}" for key in keys]]
        return [*lines[: start + 1], *insert, *lines[start + 1 :]]
    if lines[enabled].strip() == "enabled: []":
        indent = _indent(lines[enabled])
        item_indent = indent + "  "
        replacement = [f"{indent}enabled:", *[f"{item_indent}- {key}" for key in keys]]
        return [*lines[:enabled], *replacement, *lines[enabled + 1 :]]
    item_indent = _indent(lines[enabled]) + "  "
    present: set[str] = set()
    last = enabled
    for index in range(enabled + 1, end):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            last = index
            continue
        if not stripped.startswith("- ") or len(_indent(lines[index])) < len(item_indent):
            break
        present.add(stripped[2:].strip())
        item_indent = _indent(lines[index])
        last = index
    missing = [key for key in keys if key not in present]
    if not missing:
        return lines
    addition = [f"{item_indent}- {key}" for key in missing]
    return [*lines[: last + 1], *addition, *lines[last + 1 :]]


def _ensure_toolset(lines: list[str], block: str, anchor: str, extra: str) -> list[str]:
    span = _top_span(lines, block)
    if span is None:
        return lines
    start, end = span
    index = start + 1
    while index < end:
        if lines[index].strip() != f"- {anchor}":
            index += 1
            continue
        indent = _indent(lines[index])
        siblings = _sibling_items(lines, index, end, indent)
        if f"- {extra}" in siblings:
            index += 1
            continue
        lines = [*lines[: index + 1], f"{indent}- {extra}", *lines[index + 1 :]]
        end += 1
        index += 2
    return lines


def _ensure_cli_tools(lines: list[str], names: tuple[str, ...]) -> list[str]:
    """Add the generation toolsets to a profile's CLI list when that list exists."""
    span = _top_span(lines, "platform_toolsets")
    if span is None:
        return lines
    start, end = span
    cli = None
    for index in range(start + 1, end):
        if lines[index].strip() == "cli:":
            cli = index
            break
    if cli is None:
        return lines
    cli_indent = _indent(lines[cli])
    item_indent = cli_indent + "  "
    present: set[str] = set()
    last = cli
    for index in range(cli + 1, end):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            last = index
            continue
        if len(_indent(lines[index])) <= len(cli_indent):
            break
        if stripped.startswith("- "):
            present.add(stripped[2:].strip())
            item_indent = _indent(lines[index])
            last = index
    missing = [name for name in names if name not in present]
    if not missing:
        return lines
    addition = [f"{item_indent}- {name}" for name in missing]
    return [*lines[: last + 1], *addition, *lines[last + 1 :]]


def _rename_provider(lines: list[str], name: str, old: str, new: str) -> list[str]:
    """Replace the shared provider name inside one generation section."""
    span = _top_span(lines, name)
    if span is None:
        return lines
    start, end = span
    for index in range(start + 1, end):
        if lines[index].strip() != f"provider: {old}":
            continue
        indent = _indent(lines[index])
        return [*lines[:index], f"{indent}provider: {new}", *lines[index + 1 :]]
    return lines


def _ensure_section(lines: list[str], name: str, fields: tuple[str, ...]) -> list[str]:
    span = _top_span(lines, name)
    block = [f"{name}:", *[f"  {field}" for field in fields]]
    if span is None:
        if lines and lines[-1].strip():
            return [*lines, "", *block]
        return [*lines, *block]
    start, end = span
    present = {lines[index].strip().split(":", 1)[0] for index in range(start + 1, end)}
    if "provider" in present:
        return lines
    missing = [field for field in fields if field.split(":", 1)[0] not in present]
    addition = [f"  {field}" for field in missing]
    return [*lines[: start + 1], *addition, *lines[start + 1 :]]


def _sibling_items(lines: list[str], index: int, end: int, indent: str) -> set[str]:
    found: set[str] = set()
    cursor = index
    while cursor > 0:
        previous = cursor - 1
        stripped = lines[previous].strip()
        if not stripped or stripped.startswith("#"):
            cursor = previous
            continue
        if _indent(lines[previous]) == indent and stripped.startswith("- "):
            found.add(stripped)
            cursor = previous
            continue
        break
    cursor = index
    while cursor < end:
        stripped = lines[cursor].strip()
        if not stripped or stripped.startswith("#"):
            cursor += 1
            continue
        if _indent(lines[cursor]) == indent and stripped.startswith("- "):
            found.add(stripped)
            cursor += 1
            continue
        break
    return found


def _top_span(lines: list[str], key: str) -> tuple[int, int] | None:
    start = None
    for index, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            start = index
            break
    if start is None:
        return None
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped and not lines[index][0].isspace() and not stripped.startswith("#"):
            return start, index
    return start, len(lines)


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" "))]
