"""Metal source files, read as text and compiled at runtime."""

from pathlib import Path


def shader_text(*names: str) -> str:
    root = Path(__file__).resolve().parent / "shaders"
    return "".join((root / name).read_text() for name in names)
