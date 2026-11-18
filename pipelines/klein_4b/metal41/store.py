"""Packed int8 record. Matrices stay memory-mapped until the prefix is released."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Matrix:
    codes: np.ndarray
    scales: np.ndarray
    bias: np.ndarray | None
    n: int
    k: int
    stored_n: int
    stored_k: int
    kh: int | None
    kw: int | None


class Store:
    """One finished record directory."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.index = json.loads((self.path / "index.json").read_text(encoding="utf-8"))
        self._maps: list[Any] = []

    def has(self, key: str) -> bool:
        return key in self.index["matrices"] or key in self.index["vectors"]

    def matrix(self, key: str) -> Matrix:
        spec = self.index["matrices"][key]
        codes = np.memmap(
            self.path / spec["codes"],
            dtype=np.int8,
            mode="r",
            shape=(int(spec["stored_n"]), int(spec["stored_k"])),
        )
        scales = np.memmap(
            self.path / spec["scales"],
            dtype=np.float32,
            mode="r",
            shape=(int(spec["stored_n"]),),
        )
        self._maps.extend((codes, scales))
        bias = None
        bias_file = spec.get("bias")
        if isinstance(bias_file, str):
            bias = np.fromfile(self.path / bias_file, dtype=np.float16)
        kh = spec.get("kh")
        kw = spec.get("kw")
        return Matrix(
            codes,
            scales,
            bias,
            int(spec["n"]),
            int(spec["k"]),
            int(spec["stored_n"]),
            int(spec["stored_k"]),
            int(kh) if isinstance(kh, int) else None,
            int(kw) if isinstance(kw, int) else None,
        )

    def vector(self, key: str) -> np.ndarray:
        spec = self.index["vectors"][key]
        dtype = np.float32 if spec["dtype"] == "f32" else np.float16
        return np.fromfile(self.path / spec["file"], dtype=dtype)

    def close_prefix(self, prefix: str) -> None:
        """Drop maps whose key starts with ``prefix`` so those pages can leave."""
        needle = f"weights/{prefix.strip('/')}"
        kept: list[Any] = []
        for item in self._maps:
            name = str(getattr(item, "filename", "")).replace("\\", "/")
            if needle in name:
                continue
            kept.append(item)
        self._maps = kept

    def close(self) -> None:
        self._maps.clear()
