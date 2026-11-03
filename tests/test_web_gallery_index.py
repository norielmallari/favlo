"""Gallery vector index nearest and unlike search."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.gallery_index import DIM, GalleryIndex  # noqa: E402


def _vec(*values: float) -> list[float]:
    """Pad a short probe into a DIM-length embedding."""
    vector = [0.0] * DIM
    for index, value in enumerate(values):
        vector[index] = value
    norm = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / norm for item in vector]


def test_search_unlike_returns_farthest(tmp_path: Path) -> None:
    index = GalleryIndex(tmp_path / "gallery_index.db")
    workspace = "workspace_demo"
    index.upsert(
        workspace,
        "near.png",
        kind="image",
        sidecar_mtime="1",
        prompt_text="near",
        vector=_vec(1.0, 0.0, 0.0),
    )
    index.upsert(
        workspace,
        "mid.png",
        kind="image",
        sidecar_mtime="1",
        prompt_text="mid",
        vector=_vec(1.0, 1.0, 0.0),
    )
    index.upsert(
        workspace,
        "far.png",
        kind="image",
        sidecar_mtime="1",
        prompt_text="far",
        vector=_vec(0.0, 0.0, 1.0),
    )
    query = _vec(1.0, 0.0, 0.0)
    nearest = index.search(workspace, query, limit=2)
    unlike = index.search(workspace, query, limit=2, unlike=True)
    assert [row["path"] for row in nearest] == ["near.png", "mid.png"]
    assert [row["path"] for row in unlike] == ["far.png", "mid.png"]
    assert unlike[0]["score"] < nearest[0]["score"]
    assert math.isclose(unlike[0]["score"], 0.0, abs_tol=1e-5)
