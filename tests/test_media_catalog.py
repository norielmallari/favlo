"""The media catalog is registry data. It does not open a device."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from favlo.media_catalog import build_catalog, publish


def test_publish_uses_the_pipeline_id() -> None:
    assert publish("z_image_turbo", "generate_image") == "z_image_turbo__generate_image"


def test_a_broken_pipeline_is_skipped_so_serve_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "registry.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "broken-model": {
                        "pipeline": "broken_pipe",
                        "runtime": "metal41",
                        "alias": None,
                        "variants": {"generate": {}},
                    },
                    "still-model": {
                        "pipeline": "still_pipe",
                        "runtime": "metal41",
                        "alias": None,
                        "variants": {"generate": {}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    def load(name: str, root=None, *, runtime=None, pipeline_root=None):
        del root, runtime, pipeline_root
        if name == "broken_pipe":
            raise SyntaxError("from __future__ imports must occur at the beginning of the file")
        from collections.abc import Sequence
        from pathlib import Path
        from types import SimpleNamespace
        from typing import Any

        from favlo.media_pipeline import MediaCall, MediaPipeline, MediaTool
        from favlo.pipeline import Runtime

        class _Handle(MediaPipeline):
            kind = "image"
            id_prefix = "image"
            object_name = "image"

            def parse(self, body):
                del body
                raise NotImplementedError

            def generate(self, runtime, request, hold, job, stop, call):
                del runtime, request, hold, job, stop, call
                raise NotImplementedError

            def pack(self, runtime, request, result, job, home, table, call):
                del runtime, request, result, job, home, table, call
                raise NotImplementedError

            def main(self, argv: Sequence[str]) -> int:
                del argv
                return 0

            def open_runtime(self, path: Path, **kwargs: Any) -> Runtime:
                del path, kwargs
                return Runtime()

            def mcp_tools(self) -> Sequence[MediaTool]:
                return [
                    MediaTool(
                        name="generate_image",
                        description="One still.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "Caption."},
                            },
                            "required": ["prompt"],
                        },
                        handler="handle_image",
                        result="image",
                        task=True,
                    )
                ]

        handle = _Handle()
        return SimpleNamespace(pipeline=handle, handle_image=handle)

    monkeypatch.setattr("favlo.cli.load_pipeline", load)
    catalog = build_catalog(tmp_path)
    names = [name for name, _record, _spec in catalog.published()]
    assert names == ["still_pipe__generate_image"]
    err = capsys.readouterr().err
    assert "broken-model did not import (SyntaxError" in err


def test_registered_media_tools_publish_without_a_device() -> None:
    names = [name for name, _record, _spec in build_catalog().published()]
    assert "z_image_turbo__generate_image" in names
    assert "wan22_i2v__generate_video" in names
    assert "generate_image" not in names
    assert all("__" in name for name in names)
