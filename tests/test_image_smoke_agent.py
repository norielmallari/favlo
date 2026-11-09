"""The image agent registers the tool names the harness passes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness.image_smoke_agent import image_mcp, registered, smoke_note, smoke_runs
from harness.mcp_media import published_name
from tests.test_server import _QueueMedia, _Runtime, _serve_background, _stop_background


def test_published_name_prefixes_the_pipeline() -> None:
    assert published_name("z-image-turbo", "generate_image") == "z_image_turbo__generate_image"
    assert published_name("", "z_image_turbo__generate_image") == "z_image_turbo__generate_image"


def test_smoke_runs_follow_the_pipeline_note() -> None:
    root = Path(__file__).resolve().parents[1]
    note = smoke_note(root, "z-image-turbo")
    runs = smoke_runs(note, None)
    assert len(runs) == 1
    assert "quality" not in runs[0]
    assert runs[0]["prompt"] == note["prompt"]
    assert "red cube" in note["question"]
    assert runs[0]["size"] == "512x512"
    assert "model" not in runs[0]
    named = smoke_runs(note, "example-model")
    assert named[0]["model"] == "example-model"


def test_the_named_tool_is_registered_and_returns_the_file(tmp_path: Path) -> None:
    asyncio.run(_registered(tmp_path))


async def _registered(tmp_path: Path) -> None:
    state, thread = _serve_background(_Runtime())
    state["image_home"] = tmp_path
    state["module"] = _QueueMedia(tmp_path)
    port = int(state["server"].server_address[1])
    name = published_name("z-image-turbo", "generate_image")
    png: list[bytes] = []
    try:
        async with image_mcp(port, [name], png) as mcp:
            registered(mcp, [name])
            assert [func.name for func in mcp.functions] == [name]
            result = await mcp.functions[0].invoke(arguments={"prompt": "later"})
    finally:
        _stop_background(state, thread)
    text = _text(result)
    assert "image_later" in text
    assert png == [b"later"]


def _text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(str(getattr(item, "text", "") or "") for item in result)
    return str(result)
