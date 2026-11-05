"""Every Favlo MCP tool description and param description fits the Favlo caps."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from favlo.mcp_server import (
    TOOL_DESCRIPTION_MAX_CHARS,
    TOOL_PARAM_DESCRIPTION_MAX_CHARS,
    build_mcp,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.mcp_tools import _description  # noqa: E402


def _tools():
    mcp = build_mcp(lambda *_args: {}, {}, lambda _text: [0.0] * 1024)
    return asyncio.run(mcp.list_tools(run_middleware=False))


def _param_descriptions(tools: list[object]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for tool in tools:
        schema = tool.to_mcp_tool().input_schema or {}
        props = schema.get("properties") or {}
        for name, prop in props.items():
            if not isinstance(prop, dict):
                continue
            rows.append((tool.name, name, prop.get("description") or ""))
    return rows


def test_every_mcp_tool_description_fits_the_favlo_limit() -> None:
    tools = _tools()
    assert tools
    overs = [
        (tool.name, len(tool.description or ""))
        for tool in tools
        if len(tool.description or "") > TOOL_DESCRIPTION_MAX_CHARS
    ]
    assert overs == [], overs


def test_every_mcp_param_description_fits_the_favlo_limit() -> None:
    tools = _tools()
    overs = [
        (tool_name, param, len(text))
        for tool_name, param, text in _param_descriptions(tools)
        if text and len(text) > TOOL_PARAM_DESCRIPTION_MAX_CHARS
    ]
    assert overs == [], overs


def test_agent_facing_descriptions_fit_after_web_enrichment() -> None:
    """Web appends the list/set model note and may prefix the server name."""
    tools = _tools()
    overs: list[tuple[str, int]] = []
    for tool in tools:
        raw = tool.description or ""
        for exposed in (tool.name, f"favlo_{tool.name}"):
            text = _description({"name": tool.name, "description": raw}, exposed=exposed)
            if len(text) > TOOL_DESCRIPTION_MAX_CHARS:
                overs.append((exposed, len(text)))
    assert overs == [], overs


def test_a_finished_task_result_names_the_file_and_the_link() -> None:
    from favlo.mcp_server import _finished_handles

    found = _finished_handles(
        {
            "content": [
                {"type": "text", "text": "image_abc"},
                {"type": "resource_link", "uri": "favlo://images/image_abc.png"},
            ]
        }
    )
    assert found["names"] == ["image_abc"]
    assert found["links"] == ["favlo://images/image_abc.png"]


def test_tool_description_max_is_256() -> None:
    assert TOOL_DESCRIPTION_MAX_CHARS == 256
    assert TOOL_DESCRIPTION_MAX_CHARS < 1024
    assert TOOL_PARAM_DESCRIPTION_MAX_CHARS == 48
    assert TOOL_PARAM_DESCRIPTION_MAX_CHARS < TOOL_DESCRIPTION_MAX_CHARS
