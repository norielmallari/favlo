# Listener

`MCP.md` at the repository root is the public catalog of `POST /mcp` tools and `favlo://` resources.

When a change adds, removes, or renames a tool or resource, changes tool parameters or defaults, changes optional-task behavior, or changes what generate tools return, update `MCP.md` in the same change.

`docs/serve.md` stays the listener note. Do not duplicate the tool tables there. Do not document chat completion or OpenAI HTTP routes in `MCP.md`. Do not add HTTP `/v1/images`, `/v1/videos`, or `/v1/audio` routes.

Public wording is `docs/AGENTS.md`.

Tool descriptions stay at or under `TOOL_DESCRIPTION_MAX_CHARS` (256) in `mcp_server.py`. A pipeline tool is not added to `MCP.md`. Each parameter description stays at or under `TOOL_PARAM_DESCRIPTION_MAX_CHARS` (48). Prefer an empty property description when the name and type are enough.

When a listener tool docstring, an Args line, an explicit `description=`, or memory tool copy changes, run:

```bash
uv run pytest tests/test_mcp_tool_descriptions.py -q
```

That suite must stay green.
