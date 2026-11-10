# Harness

`harness/` scores packages in this checkout. Do not add a harness here that names or imports a package from outside this checkout.

A media agent that calls one generate tool takes `--pipeline` and one or more `--tool` local names. It builds `{pipeline}__{tool}` with `harness.mcp_media.published_name` and registers those names on a Microsoft Agent Framework `MCPStreamableHTTPTool`. The smoke body comes from `docs/<pipeline>/smoke.json`.

Wrong: a published tool name hardcoded in source, a smoke path with no `--pipeline`, or a package import inside a harness.

Language, coding, needle, synthetic, and benchmark harnesses take `--model`. They are not media generate agents. `harness.media_pipeline` discovers registered media packages. Helpers are not end-to-end scores.

A harness CLI change updates `docs/harness.md` in the same change.

How to run a smoke, a coding agent, or a listener session is the smoke-testing skill and the favlo-server skill. Do not paste those commands here. A language check takes `--model`. It does not hardcode a model id.
