# MADMetal

**MADMetal Accelerated Decoding on Metal.**

The author admires the work of the authors of [oMLX](https://github.com/jundot/omlx) and recommends their app before trying MADMetal.

One model, one inference pipeline. MADMetal does not host a family of models behind a shared runner. Each model owns its forward, its weight records, and its Metal kernels. `madmetal serve` is the entry point for inference. A client selects a model with the OpenAI `model` field. Quantize and the other non-serve commands use that model's flag, as in `madmetal --qwen38-flash-next quantize`.

The first pipeline is Qwen3.8-Flash-Next on an M5 Pro: native Metal tensor records, a single-request OpenAI-compatible server, and a prefix cache for agentic coding. `qwen38-flash-next.py` loads the `qwen38_flash_next` package. Each runtime keeps its own package, kernels, and as-built note. The process contract is [ARCHITECTURE.md](ARCHITECTURE.md). The papers and specifications are [REFERENCES.md](REFERENCES.md). Agents start at [AGENTS.md](AGENTS.md).

## Running the Server

`serve` listens on `127.0.0.1:8129`. The registry starts empty. `quantize` registers the record it wrote. A request's `model` field selects one of those rows. One pipeline is resident. A different pipeline releases the resident record before it loads.

```text
uv sync --extra dev
uv run madmetal serve
uv run python -m harness --workspace ./note-workspace
```

The harness is a [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) client. It creates a file, replaces its text, and confirms the result. `python -m harness.long_prompt` reads repository files across turns until the prompt reaches 65,536 tokens and prints prefill and decode rates. The server keeps that live conversation in memory. A request that does not extend it is written under `~/.madmetal/prefix` and loaded again when those token ids return.

## Milestone

The first milestone is that agentic loop on this server.

## Development

Code in this repository is written with AI assistance and reviewed before it is kept. This note is disclosure only. It does not replace the license, and it does not grant or limit any rights under the license.

## Problems

Report a problem in this repository's issue tracker.

## License

Copyright 2026 Noriel Mallari

MADMetal is licensed under the Apache License, Version 2.0. The terms are in [LICENSE.md](LICENSE.md).
