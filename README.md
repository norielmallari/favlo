# Favlo

Favlo AI Runtime: Your Local AI Workhorse

<p align="center">
  <img src="docs/images/favlo.png" alt="Favlo AI Runtime" width="480">
</p>

>This app is directly inspired by [oMLX](https://github.com/jundot/omlx). It's an amazing app - I recommend trying that first for general purpose inference before using my more specialized runtime.

Favlo is a local runtime for Apple Silicon. It serves language-model inference and text-to-image generation. One model is one pipeline. Each pipeline owns its forward, its weight records, and its Metal kernels. `favlo serve` is the entry point. A client selects a registered model with the OpenAI `model` field. A language row answers `POST /v1/chat/completions`. A [Z-Image-Turbo](docs/z-image-turbo/README.md) row, packed from [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), answers `POST /v1/images/generations` and returns a PNG.

The process contract is [ARCHITECTURE.md](ARCHITECTURE.md). The papers are [REFERENCES.md](REFERENCES.md). Measured research claims are [RESEARCH.md](RESEARCH.md). Cite this software from [CITATION.cff](CITATION.cff). Agents start at [AGENTS.md](AGENTS.md). AI-assisted features are [docs/ai-assistance](docs/ai-assistance/README.md).

| Pipeline | Note |
|---|---|
| `qwen38-flash-next` | [Qwen3.8-Flash-Next](docs/qwen38-flash-next/architecture.md) |
| `qwen38-27b` | [Qwen3.8-27B](docs/qwen38-27b/README.md) |
| `qwen36-35b-a3b` | [Qwen3.6-35B-A3B](docs/qwen36-35b-a3b/README.md) |
| `z-image-turbo` | [Z-Image-Turbo](docs/z-image-turbo/README.md) |

## Requirements

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

Shaders compile at Metal 4.1. A machine that cannot compile that version cannot run a pipeline. 8 GB is not enough to run a model capable of real-world agentic workflows.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

This installs the `favlo` package and the `dev` extra, including pytest. The package is not published to PyPI.

## Usage

```bash
uv run favlo serve
uv run favlo serve --non-interactive
```

`favlo serve` on a terminal is the CLI. `favlo serve --non-interactive` starts the server daemon and exits. The daemon listens on `127.0.0.1:8129`. On a terminal, an empty registry downloads [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), packs `models/Qwen3.6-35B-A3B-fp4`, and then the daemon listens. Quantize and the other non-serve commands use that model's flag, as in `favlo --qwen38-flash-next quantize`. Chat completions and image generation are both served from that daemon. The listener is [docs/serve.md](docs/serve.md). The image request is [docs/z-image-turbo/README.md](docs/z-image-turbo/README.md). The terminal session is [docs/guides/clients.md](docs/guides/clients.md). The assisted features on that session are [docs/ai-assistance](docs/ai-assistance/README.md).

## Methodology

A pipeline equation comes from a public paper, a published specification, or a vendor document, then maps onto a Metal operation. Another application's source is not a reference. The process citations are [REFERENCES.md](REFERENCES.md). A pipeline's own citations are in that pipeline's note.

The prompt and the generated tokens are separate phases. Patel et al., *Splitwise* ([arXiv:2311.18677](https://arxiv.org/abs/2311.18677)), and Zhong et al., *DistServe* ([arXiv:2401.09670](https://arxiv.org/abs/2401.09670)), split those phases. The key/value cache is what crosses the boundary. Agrawal et al., *SARATHI* ([arXiv:2308.16369](https://arxiv.org/abs/2308.16369)), compute a prompt in chunks. Prefill and decode are Metal shaders. A missing kernel stays missing.

Weights follow the Open Compute Project *Microscaling Formats (MX) Specification*: a block of 32 values and an E8M0 scale. Apple's Metal tensor documents name `matmul2d` and that scale plane. WWDC26 session 330 is the tensor-op session. The Metal shading-language specification is the unpack and storage reference.

## Development

Favlo is an AI-First application. Agentic software delivery is how this repository is built: agents design the change, write the code, and land it. The author directs the work.

## Problems

Report a problem in this repository's issue tracker.

## License

Copyright 2026 Noriel Mallari

Favlo is licensed under the Apache License, Version 2.0. The terms are in [LICENSE.md](LICENSE.md).
