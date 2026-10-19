# MADMetal

>Author's Note: This app is directly inspired by [oMLX](https://github.com/jundot/omlx). It's an amazing app - I recommend trying that first for general purpose inference before using my more specialized runtime.

**MADMetal Accelerated Decoding on Metal.**

One model, one inference pipeline. MADMetal does not host a family of models behind a shared runner. Each model owns its forward, its weight records, and its Metal kernels. `madmetal serve` is the entry point for inference. A client selects a model with the OpenAI `model` field. Quantize and the other non-serve commands use that model's flag, as in `madmetal --qwen38-flash-next quantize`.

Each runtime keeps its own package, kernels, and as-built note. The pipelines are listed in [ARCHITECTURE.md](ARCHITECTURE.md). The papers and specifications are [REFERENCES.md](REFERENCES.md). Agents start at [AGENTS.md](AGENTS.md).

## Running the Server

`serve` listens on `127.0.0.1:8129`. The registry starts empty. `quantize` registers the record it wrote. A request's `model` field selects one of those rows. One pipeline is resident. A different pipeline releases the resident record before it loads.

```text
uv sync --extra dev
uv run madmetal serve
uv run python -m harness.file_agent --base-url http://127.0.0.1:8129/v1 --workspace ./note-workspace
```

The harness is a [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) client. `python -m harness.file_agent` creates a file, replaces its text, and confirms the result against a server that is already listening. `python -m harness --model <pipeline>` starts `serve`, reads repository files until the prompt reaches the `--tokens` target (default 65536), and prints prefill and decode rates. The request sets `enable_thinking` false. The server stops after the first turn and starts again, so the second turn loads the prefix from disk. Later turns stay on that server. A request that does not extend the stored token ids stops the run. `python -m harness.coding --model <pipeline>` sets `enable_thinking` true. The model writes `ledger.py`. Exit 0 is a pass, and the command prints that run's prefill and decode lines.

## Development

MADMetal is an AI-first application. Agentic software delivery is how this repository is built: agents design the change, write the code, and land it. The author directs the work.

## Local Inference First

The objective is several models on one machine, under the memory that machine can give them. The registry holds the records. One pipeline is resident. A request for a different model releases that record, its device buffers, and its live prefix, then loads the pipeline the request names. The listener serves one completion at a time, on the machine that stores the weights.

`--max-memory` is the byte cap for a pipeline that enforces one. Sequence state that is not the live prompt is written to the SSD prefix store and read back when a later turn extends those token ids. Each pipeline keeps its own weight records and Metal kernels, sized for the memory it is given.

Batching, and other work that spends memory on extra sequences or extra copies of the weights, is deprioritized. The resident set stays one pipeline, so the machine can switch among the models it has registered. The process contract is [ARCHITECTURE.md](ARCHITECTURE.md).

## Problems

Report a problem in this repository's issue tracker.

## License

Copyright 2026 Noriel Mallari

MADMetal is licensed under the Apache License, Version 2.0. The terms are in [LICENSE.md](LICENSE.md).
