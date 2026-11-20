# Gemma 4 26B-A4B

Summary for the `gemma4_26b_a4b` language pipeline. Kernels, the packed record, the forward path, chat template, and registry fields are in [ARCHITECTURE.md](ARCHITECTURE.md).

- `handle_responses` is a `LanguagePipeline` served through `favlo serve`.
- After a record is registered, `GET /v1/models` lists it and `model` on `POST /v1/responses` loads it.
- `plan`, `quantize`, `kernels`, and `complete` require `favlo --gemma4_26b_a4b`.
- Windows loads `gemma4_26b_a4b.cuda`. Every other host loads `gemma4_26b_a4b.metal41`.

- The official checkpoint is [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).
- Those weights are under the [Apache License, Version 2.0](https://ai.google.dev/gemma/docs/gemma_4_license).
- A bare `gemma4_26b_a4b` uses the thinking card unless the request names instruct.
- Registration sets `kv_cache` to `half`.
- The served window stops at 131072 tokens.
- The Metal record is `gemma-4-26B-A4B-it-fp8-e20`.
- The CUDA record is `gemma-4-26B-A4B-it-int8-e10`. Dense matrices are int8. Routed experts are int4.
- `quantize` does not download.

After changing an open-time registry field, unload the resident model:

```text
uv run python -c 'from favlo.smoke_lock import unload_model; unload_model()'
```

## Setup

How to pack and talk to `gemma4_26b_a4b`. The kernel contract is [ARCHITECTURE.md](ARCHITECTURE.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

### What you need

- Apple Silicon
- Metal 4.1
- 48 GB of memory for the full expert set
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A local bfloat16 checkout of that repository
- Free disk for that checkout and the packed record at the same time

`plan` prints the packed sizes:

- `resident` 2.824 GiB
- `half` 0.066 GiB
- `experts` 13.833 GiB

The checkout stays after `quantize`.

- 16 GB and 24 GB run in degraded mode.
- The runtime keeps as many expert layers as fit and reads the rest from the record.
- 8 GB is not enough to run a model capable of real-world agentic workflows.

### Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

This checkout is release 0.6.0. PyPI publishes the runtime under `src/favlo`. The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

### Pack

`--src` is the bfloat16 directory, under `models/` or elsewhere on disk. With no `--src`, `quantize` reads `models/gemma-4-26B-A4B-it`.

```bash
uv run favlo --gemma4_26b_a4b quantize
uv run favlo --gemma4_26b_a4b quantize --src models/example-model
```

Either command packs `models/gemma-4-26B-A4B-it-fp8-e20`. It registers the folder under the alias `gemma4_26b_a4b` in `models/registry.json`. The row stores `models/gemma-4-26B-A4B-it-fp8-e20`. The checkout stays where it is.

A missing directory prints `missing checkpoint:` and that directory's name. The default line is `missing checkpoint: models/gemma-4-26B-A4B-it`.

- Resident matrices, including the token embedding and the vision tower, are fp8.
- Of the 3,840 routed experts, 768 are fp8 and the rest are fp4.
- Gate-up and down move together.
- A finished directory has `manifest.json` with `complete: true`, resident plane `fp8`, and expert plane `fp8-e20`.
- Running `quantize` again on that directory returns.
- An unfinished directory resumes from `quantize_progress.json`.
- The tensors must match the Gemma 4 map.

`--name` is the packed folder. The alias on that row stays `gemma4_26b_a4b`. A second folder cannot take an alias another row already uses.

```bash
uv run favlo --gemma4_26b_a4b quantize --name example-model
```

That registers `example-model` and stores `models/example-model` on the row.

### Talk to it

Start the listener:

```bash
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. A terminal `favlo serve` is that process. Ctrl+C stops it. Chat is `POST /v1/responses`. `POST /v1/chat/completions` translates into that request and translates the Response back. Favlo web uses the same listener.

From another terminal:

```bash
curl http://127.0.0.1:8129/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4_26b_a4b:instruct","input":"Hello"}'
```

- `GET /v1/models` lists `gemma4_26b_a4b`, `gemma4_26b_a4b:thinking`, and `gemma4_26b_a4b:instruct`.
- The packed folder name is not a request id.
- A bare `gemma4_26b_a4b` uses the thinking card.
- Both cards are temperature 0.95, top-p 0.95, top-k 64, and presence penalty 0.
- The thinking card stores `reasoning_effort` `high`.
- The instruct card stores `minimal`.
- A request that omits effort uses that card's effort.
- A request that sets `reasoning.effort` still keeps the selected card's sampler numbers.
- A request `preserve_thinking` value is ignored.
- A tool-call thought stays on the next user turn.
- A plain assistant thought drops.
- An open thought closes after 2048 reasoning tokens, and the reply continues.
- The server does not check an API key.
- A client that requires one can send any placeholder, such as `favlo`.

A user message can include a still image. The served window stops at 131072 tokens. The record has no drafter.

One carrot reply, with thinking off:

```bash
uv run favlo --gemma4_26b_a4b complete
```

### If quantize stops

- Run the same command again.
- Files already written stay on disk, and the next run continues from them.
- A missing checkout is the `missing checkpoint` line.
- A tensor the Gemma 4 map does not name stops the pack.

## Further reading

- [ceilings.md](ceilings.md) — bandwidth and kernel ceilings
- [benchmarks.md](benchmarks.md) — harness scores
- [research/kernel-ceilings.md](research/kernel-ceilings.md) — packed kernel ceiling
- [research/recorded-forward.md](research/recorded-forward.md) — recorded prefill and decode runs
- [research/coding-agent.md](research/coding-agent.md) — vision coding-agent rates on fp8 keys and values
- [research/coding-agent-half.md](research/coding-agent-half.md) — vision coding-agent rates on the fp8-e20 pack with half keys and values
- [research/coding-text.md](research/coding-text.md) — text coding-agent rate
- [research/coding-text-cuda.md](research/coding-text-cuda.md) — text coding-agent rate on CUDA with int4 experts and half keys and values
- [research/needle-page.md](research/needle-page.md) — page-needle rate
- [research/synthetic-prompt.md](research/synthetic-prompt.md) — cold synthetic prompts on fp8 keys and values
- [research/synthetic-prompt-half.md](research/synthetic-prompt-half.md) — cold 512, 1024, 4096, and 10000-token prompts on the fp8-e20 pack with half keys and values
- [research/synthetic-prompt-cuda.md](research/synthetic-prompt-cuda.md) — cold 512, 1024, 4096, and 10000-token prompts on CUDA with int4 experts and half keys and values
