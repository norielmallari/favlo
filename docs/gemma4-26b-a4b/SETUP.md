# Set up Gemma 4 26B-A4B

How to pack and talk to `gemma4-26b-a4b`. The kernel contract is [README.md](README.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). Those weights are under the [Apache License, Version 2.0](https://ai.google.dev/gemma/docs/gemma_4_license).

## What you need

- Apple Silicon
- Metal 4.1
- 48 GB of memory for the full expert set
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A local bfloat16 checkout of that repository
- Free disk for that checkout and the packed record at the same time

`plan` prints the packed sizes: resident 2.824 GiB, float16 vectors 0.066 GiB, and expert images 13.833 GiB. The checkout stays after `quantize`.

16 GB and 24 GB run in degraded mode. The runtime keeps as many expert layers as fit and reads the rest from the record. 8 GB is not enough to run a model capable of real-world agentic workflows.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

Release 0.3.0 is on PyPI as the runtime under `src/favlo`. The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

## Pack

`--src` is the bfloat16 directory, under `models/` or elsewhere on disk. With no `--src`, `quantize` reads `models/gemma-4-26B-A4B-it`.

```bash
uv run favlo --gemma4-26b-a4b quantize
uv run favlo --gemma4-26b-a4b quantize --src models/example-model
```

Either command packs `models/gemma-4-26B-A4B-it-fp8-e20` and registers the folder under the alias `gemma4-26b-a4b` in `models/registry.json`. The row stores `models/gemma-4-26B-A4B-it-fp8-e20`. The checkout stays where it is. A missing directory prints `missing checkpoint:` and that directory's name. The default line is `missing checkpoint: models/gemma-4-26B-A4B-it`.

Resident matrices, including the token embedding and the vision tower, are fp8. Of the 3,840 routed experts, 768 are fp8 and the rest are fp4. Gate-up and down move together. A finished directory has `manifest.json` with `complete: true`, resident plane `fp8`, and expert plane `fp8-e20`. Running `quantize` again on that directory returns. An unfinished directory resumes from `quantize_progress.json`. The tensors must match the Gemma 4 map.

`--name` is the packed folder. The alias on that row stays `gemma4-26b-a4b`. A second folder cannot take an alias another row already uses.

```bash
uv run favlo --gemma4-26b-a4b quantize --name example-model
```

That registers `example-model` and stores `models/example-model` on the row.

## Talk to it

Start the listener:

```bash
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. A terminal `favlo serve` is that process. Ctrl+C stops it. Chat is `POST /v1/responses`. `POST /v1/chat/completions` translates into that request and translates the Response back. Favlo web uses the same listener.

From another terminal:

```bash
curl http://127.0.0.1:8129/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-26b-a4b:instruct","input":"Hello"}'
```

`GET /v1/models` lists the registered ids. A bare `gemma4-26b-a4b` uses the thinking sampling row. `gemma4-26b-a4b:instruct` is the same sampler numbers under another suffix. Both rows are temperature 0.95, top-p 0.95, top-k 64, and presence penalty 0. Thinking follows the resolved `reasoning.effort` (`minimal` off, `high` on) or the row's `enable_thinking`, not the suffix. The server does not check an API key. A client that requires one can send any placeholder, such as `favlo`.

A user message can include a still image. The served window stops at 131072 tokens. The record has no drafter.

One carrot reply, with thinking off:

```bash
uv run favlo --gemma4-26b-a4b complete
```

## If quantize stops

Run the same command again. Files already written stay on disk, and the next run continues from them. A missing checkout is the `missing checkpoint` line. A tensor the Gemma 4 map does not name stops the pack.
