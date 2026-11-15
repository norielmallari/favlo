# Qwen3.5-4B

Summary for the `qwen35_4b` language pipeline. Kernels, the packed record, the forward path, chat template, and registry fields are in [ARCHITECTURE.md](ARCHITECTURE.md).

- `handle_responses` is a `LanguagePipeline` served through `favlo serve`.
- After a record is registered, `GET /v1/models` lists it and `model` on `POST /v1/responses` loads it.
- Package commands (`download`, `quantize`, `setup`, `mtp-half`, `kernels`, `complete`, `models`) require `favlo --qwen35_4b`, which loads `qwen35_4b.metal41`.

- The official checkpoint is [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- Those weights are under the Apache License, Version 2.0.
- Registration sets `default_variant` to `thinking` and `kv_cache` to `fp8_e4m3` unless the request overrides them.
- The packed record is `models/Qwen3.5-4B-fp8`.
- The served context window is 262144 tokens.

After changing an open-time registry field, unload the resident model:

```text
uv run python -c 'from favlo.smoke_lock import unload_model; unload_model()'
```

## Setup

How to download, pack, and talk to `qwen35_4b`. The kernel contract is [ARCHITECTURE.md](ARCHITECTURE.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

### What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the BF16 checkout and the packed record at the same time

- `setup` downloads the official checkpoint, packs `models/Qwen3.5-4B-fp8`, registers the alias `qwen35_4b`, and removes the BF16 checkout.
- `--keep-bf16` leaves that checkout on disk.
- A sharded checkout and a single `model.safetensors` both pack.
- Prefer the official `chat_template.jinja` (tools and thinking) over a short finetune template.
- 8 GB is not enough to run a model capable of real-world agentic workflows.

### Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

This checkout is release 0.6.0. PyPI publishes the runtime under `src/favlo`. The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

### Download and pack

```bash
uv run favlo --qwen35_4b setup
```

That command does four things:

1. Downloads [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) into `models/Qwen3.5-4B-bf16` with plain HTTP. A public checkout does not need a token. A short file continues on the next run.
2. Packs `models/Qwen3.5-4B-fp8`. Language and vision matrices are fp8 E4M3. Norms, gains, and related vectors stay Metal `half`. MTP weights are left unpacked.
3. Registers the folder under the alias `qwen35_4b` in `models/registry.json`. The row stores `models/Qwen3.5-4B-fp8`.
4. Removes `models/Qwen3.5-4B-bf16`. `--keep-bf16` leaves that checkout on disk.

A finished directory has `manifest.json` with `complete: true`. Running `setup` again on that directory registers it, if it is not already registered, and returns.

- An empty registry on a terminal runs `favlo --qwen35_4b setup` before the daemon starts.
- It first checks that a small shader compiles at Metal 4.1.
- A machine that cannot compile that version prints `this runtime cannot run here` and exits.
- `uv run favlo serve --non-interactive` does not download.
- A pipe does not download.

Keep the BF16 checkout:

```bash
uv run favlo --qwen35_4b setup --keep-bf16
```

Pack a checkout that is already on disk:

```bash
uv run favlo --qwen35_4b quantize --src models/Qwen3.5-4B-bf16
```

- With no `--src`, `quantize` reads `models/Qwen3.5-4B-bf16`.
- `--src` names another checkout.
- `quantize` leaves the checkout in place.
- A finished record whose tensor list matches that pack is left in place.
- A record packed before the vision tensors were added is rebuilt.
- `serve --max-memory` and `complete --max-memory` cap the serve ledger; this package does not parse that string.

Download only:

```bash
uv run favlo --qwen35_4b download
uv run favlo --qwen35_4b download --check
```

### MTP half

`mtp-half` appends the official MTP matrices as Metal `half` from the SSD BF16 checkout onto a finished fp8 record. The checkout may include `mtp.safetensors` when the language shards omit those tensors.

```bash
uv run favlo --qwen35_4b mtp-half --src models/Qwen3.5-4B-bf16
```

- A request field `mtp` wins; omitted uses the registry row (default off).
- `mtp_depth` defaults to 3.

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
  -d '{"model":"qwen35_4b:instruct","input":"Hello"}'
```

- `GET /v1/models` lists the registered ids.
- A bare `qwen35_4b` uses the thinking card (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, `reasoning_effort` `high`).
- `qwen35_4b:instruct` uses temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, and `reasoning_effort` `none`.
- A request `reasoning.effort` wins when set (`none` or `high`; the other listed names map onto those).
- When the request omits effort, the selected variant's `reasoning_effort` is used.
- The server does not check an API key.
- A client that requires one can send any placeholder, such as `favlo`.

A user message can include a still image. The template does not accept an audio part. The served window stops at 262144 tokens.

### Package checks

```bash
uv run favlo --qwen35_4b kernels
uv run favlo --qwen35_4b complete
```

### If setup stops

- Run the same command again.
- Files already written stay on disk, and the next run continues from them.
- A `download failed` line is the HTTP error from `huggingface.co`.

## Further reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — registry, kernels, record, forward, chat template, sampling
- [references.md](references.md) — papers and config fields
- [ceilings.md](ceilings.md) — kernel ceilings
- [research/int8-prefill.md](research/int8-prefill.md) — int8 prefill remainder
- [research/kernel-ceilings.md](research/kernel-ceilings.md) — isolated dispatch sum
- [research/recorded-forward.md](research/recorded-forward.md) — carrot smoke and coding-agent rows
- [research/needle-page.md](research/needle-page.md) — page-mode needle
- [research/coding-agent.md](research/coding-agent.md) — vision coding agent
- [research/coding-agent-mtp.md](research/coding-agent-mtp.md) — coding agent with MTP
- [research/coding-text.md](research/coding-text.md) — text coding agent
- [research/synthetic-prompt.md](research/synthetic-prompt.md) — cold 512, 1024, 4096, and 10000-token prompts on fp8 weights and `fp8_e4m3` keys and values
