# Qwen3.5-4B

Language pipeline `qwen35_4b` serves [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache License 2.0) through `favlo serve` after `setup` registers the packed record at `models/Qwen3.5-4B-fp8`. Chat Completions and Responses bodies reach it on the listener once that row is present.

```text
uv run favlo --qwen35_4b --help
uv run favlo --qwen35_4b kernels
uv run favlo --qwen35_4b complete
```

## Setup

Official weights are [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).

- `favlo serve` on a terminal with an empty registry downloads that checkpoint, packs it, and removes the BF16 checkout.
- A sharded checkout and a single `model.safetensors` both pack.
- Prefer the official `chat_template.jinja` (tools and thinking) over a short finetune template.

```text
uv run favlo --qwen35_4b setup
uv run favlo --qwen35_4b setup --keep-bf16
uv run favlo --qwen35_4b quantize --src models/Qwen3.5-4B-bf16
uv run python -m favlo.smoke_lock run --pipeline qwen35_4b --command kernels -- \
  uv run favlo --qwen35_4b kernels
uv run python -m favlo.smoke_lock run --pipeline qwen35_4b --command complete -- \
  uv run favlo --qwen35_4b complete
```

- `quantize` / `setup` write `models/Qwen3.5-4B-fp8`.
- A finished record whose tensor list matches that pack is left in place.
- A record packed before the vision tensors were added is rebuilt.
- `serve --max-memory` and `complete --max-memory` cap the serve ledger; this package does not parse that string.

`GET /v1/models` lists the registered ids.

- A bare `qwen35_4b` uses the thinking card (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, `reasoning_effort` `high`).
- `qwen35_4b:instruct` uses temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, and `reasoning_effort` `none`.
- A request `reasoning.effort` wins when set (`none` or `high`; the other listed names map onto those).
- When the request omits effort, the selected variant's `reasoning_effort` is used.

`mtp-half` appends the official MTP matrices as Metal `half` from the SSD BF16 checkout (`models/Qwen3.5-4B-bf16`, including `mtp.safetensors` when the language shards omit those tensors) onto a finished fp8 record.

- A request field `mtp` wins; omitted uses the registry row (default off).
- `mtp_depth` defaults to 3.

```text
uv run favlo --qwen35_4b mtp-half --src models/Qwen3.5-4B-bf16
```

Kernels, packed record, forward, and registry: [ARCHITECTURE.md](ARCHITECTURE.md).

A bare `qwen35_4b` id uses the thinking variant unless the request names `:instruct`.

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
