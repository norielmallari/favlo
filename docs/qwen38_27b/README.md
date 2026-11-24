# Qwen3.8-27B

Summary for the `qwen38_27b` language pipeline on Metal 4.1. Kernels, the packed record, the forward path, and registry fields are in [ARCHITECTURE.md](ARCHITECTURE.md).

- The official checkpoint is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- Those weights are under the Apache License, Version 2.0.
- Clients reach this pipeline through `favlo serve`, which loads `qwen38_27b.portable`.
- `portable` is the serve entry. `metal41` is the Metal device that entry opens on Apple Silicon.
- `GET /v1/models` lists registered ids.
- The `model` field on `POST /v1/responses` or `POST /v1/chat/completions` loads it.
- Package commands (`download`, `quantize`, `mtp-half`, `delete`, `kernels`, `bf16`, `complete`, `models`) require `favlo --qwen38_27b`, which loads that same entry.

- A bare model id uses the thinking sampling card.
- The served context window stops at 131072 tokens.

After changing an open registry field, close the resident model:

```text
uv run python -c 'from favlo.smoke_lock import unload_model; unload_model()'
```

## Setup

How to download, pack, and talk to `qwen38_27b`. The kernel contract is [ARCHITECTURE.md](ARCHITECTURE.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

### What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the BF16 checkout and, after packing, the packed record beside it

- `download` asks Hugging Face for the file list and stops when the volume cannot hold the files still missing.
- The checkout stays on disk after `quantize`.
- 8 GB is not enough to run a model capable of real-world agentic workflows.
- On 16 GB the runtime wires as many feed-forward layers as fit and reads the rest from the record.

### Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

This checkout is release 0.6.0. PyPI publishes the runtime under `src/favlo`. The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

### Download

```bash
uv run favlo --qwen38_27b download
```

- That command downloads [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) into `models/Qwen3.8-27B-bf16` with plain HTTP.
- A public checkout does not need a token.
- A short file continues on the next run.
- A finished download writes `fetch_complete.json`.
- Running `download` again prints that directory and does not fetch the files again.

Check free space without writing files:

```bash
uv run favlo --qwen38_27b download --check
```

`ok` means the volume can hold the bytes still missing.

`download` does not pack the record and does not register it. Packing stays a separate command.

### Pack

The default pack writes `models/Qwen3.8-27B-fp4`. Language projections and the feed-forward are fp4. The embedding, `lm_head`, MTP, and the vision tower are fp8.

```bash
uv run favlo --qwen38_27b quantize --src models/Qwen3.8-27B-bf16
```

- That command registers the folder under the alias `qwen38_27b` in `models/registry.json`.
- The row stores the directory that was written.
- It sets `mtp` true and `mtp_depth` 3 when the row does not already store `mtp`.
- `models add` does the same.
- The BF16 checkout stays.
- A finished directory has `manifest.json` with `complete: true`.
- Running `quantize` again on that directory returns it when the feed-forward percentage matches.
- An unfinished directory resumes from `quantize_progress.json`. A tensor is recorded only after its bytes are on disk. A layout 2 tensor with no digest, or whose bytes do not match, is written again.

- With no `--src`, `quantize` reads the Hugging Face cache snapshot for [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- A missing cache is an error.
- `--src` names a local BF16 checkout, including `models/Qwen3.8-27B-bf16`.
- `--output` names the packed directory.
- The alias stays `qwen38_27b`.

### Quantization

`--ffn-fp8` is a percentage of the 64 language feed-forward layers. Omitting it stores the language projections and the feed-forward as fp4.

- No `--ffn-fp8` writes `models/Qwen3.8-27B-fp4`. This is the default.
- `100` stores all 64 language feed-forward layers as fp8 and writes `models/Qwen3.8-27B-fp8`.
- Any other percentage stores that share of the language feed-forward layers as fp8 and writes `models/Qwen3.8-27B-fp8-e<P>`, where `<P>` is that percentage.
- `20` writes `models/Qwen3.8-27B-fp8-e20`.
- The count is the largest whole number that does not exceed that percentage, so `20` is 12 layers.
- Those layers are the highest BF16-to-fp4 relative errors.
- Gate, up, and down move together.

- The embedding and `lm_head` are fp8 E4M3. Language projections stay fp4.
- Vision-tower matrices are fp8 E4M3. Vision norms and biases stay Metal `half`.
- MTP matrices are fp8 E4M3 in `mtp.bin`. One-dimensional MTP parameters stay Metal `half`. The file is opened only when MTP is on.

`calibrate` rewrites a finished record. Language projections use GPTQ with the FOEM update. The embedding, `lm_head`, the vision tower, and MTP stay as `quantize` wrote them.

```bash
uv run favlo --qwen38_27b calibrate --src models/Qwen3.8-27B-bf16 --record models/Qwen3.8-27B-fp4
```

- A missing record is packed first. Language projections are round-to-nearest fp4. The embedding, `lm_head`, MTP, and the vision tower are round-to-nearest fp8.
- Each pass covers eight language layers. The float32 input Gram accumulates on device, and the column update runs on device. BF16 rows are read in sequential 8 MiB pieces.
- The command registers that folder under the alias `qwen38_27b` when it finishes.
- Running it again on a manifest whose `quantization` is `gptq-foem` returns the directory.
- MTP matrices are fp8 E4M3 in `mtp.bin`. Vision tensors are `vision.bin`. A text open does not open either file.
- One-dimensional parameters, including MTP norms, stay Metal `half`.
- `mtp-half` rewrites MTP matrices in `mtp.bin` to Metal `half`.

```bash
uv run favlo --qwen38_27b quantize --src models/Qwen3.8-27B-bf16 --ffn-fp8 100
uv run favlo --qwen38_27b quantize --src models/Qwen3.8-27B-bf16 --ffn-fp8 20
```

### Talk to it

Start the listener:

```bash
uv run favlo serve
```

- The daemon listens on `http://127.0.0.1:8129`.
- A terminal `favlo serve` is that process.
- Ctrl+C stops it.
- Chat is `POST /v1/responses`.
- `POST /v1/chat/completions` translates into that request and translates the Response back.
- Favlo web uses the same listener.

From another terminal:

```bash
curl http://127.0.0.1:8129/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38_27b:instruct","input":"Hello"}'
```

`GET /v1/models` lists the registered ids.

- A bare `qwen38_27b` uses the thinking sampling card: temperature 1.0, top-p 0.95, top-k 20, presence penalty 0.
- `qwen38_27b:instruct` uses temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5.
- Those suffixes are sampling presets only.
- Thinking follows the resolved `reasoning.effort` (`none`, `low`, `medium`, or `xhigh`; the other listed names map onto those) or the row's `reasoning_effort` / `enable_thinking`.
- A request that omits effort leaves the template default, xhigh, when thinking is open.
- The server does not check an API key.
- A client that requires one can send any placeholder, such as `favlo`.

- A user message can include a still image.
- The served window stops at 131072 tokens.
- `preserve_thinking`, tool-call markup, and prefix partition fields are in [ARCHITECTURE.md](ARCHITECTURE.md).

### Delete a model

```bash
uv run favlo --qwen38_27b delete Qwen3.8-27B-fp4
uv run favlo --qwen38_27b delete qwen38_27b
```

`delete` takes the folder name or the alias. A `:instruct` or `:thinking` suffix deletes that same model.

The command removes:

- the registry row
- the packed directory under `models/`
- the BF16 checkout when that checkout is still under `models/` and no other row uses it
- that record's prefix files under `~/.favlo/prefix/qwen38_27b/`

A checkout outside `models/` is left in place. Another model's directory and row stay.

### If download stops

- Run `download` again.
- Files already written stay on disk, and the next run continues from them.
- A line that names a GiB count is free space.
- A failed download prints `the next download continues from the files already written`.

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — registry, kernels, record, forward, chat template, sampling
- [ceilings.md](ceilings.md) — per-dispatch kernel ceilings
- [memory-usage.md](memory-usage.md) — device bytes at the served window
- [benchmarks.md](benchmarks.md) — harness scores
- Research: [kernel-ceilings](research/kernel-ceilings.md), [recorded-forward](research/recorded-forward.md), [int8-prefill](research/int8-prefill.md), [mixed-agent-forward](research/mixed-agent-forward.md), [coding-agent](research/coding-agent.md), [coding-agent-half](research/coding-agent-half.md), [needle-page](research/needle-page.md), [needle-retrieve](research/needle-retrieve.md), [synthetic-prompt](research/synthetic-prompt.md), [synthetic-prompt-half](research/synthetic-prompt-half.md), [synthetic-prompt-fp4](research/synthetic-prompt-fp4.md), [synthetic-prompt-mtp](research/synthetic-prompt-mtp.md), [synthetic-prompt-mtp-half](research/synthetic-prompt-mtp-half.md), [synthetic-prompt-mtp-half-depth2](research/synthetic-prompt-mtp-half-depth2.md), [synthetic-prompt-mtp-half-depth4](research/synthetic-prompt-mtp-half-depth4.md), [synthetic-prompt-mtp-half-depth6](research/synthetic-prompt-mtp-half-depth6.md), [span-draft](research/span-draft.md), [mtp-window](research/mtp-window.md), [ffn-fp8-budget](research/ffn-fp8-budget.md)
