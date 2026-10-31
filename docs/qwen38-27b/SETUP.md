# Set up Qwen3.8-27B

How to download, pack, and talk to `qwen38-27b`. The kernel contract is [README.md](README.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Those weights are under the Apache License, Version 2.0.

## What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the BF16 checkout and, after packing, the packed record beside it

`download` asks Hugging Face for the file list and stops when the volume cannot hold the files still missing. The checkout stays on disk after `quantize`.

8 GB is not enough to run a model capable of real-world agentic workflows. On 16 GB the runtime wires as many feed-forward layers as fit and reads the rest from the record.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

The package is not published to PyPI. Later commands use `uv run favlo` from that checkout.

## Download

```bash
uv run favlo --qwen38-27b download
```

That command downloads [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) into `models/Qwen3.8-27B-bf16` with plain HTTP. A public checkout does not need a token. A short file continues on the next run. A finished download writes `fetch_complete.json`. Running `download` again prints that directory and does not fetch the files again.

Check free space without writing files:

```bash
uv run favlo --qwen38-27b download --check
```

`ok` means the volume can hold the bytes still missing.

`download` does not pack the record and does not register it. An idle `uv run favlo serve` may start this same download when the machine has at least 16 GB of memory and the volume has room. Packing stays a separate command.

## Pack

The default pack stores every quantized matrix as fp4 and writes `models/Qwen3.8-27B-fp4`.

```bash
uv run favlo --qwen38-27b quantize --src models/Qwen3.8-27B-bf16
```

That command registers the folder under the alias `qwen38-27b` in `models/registry.json`. The row stores the directory that was written, and it sets `mtp` true and `mtp_depth` 3 when the row does not already store `mtp`. `models add` does the same. The BF16 checkout stays. A finished directory has `manifest.json` with `complete: true`. Running `quantize` again on that directory returns it when the feed-forward percentage matches. An unfinished directory resumes from `quantize_progress.json`.

With no `--src`, `quantize` reads the Hugging Face cache snapshot for [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). A missing cache is an error. `--src` names a local BF16 checkout, including `models/Qwen3.8-27B-bf16`. `--output` names the packed directory. The alias stays `qwen38-27b`.

## Quantization

`--ffn-fp8` is a percentage of the 64 language feed-forward layers. Omitting it stores every quantized matrix as fp4.

- No `--ffn-fp8` writes `models/Qwen3.8-27B-fp4`. This is the default.
- `100` stores all 64 language feed-forward layers as fp8 and writes `models/Qwen3.8-27B-fp8`.
- Any other percentage stores that share of the language feed-forward layers as fp8 and writes `models/Qwen3.8-27B-fp8-e<P>`, where `<P>` is that percentage. `20` writes `models/Qwen3.8-27B-fp8-e20`. The count is the largest whole number that does not exceed that percentage, so `20` is 12 layers. Those layers are the highest BF16-to-fp4 relative errors. Gate, up, and down move together.

The embedding, the projections, `lm_head`, and the vision tower stay fp4. MTP matrices are fp8 E4M3. One-dimensional parameters, including MTP norms, stay float16. `mtp-fp8` rewrites a finished record whose MTP matrices are still fp4.

```bash
uv run favlo --qwen38-27b quantize --src models/Qwen3.8-27B-bf16 --ffn-fp8 100
uv run favlo --qwen38-27b quantize --src models/Qwen3.8-27B-bf16 --ffn-fp8 20
```

## Talk to it

Start the listener:

```bash
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. The terminal is the operator console. `quit` stops the daemon. `reload` unloads the open pipelines and stays attached. Chat is `POST /v1/chat/completions` or Favlo web.

From another terminal:

```bash
curl http://127.0.0.1:8129/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b:instruct","messages":[{"role":"user","content":"Hello"}]}'
```

`GET /v1/models` lists the registered ids. A bare `qwen38-27b` uses the thinking card (temperature 1.0, top-p 0.95, top-k 20, presence penalty 0). `qwen38-27b:instruct` uses temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5. `enable_thinking` set to false selects the instruct card when the id has no variant suffix. A request that omits `reasoning_effort`, or sends `none`, leaves the template default, xhigh. The server does not check an API key. A client that requires one can send any placeholder, such as `favlo`.

A user message can include a still image. The served window stops at 131072 tokens.

## Delete a model

```bash
uv run favlo --qwen38-27b delete Qwen3.8-27B-fp4
uv run favlo --qwen38-27b delete qwen38-27b
```

`delete` takes the folder name or the alias. A `:instruct` or `:thinking` suffix deletes that same model. The command removes the registry row, the packed directory under `models/`, and the BF16 checkout when that checkout is still under `models/` and no other row uses it. It also removes that record's prefix files under `~/.favlo/prefix/qwen38-27b/`. A checkout outside `models/` is left in place. Another model's directory and row stay.

## If download stops

Run `download` again. Files already written stay on disk, and the next run continues from them. A line that names a GiB count is free space. A failed download prints `the next download continues from the files already written`.
