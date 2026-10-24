# Set up Qwen3.6-35B-A3B

How to download, pack, and talk to `qwen36-35b-a3b`. The kernel contract is [README.md](README.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Those weights are under the Apache License, Version 2.0.

## What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the BF16 checkout and the packed record at the same time

The default pack is `fp8-e20`. `setup` reserves 22.3 GiB for that record (the routed expert images and the resident weights). Another `--expert-fp8` value reserves the size of that pack instead. The checkout is whatever the repository publishes. `setup` adds those two sizes and stops when the volume does not have that much free. After the record is registered, the checkout is removed.

8 GB is not enough to run a model capable of real-world agentic workflows. On 16 GB the runtime keeps a smaller set of expert images in memory and reads the rest from the record.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

The package is not published to PyPI. Later commands use `uv run favlo` from that checkout.

## Download and pack

```bash
uv run favlo --qwen36-35b-a3b setup
```

That command does four things:

1. Downloads [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) into `models/Qwen3.6-35B-A3B-bf16` with plain HTTP. A public checkout does not need a token. A short file continues on the next run.
2. Packs `models/Qwen3.6-35B-A3B-fp8-e20`. Twenty percent of the routed experts are fp8. The rest of those experts are fp4. The language projections, the token embedding, `lm_head`, the vision tower, and the shared expert are fp8.
3. Registers the folder under the alias `qwen36-35b-a3b` in `models/registry.json`. The row stores `models/Qwen3.6-35B-A3B-fp8-e20`.
4. Removes `models/Qwen3.6-35B-A3B-bf16`. `--keep-bf16` leaves that checkout on disk.

A finished directory has `manifest.json` with `complete: true`. Running `setup` again on that directory registers it, if it is not already registered, and returns.

`uv run favlo serve` on a terminal does the same download when `models/registry.json` has no rows. It first checks that a small shader compiles at Metal 4.1. A machine that cannot compile that version prints `this runtime cannot run here` and exits. `uv run favlo serve --non-interactive` does not download. A pipe does not download.

## Talk to it

Start the listener:

```bash
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. The terminal prompt is `you:`. A line is sent when it is ready. The reply is `favlo:`. `/quit` stops the daemon.

From another terminal:

```bash
curl http://127.0.0.1:8129/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-35b-a3b:instruct","messages":[{"role":"user","content":"Hello"}]}'
```

`GET /v1/models` lists the registered ids. A bare `qwen36-35b-a3b` uses the thinking card (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5). `qwen36-35b-a3b:instruct` uses temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5. `enable_thinking` set to false selects the instruct card when the id has no variant suffix. The server does not check an API key. A client that requires one can send any placeholder, such as `favlo`.

A user message can include a still image. The template does not accept an audio part. The served window stops at 131072 tokens.

## If setup stops

Run the same command again. Files already written stay on disk, and the next run continues from them. A line that names a GiB count is free space: the volume needs room for the checkout and the record together until the checkout is removed. A `download failed` line is the HTTP error from `huggingface.co`.

## Quantization

`--expert-fp8` is a percentage from 0 to 100. The default is 20, which is the `fp8-e20` pack. When `--name` is omitted, the packed folder is the repository's model name plus that format.

- `0` stores every quantized matrix as fp4. The folder ends in `-fp4`.
- `20` stores 20 percent of the routed experts as fp8 and the rest as fp4. The language projections, the token embedding, `lm_head`, the vision tower, and the shared expert are fp8. The folder ends in `-fp8-e20`.
- `100` stores every routed expert as fp8. The folder ends in `-fp8-e100`.
- Any other percentage stores that share of the routed experts as fp8. The folder ends in `-fp8-e<P>`, where `<P>` is the percentage. The count is the largest whole number that does not exceed that percentage.

```bash
uv run favlo --qwen36-35b-a3b setup --expert-fp8 0
uv run favlo --qwen36-35b-a3b setup --repo example-org/example-model --expert-fp8 100
```

The first command packs `models/Qwen3.6-35B-A3B-fp4`. The second packs `models/example-model-fp8-e100`. `quantize` accepts the same `--expert-fp8` and does not download. With no `--src`, it reads `models/Qwen3.6-35B-A3B-bf16`. `--src` names another checkout. After a successful quantize, a checkout under `models/` is removed. `--keep-bf16` leaves it. A `--src` outside `models/` is left in place.

## Another checkpoint

`setup` can download a different Hugging Face repository that uses the same tensor names as [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). With no `--name`, the folder and the registry id are the model name plus the quant format. The default is `fp8-e20`.

`--repo` accepts `org/name` or a `huggingface.co` model link. A link's extra path, such as `/tree/main`, is ignored.

```bash
uv run favlo --qwen36-35b-a3b setup --repo example-org/example-model
uv run favlo --qwen36-35b-a3b setup --repo https://huggingface.co/example-org/example-model
```

The checkout is `models/example-org--example-model-bf16`. It is removed after `models/example-model-fp8-e20` is registered. `--keep-bf16` leaves the checkout. The client model field is `example-model-fp8-e20`. `example-model-fp8-e20:instruct` and `example-model-fp8-e20:thinking` select the same cards as the official alias.

`--name` replaces that folder. The name is used as written, and it is the registry id.

```bash
uv run favlo --qwen36-35b-a3b setup --repo example-org/example-model --name example-model
```

That registers `example-model` and stores `models/example-model` on the row. The official alias `qwen36-35b-a3b` stays on `models/Qwen3.6-35B-A3B-fp8-e20`. A different percentage of the official checkpoint is registered under its folder name, such as `Qwen3.6-35B-A3B-fp4`.

`quantize --name example-model` packs a checkout that is already on disk into that folder and registers the same id. With no `--name` and no `--output`, the folder is the checkout directory name, dropping a trailing `-bf16`, plus the quant format.

## Delete a model

```bash
uv run favlo --qwen36-35b-a3b delete example-model-fp8-e20
uv run favlo --qwen36-35b-a3b delete qwen36-35b-a3b
```

`delete` takes the folder name or the alias. A `:instruct` or `:thinking` suffix deletes that same model. The command removes the registry row, the packed directory under `models/`, and the BF16 checkout when that checkout is still under `models/` and no other row uses it. It also removes that record's prefix files under `~/.favlo/prefix/qwen36-35b-a3b/`. A checkout outside `models/` is left in place. Another model's directory and row stay.
