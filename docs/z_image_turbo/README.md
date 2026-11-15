# Z-Image-Turbo

Eight-step text-to-image pipeline for [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). Those weights are under the Apache License, Version 2.0. `favlo --z_image_turbo` loads `pipelines/z_image_turbo/metal41/z-image-turbo.py`. With `favlo serve` listening, clients draw through `favlo media --pipeline z_image_turbo` or the published `z_image_turbo__generate_image` MCP tool.

- [ARCHITECTURE.md](ARCHITECTURE.md) is the as-built record: packed weights, registry, listener contract, forward path, smoke, and citations.
- [ceilings.md](ceilings.md) lists kernel and bandwidth ceilings.
- [research/generation-ceiling.md](research/generation-ceiling.md) records a measured generation sum.

- The default image is 1024×1024.
- Width and height are multiples of 16 up to 2560.
- `--steps` must be 8.

```text
uv run favlo --z_image_turbo kernels
```

```text
uv run pytest tests/test_z_image_turbo_smoke.py -s
ZIMAGE_SMOKE=1 uv run pytest tests/test_z_image_turbo_smoke.py -s
python -m harness.image --pipeline z_image_turbo --tool generate_image --model Z-Image-Turbo-int8
python -m harness.image_smoke_agent --pipeline z_image_turbo --tool generate_image --model Z-Image-Turbo-int8
```

## Setup

How to download, pack, and draw with `z_image_turbo`. The kernel contract is [ARCHITECTURE.md](ARCHITECTURE.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). Those weights are under the Apache License, Version 2.0.

### What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the checkout and, after packing, the packed record beside it

The seven weight files in the download are 30.6 GiB. The checkout stays on disk after `quantize`. The tokenizer is read from that checkout.

8 GB is not enough to run a model capable of real-world agentic workflows.

### Install

From a checkout of this repository:

```text
uv sync --extra dev
```

This checkout is release 0.6.0. PyPI publishes the runtime under `src/favlo`. The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

### Download

```text
uv run favlo --z_image_turbo setup
```

That command downloads [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) into `models/Z-Image-Turbo` with plain HTTP. A public checkout does not need a token. A file that already has its published size is left in place. A short file continues on the next run.

`setup` does not pack the record and does not register it.

### Pack

```text
uv run favlo --z_image_turbo quantize --int8
```

That command reads `models/Z-Image-Turbo` and registers the folder `Z-Image-Turbo-int8` in `models/registry.json`.

- The row stores `models/Z-Image-Turbo-int8`.
- The client model field is `Z-Image-Turbo-int8`.
- When the registry image default is empty, that registration stores `Z-Image-Turbo-int8` as the default.
- A default that is already set stays.
- A later pack of another folder does not replace it.
- It copies `tokenizer/` into that folder. Generation reads that copy.
- A finished directory has `manifest.json` with `complete: true`.
- Running `quantize --int8` again on that directory returns it.
- An unfinished directory is replaced.

On an M5 Pro with 48 GB a 1280 generation of the smoke prompt from that record printed 51.74 s, caption 0.15 s, decoder 9.27 s, steps 5.24–5.37 s. A 1024 generation printed 26.01 s, caption 0.16 s, decoder 4.63 s, steps 2.65–2.66 s. A 512 generation printed 4.43 s, with each step at 0.44 s.

- `quantize` without `--int8` writes `models/Z-Image-Turbo-fp4`.
- It does not read the int8 record.
- The fp4 directory is not the served record.
- Run the int8 pack first.
- A fp4 pack fills the image default only when that key is still empty.

```text
uv run favlo --z_image_turbo quantize
```

### Another safetensors file

`--weights` is one `.safetensors` file, or a directory of those files. With no `--name` and no `--out`, the packed folder is that file name without `.safetensors`, or that directory name, plus `-int8` when `--int8` is set, or `-fp4` otherwise, under `models/`. A name that already ends in that suffix is kept. The folder is the registry id.

```text
uv run favlo --z_image_turbo quantize --int8 --weights models/example-model.safetensors
```

That registers `example-model-int8`. The client model field is `example-model-int8`.

```text
uv run favlo --z_image_turbo quantize --weights models/example-model.safetensors
```

- `--name` replaces that folder.
- The name is used as written, and it is the registry id.

```text
uv run favlo --z_image_turbo quantize --weights models/example-model.safetensors --name example-model
```

That registers `example-model`. The official folder `Z-Image-Turbo-int8` stays on the official pack.

- A weights-only file copies the tokenizer from `models/Z-Image-Turbo` when that checkout contains `tokenizer/tokenizer.json`.
- Otherwise pass `--tokenizer-source` pointing at a tree that contains `tokenizer/`.
- The copy is written into the packed record.
- `--out` names the packed directory.
- `--name` and `--out` are not used together.
- A folder name is one token of letters, digits, dots, underscores, or hyphens.
- A file name that is not such a token needs `--name`.

### Draw a picture

Start the listener:

```text
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. An image client calls `generate_image`.

- Width and height are multiples of 16, from 16 to 2560.
- The step count is 8.
- `--model` is optional.
- An omitted `--model` uses the image default.
- After the int8 pack above, on an empty default, that id is `Z-Image-Turbo-int8`.

```text
uv run favlo media --pipeline z_image_turbo prompt="a red cube on a white table" size=1024x1024 seed=0
uv run favlo media --pipeline z_image_turbo prompt="a red cube on a white table" size=1024x1024 seed=0 quality=high
uv run favlo media --pipeline z_image_turbo --model Z-Image-Turbo-int8 --out pack.json prompt="a red cube on a white table"
```

- `favlo media` does not start the listener.
- It exits when that port is closed.
- Stdout is one `file://` URI.
- This is the command that draws. The pipeline command does not.

The default image is 1024×1024. `--seed` selects the initial noise. An omitted `--model` uses the image default, which the int8 pack sets when that key is empty.

### If download stops

Run `setup` again. Files that already have their published size stay on disk, and a short file continues from the bytes already written.