# Z-Image-Turbo

Eight-step text-to-image. An optional source PNG is image-to-image: the VAE encodes it, `strength` noises that latent, and the remaining Euler steps denoise it.

The official checkpoint is [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).

Those weights are under the Apache License, Version 2.0.

`z_image_turbo.portable` is the serve entry. `metal41` is the Metal device that entry opens. `favlo serve` and `favlo --z_image_turbo` import `portable`. Setup, quantize, and kernels run from that entry and open the device on Apple Silicon.

Start `favlo serve`.

Then call `z_image_turbo__generate_image` on the listener, or use `favlo media --pipeline z_image_turbo`.

The registry model id is `Z-Image-Turbo-int8`.

```text
uv run favlo --z_image_turbo kernels
```

```text
uv run favlo media --pipeline z_image_turbo prompt="a red cube on a white table" size=1024x1024 seed=0
uv run favlo media --pipeline z_image_turbo prompt="a real tan horse galloping left" image=file:///absolute/still.png strength=0.75
```

Kernels, packed layout, registry, listener fields, and the forward path are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

How to download, pack, and draw with `z_image_turbo`.

- The kernel contract is [ARCHITECTURE.md](ARCHITECTURE.md).
- The listener is [../serve.md](../serve.md).
- Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).

Those weights are under the Apache License, Version 2.0.

### What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the checkout and, after packing, the packed record beside it

The seven weight files in the download are 30.6 GiB.

The checkout stays on disk after `quantize`.

The tokenizer is read from that checkout.

8 GB is not enough to run a model capable of real-world agentic workflows.

### Install

From a checkout of this repository:

```text
uv sync --extra dev
```

This checkout is release 0.6.0.

PyPI publishes the runtime under `src/favlo`.

The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

### Download

```text
uv run favlo --z_image_turbo setup
```

That command downloads [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) into `models/Z-Image-Turbo` with plain HTTP.

- A public checkout does not need a token.
- A file that already has its published size is left in place.
- A short file continues on the next run.

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
- Running `quantize` again on that directory packs any missing VAE encoder tensors and leaves the other planes in place.
- An unfinished directory is replaced.

`quantize` without `--int8` writes `models/Z-Image-Turbo-fp4`.

- It does not read the int8 record.
- The fp4 directory is not the served record.
- Run the int8 pack first.
- A fp4 pack fills the image default only when that key is still empty.

```text
uv run favlo --z_image_turbo quantize
```

### Another safetensors file

`--weights` is one `.safetensors` file, or a directory of those files.

With no `--name` and no `--out`, the packed folder is that file name without `.safetensors`, or that directory name, plus `-int8` when `--int8` is set, or `-fp4` otherwise, under `models/`.

A name that already ends in that suffix is kept.

The folder is the registry id.

```text
uv run favlo --z_image_turbo quantize --int8 --weights models/example-model.safetensors
```

That registers `example-model-int8`.

The client model field is `example-model-int8`.

```text
uv run favlo --z_image_turbo quantize --weights models/example-model.safetensors
```

- `--name` replaces that folder.
- The name is used as written, and it is the registry id.

```text
uv run favlo --z_image_turbo quantize --weights models/example-model.safetensors --name example-model
```

That registers `example-model`.

The official folder `Z-Image-Turbo-int8` stays on the official pack.

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

The daemon listens on `http://127.0.0.1:8129`.

The model id is `Z-Image-Turbo-int8`.

An image client calls `generate_image`.

- Width and height are multiples of 16, from 16 to 2560.
- The step count is 8.
- `--model` is optional.
- An omitted `--model` uses the image default.
- After the int8 pack above, on an empty default, that id is `Z-Image-Turbo-int8`.

```text
uv run favlo media --pipeline z_image_turbo prompt="a red cube on a white table" size=1024x1024 seed=0
uv run favlo media --pipeline z_image_turbo prompt="a red cube on a white table" size=1024x1024 seed=0 quality=high
uv run favlo media --pipeline z_image_turbo --model Z-Image-Turbo-int8 --out pack.json prompt="a red cube on a white table"
uv run favlo media --pipeline z_image_turbo prompt="A real tan horse galloping left on a white field." image=file:///absolute/still.png strength=0.75
```

An optional `image` is an 8-bit RGB PNG. `strength` is from 0 to 1 and defaults to 0.6. An omitted `size` uses that image's edges.

`favlo media` does not start the listener.

It exits when that port is closed.

Stdout is one `file://` URI.

This is the command that draws.

The pipeline command does not.

The default image is 1024×1024.

`--seed` selects the initial noise.

Body fields, the MCP tool, cancel, and preempt are in [ARCHITECTURE.md](ARCHITECTURE.md).

### If download stops

Run `setup` again.

Files that already have their published size stay on disk, and a short file continues from the bytes already written.

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md) — as-built serving and inference detail
- [research/generation-ceiling.md](research/generation-ceiling.md) — measured generation sum
- [smoke.json](smoke.json) — image-agent prompt and latent hashes
