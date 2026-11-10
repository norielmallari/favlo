# Set up Z-Image

How to download, pack, and draw with `z_image_base`. The kernel contract is [README.md](README.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image). Those weights are under the Apache License, Version 2.0.

## What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the checkout and, after packing, the packed record beside it

The six weight files in the download are 19.1 GiB. The checkout stays on disk after `quantize`. The tokenizer is read from that checkout.

8 GB is not enough to run a model capable of real-world agentic workflows.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

Release 0.3.0 is on PyPI as the runtime under `src/favlo`. The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

## Download

```bash
uv run favlo --z_image_base setup
```

That command downloads [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) into `models/Z-Image` with plain HTTP. A public checkout does not need a token. A file that already has its published size is left in place. A short file continues on the next run.

`setup` does not pack the record and does not register it.

## Pack

The served pack is symmetric int8. Each matrix has one float32 scale per output row. `N` is padded to 128. Rank-1 weights and the two pad tokens stay float16. The activation is quantized in groups of 128 along K.

```bash
uv run favlo --z_image_base quantize
```

That command reads `models/Z-Image` and registers the folder `Z-Image-int8` in `models/registry.json`. The row stores `models/Z-Image-int8`. The client model field is `Z-Image-int8`. It copies `tokenizer/` into that folder. Generation reads that copy. A finished directory has `manifest.json` with `complete: true`. Running `quantize` again on that directory returns it. An unfinished directory is replaced.

The row records the sampler defaults `steps`, `guidance`, `shift`, `cfg_truncation`, and `cfg_normalization`. A new row receives 28, 4, 6, 1, and 0. A key already on the row is left as written. A request that omits one of those fields uses the row. A request field replaces that one value for that call.

`--fp4` writes `models/Z-Image-fp4`. That pack is fp4 E2M1. It does not read the int8 record. The fp4 directory is not the served record.

## Another safetensors file

`--weights` is one `.safetensors` file, or a directory of those files. With no `--name` and no `--out`, the packed folder is that file name without `.safetensors`, or that directory name, plus `-int8`, under `models/`. `--fp4` uses `-fp4` instead. A name that already ends in that suffix is kept. The folder is the registry id.

```bash
uv run favlo --z_image_base quantize --weights models/example-model.safetensors
```

That registers `example-model-int8`. The client model field is `example-model-int8`.

`--name` replaces that folder. The name is used as written, and it is the registry id.

```bash
uv run favlo --z_image_base quantize --weights models/example-model.safetensors --name example-model
```

That registers `example-model`. The official folder `Z-Image-int8` stays on the official pack.

A weights-only file copies the tokenizer from `models/Z-Image` when that checkout contains `tokenizer/tokenizer.json`. Otherwise pass `--tokenizer-source` pointing at a tree that contains `tokenizer/`. The copy is written into the packed record. `--out` names the packed directory. `--name` and `--out` are not used together. A folder name is one token of letters, digits, dots, underscores, or hyphens. A file name that is not such a token needs `--name`.

## Draw a picture

Start the listener:

```bash
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. The model id is `Z-Image-int8`. An image client calls `generate_image`, or:

```bash
uv run favlo media --pipeline z-image-base --model Z-Image-int8 prompt="a red cube on a white surface" size=1024x1024 seed=0
```

`favlo media` does not start the listener. Stdout is one `file://` URI. This is the command that draws. The pipeline command does not. The body fields, the MCP tool, cancel, and preempt are in [README.md](README.md). Width and height are multiples of 16, from 16 to 2560. An omitted sampler field uses the registry row.

The default image is 1024×1024. The seed selects the initial noise. The default record is `models/Z-Image-int8`.

## If download stops

Run `setup` again. Files that already have their published size stay on disk, and a short file continues from the bytes already written.
