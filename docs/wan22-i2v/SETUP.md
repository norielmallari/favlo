# Set up Wan 2.2 I2V

How to download, pack, and make a clip with `wan22-i2v`. The kernel contract is [README.md](README.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B). The text encoder is [google/umt5-xxl](https://huggingface.co/google/umt5-xxl), packed by [../umt5-xxl/SETUP.md](../umt5-xxl/SETUP.md). Those weights are under the Apache License, Version 2.0.

## What you need

- Apple Silicon
- Metal 4.1
- 48 GB of unified memory for the listed resolutions at 81 frames. A machine with 16 GB cannot hold one resident int8 expert. A machine with 24 GB covers short or 320p clips only when the ledger still fits `expert_bytes` plus `forward_bytes`
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the download and, after packing, the packed record beside it

The VAE and the two experts are about 106 GiB together. `setup` stops when the volume cannot hold the bytes still missing. The download stays on disk after packing.

8 GB is not enough to run a model capable of real-world agentic workflows.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

Release 0.3.0 is on PyPI as the runtime under `src/favlo`. The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

## Pack the text encoder

```bash
uv run favlo --umt5-xxl setup
uv run favlo --umt5-xxl quantize
```

That writes `models/google/umt5-xxl-int8` with the tokenizer inside the pack and removes the BF16 checkout. A clip reads that directory. The steps are [../umt5-xxl/SETUP.md](../umt5-xxl/SETUP.md).

## Download the VAE and the experts

```bash
uv run favlo --wan22-i2v setup
```

That command downloads with plain HTTP. A public file does not need a token. Sixteen connections write one file. A short file continues from the ranges already on disk.

The download lands in two places:

- `models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth`
- the two expert files, `high.safetensors` and `low.safetensors`

`setup` prints the VAE directory, the expert directory, and those two files. It does not pack a record and does not register one. It does not download umT5.

## Pack the experts

`quantize --int8` reads the two expert files `setup` printed. It writes `models/Wan2.2-I2V-A14B-int8` and registers that folder. The registry row stores that directory and points `parts.umt5` at the shared pack. The client model field is `Wan2.2-I2V-A14B-int8`. Every DiT linear is symmetric int8 with one float32 scale per output channel. The head weight and its bias are float16. Biases, QK RMSNorm, `norm3`, modulation, and the patch convolution stay float16. The text encoder is the shared int8 pack `models/google/umt5-xxl-int8`.

```bash
uv run favlo --wan22-i2v quantize --int8 --high models/example-model/high.safetensors --low models/example-model/low.safetensors
```

`quantize --fp8-percent 50` writes `models/Wan2.2-I2V-A14B-fp8-e50`. Twenty of the forty blocks in each expert are stored as fp8. The rest of those blocks stay fp4. Text and time stay fp4. The head weight and its bias are float16. Biases, QK RMSNorm, `norm3`, modulation, and the patch convolution stay float16.

Replace the two paths below with the high-noise file and the low-noise file `setup` printed.

```bash
uv run favlo --wan22-i2v quantize --high models/example-model/high.safetensors --low models/example-model/low.safetensors
```

`--name` is the folder under `models/` and the registry id. With no `--name` and no `--out`, `--fp8-percent 50` writes `Wan2.2-I2V-A14B-fp8-e50`. The default percent is 20, which writes `Wan2.2-I2V-A14B-fp4`. `--name` and `--out` are not used together. `--high-dir` and `--low-dir` are the quantized expert directories. With no flag, those directories are `<record>/high` and `<record>/low`. The row stores the record directory and those two directories. `--config` is a JSON file of the checkpoint config. With no `--config`, the published Wan 2.2 I2V config is used.

```bash
uv run favlo --wan22-i2v quantize --high models/example-model/high.safetensors --low models/example-model/low.safetensors --name example-model
```

That registers `example-model`. The client model field is `example-model`. The row's directory is `models/example-model`.

```bash
uv run favlo --wan22-i2v quantize --high models/example-model/high.safetensors --low models/example-model/low.safetensors --name example-model --high-dir models/example-high --low-dir models/example-low
```

That stores `models/example-high` and `models/example-low` on the `example-model` row.

## Make a clip

Start the listener:

```bash
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. The model id is `Wan2.2-I2V-A14B-int8`. A video client calls `wan22_i2v__generate_video`. The still is an image id from `generate_image`, a PNG block, or a `file://` URI. `size` is required. Landscape and portrait sizes for 240p, 320p, 480p, and 720p are accepted. An omitted duration is 81 frames. A frame count is `4n+1`. The shortest moving clip is 5 frames. The default step count is 4.

```bash
uv run favlo media --pipeline wan22-i2v prompt="The cube slides to the right. Fixed camera." image=file:///absolute/still.png size=576x320 frames=5
```

`favlo media` does not start the listener. Stdout is one `file://` URI. This is the command that writes the clip. The pipeline command does not. A still that is not the clip size is resized to that size.

## If download stops

Run `setup` again. Finished ranges stay on disk, and the next run continues from the bytes still missing. A line that names a GiB count is free space.
