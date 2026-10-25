# Set up Wan 2.2 I2V

How to download, pack, and make a clip with `wan22-i2v`. The kernel contract is [README.md](README.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B). The text encoder is [google/umt5-xxl](https://huggingface.co/google/umt5-xxl). Those weights are under the Apache License, Version 2.0.

## What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the download and, after packing, the packed record beside it

The five downloaded files are 117.5 GiB together. `setup` stops when the volume cannot hold the bytes still missing. The download stays on disk after packing.

8 GB is not enough to run a model capable of real-world agentic workflows.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

The package is not published to PyPI. Later commands use `uv run favlo` from that checkout.

## Download

```bash
uv run favlo --wan22-i2v setup
```

That command downloads with plain HTTP. A public file does not need a token. Sixteen connections write one file. A short file continues from the ranges already on disk.

The download lands in three places:

- `models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth`
- `models/google/umt5-xxl/`, the umT5 encoder and `tokenizer.json`
- the two expert files, `high.safetensors` and `low.safetensors`

`setup` prints the VAE directory, the umT5 directory, the expert directory, and those two files. It does not pack a record and does not register one.

`--umt5` downloads a different text encoder. The value is `org/name` or a `https://huggingface.co/org/name` link. That repository must publish `models_t5_umt5-xxl-enc-bf16.pth` and `tokenizer.json`. Those two files are stored under `models/<org>/<name>/`. The VAE and the two experts stay the default download. Their published sizes replace the default text-encoder sizes in the disk check.

```bash
uv run favlo --wan22-i2v setup --umt5 example-org/example-model
```

## Pack the experts

`quantize --int8` reads the two expert files `setup` printed. It writes `models/Wan2.2-I2V-A14B-int8` and registers that folder. The registry row stores that directory. The client model field is `Wan2.2-I2V-A14B-int8`. Every DiT linear is symmetric int8 with one float32 scale per output channel. The head weight and its bias are float16. Biases, QK RMSNorm, `norm3`, modulation, and the patch convolution stay float16. The text encoder is the shared int8 pack `models/google/umt5-xxl-int8`.

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

## Pack the text encoder

```bash
uv run favlo --wan22-i2v quantize-text
```

That command reads `models/google/umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth` and writes `models/google/umt5-xxl-int8`. Every linear is symmetric int8. The token embedding, the RMSNorm weights, and the relative buckets stay float16. The registry row stores that directory, and another pipeline can use the same folder. Without this pack, the text context is zeros.

`--name` writes `models/<name>/umt5` and stores that directory on the row `quantize` registered. `--out` is the quantized umT5 directory, and `--name` is the registry id it belongs to. `--umt5` reads the encoder downloaded for that repository. `quantize` registers the row before `quantize-text` records the umT5 directory.

```bash
uv run favlo --wan22-i2v quantize-text --name example-model --umt5 example-org/example-model
```

```bash
uv run favlo --wan22-i2v quantize-text --name example-model --out models/example-umt5
```

That stores `models/example-umt5` on the `example-model` row.

## Make a clip

Start the listener:

```bash
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. The model id is `Wan2.2-I2V-A14B-int8`. A video client calls `generate_video`. The still is an image id from `generate_image`, or a PNG block. A filesystem path is refused. The default size is 576×320. The default length is 81 frames, which is 5 seconds at 16 fps. A frame count is `4n+1`. The shortest moving clip is 5 frames. The default step count is 4.

From the checkout, without the listener:

```bash
uv run favlo --wan22-i2v generate --record models/Wan2.2-I2V-A14B-int8 --image still.png --prompt "The cube slides to the right. Fixed camera." --frames 5 --width 576 --height 320 --out clip.avi
```

`--image` is a PNG. A still that is not the clip size is resized to that size. The command writes an uncompressed AVI.

## If download stops

Run `setup` again. Finished ranges stay on disk, and the next run continues from the bytes still missing. A line that names a GiB count is free space.
