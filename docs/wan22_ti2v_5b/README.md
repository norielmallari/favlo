# Wan2.2 TI2V-5B

Text-to-video and image-to-video on the Metal 4.1 GPU.

The weights are the 5B transformer from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

The packed record is `models/Wan2.2-TI2V-5B-int8`.

`wan22_ti2v_5b.portable` is the serve entry. `favlo --wan22_ti2v_5b` and `favlo serve` import that package. That entry opens `metal41` on Apple Silicon and `cuda` on Windows. The CUDA generate path raises until the int4 DiT forward is packed.

With `favlo serve` listening, `favlo media --pipeline wan22_ti2v_5b` posts a clip job.

MCP tools `wan22_ti2v_5b__generate_video_t2v` and `wan22_ti2v_5b__generate_video_i2v` are the same path for clients.

As-built detail: [ARCHITECTURE.md](ARCHITECTURE.md).

```text
uv run favlo --wan22_ti2v_5b umt5 setup
uv run favlo --wan22_ti2v_5b umt5 quantize --format wan
uv run favlo --wan22_ti2v_5b setup
uv run favlo --wan22_ti2v_5b quantize --weights <checkpoint>
uv run favlo --wan22_ti2v_5b kernels
uv run favlo --wan22_ti2v_5b ceilings
```

```text
uv run favlo media --pipeline wan22_ti2v_5b prompt="A horse gallops at speed. Fixed camera." size=832x480 frames=9
uv run favlo media --pipeline wan22_ti2v_5b prompt="A horse gallops at speed. Fixed camera." image=file:///absolute/still.png size=832x480 frames=9
```

- [../umt5_xxl/README.md](../umt5_xxl/README.md) — shared umT5 pack
- [ceilings.md](ceilings.md) — kernel ceilings for a 9-frame 576×320 grid
- [research/int8-weight-error.md](research/int8-weight-error.md) — int8 weight error
- [research/clip-ceiling.md](research/clip-ceiling.md) — isolated DiT sum and printed decode lines

The base card is Apache-2.0. The source checkpoint's license is unset. Confirm those terms before redistributing the packed record.

## Setup

How to download, pack, and make a clip with `wan22_ti2v_5b`.

- The kernel contract is [ARCHITECTURE.md](ARCHITECTURE.md).
- The listener is [../serve.md](../serve.md).
- Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

The text encoder is [google/umt5-xxl](https://huggingface.co/google/umt5-xxl).

It is packed by this pipeline’s `umt5` commands ([../umt5_xxl/README.md](../umt5_xxl/README.md)).

### What you need

- Apple Silicon
- Metal 4.1
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the VAE download and, after packing, the packed record
- A local DiT checkpoint for `quantize --weights`

Open refuses when three decoder frames do not fit. The ledger detail is [ARCHITECTURE.md](ARCHITECTURE.md).

8 GB is not enough to run a model capable of real-world agentic workflows.

### Install

From a checkout of this repository:

```text
uv sync --extra dev
```

This checkout is release 0.6.0.

PyPI publishes the runtime under `src/favlo`.

The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

### Pack the text encoder

```text
uv run favlo --wan22_ti2v_5b umt5 setup
uv run favlo --wan22_ti2v_5b umt5 quantize --format wan
```

That writes `models/google/umt5-xxl-int8` with the tokenizer inside the pack and removes the BF16 checkout.

A clip reads that directory.

`favlo --wan22_i2v umt5` writes the same pack.

### Download the VAE

```text
uv run favlo --wan22_ti2v_5b setup
```

That command downloads `Wan2.2_VAE.pth` from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) into `models/Wan-AI/Wan2.2-TI2V-5B/`.

- It does not download the DiT.
- It does not pack a record.
- It does not register one.
- It does not download umT5.

`--student-dir` optionally downloads the Turbo student `model.pt` into that directory.

### Pack the DiT

`quantize` reads a local DiT checkpoint.

It writes `models/Wan2.2-TI2V-5B-int8` and registers that folder.

The client model field is `Wan2.2-TI2V-5B-int8`.

It does not pack the VAE.

```text
uv run favlo --wan22_ti2v_5b quantize --weights <checkpoint>
```

### Make a clip

Start the listener:

```text
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`.

The model id is `Wan2.2-TI2V-5B-int8`.

A video client calls `wan22_ti2v_5b__generate_video_t2v` or `wan22_ti2v_5b__generate_video_i2v`.

- `prompt` and `size` are required.
- Edges are multiples of 32.
- An omitted `image` is text-to-video.
- A provided `image` is image-to-video.
- The clip is 24 fps.
- `frames` is 4n+1 and at least 5. `duration` wins when both are set.
- An omitted duration and frame count is 121 frames.
- `steps` is 0 or 4.
- `guidance_scale` is omitted or 1.

```text
uv run favlo media --pipeline wan22_ti2v_5b prompt="A horse gallops at speed. Fixed camera." size=832x480 frames=9
uv run favlo media --pipeline wan22_ti2v_5b prompt="A horse gallops at speed. Fixed camera." image=file:///absolute/still.png size=832x480 frames=9
```

- `favlo media` does not start the listener.
- Stdout is one `file://` URI.
- This is the command that writes the clip.
- The pipeline command does not.

A still that is not the clip size is resized to that size.

### If download stops

Run `setup` again.

Finished ranges stay on disk, and the next run continues from the bytes still missing.
