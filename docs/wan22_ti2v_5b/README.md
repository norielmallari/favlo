# Wan2.2 TI2V-5B

Text-to-video and image-to-video on the Metal 4.1 GPU. The weights are the
5B transformer from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).
The packed record is `models/Wan2.2-TI2V-5B-int8`. MCP tools
`wan22_ti2v_5b__generate_video_t2v` and `wan22_ti2v_5b__generate_video_i2v`
reach the pipeline through `favlo serve` once that row is registered.

## Setup

- `setup` downloads `Wan2.2_VAE.pth` from
  [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
  into `models/Wan-AI/Wan2.2-TI2V-5B/`.
- Quantize reads a local DiT checkpoint.
- It does not download the VAE again.
- It does not pack the VAE.
- The shared umT5 pack is owned by this pipeline’s `umt5` commands (same pack as I2V).

```text
uv run favlo --wan22_ti2v_5b umt5 setup
uv run favlo --wan22_ti2v_5b umt5 quantize --format wan
uv run favlo --wan22_ti2v_5b setup
uv run favlo --wan22_ti2v_5b quantize --weights <checkpoint>
uv run favlo --wan22_ti2v_5b kernels
uv run favlo --wan22_ti2v_5b ceilings
```

- `favlo --wan22_ti2v_5b umt5 quantize` writes `models/google/umt5-xxl-int8` and removes the BF16 checkout.
- DiT `quantize` writes `models/Wan2.2-TI2V-5B-int8`.

Clips come from `favlo media` only.

```text
uv run favlo media --pipeline wan22_ti2v_5b prompt="A horse gallops at speed. Fixed camera." size=832x480 frames=9
uv run favlo media --pipeline wan22_ti2v_5b prompt="A horse gallops at speed. Fixed camera." image=file:///absolute/still.png size=832x480 frames=9
uv run favlo media --pipeline wan22_ti2v_5b prompt="A horse gallops at speed, its raised foreleg driving down and the trailing legs swinging forward into the next stride, mane and tail whipping. Fixed camera." image=file:///absolute/still.png size=832x480 frames=9
```

The base card is Apache-2.0. The source checkpoint's license is unset.
Confirm those terms before redistributing the packed record.

As-built detail: [ARCHITECTURE.md](ARCHITECTURE.md).

- [research/int8-weight-error.md](research/int8-weight-error.md) — int8 weight error
- [research/clip-ceiling.md](research/clip-ceiling.md) — clip ceiling