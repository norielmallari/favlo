# Wan2.2 TI2V-5B setup

`setup` downloads `Wan2.2_VAE.pth` from
[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
into `models/Wan-AI/Wan2.2-TI2V-5B/`. Quantize reads a local DiT checkpoint.
It does not download the VAE again, and it does not pack the VAE. The shared
umT5 pack is a separate pipeline.

```text
uv run favlo --umt5_xxl setup
uv run favlo --umt5_xxl quantize
uv run favlo --wan22_ti2v_5b setup
uv run favlo --wan22_ti2v_5b quantize --weights <checkpoint>
uv run favlo --wan22_ti2v_5b kernels
uv run favlo --wan22_ti2v_5b smoke
```

`favlo --umt5_xxl quantize` writes `models/google/umt5-xxl-int8` with the
tokenizer inside that pack and removes the BF16 checkout. DiT `quantize`
writes `models/Wan2.2-TI2V-5B-int8`, prints the relative RMS of each int8
matrix, and points `parts.umt5` at that shared pack. Smoke opens that part. Those commands do not pack the encoder, and they do not open the
BF16 file. `kernels` compiles the shader and checks fp4, fp8, int8, a narrow
convolution, and a wide convolution that crosses the 4096-row tile.
`smoke` serves both 832×480 clips, nine frames each. The still is resized
to the clip. After both exist, it deletes the DiT
checkpoint named in the record and leaves `Wan2.2_VAE.pth` in place.

A clip from the checkout uses `favlo media`. The pipeline command does not draw.

```text
uv run favlo media --pipeline wan22-ti2v-5b prompt="A horse gallops at speed, its raised foreleg driving down and the trailing legs swinging forward into the next stride, mane and tail whipping. Fixed camera." image=file:///absolute/still.png size=832x480 frames=9
```

Serve the registered id `Wan2.2-TI2V-5B-int8` or the alias `wan22_ti2v_5b`.
The base card is Apache-2.0. The source checkpoint's license is unset.
Confirm those terms before redistributing the packed record.
