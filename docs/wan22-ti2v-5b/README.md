# Wan2.2 TI2V-5B

Text-to-video and image-to-video on the Metal 4.1 GPU. The weights are the
5B transformer from the [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
card. The record is `models/Wan2.2-TI2V-5B-int8`. The umT5 encoder is the int8 pack named by `parts.umt5` on the registry
row, `models/google/umt5-xxl-int8`. A clip reads that pack and does not
open the BF16 encoder file. The tokenizer is `tokenizer.json` beside the
source download. This pipeline does not pack the encoder. The VAE is the
float32 `Wan2.2_VAE.pth` from that card.

`generate_video` takes an optional still. Omitting it samples from text.
Passing a PNG conditions the first latent frame. The clip is 24 fps. The
default body is 121 frames at `1280x704`. `frames` wins when `seconds` is
also set. `seconds` snaps to `4n+1`. Edges are multiples of 32. `steps` is
0 or 4. `guidance_scale` is omitted or 1. `negative_prompt` is accepted and
not encoded.

The sampler uses four knots, 1000, 960, 888.888889, and 727.272727. Each knot is
one forward. The update reads `x0 = x_t - sigma * v`. A remaining knot mixes
that prediction with fresh noise, `x = (1 - sigma_next) * x0 + sigma_next * eps`.
The last knot decodes `x0`.

Self-attention, cross-attention, and the feed-forward are symmetric int8,
one float scale per output row. `quantize` prints the relative RMS of each
matrix. The worst matrix was `blocks.11.ffn.2` at 0.014504. The feed-forward
activation is tanh-GELU (tanh versus erf max abs 0.000473). Time,
text, the patch, the head, norms, and modulation stay float16. The VAE stays
float32. It is the 48-channel Wan2.2 autoencoder. Pixels are grouped 2×2
into 12 channels with the width offset outside the height offset, then
restored the same way. The VAE walks one latent frame at a time and keeps
the causal cache. A wide convolution is float16 im2col plus `matmul2d`,
4096 output rows at a time, and those two kernels share one command buffer.
A narrow one is float `conv3d`. An int8 product and its scale share one
command buffer. Attention submits every head before each wait. The prompt's
cross-attention key and value stay for the four knots. Weights already on
the device stay until close.

The decoder's resident frame tile shrinks until one latent frame fits. A
request keeps the resolution, frame count, and duration it named. Open
refuses when that minimum does not fit. The 30 DiT blocks and the 24 text
blocks stay resident until close.

## Commands

```text
uv run favlo --wan22-ti2v-5b setup
uv run favlo --wan22-ti2v-5b quantize --weights <checkpoint>
uv run favlo --wan22-ti2v-5b kernels
uv run favlo --wan22-ti2v-5b ceilings
uv run favlo --wan22-ti2v-5b smoke
```

`smoke` writes an 832×480 text clip and an 832×480 clip from `still.png`,
nine frames each, then deletes the DiT checkpoint it packed. The VAE file
stays. `still.png` is the horse still. A still that is not the clip size is
resized to that size, and that frame is the first latent. A canvas smaller
than 832×480 is not the smoke. [MLX-Gen](https://github.com/lpalbou/mlx-gen/blob/main/docs/wan-video.md)
already treats that smaller canvas as a routing check. This note does not
add a score for the cutoff.

Setup and the first clip are in [SETUP.md](SETUP.md).

## Papers

The transformer and the causal VAE are Wan, [arXiv:2503.20314](https://arxiv.org/abs/2503.20314).
Channel mean and standard deviation are the Diffusers VAE config on
[Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers).
Self-forcing, [arXiv:2506.08009](https://arxiv.org/abs/2506.08009), and
distribution matching, [arXiv:2311.18828](https://arxiv.org/abs/2311.18828),
describe how a four-step checkpoint is trained. The served loop stays the
bidirectional update above. `matmul2d` is Metal Performance Primitives,
described in WWDC26 session 330.
