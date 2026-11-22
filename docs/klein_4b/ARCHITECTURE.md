# FLUX.2 [klein] 4B

As-built detail for the `klein_4b` package. [README.md](README.md) is the summary.

The checkpoint is [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), Apache-2.0. The serve entry is `pipelines/klein_4b/portable/`. It subclasses `MediaPipeline` with `kind` `image` and publishes `generate_image` and `edit_image`. `metal41` is the device that entry opens. The shell does not import that device until `open_runtime`. It does not import another pipeline.

The DiT is the rectified-flow MMDiT in Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (2024), with the FLUX.2 changes in the Hugging Face model card and `transformer/config.json`: shared modulation, no linear bias, SwiGLU, and a parallel single block. The text tower is Qwen3-4B (Yang et al., *Qwen3 Technical Report*, 2025). The VAE is `AutoencoderKLFlux2` from `vae/config.json`.

`model_index.json` names `Flux2KleinPipeline` with `is_distilled` true. The card's call is 4 steps, guidance 1.0, and, for text-to-image, 1024×1024. The same checkpoint edits from one to four references. Guidance is baked in, so a step is one velocity and one Euler update.

## Forward

The prompt is one user turn, with the generation prompt and thinking off:

```text
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

That string is padded or truncated to 512. The encoder runs 27 of the 36 layers. Hidden states 9, 18, and 27 are concatenated on the channel axis to 7680. Layers 28–36 and `lm_head` are not packed. Q, K, and V have no bias. QK RMSNorm is per head. RoPE θ is `1e6`, rotate-half. Attention is causal. The feed-forward is SwiGLU.

The DiT width is 3072, with 24 heads of 128. Five double blocks keep an image stream and a text stream and join them only for attention, text tokens then image tokens. Twenty single blocks run on the concatenated sequence. One image modulation and one text modulation, each six slices of 3072, are shared by every double block. One modulation of three slices is shared by every single block. The order in each triple is shift, scale, gate. The norm inside a block is an affine-free LayerNorm, ε `1e-6`, then `(1 + scale) · norm + shift`. The residual is `x + gate · branch`.

A double-block feed-forward maps 3072 to 18432, applies SiLU to the first 9216, multiplies by the second 9216, and maps back. A single block's one projection emits QKV and that same SwiGLU input (`27648`). Attention and the gated MLP run side by side. One projection maps the concatenation (`12288`) back to 3072, and the single-block gate multiplies that result.

QK RMSNorm is a learned 128-vector. RoPE has four axes of 32, θ 2000, adjacent-pair rotation. Text sits at `(0, 0, 0, token)`. A target latent sits at `(0, row, col, 0)`. Reference `i` uses `T = 10 + 10·i`. There is no guidance embedding. The sinusoid is 256 wide with cosine before sine, on the scheduler timestep. Two bias-free linears, with SiLU between them, make the 3072-wide conditioning vector. The readout is an affine-free LayerNorm, then scale and shift from that vector, then a linear to 128.

The VAE compresses 8× in the conv stack and packs 2×2 of the 32-channel mean into 128 channels, so pixels are 16× the token grid. Each stride-2 downsample adds one zero on the bottom and the right, then convolves with no further padding. Each edge is a multiple of 16. After packing, BatchNorm with no affine term and ε `1e-4` whitens with the stored mean and variance. Decode inverts that. Mid-block attention is one head over the 512 channels. GroupNorm uses 32 groups and ε `1e-6`.

A text-to-image call has no reference tokens. The image sequence is the target latent, and the velocity is that whole sequence. An edit encodes each of one to four references, concatenates them in front of the target tokens, and leaves them out of the Euler update. The velocity tail is the target.

For `N` steps the base sigmas are `linspace(1, 1/N, N)`. μ is the publisher's empirical shift: above 4300 tokens, `0.00016927 · tokens + 0.45666666`; otherwise the blend of the 10-step and 200-step lines at `N`. The exponential map is `exp(μ) / (exp(μ) + 1/σ − 1)`. A trailing 0 is the last sigma. The update is `x + (σ_next − σ) · v`. The card's `N` is 4. Token count for μ is the target grid only.

## Plane

Rank-2 weights, and every convolution flattened to `[N, K]`, are symmetric int8. Codes are `−127..127`. `−128` is unused. One float32 scale per output row. A zero row stores scale 0. `N` and stored `K` pad to a multiple of 128. Token rows pad to 64.

The activation is quantized from the resident row in groups of 128 along K. A zero group stores scale 1 and zero codes. The text tower, the DiT, and the VAE encoder quantize a float16 row. The VAE decoder keeps its volume in float32 and quantizes that. A decoder convolution gathers the float volume into the int8 codes and does not store the patch. A frame whose product would pass 2^31 elements is split into row spans, because the int8 tensor index is 32-bit. `matmul2d` is int8 × int8 → int32, tile 64×128, `execution_simdgroups<4>`, `transpose_right` set. The epilogue multiplies by the group scale and the row scale in float32, adds a float16 bias when the layer has one, and the next consumer stores float16, or float32 for the decoder volume.

Attention is float32: QK RMSNorm, RoPE, the score, the softmax, and the mix. Head dim 128 uses a 32×32 query tile and a 32×32 key tile. The score tile stays in the threadgroup. The query-key product and the value mix are `matmul2d`. Head dim 512, the VAE mid-block, stays one threadgroup per query. The score is not stored. Head scale is `1/sqrt(head dim)`. Outside that attention math, the residual, the MLP, LayerNorm, modulation, the timestep vector, the latent, and the Euler update are float16. The decoder volume stays float32 through GroupNorm, the residual, and nearest-2. Those ops, and the float32 RoPE, are Metal kernels. The caption, the reference encode, each DiT step, and the decode stay on the device. One step is one command buffer. The host waits when that step finishes, reads the latent back, and then advances progress. The text tower is dropped after the caption, the encoder after the references, and the DiT after the last step. The decoder volume stays float32 until each int8 quantize. A convolution gathers that volume into the int8 product without storing the patch. Each decoder block then drops its weights and its pooled tensors. The decoded still is read at the end.

The packed folder is `models/FLUX.2-klein-4B-int8`. Serve reads each plane file in 8 MiB steps into the shared span the kernel binds. The files stay one plane each. A bias file is logical `n` float16 values. `setup` reads the Hugging Face snapshot. `quantize` writes that folder and registers `klein_4b`. `kernels` checks one int8 product, one float32 attention, and the elementwise kernels against the host formulas.

## Listener

`generate_image` takes `prompt` and optional `size`, `steps`, and `seed`. An omitted size is 1024×1024.

`edit_image` takes `prompt`, `image`, and optional `size`, `steps`, and `seed`. `image` is one still or a list of one to four (`file://`, image id, PNG, or a filename in `images`). A present `image` on either tool is an edit. A blank `image` is rejected.

`size` is `WxH` with both edges multiples of 16, from 16 to 2048. That size is the output canvas. An omitted edit size is the first reference after it is fitted. A reference whose long side is above 1024 is shrunk so that side is 1024, aspect kept, then both edges snap to a multiple of 16. Smaller references are only center-cropped to that grid. Each reference keeps its own token grid. The default is 4 steps. The image smoke reads `docs/klein_4b/smoke.json` and calls `klein_4b__generate_image`.

Cancel is read after the caption, each Euler step, and the decode. Preempt keeps the target latent and the next step index. A resume encodes the caption and the references again and continues the held latent.
