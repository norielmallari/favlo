# FLUX.2 [klein] 4B

As-built detail for the `klein_4b` package. [README.md](README.md) is the summary.

Apple Silicon loads `pipelines/klein_4b/metal41/`. Windows loads `pipelines/klein_4b/cuda/`.

## Apple Silicon

The checkpoint is [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), Apache-2.0. The package is `pipelines/klein_4b/metal41/`. It subclasses `MediaPipeline` with `kind` `image` and publishes `generate_image` and `edit_image`. It does not import another pipeline.

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

## Windows

Windows loads `pipelines/klein_4b/cuda`. The public id is `klein_4b`. `favlo --klein_4b` runs `setup`, `quantize`, and `kernels`. The registry folder is `FLUX.2-klein-4B-int4`, alias `klein_4b`, runtime `cuda`.

The checkpoint is `black-forest-labs/FLUX.2-klein-4B`. Quantize calls `favlo.quantize.quantize_int4` and leaves a finished manifest in place. Format string `flux2-klein-4b-int4`.

## Record

Rank-2 weights and convolutions flattened to `[N, K]` are symmetric int4. Codes are −7..7. The low nibble is the even column. One fp32 scale per output row. `N` and stored `K` pad to a multiple of 128. Code bytes are `stored_N × stored_K / 2`. Vectors are fp16, except BatchNorm `running_mean` and `running_var`, which stay fp32. A bias is stored only when the checkpoint has one. `tokenizer.json` is copied. Text layers 27–35, `model.norm`, and `lm_head` are not packed.

| Stage | Bytes |
|---|---:|
| Text, 27 layers | 1,561,187,328 |
| DiT | 1,941,851,648 |
| VAE | 42,486,342 |

Text hidden size is 2560, 32 query heads, 8 key and value heads, head dim 128, intermediate 9728, vocab 151936. The DiT width is 3072, with 5 double blocks and 20 single blocks. The VAE is `AutoencoderKLFlux2` with `block_out_channels` `(128, 256, 512, 512)`.

## Where the bytes live

One stage is resident. The text tower is dropped after the caption, the encoder after the references, and the DiT after the last step. The decoder is dropped after the still is read. The caption, 512×7680 fp16, stays with the DiT.

The open uses `memory.budget` through `open_gate` and `host.device_budget` for the device. Usable device memory is the driver's free count minus the 256 MiB hold-back. The named canvas is kept. The open refuses when the resident stage does not fit. Denoise counts the DiT, the joint residual, a second copy of the single-block projection, and the caption. Decode counts the VAE plus the input volume and the output volume.

## Kernels

`engine.cu` is the step. An int4 weight row is widened to fp16 and the product is cuBLAS, with fp32 accumulation and fp16 store. Activations are not quantized. A fp16 store saturates at the finite range. A convolution gathers its window into its own kernel and does not store the patch. The decoder volume stays float32 through that gather. Attention is fp16. Head dim 128 keeps a query tile, a key tile, and that pair's score tile, and the products use the same fp16 MMA. Head dim 512 does not store the score. Head dim 128 is the text tower and the DiT. Head dim 512 is the VAE mid-block, one head. The decoder volume stays float32 through GroupNorm, the residual, and nearest-2. The initial latent is `np.random.default_rng(seed).standard_normal`, then fp16. An omitted seed on a direct call is 0. The listener still assigns a seed before parse when the body has none.

`kernels` compiles for sm_86 with the installed CUDA toolkit and checks one fp16 MMA, one int4 nibble, and the LayerNorm, SwiGLU, RoPE, GroupNorm, Euler, and attention kernels against the host formulas. Head dim 128 keeps a query tile, a key tile, and that pair's score tile. Head dim 512 does not store the score.

## Forward

The prompt is one user turn with thinking closed, padded or truncated to 512 with pad id 151643. Keys at and past the unpadded length stay outside the causal window. RMSNorm ε is 10⁻⁶. Text RoPE is rotate-half with θ `1e6`. The feed-forward is SwiGLU. Hidden states after layers 9, 18, and 27 are concatenated on the channel axis to 7680.

The scheduler timestep is `sigma × 1000`. The sinusoid is 256 wide, cosine then sine, period 10000. Two bias-free linears with SiLU between them make the conditioning vector. SiLU of that vector feeds the modulation linears. Image and text each have six slices of 3072. The single stream has three. Each triple is shift, scale, gate.

A block norm is an affine-free LayerNorm, then `(1 + scale) · norm + shift`. The residual is `x + gate · branch`. Five double blocks join the streams only for attention, text tokens then image tokens. Attention is full, scaled by `1/sqrt(128)`. Twenty single blocks run on the concatenated sequence. `to_qkv_mlp_proj` is 27648 wide. `to_out` maps 12288 back to 3072.

DiT RoPE has four axes of 32, θ 2000, adjacent-pair rotation. Text sits at `(0, 0, 0, token)`. A target latent sits at `(0, row, col, 0)`. Reference `i` uses `T = 10 + 10·i`. References are in front of the target. The velocity is the target tail. Each step runs the references again.

For `N` steps the base sigmas are `linspace(1, 1/N, N)`. Above 4300 target tokens, μ is `0.00016927 · tokens + 0.45666666`. Otherwise μ blends the 10-step line `8.73809524e-05 · tokens + 1.89833333` and the 200-step line at this `N`. The map is `exp(μ) / (exp(μ) + 1/σ − 1)`. A trailing 0 is the last sigma. The update is `x + (σ_next − σ) · v`. Token count for μ is the target grid only. The default `N` is 4.

The VAE compresses 8×. Three stride-2 downsamples each add one zero on the bottom and the right, then a 3×3 with no further padding. The first 32 quant channels are the mean. A 2×2 pack makes 128 channels. BatchNorm has no affine term and ε `10⁻⁴`. Decode inverts that. Upsample is nearest-2 and a 3×3. GroupNorm uses 32 groups and ε `10⁻⁶`, with SiLU on the resnet norms and the outer conv norm.

Pixels are float CHW in `[-1, 1]`. uint8 is `rint((clip(x, −1, 1) + 1) · 127.5)`.

`size` is `WxH`, both edges multiples of 16, from 16 to 2048. An omitted text-to-image size is 1024×1024. A reference whose long side is above 1024 is shrunk so that side is 1024, nearest, then both edges snap down to a multiple of 16. A shorter reference is only center-cropped. An omitted edit size is the fitted first reference. `image` is one still or a list of one to four.

Cancel is read after the caption, after each Euler step once the latent is on the host, and before the decode. The hold is the phase, the next step, the float16 target latent, the grid, and the canvas. A decode hold skips denoising. Device buffers are not in the hold. A resume encodes the caption and the references again.

## Registry

`quantize` calls `register_quantized` with folder `FLUX.2-klein-4B-int4`, alias `klein_4b`, pipeline `klein_4b`, and runtime `cuda`. The sampling card is `generate`, temperature 0. `generate_image` and `edit_image` are the MCP tools. The handler is `handle_image`.
