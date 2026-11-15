# Z-Image shared forward

This note is the as-built detail for this package. [README.md](README.md) is the summary.

Shared Metal forward for Z-Image-Turbo and Z-Image. The importable package is `pipelines/z_image/metal41/` (`import z_image.metal41`). It has no typed `pipeline` export. The command scan does not list it.

The package holds:

- the int8 and fp4 pack
- the Metal shaders in `shaders/kernels.metal`
- the Qwen3 caption encoder
- the S3-DiT velocity
- the Flux VAE decoder
- the flow-matching knots
- the host image hold for preempt
- the resumable Hugging Face file download

A caller passes:

- the step count
- the shift
- the format id
- the source repository

Served pipelines import this package. `z_image_turbo` re-exports many modules under its own runtime. `z_image_base` imports `z_image.metal41` directly for the shared pieces and keeps its own schedule and guidance.

## Pack

- `quantize` packs generate-path weights. The VAE encoder is left unpacked.
- `kind` is `int8` or `fp4`.
- Int8 stores one float32 scale per output row and pads that row count to 128.
- The activation is quantized in groups of 128 along K.
- Stored K is the logical K rounded up to a multiple of 256.
- fp4 is E2M1 with one E8M0 scale per 32 values along K.
- Biases, norms, and the two pad tokens stay float16.
- The caller supplies `format_id` and `source_repo`. When those are omitted, the pack defaults match the Turbo line.

## Schedule

- The scheduler is FlowMatchEulerDiscreteScheduler.
- The shift is `s·u / (1 + (s − 1)·u)`.
- Sigma 1 is noise and sigma 0 is the image.
- The embedded time is `(1 − sigma) · 1000`.
- Library defaults in this package are eight steps and shift 3.
- A caller may pass another positive step count and shift.

## Forward pieces

- Caption tokens come from the Qwen3-4B chat template. The vector that enters `cap_embedder` is the penultimate layer, before `model.norm`.
- Caption refine has no AdaLN. Image refine uses AdaLN.
- The main DiT stack sees image tokens, then caption tokens.
- RoPE puts the caption index on the temporal axis and the patch coordinate on the two spatial axes.
- The Euler update and the VAE decode stay on device shaders.
- The tokenizer and the PNG container stay on the host.
- `ImageHold` keeps host values for a paused job. Device buffers do not survive `close`.

Scratch accepts edges that are multiples of 16 up to 2560. The publisher's last training stage keeps each edge within 256 pixels of the 1024 grid, from 768 to 1280.

## Registry

- This package has no registry row.
- `z_image_turbo` and `z_image_base` each store their own packed folder.
- The fields those rows read are on [the Turbo note](../z_image_turbo/ARCHITECTURE.md#registry) and [the base note](../z_image_base/ARCHITECTURE.md#registry).

## Citations

Cai, Zheng, et al. *Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer.* arXiv:2511.22699.

Qwen Team. *Qwen3 Technical Report.* The caption encoder is the 36-layer Qwen3 configuration shipped with the official checkpoints.

Peebles and Xie. *Scalable Diffusion Models with Transformers.* AdaLN-Zero.

Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.*

Lipman et al. *Flow Matching for Generative Modeling.*

Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* The decoder byte map.

Open Compute Project. *OCP Microscaling Formats (MX) Specification.* fp4 E2M1 and E8M0.

Apple. *Metal 4 tensor operations,* WWDC26 session 330. `matmul2d`.
