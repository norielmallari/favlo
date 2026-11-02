# References

Gemma Team. *Gemma 4 Technical Report.* 2026. [arXiv:2607.02770](https://arxiv.org/abs/2607.02770). The 26B-A4B decoder, the sliding and global layers, the per-layer scalar, and the final logit soft cap. Section 2.5 names Q4_0 as blockwise quantization of 32 values. This record stores resident matrices as fp8 E4M3 and routed experts as fp4 or fp8. Algorithm 1 and Figure 2 resize an image to a multiple of the pooled patch and pool the encoder 3×3. Table 10 is the 550M width, MLP, heads, and depth. Table 12 evaluates this checkpoint at `N_max` = 280.

Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929). The 550M tower is this ViT: patch 16, pre-norm residuals, and scaled dot-product attention.

Heo, Park, Han, and Yun. *Rotary Position Embedding for Vision Transformer.* 2024. [arXiv:2403.13298](https://arxiv.org/abs/2403.13298). Axial 2D-RoPE. This tower uses θ = 100 from the vision config. The first half of each head rotates with the column (x) and the second half with the row (y).

mlx-lm. `gemma4_text.py`. The text layer adds the attention residual and both feed-forward residuals at scale one, then multiplies by `layer_scalar`. The dense MLP uses `pre_feedforward_layernorm` and `post_feedforward_layernorm_1`. Routed experts use `pre_feedforward_layernorm_2` and `post_feedforward_layernorm_2`. The router normalizes with `router.scale` times `hidden**-0.5`. The token embedding is multiplied by `sqrt(hidden)`. Logits use the tied embedding with no extra scale. Attention uses scale 1. Values use RMSNorm with no scale: a global value is that norm of the key projection and is not rotated.

Google. *google/gemma-4-26B-A4B-it.* Model card. [Hugging Face](https://huggingface.co/google/gemma-4-26B-A4B-it). Those weights are under the [Apache License, Version 2.0](https://ai.google.dev/gemma/docs/gemma_4_license). This pipeline packs a local bfloat16 checkout of those shapes. It does not download.

Barbero, Vitvitskyi, Perivolaropoulos, Pascanu, and Veličković. *Round and Round We Go! What Makes Rotary Positional Encodings Useful?* 2024. [arXiv:2410.06205](https://arxiv.org/abs/2410.06205). Proportional rotary embeddings rotate the first fraction of pairs and leave the rest as the identity. The rotation pairs the first half of the head with the second half.

Zhang and Sennrich. *Root Mean Square Layer Normalization.* 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467). The stored scale is the multiplier. ε is 10⁻⁶.

Hendrycks and Gimpel. *Gaussian Error Linear Units (GELUs).* 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415). The feed-forward uses the tanh approximation.

Georgi Gerganov and the ggml authors. *ggml.* `quantize_row_q4_0_ref` in `ggml-quants.c`. The code is `trunc(x / d + 8.5)` clipped to 0..15. The low nibble is element `i` and the high nibble is element `i + 16`. Reconstruction is `(q − 8) · d`. A QAT block keeps the lattice step when a step `absmax/k` for `k` from 8 down to 1 reconstructs the block within `0.075·step`. The stored scale is the negative of that step, so code −8 is the positive peak. Least squares refits the step, then the codes snap again. A block that fails the test uses the ggml peak.

Open Compute Project. *Microscaling Formats (MX) Specification.* The prefix file stores Metal half in the live cache layout. Routed experts are fp4 E2M1 or fp8 E4M3, with one E8M0 scale per 32 values. The fp4 encoder tries the exponents around the block peak and keeps the codes with the smallest squared error. The highest 20% of experts are stored as fp8, with the block-peak E8M0 scale, and the gate-up and down projections of an expert stay on the same width. The prefix reconstruction and the power-of-two rounding are the same rules [REFERENCES.md](../../REFERENCES.md) records for this repository.

Apple. `MTLTensor` in the macOS 27 SDK, and WWDC26 session 330. `matmul2d` dequantizes the right-hand tensor. The left operand is Metal half and the destination is float32. This pipeline's `kernels` command is the acceptance for that dispatch.
