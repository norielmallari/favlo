# References

Public documents this process is built from. A pipeline's papers live in that pipeline's note. Another application's source is not a reference.

Qwen3.8-Flash-Next citations are [docs/qwen38-flash-next/references.md](docs/qwen38-flash-next/references.md).

## Qwen3.8-27B

Qwen Team. *Qwen3.8-27B.* The 64-layer 3:1 mix of Gated DeltaNet and full attention, 24 query heads over 4 key/value heads, head size 256, and the 262,144-token context. The as-built note is [docs/qwen38-27b/README.md](docs/qwen38-27b/README.md).

Qwen Team. *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability.* Alibaba Group, August 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320). Equations (9) and (10) for the Gated DeltaNet gate.

The step in this pipeline decays the \(128 \times 128\) state by \(\alpha\), then writes \(\beta(v - S k) k^\top\), then reads with the query. \(\alpha = \exp(-\exp(A_{\log})\mathrm{softplus}(a + \mathrm{dt\_bias}))\) and \(\beta = \mathrm{sigmoid}(b)\). Decode applies that step once per token. Prefill uses the same recurrence over a chunk of at most 2048 tokens. The output gate is SiLU. The head norm multiplies by the stored gain, \(x / \mathrm{rms} \cdot w\). The residual stream, the final norm, and the query and key norms are zero-centered, \(x / \mathrm{rms} \cdot (1 + w)\). A BF16 step on the stored weights still has to accept this layout.

Yang, Kautz, and Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* [arXiv:2412.06464](https://arxiv.org/abs/2412.06464), 2024.

Ashish Vaswani et al. *Attention Is All You Need.* 2017. The scaled dot-product on the full-attention layers.

Tri Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* 2022. The tiled score and value mix.

Noam Shazeer. *GLU Variants Improve Transformer.* 2020. SwiGLU on the feed-forward.

Jianlin Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021. Partial RoPE on the first 64 of 256 channels, \(\theta = 10^{7}\).

Peng Wang et al. *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.* 2024. Text writes the same position on every mRoPE axis, so the rotation is plain RoPE.

Zihan Qiu et al. *Gated Attention for Large Language Models.* [arXiv:2505.06708](https://arxiv.org/abs/2505.06708), 2025. The sigmoid that multiplies the softmax output.

## Qwen3.6-35B-A3B

Qwen Team. *Qwen3.6-35B-A3B.* Model card, April 2026. [Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The 40-layer 3:1 mix, 256 routed experts plus one shared expert, the sampling card, and the 262,144-token default RoPE. The YaRN factor-4 extension in that card is not enabled. The as-built note is [docs/qwen36-35b-a3b/README.md](docs/qwen36-35b-a3b/README.md).

Qwen Team. *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability.* Alibaba Group, August 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320).

The Gated DeltaNet step in this pipeline decays the \(128 \times 128\) state by \(\alpha\), then writes \(\beta(v - S k) k^\top\), then reads with the query. \(\alpha = \exp(g)\) and \(\beta = \mathrm{sigmoid}(b)\) with \(g = -\exp(A_{\log})\mathrm{softplus}(a + \mathrm{dt\_bias})\), from Yang et al. and Qwen3-Next equations (9) and (10) in the Qwen3.8 report. Decode applies that step once per token. Prefill uses the chunkwise form in Yang, Kautz, and Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464), §3.3: chunks of 64, the ungated Householder factor on the keys, and the decay carried in the value factor. The output gate is SiLU. The state is not the transposed state from that report. Query and key are L2-normalized and query is scaled by \(1/\sqrt{128}\). A BF16 step on the stored weights still has to accept this layout.

Zihan Qiu et al. *Gated Attention for Large Language Models.* [arXiv:2505.06708](https://arxiv.org/abs/2505.06708), 2025. The sigmoid that multiplies the softmax attention output.

Gemma Team. *Gemma: Open Models Based on Gemini Research and Technology.* The zero-centered RMSNorm, \(x / \mathrm{rms} \cdot (1 + w)\).

Dan Hendrycks and Kevin Gimpel. *Gaussian Error Linear Units.* 2016. The tanh approximation the vision MLP names `gelu_pytorch_tanh`.

## Block scaling

Open Compute Project. *OCP Microscaling Formats (MX) Specification*, version 1.0, September 2023. [PDF](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

Used from that spec:

- Block size 32.
- E8M0 is an unsigned biased exponent, bias 127, range \(2^{-127}\) through \(2^{127}\). The encoding `0xFF` is NaN and is not a scale this converter writes.
- Section 6.1: the dot product applies the two block scales outside the element reduction.
- Section 6.3 requires round-ties-to-even and saturation. Its minimally supported scale is the largest power of two less than or equal to absmax divided by the element type's largest power of two. The same section allows other conversion algorithms.

The converter uses the other algorithm: round the power of two **up**, so the peak fits in the format without clamping. For fp4 E2M1 the positive endpoint is 6. For fp8 E4M3 it is 448. For int2 it is 1. `scale = 2^{ceil(log2(amax / qmax))}`, stored as `e = log2(scale) + 127`, clamped to `0..254`. A zero block stores exponent 0. Reconstruct `value × 2^(e − 127)`.

Open Compute Project. *OCP Microscaling Formats (MX) Specification.* The fp4 E2M1 magnitudes are `0, 0.5, 1, 1.5, 2, 3, 4, 6`, and bit 3 of the nibble is the sign. *OCP 8-bit Floating Point Specification.* E4M3FN is the finite format whose largest value is 448 and whose all-ones mantissa at the top exponent is NaN. Metal names these `metal_fp4_e2m1` and `metal_fp8_e4m3`. The half × int8 `matmul2d` op compiles on this GPU and its dispatch does not complete, so integer int8 is not a weight plane.

Bita Darvish Rouhani et al. *OCP Microscaling Formats for Deep Learning.* September 2023. Cited by the MX spec.

IEEE Std 754-2019. Round-ties-to-even.

## Metal storage

Apple. *Metal Shading Language Specification.* Metal 4.1 on this machine. `unpack` of `packed_int4b` is signed two's complement, −8..7, and element 0 is the low nibble. That is the int4 layout a pipeline probe measures. `unpack` does not accept `int2b` here, so int2 is packed by a shift: the low two bits are element 0, and the codes are two's complement −2..1.

Apple. `MTLTensor` in the macOS 27 SDK. For int4, int8, int2, fp8, and E8M0, every stride after the innermost is a multiple of 128 bytes, and a quantized data plane's buffer offset is 128-byte aligned. The innermost stride is 1. The scales plane is `MTLTensorDataTypeMetalFloat8UE8M0` with block factors `{32, 1}`. A buffer-backed tensor with auxiliary planes is created with per-plane attachments. A device-allocated tensor uses an opaque layout and is the resident copy.

Apple. WWDC25 session 262, *Combine Metal 4 machine learning and graphics.* A tensor created from a buffer names its strides. A tensor created from the device is the opaque layout.

Apple. WWDC26 session 330, *Optimize custom machine learning operations with Metal tensors.* Quantized data and the E8M0 scale plane are one multi-plane tensor. `matmul2d` dequantizes that tensor. The scale block in the sample is 32 by 1.

## Phase split

Pratyush Patel et al. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677). The prompt and the generated tokens are different phases, and the key/value cache is what crosses the boundary.

Yinmin Zhong et al. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670). The same split, placed so the cache transfer fits the link.

Amey Agrawal et al. *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* [arXiv:2308.16369](https://arxiv.org/abs/2308.16369), 2023. A prompt can be computed in chunks. This process does not batch a chunk with other requests.

Yuning Zhang, Grant Pinkert, Nan Yang, Yanli Li, and Dong Yuan. *DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference.* [arXiv:2509.07379](https://arxiv.org/abs/2509.07379), 2025. A prompt activates many experts. A generated token activates a few. Prefill loads overlap the non-expert work.

Xin He et al. *ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling.* DAC 2026. [arXiv:2410.17954](https://arxiv.org/abs/2410.17954). Later routes can be predicted. A miss still loads the routed expert.

Kwanhee Kyung, Sungmin Yun, and Jung Ho Ahn. *SSD Offloading for LLM Mixture-of-Experts Weights Considered Harmful in Energy Efficiency.* [arXiv:2508.06978](https://arxiv.org/abs/2508.06978), 2025. Hiding the wait does not shrink the bytes moved.

Apple. WWDC22 session 10104, *Load resources faster with Metal 3.* An IO command queue runs beside the compute queue. Shared events order them. The queue can be concurrent, and a command buffer can load a buffer directly from a file handle.

Apple. WWDC22 session 10066, *Discover Metal 3.* Many small loads are the case that fast resource loading adds. A long sequential file is the bulk transfer.

The plan that uses these is a local note under `artifacts/implementation-plan/docs/prefill-decode-split/` (gitignored). It is not an as-built forward.

## Benchmarks

Jeffrey Zhou et al. *Instruction-Following Evaluation for Large Language Models.* [arXiv:2311.07911](https://arxiv.org/abs/2311.07911), 2023. IFEval. Prompt-level and instruction-level accuracy, strict and loose.

Karl Cobbe et al. *Training Verifiers to Solve Math Word Problems.* [arXiv:2110.14168](https://arxiv.org/abs/2110.14168), 2021. GSM8K. The final number is the `####` line.

Mark Chen et al. *Evaluating Large Language Models Trained on Code.* [arXiv:2107.03374](https://arxiv.org/abs/2107.03374), 2021. HumanEval. pass@1 is one sample against the published tests.

The harness writes a score into `docs/<pipeline>/benchmarks.md` when that file exists for the requested model id.
