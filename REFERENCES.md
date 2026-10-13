# References

Public documents this runtime is built from. Algorithms are taken from these papers and specifications. Another application's source is not a reference.

## This model

Qwen Team. *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability.* Alibaba Group, August 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320).

The forward follows this report:

- Gated DeltaNet, equations (1)–(11). State is \(d_k \times d_v\). Decay \(\alpha_t\) and write \(\beta_t\) use (9)–(10). The output gate is a sigmoid, and RMSNorm is zero-centered.
- Qwen Sparse Attention, equations (12)–(16) and (19). Indexer is multi-query, 4 query heads and 1 key head. Keys are mean-pooled in blocks of 4, then normalized, then partial RoPE. Scores are a sum of ReLUs. The tail of an unfinished block is always attended.
- Gated Residual, equations (29)–(34). Four branches. Each branch has its own RMSNorm gain. The read is an elementwise sigmoid gate from a rank-320 bottleneck. The write is one scalar per branch, in \((0, 2)\).
- N-gram embedding at layer 2. Multi-head hashing into a host-memory table, injected by a contextual gate. Placement is Table 7.

Qwen Team. *Qwen3.8-Flash-Next.* Model card and blog, August 2026. [Blog](https://qwen.ai/blog?id=qwen3.8-flash-next).

Shapes, the 3:1 GDN/QSA schedule, the 512-expert top-10 MoE, the shared expert, the 262,144 context, and the sampling defaults come from the model card. Thinking mode is `temperature=1.0`, `top_p=0.95`, `top_k=20`. Instruct mode is `temperature=0.7`, `top_p=0.8`, `top_k=20`. The checkpoint tokenizer and chat template are part of that release. Vision weights stay unloaded.

Qwen Team. *Qwen3-Next: Towards Ultimate Training and Inference Efficiency.* [Alibaba Cloud](https://www.alibabacloud.com/blog/602580). The 3:1 hybrid, partial RoPE on the first quarter of the head, and the head dimension of 256 are introduced here and kept by the Qwen3.8 report.

## Token mixing

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* [arXiv:2412.06464](https://arxiv.org/abs/2412.06464), 2024. The delta recurrence. The Qwen3.8 report changes the output gate from SiLU to sigmoid and transposes the state. Q and K are L2-normalized, and Q is scaled by \(1/\sqrt{d_k}\).

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021. The matrix-state reading of linear attention.

Ashish Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. The softmax attention that QSA sparsifies.

Noam Shazeer. *Fast Transformer Decoding: One Write-Head Is All You Need.* [arXiv:1911.02150](https://arxiv.org/abs/1911.02150), 2019. The indexer is multi-query attention.

Noam Shazeer. *GLU Variants Improve Transformer.* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202), 2020. Routed and shared experts are SwiGLU: the gate and up projections, SiLU on the gate, then the down projection.

Jianlin Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing 568, 2024. Partial RoPE on 64 of 256 core channels, and on 64 of 128 indexer channels. \(\theta = 10^7\). Text uses the three-section interleaved mRoPE widths in the checkpoint config, which sum to that rotary dimension.

A. Liu et al. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.* [arXiv:2512.02556](https://arxiv.org/abs/2512.02556), 2025. QSA follows its sparse-indexer route and replaces token-level scoring with the micro-block indexer in the Qwen3.8 report.

## Residual stream

Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. *Highway Networks.* [arXiv:1505.00387](https://arxiv.org/abs/1505.00387), 2015.

Cenk Baykal et al. *Alternating Updates for Efficient Transformers.* [arXiv:2301.13310](https://arxiv.org/abs/2301.13310), 2023.

Defa Zhu et al. *Hyper-Connections.* [arXiv:2409.19606](https://arxiv.org/abs/2409.19606), 2024.

Zhenda Xie et al. *mHC: Manifold-Constrained Hyper-Connections.* [arXiv:2512.24880](https://arxiv.org/abs/2512.24880), 2025.

Zihan Qiu et al. *A Unified View of Attention and Residual Sinks.* [arXiv:2601.22966](https://arxiv.org/abs/2601.22966), 2026. GatedNorm, which the Qwen3.8 report merges into the Gated Residual read.

Zihan Qiu et al. *Gated Attention for Large Language Models.* [arXiv:2505.06708](https://arxiv.org/abs/2505.06708), 2025. The sigmoid output gate on attention.

Ruibin Xiong et al. *On Layer Normalization in the Transformer Architecture.* ICML 2020. Pre-norm. Gated Residual replaces the block pre-norm, as the report states after equation (34).

## N-gram memory

Aurko Roy et al. *N-Grammer: Augmenting Transformers with Latent N-grams.* [arXiv:2207.06366](https://arxiv.org/abs/2207.06366), 2022. Multi-head hashing.

Hongzhi Huang et al. *Over-Tokenized Transformer.* ICML 2025.

Xin Cheng et al. *Conditional Memory via Scalable Lookup.* ACL 2026. Equations (5) and (6): the branch score starts from a scaled dot of RMSNorm(query) and RMSNorm(key), and the delta adds a dilated depthwise convolution of that gated value. Qwen takes the signed square root of that score before the sigmoid. Qwen does not use the paper's tokenizer compression.

Eight bigram heads then eight trigram heads. The delta is injected into the four residual branches before the layer mix. The convolution history is nine frames.

The n-gram hash is a wrapping int64 XOR of token × multiplier, then a positive remainder of the checkpoint primes, plus the checkpoint offsets. Shift 0 uses the first multiplier. A lookback that crosses EOS, or that starts before the sequence, is the EOS id. Dilation equals the n-gram size. The multipliers, primes, and offsets are the checkpoint tensors, not a second copy of those constants.

Google DeepMind. *Gemma 3n model overview.* 2025. Host-side embedding tables as a capacity axis.

Haoran Liu et al. *Scaling Embeddings Outperforms Scaling Experts in Language Models.* [arXiv:2601.21204](https://arxiv.org/abs/2601.21204), 2026.

## Experts

Zunhai Su et al. *Unveiling Super Experts in Mixture-of-Experts Large Language Models.* [arXiv:2507.23279](https://arxiv.org/abs/2507.23279), 2025. Equation (6): \(a_{l,e} > P_{99.5}(\mathcal{A})\), \(a_{l,e} > a_{\max}/10\), and \(l \in L\). \(a_{l,e}\) is the max absolute `down_proj` output. Those experts are the int8 tier. \(L\) is the early layers that form the massive activations. This pipeline takes the first quarter of the 48 layers, and at least the first four.

Mike Lasby et al. *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression.* ICLR 2026. [arXiv:2510.13999](https://arxiv.org/abs/2510.13999). The int4 keep set, after Super Experts and the per-bucket pins, is filled by \(S_j\), the mean of router weight times expert-output norm over tokens that routed to expert \(j\). Experts outside the keep set, including ones that never fired, are int2. The keep count is a CLI argument, aligned to a multiple of 16.

## Block scaling

Open Compute Project. *OCP Microscaling Formats (MX) Specification*, version 1.0, September 2023. [PDF](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

Used from that spec:

- Block size 32.
- E8M0 is an unsigned biased exponent, bias 127, range \(2^{-127}\) through \(2^{127}\). The encoding `0xFF` is NaN and is not a scale this converter writes.
- Section 6.1: the dot product applies the two block scales outside the element reduction.
- Section 6.3 requires round-ties-to-even and saturation. Its minimally supported scale is the largest power of two less than or equal to absmax divided by the element type's largest power of two. The same section allows other conversion algorithms.

The converter uses the other algorithm: round the power of two **up**, so the peak fits in the signed code range without clamping. For signed int4 the positive endpoint is 7. For int8 it is 127. For int2 it is 1. `scale = 2^{ceil(log2(amax / qmax))}`, stored as `e = log2(scale) + 127`, clamped to `0..254`. A zero block stores exponent 0. Reconstruct `code × 2^(e − 127)`.

Bita Darvish Rouhani et al. *OCP Microscaling Formats for Deep Learning.* September 2023. Cited by the MX spec.

IEEE Std 754-2019. Round-ties-to-even.

## Metal storage

Apple. *Metal Shading Language Specification.* Metal 4.1 on this machine. `unpack` of `packed_int4b` is signed two's complement, −8..7, and element 0 is the low nibble. That is the int4 layout the probe measured. `unpack` does not accept `int2b` here, so int2 is packed by this runtime's own shift: the low two bits are element 0, and the codes are two's complement −2..1.

Apple. `MTLTensor` in the macOS 27 SDK. For int4, int8, int2, fp8, and E8M0, every stride after the innermost is a multiple of 128 bytes, and a quantized data plane's buffer offset is 128-byte aligned. The innermost stride is 1. The scales plane is `MTLTensorDataTypeMetalFloat8UE8M0` with block factors `{32, 1}`. A buffer-backed tensor with auxiliary planes is created with per-plane attachments. A device-allocated tensor uses an opaque layout and is the resident copy.

Apple. WWDC25 session 262, *Combine Metal 4 machine learning and graphics.* A tensor created from a buffer names its strides. A tensor created from the device is the opaque layout.

Apple. WWDC26 session 330, *Optimize custom machine learning operations with Metal tensors.* Quantized data and the E8M0 scale plane are one multi-plane tensor. `matmul2d` dequantizes that tensor. The scale block in the sample is 32 by 1.

The probe in `qwen38-flash-next.py` is the measurement of those rules on this M5 Pro. It is recorded in [docs/qwen38/probe.md](docs/qwen38/probe.md).

## Phase split

Pratyush Patel et al. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677). The prompt and the generated tokens are different phases, and the key/value cache is what crosses the boundary.

Yinmin Zhong et al. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670). The same split, placed so the cache transfer fits the link.

Amey Agrawal et al. *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* [arXiv:2308.16369](https://arxiv.org/abs/2308.16369), 2023. A prompt can be computed in chunks. This runtime does not batch a chunk with other requests.

Yuning Zhang, Grant Pinkert, Nan Yang, Yanli Li, and Dong Yuan. *DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference.* [arXiv:2509.07379](https://arxiv.org/abs/2509.07379), 2025. A prompt activates many experts. A generated token activates a few. Prefill loads overlap the non-expert work.

Xin He et al. *ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling.* DAC 2026. [arXiv:2410.17954](https://arxiv.org/abs/2410.17954). Later routes can be predicted. A miss still loads the routed expert.

Kwanhee Kyung, Sungmin Yun, and Jung Ho Ahn. *SSD Offloading for LLM Mixture-of-Experts Weights Considered Harmful in Energy Efficiency.* [arXiv:2508.06978](https://arxiv.org/abs/2508.06978), 2025. Hiding the wait does not shrink the bytes moved.

Apple. WWDC22 session 10104, *Load resources faster with Metal 3.* An IO command queue runs beside the compute queue. Shared events order them. The queue can be concurrent, and a command buffer can load a buffer directly from a file handle.

Apple. WWDC22 session 10066, *Discover Metal 3.* Many small loads are the case that fast resource loading adds. A long sequential file is the bulk transfer.

The plan that uses these is [docs/prefill-decode-split/prefill-decode-split.md](docs/prefill-decode-split/prefill-decode-split.md). It is not the as-built forward.

## Decode

Mohamed Amine Bergach. *When Quantization Is Free: An int4 KV Cache That Outruns fp16 on Apple Silicon.* [arXiv:2605.05699](https://arxiv.org/abs/2605.05699), May 2026. A fused int4 key/value cache can transfer less per decode step than fp16 on unified memory. The measured models are not this checkpoint.

The plan that uses the Qwen3.8 report, the Metal tensor path above, and that cache result is [docs/decode-options/decode-options.md](docs/decode-options/decode-options.md). It is not the as-built decode.

## Prompt text

The calibration strings are data, not an algorithm. They are a bucket JSON: `tool_calling`, `chat`, `mixed`, `reasoning`, `code`, `en`, `ko`, `zh`, `ja`. `tool_calling` and `chat` are already chat-templated.
