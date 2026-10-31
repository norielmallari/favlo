# Qwen3.8-Flash-Next references

Papers and specifications this pipeline is built from. The as-built forward is [architecture.md](architecture.md). Shared hardware citations stay in [../../REFERENCES.md](../../REFERENCES.md).

## Model

Qwen Team. *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability.* Alibaba Group, August 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320).

The forward follows this report:

- Gated DeltaNet, equations (1)–(11). State is \(d_k \times d_v\). Decay \(\alpha_t\) and write \(\beta_t\) use (9)–(10). The output gate is a sigmoid, and RMSNorm is zero-centered.
- Qwen Sparse Attention, equations (12)–(16) and (19). Indexer is multi-query, 4 query heads and 1 key head. Keys are mean-pooled in blocks of 4, then normalized, then partial RoPE. Scores are a sum of ReLUs. The tail of an unfinished block is always attended.
- Gated Residual, equations (29)–(34). Four branches. Each branch has its own RMSNorm gain. The read is an elementwise sigmoid gate from a rank-320 bottleneck. The write is one scalar per branch, in \((0, 2)\).
- N-gram embedding at layer 2. Multi-head hashing into a host-memory table, injected by a contextual gate. Placement is Table 7.

Qwen Team. *Qwen3.8-Flash-Next.* Model card and blog, August 2026. [Blog](https://qwen.ai/blog?id=qwen3.8-flash-next). The checkpoint is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Those weights are under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).

Shapes, the 3:1 GDN/QSA schedule, the 512-expert top-10 MoE, the shared expert, the 262,144 context, and the sampling defaults come from the model card. Thinking mode is `temperature=1.0`, `top_p=0.95`, `top_k=20`. Instruct mode is `temperature=0.7`, `top_p=0.8`, `top_k=20`. The checkpoint tokenizer and chat template are part of that release. Vision weights stay unloaded. Those rows are `sampling.py`.

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

The n-gram hash is a wrapping int64 XOR of token × multiplier, then a positive remainder of the checkpoint primes, plus the checkpoint offsets. Shift 0 uses the first multiplier. A lookback that crosses EOS, or that starts before the sequence, is the EOS id. Dilation equals the n-gram size. The multipliers, primes, and offsets are the checkpoint tensors, not a second copy of those constants. The as-built hash is [ngram.md](ngram.md).

Google DeepMind. *Gemma 3n model overview.* 2025. Host-side embedding tables as a capacity axis.

Haoran Liu et al. *Scaling Embeddings Outperforms Scaling Experts in Language Models.* [arXiv:2601.21204](https://arxiv.org/abs/2601.21204), 2026.

## Experts

Zunhai Su et al. *Unveiling Super Experts in Mixture-of-Experts Large Language Models.* [arXiv:2507.23279](https://arxiv.org/abs/2507.23279), 2025. Equation (6): \(a_{l,e} > P_{99.5}(\mathcal{A})\), \(a_{l,e} > a_{\max}/10\), and \(l \in L\). \(a_{l,e}\) is the max absolute `down_proj` output. Those experts are the int8 tier. \(L\) is the early layers that form the massive activations. This pipeline takes the first quarter of the 48 layers, and at least the first four.

Mike Lasby et al. *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression.* ICLR 2026. [arXiv:2510.13999](https://arxiv.org/abs/2510.13999). The int4 keep set, after Super Experts and the per-bucket pins, is filled by \(S_j\), the mean of router weight times expert-output norm over tokens that routed to expert \(j\). Experts outside the keep set, including ones that never fired, are int2. The keep count is a CLI argument, aligned to a multiple of 16.

## Decode

Mohamed Amine Bergach. *When Quantization Is Free: An int4 KV Cache That Outruns fp16 on Apple Silicon.* [arXiv:2605.05699](https://arxiv.org/abs/2605.05699), May 2026. A fused int4 key/value cache can transfer less per decode step than fp16 on unified memory. The measured models are not this checkpoint.

The as-built decode paths are in [schedule.md](schedule.md), including MTP. The int4 cache in the paper above is not one of those paths.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). One target call scores the drafted span. A drafted token \(x\) is kept with probability \(\min(1, p(x)/q(x))\). A rejection draws from \(\mathrm{norm}(\max(0, p-q))\). The output distribution is the target's, at every temperature. Temperature 0 is that rule when \(p\) is a point mass on the argmax.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling.* [arXiv:2302.01318](https://arxiv.org/abs/2302.01318), 2023. Temperature, nucleus, and top-k are applied to both distributions before the same test. Scoring a short continuation costs about one sample when the call is bound by reading the weights.

The decode that uses these is [schedule.md](schedule.md).

The key/value cache defaults to E4M3 for completed blocks of four. The unfinished tail stays BF16. `--kv-cache bf16` stores every K and V token as BF16. The indexer key stays BF16.

`probe` writes one block of 32 and reads it back on the GPU. It is the measurement in [probe.md](probe.md). It does not pack an expert. `matmul` runs one fp4 E2M1 tile and one fp8 E4M3 tile through `matmul2d`, and the fp4 tile through the decode shader dot.

## Prompt text

The calibration strings are data, not an algorithm. They are a bucket JSON: `tool_calling`, `chat`, `mixed`, `reasoning`, `code`, `en`, `ko`, `zh`, `ja`. `tool_calling` and `chat` are already chat-templated.

## Expert cache

Juncheng Yang, Yazhuo Zhang, Ziyue Qiu, Yao Yue, and K. V. Rashmi. *FIFO Queues are All You Need for Cache Eviction.* Symposium on Operating Systems Principles, 2023. [doi:10.1145/3600006.3613147](https://doi.org/10.1145/3600006.3613147). S3-FIFO is a selectable atlas strategy. A new run enters a FIFO of one tenth of the atlas. A run requested only once leaves that queue. A run requested again moves to the main FIFO.

Leyang Xue, Yao Fu, Zhan Lu, Chuanhao Sun, Luo Mai, and Mahesh Marina. *MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache.* [arXiv:2401.14361](https://arxiv.org/abs/2401.14361), 2024. Expert replacement follows how often each expert is activated.

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang. *fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving.* [arXiv:2502.05370](https://arxiv.org/abs/2502.05370), 2025. The expert cache is LFU. A layer-sequential pass makes recency the wrong signal.

Kexin Li, Wenkan Huang, Qinggang Wang, Long Zheng, Xiaofei Liao, Hai Jin, and Jingling Xue. *Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching.* Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 2025. [doi:10.1145/3712285.3759903](https://doi.org/10.1145/3712285.3759903). Cold experts are the first victims. That split plus LFU is the selectable ``cold_lfu`` strategy. The atlas default is S3-FIFO.
