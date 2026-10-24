# Ternary Bonsai 2

Failed pipeline. The command scan does not load it, and `serve` does not serve it. The entry is `pipelines/failed/ternary-bonsai-2.py`. The package is `pipelines/failed/ternary_bonsai_2/`. The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md).

Greedy decode of the packed int2 record did not collect six carrots. The reply was not readable text. The host checks and the carrot file moved with the package to `pipelines/failed/tests/` and are outside the default test run.

The architecture card is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Those weights are under the Apache License, Version 2.0. The attempt packed an F16 GGUF as int2 and wrote `models/Ternary-Bonsai-2-int2`. It did not open a safetensors checkout. The package does not import `qwen38_27b`. The sections below describe that attempt.

## Kernels

`kernels` compiles this package's shader library and fails if any kernel in that library does not produce a pipeline. It then dispatches the int2 `matmul2d` tiles, the int2 GEMV, the 1024-point Hadamard and its inverse, the fp8 key/value tile, multiplicative RMSNorm, SwiGLU, one Gated DeltaNet step, the 10240-channel convolution, an E4M3 decode, partial RoPE at position 0, and a four-token causal fp8 attention.

The Hadamard is \(R = (1/\sqrt{1024})\, H_{1024} S\) along K. Forward multiplies the fixed \(\pm 1\) signs, runs the Walsh–Hadamard, then scales by \(1/32\). The inverse scales after the Hadamard and then multiplies the signs. A gathered embedding row uses the inverse. `in_proj_a` and `in_proj_b` are not in that basis. The sign vectors are float16, widths 5120, 6144, and 17408.

The DeltaNet state is float32, 48 heads of `128×128`. The step decays the state by \(\alpha\), writes \(\beta(v - Sk)k^\top\), and reads with the query. \(\alpha = \exp(-\exp(A_{\log})\mathrm{softplus}(a + \mathrm{dt\_bias}))\) and \(\beta = \mathrm{sigmoid}(b)\). Decode applies that step once per token. Prefill uses the same recurrence across a chunk of at most 2048. Query and key are repeated three times across the 48 value heads. The mixed vector is 10240 wide and the output gate is 6144 wide. Before `out_proj`, that state is reordered from `[128, 16, 3]` to `[128, 3, 16]`. The head RMSNorm multiplies by the stored gain, \(x / \mathrm{rms} \cdot w\), then the gate is SiLU. `in_proj_a` and `in_proj_b` are stored with 64 rows. The recurrence reads heads 0 through 47.

## Record

Every matrix is int2, codes \(-2..1\), low two bits element 0, with an E8M0 scale every 32 along K. That includes the embedding, the projections, the feed-forward, and `lm_head`. K is a multiple of 512 and rows are a multiple of 64. One-dimensional parameters stay float16: RMSNorm gains, `A_log`, and `dt_bias`. The depthwise convolution stays float16 with shape `[10240, 4]`. There is no fp8 or fp4 weight plane and no feed-forward error budget. The record does not contain an MTP block. A finished directory has `manifest.json` with `complete: true`. An unfinished directory resumes from `quantize_progress.json`.

## Forward

The attempt applied the prompt in chunks of at most 2048 tokens, then drew one token at a time. A chunk submits every 4 layers as its own command buffer while every feed-forward layer it runs is in the device buffer. Temperature 0 is the on-device argmax. A chunk's projections are `matmul2d` tiles of 64 by 64. The down projection uses a 32 by 128 tile. Decode of one token is the int2 GEMV. The vocabulary reduction stays on device.

The position limit is 262144. The served window stops at 131072. The live length starts at 2048 and grows by 2048. The fit claims the prefill workspace, the decode buffers, the DeltaNet state, and the convolution history first, then the key/value cache. Feed-forward layers that do not fit are read from the record for that layer. The cache is not shortened to bring a layer back. There is no expert atlas and no MTP plane.

The live key/value cache is fp8 E4M3 on the 16 full-attention layers. A key is `[head, token, channel]` with an E8M0 scale every 32 channels. A value is `[head, channel, token]` with an E8M0 scale every 32 tokens. One token across the 16 layers is 33792 bytes. The prefix file holds the packed key and value positions, the DeltaNet state, and the convolution history. An exact-length file is not sampled.

RMSNorm on the residual stream, the final norm, and the query and key is \(x / \mathrm{rms} \cdot (1 + w)\). Partial RoPE is the first 64 of 256 channels, \(\theta = 10^{7}\). A text position is written on every mRoPE axis. The attention output gate is the sigmoid of the second 256 channels of `q_proj`. The feed-forward is SwiGLU, width 17408.

One decode step reads the int2 language matrices except the embedding table, plus one embedding row: 7208388000 bytes. At 307 GB/s that read is 42.6 tok/s before the key/value cache. That figure is the byte count divided by that bandwidth. It is not a measured rate.

Stop ids are 248046 and 248044. Thinking is temperature 1.0, top-p 0.95, top-k 20, presence penalty 0. Instruct is 0.7, 0.8, 20, 1.5. The template default reasoning effort is xhigh. A request that omits the variant uses the thinking row. `enable_thinking` false uses the instruct row.
