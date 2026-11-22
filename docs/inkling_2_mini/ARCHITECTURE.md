# Inkling 2 Mini

The language pipeline loads the safetensors checkpoint as stored. Attention projections, the router, and the output projection are bf16 in the file. The fused attention input plane, the attention output projection, and the head are Metal fp4: E2M1 with an E8M0 scale per 32 along K, with the power of two rounded up so the group peak fits. The router, the sink, the dynamic convolution taps, and the message gates stay half. The embedding stays half. Layers 0–19 store NVFP4 experts. `bake` requantizes them once to the same fp4 plane and writes `experts-fp4.safetensors`. Layers 20–23 stay MXFP8 in that file. The open reads the dense pack and the expert pack straight into device buffers, in 8 MiB pieces. Without the expert pack, the open requantizes. Prefill multiplies with `matmul2d`. A single token uses the GEMV.

## Setup

`setup` lists the repository on `huggingface.co`, checks that the volume can hold the bytes still missing, and downloads the served files. A file that is already the published length is kept. A published checksum is checked once and recorded in `fetch_complete.json`, so the next run does not read the weight file again. `mtp.safetensors` is the drafter and is not downloaded.

The registry row then stores `kv_cache` `half`, context 16384, `enable_thinking` true, `reasoning_effort` `high`, and `preserve_thinking` true. The variant cards are temperature 1.0, top-p 1.0, top-k 0, and presence penalty 0. `thinking` and `high` use effort `high`. `instruct` and `none` use effort `none`. `minimal`, `low`, `medium`, `xhigh`, and `max` use those efforts.

When `dense-fp4.safetensors` is missing or does not match `config.json`, `setup` runs the calibration below. A replaced weight file drops that pack, the expert pack, and the saved calibration inputs first, so the codes are rounded for the file just downloaded. `--no-calibrate` leaves the nearest-code dense planes. `setup` still writes `experts-fp4.safetensors` when that file is missing or does not match `config.json`.

## Dense calibration

`calibrate` writes `dense-fp4.safetensors` beside the checkpoint. When that file is present and its `config` digest matches `config.json`, the open reads the fp4 planes from it in 8 MiB pieces. Otherwise the open rounds each weight to the nearest code. The file has a `data` and a `scale` tensor for `layers.{i}.attn`, `layers.{i}.wo`, and `unembed`, in the layout the kernels read.

The codes come from GPTQ (Frantar et al., 2022, Algorithm 1). The model answers 24 calibration prompts twice, with thinking on and off, from the nearest-code planes. A capture prefill then runs each prompt and its answer again and sums `X^T X` on the host for every plane input. The query, key, and value rows of the fused plane each use their own input stream. A layer whose three streams agree adds that input to all three. Columns are rounded in order. Each group's E8M0 scale is fixed when its first column is reached. Each rounding error is spread over the columns not yet rounded through the upper Cholesky factor of the damped inverse Hessian (damping 1% of the mean diagonal, blocks of 128). The input sums stay in `dense-fp4-inputs.npz`, so `calibrate --reuse` rounds again without the capture.

On the calibration inputs, the relative output error of a plane falls from 0.06–0.13 to 0.02–0.06 for the attention rows, from 0.11–0.21 to 0.04–0.07 for the output projection, and from 0.041 to 0.019 for the head. The prefix-store tag names the dense pack, so a record written under other planes is not loaded.

MR-GPTQ ([Egiazarian et al., ICLR 2026](https://arxiv.org/abs/2509.23202)) applies GPTQ to MXFP4 and NVFP4 with block Hadamard rotations and format-specific grids. This pipeline uses plain GPTQ on the E8M0-per-32 grid, without rotations, so the kernels are unchanged.

## Schedule

24 stored layers, 48 visits. Layers 0–5 run once. Layers 6–17 run three times, and every pass starts from the embedding plus the prelude block. Layers 18–23 run once. Six layers make one block. Inside a block, each attention output and each expert output goes through the residual short convolution and is added to that block's partial sum. The next block stores the partial as a finished summary and starts again at zero. The embedding is the first summary.

A sublayer does not read the partial directly. It mixes the finished summaries and the current partial: keys are RMS-normalized, the learned query is scaled by `1/sqrt(2304)`, and the values are the normalized sources. Checkpoint `attn_queries.{i}` feeds layer `i + 1`. Layer 0 has no attention query and reads the embedding. The three-query tensors land on the full-attention layers, so their query, key, and value streams are separate mixes. MLP queries stay on the layer they are stored with. The output query mixes the finished summaries for the final RMSNorm.

A later core pass reads the previous pass through four messages per layer. For core index `i`, the query message is row `2i` and the MLP message is row `2i+1` of `middle_cycle_cross_pass_queries`. The key message is row `2i` and the value message is row `2i+1` of `middle_cycle_cross_pass_kv_queries`. The message gate is the published rank-64 blend

```text
h = W_down · [rms(input), rms(message)]
out = (1 − tanh(W_forget · h)) · input + tanh(W_gain · h) · message
```

The blend is applied to the mix, before the sublayer RMSNorm. Full-attention layers are 5, 11, 17, and 23. The other twenty layers use a sliding window of 1024. Each visit has its own key, value, and convolution state.

## Attention

24 query heads, 8 key/value heads, head dim 128. There is no rotary embedding on the text stack. A relative table of extent 1024 adds a per-head bias for distances below that extent. Query, key, and relative vectors are L2-clipped at the thresholds in `config.json`. The query and the relative vector use their learned per-channel scales. The key scale is one. An input-dependent sink logit is an extra attention slot with no value. The output is gated before the output projection, and that projection then passes through the residual short convolution

```text
y[t] = (x[t] + sum_lag w[W-1-lag] x[t-lag]) / sqrt(W+1)
```

with `W = 4`. The last stored tap multiplies the current step. Values are RMS-normalized per head with no gamma. Local layers share one attention RMSNorm. Full layers use separate query, key, and value norms. Their key and value convolutions add the current token and a per-token tap to the static tap, then divide by `sqrt(2W+1)`.

Keys and values are stored half, 1024 channels per position, one plane per visit at the full context. A local visit reads only its last 1024 positions.

Prefill attention is `attend_flash`. One threadgroup of four simdgroups owns 32 query rows of one head. It walks 64-key tiles. `matmul2d` multiplies the half query tile by the key tile, and a second `matmul2d` multiplies the half probabilities by the value tile. Each lane then adds the relative bias for its scores, applies the window and the causal mask, and updates the running maximum and sum. The sink logit seeds that maximum and sum, and the sigmoid gate multiplies the output per channel.

Decode attention is `attend_group`. One threadgroup of eight simdgroups owns one key/value head and its three query heads. The keys split into spans of at least 256, at most four spans per head. Inside a span, eight lanes share one key: each holds sixteen channels of the three queries, and a shuffle reduction finishes the dot. The relative bias is skipped past distance 1024. Each simdgroup keeps an online softmax. The threadgroup merges its eight partials in threadgroup memory and writes one partial per query head. The last span to finish, counted with an atomic, merges the spans, adds the sink, and applies the gate.

## Experts

Eight of 128 experts, no shared expert. `bake` writes `experts-fp4.safetensors`: layers 0–19 are the Metal fp4 the kernels read, and layers 20–23 are the MXFP8 bytes as stored. The open reads that file in 8 MiB pieces after the dense planes. A missing pack still reads the checkpoint and requantizes layers 0–19 on the device. When the budget holds every image, the atlas is complete and each tensor is read once. Otherwise it keeps eight images, the experts one token routes, and copies a miss in before that expert runs.

One layer records its attention kernels into one command buffer and waits when the CPU reads the router. A complete atlas then records that layer's expert kernels into the next buffer. A single token records the whole forward in one command buffer. Its fp4 projections are a four-row GEMV, and its RMSNorm and block mix each run as one threadgroup of 256. `route_gemv` scores the router in 32 threadgroups of four rows. The last group to finish, counted with an atomic, picks the eight experts on the GPU, and those experts run as one grid of four-row groups. A decode step reads its token from the device, so the host encodes step n + 1 while the device runs step n. When sampling allows it (no top-k, top-p, penalty, or forbidden token), the GPU draws the token with a Gumbel maximum. Prefill still waits at the router so the CPU can group tokens. A partial atlas waits after each expert, because the next expert reuses the same slot. The router scores with a sigmoid. The gate bias is added only to choose the eight experts. The selected sigmoid scores are renormalized and multiplied by `route_scale` 8. The fused gate and up rows are interleaved, even row then odd row. The activation is `softcap(gate) * sigmoid(gate) * softcap(up)`, with caps 4 and 25. The expert output then takes the same residual short convolution as attention.

NVFP4 reconstruction is `e2m1 * e4m3(block_scale) * (amax / 2688)`. The 2688 is the file's outer-scale denominator, which is `448 * 6`. The load kernel turns each block of 32 into an E2M1 code and one E8M0 scale. K is padded to a multiple of 256, so the down projection's 1152 becomes 1280 and the extra columns stay zero. MXFP8 weights are E4M3 with one E8M0 scale per 32 along K, which is the native scale plane `{32, 1}`.

## Chat

The checkpoint template injects `Thinking effort level:` and, when tools are present, a `tool_declare` block. A request effort is converted to the template's number before that render: a listed name uses the template map, and a number is clamped to `[0.0, 0.99]`. The template has no generation prompt: the model writes `<|message_model|>` itself, then one body marker. The stop token is `<|content_model_end_sampling|>` (200006). Logits past the unpadded vocab of 200058 are ignored. Thinking text is `<|content_thinking|>`. Reply text is `<|content_text|>`. A tool call is `<|message_model|>name<|content_invoke_tool_json|>` and a JSON object with `name` and `args`. Each message ends with `<|end_message|>`.

The parser splits the completion at those markers. A body cut off by the length limit keeps its text. A tool body is read with a JSON decoder that tolerates raw control characters. A backslash that does not start a JSON escape is doubled and the body is read again. Text with no markers is the reply.

A required tool forces the opening `<|message_model|>name<|content_invoke_tool_json|>{"name":"…","args":`. With thinking off and no tool to call, the opening is `<|message_model|><|content_text|>`. Otherwise the model opens its own message. `encode_chat` appends the forced text to the prompt tokens, so a prompt length counts it. The parser treats it as part of the completion.

A stream decodes tokens incrementally and holds back a split UTF-8 tail for at most eight tokens. A gate over the markers emits reasoning deltas, reply deltas, and, for a call, an added item and argument deltas as the JSON arrives. A writer thread sends the events from a queue. A hosted `web_search` request adds its search item first, in the stream and in the stored output.

A Gumbel draw takes 23 bits of a hash, adds one half, and scales by `2^-23`, so the uniform is exact in float and never 0 or 1.

## Prefix store

The pipeline uses the shared SSD prefix store. A record has one `conv` region with every short-convolution state of the 48 visits, then `kNN` and `vNN` for each visit. A full-attention visit stores positions `0 .. count-1`. A local visit stores only the last 1024 of them. A record continues a prompt only when the prompt is longer, so an exact-length file is never sampled and the next prefill steps at least one token.

The live sequence is written twice per request: after prefill, and after decode. The device must then hold exactly the prompt and every returned token. Pipelined decode may already have applied the stop token. Each step therefore first copies the convolution states into one of two snapshot slots, and the step that applied the stop token is taken back from its slot. Keys past the returned tokens are never read. A decode that ended at its limit runs one more step to apply its last token. With `favlo:fill_decode` no snapshot is taken.

## Memory

The open reads `memory.budget()`. Dense projections load first, from the calibrated file when present, in 8 MiB pieces. The expert pack is read the same way. The half key/value cache for 48 visits is reserved at context 16384. Activation rows stay at 16384 when the complete atlas fits beside that cache. Otherwise the rows shrink, down to 512, and prefill walks the prompt in those rows, so the atlas can still hold every expert image. The atlas takes eight images only when a complete atlas does not fit at any of those widths. The context halves, down to 512, only when those eight images and the cache do not fit. Below the minimum, open raises `MemoryError`. The key/value cache is half. The convolution states take one buffer and two snapshot slots, about 11.5 MB together.
