# MADMetal architecture

As-built contract for **MADMetal**. The name expands to MADMetal Accelerated Decoding on Metal.

One model is one pipeline. `madmetal --qwen38-flash-next` loads `qwen38-flash-next.py`, which loads the `qwen38_flash_next` package. A future model is a new entry file, a new package, and a new flag. It does not share this forward, these records, or these kernels.

The target machine is an M5 Pro: 20 shader cores, a neural accelerator beside the shader pipelines in each core, 307 GB/s unified DRAM, and an APFS volume whose device block and allocation block are 4096 bytes. Requires macOS 27 and the matching Metal SDK. The 16-core Neural Engine is not used. Sub-byte tensor planes also require 128-byte alignment inside the record. A 4 KB file offset does not by itself satisfy that.

## In scope

- `qwen38_flash_next` runs three stages for this model: a BF16 calibration prefill, a converter from Hugging Face BF16 shards to native records, and the text runtime. `qwen38-flash-next.py` is the entry the command loads.
- One sequence at a time. Prefill and decode. Layered kernels.
- `GET /v1/models` and `POST /v1/chat/completions`, including streaming.
- The checkpoint chat template. Sampling is temperature, top-p, and top-k.
- Tool calls. The template renders `tools`. During a tool-call span the sampler masks the 248320-way logits to that call's schema, and the mask is applied before top-k and top-p. Thinking text is unconstrained. The stream emits OpenAI `tool_calls` deltas.
- A prefix cache so a continuation prefills only the new tokens.
- Expert streaming from SSD into a resident atlas. Consecutive expert ids are one sequential Metal IO. The shader reads the live image from that atlas.

## Out of scope

- A generic runner and any edit to `qwen38_flash_next` on behalf of another model.
- Affine group-64 packing and per-layer bit allocation.
- Batching and the Neural Engine.
- The vision tower. It stays unloaded.
- Multi-token prediction during serving. MTP experts are packed. The endpoint decodes one token at a time, because a draft token would have to pass the same grammar mask.
- Fusing a whole layer into one kernel. The runtime is a layered loop. Shaders share a command buffer until the host needs an expert file offset or the sampled token id.
- fp4 and fp8 as expert data planes. Further key/value widths: unscaled fp8, int8, int4, fp4, int2, and a wider K than V.
- Tool-call grammars on speculative drafts.
- A dependency on another inference app: its packages, its virtualenv, or its files. The README credit is a link, not a library.
- A second, slower forward. Host GEMV and CPU matmul of activations are not stand-ins for a missing shader. The converter's packing runs on the CPU because it writes the file. The request path does not.

## Qwen3.8-Flash-Next

48 layers, hidden 2560. Twelve repeats of three Gated DeltaNet layers and one Qwen Sparse Attention layer, each followed by MoE. The router is 512-wide, top 10, plus one shared expert. Expert intermediate size is 640. The token embedding and the output head are separate untied matrices, vocabulary 248320.

Gated DeltaNet (36 layers) keeps a fixed 128×128 state per value head, 48 value heads. Sparse attention (12 layers) stores K and V for 2 heads of 256 channels, and a 128-channel indexer key per token. The indexer pools keys in blocks of 4, then picks up to 512 blocks (2048 tokens) plus the unfinished tail.

The n-gram table is injected at decoder layer 1 (`ple_layer_ids` is `[2]`, compared as `layer_idx + 1`), before that layer's residual read. The first eight heads hash the bigram ending at the current token. The next eight hash the trigram. Shift 0 is the current token and uses the first checkpoint multiplier; each older token uses the next multiplier. A lookback that starts before the segment, or that crosses EOS, contributes the EOS id. The mix is a wrapping int64 XOR of token × multiplier, then the positive remainder of that head's prime, plus the head's offset. Each row is 160 channels, and the sixteen rows concatenate to 2560. `value_proj` is shared. `key_proj` produces one key per branch. The branch score is `sign(s) √|s|` where `s = RMSNorm(R_m) · RMSNorm(K_m) / √d`, and the gate is σ of that score. The delta is the gated value plus SiLU of a depthwise causal convolution of its branch RMSNorm, kernel 4 and dilation 3, with nine frames of history. About 320 million rows. An int4 row plus five E8M0 bytes is 85 bytes, stored in a 128-byte slot. The request path loads the 16 KB page that holds that slot.

The residual read normalizes each branch with a Gemma gain, `x / rms · (1 + w)`, then a low-rank sigmoid gate. The low-rank map is scaled by `1/n_r` before SiLU. The block input is the mean of the gated branches, the second `1/n_r`. The write is `2 σ(proj / n_r)` per branch. The final mixer is the read only.

MTP is one extra sparse-attention layer with its own 512 experts.

## Records

Record version 2. The file is the buffer image Metal reads. Three files match three reads. `resident.bin` is one sequential load at startup, and its length is a multiple of 1 MiB. `ngram.bin` is random 16 KB pages. Each expert layer file is the live images packed back to back. Inside a live image, the data plane comes first and the E8M0 scale plane follows. Each plane starts on a 128-byte boundary. One Metal IO load covers one expert, both projections, both planes, and stops at the end of that image.

That image is what `MTLTensor` documents for a buffer-backed multi-plane tensor, and what WWDC26 session 330 passes to `matmul2d`. The innermost stride is 1 element. For int4, int8, int2, and E8M0, every later stride is a multiple of 128 bytes. The scale plane is `MTLTensorDataTypeMetalFloat8UE8M0` with block factors `{32, 1}`: one byte covers 32 values along K and the other dimension is not blocked. TensorOps slices both planes together and dequantizes in the matmul. A device-allocated tensor is the opaque layout Apple names as the fastest resident copy. The file stays the buffer image. After a weight is resident (the output head, and later a Super Expert), the runtime may upload that image into a device tensor once. A cold expert is that load. The decode shader reads the data plane and the scale plane at their offsets in the buffer.

int4 rows are stored at a K rounded up to a multiple of 256, so the row is a whole number of 128-byte groups. int8 rounds K up to a multiple of 128. int2 rounds K up to a multiple of 512. The extra K columns are zero codes. The scale row is `K/32` bytes, packed tight. `tensor_inline` takes that stride from the data extent and the block of 32, so a padded scale stride would make `matmul2d` read the wrong byte. The data plane length is a multiple of 128, and the scale plane starts on that boundary. The reconstruct is `code × 2^(e − 127)`. Exponents stay in `0..254`. There is no zero point.

One middle-tier expert is `gate_up` at K 2560 and `down` at stored K 768 (logical 640). The live image is 2,785,280 bytes, which is 170 pages of 16 KB, and that image is the on-disk stride. Expert `i` starts at `i × live`. A run of consecutive ids is that many images and nothing between them. The layer file is preallocated before the images are written, so the logical range is real SSD extents rather than a hole. A wide image is 5,068,800 bytes and a narrow image is 1,658,880 bytes; those tiers are not in this record. A live image that is short of a 16 KB page is read through the next page boundary; that tail is zeros. The same expert id is the same tier in both projections. `num_experts` stays 512. The router id maps to `(tier, row)`. One layer is one file, written in expert-id order.

| Tier | Data plane | One expert | Who |
|---|---|---|---|
| Wide | `metal_int8` | 4.83 MB | Super Experts. Resident |
| Middle | `metal_int4` | 2.66 MB | Domain pin and the REAP fill |
| Narrow | `metal_int2` | 1.58 MB | The complement, including experts that never fired |

A middle-tier expert can stay on SSD. A narrow-tier expert can be cached without changing its tier. A high activation energy must not promote a narrow row. Until a profile supplies the keep set, every expert record is the middle tier.

A record directory starts with the Hugging Face model name, then `-madmetal-`, then each tier that contains experts. Counts are experts per layer. `s` is the super-expert pin and its plane, `h` is the hot keep set, `c` is the cold complement. The planes are `int8`, `int4`, and `int2`. A tier with zero experts is omitted. Super experts are the wide resident pin, hot experts are the middle tier, and cold experts are the narrow tier. The default directory is `models/Qwen3.8-Flash-Next-madmetal-h512-int4`: 512 hot experts at int4.

| Weight | Record |
|---|---|
| Output head | Middle int4. Loaded once into a device tensor |
| Token embedding | Middle int4. Loaded once. A token gathers one row from that copy |
| Gated residual, DeltaNet projections, sparse-attention projections, indexer, shared expert, n-gram `key_proj` and `value_proj` | Middle int4. Loaded once |
| Router, 512×2560 | 16-bit. Loaded once. It picks the tier |
| N-gram table | One int4 table. Each row occupies 128 bytes. The request path loads that slot |
| MTP experts | The same middle image, packed with no hole between experts |
| Norms, DeltaNet convolution, n-gram convolution | 16-bit. Loaded once |
| Rotary embedding | Computed, not stored |
| Running residual, four branches | 16-bit |

The n-gram table is not tiered. A row's payload is 160 int4 channels plus five E8M0 bytes, stored in a 128-byte slot so a 16 KB page holds 128 complete rows. The request path loads the 16 KB page that contains each selected slot, through a small page cache, and a shader reconstructs the 160 channels. The matmul that consumes a gathered row is `value_proj`, and that projection is one of the matrices loaded at startup. The hash is a wrapping int64 XOR and then a positive remainder. There is no per-row score.

Expert loads use an uncached Metal IO handle. Uncached reads on this machine stay on the direct path when the file offset, the length, and the destination offset are multiples of the 16 KB VM page. The volume's preferred transfer is 1 MiB, so a sequential file is a multiple of that and a single expert image is larger than that. Missing ids are sorted, and each run of consecutive ids is one sequential load of `(last − first + 1) × live` bytes from file offset `first × live` into one resident atlas. The destination offset in that atlas is the same multiple of 16 KB. The runtime keeps the largest of 20, 16, 12, and 8 GiB that the device accepts. An int64 table stores each image base, and an empty entry is −1. The shader reads the live bytes at that base. A run that does not intersect the ids being staged can be evicted. Matrices named above as loaded once are read at startup into one shared buffer. Their per-token path does not return to the SSD. A single load of `resident.bin` returns `MTLIOStatusError` on this GPU, so that startup read is split into 256 MB ranges inside one IO command. `MTLIOStatusComplete` is 3. An expert or n-gram load signals a shared event, and the compute command buffer waits on that event, so the host keeps encoding while the load runs.

The scale for a block of 32 is one E8M0 byte, bias 127. It is the power of two at or above `amax / qmax`, so the peak fits in the signed codes. `qmax` is 7 for int4, 127 for int8, and 1 for int2. The stored exponent is `log2(scale) + 127`, clamped to `0..254`. `0xFF` is not written. A zero block stores 0. Elements use round-ties-to-even and then saturate. The citations are [REFERENCES.md](REFERENCES.md).

int4 is packed low nibble first, signed −8..7, as measured below. int2 is packed low two bits first, signed −2..1. Metal 4.1 `unpack` does not accept `int2b` on this SDK, so that int2 layout is the one the shader shifts. The measured int4 layout is below.

## Key/value cache

The serving default stores the full sparse-attention K and V in BF16. Every cached token, both heads, all twelve layers. That is 24,576 bytes per token, and 6.0 GiB at the native 262,144-token context.

`metal_fp8_e4m3` and `metal_fp8_e5m2` are implemented, each with an E8M0 scale every 32 channels along the head. They are a switch. A tool-use journey has to stay good on fp8 before that switch is the one the server starts with. At E4M3 the same K and V are 12,672 bytes per token, 3.094 GiB at 262,144 tokens.

When the fp8 switch is on, completed blocks of four tokens are stored fp8 and the unfinished tail stays BF16. That tail is not a cache hit. When the default is on, the whole history including the tail is BF16.

Indexer keys stay BF16 in every mode. At 262,144 tokens they add 0.75 GiB. Pooling is mean, then cast back, then norm, then rotary, in that order.

## Prefix cache

One request at a time. A hit is an exact token prefix that was stored, after the checkpoint template has been applied. The usual hit is the previous turn of the same chat. Two chats that share a system prompt hit only when that prompt was stored as its own prefix. With the BF16 default, the stored length is the real token count. The multiple-of-four rule applies only while the fp8 switch is on.

Each cached prefix stores, at that boundary:

- K and V for every cached token in the active format, and the indexer key for every cached token
- the Gated DeltaNet recurrent state once, not once per token
- the Gated DeltaNet depthwise convolution history (kernel 4), once
- the n-gram convolution state and the tokens already stepped, once

Grammar state is not cached. It starts again on the new generation. When RAM is short, the starting policy is to drop prefix-cache entries before the resident expert tiers. That policy is a default to be measured: a short prefix can be cheaper to prefill again than a cold expert read. One cold middle expert in every layer is 2,785,280 bytes, about 128 MiB. Ten misses, a fully cold token, are about 1.25 GB.

## Endpoint

`madmetal --qwen38-flash-next serve` listens for `GET /v1/models` and `POST /v1/chat/completions`, including a streaming response. Messages are tokenized with the checkpoint chat template. A request may carry `tools`, and the template renders them. The sampler applies temperature, top-p, and top-k. Generation stops on the tokenizer end-of-turn id and on the text-config eos. A request that omits `max_tokens` and `max_completion_tokens` runs until one of those ids. A request that sets either field stops after that many tokens. Either way, generation stops before the sequence passes the 2048-token indexer budget. The n-gram segment boundary stays the text-config eos. A tool-call span does not yet mask logits. Thinking is off unless the request sets `enable_thinking`. Tool results on the next turn are ordinary messages, and the cache key is the token ids the template actually emitted. The server keeps one stepped prefix. A request whose ids extend that prefix continues; any other prefix clears the state and steps from the start.

The checkpoint's tool-call syntax is whatever the template emits. This server returns the sampled text as message content. It does not yet parse that span or emit `tool_calls` deltas.

## Calibration

The profiler does not quantize a weight. It runs this app's BF16 prefill. The default collect is 128 sequences of 512 tokens, drawn from prompt strings in a bucket JSON (`tool_calling`, `chat`, `mixed`, `reasoning`, `code`, `en`, `ko`, `zh`, `ja`). That file is prompt text. The citations are [REFERENCES.md](REFERENCES.md).

Only the strings matter. A later corpus file can use another layout. `tool_calling` and `chat` are already wrapped in `<|im_start|>` turns, so those strings are tokenized as they stand. The other buckets are raw text. The activation file is rebuilt when the strings or the tokenizer change.

For each routed expert it records how many tokens selected it, the input energy to `gate_up` along K = 2560, and the input energy to `down_proj` along K = 640. An expert only accumulates tokens that routed to it. Each dense projection stores one energy vector. The router, the n-gram table, and the key/value cache are not in the file. Tier lists are copied from selection so a hot block cannot promote a narrow expert.

The collect signature is the corpus hash, the sample count, the sequence length, and the tier assignment. Resume is per sequence.

## Schedule

Prefill and decode run as Metal shaders on the Apple Silicon GPU. There is no host GEMV on this path, and a kernel that is not written yet is not replaced by one.

`madmetal --qwen38-flash-next bf16` runs this same forward on the Hugging Face BF16 checkpoint. Each linear map is a float32 shader dot of the exact stored value. Dense matrices are uploaded once. An expert and an n-gram row are read from the shards when that token needs them. The command isolates the formula from the int4 record. Thinking stays off unless `--thinking` is set, matching the server. `--raw` tokenizes the prompt with no chat template. `complete` accepts the same flag and reads the int4 record. Shard reads bypass the file cache. An expert's float weights leave the cache at the end of that layer. The command buffer that used them stays open until the next expert-id read. `serve` and `complete` still read the record and do not open those shards.

`madmetal --qwen38-flash-next quantize` writes the record from the Hugging Face BF16 shards. That command is the only place a weight is quantized. The same command continues a partial output: `quantize_progress.json` records the resident catalog, each finished expert layer, the MTP expert file, and the n-gram row. Inside a layer, the last finished expert is recorded every 64 experts, and the next run writes the rest of that file without clearing the experts already stored. The n-gram cursor is pulled back to the bytes present in `ngram.bin` when the checkpoint is ahead of the file. `manifest.json` is written last, with `complete: true`. `serve` and `complete` open the record and do not open those shards. The completion that returned before this file existed was packed during the request. The router stays float16 and uses its own shader dot. The token embedding is one int4 row, gathered and reconstructed. Sequences at or under the 2048-token budget attend every token, which is the QSA rule when every complete block fits in the budget. A longer sequence stops, because the sparse block mask is not a kernel in this path yet.

A prompt chunk whose token count is greater than one runs an int4 map through `mpp::tensor_ops::matmul2d` when the row count is a multiple of 32 and the inner dimension is the stored width. The right operand is the quantized weight with its E8M0 scale plane. That plane requires `transpose_right`. The tile is 64 by 32, four simdgroups per threadgroup. The activation is float16. The destination is float32, which is what the measured kernel writes. The batch is padded up to a multiple of 64 and the padded rows are dropped. A matrix that misses either condition uses a batched shader dot. A single-token int4 map uses `gemv_int4_f32`: four rows per simdgroup, each lane reading its own 32-wide block, with the SiLU and sigmoid epilogues in that kernel.

Decode int4 maps are that GEMV. Gated DeltaNet is one convolution scan and one state scan for the chunk. Sparse attention projects the chunk, then one attend kernel. Routed experts in a prompt chunk are grouped so one expert's weights are applied to the tokens that selected it. A decode token reads the ten ids, stages any missing run, and then runs the gate and down kernels. Layer scratch comes from one arena that is rewound after the layer. The residual stays in its own buffer. RMSNorm, SiLU, sigmoid, softplus, L2 norm, RoPE, both depthwise convolutions, residual adds, the branch mean, and the router softmax and top-10 are shaders in that same command buffer. Hyper-connection norms are Gemma-style, `x / rms · (1 + w)`. The GDN head norm is the multiplicative form, `x / rms · w`, with the sigmoid output gate applied after it. Q and K are L2-normalized, and Q is then scaled by `1/√128`. The attention query projection splits into the query and a sigmoid gate. Activations stay in device buffers. The dependence is the embedding, then for each of the 48 layers a residual read, a mixer, a residual write, the router, the shared expert, and up to ten routed experts. Layer 1 adds the n-gram injection into the four residual branches before its mixer. The output head runs once, on the last token of the chunk. MTP does not run. Under the BF16 default, every sparse-attention token's K and V are kept, and the indexer key is kept beside them. The fp8 KV switch is not selected by the server. The log reports tokens per second for the prompt forward, then for the single-token forwards that follow it. A prefix hit that steps nothing logs prefill as zero tokens. `complete`, `bf16`, and `serve` accept `--transcript`. That file receives the prompt and the completion. Token rates stay on the console.

The host commits a command buffer when the next load needs a value that buffer will produce. These steps stay on the host:

- The chat template and the tokenizer emit token ids. Those ids are copied into the buffer the embedding gather reads.
- The n-gram mix is sixteen wrapping int64 XORs of token ids the host already stores. Each result selects a 128-byte slot. The request path loads the 16 KB page that contains that slot through Metal IO into a small page cache, and a shader reconstructs the row. The multipliers, head offsets, and vocabulary sizes are 280 bytes of int64 at the tail of `resident.bin`, copied out of that buffer once after it loads.
- The router shader writes ten expert ids and ten weights. The host reads the ten ids, once per layer, and stages any missing run before that layer's expert kernels. The weights stay on the device. That read is what ends the command buffer.
- Sampling returns one token id. Temperature at or below zero is a GPU argmax. A positive temperature builds the probability vector and walks the CDF on the GPU; the host passes in one uniform draw. The id is the stop check, the prefix-cache key, and the next step's token.
- The converter packs on the host because that command writes the file the shaders read.
- `bf16` reads a shard row for an expert or an n-gram vector, uploads it, and then uses these shaders. `serve` and `complete` do not open those shards.

On this M5 Pro, a 58-token chat prefill whose experts are already in the atlas runs at about 120 tokens per second. The same prefill cold streams about 18 GiB and runs at about 17 tokens per second. The tokens that follow still request experts the prompt did not, on the order of 100–300 MiB each, and those steps run at about 10 tokens per second. One of those steps spends about 60–90 ms on the GPU.

## Measured block of 32

`madmetal --qwen38-flash-next probe` on this M5 Pro, Metal 4.1. Host `replaceSlice` and `getBytes` copy packed bytes: 32 int4 values are 16 bytes, 32 int2 values are 8 bytes, and int8 / fp8 values are one byte each.

int4 element 0 is the low nibble. Shader `unpack` of `int4b` is signed two's complement, −8..7. Code 7 is 7, code 8 is −8, code 15 is −1. An E8M0 scale of 130 reads back as 130, and `code × 2^(e − 127)` matches the shader (130 is ×8).

`madmetal --qwen38-flash-next matmul` runs one 64×32×32 tile through `matmul2d` and the same rows through the decode shader dot. The activation is 1 on the first inner column, each weight row stores `(n mod 7) − 3` there, and the E8M0 byte is 130. Both kernels return that code times 8. A second prefill, 128×64×64, covers two row tiles and two column tiles and matches the reconstructed int4 product. The column tile walks the weight's outer extent. Offsetting that slice along K leaves every output past the first 32 columns wrong, and a single 32-column tile does not catch it.

fp8 E4M3 matches OCP E4M3FN, including `0x7E = 448` and `0x7F` NaN. fp8 E5M2 matches OCP E5M2. int8 bytes round-trip, including −128 and 127.

int2 bytes round-trip. Metal 4.1 `unpack` does not accept `int2b`, and this SDK's compute encoder has no tensor argument, so int2 element order and its signed endpoints are not known yet.

## Build order

1. The block-of-32 probe above is the int4 measurement. int2 element order against Apple's `unpack` is still open; the shader-dot layout is low two bits first. Prefill `matmul2d` is in Metal Performance Primitives. One 64×32×32 int4 tile with an E8M0 scale matches the signed dot on this GPU, and a 128×64×64 prefill matches the reconstructed product across two row tiles and two column tiles.
2. The profiler writes the activation file from this app's BF16 Metal prefill. The prompt strings come from that bucket JSON.
3. The converter writes the native records with the E8M0 rule above. Packing is the host writing the file the shaders read.
4. `serve` steps the shaders in the schedule and returns those tokens from `POST /v1/chat/completions`. A second request that extends the stepped prefix does not repeat that span. Hidden-state error against a BF16 prefill is not measured yet.
