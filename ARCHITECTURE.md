# MADMetal architecture

As-built contract for **MADMetal**. The name expands to MADMetal Accelerated Decoding on Metal.

One model is one pipeline. `madmetal --qwen38-flash-next` loads `qwen38-flash-next.py`, which loads the `qwen38_flash_next` package. A future model is a new entry file, a new package, and a new flag. It does not share this forward, these records, or these kernels.

The target machine is an M5 Pro: 20 shader cores, a neural accelerator beside the shader pipelines in each core, 307 GB/s unified DRAM, and an APFS volume whose device block and allocation block are 4096 bytes. Requires macOS 27 and the matching Metal SDK. The 16-core Neural Engine is not used. Sub-byte tensor planes also require 128-byte alignment inside the record. A 4 KB file offset does not by itself satisfy that.

Layouts are in [docs/qwen38/record.md](docs/qwen38/record.md). The n-gram hash is in [docs/qwen38/ngram.md](docs/qwen38/ngram.md). The kernel schedule is in [docs/qwen38/schedule.md](docs/qwen38/schedule.md). The probe is in [docs/qwen38/probe.md](docs/qwen38/probe.md). This file wins if one of those notes disagrees with it.

## In scope

- `qwen38_flash_next` runs three stages for this model: a BF16 calibration prefill, a converter from Hugging Face BF16 shards to native records, and the text runtime. `qwen38-flash-next.py` is the entry the command loads.
- One sequence at a time. Prefill and decode. Layered kernels.
- `GET /v1/models` and `POST /v1/chat/completions`, including streaming.
- The checkpoint chat template. Sampling is temperature, top-p, and top-k.
- Tool calls. The template renders `tools` and tool results. A completion that contains `<tool_call>` blocks is returned as OpenAI `tool_calls`, on the JSON response and as stream deltas. While a function name is open, the sampler keeps only tokens that continue a name from that request. A `tool_choice` that names one function forces the opening `<tool_call><function=name>` before the arguments. Those opening tokens are the tokenizer's encoding of that text, so a later render of the same opening is the same ids. A function with no parameters forces the closing tags the same way. Generation stops when that call's `</tool_call>` has closed.
- A prefix cache so a continuation prefills only the new tokens.
- Expert streaming from SSD into a resident atlas. Consecutive expert ids are one sequential Metal IO. The shader reads the live image from that atlas.

## Out of scope

- A generic runner and any edit to `qwen38_flash_next` on behalf of another model.
- Affine group-64 packing and per-layer bit allocation.
- Batching and the Neural Engine.
- The vision tower. It stays unloaded.
- Multi-token prediction during serving. MTP experts are packed. The endpoint decodes one token at a time, because a draft token would have to pass the same grammar mask.
- Fusing a whole layer into one kernel. The runtime is a layered loop. Shaders share a command buffer until the host needs an expert file offset or the sampled token id.
- fp4 and fp8 as expert data planes. Further key/value widths: fp8 E5M2, unscaled fp8, int8, int4, fp4, int2, and a wider K than V.
- Tool-call grammars on speculative drafts.
- A dependency on another inference app: its packages, its virtualenv, or its files. The README credit is a link, not a library.
- A second, slower forward. Host GEMV and CPU matmul of activations are not stand-ins for a missing shader. The converter's packing runs on the CPU because it writes the file. The request path does not.

## Qwen3.8-Flash-Next

48 layers, hidden 2560. Twelve repeats of three Gated DeltaNet layers and one Qwen Sparse Attention layer, each followed by MoE. The router is 512-wide, top 10, plus one shared expert. Expert intermediate size is 640. The token embedding and the output head are separate untied matrices, vocabulary 248320.

Gated DeltaNet is 36 layers and keeps a fixed 128×128 state per value head. Sparse attention is 12 layers: K and V for 2 heads of 256 channels, and a 128-channel indexer key per token. The indexer pools keys in blocks of 4, then picks up to 512 blocks (2048 tokens) plus the unfinished tail. MTP is one extra sparse-attention layer with its own 512 experts. Serving samples one token at a time.

The n-gram table is injected at decoder layer 1, before that layer's residual read. The hash, the branch gate, and the residual `1/n_r` read are in [docs/qwen38/ngram.md](docs/qwen38/ngram.md).

## Records

Record version 4. Three storage planes: resident int8 for the super-expert pin, int4 for the hot keep set, int2 for the cold complement. A narrow row stays narrow. A middle-tier expert can stay on SSD. `num_experts` stays 512. `h384` in the directory name is the keep-set width, not the per-layer hot count. Do not pad an expert count to a multiple of 32. The matmul tile of 32 is a kernel constraint, not a count constraint.

`quantize` is the only command that quantizes a weight. `serve` and `complete` read the record and do not open the BF16 shards.

Image sizes, the five super ids, the directory name, and the E8M0 rule are in [docs/qwen38/record.md](docs/qwen38/record.md).

## Key/value cache

`serve`, `complete`, and `bf16` take `--kv-cache e4m3` or `--kv-cache bf16`. The default is `e4m3`.

E4M3 stores each completed block of four K and V tokens as `metal_fp8_e4m3` with an E8M0 scale every 32 channels along the head. The scale rounds up so the peak of the block fits in the finite E4M3 range, whose largest value is 448. The unfinished tail, at most three tokens, stays BF16. K and V for a completed token are 1,056 bytes, 12,672 bytes across the twelve layers.

`--kv-cache bf16` stores every K and V token as BF16, including the tail. That is 24,576 bytes per token across the twelve layers.

Indexer keys stay BF16 in both modes. At 262,144 tokens they add 0.75 GiB. The live prefix keeps the tail, and the next token of the same sequence reads it. Pooling is mean, then cast back, then norm, then rotary, in that order.

## Context

The prompt length is the number of tokens that fit in memory, and at most 131,072. The reservation is the resident hot-expert budget, the weights that stay mapped, and the scratch for one 2048-token chunk. What remains is divided by the bytes the cache stores per token. Prefill also holds the residual and the two layer staging buffers, and one chunk of indexer block scores. The limit uses the width of the buffers the runtime allocates for K, V, and the indexer key. A machine that cannot hold 131,072 tokens keeps the shorter fit.

`serve` and `complete` take `--resident-percent`. That is the atlas budget as a percent of the hot-tier bytes, page-aligned, stepping down by 4 GiB until 8 GiB or the device accepts the buffer. Omitting it keeps the 24, 20, 16, 12, 8 GiB ladder. The prompt limit reserves the first candidate, so a prompt that was accepted still fits when the atlas is allocated. A prompt past the limit is refused. Generation stops when the next token would pass it.

The indexer attends at most 512 complete blocks of 4 tokens, plus the unfinished tail. A visible span that fits in the dense attend kernel attends every token. 2048 is that attended window, not a cap on the prompt.

## Prefix cache

One request at a time. A hit is an exact token prefix that was stored, after the checkpoint template has been applied. The usual hit is the previous turn of the same chat. Two chats that share a system prompt hit only when that prompt was stored as its own prefix. The stored length is the real token count, including an unfinished E4M3 tail. This process does not write a prefix file. A persisted boundary would end on a completed block of four.

Each cached prefix stores, at that boundary:

- K and V for every cached token in the active format, and the indexer key for every cached token
- the Gated DeltaNet recurrent state once, not once per token
- the Gated DeltaNet depthwise convolution history (kernel 4), once
- the n-gram convolution state and the tokens already stepped, once

Grammar state is not cached. It starts again on the new generation. When RAM is short, the starting policy is to drop prefix-cache entries before the resident expert tiers. That policy is a default to be measured: a short prefix can be cheaper to prefill again than a cold expert read.

## Endpoint

`madmetal --qwen38-flash-next serve` listens on `127.0.0.1:8129` for `GET /v1/models` and `POST /v1/chat/completions`, including a streaming response. The process serves one request at a time. A connection that arrives while a request is open waits in the listen queue until that request has returned, including its error response. The decode loop enqueues each sampled id and continues to the next step. Once a decode token's routed experts were resident, the next token leaves the ids on the device and reads them once at the end of the token. A miss loads the missing experts and replays the token. A side thread detokenizes those ids and writes the events. Tokens inside the open think block stream as `reasoning_content`. Tokens after `</think>` stream as `content`. Messages are tokenized with the checkpoint chat template. A request may carry `tools`, and the template renders them. The sampler applies temperature, top-p, and top-k. A token span that has already been emitted three times is removed from the next draw. A 12-token span that has already occurred three times, after 512 tokens of an open think, closes that block. A recent window that uses only a handful of distinct tokens is removed as well. While thinking is on, the close tag stays out of the draw until 8 tokens have been written. Until the close tag, and for 16 tokens after it, the end-of-turn ids stay out of the draw. A think that runs 64 tokens without that tag may end. A request may set `presence_penalty` or `frequency_penalty`, and `penalty_window`. Presence is subtracted once from the logit of each distinct token in that window. Frequency is subtracted once for every copy in the same window. The window defaults to the last 20 tokens. A window of 0 uses the whole completion. Omitting either penalty leaves it at 0, which is the thinking card. A request that omits a field uses the Qwen3.8-Flash-Next model card for the active mode. Thinking, used when `enable_thinking` is omitted or true, is temperature 1.0, top-p 0.95, top-k 20. Instruct, when `enable_thinking` is false, is temperature 0.7, top-p 0.8, top-k 20. A field the request sets is kept. Temperature 0 is an argmax. `complete` and `bf16` follow the same rule from `--thinking`. Generation stops on the tokenizer end-of-turn id and on the text-config eos. A request that omits `max_tokens` and `max_completion_tokens` runs until one of those ids. A request that sets either field stops after that many tokens. Either way, generation stops when the next token would pass the limit in Context. The n-gram segment boundary stays the text-config eos. A completion that contains `<tool_call>` blocks is returned with `finish_reason` `tool_calls`. While the function name is still open, the next token has to continue one of the names in `tools`. A `tool_choice` that names one function forces that call's opening before the arguments are sampled. The parameter text is not masked. A request that omits `enable_thinking` leaves `<think>` open, so decode writes the reasoning. Setting it false closes that block in the prompt, and decode starts after it. Tool results on the next turn are `role: tool` messages, and the cache key is the token ids the template actually emitted. The server keeps one stepped prefix. A request whose ids extend that prefix continues; any other prefix clears the state and steps from the start.

The checkpoint's tool-call syntax is `<tool_call><function=name><parameter=name>value</parameter></function></tool_call>`. The JSON response includes `created` as unix seconds and puts those calls on `message.tool_calls` with the arguments as a JSON object string. A streaming response emits the text before the call as content, then one `tool_calls` delta per call. Text with no call stays `message.content`.

## Calibration

The profiler does not quantize a weight. It runs this app's BF16 prefill. The default collect is 128 sequences of 512 tokens, drawn from prompt strings in a bucket JSON (`tool_calling`, `chat`, `mixed`, `reasoning`, `code`, `en`, `ko`, `zh`, `ja`). That file is prompt text. The citations are [REFERENCES.md](REFERENCES.md).

Only the strings matter. A later corpus file can use another layout. `tool_calling` and `chat` are already wrapped in `<|im_start|>` turns, so those strings are tokenized as they stand. The other buckets are raw text. The activation file is rebuilt when the strings or the tokenizer change.

For each routed expert it records how many tokens selected it, the input energy to `gate_up` along K = 2560, and the input energy to `down_proj` along K = 640. An expert only accumulates tokens that routed to it. Each dense projection stores one energy vector. The router, the n-gram table, and the key/value cache are not in the file. Tier lists are copied from selection so a hot block cannot promote a narrow expert.

The collect signature is the corpus hash, the sample count, the sequence length, and the tier assignment. Resume is per sequence.

## Schedule

Prefill and decode are Metal shaders. A missing kernel stays missing. A prompt call streams each layer's hot and cold files into one of two staging buffers and does not read the atlas. An atlas built by an earlier decode stays allocated when the staging buffers fit beside it. A cold start, or a device that refuses those buffers, allocates the atlas after the call. A prompt longer than one 2048-token chunk runs each int4 expert that received at least 64 tokens through matmul2d, once for the whole layer. The int2 tier, the int8 hot tier, and smaller int4 groups stay on the grouped shader for that width. A manifest without `hot_bits` and `cold_bits` is the int4 hot file and the int2 cold file. Decode of an int8 hot expert is the int8 GEMV on the atlas image. Decode stages missing expert runs and uses a GEMV for one token. Scratch for a prompt chunk is 2 MiB per token, or 256 MiB, whichever is larger. On the n-gram layer, a longer prompt finishes that chunk's page loads before the next chunk rewinds this arena. The half × int8 `matmul2d` dispatch does not complete on this GPU, so it is not on the request path.

The kernel map, the quantize workers, and the host steps are in [docs/qwen38/schedule.md](docs/qwen38/schedule.md).

## Probe

`madmetal --qwen38-flash-next probe` measures one block of 32 on this GPU before an expert table is packed. Nibble order and the int4 endpoints are in [docs/qwen38/probe.md](docs/qwen38/probe.md). `matmul` checks one int4 E8M0 tile against the decode shader dot. It does not load the checkpoint.

## Build order

1. The block-of-32 probe is the int4 measurement. int2 element order against Apple's `unpack` is still open.
2. The profiler writes the activation file from this app's BF16 Metal prefill. The prompt strings come from that bucket JSON.
3. The converter writes the native records with the E8M0 rule in [docs/qwen38/record.md](docs/qwen38/record.md). Packing is the host writing the file the shaders read.
4. `serve` steps the shaders in [docs/qwen38/schedule.md](docs/qwen38/schedule.md) and returns those tokens from `POST /v1/chat/completions`. A second request that extends the stepped prefix does not repeat that span. Hidden-state error against a BF16 prefill is not measured yet.
