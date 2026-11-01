# Spark-X2.6-1.7B

Served language pipeline for [XHToken/Spark-X2.6-1.7B](https://huggingface.co/XHToken/Spark-X2.6-1.7B). Those weights are under the Apache License, Version 2.0. The record directory is `Spark-X2.6-1.7B-fp8`. The flag is `spark-x26-1-7b`. `favlo serve` on a terminal with an empty registry runs `setup` for this checkpoint. The steps are [SETUP.md](SETUP.md).

`weights.bin` is loaded once into one Metal buffer, in 8 MiB pieces. Each matrix is a byte offset in that buffer. The language matrices are fp8 E4M3 with an E8M0 scale every 32 along K. `g_proj` and the RMSNorm gains stay float16. The live key and value cache is fp8 E4M3. Full attention grows from 2,048 tokens. Sliding attention keeps a ring of 512. The sampling card stores a context of 1,048,576. The opening keeps the longest full-attention length that fits `memory.budget()` and refuses to open when a 2,048-token cache does not fit. NumPy is not on prefill, decode, or sampling. The sampled token id comes back as four bytes.

A chunk of up to 64 tokens is one command buffer. That chunk is projected by the 32×128 fp8 matmul; only the valid rows are rotated, stored, and attended. Keys in that chunk are one threadgroup per token. Values in a 32-token block are quantized once for the block. Attention is one threadgroup per query head per token. A decode step longer than 64 keys splits those keys into tiles of 64. Decode reads eight fp8 weights per thread. Each projection is 32 rows from one 256-thread group, and that group keeps the activation in threadgroup memory. RMSNorm is folded into the QKV, gate, up, and vocabulary projections. The output and down projections add into the residual. Isolated kernel times are in [ceilings.md](ceilings.md). The isolated sums are in [research/kernel-ceilings.md](research/kernel-ceilings.md). The recorded prefill and decode runs are in [research/recorded-forward.md](research/recorded-forward.md).

`POST /v1/responses` is the generation path. `POST /v1/chat/completions` is translated into that body, and the Response is translated back. `reasoning.effort` selects the template. `none` turns thinking off. `minimal` and `low` use the low instruction. `medium`, `high`, `xhigh`, and `max` use the high instruction. An omitted effort uses the id's variant, or `high`. The three variants share one sampler. A request may set `temperature` and `top_p`. `top_k`, presence, and frequency stay on the card. The stop ids are 0, 1, and 2: the start-of-sentence marker, the end-of-sentence marker, and pad. None of them is copied into the message. `<think>` is a reasoning item. `<tool_call>` is a `function_call` item. A tool call before `</think>` ends the trace. `tool_choice` `none` leaves the call out. A named `tool_choice` starts that call. `input_image` is an invalid request. This model has no draft module, so `favlo:mtp` does not change the draw. Text deltas carry an empty `logprobs` list. Sampled ids are queued, and a side thread decodes them and writes the events. The decode loop does not wait on that thread. The prefix cache is the prompt-cache cut.

A prefix file stores the packed planes. A file as long as the prompt is not installed, because the next token still has to be scored.

```text
uv run favlo --spark-x26-1-7b download
uv run favlo --spark-x26-1-7b quantize
uv run favlo --spark-x26-1-7b kernels
```

Setup steps are in [SETUP.md](SETUP.md). Papers and the checkpoint fields are in [references.md](references.md).
