# Spark-X2.6-1.7B

Served language pipeline for [XHToken/Spark-X2.6-1.7B](https://huggingface.co/XHToken/Spark-X2.6-1.7B). The record directory is `Spark-X2.6-1.7B-fp4`. The flag is `spark-x26-1-7b`.

The language matrices are fp4 E2M1 with an E8M0 scale every 32 along K. `g_proj` and the RMSNorm gains stay float16. The live key and value cache is fp8 E4M3. Full attention grows from 2,048 tokens. Sliding attention keeps a ring of 512. The sampling card stores a context of 1,048,576. The opening keeps the longest full-attention length that fits `memory.budget()` and refuses to open when a 2,048-token cache does not fit. NumPy is not on prefill, decode, or sampling. The sampled token id comes back as four bytes.

A chunk of up to 64 tokens is one command buffer. That chunk is projected by the 32×128 fp4 matmul; only the valid rows are rotated, stored, and attended. Keys in that chunk are one threadgroup per token. Values in a 32-token block are quantized once for the block. Attention is one threadgroup per query head per token. A decode step longer than 64 keys splits those keys into tiles of 64. Decode reads eight fp4 weights per thread. RMSNorm is folded into the QKV, gate, and up projections, and the output and down projections add into the residual. The vocabulary projection keeps the activation in threadgroup memory and writes 32 rows from each threadgroup. Isolated kernel times are in [ceilings.md](ceilings.md).

`high` is the default variant. `low` and `none` share the sampler and change the chat template. `enable_thinking` false selects `none` when the request does not name a variant. The stop id is 1. `<think>` stays in `reasoning_content`. `<tool_call>` becomes `tool_calls`. A tool call before `</think>` ends the trace.

A prefix file stores the packed planes. A file as long as the prompt is not installed, because the next token still has to be scored.

```text
uv run favlo --spark-x26-1-7b download
uv run favlo --spark-x26-1-7b quantize
uv run favlo --spark-x26-1-7b kernels
```

Setup steps are in [SETUP.md](SETUP.md). Papers and the checkpoint fields are in [references.md](references.md).
