# Qwen3.5-4B

As-built note for the `qwen35-4b` language pipeline. The official checkpoint is [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache License 2.0). The local BF16 source for the first pack was the same-tensor MiniFantasy checkout copied to `models/Qwen3.5-4B-bf16`, then quantized on SSD into `models/Qwen3.5-4B-fp8`. Papers and config fields are [references.md](references.md). Setup commands are [SETUP.md](SETUP.md).

`handle_responses` is a `LanguagePipeline`. `handle_chat` translates a Chat Completions body into that request. Clients reach this pipeline through `favlo serve` once the record is registered.

```text
uv run favlo --qwen35-4b --help
uv run favlo --qwen35-4b kernels
uv run favlo --qwen35-4b complete
```

## Memory

Device spans come from `favlo.memory`. The serve process keeps the ledger. This runtime asks for the budget at open and again at the start of each generate. A `--max-memory` cap is applied by that ledger, not by this package. `open_runtime` calls `memory.set_max_memory` then opens; it does not keep a private parser. Spans go through `device_buffer`. The live full-attention cache length is the one knob: it shrinks at open until the resident record, Gated DeltaNet state, convolution history, workspace, and that cache fit. A 2048-token cache that does not fit refuses to open. `open_runtime` takes `owner`; every allocate in that opening uses it; `close` sweeps that string.

## Architecture

Dense hybrid text backbone: 24 Gated DeltaNet layers and 8 full-attention layers (`[linear × 3, full] × 8`), SwiGLU feed-forward, tied embeddings, optional MTP and vision tower (not packed in the first language record). Hidden size 2560, intermediate 9216, vocab 248320. Full attention uses 16 query heads with a fused output gate, 4 KV heads, head dim 256, partial RoPE (64 of 256), and M-RoPE. Gated DeltaNet uses 16 key heads, 32 value heads, head dim 128, and a depthwise conv of width 4.

Weights are fp8 E4M3 with an E8M0 scale every 32 along K. Norms, `A_log`, `dt_bias`, convolution taps, and Q/K gains stay float16. Layer / Q / K / final RMSNorm apply gain as `1 + w`. The GDN gated norm applies `w` then SiLU of `z`. The eight full-attention layers keep an fp8 key/value cache. Prefill is a 1024-token Metal tile: `matmul_fp8` / `matmul_fp8_n32`, `gdn_chunk`, and `attend_prefill`. Only the last chunk runs the head. Decode stays token-by-token. Kernel ceilings are [ceilings.md](ceilings.md).

The packed record's `chat_template.jinja` is the official Qwen3.5 template (tools and `enable_thinking`). MiniFantasy's short template is not used for serve.

## Sampling

Variants in `sampling.py`: `thinking` (default) and `instruct`. Registry context length is 262144. Temperature 0 is argmax. A presence penalty subtracts once per distinct completion token already drawn. First ship does not run nucleus / top-k on device yet.

## Measured carrot smoke

Pipeline smoke under the smoke lock (`tests/test_qwen35_4b_smoke.py`), fp8 record on device (2 November 2026):

| Phase | Rate | Tokens | Seconds |
|---|---|---|---|
| prefill | 1014.25 tok/s | 51 | 0.05 |
| decode (instruct card) | 44.95 tok/s | 105 | 2.34 |
| decode (thinking card) | 43.34 tok/s | 558 | 12.88 |

A 2012-token prompt prefills at 1027.68 tok/s (1.96 s). The prefill kernel ceiling for a 256-token chunk is 1024.64 tok/s ([ceilings.md](ceilings.md)).

Coding agent printed `prompt tokens` / `cache miss` per request:

| Card | Prefill | Decode |
|---|---|---|
| thinking | 909.40 tok/s (25736 tok in 28.30s) | 38.04 tok/s (2321 tok in 61.01s) |
| instruct | 968.42 tok/s (23058 tok in 23.81s) | 39.43 tok/s (1920 tok in 48.69s) |

Progress uses `favlo.console` begin / end / note on the console thread (`prompt`, `cache`, load / prefill / decode). Decode streaming uses a separate queue thread. NumPy stays off the hot path.
