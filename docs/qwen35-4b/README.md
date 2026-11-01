# Qwen3.5-4B

As-built note for the `qwen35-4b` language pipeline. The official checkpoint is [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache License 2.0). `favlo serve` on a terminal with an empty registry runs this pipeline's `setup`. The packed directory is `models/Qwen3.5-4B-fp8`. Papers and config fields are [references.md](references.md). Setup commands are [SETUP.md](SETUP.md).

`handle_responses` is a `LanguagePipeline`. `handle_chat` translates a Chat Completions body into that request. Clients reach this pipeline through `favlo serve` once the record is registered.

```text
uv run favlo --qwen35-4b --help
uv run favlo --qwen35-4b kernels
uv run favlo --qwen35-4b complete
```

## Memory

Device spans come from `favlo.memory`. The serve process keeps the ledger. This runtime asks for the budget at open and again at the start of each generate. A `--max-memory` cap is applied by that ledger, not by this package. `open_runtime` calls `memory.set_max_memory` then opens; it does not keep a private parser. Spans go through `device_buffer`. The live full-attention cache length is the one knob: it shrinks at open until the resident record, Gated DeltaNet state, convolution history, workspace, and that cache fit. A 2048-token cache that does not fit refuses to open. `open_runtime` takes `owner`; every allocate in that opening uses it; `close` sweeps that string.

## Architecture

Dense hybrid text backbone: 24 Gated DeltaNet layers and 8 full-attention layers (`[linear × 3, full] × 8`), SwiGLU feed-forward, tied embeddings, and a vision tower in the same packed record. Hidden size 2560, intermediate 9216, vocab 248320. Full attention uses 16 query heads with a fused output gate, 4 KV heads, head dim 256, partial RoPE (64 of 256), and M-RoPE. Gated DeltaNet uses 16 key heads, 32 value heads, head dim 128, and a depthwise conv of width 4.

Weights are fp8 E4M3 with an E8M0 scale every 32 along K. Norms, `A_log`, `dt_bias`, convolution taps, and Q/K gains stay float16. Layer / Q / K / final RMSNorm apply gain as `1 + w`. The GDN gated norm applies `w` then SiLU of `z`. The eight full-attention layers keep a key/value cache. The registry field is `kv_cache`. `f16` is the default (two bytes per element, no scale plane). `e4m3` remains the fp8 cache. Prefill is a 1024-token Metal tile: `matmul_fp8` / `matmul_fp8_n32`, `gdn_chunk`, and `attend_prefill` or `attend_prefill_f16`. Only the last chunk runs the head. Decode stays token-by-token. Kernel ceilings are [ceilings.md](ceilings.md).

The vision tower is packed beside the language weights and runs for a still image. Its matrices are fp8 E4M3 with an E8M0 scale every 32 along K. LayerNorm weights, LayerNorm biases, and the other vision biases stay float16. The Conv3d patch embed is stored as `[1024, 1536]`. The position table is `[2304, 1024]`. Each block is LayerNorm, full attention, and tanh GELU. The merger writes 2560-wide rows, and prefill overwrites the image-pad tokens with those rows. A repeated image reuses the embedding. Video is rejected. An image prompt uses interleaved M-RoPE on the three axes; decode adds the position delta from that table.

The packed record's `chat_template.jinja` is the official Qwen3.5 template (tools and `enable_thinking`). MiniFantasy's short template is not used for serve. The model card's multi-turn rule is that history keeps the final output and leaves the thinking content out. The template already does that for an assistant turn at or before the latest user message, and this pipeline drops that reasoning before render. An assistant turn after that user message, such as a tool step in the current turn, still carries `reasoning_content`.

## Sampling

Variants in `sampling.py`: `thinking` (default) and `instruct`. Registry context length is 262144. Temperature 0 is argmax. A presence penalty subtracts once per distinct completion token already drawn. First ship does not run nucleus / top-k on device yet. A thinking request that omits `max_output_tokens` runs until a stop id or the last context position. An instruct request that omits it stops at 2048 tokens. An explicit cap applies to both.

## Measured carrot smoke

Pipeline smoke under the smoke lock (`tests/test_qwen35_4b_smoke.py`), fp8 record, e4m3 cache (2 November 2026):

| Phase | Rate | Tokens | Seconds |
|---|---|---|---|
| prefill | 1014.25 tok/s | 51 | 0.05 |
| decode (instruct card) | 44.95 tok/s | 105 | 2.34 |
| decode (thinking card) | 43.34 tok/s | 558 | 12.88 |

A 2012-token prompt on that e4m3 cache prefills at 1060.88 tok/s (1.90 s). The prefill kernel ceiling for a 256-token chunk of the e4m3 attend path is 1007.60 tok/s ([ceilings.md](ceilings.md)).

`complete` on the default f16 cache, same 51-token prompt (2 November 2026): prefill 897.90 tok/s (0.06 s), decode 44.81 tok/s (64 tok in 1.43 s).

Coding agent on the e4m3 cache printed `prompt tokens` / `cache miss` per request (2 November 2026):

| Card | Prefill | Decode |
|---|---|---|
| thinking | 993.10 tok/s (25781 tok in 25.96s) | 37.64 tok/s (2384 tok in 63.34s) |
| instruct | 1049.04 tok/s (23058 tok in 21.98s) | 40.21 tok/s (1863 tok in 46.33s) |

Thinking prefill is 98.6% of the 1007.60 tok/s kernel ceiling ([ceilings.md](ceilings.md)).

Progress uses `favlo.console` begin / end / note on the console thread (`prompt`, `cache`, load / prefill / decode). Decode streaming uses a separate queue thread. NumPy stays off the hot path.
