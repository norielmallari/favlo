# Qwen3.5-4B

As-built note for the `qwen35-4b` language pipeline. The official checkpoint is [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache License 2.0). `favlo serve` on a terminal with an empty registry runs this pipeline's `setup`. The packed directory is `models/Qwen3.5-4B-fp8`. Papers and config fields are [references.md](references.md). Setup commands are [SETUP.md](SETUP.md).

`handle_responses` is a `LanguagePipeline`. The listener translates a Chat Completions body into that request. Clients reach this pipeline through `favlo serve` once the record is registered.

```text
uv run favlo --qwen35-4b --help
uv run favlo --qwen35-4b kernels
uv run favlo --qwen35-4b complete
```

## Memory

Device spans come from `favlo.memory`. The serve process keeps the ledger. This runtime asks for the budget at open and again at the start of each generate. A `--max-memory` cap is applied by that ledger, not by this package. `open_runtime` calls `memory.set_max_memory` then opens; it does not keep a private parser. Spans go through `device_buffer`. The live full-attention cache length is the one knob: it shrinks at open until the resident record, Gated DeltaNet state, convolution history, workspace, and that cache fit. A 2048-token cache that does not fit refuses to open. `open_runtime` takes `owner`; every allocate in that opening uses it; `close` sweeps that string.

## Architecture

Dense hybrid text backbone: 24 Gated DeltaNet layers and 8 full-attention layers (`[linear × 3, full] × 8`), SwiGLU feed-forward, tied embeddings, and a vision tower in the same packed record. Hidden size 2560, intermediate 9216, vocab 248320. Full attention uses 16 query heads with a fused output gate, 4 KV heads, head dim 256, partial RoPE (64 of 256), and M-RoPE. Gated DeltaNet uses 16 key heads, 32 value heads, head dim 128, and a depthwise conv of width 4.

Weights are fp8 E4M3 with an E8M0 scale every 32 along K. Norms, `A_log`, `dt_bias`, convolution taps, and Q/K gains stay Metal `half`. Layer / Q / K / final RMSNorm apply gain as `1 + w`. The GDN gated norm applies `w` then SiLU of `z`. The eight full-attention layers keep a key/value cache. The registry field is `kv_cache`. `fp8_e4m3` is the default (Metal `metal_fp8_e4m3`). `half` remains accepted (Metal `half`, two bytes per element, no scale plane). Both live planes are token-major `[position, kv_head, channel]` (with an E8M0 scale plane for `fp8_e4m3`). The SSD prefix store under `~/.favlo/prefix/qwen35-4b/` writes that same layout for positions `0 .. count-1` as regions `k.0`…`k.7` and `v.0`…`v.7`, plus `history` (convolution taps), `state` (Gated DeltaNet), and `rope` (the M-RoPE position delta). An exact-length file is not sampled; the next token is still stepped. Prefill is a 1024-token Metal tile. The FFN gate, up, and down planes and the GDN qkv, z, and out planes run int8: `restore_fp8_i8` collapses each fp8 weight row to int8 codes and one channel scale, `quantize_half_i8` does the same per activation row, `prefill_i8_n64` (`prefill_i8_m32` for down) is the int8×int8→int32 `matmul2d`, and `scale_i8_f32` rewrites the accumulators as floats (a 2012-token prompt prefilled at 1730 tok/s against 1520 tok/s on `matmul_fp8`, 2 November 2026). The GDN a/b and attention q/k/v/o planes stay on `matmul_fp8` / `matmul_fp8_n32`. The rest of the tile is `gdn_chunk`, and `attend_prefill` or `attend_prefill_half`. On both planes, `attend_prefill` / `attend_prefill_half` and `attend_decode` / `attend_decode_half` keep the softmax state in registers (no score plane), so a long prompt does not fall back to one-token attend. Only the last chunk runs the head. Decode stays token-by-token. Kernel ceilings are [ceilings.md](ceilings.md). Measured remainders for int8 prefill, the isolated dispatch sum, the harness rates, the page needle, and the vision coding agent are [research/int8-prefill.md](research/int8-prefill.md), [research/kernel-ceilings.md](research/kernel-ceilings.md), [research/recorded-forward.md](research/recorded-forward.md), [research/needle-page.md](research/needle-page.md), and [research/coding-agent.md](research/coding-agent.md).

The vision tower is packed beside the language weights and runs for a still image. Its matrices are fp8 E4M3 with an E8M0 scale every 32 along K. LayerNorm weights, LayerNorm biases, and the other vision biases stay Metal `half`. The Conv3d patch embed is stored as `[1024, 1536]`. The position table is `[2304, 1024]`. Each block is LayerNorm, full attention, and tanh GELU. The merger writes 2560-wide rows, and prefill overwrites the image-pad tokens with those rows. A repeated image reuses the embedding; rows also land under `~/.favlo/vision-cache/qwen35-4b/` so a later serve process can skip the tower. Video is rejected. An image prompt uses interleaved M-RoPE on the three axes; decode adds the position delta from that table.

The packed record's `chat_template.jinja` is the official Qwen3.5 template (tools and `enable_thinking`). MiniFantasy's short template is not used for serve. Historical reasoning stays in the prompt when `preserve_thinking` is on (the default, including `settings.json` omitted). That keeps a later turn a prefix of the tokens already applied, including the open `<think>` from a thinking generation prompt. `preserve_thinking` false matches the model card's multi-turn rule: history keeps the final output and leaves thinking out for assistant turns at or before the latest user message. An assistant turn after that user message, such as a tool step in the current turn, still carries `reasoning_content`. Tool schemas stay in the encoded prompt under `tool_choice` none so a later turn that re-enables tools still matches the stored prefix. A required named call forces the opening markup (or the whole call when that function takes no arguments), turns thinking off for that completion, and stops at the first `</tool_call>` so a second call is not drawn.

## Sampling

Variants in `sampling.py`: `thinking` (default) and `instruct`. Each card stores temperature, top-p, top-k, presence penalty, and `reasoning_effort` (`high` on thinking, `none` on instruct). Registry context length is 262144. Registration copies those cards and stores `enable_thinking` from the pipeline default. Temperature 0 is argmax. A presence penalty subtracts once per distinct completion token already drawn. First ship does not run nucleus / top-k on device yet. The official card exposes thinking only as `enable_thinking` on/off (no soft `/think` switch and no graduated effort ladder). On the Responses route, every listed `reasoning.effort` is accepted. Supported values are `none` and `high` (thinking on). An unsupported name steps up to the next supported value; when none is higher, it falls back to the next lower (`minimal` / `low` / `medium` → `high`; `xhigh` / `max` → `high`). A request effort wins and does not swap the sampling card. An omitted effort uses the selected variant's `reasoning_effort` (bare id and `:thinking` → `high`, `:instruct` → `none`), then the row field, then `enable_thinking`, then the pipeline default. Do not infer thinking from the suffix spelling alone. A thinking request that omits `max_output_tokens` runs until a stop id or the last context position. A non-thinking request that omits it stops at 2048 tokens. An explicit cap applies to both.

## Measured carrot smoke

Pipeline smoke under the smoke lock (`tests/test_qwen35_4b_smoke.py`), fp8 record, fp8_e4m3 cache (2 November 2026):

| Phase | Rate | Tokens | Seconds |
|---|---|---|---|
| prefill | 1014.25 tok/s | 51 | 0.05 |
| decode (instruct card) | 44.95 tok/s | 105 | 2.34 |
| decode (thinking card) | 43.34 tok/s | 558 | 12.88 |

A 2012-token prompt on that fp8_e4m3 cache prefills at 1060.88 tok/s (1.90 s). The prefill kernel ceiling for a 256-token chunk on the default half cache is 1061.69 tok/s ([ceilings.md](ceilings.md)). The earlier fp8_e4m3 attend path summed to 1007.60 tok/s.

`complete` on the default half cache, same 51-token prompt (2 November 2026): prefill 897.90 tok/s (0.06 s), decode 44.81 tok/s (64 tok in 1.43 s).

Coding agent on the fp8_e4m3 cache printed `prompt tokens` / `cache miss` per request (2 November 2026):

| Card | Prefill | Decode |
|---|---|---|
| thinking | 993.10 tok/s (25781 tok in 25.96s) | 37.64 tok/s (2384 tok in 63.34s) |
| instruct | 1049.04 tok/s (23058 tok in 21.98s) | 40.21 tok/s (1863 tok in 46.33s) |

Coding agent on the half cache, thinking on, no vision (2 November 2026): prefill 1393.99 tok/s (25747 tok in 18.47s), decode 39.96 tok/s (2028 tok in 50.75s).

After int8 prefill for FFN and GDN qkv/z/out (2 November 2026), the same coding agent printed prefill 1548.58 tok/s (25598 tok in 16.53s) and decode 39.85 tok/s (2384 tok in 59.83s). A 2012-token warm complete on that path prefills at 1739.79 tok/s. After the resident int8 weight cache (2 November 2026), coding printed prefill 1671.59 tok/s (25592 tok in 15.31s) and decode 40.40 tok/s (2188 tok in 54.16s). After the mid-width `gemv_fp8_x4` decode path (2 November 2026), coding printed prefill 1673.36 tok/s (25569 tok in 15.28s) and decode 40.77 tok/s (2172 tok in 53.28s). When the budget fits the restored int8 codes and channel scales for every int8 plane beside the rest of the opening, `restore_fp8_i8` runs once per plane into a resident atlas on first use and later prefill matmuls only bind it (a 2012-token prompt prefilled at 1781 tok/s on the first call and about 1852 tok/s warm, 2 November 2026); otherwise each matmul restores into the single stage buffer.

The packed record may also hold the official MTP head as Metal `half` (`mtp-half` from the SSD BF16 checkout). A request `mtp` wins; omitted uses the registry row (default off). `mtp_depth` defaults to 3. Greedy verify only steps confirmed tokens. On the carrot smoke, MTP off decoded faster than MTP on (47.45 tok/s vs 37.44 tok/s, 2 November 2026).

Progress uses `favlo.console` begin / count / end / note on the console thread (`prompt`, `cache`, load / prefill / decode, plus `mtp_depth` / `mtp` when drafting). Prefill ends at the first sampled token; decode runs from that sample to the last kept sample; prefix writes stay outside those clocks. Chat Completions `usage` carries `context_length` and `timings` (`prompt_per_second`, `predicted_per_second`, `prompt_n`, `predicted_n`) from those clocks. Decode streaming uses a separate queue thread. NumPy stays off the hot path.
