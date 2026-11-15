# Qwen3.5-4B architecture

As-built detail for the `qwen35_4b` language pipeline. [README.md](README.md) is the summary.

- `handle_responses` is a `LanguagePipeline`.
- The listener translates a Chat Completions body into that request.
- Clients reach this pipeline through `favlo serve` once the record is registered.
- The packed directory is `models/Qwen3.5-4B-fp8`.

## Registry

- The row key in `models/registry.json` is the packed folder, `Qwen3.5-4B-fp8`.
- The alias `qwen35_4b` is the public id and is not the key.
- Write the field on that folder and save.
- `set_model_mtp` and `set_alias` take the public id.
- A field marked open is copied when the pipeline process starts, so close the resident model after the write.
- The listener stays up.
- A field marked request is read from the file on that request.

```python
from favlo.registry import load_registry, save_registry, set_model_mtp

data = load_registry()
data["models"]["Qwen3.5-4B-fp8"]["kv_cache"] = "fp8_e4m3"
save_registry(data)
set_model_mtp("qwen35_4b", enabled=False, depth=3)
```

```text
uv run python -c 'from favlo.smoke_lock import unload_model; unload_model()'
```

| Field | When | Values |
|---|---|---|
| `kv_cache` | open | `half` or `fp8_e4m3`. Registration writes `fp8_e4m3`. This open accepts those two names only. A missing key is read as `f16`, and this open rejects that name. |
| `mtp` | request | `true` or `false`. A missing key is off. A request `mtp` or `favlo:mtp` wins. |
| `mtp_depth` | request | An integer of 1 or more. A missing key is 3. A request `mtp_depth` or `favlo:mtp_depth` wins. |
| `enable_thinking` | open | `true` or `false`. Registration writes `true`. Used when the request omits effort and the selected card has no `reasoning_effort`. |
| `reasoning_effort` | open | `none` or `high`. An unsupported name resolves the same way a request does (`minimal`, `low`, and `medium` become `high`; `xhigh` and `max` become `high`). Used only when the selected card omits `reasoning_effort`. Both registered cards store one. |
| `default_variant` | open | `thinking` or `instruct`. Registration writes `thinking`. A bare id uses that card. |
| `variants.thinking` | open | `temperature` 1.0, `top_p` 0.95, `top_k` 20, `presence_penalty` 1.5, `reasoning_effort` `high`. |
| `variants.instruct` | open | `temperature` 0.7, `top_p` 0.8, `top_k` 20, `presence_penalty` 1.5, `reasoning_effort` `none`. |
| `alias` | resolve | One name, with no colon and no slash, or `null`. `null` makes the folder the public id. |
| `context_length` | listing | A positive integer. Registration writes 262144. `GET /v1/models` lists it, and the usage prompt cap uses it when `max_prompt_tokens` is absent. The next completion reads the file. The served window stays 262144 in this package. |
| `max_prompt_tokens` | listing | A positive integer. Registration writes 262144. The usage prompt cap. The next completion reads the file. The served window stays 262144. |
| `pipeline_default` | omitted model | `true` on one row of this pipeline. Enabling it clears the mark on the other rows of this pipeline. |

- `variants` accepts only `thinking` and `instruct`.
- A third name is listed and does not select its own card.
- A request still overrides one sampling field for that call.
- Changing the effort does not swap the card.

- The file key `default` is the id harnesses and `favlo web` use when none is named.
- Language registration initially writes `qwen35_4b:instruct`.
- A stored value stays until it is replaced.

- `pipeline`, `runtime`, `directory`, `source`, and `pipeline_root` are the row's identity.
- `runtime` is `metal41`, `cuda`, or `portable`.
- `serve` opens `directory` and imports `{pipeline}.{runtime}`.
- `source` is the BF16 checkout quantize checks.
- `resident_percent` is stored as null and discarded at open.
- This pipeline does not read `parts`.

## Memory

- Device spans come from `favlo.memory`.
- The serve process keeps the ledger.
- This runtime asks for the budget at open and again at the start of each generate.
- A `--max-memory` cap is applied by that ledger, not by this package.
- `open_runtime` calls `memory.set_max_memory` then opens; it does not keep a private parser.
- Spans go through `device_buffer`.
- The live full-attention cache length is the one knob: it shrinks at open until the resident record, Gated DeltaNet state, convolution history, workspace, and that cache fit.
- A 2048-token cache that does not fit refuses to open.
- `open_runtime` takes `owner`; every allocate in that opening uses it; `close` sweeps that string.

## Architecture

Dense hybrid text backbone: 24 Gated DeltaNet layers and 8 full-attention layers (`[linear × 3, full] × 8`), SwiGLU feed-forward, tied embeddings, and a vision tower in the same packed record.

- Hidden size 2560, intermediate 9216, vocab 248320.
- Full attention uses 16 query heads with a fused output gate, 4 KV heads, head dim 256, partial RoPE (64 of 256), and M-RoPE.
- Gated DeltaNet uses 16 key heads, 32 value heads, head dim 128, and a depthwise conv of width 4.

- Weights are fp8 E4M3 with an E8M0 scale every 32 along K.
- Norms, `A_log`, `dt_bias`, convolution taps, and Q/K gains stay Metal `half`.
- Layer / Q / K / final RMSNorm apply gain as `1 + w`.
- The GDN gated norm applies `w` then SiLU of `z`.
- The eight full-attention layers keep a key/value cache.
- The registry field is `kv_cache`.
- `fp8_e4m3` is the default (Metal `metal_fp8_e4m3`).
- `half` remains accepted (Metal `half`, two bytes per element, no scale plane).
- Both live planes are token-major `[position, kv_head, channel]` (with an E8M0 scale plane for `fp8_e4m3`).

- The SSD prefix store under `~/.favlo/prefix/qwen35_4b/` writes that same layout for positions `0 .. count-1` as regions `k.0`…`k.7` and `v.0`…`v.7`, plus `history` (convolution taps), `state` (Gated DeltaNet), and `rope` (the M-RoPE position delta).
- An exact-length file is not sampled; the next token is still stepped.
- With `preserve_thinking` off, prefill stops at the latest user message and stores that prefix before the generation suffix.
- That user-turn file is pinned.
- A prompt that already ends on the user message pins the generation-cut file instead.
- A tool-step generation cut is stored and not pinned.
- Those boundary stores are silent.
- Thinking ends the generation cut before `</think>\n`.
- Instruct ends it before the closed empty think block.
- The next user message shares the user-turn file.
- A simple follow-up shares the generation cut, including a still that sits inside it.
- With the flag on, the user-turn cut is not written.
- A request `favlo:prefix_scratch` stores that run under `~/.favlo/prefix-scratch/<name>`, and `favlo:prefix_budget` is that partition's byte cap.

- Prefill is a 1024-token Metal tile.
- The FFN gate, up, and down planes and the GDN qkv, z, and out planes run int8: `restore_fp8_i8` collapses each fp8 weight row to int8 codes and one channel scale, `quantize_half_i8` does the same per activation row, `prefill_i8_n64` (`prefill_i8_m32` for down) is the int8×int8→int32 `matmul2d`, and `scale_i8_f32` rewrites the accumulators as floats (a 2012-token prompt prefilled at 1730 tok/s against 1520 tok/s on `matmul_fp8`, 2 November 2026).
- The GDN a/b and attention q/k/v/o planes stay on `matmul_fp8` / `matmul_fp8_n32`.
- The rest of the tile is `gdn_chunk`, and `attend_prefill` or `attend_prefill_half`.
- On both planes, `attend_prefill` / `attend_prefill_half` and `attend_decode` / `attend_decode_half` keep the softmax state in registers (no score plane), so a long prompt does not fall back to one-token attend.
- Only the last chunk runs the head.
- Decode stays token-by-token.
- Kernel ceilings are [ceilings.md](ceilings.md).
- Measured remainders for int8 prefill, the isolated dispatch sum, the harness rates, the page needle, the vision coding agent, the text coding agent, and the cold synthetic prompts are [research/int8-prefill.md](research/int8-prefill.md), [research/kernel-ceilings.md](research/kernel-ceilings.md), [research/recorded-forward.md](research/recorded-forward.md), [research/needle-page.md](research/needle-page.md), [research/coding-agent.md](research/coding-agent.md), [research/coding-agent-mtp.md](research/coding-agent-mtp.md), [research/coding-text.md](research/coding-text.md), and [research/synthetic-prompt.md](research/synthetic-prompt.md).

The vision tower is packed beside the language weights and runs for a still image.

- Its matrices are fp8 E4M3 with an E8M0 scale every 32 along K.
- LayerNorm weights, LayerNorm biases, and the other vision biases stay Metal `half`.
- The Conv3d patch embed is stored as `[1024, 1536]`.
- The position table is `[2304, 1024]`.
- Each block is LayerNorm, full attention, and tanh GELU.
- The merger writes 2560-wide rows.
- The prompt expands each still to its merged grid before the prefix check, and prefill overwrites only the image-pad tokens past that hit.
- A span already in the loaded prefix logs `vision cached` and uploads no rows.
- The file stored before the generation suffix includes the still, so a follow-up that drops historical thinking hits that span.
- A span past the hit reads `~/.favlo/vision-cache/qwen35_4b/` and logs `vision cache`, or runs the tower and logs `vision patches` and `vision merged`.
- That cache is the file only.
- The turn uploads the rows it splices and releases them when it finishes.
- `has_vision` is true, so the listener accepts a still.
- Video is rejected.
- An image prompt uses interleaved M-RoPE on the three axes; decode adds the position delta from that table.

The packed record's `chat_template.jinja` is the official Qwen3.5 template (tools and `enable_thinking`).

- MiniFantasy's short template is not used for serve.
- Historical reasoning stays in the prompt when `preserve_thinking` is on.
- The request wins over this pipeline's registry row.
- An omitted row is off.
- A string is an error.
- `settings.json` is not read.
- Omitted is off, matching the model card's multi-turn rule: history keeps the final output and leaves thinking out for assistant turns at or before the latest user message.
- True keeps historical traces so a later turn can stay a prefix of the tokens already applied, including the open `</think>` from a thinking generation prompt.
- An assistant turn after that user message, such as a tool step in the current turn, still carries `reasoning_content`.
- Tool schemas stay in the encoded prompt under `tool_choice` none so a later turn that re-enables tools still matches the stored prefix.
- A required named call forces the opening markup (or the whole call when that function takes no arguments), turns thinking off for that completion, and stops at the initial `</tool_call>` so a second call is not drawn.

## Sampling

Variants in `sampling.py`: `thinking` (default) and `instruct`.

- Each card stores temperature, top-p, top-k, presence penalty, and `reasoning_effort` (`high` on thinking, `none` on instruct).
- Registry context length is 262144.
- Registration copies those cards and stores `enable_thinking` from the pipeline default.
- Temperature 0 is argmax.
- A presence penalty subtracts once per distinct completion token already drawn.
- The shipped build does not run nucleus / top-k on device yet.
- The official card exposes thinking only as `enable_thinking` on/off (no soft `/think` switch and no graduated effort ladder).
- On the Responses route, every listed `reasoning.effort` is accepted.
- The listener response chain, not this pipeline, keeps a `store` true response and rebuilds the next transcript.
- This pipeline does not read `store` or `previous_response_id`.
- Supported values are `none` and `high` (thinking on).
- An unsupported name steps up to the next supported value; when none is higher, it falls back to the next lower (`minimal` / `low` / `medium` → `high`; `xhigh` / `max` → `high`).
- A request effort wins and does not swap the sampling card.
- An omitted effort uses the selected variant's `reasoning_effort` (bare id and `:thinking` → `high`, `:instruct` → `none`), then the row field, then `enable_thinking`, then the pipeline default.
- Do not infer thinking from the suffix spelling alone.
- A thinking request that omits `max_output_tokens` runs until a stop id or the last context position.
- A non-thinking request that omits it stops at 2048 tokens.
- An explicit cap applies to both.

## Measured carrot smoke

Pipeline smoke under the smoke lock (`tests/test_qwen35_4b_smoke.py`), fp8 record, fp8_e4m3 cache (2 November 2026):

| Phase | Rate | Tokens | Seconds |
|---|---|---|---|
| prefill | 1014.25 tok/s | 51 | 0.05 |
| decode (instruct card) | 44.95 tok/s | 105 | 2.34 |
| decode (thinking card) | 43.34 tok/s | 558 | 12.88 |

- A 2012-token prompt on that fp8_e4m3 cache prefills at 1060.88 tok/s (1.90 s).
- The prefill kernel ceiling for a 256-token chunk on the default half cache is 1061.69 tok/s ([ceilings.md](ceilings.md)).
- The earlier fp8_e4m3 attend path summed to 1007.60 tok/s.

`complete` on the default half cache, same 51-token prompt (2 November 2026): prefill 897.90 tok/s (0.06 s), decode 44.81 tok/s (64 tok in 1.43 s).

Coding agent on the fp8_e4m3 cache printed `prompt tokens` / `cache miss` per request (2 November 2026):

| Card | Prefill | Decode |
|---|---|---|
| thinking | 993.10 tok/s (25781 tok in 25.96s) | 37.64 tok/s (2384 tok in 63.34s) |
| instruct | 1049.04 tok/s (23058 tok in 21.98s) | 40.21 tok/s (1863 tok in 46.33s) |

Coding agent on the half cache, thinking on, no vision (2 November 2026): prefill 1393.99 tok/s (25747 tok in 18.47s), decode 39.96 tok/s (2028 tok in 50.75s).

- After int8 prefill for FFN and GDN qkv/z/out (2 November 2026), the same coding agent printed prefill 1548.58 tok/s (25598 tok in 16.53s) and decode 39.85 tok/s (2384 tok in 59.83s).
- A 2012-token warm complete on that path prefills at 1739.79 tok/s.
- After the resident int8 weight cache (2 November 2026), coding printed prefill 1671.59 tok/s (25592 tok in 15.31s) and decode 40.40 tok/s (2188 tok in 54.16s).
- After the mid-width `gemv_fp8_x4` decode path (2 November 2026), coding printed prefill 1673.36 tok/s (25569 tok in 15.28s) and decode 40.77 tok/s (2172 tok in 53.28s).

- When the budget fits the restored int8 codes and channel scales for every int8 plane beside the rest of the opening, `restore_fp8_i8` runs once per plane into a resident atlas on initial use and later prefill matmuls only bind it (a 2012-token prompt prefilled at 1781 tok/s on the initial call and about 1852 tok/s warm, 2 November 2026).
- Otherwise each matmul restores into the single stage buffer.

The packed record may also hold the official MTP head as Metal `half` (`mtp-half` from the SSD BF16 checkout).

- A request `mtp` wins; omitted uses the registry row (default off).
- `mtp_depth` defaults to 3.
- A window of two or three feeds reads each language weight once, then the Gated DeltaNet and the full-attention mix run in token order.
- A rejected tail restores the recurrent state and the token counts and replays only the feeds that stay.
- A confirmed stop id is not stepped and is not emitted, so the next turn continues from the last kept token.
- A longer window stays on the serial step.

`serve` draws a load row while the record is read, then a prefill row that counts uncached prompt tokens as each tile finishes, then a decode row that counts sampled tokens.

- Each count posts onto `favlo.host_progress`; that host thread builds the fact for the console queue, and the CLI thread paints it.
- Notes (`prompt`, `cache`, `prefix_load`, `mtp_depth`, `mtp`, prefix store) go through `console.note` the same way.
- Prefill ends at the initial sampled token; decode runs from that sample to the last kept sample; prefix writes stay outside those clocks.
- Chat Completions `usage` carries `context_length` and `timings` (`prompt_per_second`, `predicted_per_second`, `prompt_n`, `predicted_n`) from those clocks.
- Decode streaming uses a separate queue thread.
- NumPy stays off the hot path.
- Presence and frequency stay on device (`penalize`, `bump_count`).
- Those kernels, plus argmax, share the decode command buffer with the token step, and the draft buffer when MTP is on.
- The initial sample after prefill is one submit.
- Prefix flush stays on the generate thread, outside both clocks.
- A stop check stays on that thread once per chunk and once per token.
- This pipeline is dense; it has no expert-firing histogram and no deferred telemetry pull.