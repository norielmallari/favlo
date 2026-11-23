# Documentation

The public tree under `docs/` is **as-built** detail. The process and the write set for each runtime are in [ARCHITECTURE.md](../ARCHITECTURE.md). That file wins for the listener and for which files an agent may edit. A pipeline directory wins for that pipeline. Its README is a summary. `ARCHITECTURE.md` in that folder is the as-built detail.

| Path | Contents |
|---|---|
| [serve.md](serve.md) | Listener, daemon, and the serve log |
| [../web-api/README.md](../web-api/README.md) | Web API for `favlo web` |
| [../web-spa/README.md](../web-spa/README.md) | Expo app for `favlo web` |
| [console.md](console.md) | Shared status row |
| [progress-reporting.md](progress-reporting.md) | Progress calls and which pipeline uses each |
| [prefix.md](prefix.md) | SSD prefix store |
| [harness.md](harness.md) | File agent, coding agent, and worktree agent |
| [coding-primitives.md](coding-primitives.md) | Tool contract for the coding harness |
| [synthetic-matmul2d.md](synthetic-matmul2d.md) | Synthetic `matmul2d` dtype rates |
| [synthetic-decode.md](synthetic-decode.md) | Synthetic one-token GEMV and `matmul2d` rates |
| [ai-assistance/README.md](ai-assistance/README.md) | AI-assisted features on `favlo serve` |
| [qwen38_27b/README.md](qwen38_27b/README.md) | Qwen3.8-27B summary |
| [qwen38_27b/research/ffn-fp8-budget.md](qwen38_27b/research/ffn-fp8-budget.md) | Qwen3.8-27B feed-forward fp8 budget measurement |
| [qwen38_27b/research/span-draft.md](qwen38_27b/research/span-draft.md) | Qwen3.8-27B two-token draft measurement |
| [qwen38_27b/research/kernel-ceilings.md](qwen38_27b/research/kernel-ceilings.md) | Qwen3.8-27B isolated projection sum |
| [qwen38_27b/research/mtp-window.md](qwen38_27b/research/mtp-window.md) | Qwen3.8-27B MTP verify window |
| [qwen38_27b/research/recorded-forward.md](qwen38_27b/research/recorded-forward.md) | Qwen3.8-27B recorded prefill and decode sums |
| [qwen38_27b/research/mixed-agent-forward.md](qwen38_27b/research/mixed-agent-forward.md) | Qwen3.8-27B mixed-record agent forward |
| [qwen38_27b/research/int8-prefill.md](qwen38_27b/research/int8-prefill.md) | Qwen3.8-27B int8 prefill recorded rates |
| [qwen38_27b/research/needle-page.md](qwen38_27b/research/needle-page.md) | Qwen3.8-27B page needle at 65536 and 131072 on half and fp8 keys and values |
| [qwen38_27b/research/needle-retrieve.md](qwen38_27b/research/needle-retrieve.md) | Qwen3.8-27B retrieve needle on fp8 keys and values |
| [qwen38_27b/research/coding-agent.md](qwen38_27b/research/coding-agent.md) | Qwen3.8-27B coding agent on fp8 keys and values |
| [qwen38_27b/research/coding-agent-half.md](qwen38_27b/research/coding-agent-half.md) | Qwen3.8-27B coding agent on half keys and values |
| [qwen38_27b/research/synthetic-prompt.md](qwen38_27b/research/synthetic-prompt.md) | Qwen3.8-27B cold 512-token and 10000-token prompts |
| [qwen38_27b/research/synthetic-prompt-mtp.md](qwen38_27b/research/synthetic-prompt-mtp.md) | Qwen3.8-27B cold synthetic prompts with MTP on |
| [qwen38_27b/research/synthetic-prompt-half.md](qwen38_27b/research/synthetic-prompt-half.md) | Qwen3.8-27B cold 512, 1024, 4096, and 10000-token prompts on the fp8-e20 pack with half keys and values |
| [qwen38_27b/research/synthetic-prompt-mtp-half.md](qwen38_27b/research/synthetic-prompt-mtp-half.md) | Qwen3.8-27B cold 512, 1024, 4096, and 10000-token prompts on the fp8-e20 pack with half keys and values and MTP on |
| [qwen36_35b_a3b/README.md](qwen36_35b_a3b/README.md) | Qwen3.6-35B-A3B summary |
| [inkling_2_mini/README.md](inkling_2_mini/README.md) | Inkling 2 Mini summary |
| [inkling_2_mini/ceilings.md](inkling_2_mini/ceilings.md) | Inkling 2 Mini bandwidth ceiling and kernel costs |
| [inkling_2_mini/research/synthetic-prompt-half.md](inkling_2_mini/research/synthetic-prompt-half.md) | Inkling 2 Mini cold 512, 1024, 4096, and 10000-token prompts on the GPTQ dense fp4 pack with half keys and values |
| [inkling_2_mini/research/coding-text-half.md](inkling_2_mini/research/coding-text-half.md) | Inkling 2 Mini text coding agent on the GPTQ dense fp4 pack with half keys and values |
| [inkling_2_mini/research/coding-vision-half.md](inkling_2_mini/research/coding-vision-half.md) | Inkling 2 Mini vision coding agent on the GPTQ dense fp4 pack with half keys and values |
| [inkling_2_mini/research/file-agent-half.md](inkling_2_mini/research/file-agent-half.md) | Inkling 2 Mini file agent on the GPTQ dense fp4 pack with half keys and values |
| [inkling_2_mini/research/synthetic-prompt-fp8.md](inkling_2_mini/research/synthetic-prompt-fp8.md) | Inkling 2 Mini cold 512, 1024, 4096, and 10000-token prompts on the FOEM fp8 attention pack with half keys and values |
| [inkling_2_mini/research/coding-text-fp8.md](inkling_2_mini/research/coding-text-fp8.md) | Inkling 2 Mini text coding agent on the FOEM fp8 attention pack with half keys and values |
| [inkling_2_mini/research/file-agent-fp8.md](inkling_2_mini/research/file-agent-fp8.md) | Inkling 2 Mini file agent on the FOEM fp8 attention pack with half keys and values |
| [inkling_2_mini/research/intelligence-smoke.md](inkling_2_mini/research/intelligence-smoke.md) | Inkling 2 Mini first 30 rows of IFEval, GSM8K, and HumanEval |
| [inkling_2_mini/benchmarks.md](inkling_2_mini/benchmarks.md) | Inkling 2 Mini harness scores against `serve` |
| [qwen36_35b_a3b/research/fp16-peak-restore.md](qwen36_35b_a3b/research/fp16-peak-restore.md) | Qwen3.6-35B-A3B float16 peak-restore bound |
| [qwen36_35b_a3b/research/kernel-ceilings.md](qwen36_35b_a3b/research/kernel-ceilings.md) | Qwen3.6-35B-A3B isolated dispatch sum |
| [qwen36_35b_a3b/research/recorded-forward.md](qwen36_35b_a3b/research/recorded-forward.md) | Qwen3.6-35B-A3B recorded prefill and decode sums |
| [qwen36_35b_a3b/research/needle-retrieve.md](qwen36_35b_a3b/research/needle-retrieve.md) | Qwen3.6-35B-A3B retrieve needle on fp8 keys and values |
| [qwen36_35b_a3b/research/needle-page.md](qwen36_35b_a3b/research/needle-page.md) | Qwen3.6-35B-A3B page needle on half keys and values |
| [qwen36_35b_a3b/research/coding-agent.md](qwen36_35b_a3b/research/coding-agent.md) | Qwen3.6-35B-A3B coding agent on fp8 keys and values |
| [qwen36_35b_a3b/research/coding-agent-half.md](qwen36_35b_a3b/research/coding-agent-half.md) | Qwen3.6-35B-A3B coding agent on half keys and values |
| [qwen36_35b_a3b/research/coding-text.md](qwen36_35b_a3b/research/coding-text.md) | Qwen3.6-35B-A3B text coding agent on half keys and values |
| [qwen36_35b_a3b/research/synthetic-prompt.md](qwen36_35b_a3b/research/synthetic-prompt.md) | Qwen3.6-35B-A3B cold 512-token and 10000-token prompts |
| [qwen36_35b_a3b/research/synthetic-prompt-half.md](qwen36_35b_a3b/research/synthetic-prompt-half.md) | Qwen3.6-35B-A3B cold 512, 1024, 4096, and 10000-token prompts on the fp8-e30-24h pack with half keys and values |
| [qwen36_35b_a3b/performance.md](qwen36_35b_a3b/performance.md) | Qwen3.6-35B-A3B recorded prefill and decode rates |
| [gemma4_26b_a4b/README.md](gemma4_26b_a4b/README.md) | Gemma 4 26B-A4B summary |
| [qwen35_4b/README.md](qwen35_4b/README.md) | Qwen3.5-4B language pipeline |
| [qwen35_4b/references.md](qwen35_4b/references.md) | Papers and specifications for the Qwen3.5-4B pipeline |
| [qwen35_4b/ceilings.md](qwen35_4b/ceilings.md) | Qwen3.5-4B kernel ceilings |
| [qwen35_4b/research/int8-prefill.md](qwen35_4b/research/int8-prefill.md) | Qwen3.5-4B int8 prefill recorded rates |
| [qwen35_4b/research/kernel-ceilings.md](qwen35_4b/research/kernel-ceilings.md) | Qwen3.5-4B isolated dispatch sum |
| [qwen35_4b/research/recorded-forward.md](qwen35_4b/research/recorded-forward.md) | Qwen3.5-4B recorded prefill and decode runs |
| [qwen35_4b/research/needle-page.md](qwen35_4b/research/needle-page.md) | Qwen3.5-4B page needle on fp8 keys and values |
| [qwen35_4b/research/coding-agent.md](qwen35_4b/research/coding-agent.md) | Qwen3.5-4B vision coding agent on fp8 keys and values |
| [qwen35_4b/research/coding-agent-mtp.md](qwen35_4b/research/coding-agent-mtp.md) | Qwen3.5-4B vision coding agent with MTP on |
| [qwen35_4b/research/coding-text.md](qwen35_4b/research/coding-text.md) | Qwen3.5-4B text coding agent on fp8 keys and values |
| [qwen35_4b/research/synthetic-prompt.md](qwen35_4b/research/synthetic-prompt.md) | Qwen3.5-4B cold 512, 1024, 4096, and 10000-token prompts |
| [gemma4_26b_a4b/ceilings.md](gemma4_26b_a4b/ceilings.md) | Gemma 4 26B-A4B kernel and bandwidth ceilings |
| [gemma4_26b_a4b/references.md](gemma4_26b_a4b/references.md) | Papers and specifications for the Gemma 4 pipeline |
| [gemma4_26b_a4b/research/kernel-ceilings.md](gemma4_26b_a4b/research/kernel-ceilings.md) | Gemma 4 26B-A4B packed kernel ceiling |
| [gemma4_26b_a4b/research/recorded-forward.md](gemma4_26b_a4b/research/recorded-forward.md) | Gemma 4 26B-A4B recorded prefill and decode runs |
| [gemma4_26b_a4b/research/coding-agent.md](gemma4_26b_a4b/research/coding-agent.md) | Gemma 4 26B-A4B vision coding-agent rates on fp8 keys and values |
| [gemma4_26b_a4b/research/coding-agent-half.md](gemma4_26b_a4b/research/coding-agent-half.md) | Gemma 4 26B-A4B vision coding-agent rates on the fp8-e20 pack with half keys and values |
| [gemma4_26b_a4b/research/coding-text.md](gemma4_26b_a4b/research/coding-text.md) | Gemma 4 26B-A4B text coding-agent rate |
| [gemma4_26b_a4b/research/needle-page.md](gemma4_26b_a4b/research/needle-page.md) | Gemma 4 26B-A4B page-needle rate |
| [gemma4_26b_a4b/research/synthetic-prompt.md](gemma4_26b_a4b/research/synthetic-prompt.md) | Gemma 4 26B-A4B cold 512-token and 10000-token prompts on fp8 keys and values |
| [gemma4_26b_a4b/research/synthetic-prompt-half.md](gemma4_26b_a4b/research/synthetic-prompt-half.md) | Gemma 4 26B-A4B cold 512, 1024, 4096, and 10000-token prompts on the fp8-e20 pack with half keys and values |
| [gemma4_26b_a4b/benchmarks.md](gemma4_26b_a4b/benchmarks.md) | Gemma 4 26B-A4B harness scores against `serve` |
| [qwen38_27b/benchmarks.md](qwen38_27b/benchmarks.md) | Harness scores against `serve` |
| [qwen36_35b_a3b/benchmarks.md](qwen36_35b_a3b/benchmarks.md) | Harness scores against `serve` |
| [z_image/README.md](z_image/README.md) | Shared Z-Image forward used by Turbo and the base checkpoint |
| [z_image_turbo/README.md](z_image_turbo/README.md) | Z-Image-Turbo summary |
| [z_image_base/README.md](z_image_base/README.md) | Z-Image summary |
| [klein_4b/README.md](klein_4b/README.md) | FLUX.2 [klein] 4B summary |
| [klein_4b/ceilings.md](klein_4b/ceilings.md) | FLUX.2 [klein] 4B kernel ceilings for one 576×320 edit |
| [wan22_i2v/README.md](wan22_i2v/README.md) | Wan 2.2 I2V summary |
| [wan22_ti2v_5b/README.md](wan22_ti2v_5b/README.md) | Wan 2.2 TI2V-5B summary |
| [umt5_xxl/README.md](umt5_xxl/README.md) | Shared umT5-XXL int8 pack (Wan `umt5` CLI) |
| [wan22_ti2v_5b/ceilings.md](wan22_ti2v_5b/ceilings.md) | Wan 2.2 TI2V-5B kernel ceilings |
| [ace_step_15_turbo/README.md](ace_step_15_turbo/README.md) | ACE-Step 1.5 XL Turbo summary |
| [bge_m3/README.md](bge_m3/README.md) | BGE-M3 summary |
| [bge_m3/references.md](bge_m3/references.md) | Papers and specifications for the BGE-M3 pipeline |
| [bge_m3/failed/failed.md](bge_m3/failed/failed.md) | Failed BGE-M3 Neural Engine window above 512 |
| [voxcpm2/README.md](voxcpm2/README.md) | VoxCPM2 summary |
| [voxcpm2/references.md](voxcpm2/references.md) | Papers and specifications for the VoxCPM2 pipeline |
| [ace_step_15_turbo/research/dit-mlp-fp4-error.md](ace_step_15_turbo/research/dit-mlp-fp4-error.md) | ACE-Step XL Turbo DiT MLP fp4 error ranking |
| [wan22_i2v/research/clip-ceiling.md](wan22_i2v/research/clip-ceiling.md) | Wan 2.2 I2V 9-frame kernel sum |
| [wan22_i2v/research/five-second-clip.md](wan22_i2v/research/five-second-clip.md) | Wan 2.2 I2V 81-frame printed stage times |
| [wan22_i2v/research/block-fp4-error.md](wan22_i2v/research/block-fp4-error.md) | Wan 2.2 I2V fp8 block ranking |
| [wan22_i2v/research/int8-weight-error.md](wan22_i2v/research/int8-weight-error.md) | Wan 2.2 I2V int8 weight error |
| [wan22_ti2v_5b/research/int8-weight-error.md](wan22_ti2v_5b/research/int8-weight-error.md) | Wan 2.2 TI2V-5B int8 weight error |
| [wan22_ti2v_5b/research/clip-ceiling.md](wan22_ti2v_5b/research/clip-ceiling.md) | Wan 2.2 TI2V-5B 9-frame kernel sum and printed clips |
| [z_image_turbo/research/generation-ceiling.md](z_image_turbo/research/generation-ceiling.md) | Z-Image-Turbo 1280 generation sum |
| [z_image_base/research/generation-time.md](z_image_base/research/generation-time.md) | Z-Image base 512 and 1024 generation times |
| [bge_m3/research/recorded-encode.md](bge_m3/research/recorded-encode.md) | BGE-M3 warm encode times |

Do not commit plan or option-comparison notes here. A measured research remainder lives at `docs/<pipeline>/research/<slug>.md` and does not change the serving note until that behavior ships.

| [research/history/zonos2/moe-fp4-error.md](research/history/zonos2/moe-fp4-error.md) | Historical ZONOS2 MoE fp4 error (pipeline only in git history) |
| [research/history/spark_x26_1_7b/kernel-ceilings.md](research/history/spark_x26_1_7b/kernel-ceilings.md) | Historical Spark kernel ceilings (pipeline only in git history) |
| [research/history/spark_x26_1_7b/recorded-forward.md](research/history/spark_x26_1_7b/recorded-forward.md) | Historical Spark recorded forward (pipeline only in git history) |
