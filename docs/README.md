# Documentation

The public tree under `docs/` is **as-built** detail. The process and the write set for each runtime are in [ARCHITECTURE.md](../ARCHITECTURE.md). That file wins for the listener and for which files an agent may edit. A pipeline directory wins for that pipeline.

| Path | Contents |
|---|---|
| [serve.md](serve.md) | Listener, daemon, and the serve log |
| [../web-api/README.md](../web-api/README.md) | Web API for `favlo web` |
| [../web-spa/README.md](../web-spa/README.md) | Expo app for `favlo web` |
| [console.md](console.md) | Shared status row |
| [prefix.md](prefix.md) | SSD prefix store |
| [harness.md](harness.md) | File agent, coding agent, and worktree agent |
| [coding-primitives.md](coding-primitives.md) | Tool contract for the coding harness |
| [synthetic-matmul2d.md](synthetic-matmul2d.md) | Synthetic `matmul2d` dtype rates |
| [synthetic-decode.md](synthetic-decode.md) | Synthetic one-token GEMV and `matmul2d` rates |
| [ai-assistance/README.md](ai-assistance/README.md) | AI-assisted features on `favlo serve` |
| [qwen38-27b/README.md](qwen38-27b/README.md) | Qwen3.8-27B pipeline contract |
| [qwen38-27b/SETUP.md](qwen38-27b/SETUP.md) | Qwen3.8-27B download, pack, and first chat |
| [ternary-bonsai-2/README.md](ternary-bonsai-2/README.md) | Failed Ternary Bonsai 2 pipeline |
| [qwen38-27b/research/ffn-fp8-budget.md](qwen38-27b/research/ffn-fp8-budget.md) | Qwen3.8-27B feed-forward fp8 budget measurement |
| [qwen38-27b/research/span-draft.md](qwen38-27b/research/span-draft.md) | Qwen3.8-27B two-token draft measurement |
| [qwen38-27b/research/kernel-ceilings.md](qwen38-27b/research/kernel-ceilings.md) | Qwen3.8-27B isolated projection sum |
| [qwen38-27b/research/mtp-window.md](qwen38-27b/research/mtp-window.md) | Qwen3.8-27B MTP verify window |
| [qwen38-27b/research/recorded-forward.md](qwen38-27b/research/recorded-forward.md) | Qwen3.8-27B recorded prefill and decode sums |
| [qwen38-27b/research/mixed-agent-forward.md](qwen38-27b/research/mixed-agent-forward.md) | Qwen3.8-27B mixed-record agent forward |
| [qwen38-27b/research/int8-prefill.md](qwen38-27b/research/int8-prefill.md) | Qwen3.8-27B int8 prefill recorded rates |
| [qwen36-35b-a3b/README.md](qwen36-35b-a3b/README.md) | Qwen3.6-35B-A3B pipeline contract |
| [qwen36-35b-a3b/SETUP.md](qwen36-35b-a3b/SETUP.md) | Qwen3.6-35B-A3B download, pack, and first chat |
| [qwen36-35b-a3b/research/fp16-peak-restore.md](qwen36-35b-a3b/research/fp16-peak-restore.md) | Qwen3.6-35B-A3B float16 peak-restore bound |
| [qwen36-35b-a3b/research/kernel-ceilings.md](qwen36-35b-a3b/research/kernel-ceilings.md) | Qwen3.6-35B-A3B isolated dispatch sum |
| [qwen36-35b-a3b/research/recorded-forward.md](qwen36-35b-a3b/research/recorded-forward.md) | Qwen3.6-35B-A3B recorded prefill and decode sums |
| [qwen36-35b-a3b/performance.md](qwen36-35b-a3b/performance.md) | Qwen3.6-35B-A3B recorded prefill and decode rates |
| [gemma4-26b-a4b/README.md](gemma4-26b-a4b/README.md) | Gemma 4 26B-A4B pipeline contract |
| [qwen35-4b/README.md](qwen35-4b/README.md) | Qwen3.5-4B language pipeline |
| [qwen35-4b/references.md](qwen35-4b/references.md) | Papers and specifications for the Qwen3.5-4B pipeline |
| [qwen35-4b/SETUP.md](qwen35-4b/SETUP.md) | Qwen3.5-4B download and pack |
| [qwen35-4b/ceilings.md](qwen35-4b/ceilings.md) | Qwen3.5-4B kernel ceilings |
| [spark-x26-1-7b/README.md](spark-x26-1-7b/README.md) | Spark-X2.6-1.7B pipeline contract |
| [spark-x26-1-7b/references.md](spark-x26-1-7b/references.md) | Papers and specifications for the Spark pipeline |
| [spark-x26-1-7b/ceilings.md](spark-x26-1-7b/ceilings.md) | Spark-X2.6-1.7B kernel ceilings |
| [spark-x26-1-7b/research/kernel-ceilings.md](spark-x26-1-7b/research/kernel-ceilings.md) | Spark-X2.6-1.7B isolated dispatch sum |
| [spark-x26-1-7b/research/recorded-forward.md](spark-x26-1-7b/research/recorded-forward.md) | Spark-X2.6-1.7B recorded prefill and decode runs |
| [gemma4-26b-a4b/ceilings.md](gemma4-26b-a4b/ceilings.md) | Gemma 4 26B-A4B bandwidth ceilings |
| [gemma4-26b-a4b/references.md](gemma4-26b-a4b/references.md) | Papers and specifications for the Gemma 4 pipeline |
| [gemma4-26b-a4b/research/kernel-ceilings.md](gemma4-26b-a4b/research/kernel-ceilings.md) | Gemma 4 26B-A4B isolated decode sum |
| [gemma4-26b-a4b/research/recorded-forward.md](gemma4-26b-a4b/research/recorded-forward.md) | Gemma 4 26B-A4B recorded prefill and decode runs |
| [gemma4-26b-a4b/benchmarks.md](gemma4-26b-a4b/benchmarks.md) | Gemma 4 26B-A4B harness scores against `serve` |
| [qwen38-27b/benchmarks.md](qwen38-27b/benchmarks.md) | Harness scores against `serve` |
| [qwen36-35b-a3b/benchmarks.md](qwen36-35b-a3b/benchmarks.md) | Harness scores against `serve` |
| [z-image-turbo/README.md](z-image-turbo/README.md) | Z-Image-Turbo pipeline contract |
| [z-image-turbo/SETUP.md](z-image-turbo/SETUP.md) | Z-Image-Turbo download, pack, and first image |
| [wan22-i2v/README.md](wan22-i2v/README.md) | Wan 2.2 I2V pipeline contract |
| [wan22-i2v/SETUP.md](wan22-i2v/SETUP.md) | Wan 2.2 I2V download, pack, and first clip |
| [wan22-ti2v-5b/README.md](wan22-ti2v-5b/README.md) | Wan 2.2 TI2V-5B pipeline contract |
| [wan22-ti2v-5b/SETUP.md](wan22-ti2v-5b/SETUP.md) | Wan 2.2 TI2V-5B download, pack, and first clip |
| [wan22-ti2v-5b/ceilings.md](wan22-ti2v-5b/ceilings.md) | Wan 2.2 TI2V-5B kernel ceilings |
| [ace-step-15-turbo/README.md](ace-step-15-turbo/README.md) | ACE-Step 1.5 XL Turbo pipeline contract |
| [ace-step-15-turbo/SETUP.md](ace-step-15-turbo/SETUP.md) | ACE-Step download, pack, and first clip |
| [zonos2/README.md](zonos2/README.md) | Failed ZONOS2 pipeline |
| [zonos2/references.md](zonos2/references.md) | Papers and specifications for the ZONOS2 pipeline |
| [bge-m3/README.md](bge-m3/README.md) | BGE-M3 pipeline contract |
| [bge-m3/references.md](bge-m3/references.md) | Papers and specifications for the BGE-M3 pipeline |
| [bge-m3/failed/failed.md](bge-m3/failed/failed.md) | Failed BGE-M3 Neural Engine window above 512 |
| [voxcpm2/README.md](voxcpm2/README.md) | VoxCPM2 pipeline contract |
| [voxcpm2/references.md](voxcpm2/references.md) | Papers and specifications for the VoxCPM2 pipeline |
| [ace-step-15-turbo/research/dit-mlp-fp4-error.md](ace-step-15-turbo/research/dit-mlp-fp4-error.md) | ACE-Step XL Turbo DiT MLP fp4 error ranking |
| [zonos2/research/moe-fp4-error.md](zonos2/research/moe-fp4-error.md) | ZONOS2 MoE fp4 error ranking |
| [wan22-i2v/research/clip-ceiling.md](wan22-i2v/research/clip-ceiling.md) | Wan 2.2 I2V 9-frame kernel sum |
| [wan22-i2v/research/block-fp4-error.md](wan22-i2v/research/block-fp4-error.md) | Wan 2.2 I2V fp8 block ranking |
| [wan22-i2v/research/int8-weight-error.md](wan22-i2v/research/int8-weight-error.md) | Wan 2.2 I2V int8 weight error |
| [wan22-ti2v-5b/research/int8-weight-error.md](wan22-ti2v-5b/research/int8-weight-error.md) | Wan 2.2 TI2V-5B int8 weight error |
| [wan22-ti2v-5b/research/clip-ceiling.md](wan22-ti2v-5b/research/clip-ceiling.md) | Wan 2.2 TI2V-5B 9-frame kernel sum and printed clips |
| [z-image-turbo/research/generation-ceiling.md](z-image-turbo/research/generation-ceiling.md) | Z-Image-Turbo 1280 generation sum |
| [bge-m3/research/recorded-encode.md](bge-m3/research/recorded-encode.md) | BGE-M3 warm encode times |

Do not commit plan or option-comparison notes here. A measured research remainder lives at `docs/<pipeline>/research/<slug>.md` and does not change the serving note until that behavior ships.
