# Documentation

The public tree under `docs/` is **as-built** detail. The process and the write set for each runtime are in [ARCHITECTURE.md](../ARCHITECTURE.md). That file wins for the listener and for which files an agent may edit. A pipeline directory wins for that pipeline.

| Path | Contents |
|---|---|
| [serve.md](serve.md) | Listener, daemon, and the serve log |
| [console.md](console.md) | Shared status row |
| [prefix.md](prefix.md) | SSD prefix store |
| [harness.md](harness.md) | File agent, coding agent, and worktree agent |
| [ai-assistance/README.md](ai-assistance/README.md) | AI-assisted features on `favlo serve` |
| [qwen38-flash-next/architecture.md](qwen38-flash-next/architecture.md) | Qwen3.8-Flash-Next pipeline contract |
| [qwen38-flash-next/research/int2-miss.md](qwen38-flash-next/research/int2-miss.md) | Qwen3.8-Flash-Next int2 reconstruction bound |
| [qwen38-flash-next/research/kernel-ceilings.md](qwen38-flash-next/research/kernel-ceilings.md) | Qwen3.8-Flash-Next isolated dispatch sum |
| [qwen38-flash-next/research/expert-planes.md](qwen38-flash-next/research/expert-planes.md) | Qwen3.8-Flash-Next expert-width dispatch times |
| [qwen38-flash-next/research/recorded-forward.md](qwen38-flash-next/research/recorded-forward.md) | Qwen3.8-Flash-Next recorded prefill and decode sums |
| [qwen38-flash-next/performance.md](qwen38-flash-next/performance.md) | Qwen3.8-Flash-Next recorded prefill and decode rates |
| [qwen38-flash-next/references.md](qwen38-flash-next/references.md) | Papers and specifications for that pipeline |
| [qwen38-27b/README.md](qwen38-27b/README.md) | Qwen3.8-27B pipeline contract |
| [qwen38-27b/SETUP.md](qwen38-27b/SETUP.md) | Qwen3.8-27B download, pack, and first chat |
| [ternary-bonsai-2/README.md](ternary-bonsai-2/README.md) | Failed Ternary Bonsai 2 pipeline |
| [qwen38-27b/research/ffn-fp8-budget.md](qwen38-27b/research/ffn-fp8-budget.md) | Qwen3.8-27B feed-forward fp8 budget measurement |
| [qwen38-27b/research/span-draft.md](qwen38-27b/research/span-draft.md) | Qwen3.8-27B two-token draft measurement |
| [qwen38-27b/research/kernel-ceilings.md](qwen38-27b/research/kernel-ceilings.md) | Qwen3.8-27B isolated projection sum |
| [qwen38-27b/research/recorded-forward.md](qwen38-27b/research/recorded-forward.md) | Qwen3.8-27B recorded prefill and decode sums |
| [qwen38-27b/research/mixed-agent-forward.md](qwen38-27b/research/mixed-agent-forward.md) | Qwen3.8-27B mixed-record agent forward |
| [qwen36-35b-a3b/README.md](qwen36-35b-a3b/README.md) | Qwen3.6-35B-A3B pipeline contract |
| [qwen36-35b-a3b/SETUP.md](qwen36-35b-a3b/SETUP.md) | Qwen3.6-35B-A3B download, pack, and first chat |
| [qwen36-35b-a3b/research/fp16-peak-restore.md](qwen36-35b-a3b/research/fp16-peak-restore.md) | Qwen3.6-35B-A3B float16 peak-restore bound |
| [qwen36-35b-a3b/research/kernel-ceilings.md](qwen36-35b-a3b/research/kernel-ceilings.md) | Qwen3.6-35B-A3B isolated dispatch sum |
| [qwen36-35b-a3b/research/recorded-forward.md](qwen36-35b-a3b/research/recorded-forward.md) | Qwen3.6-35B-A3B recorded prefill and decode sums |
| [qwen36-35b-a3b/performance.md](qwen36-35b-a3b/performance.md) | Qwen3.6-35B-A3B recorded prefill and decode rates |
| [qwen38-flash-next/record.md](qwen38-flash-next/record.md) | Metal record layout and tier rules |
| [qwen38-flash-next/ngram.md](qwen38-flash-next/ngram.md) | n-gram hash and injection |
| [qwen38-flash-next/schedule.md](qwen38-flash-next/schedule.md) | Prefill, decode, and quantize schedule |
| [qwen38-flash-next/probe.md](qwen38-flash-next/probe.md) | Block-of-32 GPU probe |
| [qwen38-flash-next/console.md](qwen38-flash-next/console.md) | Status row binding |
| [qwen38-flash-next/benchmarks.md](qwen38-flash-next/benchmarks.md) | Harness scores against `serve` |
| [qwen38-27b/benchmarks.md](qwen38-27b/benchmarks.md) | Harness scores against `serve` |
| [qwen36-35b-a3b/benchmarks.md](qwen36-35b-a3b/benchmarks.md) | Harness scores against `serve` |
| [z-image-turbo/README.md](z-image-turbo/README.md) | Z-Image-Turbo pipeline contract |
| [wan22-i2v/README.md](wan22-i2v/README.md) | Wan 2.2 I2V pipeline contract |
| [ace-step-15-turbo/README.md](ace-step-15-turbo/README.md) | ACE-Step 1.5 XL Turbo pipeline contract |
| [zonos2/README.md](zonos2/README.md) | Failed ZONOS2 pipeline |
| [zonos2/references.md](zonos2/references.md) | Papers and specifications for the ZONOS2 pipeline |
| [voxcpm2/README.md](voxcpm2/README.md) | VoxCPM2 pipeline contract |
| [voxcpm2/references.md](voxcpm2/references.md) | Papers and specifications for the VoxCPM2 pipeline |
| [ace-step-15-turbo/research/dit-mlp-fp4-error.md](ace-step-15-turbo/research/dit-mlp-fp4-error.md) | ACE-Step XL Turbo DiT MLP fp4 error ranking |
| [zonos2/research/moe-fp4-error.md](zonos2/research/moe-fp4-error.md) | ZONOS2 MoE fp4 error ranking |
| [wan22-i2v/research/clip-ceiling.md](wan22-i2v/research/clip-ceiling.md) | Wan 2.2 I2V 9-frame kernel sum |
| [wan22-i2v/research/block-fp4-error.md](wan22-i2v/research/block-fp4-error.md) | Wan 2.2 I2V fp8 block ranking |
| [z-image-turbo/research/generation-ceiling.md](z-image-turbo/research/generation-ceiling.md) | Z-Image-Turbo 1280 generation sum |

Implementation plans, option comparisons, and design sketches are **not** committed here. They belong under `artifacts/implementation-plan/` (gitignored). A measured research remainder lives at `docs/<pipeline>/research/<slug>.md` and does not change the serving note until that behavior ships.
