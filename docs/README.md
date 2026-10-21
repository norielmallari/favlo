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
| [qwen38-flash-next/performance.md](qwen38-flash-next/performance.md) | Qwen3.8-Flash-Next recorded prefill and decode rates |
| [qwen38-flash-next/references.md](qwen38-flash-next/references.md) | Papers and specifications for that pipeline |
| [qwen38-27b/README.md](qwen38-27b/README.md) | Qwen3.8-27B pipeline contract |
| [qwen36-35b-a3b/README.md](qwen36-35b-a3b/README.md) | Qwen3.6-35B-A3B pipeline contract |
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

Implementation plans, option comparisons, and design sketches are **not** committed here. They belong under `artifacts/implementation-plan/` (gitignored).
