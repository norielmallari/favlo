# Documentation

The public tree under `docs/` is **as-built** detail. The process and the write set for each runtime are in [ARCHITECTURE.md](../ARCHITECTURE.md). That file wins for the listener and for which files an agent may edit. A pipeline directory wins for that pipeline.

| Path | Contents |
|---|---|
| [qwen38-flash-next/architecture.md](qwen38-flash-next/architecture.md) | Qwen3.8-Flash-Next pipeline contract |
| [qwen38-flash-next/performance.md](qwen38-flash-next/performance.md) | Qwen3.8-Flash-Next long-prompt rates |
| [qwen38-flash-next/references.md](qwen38-flash-next/references.md) | Papers and specifications for that pipeline |
| [qwen38-27b/README.md](qwen38-27b/README.md) | Qwen3.8-27B pipeline contract |
| [qwen36-35b-a3b/README.md](qwen36-35b-a3b/README.md) | Qwen3.6-35B-A3B pipeline contract |
| [qwen36-35b-a3b/performance.md](qwen36-35b-a3b/performance.md) | Qwen3.6-35B-A3B long-prompt rates |
| [qwen38-flash-next/record.md](qwen38-flash-next/record.md) | Metal record layout and tier rules |
| [qwen38-flash-next/ngram.md](qwen38-flash-next/ngram.md) | n-gram hash and injection |
| [qwen38-flash-next/schedule.md](qwen38-flash-next/schedule.md) | Prefill, decode, and quantize schedule |
| [qwen38-flash-next/probe.md](qwen38-flash-next/probe.md) | Block-of-32 GPU probe |
| [qwen38-flash-next/console.md](qwen38-flash-next/console.md) | Status row binding |
| [qwen38-flash-next/benchmarks.md](qwen38-flash-next/benchmarks.md) | Harness scores against `serve` |

Implementation plans, option comparisons, and design sketches are **not** committed here. They belong under `artifacts/implementation-plan/` (gitignored).
