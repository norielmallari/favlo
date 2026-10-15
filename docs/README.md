# Documentation

The public tree under `docs/` is **as-built** detail for what this repository ships today. Product scope and invariants are in [ARCHITECTURE.md](../ARCHITECTURE.md). If a note here disagrees with that file, ARCHITECTURE.md wins.

| Path | Contents |
|---|---|
| [qwen38/record.md](qwen38/record.md) | Metal record layout and tier rules |
| [qwen38/ngram.md](qwen38/ngram.md) | n-gram hash and injection |
| [qwen38/schedule.md](qwen38/schedule.md) | Prefill, decode, and quantize schedule |
| [qwen38/probe.md](qwen38/probe.md) | Block-of-32 GPU probe |
| [qwen38/console.md](qwen38/console.md) | Status row binding |
| [qwen38/benchmarks.md](qwen38/benchmarks.md) | Harness scores against `serve` |

Implementation plans, option comparisons, and design sketches are **not** committed here. They belong under `artifacts/implementation-plan/` (gitignored).
