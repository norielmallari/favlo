# Research

Catalog of measured claims. One row is one note. The note holds the table, the predecessors, and the command. This file does not restate them.

`measured` means the serving note is unchanged. `shipped` means that note describes the behavior. `narrowed` means a correction was appended to the note. A row stays.

| Slug | Pipeline | Status | Searched | Remainder | Note |
|---|---|---|---|---|---|
| `ffn-fp8-budget` | `qwen38-27b` | shipped | 2026-10-21 | Under the round-up E8M0 scale, the widest fp4 block miss is 1/5 of the peak and the widest fp8 miss is 1/17, so fp8 shrinks that miss by 17/5. | [docs/qwen38-27b/research/ffn-fp8-budget.md](docs/qwen38-27b/research/ffn-fp8-budget.md) |
| `span-draft` | `qwen38-27b` | shipped | 2026-10-21 | On one prompt the two-token projection matched six serial argmax pairs and the Gated DeltaNet input differed by at most 2e-6, and keeping that state changed the completion by the tenth token. | [docs/qwen38-27b/research/span-draft.md](docs/qwen38-27b/research/span-draft.md) |
| `kernel-ceilings` | `qwen38-27b` | shipped | 2026-10-21 | The isolated projection sum is 325.9 ms, 3.07 tok/s, and a warm fused step measured 63.4 to 64.6 ms. | [docs/qwen38-27b/research/kernel-ceilings.md](docs/qwen38-27b/research/kernel-ceilings.md) |
| `recorded-forward` | `qwen38-27b` | shipped | 2026-10-21 | A warm run recorded 353.54 tok/s on a 2048-token prefill and 15.29 tok/s on 64 argmax tokens. | [docs/qwen38-27b/research/recorded-forward.md](docs/qwen38-27b/research/recorded-forward.md) |
| `fp16-peak-restore` | `qwen36-35b-a3b` | shipped | 2026-10-21 | A float16 cast flushes an attention weight at 2^-25; dividing by the window peak first keeps that window's largest weight. | [docs/qwen36-35b-a3b/research/fp16-peak-restore.md](docs/qwen36-35b-a3b/research/fp16-peak-restore.md) |
| `kernel-ceilings` | `qwen36-35b-a3b` | shipped | 2026-10-21 | The isolated dispatch sum is 97.25 tok/s decode at context 2048 and 815.71 tok/s prefill at 65536 tokens. | [docs/qwen36-35b-a3b/research/kernel-ceilings.md](docs/qwen36-35b-a3b/research/kernel-ceilings.md) |
| `recorded-forward` | `qwen36-35b-a3b` | shipped | 2026-10-21 | The file-agent forward recorded 597.98 tok/s prefill over 88004 tokens and 51.26 tok/s decode over 1298 tokens. | [docs/qwen36-35b-a3b/research/recorded-forward.md](docs/qwen36-35b-a3b/research/recorded-forward.md) |
| `int2-miss` | `qwen38-flash-next` | shipped | 2026-10-21 | Under the round-up E8M0 scale, an int2 block whose scaled peak equals 1 misses by 1/2 of that peak, and a peak that has just rounded up misses by the whole peak. | [docs/qwen38-flash-next/research/int2-miss.md](docs/qwen38-flash-next/research/int2-miss.md) |
| `kernel-ceilings` | `qwen38-flash-next` | shipped | 2026-10-21 | Decode at context 2048 sums to 13.29 tok/s, and one prefill chunk without routed experts is 518.4 tok/s. | [docs/qwen38-flash-next/research/kernel-ceilings.md](docs/qwen38-flash-next/research/kernel-ceilings.md) |
| `recorded-forward` | `qwen38-flash-next` | shipped | 2026-10-21 | The file-reading run recorded 200.37 tok/s prefill over 92152 tokens and 4.65 tok/s decode over 318 tokens. | [docs/qwen38-flash-next/research/recorded-forward.md](docs/qwen38-flash-next/research/recorded-forward.md) |
| `generation-ceiling` | `z-image-turbo` | shipped | 2026-10-21 | One 1280 generation sums to 202.87 s, and a printed run of the smoke prompt was 200.58 s. | [docs/z-image-turbo/research/generation-ceiling.md](docs/z-image-turbo/research/generation-ceiling.md) |
