# Recorded rates

Recorded prefill and decode for `qwen38-flash-next`. Kernel and bandwidth ceilings are in [ceilings.md](ceilings.md). The pipeline contract is [architecture.md](architecture.md).

A file-agent run is the instruct series. The command is `uv run python -m harness --model qwen38-flash-next`. The request sets `enable_thinking` false and leaves the other sampling fields unset. A coding-agent run is the thinking series. The command is `uv run python -m harness.coding --model qwen38-flash-next`. The request sets `enable_thinking` true. The two series stay separate.

The table is one finished file-reading run. That request set temperature 0 and left thinking on. It continued past 65536 prompt tokens and stopped at 92739 over four requests. The server stopped after the first task and started again, so the next task loaded the stored prefix. Prefill is the summed suffix time. Decode is the summed sampled-token time.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 397 | hit 397 | prefix | 4.41 tok/s (31 tok in 7.03 s) |
| 2 | 46485 | hit 428 | 214.86 tok/s (46057 tok in 214.36 s) | 5.18 tok/s (128 tok in 24.73 s) |
| 3 | 46651 | hit 46613 | 7.21 tok/s (38 tok in 5.27 s) | 4.05 tok/s (31 tok in 7.65 s) |
| 4 | 92739 | hit 46682 | 191.68 tok/s (46057 tok in 240.28 s) | 4.41 tok/s (128 tok in 29.01 s) |

The summary is prefill 200.37 tok/s (92152 tok in 459.91 s) and decode 4.65 tok/s (318 tok in 68.42 s). The prefill kernel ceiling for one 2048-token chunk, routed experts excluded, is 518.4 tok/s. The decode kernel ceiling is 12.67 tok/s at 32768 tokens and 11.20 tok/s at 262144 tokens. Both assume the ten experts per layer are already resident.
