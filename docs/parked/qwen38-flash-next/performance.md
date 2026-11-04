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

## Instruct series

One file-agent run reached 68987 prompt tokens over 7 requests. The server stopped after the first task and started again. The next task loaded 46424 tokens, the previous prompt plus its completion. Later tasks extended that prefix. Request 1 was a stored prefix of the whole prompt, so the server printed no prefill line.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 361 | load 361 |  | 6.69 tok/s (31 tok in 4.63 s) |
| 2 | 46417 | hit 392 | 204.39 tok/s (46025 tok in 225.19 s) | 1.03 tok/s (7 tok in 6.81 s) |
| 3 | 46461 | load 46424 | 11.58 tok/s (37 tok in 3.20 s) | 6.87 tok/s (79 tok in 11.49 s) |
| 4 | 46564 | hit 46540 | 21.40 tok/s (24 tok in 1.12 s) | 8.73 tok/s (106 tok in 12.15 s) |
| 5 | 46698 | hit 46670 | 21.94 tok/s (28 tok in 1.28 s) | 7.84 tok/s (99 tok in 12.64 s) |
| 6 | 46831 | hit 46797 | 22.12 tok/s (34 tok in 1.54 s) | 8.04 tok/s (73 tok in 9.08 s) |
| 7 | 68987 | hit 46904 | 175.08 tok/s (22083 tok in 126.13 s) | 5.71 tok/s (126 tok in 22.06 s) |

The summary is prefill 190.34 tok/s (68231 tok in 358.46 s) and decode 6.61 tok/s (521 tok in 78.86 s).

## Thinking series

One coding-agent run wrote the graded file. The hidden suite passed. The second decode line is the server log. The harness summary ran before that line was copied, so its decode total is the first request only.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 437 | miss | 62.02 tok/s (437 tok in 7.05 s) | 10.37 tok/s (893 tok in 86.11 s) |
| 2 | 1350 | hit 1330 | 32.19 tok/s (20 tok in 0.62 s) | 9.38 tok/s (369 tok in 39.33 s) |

The harness summary is prefill 59.58 tok/s (457 tok in 7.67 s) and decode 10.37 tok/s (893 tok in 86.11 s).
