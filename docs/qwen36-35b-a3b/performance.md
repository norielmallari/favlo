# Rates

Kernel and bandwidth ceilings are in [ceilings.md](ceilings.md). The pipeline contract is [README.md](README.md).

The file-agent series and the coding-agent series stay separate. Prefill is the summed suffix time. Decode is the summed sampled-token time.

## Recorded file-reading run

The rates below are from a finished file-reading run. The prompt reached 73810 tokens over five requests. The request set temperature 0 and left thinking on.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 359 | miss | 36.36 tok/s (359 tok in 9.87 s) | 12.16 tok/s (33 tok in 2.71 s) |
| 2 | 46355 | hit 392 | 782.95 tok/s (45963 tok in 58.71 s) | 20.95 tok/s (84 tok in 4.01 s) |
| 3 | 46474 | hit 46439 | 2.98 tok/s (35 tok in 11.73 s) | 13.74 tok/s (42 tok in 3.06 s) |
| 4 | 46583 | hit 46516 | 6.21 tok/s (67 tok in 10.80 s) | 11.33 tok/s (34 tok in 3.00 s) |
| 5 | 73810 | hit 46617 | 516.28 tok/s (27193 tok in 52.67 s) | 18.19 tok/s (70 tok in 3.85 s) |

The summary is prefill 512.01 tok/s (73617 tok in 143.78 s) and decode 15.81 tok/s (263 tok in 16.63 s). The prefill kernel ceiling for 65536 tokens from position 0 was 1047.86 tok/s. The decode kernel ceiling at context 65536 was 82.97 tok/s. Those ceilings were measured when the value mix read float16 attention weights. The ceilings for the current value mix are in [ceilings.md](ceilings.md).

## File agent

Instruct card. `enable_thinking` is false. Omitted sampling fields come from that card. The command is:

```text
uv run python -m harness --model qwen36-35b-a3b
```

The server stops after the first turn and starts again, so the next turn loads the stored prefix. Later turns stay on that server. This run reached 87799 prompt tokens over four requests.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 361 | miss | 32.89 tok/s (361 tok in 10.98 s) | 10.81 tok/s (31 tok in 2.87 s) |
| 2 | 46417 | hit 392 | 651.76 tok/s (46025 tok in 70.62 s) | 33.26 tok/s (397 tok in 11.94 s) |
| 3 | 46851 | prefix load 46814, hit 46814 | 2.85 tok/s (37 tok in 12.97 s) | 11.64 tok/s (40 tok in 3.44 s) |
| 4 | 87799 | hit 46891 | 407.56 tok/s (40908 tok in 100.37 s) | 36.39 tok/s (873 tok in 23.99 s) |

The summary is prefill 447.99 tok/s (87331 tok in 194.94 s) and decode 31.75 tok/s (1341 tok in 42.24 s).

## Coding agent

Thinking card. `enable_thinking` is true. Omitted sampling fields come from that card. The command is:

```text
uv run python -m harness.coding --model qwen36-35b-a3b
```

The hidden suite passed. The run used two requests and stopped at 612 prompt tokens.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 399 | miss | 32.53 tok/s (399 tok in 12.26 s) | 33.64 tok/s (193 tok in 5.74 s) |
| 2 | 612 | hit 592 | 1.63 tok/s (20 tok in 12.30 s) | 24.21 tok/s (114 tok in 4.71 s) |

The summary is prefill 17.06 tok/s (419 tok in 24.56 s) and decode 29.38 tok/s (307 tok in 10.45 s).
