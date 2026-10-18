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

The server stops after the first turn and starts again, so the next turn loads the stored prefix. Later turns stay on that server. This run reached 87318 prompt tokens over four requests.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 361 | miss | 32.11 tok/s (361 tok in 11.24 s) | 11.15 tok/s (31 tok in 2.78 s) |
| 2 | 46417 | hit 392 | 677.55 tok/s (46025 tok in 67.93 s) | 13.31 tok/s (475 tok in 35.69 s) |
| 3 | 46929 | prefix load 46892, hit 46892 | 3.11 tok/s (37 tok in 11.88 s) | 7.64 tok/s (40 tok in 5.24 s) |
| 4 | 87318 | hit 46969 | 412.79 tok/s (40349 tok in 97.75 s) | 8.28 tok/s (1182 tok in 142.80 s) |

The summary is prefill 459.60 tok/s (86772 tok in 188.80 s) and decode 9.26 tok/s (1728 tok in 186.51 s).

## Coding agent

Thinking card. `enable_thinking` is true. Omitted sampling fields come from that card. The command is:

```text
uv run python -m harness.coding --model qwen36-35b-a3b
```

The hidden suite passed. The run used two requests and stopped at 599 prompt tokens.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 399 | miss | 32.74 tok/s (399 tok in 12.19 s) | 33.40 tok/s (180 tok in 5.39 s) |
| 2 | 599 | hit 579 | 1.87 tok/s (20 tok in 10.69 s) | 25.62 tok/s (105 tok in 4.10 s) |

The summary is prefill 18.31 tok/s (419 tok in 22.88 s) and decode 30.03 tok/s (285 tok in 9.49 s).
