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

The server stops after the first turn and starts again, so the next turn loads the stored prefix. Later turns stay on that server. This run reached 87769 prompt tokens over four requests.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 361 | miss | 30.42 tok/s (361 tok in 11.87 s) | 10.80 tok/s (31 tok in 2.87 s) |
| 2 | 46417 | hit 392 | 652.26 tok/s (46025 tok in 70.56 s) | 41.09 tok/s (367 tok in 8.93 s) |
| 3 | 46821 | prefix load 46784, hit 46784 | 3.13 tok/s (37 tok in 11.81 s) | 12.57 tok/s (40 tok in 3.18 s) |
| 4 | 87769 | hit 46861 | 402.46 tok/s (40908 tok in 101.64 s) | 39.79 tok/s (673 tok in 16.91 s) |

The summary is prefill 445.84 tok/s (87331 tok in 195.88 s) and decode 34.84 tok/s (1111 tok in 31.89 s).

## Coding agent

Thinking card. `enable_thinking` is true. Omitted sampling fields come from that card. The command is:

```text
uv run python -m harness.coding --model qwen36-35b-a3b
```

The hidden suite passed. The run used two requests and stopped at 637 prompt tokens.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 399 | miss | 36.41 tok/s (399 tok in 10.96 s) | 39.86 tok/s (218 tok in 5.47 s) |
| 2 | 637 | hit 617 | 1.69 tok/s (20 tok in 11.83 s) | 21.51 tok/s (76 tok in 3.53 s) |

The summary is prefill 18.39 tok/s (419 tok in 22.79 s) and decode 32.67 tok/s (294 tok in 9.00 s).
