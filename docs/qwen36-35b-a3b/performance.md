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

The server stops after the first turn and starts again, so the next turn loads the stored prefix. Later turns stay on that server. This run reached 88443 prompt tokens over four requests. The prefill kernel ceiling for 65536 tokens from position 0 is 815.71 tok/s.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 361 | miss | 500.50 tok/s (361 tok in 0.72 s) | 76.42 tok/s (31 tok in 0.41 s) |
| 2 | 46417 | hit 392 | 811.33 tok/s (46025 tok in 56.73 s) | 57.99 tok/s (368 tok in 6.35 s) |
| 3 | 46822 | prefix load 46785, hit 46785 | 37.59 tok/s (37 tok in 0.98 s) | 57.81 tok/s (40 tok in 0.69 s) |
| 4 | 88443 | hit 46862 | 468.56 tok/s (41581 tok in 88.74 s) | 48.06 tok/s (859 tok in 17.87 s) |

The summary is prefill 597.98 tok/s (88004 tok in 147.17 s) and decode 51.26 tok/s (1298 tok in 25.32 s).

One resident load then applied four chunks in order. The first 2048-token chunk of a prompt, the kernel ceiling in [ceilings.md](ceilings.md), is 1521.37 tok/s.

| Chunk | Position | Tokens | Prefill |
|---|---:|---:|---|
| 1 | 0 | 64 | 57.36 tok/s (64 tok in 1.12 s) |
| 2 | 64 | 2048 | 1370.25 tok/s (2048 tok in 1.49 s) |
| 3 | 2112 | 2048 | 1297.27 tok/s (2048 tok in 1.58 s) |
| 4 | 4160 | 4096 | 1229.00 tok/s (4096 tok in 3.33 s) |

## Coding agent

Thinking card. `enable_thinking` is true. Omitted sampling fields come from that card. The command is:

```text
uv run python -m harness.coding --model qwen36-35b-a3b
```

The hidden suite passed. The run used two requests and stopped at 563 prompt tokens.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 399 | miss | 470.49 tok/s (399 tok in 0.85 s) | 73.67 tok/s (144 tok in 1.95 s) |
| 2 | 563 | hit 543 | 98.87 tok/s (20 tok in 0.20 s) | 74.83 tok/s (53 tok in 0.71 s) |

The summary is prefill 399.05 tok/s (419 tok in 1.05 s) and decode 74.06 tok/s (197 tok in 2.66 s).
