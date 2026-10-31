# Rates

Kernel and bandwidth ceilings are in [ceilings.md](ceilings.md). The pipeline contract is [README.md](README.md).

The file-agent series and the coding-agent series stay separate. Prefill is the summed suffix time. Decode is the summed sampled-token time.

## Recorded file-reading run

The rates below are from a finished file-reading run on the earlier fp4 record. The prompt reached 73810 tokens over five requests. The request set temperature 0 and left thinking on.

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

The server stops after the first turn and starts again, so the next turn loads the stored prefix. Later turns stay on that server. This run is `Qwen3.6-35B-A3B-fp8-e20`, after `moe_group` splits assignments into 16 slices. It reached 102507 prompt tokens over four requests on 23 October 2026, before the resident projections restored into an int8 stage.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 361 | miss | 123.61 tok/s (361 tok in 2.92 s) | 56.14 tok/s (31 tok in 0.55 s) |
| 2 | 46901 | hit 392 | 841.19 tok/s (46509 tok in 55.29 s) | 49.80 tok/s (568 tok in 11.41 s) |
| 3 | 47506 | prefix load 47469, hit 47469 | 13.96 tok/s (37 tok in 2.65 s) | 30.62 tok/s (40 tok in 1.31 s) |
| 4 | 102507 | hit 47546 | 436.31 tok/s (54961 tok in 125.97 s) | 39.96 tok/s (821 tok in 20.54 s) |

The summary is prefill 545.24 tok/s (101868 tok in 186.83 s) and decode 43.18 tok/s (1460 tok in 33.81 s). The previous e20 file agent, before that grouping, was 530.88 tok/s prefill over 101313 tokens and 38.12 tok/s decode over 783 tokens.

On 25 October 2026 the same command ran after the resident projections restored into an int8 stage. It reached 104740 prompt tokens over four requests.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 366 | miss | 670.63 tok/s (366 tok in 0.55 s) | 57.13 tok/s (37 tok in 0.65 s) |
| 2 | 47117 | hit 403 | 847.61 tok/s (46714 tok in 55.11 s) | 49.52 tok/s (356 tok in 7.19 s) |
| 3 | 47515 | prefix load 47473, hit 47473 | 139.57 tok/s (42 tok in 0.30 s) | 35.00 tok/s (46 tok in 1.31 s) |
| 4 | 104740 | hit 47561 | 443.18 tok/s (57179 tok in 129.02 s) | 40.36 tok/s (910 tok in 22.55 s) |

The summary is prefill 563.85 tok/s (104301 tok in 184.98 s) and decode 42.56 tok/s (1349 tok in 31.70 s). The prefill kernel ceiling for 65536 tokens is 832.32 tok/s. The decode kernel ceiling at context 65536 is 58.23 tok/s.

One resident load of the earlier fp4 record then applied four chunks in order. The first 2048-token chunk ceiling for that fp4 forward was 1521.37 tok/s.

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

The hidden suite passed. This run is `Qwen3.6-35B-A3B-fp8-e20`, on 23 October 2026, with the same 16-slice grouping. It used two requests and stopped at 638 prompt tokens.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 399 | miss | 138.43 tok/s (399 tok in 2.88 s) | 64.26 tok/s (219 tok in 3.41 s) |
| 2 | 638 | prefix load 399, hit 399 | 624.60 tok/s (239 tok in 0.38 s) | 60.02 tok/s (95 tok in 1.58 s) |

The summary is prefill 195.71 tok/s (638 tok in 3.26 s) and decode 62.93 tok/s (314 tok in 4.99 s). The previous e20 coding agent was 181.59 tok/s prefill over 572 tokens and 54.82 tok/s decode over 199 tokens.

On 25 October 2026 the same command ran after the resident projections restored into an int8 stage. The hidden suite passed. It used two requests and stopped at 663 prompt tokens.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 399 | miss | 692.67 tok/s (399 tok in 0.58 s) | 63.22 tok/s (244 tok in 3.86 s) |
| 2 | 663 | prefix load 399, hit 399 | 663.69 tok/s (264 tok in 0.40 s) | 64.00 tok/s (161 tok in 2.52 s) |

The summary is prefill 676.53 tok/s (663 tok in 0.98 s) and decode 63.48 tok/s (405 tok in 6.38 s).

On 31 October 2026 the same command ran on an M5 Pro with 48 GB, on `Qwen3.6-35B-A3B-fp8-e20`, with MTP on. The harness printed `mtp on depth 3`. The language forward verifies one draft. The hidden suite passed. It reached 3436 prompt tokens over six requests.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 1998 | miss | 698.12 tok/s (1998 tok in 2.86 s) | 44.88 tok/s (341 tok in 7.60 s) |
| 2 | 2532 | prefix load 1998, hit 1998 | 564.90 tok/s (534 tok in 0.95 s) | 44.17 tok/s (157 tok in 3.55 s) |
| 3 | 2773 | prefix load 2532, hit 2532 | 410.13 tok/s (241 tok in 0.59 s) | 45.02 tok/s (119 tok in 2.64 s) |
| 4 | 3091 | hit 2892 | 322.50 tok/s (199 tok in 0.62 s) | 43.13 tok/s (159 tok in 3.69 s) |
| 5 | 3288 | hit 3250 | 175.79 tok/s (38 tok in 0.22 s) | 41.30 tok/s (89 tok in 2.16 s) |
| 6 | 3436 | hit 3377 | 214.87 tok/s (59 tok in 0.27 s) | 40.66 tok/s (339 tok in 8.34 s) |

The summary is prefill 556.99 tok/s (3069 tok in 5.51 s) and decode 43.03 tok/s (1204 tok in 27.98 s).

Later the same day the same command ran again on that machine and record, with MTP on. The harness printed `mtp on depth 3`. The language forward verifies one draft. The hidden suite passed. It reached 3512 prompt tokens over seven requests.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 1998 | miss | 705.67 tok/s (1998 tok in 2.83 s) | 46.46 tok/s (401 tok in 8.63 s) |
| 2 | 2592 | prefix load 1998, hit 1998 | 585.40 tok/s (594 tok in 1.01 s) | 44.21 tok/s (101 tok in 2.28 s) |
| 3 | 2777 | prefix load 2592, hit 2592 | 374.21 tok/s (185 tok in 0.49 s) | 45.49 tok/s (110 tok in 2.42 s) |
| 4 | 3086 | hit 2887 | 386.56 tok/s (199 tok in 0.51 s) | 40.35 tok/s (119 tok in 2.95 s) |
| 5 | 3243 | hit 3205 | 175.93 tok/s (38 tok in 0.22 s) | 39.93 tok/s (108 tok in 2.70 s) |
| 6 | 3382 | hit 3351 | 154.98 tok/s (31 tok in 0.20 s) | 34.14 tok/s (71 tok in 2.08 s) |
| 7 | 3512 | hit 3453 | 220.53 tok/s (59 tok in 0.27 s) | 39.79 tok/s (188 tok in 4.72 s) |

The summary is prefill 561.30 tok/s (3104 tok in 5.53 s) and decode 42.59 tok/s (1098 tok in 25.78 s).

## Greedy smoke

`tests/test_qwen36_35b_a3b_smoke.py` on 31 October 2026, on an M5 Pro with 48 GB, on `Qwen3.6-35B-A3B-fp8-e20`. The greedy carrot ran twice. The second run set `mtp` and printed `mtp depth 3`. It kept 67 of 82 drafts. The instruct card and the thinking card omit `mtp`.

| Case | Prefill | Decode |
|---|---|---|
| Greedy, MTP off | 176.29 tok/s (51 tok in 0.29 s) | 68.94 tok/s (108 tok in 1.57 s) |
| Greedy, MTP on | 212.43 tok/s (51 tok in 0.24 s) | 57.54 tok/s (149 tok in 2.59 s) |
| Instruct card | 245.73 tok/s (51 tok in 0.21 s) | 63.63 tok/s (154 tok in 2.42 s) |
| Thinking card, seed 0 | 237.91 tok/s (49 tok in 0.21 s) | 64.59 tok/s (549 tok in 8.50 s) |

## File agent, 16GiB

Same instruct card as the file agent above, on the earlier fp4 record. The command passes a memory cap, so the atlas holds fewer than every expert image:

```text
uv run python -m harness --model qwen36-35b-a3b --max-memory 16GiB
```

This run reached 88347 prompt tokens over four requests.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 361 | miss | 106.23 tok/s (361 tok in 3.40 s) | 27.55 tok/s (31 tok in 1.13 s) |
| 2 | 46417 | hit 392 | 656.38 tok/s (46025 tok in 70.12 s) | 33.64 tok/s (272 tok in 8.09 s) |
| 3 | 46726 | prefix load 46689, hit 46689 | 25.23 tok/s (37 tok in 1.47 s) | 22.27 tok/s (40 tok in 1.80 s) |
| 4 | 88347 | hit 46766 | 402.77 tok/s (41581 tok in 103.24 s) | 29.08 tok/s (691 tok in 23.77 s) |

The summary is prefill 493.77 tok/s (88004 tok in 178.23 s) and decode 29.72 tok/s (1034 tok in 34.79 s).

## Coding agent, 16GiB

Same thinking card as the coding agent above, on the earlier fp4 record. The hidden suite passed. The command is:

```text
uv run python -m harness.coding --model qwen36-35b-a3b --max-memory 16GiB
```

The run used two requests and stopped at 611 prompt tokens.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 399 | miss | 140.14 tok/s (399 tok in 2.85 s) | 33.02 tok/s (192 tok in 5.82 s) |
| 2 | 611 | hit 591 | 74.87 tok/s (20 tok in 0.27 s) | 38.60 tok/s (60 tok in 1.55 s) |

The summary is prefill 134.29 tok/s (419 tok in 3.12 s) and decode 34.19 tok/s (252 tok in 7.37 s).
