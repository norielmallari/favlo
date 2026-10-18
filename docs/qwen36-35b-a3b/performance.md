# Long prompt

One run of the file-reading prompt for `qwen36-35b-a3b`. Kernel and bandwidth ceilings are in [ceilings.md](ceilings.md). The pipeline contract is [README.md](README.md).

The harness starts `madmetal serve` and runs an agent until the prompt reaches 65536 tokens. The model id is the thinking variant. The agent calls `read_file` for the next repository source, largest file first, then analyzes that file. The server stops after the first task and starts again, so the next task loads the stored prefix and prefills the new suffix. Later tasks stay on that server. The command is:

```text
uv run python -m harness.long_prompt --model qwen36-35b-a3b
```

The printed summary is the measurement. The request sets temperature 0 and leaves thinking on. The run below continued past 65536 prompt tokens and stopped at 73810 over five requests. Prefill is the summed suffix time. Decode is the summed sampled-token time.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 359 | miss | 36.36 tok/s (359 tok in 9.87 s) | 12.16 tok/s (33 tok in 2.71 s) |
| 2 | 46355 | hit 392 | 782.95 tok/s (45963 tok in 58.71 s) | 20.95 tok/s (84 tok in 4.01 s) |
| 3 | 46474 | hit 46439 | 2.98 tok/s (35 tok in 11.73 s) | 13.74 tok/s (42 tok in 3.06 s) |
| 4 | 46583 | hit 46516 | 6.21 tok/s (67 tok in 10.80 s) | 11.33 tok/s (34 tok in 3.00 s) |
| 5 | 73810 | hit 46617 | 516.28 tok/s (27193 tok in 52.67 s) | 18.19 tok/s (70 tok in 3.85 s) |

The summary is prefill 512.01 tok/s (73617 tok in 143.78 s) and decode 15.81 tok/s (263 tok in 16.63 s). The prefill kernel ceiling for 65536 tokens from position 0 is 1047.86 tok/s. The decode kernel ceiling at context 65536 is 82.97 tok/s.
