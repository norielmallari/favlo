# Long prompt

One run of the file-reading prompt for `qwen38-flash-next`. Kernel and bandwidth ceilings are in [ceilings.md](ceilings.md). The pipeline contract is [architecture.md](architecture.md).

The harness starts `madmetal serve` and runs an agent until the prompt reaches 65536 tokens. The model id is the thinking variant. The agent calls `read_file` for the next repository source, largest file first, then analyzes that file. The server stops after the first task and starts again, so the next task loads the stored prefix and prefills the new suffix. Later tasks stay on that server. The command is:

```text
uv run python -m harness.long_prompt --model qwen38-flash-next
```

The printed summary is the measurement. The request sets temperature 0 and leaves thinking on. The run below continued past 65536 prompt tokens and stopped at 92739 over four requests. Prefill is the summed suffix time. Decode is the summed sampled-token time.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 397 | hit 397 | prefix | 4.41 tok/s (31 tok in 7.03 s) |
| 2 | 46485 | hit 428 | 214.86 tok/s (46057 tok in 214.36 s) | 5.18 tok/s (128 tok in 24.73 s) |
| 3 | 46651 | hit 46613 | 7.21 tok/s (38 tok in 5.27 s) | 4.05 tok/s (31 tok in 7.65 s) |
| 4 | 92739 | hit 46682 | 191.68 tok/s (46057 tok in 240.28 s) | 4.41 tok/s (128 tok in 29.01 s) |

The summary is prefill 200.37 tok/s (92152 tok in 459.91 s) and decode 4.65 tok/s (318 tok in 68.42 s). The prefill kernel ceiling for one 2048-token chunk, routed experts excluded, is 518.4 tok/s. The decode kernel ceiling is 12.67 tok/s at 32768 tokens and 11.20 tok/s at 262144 tokens. Both assume the ten experts per layer are already resident.
