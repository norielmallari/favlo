# Long prompt

One run of the file-reading prompt for `qwen36-35b-a3b`. Kernel and bandwidth ceilings are in [ceilings.md](ceilings.md). The pipeline contract is [README.md](README.md).

The harness starts `madmetal serve` and runs an agent until the prompt reaches 65536 tokens. The model id is the thinking variant. The agent calls `read_file` for the next repository source, largest file first, then analyzes that file. The server stops after the first task and starts again, so the next task loads the stored prefix and prefills the new suffix. Later tasks stay on that server. The command is:

```text
uv run python -m harness.long_prompt --model qwen36-35b-a3b
```

The printed summary is the measurement. The run below closed thinking and stopped each completion at the forced tool call, 13 tokens. It stopped at 85667 prompt tokens over three requests. Prefill is the summed suffix time. Decode is the summed sampled-token time.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 301 | miss | 165.29 tok/s (301 tok in 1.82 s) | 68.19 tok/s (13 tok in 0.19 s) |
| 2 | 45776 | hit 314 | 956.06 tok/s (45462 tok in 47.55 s) | 55.16 tok/s (13 tok in 0.24 s) |
| 3 | 85667 | hit 45789 | 615.28 tok/s (39878 tok in 64.81 s) | 47.39 tok/s (13 tok in 0.27 s) |

The summary is prefill 750.05 tok/s (85641 tok in 114.18 s) and decode 55.71 tok/s (39 tok in 0.70 s). The prefill kernel ceiling for 65536 tokens from position 0 is 1047.86 tok/s. The decode kernel ceiling at context 65536 is 82.97 tok/s.
