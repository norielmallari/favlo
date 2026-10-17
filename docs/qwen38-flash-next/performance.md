# Long prompt

One run of the file-reading prompt for `qwen38-flash-next`. Kernel and bandwidth ceilings are in [ceilings.md](ceilings.md). The pipeline contract is [architecture.md](architecture.md).

The harness starts `madmetal serve` and runs an agent until the prompt reaches 65536 tokens. The model id is the thinking variant. The agent calls `read_file` for the next repository source, largest file first, then analyzes that file. The server stops after the first task and starts again, so the next task loads the stored prefix and prefills the new suffix. Later tasks stay on that server. The command is:

```text
uv run python -m harness.long_prompt --model qwen38-flash-next
```

The printed summary is the measurement. The run below closed thinking and stopped each completion at the forced tool call, 13 tokens. It stopped at 85930 prompt tokens over three requests. Prefill is the summed suffix time. Decode is the summed sampled-token time.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 301 | miss | 52.10 tok/s (301 tok in 5.78 s) | 9.10 tok/s (13 tok in 1.43 s) |
| 2 | 46039 | hit 314 | 229.65 tok/s (45725 tok in 199.10 s) | 4.10 tok/s (13 tok in 3.17 s) |
| 3 | 85930 | hit 46052 | 202.99 tok/s (39878 tok in 196.46 s) | 4.48 tok/s (13 tok in 2.90 s) |

The summary is prefill 214.04 tok/s (85904 tok in 401.34 s) and decode 5.20 tok/s (39 tok in 7.50 s). The prefill kernel ceiling for one 2048-token chunk, routed experts excluded, is 518.4 tok/s. The decode kernel ceiling is 12.67 tok/s at 32768 tokens and 11.20 tok/s at 262144 tokens. Both assume the ten experts per layer are already resident.
