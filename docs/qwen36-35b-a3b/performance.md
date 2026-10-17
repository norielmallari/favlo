# Long prompt

One run of the file-reading prompt for `qwen36-35b-a3b`. Kernel and bandwidth ceilings are in [ceilings.md](ceilings.md). The pipeline contract is [README.md](README.md).

The harness starts `madmetal serve` and sends chat completions until the prompt reaches 65536 tokens. It reads repository source, largest file first, and each turn asks for `read_file`. Temperature is 0, `max_tokens` is 32, and thinking is off. The server stops after the first turn and starts again, so the second turn loads the stored prefix and prefills the new suffix. Later turns stay on that server. The command is:

```text
uv run python -m harness.long_prompt --model qwen36-35b-a3b
```

The printed summary is the measurement. This run stopped at 85667 prompt tokens over three requests. Prefill is the summed suffix time. Decode is the summed sampled-token time. Each decode line is 13 tokens.

| Request | Prompt tokens | Cache | Prefill | Decode |
|---|---:|---|---|---|
| 1 | 301 | miss | 165.29 tok/s (301 tok in 1.82 s) | 68.19 tok/s (13 tok in 0.19 s) |
| 2 | 45776 | hit 314 | 956.06 tok/s (45462 tok in 47.55 s) | 55.16 tok/s (13 tok in 0.24 s) |
| 3 | 85667 | hit 45789 | 615.28 tok/s (39878 tok in 64.81 s) | 47.39 tok/s (13 tok in 0.27 s) |

The summary is prefill 750.05 tok/s (85641 tok in 114.18 s) and decode 55.71 tok/s (39 tok in 0.70 s). The prefill kernel ceiling for 65536 tokens from position 0 is 1047.86 tok/s. The decode kernel ceiling at context 65536 is 82.97 tok/s.
