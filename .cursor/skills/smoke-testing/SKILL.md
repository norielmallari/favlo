---
name: smoke-testing
description: Run Favlo language-model pipeline smoke and agent smoke. Pipeline smoke is greedy decode, the instruct card, the thinking card, and each reasoning effort. Agent smoke is the file agent (instruct context ladder) and the coding agent (thinking card). Use for a text model after a pipeline change, or when the language-model optimization skill points here. Do not use for image generation.
---

# Smoke testing

One pipeline. Read this skill when another skill says to smoke. Do not copy these commands into that skill.

Take the GPU with one `run`. It waits for the lock, closes the resident model when `favlo serve` is up and no request is open, runs the smoke, and releases on every exit, including failure and SIGINT. The daemon stays up. Do not kill it.

```text
uv run python -m favlo.smoke_lock run --pipeline <stem> --command smoke -- \
  uv run python -m pytest tests/<file> -s
```

A stretch of several commands in one shell uses `acquire --wait` and one `release` at the end. `acquire` without `--wait` is for tests of exit 2 and 3.

The file-agent and coding-agent commands reuse the listener on `127.0.0.1:8129` through `python -m favlo.smoke_lock attach`. They do not take the machine lock. When that pipeline is resident and its sources are newer than the worker, the model is unloaded before the run. Do not stop a daemon the user left running.

Stop at the first failure. Do not start agent smoke while a pipeline smoke is failing.

Every smoke keeps the prefill and decode tok/s. Each line has the rate, the token count, and the seconds. A pipeline smoke prints those lines from the runtime; run it with `-s` so pytest does not hide them. An agent smoke prints each request's prefill and decode line, then the summed rates. A run that drops those lines is not a finished smoke. Do not write a rate that was not printed.

## Sampling

The model field is the pipeline name. A `:instruct` or `:thinking` id pins that row and ignores `enable_thinking`.

`enable_thinking` is the only sampling parameter the agents send. A bare id with `enable_thinking` false uses the instruct card. A bare id with `enable_thinking` true uses the thinking card. Omitted temperature, top-p, top-k, and presence come from that card. Omitted reasoning effort leaves the template default. Omitted `max_tokens` does not cap the completion.

## Pipeline smoke

In-process, this package only. The reply collects 6 carrots. Temperature 0 is used only for greedy decode.

Run the carrot file that imports only this package:

| Pipeline | File |
|---|---|
| `qwen38-27b` | `tests/test_qwen38_27b_smoke.py` |
| `qwen36-35b-a3b` | `tests/test_qwen36_35b_a3b_smoke.py` |
| `qwen38-flash-next` (parked) | `pipelines/parked/tests/test_smoke.py` |
| `spark-x26-1-7b` (parked) | `pipelines/parked/tests/test_spark_smoke.py` |
| `gemma4-26b-a4b` | `tests/test_gemma4_26b_a4b_smoke.py` |

`tests/test_qwen36_35b_a3b_smoke_16gb.py` is skipped for now. It would run the 35B file's cases with the runtime capped at 16GiB. Do not run that file until the skip is removed.

The file covers, in order: greedy decode, the instruct card, the thinking card at the template default effort, then one case for each reasoning effort that template names (`low`, `medium`, `xhigh` when the template names them).

## Agent smoke

Each command reuses the listener on `127.0.0.1:8129`. Do not start a second listener.

Instruct decode and thinking-card decode stay two series. The 90% check compares the phase being optimized with the kernel ceiling on the series for that agent.

### File agent

Instruct. Opens `<pipeline>:instruct` with `enable_thinking` false. Default target 65536 tokens. Pass `--tokens` when this pipeline's prompt limit is different.

```text
uv run python -m harness --model <pipeline>
```

A Microsoft Agent Framework agent reads `src/favlo` until `prompt tokens` reaches the target. The corpus is shared package code only. After the first turn the harness unloads the resident model, so the next turn loads the prefix from disk. Later turns extend that prefix in memory. The listener stays up. The command streams each model call, then prints each request's context, `prefill`, and `decode`, then the summed rates. Exit 0 means the target was reached and the prefix reload held.

`python -m harness.file_agent` is the short create, edit, and confirm check. Pytest runs that check. It is not this rate run.

### Coding agent

Thinking. `enable_thinking` true.

```text
uv run python -m harness.coding --model <pipeline>
```

The model writes `ledger.py`. The suite that decides the pass lives outside the workspace. Exit 0 is a pass. The command prints each request's `prefill` and `decode` line with the context that request reached. Those lines are the thinking-card series.
