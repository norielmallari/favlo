---
name: smoke-testing
description: Run Favlo language-model pipeline smoke and agent smoke. Pipeline smoke is greedy decode, the instruct card, the thinking card, and each reasoning effort. Agent smoke is the file agent (instruct context ladder) and the coding agent (thinking card). Use for a text model after a pipeline change, or when the language-model optimization skill points here. Do not use for image generation.
---

# Smoke testing

One pipeline. Read this skill when another skill says to smoke. Do not copy these commands into that skill.

Take the GPU with the smoke lock before a command that loads a checkpoint. Run acquire, the smoke, and release in the same shell. Acquire closes the resident model when `favlo serve` is up and no request is open, then writes `artifacts/pipeline-test.lock`. The daemon stays up. Do not kill it. Exit 2 means a request is open: wait and run acquire again. Exit 3 means another live pid holds the lock: wait. Release on failure too.

```text
uv run python -m favlo.smoke_lock acquire --pipeline <stem> --command smoke
uv run python -m pytest tests/<file> -s
uv run python -m favlo.smoke_lock release
```

The file-agent and coding-agent commands start their own serve. They still need port 8129 free. Do not point them at a listener already on that port, and do not stop a daemon the user left running.

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
| `qwen38-flash-next` | `tests/test_smoke.py` |

`tests/test_qwen36_35b_a3b_smoke_16gb.py` runs the 35B file's cases with the runtime capped at 16GiB. That file is the `<file>` in the lock commands above.

The file covers, in order: greedy decode, the instruct card, the thinking card at the template default effort, then one case for each reasoning effort that template names (`low`, `medium`, `xhigh` when the template names them).

## Agent smoke

Each command starts its own `serve`. Do not point it at a listener already on `8129`.

Instruct decode and thinking-card decode stay two series. The 90% check compares the phase being optimized with the kernel ceiling on the series for that agent.

### File agent

Instruct. `enable_thinking` false. Default target 65536 tokens. Pass `--tokens` when this pipeline's prompt limit is different.

```text
uv run python -m harness --model <pipeline>
```

The session reads repository source until `prompt tokens` reaches the target. The server stops after the first turn and a new server starts, so the next turn loads the prefix from disk. Later turns extend that prefix in memory. The command prints each request's context, `prefill`, and `decode`, then the summed rates. Exit 0 means the target was reached and the prefix reload held.

`python -m harness.file_agent` is the short create, edit, and confirm check. Pytest runs that check. It is not this rate run.

### Coding agent

Thinking. `enable_thinking` true.

```text
uv run python -m harness.coding --model <pipeline>
```

The model writes `ledger.py`. The suite that decides the pass lives outside the workspace. Exit 0 is a pass. The command prints each request's `prefill` and `decode` line with the context that request reached. Those lines are the thinking-card series.
