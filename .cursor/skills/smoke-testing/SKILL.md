---
name: smoke-testing
description: Run Favlo language-model pipeline smoke and agent smoke. Pipeline smoke is greedy decode, the instruct card, the thinking card, and each reasoning effort. Agent smoke is the file agent (instruct context ladder) and the coding agent (thinking card). Use for a text model after a pipeline change, or when the language-model optimization skill points here. Do not use for image generation.
---

# Smoke testing

One pipeline. Read this skill when another skill says to smoke. Do not copy these commands into that skill.

Wait until the machine is free before a command that compiles shaders, dispatches Metal, loads a checkpoint, quantizes, or starts `favlo serve` in its own process. That includes `pytest`, `kernels`, `quantize`, `complete`, `bf16`, `probe`, and `matmul`. Editing the write set does not wait. Reading another package does not.

The machine is busy when `artifacts/pipeline-test.lock` exists and the `pid` in it is still running. A running `favlo serve` is not itself a reason to wait. The file is gitignored. Do not commit it.

`run` waits until this shell owns the lock, runs the command, and releases on every exit, including a failed command and SIGINT. When the daemon is up and no request is open, it closes the resident model and holds chat until that release. The daemon stays up. Do not kill the server, and do not kill the process that holds the lock. Do not sleep and retry acquire outside the script.

```text
uv run python -m favlo.smoke_lock run --pipeline <stem> --command smoke -- \
  uv run python -m pytest tests/<file> -s
uv run python -m favlo.smoke_lock attach --pipeline <stem> --command <name> -- <command...>
```

A command that calls the listener uses `attach`. It waits until the lock is clear, reuses `127.0.0.1:8129`, and does not take the lock. It starts the daemon only when that port is closed, and stops it only when this call started it. When `--pipeline` is the resident model and that package's sources are newer than its worker, `attach` closes the model before the command.

A stretch of several commands in one shell uses one blocking acquire and one release at the end. `acquire` without `--wait` is for tests of exit 2 and 3. Agents do not use it.

```text
uv run python -m favlo.smoke_lock acquire --wait --pipeline <stem> --command <name>
uv run python -m favlo.smoke_lock release
```

A lock whose `pid` is gone is stale. `release --stale` removes it and resumes chat. Do not delete the lock file by hand. Do not use `kill`, `stop_pid`, or `POST /v1/pipelines/close` to end a hold. Close during a live hold is 409. A chat during the hold receives 503. After release, the next chat loads a model again.

The lock is one JSON object. `pipeline` is the flag stem. `pid` is the shell that ran `run` or `acquire --wait`, not the short-lived Python process. `command` is the short name of that test. Do not write a name, home directory, hostname, or device path.

```json
{"pipeline": "example-model", "pid": 12345, "command": "kernels"}
```

The file-agent and coding-agent commands reuse the listener on `127.0.0.1:8129` through `python -m favlo.smoke_lock attach`. They do not take the machine lock. When that pipeline is resident and its sources are newer than the worker, the model is unloaded before the run. Do not stop a daemon the user left running.

Stop at the first failure. Do not start agent smoke while a pipeline smoke is failing.

Every smoke keeps the prefill and decode tok/s. Each line has the rate, the token count, and the seconds. A pipeline smoke prints those lines from the runtime; run it with `-s` so pytest does not hide them. An agent smoke prints each request's prefill and decode line, then the summed rates. A run that drops those lines is not a finished smoke. Do not write a rate that was not printed.

## Sampling

The model field is the pipeline name. A `:instruct` or `:thinking` id pins that row and ignores `enable_thinking`.

`enable_thinking` is the only sampling parameter the agents send. A bare id with `enable_thinking` false uses the instruct card. A bare id with `enable_thinking` true uses the thinking card. Omitted temperature, top-p, top-k, and presence come from that card. Omitted reasoning effort leaves the template default. Omitted `max_tokens` does not cap the completion.

## Pipeline smoke

In-process, this package only. The reply collects 6 carrots. Temperature 0 is used only for greedy decode.

Run the carrot file that imports only this package: `tests/test_<pipeline>_smoke.py`. A memory-capped variant of that file stays skipped until its skip is removed.

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
