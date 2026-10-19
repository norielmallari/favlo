# Harness

As-built clients that score a running `favlo serve`. The map is [../ARCHITECTURE.md](../ARCHITECTURE.md). A pipeline's scores live in that pipeline's `benchmarks.md`.

`python -m harness.benchmarks --model <id>` scores IFEval, GSM8K, and HumanEval against `serve`. The `model` field is that id. Thinking stays off. Omitted sampling fields are filled from `GET /v1/models` row `<id>:instruct`. A CLI flag overrides that field. The checker reads `message.content`. `--limit 30` scores the first 30 rows and writes the smoke section. `--limit 100` writes the sample of 100. A run with no limit scores the smoke, then the full sets. That run is not part of `pytest`. Each finished suite is written into `docs/<pipeline>/benchmarks.md`.

`pytest` includes the file agent: it starts `serve` and checks that create, edit, and confirm land in the workspace. `python -m harness.file_agent` is that client against an already running server.

`python -m harness --model <id>` starts `serve`, reads repository files one turn at a time until the prompt reaches the token target (default 65536), and prints the server's prefill and decode rates. The model field is the pipeline name. The request sets `enable_thinking` false and leaves every other sampling field unset, so the listener fills the instruct card. The server stops after the first turn and starts another, so the second turn loads the prefix from disk. Later turns stay on that server. That run is not part of `pytest`.

`python -m harness.coding --model <id>` starts `serve` and sets `enable_thinking` true on that same bare id. The model writes `ledger.py` through tools. A pytest file outside the workspace is the result. Exit 0 is a pass. The command prints each request's prefill and decode line. That series stays separate from the file agent's. That run is not part of `pytest`.
