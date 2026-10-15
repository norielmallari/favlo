# Console

As-built status row for Qwen3.8-Flash-Next. Scope is in [ARCHITECTURE.md](../../ARCHITECTURE.md). If this note disagrees with that file, ARCHITECTURE.md wins.

`qwen38_flash_next/console.py` owns the row. It does not import the forward. A handle is a `Progress` value: phase, done, total, unit, and an optional gauge. The request thread writes those integers. A daemon thread samples them at 10 Hz. Registration takes a lock. The sample does not.

Phases are `load`, `quantize`, `prefill`, and `decode`. Units are `tok`, `layers`, `bytes`, `shards`, and `experts`. `--console` is `auto`, `pretty`, `plain`, or `json`. `auto` is pretty on a terminal. A pipe, `NO_COLOR`, or `TERM=dumb` selects plain. Plain prints the harness lines unchanged, plus one `madmetal 0.1.0` line at startup. Pretty draws the Ingot Plate once, then one row per open handle, and prints the summary when the handle ends. JSON is NDJSON. Each object has `v` 1, `pipeline` `qwen38-flash-next`, and enough fields to stand alone. A tick's `eta` stays null until the 5-second rate window has at least 500 ms of samples.

Prefill `done` is `(layers completed + the fraction of the current layer) / layers`, multiplied by the incoming token count. It does not move backward. The live prefill tok/s is the tokens completed divided by the time since prefill started. A sample that adds no tokens lowers that average. The summary line is the same ratio over the whole prefill. A cached prompt that prefills nothing prints `prefill 0 tok/s` and does not open a bar. `serve` in pretty mode keeps one decode row for the open request, with the request id. The row closes when that request ends, including when the client drops the stream. A later prefill closes a row the request left open. A request that sets `max_tokens` draws a bar. One that does not draws a spinner. `complete` and `bf16` do not draw that row. Pretty mode ends those commands with one summary line.

`madmetal serve` is a process command. It does not take `--qwen38-flash-next`. The process listens before any record is loaded. The first request for a folder in `models/registry.json` loads that record. The same folder stays resident. A different folder closes the runtime and loads the new one. `GET /v1/models` lists the folder, the alias, and both variants. `:thinking` and `:instruct` change sampling for that call. They do not load a second record. `qwen38-flash-next` selects the default folder.

The usage object carries `timings.prompt_per_second`, `timings.predicted_per_second`, `timings.prompt_n`, and `timings.predicted_n`. `prompt_n` is the prompt length minus the cached prefix. `prompt_tokens` remains the full prompt.

`complete` and `bf16` take `--model` as a folder or an alias. `--record` is still a direct path and wins when it is set. `quantize --alias` stores the alias on the directory it writes. The derived directory name is `{source name}-madmetal-h{hot}-{plane}-c{cold}-{plane}`.
