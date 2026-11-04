# Console

As-built status row for the process. The map is [../ARCHITECTURE.md](../ARCHITECTURE.md). How a shipped pipeline fills a phase is that pipeline's note.

`src/favlo/console.py` imports nothing from a pipeline. The listener does not import a pipeline until a request names one. That row's entry file provides `open_runtime` and `handle_responses`. Pipelines report through `begin`, `count`, `end`, `note`, and `format_rate`. They do not print their own status rows. Which calls each pipeline uses is [progress-reporting.md](progress-reporting.md).

`begin`, `count`, `end`, and `note` enqueue onto the `favlo-console` thread and return. That thread formats the line and writes the pipe or the terminal. A count copies `done` and the producer's elapsed time into the fact. The caller does not wait for the paint. A slow reader does not stall the next GPU step. Tests that need the sync path pass `start_thread=False`.

## Modes

`--console auto` is pretty when stderr is a terminal, and plain when it is a pipe, when `NO_COLOR` is set, or when `TERM` is `dumb`. Progress goes to stderr. [Command Line Interface Guidelines](https://clig.dev/) put the primary result on stdout and messaging on stderr, animate only on a TTY, and hide detail behind a live row while the work runs. Pretty and plain prefix each durable stderr line with the local wall clock, `[11:46:23] prefill 4.64 tok/s (58 tok in 12.50s)`. The recorded body and `format_rate` stay without that prefix, so the harness still parses them. Live rows do not carry a stamp. JSON is one NDJSON object per event, schema version 1.

`pretty` draws the Ingot Plate once, then one row per open phase at 10 Hz. A determinate phase has a bar. An open-ended decode has a spinner. Pretty folds `prompt tokens`, `cache hit`, `cache miss`, `prefix load`, `mtp`, `incoming tokens`, `output tokens`, and `context` into the daemon log. While a listener request is open, prefill, decode, sample, and the prefix-store line wait. After that request closes they print as one durable line. A later request is a new line. Load, download, and quantize still print when that row closes. A command that never opens a request, such as `complete`, still prints each summary when the row closes. A failure uses `report`, which finishes the live rows first.

The rate is the work completed divided by the time since that phase started. The clock is `perf_counter`. A tick carries `elapsed`, measured on the producer from the phase open to that count, and `ts` on that same counter. The painter adds its own time since it applied the tick, and it does not subtract a producer timestamp from its counter. `eta` stays null until 500 ms have elapsed. The painted rate, ETA, and gauge are an exponential smooth with a 1.5 s time constant. The phase summary and a JSON tick keep the raw rate. While the queue is empty, a pretty console redraws an open row every 100 ms. When the queue still holds events, a count redraw waits 300 ms and the next paint shows the latest count. `begin`, `end`, and a note paint with their event. Phases are `load`, `download`, `quantize`, `prefill`, `decode`, and `sample`. Units are `tok`, `layers`, `bytes`, `shards`, `experts`, `stages`, `files`, and `blocks`. A `sample` row counts the work that pipeline's note names. When that row has a detail, the painted label is the detail.

The pipeline process keeps a JSON console. Its thread writes each queued fact onto the protocol stream and does not paint. Each pipeline reader enqueues those facts. One CLI thread runs them through `src/favlo/progress.py`, which keys a row by the pipeline process, the phase, and the request, and drops a count for a phase that has not begun or has already ended. That same thread paints stderr. The load row is painted while a completion reads the record. Plain does not open a row. A detached daemon writes the same events as JSON.

## Notes

`note` writes the canonical diagnostic lines. `format_rate` writes the close line the harness parses, `prefill 4.64 tok/s (58 tok in 12.50s)`. A pipeline that still calls `log` with those same strings stays compatible.
