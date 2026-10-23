# Console

As-built status row for the process. The map is [../ARCHITECTURE.md](../ARCHITECTURE.md). How a pipeline fills a phase is that pipeline's note. Flash Next's binding is [qwen38-flash-next/console.md](qwen38-flash-next/console.md).

`src/favlo/console.py` imports nothing from a pipeline. The listener does not import a pipeline until a request names one. That row's entry file provides `open_runtime` and `handle_chat`.

`--console auto` is pretty when stderr is a terminal, and plain when it is a pipe, when `NO_COLOR` is set, or when `TERM` is `dumb`. `plain` prints the lines the harness parses. `pretty` draws the Ingot Plate once, then one row per open phase at 10 Hz. Phases are `load`, `download`, `quantize`, `prefill`, `decode`, and `sample`. Units are `tok`, `layers`, `bytes`, `shards`, `experts`, and `stages`. A `sample` row counts the work that pipeline's note names. `json` writes one NDJSON object per event, schema version 1. The daemon console is JSON and emits those events, including ticks. A terminal `favlo serve` paints each open phase from that stream. The load row is painted while a completion reads the record. Plain does not open a row.

A chat reply is printed after that turn's telemetry. The prompt `you:` starts on the next line. A completion from another client paints on this terminal. When that request closes, `you:` is printed again.
