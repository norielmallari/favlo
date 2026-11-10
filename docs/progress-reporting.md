# Progress reporting

The console accepts a progress report from any pipeline. `begin`, `count`, `end`, and `note` do not branch on the kind. The row key is the pipeline process, the phase, and the request, so an embeddings load and a language-model prefill do not share a row. The phase list and the unit list are shared. The unit says what the count is.

A pipeline calls these on the console it already holds. `publish_count` is the helper that calls `count`. The caller posts onto `favlo.host_progress` and returns. That host thread builds the fact and enqueues onto the console queue the CLI already drains. The CLI thread sanitizes the fact and paints it. The map of the row itself is [console.md](console.md).

Not every pipeline calls the API. The matrix below is the calls each pipeline makes now. A blank serve path means that pipeline does not open a row.

## Calls

| Call | Records |
|---|---|
| `begin(phase, total, unit, detail, request)` | Opens one row. `request` keeps two requests of the same phase apart. |
| `count(progress, done)` | One count for that row. The fact carries `done` and the elapsed time since `begin`, measured on the producer clock. |
| `end(progress, summary)` | Closes the row. `summary` is the durable line. |
| `note(topic, **fields)` | One diagnostic line. The topic is from the list below. |
| `format_rate(phase, count, seconds, unit)` | The close line a language pipeline passes to `end`. The harness parses `prefill 4.64 tok/s (58 tok in 12.50s)`. |

Phases are `load`, `download`, `quantize`, `prefill`, `decode`, and `sample`.

Units are `tok`, `layers`, `bytes`, `shards`, `experts`, `stages`, `files`, and `blocks`.

Note topics are `prompt`, `cache`, `prefix_load`, `prefix_store`, `prefix_store_failed`, `mtp_depth`, `mtp`, `incoming`, `output`, and `context`.

## Served pipelines

| Pipeline | Kind | `begin` | `count` | `end` | `note` | `format_rate` |
|---|---|---|---|---|---|---|
| `qwen36_35b_a3b` | language | `load` bytes, `prefill` tok, `decode` tok. Commands: `download` bytes, `quantize` bytes | load, prefill, decode, and those command rows | yes | `prompt`, `cache`, `prefix_load`, `prefix_store`, `prefix_store_failed`, `mtp_depth`, `mtp`, `context` | prefill, decode |
| `qwen38_27b` | language | `load` bytes, `prefill` tok, `decode` tok. Command: `download` bytes | load, prefill, decode, and the download row | yes | `prompt`, `cache`, `prefix_load`, `prefix_store`, `prefix_store_failed`, `mtp_depth`, `mtp`, `context` | prefill, decode |
| `gemma4_26b_a4b` | language | `load` bytes, `prefill` tok, `decode` tok | load, prefill, decode | yes | `prompt`, `cache`, `prefix_load`, `prefix_store`, `prefix_store_failed`, `context` | prefill, decode |
| `qwen35_4b` | language | `load` bytes, `prefill` tok, `decode` tok. Command: `quantize` files | load, prefill, decode, and the quantize row | yes | `prompt`, `cache`, `prefix_load`, `prefix_store`, `prefix_store_failed`, `mtp_depth`, `mtp` | prefill, decode |
| `z_image_turbo` | image | `load` bytes, `sample` layers | load, sample | yes | no | no |
| `wan22_i2v` | video | `load` files, `sample` layers, `sample` blocks, `decode` blocks | the open stage row | yes | no | no |
| `ace_step_15_turbo` | music | no | no | no | no | no |
| `voxcpm2` | voice | no | no | no | no | no |
| `bge_m3` | embeddings | no | no | no | no | no |
| `umt5_xxl` (library) | text encoder pack | no | no | no | no | no |
| `wan22_ti2v_5b` | video | no | no | no | no | no |

`wan22_i2v` stage names on a `sample` or `decode` row are `compile`, `text`, `encode`, `high`, `low`, and `decode`. The job percent written onto the clip stays on the job dict. It is not a console count.

`ace_step_15_turbo` increments a job percent in its own object. That object is not `begin` or `count`. `voxcpm2` keeps the console and does not call it. `bge_m3`, `umt5_xxl`, and `wan22_ti2v_5b` drop the console argument.

An open-ended language `decode` passes `total` 0. The row is a count, not a fraction of the request cap.
