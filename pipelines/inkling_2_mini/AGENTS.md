# This package

The pipeline entry is `inkling_2_mini.portable`. `metal41` is the Metal device that entry opens. Serve, the CLI, and the carrot smoke import `portable`. They do not construct the device runtime.

This package is a language pipeline. It inherits `LanguagePipeline`. The class is the disconnect shell around one response. It does not encode a prompt, sample a token, or parse a tool call. The README under `docs/` whose folder name matches this package is a summary. The as-built detail is `ARCHITECTURE.md` in that same folder. Read that file before editing kernels.

A shell change loads the seam-checks skill. A GPU smoke loads the smoke-testing skill. Do not paste those commands here.

## Responses

One concrete class: `class Pipeline(LanguagePipeline)`. Implement `handle_responses`. That method reads a Responses body and writes a Response, including its sampler and its stream. Bind `handle_responses` to the one `Pipeline` instance. The daemon calls that instance.

This package implements the Responses API only. The listener owns the Chat Completions wrapper. `content` is the reply. `tool_calls` are the calls. `reasoning` is the trace when the model wrote `<|content_thinking|>`. `finish_reason` is `stop` or `length`.

`on_open` calls `super`. `on_finish` calls `super`. `RequestClosed` subclasses `ClientClosed`.

## Variants and effort

Registration copies `CARD` from `sampling.py` onto the row. The package has no sampling suffixes. `generation_config.json` publishes temperature 1.0, and the published run settings are top-p 1.0 and top-k 0 with no presence penalty. A request sets the effort with `reasoning_effort`. The row default is `high`. The template accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`. `resolve_effort` keeps a supported name. `template_effort` converts that name, or a number, to the float in `[0.0, 0.99]` that the checkpoint template writes. `setup` sets `preserve_thinking` true. The template has no such switch: it renders an assistant `reasoning_content` when the field is present. A request that sets the field false removes it before render.

## Setup

`setup` downloads `thinkingmachines/Inkling-2-Mini` into `Inkling-2-Mini-fp4-gptq-foem`, rewrites bf16 tensors as float16, bakes any missing pack, and seals a raw record. `resident.bin` holds the float language planes, including float16 attention and the float16 output projection. `experts.bin` holds the expert payloads. `vision.bin` and `mtp.bin` are separate files. `manifest.json` indexes them. The safetensors weight files are removed after the record matches `config.json`. `calibrate` replaces layers 0–19 with GPTQ-FOEM, leaves layers 20–23 as MXFP8, and seals the record again. It reads the language shard, so a later run downloads the checkpoint again. The head is pread as float16 from `resident.bin`. `--repo` accepts another repository with the same tensor names, as `org/name` or a `huggingface.co` link. There is no separate BF16 checkout to remove. `quantize --src <folder> --name <id> --expert-fp8 <percent>` scores that BF16 checkpoint in place, records that folder as the registry source, stores attention and the output projection as float16, stores the top percent of experts as MXFP8 and the rest as Metal fp4, and seals a new folder. `--promote firing` ranks `firing.sqlite` instead. A new seal stores those float16 planes. The official folder's expert calibration is unchanged. A finished request writes `~/.favlo/telemetry/<folder>/firing.sqlite`. Hits follow the 48 visits, 384 per token.

## Memory

This package reads `memory.budget()` when it opens. The weights are the minimum working set. A machine that cannot hold `resident.bin` and `experts.bin` refuses to open. The context length is the knob it shrinks. A request keeps the prompt it named, truncated to the context that fit. Eight full-attention visits keep a half key/value plane for that context. Each of the other forty keeps a 1024-token ring. An image reads `vision.bin` for that request and releases it after the rows are written.

## Progress

Load, prefill, and decode report through `favlo.console`. The load row stays open through the dense pack, the expert pack, and the working-set allocation. Its close line is the loaded summary. Prefill opens only for an uncached suffix and advances after each chunk. Decode opens when `live_decode` is set and counts each kept token. Both close with `format_rate`. A cancel closes a row that is still open. Do not `print` a tok/s line.

## Kernels

`kernels` compiles the shader library and checks half `matmul2d`, the half GEMV, Metal fp4 `matmul2d`, the fp4 GEMV, Metal fp8 `matmul2d`, the fp8 output GEMV, NVFP4 dequant, its fp4 requant, the NVFP4 GEMV, and the NVFP4 prefill tile, the argmax and the Gumbel draw, and prefill and decode attention against a host reference. The hot path does not run a host GEMV. `bake` requantizes NVFP4 experts to Metal fp4. Attention and the output projection stay float16. The seal copies those payloads into `experts.bin` and `resident.bin`. The open preads those files and the float16 head in 8 MiB pieces. The embedding, the head, the attention plane, and the output projection stay float16. Those two products use `matmul2d` on prefill. One token uses the half GEMV. A record that still stores them as nearest E4M3 uses the fp8 GEMV for one token. The token RMSNorm and the block mix each use one threadgroup of 256.

## Calibration

`bake` requantizes NVFP4 experts to Metal fp4. Attention and the output projection stay float16. The seal stores those planes in `resident.bin` and the experts in `experts.bin`. The head stays half in `resident.bin`. `calibrate` reads the language shard, so it runs before that shard is removed.
