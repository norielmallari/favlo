# This package

This package is a language pipeline. It inherits `LanguagePipeline`. The class is the disconnect shell around one response. It does not encode a prompt, sample a token, or parse a tool call. The README under `docs/` whose folder name matches this package is a summary. The as-built detail is `ARCHITECTURE.md` in that same folder. Read that file before editing kernels.

A shell change loads the seam-checks skill. A GPU smoke loads the smoke-testing skill. Do not paste those commands here.

## Responses

One concrete class: `class Pipeline(LanguagePipeline)`. Implement `handle_responses`. That method reads a Responses body and writes a Response, including its sampler and its stream. Bind `handle_responses` to the one `Pipeline` instance. The daemon calls that instance.

This package implements the Responses API only. The listener owns the Chat Completions wrapper. `content` is the reply. `tool_calls` are the calls. `reasoning` is the trace when the model wrote `<|content_thinking|>`. `finish_reason` is `stop` or `length`.

`on_open` calls `super`. `on_finish` calls `super`. `RequestClosed` subclasses `ClientClosed`.

## Variants and effort

Registration copies `VARIANTS` and `DEFAULT_VARIANT` from `sampling.py`. The template default is `high`. `none` is the instruct card. The checkpoint template accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`, and each of those names is a variant. `thinking` is `high`. `resolve_effort` keeps a supported name. `template_effort` converts that name, or a number, to the float in `[0.0, 0.99]` that the checkpoint template writes.

`generation_config.json` publishes temperature 1.0. It does not publish top-p, top-k, or a presence penalty. Those stay unrestricted on the card. `setup` sets `preserve_thinking` true, because the template keeps an assistant `reasoning_content` when the message carries one.

## Setup

`setup` downloads `thinkingmachines/Inkling-2-Mini`, registers the folder, runs `calibrate` when `dense-fp4.safetensors` is missing or does not match `config.json`, and runs `bake` when `experts-fp4.safetensors` is missing or does not match. `--no-calibrate` skips the dense pack. `--repo` accepts another repository with the same tensor names, as `org/name` or a `huggingface.co` link. The drafter is not part of the download. There is no BF16 checkout to remove.

## Memory

This package reads `memory.budget()` when it opens. The weights are the minimum working set. A machine that cannot hold the safetensors file refuses to open. The context length is the knob it shrinks. A request keeps the prompt it named, truncated to the context that fit.

## Progress

Load, prefill, and decode report through `favlo.console`. The prefill and decode lines use `format_rate`.

## Kernels

`kernels` compiles the shader library and checks half `matmul2d`, the half GEMV, Metal fp4 `matmul2d`, the fp4 GEMV, NVFP4 dequant and its fp4 requant, the argmax and the Gumbel draw, and prefill and decode attention against a host reference. The hot path does not run a host GEMV. `bake` requantizes NVFP4 experts to Metal fp4 once and writes `experts-fp4.safetensors`. The open reads that file and the dense GPTQ pack in 8 MiB pieces. Without the expert pack, the open requantizes. The dense fp4 planes load from `dense-fp4.safetensors` when `calibrate` wrote it, and are rounded to the nearest code otherwise. The embedding stays half. A single token uses the fp4 GEMV. Prefill uses `matmul2d`. The token RMSNorm and the block mix each use one threadgroup of 256.

## Calibration

`calibrate` changes weights, so rerun the pipeline smoke and the coding agent after it. Its prompts are data in `calibration.py`. Keep them free of personal data. A change to the dense plane layout bumps `PACK_FORMAT` in `gptq.py`.
