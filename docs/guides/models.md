# Download and register a model

Three pipelines are supported. A packed record is registered in `models/registry.json`. `favlo serve` lists those rows. It does not scan `models/` for a directory that was never registered.

16 GB of memory is required. 8 GB is not enough to run a model capable of real-world agentic workflows.

## Qwen3.6-35B-A3B

The official checkpoint is `Qwen/Qwen3.6-35B-A3B`. The steps are [qwen36-35b-a3b/SETUP.md](../qwen36-35b-a3b/SETUP.md).

On a terminal, `favlo serve` with an empty registry downloads that checkpoint, packs `models/Qwen3.6-35B-A3B-fp8-e20`, registers the alias `qwen36-35b-a3b`, and removes the BF16 checkout. The next `serve` continues from the files already written.

The same download and pack, without opening the socket:

```bash
favlo --qwen36-35b-a3b setup
```

`setup --repo example-org/example-model` downloads another repository that uses the same tensor names. `--repo` also accepts a `https://huggingface.co/org/name` link. With no `--name`, the folder and the registry id are the model name plus the quant format, `models/example-model-fp8-e20` by default. `--name example-model` uses that folder instead. `--expert-fp8` is a percentage from 0 to 100. The default is 20 (`-fp8-e20`). `0` appends `-fp4`. `100` appends `-fp8-e100`. The official alias stays `qwen36-35b-a3b` on the default official pack. `--keep-bf16` leaves the checkout. `delete <folder-or-alias>` removes that row, the packed directory, and a checkout still under `models/`.

`favlo --qwen36-35b-a3b quantize` does not download. It reads `models/Qwen3.6-35B-A3B-bf16`, or the checkout named by `--src`. `--expert-fp8` is a percentage from 0 to 100. The default is 20, which writes `models/Qwen3.6-35B-A3B-fp8-e20`. 0 writes `models/Qwen3.6-35B-A3B-fp4`.

## Qwen3.8-27B

The official checkpoint is `Qwen/Qwen3.8-27B`. The steps are [qwen38-27b/SETUP.md](../qwen38-27b/SETUP.md).

`favlo --qwen38-27b download` fetches that checkpoint into `models/Qwen3.8-27B-bf16` with plain HTTP. `--check` reports whether the volume has room and does not write the files. A short file is continued on the next run.

`favlo --qwen38-27b quantize` does not download. With no `--src`, it reads the Hugging Face cache snapshot for that repository. A missing cache is an error. `--src` names a local BF16 checkout, including `models/Qwen3.8-27B-bf16`. A uniform pack is `models/Qwen3.8-27B-fp4`. `--ffn-fp8 100` writes `models/Qwen3.8-27B-fp8`. Any other percentage writes `models/Qwen3.8-27B-fp8-fp4`. The record is registered under the alias `qwen38-27b`. `delete <folder-or-alias>` removes that row, the packed directory, and a checkout still under `models/`.

An idle `favlo serve` may start that download when the machine has at least 16 GB of memory and the volume has room. It asks whether to remind them again. `download_suggestions` turns those reminders off or back on. Packing the record stays `quantize`.

## Qwen3.8-Flash-Next

The official checkpoint is `Qwen/Qwen3.8-Flash-Next`.

`favlo --qwen38-flash-next quantize` does not download. `--source` is required, and that directory must contain `config.json`. `--alias` sets the registry alias. `--output` sets the record directory. When `--output` is omitted, the directory name comes from the expert widths in the source.

## Z-Image-Turbo

The official checkpoint is `Tongyi-MAI/Z-Image-Turbo`. The steps are [z-image-turbo/SETUP.md](../z-image-turbo/SETUP.md).

`favlo --z-image-turbo setup` downloads that repository into `models/Z-Image-Turbo`. `quantize` packs `models/Z-Image-Turbo-fp4`. `quantize --int8` packs `models/Z-Image-Turbo-int8` from that checkout. `--weights` is one `.safetensors` file or a directory of shards. With no `--name`, the folder is that file or directory name plus `-fp4`, or `-int8` when `--int8` is set. `--name` is the registry id.

## Wan 2.2 I2V

The official checkpoint is `Wan-AI/Wan2.2-I2V-A14B`. The text encoder is `google/umt5-xxl`. The steps are [wan22-i2v/SETUP.md](../wan22-i2v/SETUP.md).

`favlo --wan22-i2v setup` downloads the VAE, the umT5 encoder and tokenizer, and the two expert files. `setup --umt5 example-org/example-model` downloads that text encoder instead. `quantize --int8` packs `models/Wan2.2-I2V-A14B-int8`. That folder is the served record. `quantize --high` and `--low` with `--fp8-percent 50` pack `models/Wan2.2-I2V-A14B-fp8-e50`. The default percent is 20, which writes `models/Wan2.2-I2V-A14B-fp4`. `--name` is that packed folder and the registry id. The row stores the packed directory, and `--high-dir` and `--low-dir` store the quantized expert directories. `quantize-text` writes `models/google/umt5-xxl-int8` and stores that directory on the row. A clip encodes with that directory. The BF16 checkout is the quantize source. Another pipeline can use the same folder.

## ACE-Step

The official checkpoint is `ACE-Step/acestep-v15-xl-turbo`. `setup` downloads the Diffusers split `ACE-Step/acestep-v15-xl-turbo-diffusers`. The steps are [ace-step-15-turbo/SETUP.md](../ace-step-15-turbo/SETUP.md).

`favlo --ace-step-15-turbo setup` writes `models/acestep-v15-xl-turbo`. That checkout holds the text encoder, the tokenizer, the condition encoder, the transformer, and the VAE. `quantize` packs `models/acestep-v15-xl-turbo-fp8-fp4` and stores that directory on the row. `delete <folder>` removes that row and the packed directory.

## Default chat model

`models/registry.json` may set top-level `"default"` to a variant id such as `qwen36-35b-a3b:instruct`. The first registered language model writes that field to its instruct variant when it is still unset. Agent harnesses and the terminal read it when no model flag is passed.

## Register a finished record

`models` lists the registry. `models add` registers a directory under `models/` whose manifest is already complete. `models alias` renames the alias of a folder. The same subcommands exist for each pipeline flag:

```bash
favlo --qwen36-35b-a3b models
favlo --qwen36-35b-a3b models add <folder> --alias <name>
favlo --qwen36-35b-a3b models alias <folder> <name>
favlo --qwen38-27b models
favlo --qwen38-flash-next models
```

What the running server can see:

```bash
curl http://127.0.0.1:8129/v1/models
```
