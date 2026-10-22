# Download and register a model

Three pipelines are supported. A packed record is registered in `models/registry.json`. `favlo serve` lists those rows. It does not scan `models/` for a directory that was never registered.

16 GB of memory is required. 8 GB is not enough to run a model capable of real-world agentic workflows.

## Qwen3.6-35B-A3B

The official checkpoint is `Qwen/Qwen3.6-35B-A3B`.

On a terminal, `favlo serve` with an empty registry downloads that checkpoint, packs `models/Qwen3.6-35B-A3B-fp8-e20`, registers the alias `qwen36-35b-a3b`, and removes the BF16 checkout. The next `serve` continues from the files already written.

The same download and pack, without opening the socket:

```bash
favlo --qwen36-35b-a3b setup
```

`favlo --qwen36-35b-a3b quantize` does not download. It reads `models/Qwen3.6-35B-A3B-bf16`, or the checkout named by `--src`. `--expert-fp8` is a percentage from 0 to 100. The default is 20, which writes `models/Qwen3.6-35B-A3B-fp8-e20`. 0 writes `models/Qwen3.6-35B-A3B-fp4`.

## Qwen3.8-27B

The official checkpoint is `Qwen/Qwen3.8-27B`.

`favlo --qwen38-27b download` fetches that checkpoint into `models/Qwen3.8-27B-bf16` with plain HTTP. `--check` reports whether the volume has room and does not write the files. A short file is continued on the next run.

`favlo --qwen38-27b quantize` does not download. With no `--src`, it reads the Hugging Face cache snapshot for that repository. A missing cache is an error. `--src` names a local BF16 checkout, including `models/Qwen3.8-27B-bf16`. A uniform pack is `models/Qwen3.8-27B-fp4`. `--ffn-fp8 100` writes `models/Qwen3.8-27B-fp8`. Any other percentage writes `models/Qwen3.8-27B-fp8-fp4`. The record is registered under the alias `qwen38-27b`.

An idle `favlo serve` may start that download when the machine has at least 16 GB of memory and the volume has room. It asks whether to remind them again. `download_suggestions` turns those reminders off or back on. Packing the record stays `quantize`.

## Qwen3.8-Flash-Next

The official checkpoint is `Qwen/Qwen3.8-Flash-Next`.

`favlo --qwen38-flash-next quantize` does not download. `--source` is required, and that directory must contain `config.json`. `--alias` sets the registry alias. `--output` sets the record directory. When `--output` is omitted, the directory name comes from the expert widths in the source.

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
