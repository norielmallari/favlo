---
name: language-setup
description: >-
  Standardize Favlo language-pipeline setup: download the official checkpoint,
  quantize with that pipeline's defaults, register the packed folder, and
  remove the BF16 checkout unless --keep-bf16 is set. Use when adding or
  changing setup, download, quantize, registry registration, --repo, --keep-bf16,
  or the disk check for a language model.
---

# Language setup

One `setup` command prepares a language pipeline. Read this package's `AGENTS.md` and its note under `docs/` before editing. Leave every other package untouched.

## Command

`favlo --<pipeline> setup` with no repository argument:

- Downloads that pipeline's official base checkpoint.
- Quantizes with that pipeline's defaults.
- Writes the registry row for the packed folder.
- Removes the BF16 checkout.
- `--keep-bf16` leaves the checkout.

`--repo org/name` downloads that repository instead, when it uses the same tensor names. Example: `ExampleOrg/Example-Model`. The quantize, registry, and checkout steps stay the same.

`quantize` on a checkout already on disk uses the same defaults and updates the registry. A checkout under `models/` is removed unless `--keep-bf16` is set. A checkout outside `models/` stays.

## Disk

Stop before the first write when the volume cannot hold what is still missing.

- Count the published bytes not yet in the BF16 checkout.
- Count the packed-record bytes not yet in the output directory, at this pipeline's default quant.
- The same volume must hold the sum of those two sizes, because the checkout and the record sit together until setup removes the checkout.
- Two volumes are checked separately. Each one must hold its own remainder.
- A resumed run counts only the bytes still missing.
- The refusal names the GiB required and the GiB free.

## Note

The package README states the command, the default quant, and `--keep-bf16`. The as-built sizes and the registry fields stay in `ARCHITECTURE.md`.
