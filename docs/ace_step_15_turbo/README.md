# ACE-Step 1.5 XL Turbo

Stereo music from a caption and optional lyrics.

The official base model is [ACE-Step/acestep-v15-xl-turbo](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo). Favlo reads the Diffusers split [ACE-Step/acestep-v15-xl-turbo-diffusers](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers). Both are under the MIT License. The served pack is `acestep-v15-xl-turbo-fp8-fp4`.

With `favlo serve` listening, clips come from `favlo media --pipeline ace_step_15_turbo` or the published `ace_step_15_turbo__generate_music` MCP tool.

- [ARCHITECTURE.md](ARCHITECTURE.md) is the as-built record: packed weights, registry, listener contract, forward path, smoke, and citations.
- [research/dit-mlp-fp4-error.md](research/dit-mlp-fp4-error.md) records DiT MLP fp4 layer scores.

- The default length is 30 seconds.
- A length is from 10 seconds through 10 minutes.
- The step count is 8.

```text
uv run favlo --ace_step_15_turbo kernels
```

```text
uv run favlo media --pipeline ace_step_15_turbo prompt="bright piano, soft drums" seconds=10
uv run favlo media --pipeline ace_step_15_turbo --model acestep-v15-xl-turbo-fp8-fp4 --out pack.json prompt="bright piano, soft drums" duration=10
```

## Setup

How to download, pack, and make a clip with `ace_step_15_turbo`.

- The kernel contract is [ARCHITECTURE.md](ARCHITECTURE.md).
- The listener is [../serve.md](../serve.md).
- Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [ACE-Step/acestep-v15-xl-turbo](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo). `setup` downloads the Diffusers split [ACE-Step/acestep-v15-xl-turbo-diffusers](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers). Both are under the MIT License.

### What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the checkout and, after packing, the packed record beside it

The checkout stays on disk after packing. That checkout holds the text encoder, the tokenizer, the condition encoder, the transformer, and the VAE. `setup` does not download a second repository.

### Install

From a checkout of this repository:

```text
uv sync --extra dev
```

This checkout is release 0.6.0. PyPI publishes the runtime under `src/favlo`. The pipelines stay in this checkout, so later commands use `uv run favlo` from that checkout.

### Download

```text
uv run favlo --ace_step_15_turbo setup
```

That command downloads the Diffusers tree into `models/acestep-v15-xl-turbo` with plain HTTP.

- A public file does not need a token.
- A file that already has its published size is left in place.
- A short file is downloaded again.
- `setup` skips `silence_latent.pt`.
- The silence latent used at generation is a tensor in `condition_encoder/diffusion_pytorch_model.safetensors`.
- `setup` does not pack a record and does not register one.
- `--dest` names the checkout directory.
- With no `--dest`, the directory is `models/acestep-v15-xl-turbo`.

### Pack

```text
uv run favlo --ace_step_15_turbo quantize
```

That command reads `models/acestep-v15-xl-turbo` and writes `models/acestep-v15-xl-turbo-fp8-fp4`.

- The client model field is `acestep-v15-xl-turbo-fp8-fp4`.
- The VAE encoder is not packed.
- The tokenizer is copied from `tokenizer/tokenizer.json` in the checkout.
- A finished directory has `manifest.json` with `complete: true`.
- Running `quantize` again on that directory returns it.
- `--source` names the checkout.
- `--dest` names the packed directory.
- With no flags, those are the two directories above.

### Delete

```text
uv run favlo --ace_step_15_turbo delete acestep-v15-xl-turbo-fp8-fp4
```

`delete` takes the folder name. The command removes that registry row and the packed directory stored on the row when that directory is under `models/`.

- A checkout that is already gone stays gone.
- A checkout that is still under `models/` and is not used by another row is removed with the pack.
- A checkout outside `models/` is left in place.

### Make a clip

Start the listener:

```text
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. The model id is `acestep-v15-xl-turbo-fp8-fp4`. A music client calls `generate_music`.

- The default length is 30 seconds.
- A length is from 10 seconds through 10 minutes.
- The step count is 8.

```text
uv run favlo media --pipeline ace_step_15_turbo prompt="bright piano, soft drums" seconds=10
uv run favlo media --pipeline ace_step_15_turbo --model acestep-v15-xl-turbo-fp8-fp4 --out pack.json prompt="bright piano, soft drums" duration=10
```

- `favlo media` does not start the listener.
- It exits when that port is closed.
- Stdout is one `file://` URI.
- This is the command that writes the clip. The pipeline command does not.

### If download stops

Run `setup` again. Files that already have their published size stay on disk, and a short file is downloaded again.

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md) — as-built serving and inference detail
- [research/dit-mlp-fp4-error.md](research/dit-mlp-fp4-error.md) — DiT MLP fp4 layer scores
