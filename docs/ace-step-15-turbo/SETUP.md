# Set up ACE-Step

How to download, pack, and make a clip with `ace-step-15-turbo`. The kernel contract is [README.md](README.md). The listener is [../serve.md](../serve.md). Pointing a client at the server is [../guides/clients.md](../guides/clients.md).

The official checkpoint is [ACE-Step/acestep-v15-xl-turbo](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo). `setup` downloads the Diffusers split [ACE-Step/acestep-v15-xl-turbo-diffusers](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers). Both are under the MIT License.

## What you need

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A network path to `huggingface.co`
- Free disk for the checkout and, after packing, the packed record beside it

The downloaded files are 10.3 GiB together. The checkout stays on disk after packing.

That checkout holds the text encoder, the tokenizer, the condition encoder, the transformer, and the VAE. `setup` does not download a second repository.

8 GB is not enough to run a model capable of real-world agentic workflows.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

The package is not published to PyPI. Later commands use `uv run favlo` from that checkout.

## Download

```bash
uv run favlo --ace-step-15-turbo setup
```

That command downloads the Diffusers tree into `models/acestep-v15-xl-turbo` with plain HTTP. A public file does not need a token. A file that already has its published size is left in place. A short file is downloaded again.

`setup` skips `silence_latent.pt`. The silence latent used at generation is a tensor in `condition_encoder/diffusion_pytorch_model.safetensors`. `setup` does not pack a record and does not register one.

`--dest` names the checkout directory. With no `--dest`, the directory is `models/acestep-v15-xl-turbo`.

## Pack

```bash
uv run favlo --ace-step-15-turbo quantize
```

That command reads `models/acestep-v15-xl-turbo` and writes `models/acestep-v15-xl-turbo-fp8-fp4`. It registers that folder. The row stores that directory. The client model field is `acestep-v15-xl-turbo-fp8-fp4`.

Linear and convolution weights are fp8. Six of the thirty-two DiT MLP layers are stored as fp8, and the other twenty-six stay fp4. The layer list is [research/dit-mlp-fp4-error.md](research/dit-mlp-fp4-error.md). Norms, biases, AdaLN tables, Snake parameters, and the silence latent stay float32. The VAE encoder is not packed. The tokenizer is copied from `tokenizer/tokenizer.json` in the checkout.

A finished directory has `manifest.json` with `complete: true`. Running `quantize` again on that directory returns it.

`--source` names the checkout. `--dest` names the packed directory. With no flags, those are the two directories above.

## Make a clip

Start the listener:

```bash
uv run favlo serve
```

The daemon listens on `http://127.0.0.1:8129`. The model id is `acestep-v15-xl-turbo-fp8-fp4`. A music client calls `generate_music`. The default length is 30 seconds. A length is from 10 seconds through 10 minutes. The step count is 8.

From the checkout, without the listener:

```bash
uv run favlo --ace-step-15-turbo generate --prompt "bright piano, soft drums" --seconds 10 --out clip.wav
```

`--seconds` defaults to 10 on this command. `--lyrics` is optional. `--record` selects a packed directory other than `models/acestep-v15-xl-turbo-fp8-fp4`. The command writes a WAV.

## If download stops

Run `setup` again. Files that already have their published size stay on disk, and a short file is downloaded again.
