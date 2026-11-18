# umT5-XXL

Shared text-encoder pack for Wan video pipelines. Not a Favlo CLI pipeline or flag.

Checkpoint: [google/umt5-xxl](https://huggingface.co/google/umt5-xxl). Apache License, Version 2.0.

Packed folder: `models/google/umt5-xxl-int8`.

Wan I2V and TI2V:

- obtain this library through `umt5 setup` and `umt5 quantize` on their own CLI rows
- open the packed int8 directory through `parts.umt5`
- read the same tokenizer and weights during a clip

As-built detail: [ARCHITECTURE.md](ARCHITECTURE.md).

Wan registry keys and `parts.umt5` are in [the I2V note](../wan22_i2v/ARCHITECTURE.md#registry) and [the TI2V note](../wan22_ti2v_5b/ARCHITECTURE.md#registry).

## Download

```bash
uv run favlo --wan22_i2v umt5 setup
```

- Writes `models/google/umt5-xxl/` with `models_t5_umt5-xxl-enc-bf16.pth` and the tokenizer files.
- `--umt5 org/name` downloads that repository instead.
- `favlo --wan22_ti2v_5b umt5 setup` is the same pack.

## Pack

```bash
uv run favlo --wan22_i2v umt5 quantize --format wan
```

- Writes `models/google/umt5-xxl-int8`.
- Writes `weights.bin` and `planes.json` beside the `.npy` and `.npz` files.
- Copies every tokenizer file into that directory.
- Removes the BF16 checkout.
- A clip never opens the BF16 file.
- `--keep-source` leaves the checkout in place.
- `--format hf` packs a Hugging Face `.safetensors` encoder; pass `--src` for that file.
- Point a Wan video row at the pack with `parts.umt5`.
- Or let `quantize` on that video pipeline store the shared path.
