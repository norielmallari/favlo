# umT5-XXL

Shared text-encoder pack for Wan video pipelines. It is not a Favlo CLI pipeline
or flag.

Wan I2V and TI2V:

- obtain this library through `umt5 setup` and `umt5 quantize` on their own CLI rows
- open the packed int8 directory through `parts.umt5`
- read the same tokenizer and weights during a clip

As-built detail: [ARCHITECTURE.md](ARCHITECTURE.md).

Wan registry keys and `parts.umt5` are in [the I2V note](../wan22_i2v/ARCHITECTURE.md#registry)
and [the TI2V note](../wan22_ti2v_5b/ARCHITECTURE.md#registry).

## Setup

- How to download and pack the shared umT5 int8 record.
- Wan I2V and TI2V open that pack for text.
- The encoder weights are under the Apache License, Version 2.0.
- There is no `favlo --umt5_xxl` flag: either Wan pipeline owns the commands.

### Download

```bash
uv run favlo --wan22_i2v umt5 setup
```

- That writes `models/google/umt5-xxl/` with `models_t5_umt5-xxl-enc-bf16.pth`
  and the tokenizer files.
- `--umt5 org/name` downloads that repository instead.
- `favlo --wan22_ti2v_5b umt5 setup` is the same pack.

### Pack

```bash
uv run favlo --wan22_i2v umt5 quantize --format wan
```

- That writes `models/google/umt5-xxl-int8`.
- It copies every tokenizer file into that directory.
- It removes the BF16 checkout.
- A clip never opens the BF16 file.
- `--keep-source` leaves the checkout in place.

A Hugging Face `.safetensors` encoder uses `--format hf` and `--src` for that file:

```bash
uv run favlo --wan22_i2v umt5 quantize --format hf \
  --src models/google/umt5-xxl-realistic_bf16/umt5-xxl-realistic_bf16.safetensors \
  --out models/google/umt5-xxl-realistic-int8 \
  --tokenizer models/google/umt5-xxl-int8 \
  --keep-source
```

- Point a Wan video row at the pack with `parts.umt5`.
- Or let `quantize` on that video pipeline store the shared path.