# Set up umT5-XXL

How to download and pack the shared umT5 int8 record. Wan I2V and TI2V open
that pack for text. The encoder weights are under the Apache License,
Version 2.0.

## Download

```bash
uv run favlo --umt5-xxl setup
```

That writes `models/google/umt5-xxl/` with `models_t5_umt5-xxl-enc-bf16.pth`
and the tokenizer files. `--umt5 org/name` downloads that repository instead.

## Pack

```bash
uv run favlo --umt5-xxl quantize --format wan
```

That writes `models/google/umt5-xxl-int8`, copies every tokenizer file into
that directory, and removes the BF16 checkout. A clip never opens the BF16
file. `--keep-source` leaves the checkout in place. A Hugging Face
`.safetensors` encoder uses `--format hf` and `--src` for that file:

```bash
uv run favlo --umt5-xxl quantize --format hf \
  --src models/google/umt5-xxl-realistic_bf16/umt5-xxl-realistic_bf16.safetensors \
  --out models/google/umt5-xxl-realistic-int8 \
  --tokenizer models/google/umt5-xxl-int8 \
  --keep-source
```

Point a Wan video row at the pack with `parts.umt5`, or let `quantize` on
that video pipeline store the shared path.
