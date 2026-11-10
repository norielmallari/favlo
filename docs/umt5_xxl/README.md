# umT5-XXL

Shared text-encoder pack for Wan video pipelines. Not a Favlo CLI pipeline.
`favlo --wan22_i2v umt5` (or `--wan22_ti2v_5b umt5`) downloads the BF16
encoder from [google/umt5-xxl](https://huggingface.co/google/umt5-xxl)
and packs it as int8 under `models/google/umt5-xxl-int8`. The tokenizer files
live in that pack. Wan I2V and TI2V open that directory through `parts.umt5` and read
the same bytes. The pack library lives under `pipelines/umt5_xxl/metal41/`.

```bash
uv run favlo --wan22_i2v umt5 setup
uv run favlo --wan22_i2v umt5 quantize --format wan
```

`umt5 setup --umt5 org/name` downloads that repository instead. `umt5 quantize`
defaults to `--format wan`. `--format hf` packs a Hugging Face
`.safetensors` encoder; pass `--src` for that file. `--keep-source` leaves the
BF16 checkout in place. The default pack removes it after a successful write.
`parts.umt5` on a Wan registry row is that shared directory. A clip never opens
the BF16 file.
