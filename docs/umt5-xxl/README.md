# umT5-XXL

Shared text-encoder pack for Wan video pipelines. `favlo --umt5-xxl` downloads
the BF16 encoder from [google/umt5-xxl](https://huggingface.co/google/umt5-xxl)
and packs it as int8 under `models/google/umt5-xxl-int8`. The tokenizer files
are copied into that pack. The BF16 checkout is removed after a successful
pack. Wan I2V and TI2V open that directory through `parts.umt5` and read
`tokenizer.json` from it. This pipeline does not serve a clip.

## Commands

```text
uv run favlo --umt5-xxl setup
uv run favlo --umt5-xxl quantize
```

`setup --umt5 org/name` downloads that repository instead. `quantize --src`
and `--out` name the encoder file and the pack directory. `--keep-source`
leaves the BF16 checkout in place.

Setup steps are in [SETUP.md](SETUP.md).
