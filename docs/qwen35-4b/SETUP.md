# Qwen3.5-4B setup

Official weights are [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). `favlo serve` on a terminal with an empty registry downloads that checkpoint, packs it, and removes the BF16 checkout. A sharded checkout and a single `model.safetensors` both pack. Prefer the official `chat_template.jinja` (tools and thinking) over a short finetune template.

```text
uv run favlo --qwen35-4b setup
uv run favlo --qwen35-4b setup --keep-bf16
uv run favlo --qwen35-4b quantize --src models/Qwen3.5-4B-bf16
uv run python -m favlo.smoke_lock run --pipeline qwen35-4b --command kernels -- \
  uv run favlo --qwen35-4b kernels
uv run python -m favlo.smoke_lock run --pipeline qwen35-4b --command complete -- \
  uv run favlo --qwen35-4b complete
```

`quantize` / `setup` write `models/Qwen3.5-4B-fp8` and register the row with `kv_cache` set to `half`. `fp8_e4m3` is still accepted on that field. The packer stores language and vision matrices as fp8 E4M3 with E8M0 scales. Norms, convolution taps, GDN parameters, vision LayerNorm weights and biases, and the other vision biases stay Metal `half`. A finished record whose tensor list matches that pack is left in place. A record packed before the vision tensors were added is rebuilt. `kernels` compiles the Metal library and runs GEMV, RMSNorm, Gated DeltaNet, and half cache dispatch checks. `serve --max-memory` and `complete --max-memory` cap the serve ledger; this package does not parse that string.
