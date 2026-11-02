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

`GET /v1/models` lists the registered ids. A bare `qwen35-4b` uses the thinking card (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, `reasoning_effort` `high`). `qwen35-4b:instruct` uses temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, and `reasoning_effort` `none`. A request `reasoning.effort` wins when set (`none` or `high`; the other listed names map onto those). When the request omits effort, the selected variant's `reasoning_effort` is used.

`mtp-half` appends the official MTP matrices as Metal `half` from the SSD BF16 checkout (`models/Qwen3.5-4B-bf16`, including `mtp.safetensors` when the language shards omit those tensors) onto a finished fp8 record. A request field `mtp` wins; omitted uses the registry row (default off). `mtp_depth` defaults to 3.

```text
uv run favlo --qwen35-4b mtp-half --src models/Qwen3.5-4B-bf16
```
