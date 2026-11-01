# Qwen3.5-4B setup

Official weights are [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). A same-tensor BF16 checkout may be copied into `models/Qwen3.5-4B-bf16` and packed from SSD. Prefer the official `chat_template.jinja` (tools and thinking) over a short finetune template.

```text
# Optional: copy a same-tensor BF16 checkout onto SSD first, then:
uv run favlo --qwen35-4b setup --keep-bf16
# Or pack only:
uv run favlo --qwen35-4b quantize --src models/Qwen3.5-4B-bf16
uv run python -m favlo.smoke_lock run --pipeline qwen35-4b --command kernels -- \
  uv run favlo --qwen35-4b kernels
uv run python -m favlo.smoke_lock run --pipeline qwen35-4b --command complete -- \
  uv run favlo --qwen35-4b complete
```

`quantize` / `setup` write `models/Qwen3.5-4B-fp8` and register the row. The packer stores language tensors as fp8 E4M3 with E8M0 scales; norms, convolution taps, and GDN parameters stay float16. Vision tensors are not packed in this step. `kernels` compiles the Metal library and runs GEMV, RMSNorm, and Gated DeltaNet dispatch checks. `serve --max-memory` and `complete --max-memory` cap the serve ledger; this package does not parse that string.
