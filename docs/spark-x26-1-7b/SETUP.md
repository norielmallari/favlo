# Setup

Parked pipeline. The command scan does not offer `--spark-x26-1-7b`. The entry is `pipelines/parked/spark-x26-1-7b.py`.

`download` writes the official BF16 checkout under `models/Spark-X2.6-1.7B`. `quantize` packs `models/Spark-X2.6-1.7B-fp8` and registers it for `spark-x26-1-7b`. The matrices are packed on up to eight threads and written at fixed offsets in `weights.bin`. A finished manifest is left in place.

`--keep-bf16` leaves the checkout. An empty registry downloads Qwen3.5-4B instead of this checkpoint.

```text
uv run favlo --spark-x26-1-7b setup
uv run favlo --spark-x26-1-7b download
uv run favlo --spark-x26-1-7b quantize
uv run favlo --spark-x26-1-7b kernels
```

`kernels` compiles the shader library and checks fp8 GEMV, both `matmul2d` tiles, RMSNorm, half-split RoPE at position 0, erf GELU, the sigmoid head gate, argmax, a one-hot nucleus sample, and an fp8 key score.
