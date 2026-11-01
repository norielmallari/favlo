# Setup

`download` writes the official BF16 checkout under `models/Spark-X2.6-1.7B`. `quantize` packs `models/Spark-X2.6-1.7B-fp4` and registers it for `spark-x26-1-7b`. A finished manifest is left in place.

```text
uv run favlo --spark-x26-1-7b download
uv run favlo --spark-x26-1-7b quantize
uv run favlo --spark-x26-1-7b kernels
```

`kernels` compiles the shader library and checks fp4 GEMV, both `matmul2d` tiles, RMSNorm, half-split RoPE at position 0, erf GELU, the sigmoid head gate, argmax, a one-hot nucleus sample, and an fp8 key score.
