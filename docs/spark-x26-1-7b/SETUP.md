# Setup

`download` writes the official BF16 checkout under `models/Spark-X2.6-1.7B`. `quantize` packs `models/Spark-X2.6-1.7B-fp8` and registers it for `spark-x26-1-7b`. The matrices are packed on up to eight threads and written at fixed offsets in `weights.bin`. A finished manifest is left in place.

`favlo serve` on a terminal with an empty registry runs the same download and pack before it listens. `--keep-bf16` leaves the checkout.

```text
uv run favlo --spark-x26-1-7b setup
uv run favlo --spark-x26-1-7b download
uv run favlo --spark-x26-1-7b quantize
uv run favlo --spark-x26-1-7b kernels
```

`kernels` compiles the shader library and checks fp8 GEMV, both `matmul2d` tiles, RMSNorm, half-split RoPE at position 0, erf GELU, the sigmoid head gate, argmax, a one-hot nucleus sample, and an fp8 key score.
