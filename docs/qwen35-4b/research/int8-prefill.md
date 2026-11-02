# Int8 prefill

Measurement note for the `qwen35-4b` pipeline. The serving contract is [../README.md](../README.md). The run is in that note. The isolated kernel sum is [kernel-ceilings.md](kernel-ceilings.md).

## Remainder

On `Qwen3.5-4B-fp8`, on an M5 Pro with 48 GB, a warm 2012-token prefill after the resident int8 weight atlas recorded about 1852 tok/s, and the first call that filled that atlas recorded 1781 tok/s. A search on 2 November 2026 did not find that pair.

## Predecessors

- Method. [oMLX pull 3548](https://github.com/jundot/omlx/pull/3548) routes prefill matmuls as int8 × int8 and keeps decode on packed weights. [Cider](https://github.com/Mininglamp-AI/cider/blob/main/tutorial/how_to_write_efficient_int_gemm_m5_en.md) describes INT8×INT8→INT32 `matmul2d` on M5. Favlo's [qwen38-27b int8-prefill](../../qwen38-27b/research/int8-prefill.md) restores packed rows into an int8 stage with one channel scale per row.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). [oMLX Qwen3.5-4B-8bit](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 2643 prefill tok/s at 4096 context on an M5 Pro with 20 shader cores and 48 GB for an 8-bit checkpoint. [RedHatAI W8A8](https://huggingface.co/RedHatAI/Qwen3.5-4B-quantized.w8a8) ships a static int8 weight and activation checkpoint for vLLM, not this Metal restore path.
- Measurement. Those figures are another engine or another checkpoint. They do not publish this fp8-record restore with a resident int8 atlas.
- Mechanism. Prefill restores listed fp8 planes (`gate`, `up`, `down`, GDN `qkv` / `z` / `out`) into int8 codes and one channel scale, quantizes half activations per row, runs `prefill_i8_n64` or `prefill_i8_m32`, then `scale_i8_f32`. When the budget fits, each plane restores once into a resident atlas. GDN a/b and attention q/k/v/o stay half×fp8. Decode does not read the atlas.

## Rule

Prefill is the summed prompt time on a cache miss. The 2012-token prompt repeated one token id. The first call includes the lazy restores into the atlas. The warm call reuses that atlas. Temperature 0 is an argmax. Measured on 2 November 2026 on the default `f16` cache.

## Command

The lines are the ones printed in [../README.md](../README.md). No committed command reprints the row.

## Table

| Run | Prefill |
|---|---|
| First call (fills atlas), 2012 tokens | 1781 tok/s |
| Warm (atlas resident), 2012 tokens | ~1852 tok/s |

## Search

2 November 2026. Queries: "Qwen3.5-4B prefill tok/s int8 Apple Silicon M5 Pro 2026", "Qwen3.5-4B W8A8 int8 prefill restore fp8 resident cache Apple Metal matmul2d", "Qwen3.5-4B oMLX OR mlx OR MTPLX decode tok/s M5 Pro", and "Qwen3.5-4B-fp8 1852 OR 1781 OR 1673.36 tok/s Favlo". The predecessors in this note are the overlaps those queries returned. The 2643 tok/s figure is an 8-bit oMLX checkpoint. The search did not find 1781 or 1852 tok/s on `Qwen3.5-4B-fp8` with a resident int8 atlas.

## Corrections

Later finds are appended here. The sections above stay as committed.
