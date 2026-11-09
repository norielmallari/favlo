# Generation time

Measurement note for the `z-image-base` pipeline. The serving contract is [../README.md](../README.md). This file records two printed generations of the int8 record. It is a measurement of this implementation.

## Remainder

On an M5 Pro with 48 GB, one 512 image at 50 steps and guidance 4 printed 70.47 s, and one 1024 image at the same step count and guidance printed 362.05 s. A search on 9 November 2026 did not find those prints.

## Predecessors

- Method. Cai et al. ([arXiv:2511.22699](https://arxiv.org/abs/2511.22699)) define the single-stream diffusion transformer and the flow-matching sampler. The [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) card recommends 28–50 steps and guidance 3–5, and its example uses 50 steps and guidance 4.
- Task. [mlx-teacache](https://github.com/IonDen/mlx-teacache/blob/main/docs/variants/z-image-base.md) reports a 227.4 s median for a 512 image at 50 steps and guidance 4 with 8-bit MLX weights on an M1 Max with 32 GB. [vram16](https://vram16.ullu-egg.com/en/models/z-image-base/) reports 13.21 s for a 1024 image at 25 steps with int8 weights on an RTX 5080. [ixim/Z-Image-INT8](https://huggingface.co/ixim/Z-Image-INT8) reports 46.79 s average for a 1024 image at 50 steps and guidance 4 with weight-only int8 on an RTX 5090. [ComfyUI-ZImage-Triton](https://github.com/newgrit1004/ComfyUI-ZImage-Triton/blob/main/benchmark/BENCHMARK.md) reports 15.32 s for a 1024 image at 30 steps and guidance 4 with int8 Hadamard kernels.
- Measurement. Those figures are other engines, other machines, or another step count. They do not publish these two prints.
- Mechanism. The activation is quantized in groups of 128 along K. A positive guidance runs the prompt velocity and the negative velocity inside the printed step time.

## Rule

Each printed line is one generation of the int8 record. Shift is 6 and CFG normalization is 0. The seed is 0. The step column is the range of the fifty step lines. The total includes the caption and the decoder.

## Command

```text
uv run favlo media --pipeline z-image-base --model Z-Image-int8 prompt="A single matte red cube centered on an empty white table. The tabletop is flat, clean, and uniformly white. The background is plain white. Soft even light. Nothing else is in the frame." negative_prompt="clutter, stains, debris, extra objects, text, watermark, pattern, colored background, multiple cubes" steps=50 guidance=4 shift=6 cfg_normalization=0 size=512x512 seed=0
uv run favlo media --pipeline z-image-base --model Z-Image-int8 prompt="A woman sitting on short green grass. She sits in the center of the frame with her legs folded to one side and her hands resting in her lap. The lawn fills the ground around her. Soft daylight, even and natural. A plain outdoor photograph. Correct human anatomy, natural hands and fingers. No text, no watermark, no logo." negative_prompt="extra limbs, extra fingers, distorted face, text, watermark, clutter, debris, multiple people" size=1024x1024 steps=50 guidance=4 shift=6 cfg_normalization=0 seed=0
```

The table is the pair printed on 9 November 2026 by the pipeline generate command, which is gone. The commands above are the same body through `favlo media`. A later run is a new measurement.

## Table

| Edge | Total | Caption | Decoder | Step |
|---|---|---|---|---|
| 512 | 70.47 s | 0.42 s | 0.88 s | 1.37–1.40 s |
| 1024 | 362.05 s | 0.32 s | 5.09 s | 7.00–7.17 s |

## Search

9 November 2026. Queries: "Z-Image base 50 steps inference seconds 512 OR 1024 Apple Silicon Metal int8 2026" and "Z-Image base 70.47 OR 362.05 OR 1.37s 512 50 steps Metal int8 M5". The predecessors in this note are the overlaps those queries returned. The search did not find the 70.47 s print or the 362.05 s print.

## Corrections

Later finds are appended here. The sections above stay as committed.
