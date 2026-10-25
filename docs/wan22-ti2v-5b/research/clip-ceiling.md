# Clip ceiling

Measurement note for the `wan22-ti2v-5b` pipeline. The serving contract is [../README.md](../README.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the isolated DiT sum and two printed 832×480 clips. It is a measurement of this implementation.

## Remainder

On an M5 Pro with 48 GB the isolated DiT sum at 576×320 and 9 frames is 4.267 s. A printed 832×480 9-frame text clip decoded in 18.942 s and the still clip decoded in 15.315 s. A search on 26 October 2026 did not find that sum or those two decode lines.

## Predecessors

- Method. Wan et al. ([arXiv:2503.20314](https://arxiv.org/abs/2503.20314)) define the TI2V-5B transformer and the 48-channel causal VAE. Apple, WWDC26 session 330, specifies `matmul2d`.
- Task. [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). [MLX-Gen](https://github.com/lpalbou/mlx-gen/blob/main/docs/wan-video.md) reports about 12 minutes for 832×480, 101 frames, and 25 steps, and about 35 minutes for 1280×704, on an M5 Max. [FastMetal-5B-QAD](https://haoailab.com/blogs/fastmetal/) reports 151.42 s end to end, of which 98.50 s is denoise, for 704×1280, 81 frames, and 3 steps on an M4 Max, with a tiny decoder. [Yume-1.5-5B-720P-MLX](https://huggingface.co/ckurasek/Yume-1.5-5B-720P-MLX) reports 4.5 minutes of denoise and 1.5 minutes of VAE decode for 832×480, 81 frames, and 30 steps on an M5 Pro with 48 GB. [auswolf/Wan2.2-TI2V-5B-4step-nvfp4](https://huggingface.co/auswolf/Wan2.2-TI2V-5B-4step-nvfp4) reports 18.97 s for a 4-step 1280×704 clip of about 5 seconds on a 5060 Ti, with a tiny decoder. [FlashRT](https://github.com/flashrt-project/FlashRT/blob/main/docs/wan22_usage.md) reports 178.6 s for 1280×704, 121 frames, and 20 steps on an RTX 5090.
- Measurement. Those figures are other engines, other quants, and other frame counts. They do not publish this per-kernel sum or these printed decode lines.
- Mechanism. The isolated sum times each listed kernel in its own command buffer. The printed knots submit the 30 blocks before the wait. Text and the VAE sit outside the 4.267 s sum. The printed decode is that VAE at 832×480, which is a different grid from the isolated sum.

## Rule

Each listed kernel runs in its own command buffer. The sample is the median of three dispatches after one discarded sample. The isolated grid is 576×320 and 9 frames, token rows padded from 540 to 576, four knots, and 30 blocks. The native rows are 1280×704 and 121 frames, token rows padded from 27280 to 27328. The printed clips are 832×480 and 9 frames. Host uploads sit outside the isolated times.

## Command

`uv run favlo --wan22-ti2v-5b ceilings` reprints the isolated medians. The 4.267 s sum is the arithmetic in [../ceilings.md](../ceilings.md) from 25 October 2026. `uv run favlo --wan22-ti2v-5b smoke` prints the `dit` and `decode` lines. The lines below are that print from 26 October 2026. `kernels` checks a synthetic tile and does not regenerate this table.

## Table

| Piece | Time |
|---|---|
| Isolated DiT, 576×320, 9 frames | 4.267 s |
| Native linears, 1280×704, 121 frames | 23.224 s |
| Native attention, one query tile per head | 967.822 s |
| Printed text DiT | 1.180 s, 0.665 s, 0.649 s, 0.638 s |
| Printed text decode | 18.942 s |
| Printed still DiT | 0.817 s, 0.650 s, 0.653 s, 0.661 s |
| Printed still decode | 15.315 s |

## Search

26 October 2026. Queries: "Wan2.2-TI2V-5B inference time seconds 832x480 9 frames 4 steps", "Wan2.2 TI2V-5B Apple Silicon Metal generation seconds 576x320 4.267", "Wan2.2 TI2V 5B VAE decode seconds 480p MLX ComfyUI", and "18.942 OR 15.315 OR 4.267 Wan2.2 TI2V Metal decode". The predecessors in this note are the overlaps those queries returned. [Anvil](https://fakoli.github.io/anvil-serving/benchmarks/models/wan22/) reports 9.092 s for 512×288, 17 frames, and 8 steps on an RTX 5090. A [GizAI FastVideo](https://github.com/GizAI/FastVideo) note reports about 18 s warm for 832×480, 81 frames, and 3 steps on two GPUs. A search on 26 October 2026 did not find the 4.267 s sum or the 18.942 s and 15.315 s decode lines.

## Corrections

Later finds are appended here. The sections above stay as committed.
