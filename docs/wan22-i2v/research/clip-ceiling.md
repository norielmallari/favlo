# Clip ceiling

Measurement note for the `wan22-i2v` pipeline. The serving contract is [../README.md](../README.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the isolated sum of one 9-frame 576×320 clip. It is a measurement of this shader library.

## Remainder

The DiT, the 9-token encoder, and the VAE, summed without overlap, are 82.357 s. A printed 9-frame clip was 89.49 s. A search on 23 October 2026 did not find that sum.

## Predecessors

- Method. Wan et al. ([arXiv:2503.20314](https://arxiv.org/abs/2503.20314)) define the dual-expert image-to-video transformer and the Wan2.1 causal VAE. Apple, WWDC26 session 330, specifies `matmul2d`.
- Task. [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B). [TurboDiffusion issue 51](https://github.com/thu-ml/TurboDiffusion/issues/51) reports 41 s of sampling and 79.3 s end to end for a 4-step 720p clip. The [vLLM-Omni cookbook](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/diffusion/wan2.2/index.md) lists 26.0 s for 832×480, 81 frames, and 4 steps. An M1 Max ComfyUI run of a Q5 4-step clip at 480×320 and 25 frames finished in 182.7 s, at 27.6 s per step ([Zenn](https://zenn.dev/umamon/articles/e24fbee8dc54d4)). [mlx-video](https://git.ddq.nl/n0p/mlx-video/src/commit/f4195f0118fb5caf4fbe2eeacfe12e163dcba7a0/docs/wan22-implementation-notes.md) reports 47 s against 56 s for 20 steps at 480p.
- Measurement. Those figures are other engines, other quants, and other frame counts. They do not publish this per-kernel sum at 576×320 and 9 frames.
- Mechanism. Overlapping command buffers are not added. The 90% test uses the 82.357 s kernel ceiling. The attention bandwidth ceiling is 204.951 s. A 2 GiB copy finished in 15.503 ms, 277.04 GB/s counting the read and the write.

## Rule

Each listed kernel runs in its own command buffer. The sample is the median of five dispatches after one discarded sample. The latent is 3×40×72, which is 2160 tokens, and `matmul2d` rows are padded to 2176. Four Euler steps and forty layers. Eight of the forty blocks are fp8. Text, time, and the head stay fp4. The expert pack read and the AVI pack are outside the sum.

## Command

`favlo --wan22-i2v kernels` checks a synthetic fp4 tile and an fp8 tile. That command does not print this table. The printed stage lines come from `WAN22_SMOKE=1 uv run pytest tests/test_wan22_i2v_smoke.py -s`. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Piece | Time |
|---|---|
| DiT | 72.532 s |
| 9-token umT5 | 0.049 s |
| VAE | 9.776 s |
| Kernel sum | 82.357 s |
| Attention bandwidth ceiling | 204.951 s |
| Printed clip | 89.49 s: umT5 2.49 s, encode 4.65 s, steps 18.88 s, 18.95 s, 18.83 s, 19.02 s, decode 6.67 s |

## Search

23 October 2026. Queries: "Wan2.2-I2V-A14B inference time seconds 4 steps 720p"; "Wan2.2 I2V A14B Apple Silicon Metal generation seconds 576x320"; "Wan2.2 I2V 9 frames 576x320 Metal matmul2d". The predecessors in this note are the overlaps those queries returned. The search did not find the 82.357 s sum.

## Corrections

Later finds are appended here. The sections above stay as committed.
