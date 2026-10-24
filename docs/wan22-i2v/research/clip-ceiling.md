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

23 October 2026. The shader list is unchanged, so the 82.357 s sum stands. A second search the same day used "Wan2.2-I2V-A14B inference time seconds 4 steps 720p 2026", "Wan2.2 I2V 9 frames 576x320 Metal matmul2d generation seconds Apple Silicon", and "Wan2.2 I2V 576x320 9 frames seconds kernel sum Metal fp4". The [vLLM-Omni cookbook](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/diffusion/wan2.2/index.md) also lists 21.6 s for 832×480, 81 frames, and 4 steps with USP 2, and 101.6 s for 1280×720 and 121 frames. Its H200 retro lists 22.17 s on one device for the 832×480 clip and 79.19 s with USP 2 for the 720p clip. [TokenWatt](https://www.tokenwatt.io/blog/wan2-2-image-to-video-h200-720p.html) reports 155 s for a 5-second 720p clip and 18.1 minutes for 15 seconds on one H200, for an 8-step FP8 distill. The [nalexand FP8 card](https://huggingface.co/nalexand/Wan2.2-I2V-A14B-FP8) prints 58.23 s/it, and 51.32 s/it for FP8, at 640×352 and 81 frames on a 3070 Ti. [Morphic Films](https://github.com/morphicfilms/wan2.2_optimizations/blob/main/blog.md) reports 250.70 s, and 159.55 s with TF32, for 1280×720, 81 frames, and 40 steps on 8 H100s. [ac4k](https://github.com/ac4k/Wan2.2) reports about 1240 s for 81 frames at 720p and 40 steps on one RTX 5090. A search on 23 October 2026 did not find the 82.357 s sum.

24 October 2026. Head dimension 128 now uses `attend_q8`: one online-softmax walk, Milakov's running max and sum, shared by eight queries. The self-attention median is 73.502 ms and the cross-attention median is 16.357 ms. The DiT sum is 36.476 s. The kernel sum is 46.301 s. The attention bandwidth ceiling is 17.169 s at 277.04 GB/s. A printed 9-frame 576×320 clip was 49.28 s: umT5 2.36 s, encode 4.35 s, steps 9.17 s, 9.17 s, 9.20 s, 9.20 s, decode 5.83 s. `PYTHONPATH=pipelines/metal41` and `from wan22_i2v.measure import run` regenerates the medians. Queries the same day: "Wan2.2-I2V-A14B 9 frames 576x320 Metal kernel sum 46.301 seconds attend_q8". The returned notes are the predecessors already listed, plus an MLX mixed q8 timing of 242.2 s at 384×384, 33 frames, and 12 steps. A search on 24 October 2026 did not find the 46.301 s sum.

24 October 2026. The VAE mid-block attention is `spatial_attend_lane`: four query pixels share each key and value, and thirty-two lanes split the 384 channels on one online-softmax walk. The median is 419.941 ms. Six calls are 2.520 s. The VAE sum is 8.619 s. The kernel sum is 45.144 s. A printed 9-frame 576×320 clip was 45.74 s: umT5 2.41 s, encode 2.75 s, steps 9.09 s, 9.11 s, 9.08 s, 9.12 s, decode 4.18 s. Queries the same day: "Wan2.2 I2V 576x320 9 frames 45.144 seconds spatial attention Metal". The returned notes are the predecessors already listed. A search on 24 October 2026 did not find the 45.144 s sum.

24 October 2026. DiT attention for head dimension 128 is `attend_q4`. Four queries share each key. Eight queries and sixteen queries were timed on the same shape; four was the fastest. The self-attention median is 51.875 ms and the cross-attention median is 10.664 ms. The DiT sum is 32.105 s. The kernel sum is 40.773 s. The attention bandwidth ceiling is 34.234 s at 277.04 GB/s. A printed 9-frame 576×320 clip was 41.41 s: umT5 2.41 s, encode 2.76 s, steps 8.00 s, 8.04 s, 7.99 s, 8.04 s, decode 4.17 s. Queries the same day: "Wan2.2-I2V-A14B 576x320 9 frames 40.773 seconds four query attention". The returned notes are the official config, the model card, and the predecessors already listed. A search on 24 October 2026 did not find the 40.773 s sum.

24 October 2026. Each large convolution is an im2col tile of 4096 rows. The same seven shapes with an 8192-row tile summed to 6.191 s in that session; the 4096-row tile sums to 5.658 s. The other 49 shapes stay at the earlier 1.074 s. All 207 calls sum to 6.733 s. The VAE sum is 9.777 s. The kernel sum is 41.931 s. A printed 9-frame 576×320 clip was 41.08 s: umT5 2.42 s, encode 2.61 s, steps 8.02 s, 8.03 s, 8.02 s, 8.04 s, decode 3.94 s. Queries the same day: "Wan2.2 I2V convolution im2col 4096 row tile 41.931 seconds". A Tenstorrent sweep uses a 4096-byte fp32 tile, which is a different blocking. The returned notes do not list the 41.931 s sum. A search on 24 October 2026 did not find that sum.

24 October 2026. `patch_embed` assigns one thread to each output channel. The median is 2.490 ms, and four calls are 0.010 s. The DiT sum is 32.017 s. The kernel sum is 41.843 s. A printed 9-frame 576×320 clip was 41.06 s: umT5 2.42 s, encode 2.61 s, steps 8.01 s, 8.02 s, 8.00 s, 8.04 s, decode 3.96 s. Queries the same day: "Wan2.2-I2V-A14B 9 frames 576x320 patch embed 41.843 seconds". The returned notes are the predecessors already listed, including the vLLM-Omni times of 26.0 s, 21.6 s, and 101.6 s at other shapes. A search on 24 October 2026 did not find the 41.843 s sum.
